#include "shape_decoder.h"
#include "trellis_debug.h"
#include "trellis_sched.h"
#include "sparse.h"
#include "trellis_model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <stdexcept>

namespace trellis {
using T = ggml_tensor;

// A whole ConvNeXt stage (up to 16 blocks) is built as one graph, and each block is now
// chunked over voxels (~145 nodes per chunk), so the node count scales with object density.
// This is host-side metadata only (~350 B/tensor), so headroom is cheap -- running out
// aborts in ggml_new_object, which is a much worse failure than a few MB of address space.
static constexpr size_t kGraphNodes = 262144;

// Live GTT (whole-device, so it shows anything the driver has NOT handed back) and this
// process's RSS. Probed either side of every graph: if GTT does not drop back after a
// gallocr_free, memory is being retained between stages rather than merely used by one.
void mem_probe(const char* tag) {
    if (!getenv("TRELLIS_DBG_MEM")) return;
    long long gtt = 0, rss = 0;
    if (FILE* f = fopen("/sys/class/drm/card1/device/mem_info_gtt_used", "r")) { if (fscanf(f, "%lld", &gtt) != 1) gtt = 0; fclose(f); }
    if (FILE* f = fopen("/proc/self/statm", "r")) { long long tot; if (fscanf(f, "%lld %lld", &tot, &rss) != 2) rss = 0; fclose(f); }
    fprintf(stderr, "      [mem] %-34s GTT=%6.2f GB  RSS=%6.2f GB\n",
            tag, gtt / 1e9, rss * 4096.0 / 1e9);
}

// run a graph producing `out`, with the given input tensors set from host data
static std::vector<float> run1(const Model& m, ggml_context* c, T* out,
                               std::vector<std::pair<T*, const void*>> ins) {
    ggml_set_output(out);
    ggml_cgraph* g = ggml_new_graph_custom(c, kGraphNodes, false);
    ggml_build_forward_expand(g, out);
    trellis::GraphExec ex(m);
    if (!ex.alloc(g)) throw std::runtime_error("shape_dec alloc");
    if (getenv("TRELLIS_DBG_ALLOC"))
        fprintf(stderr, "      [stage-alloc] nodes=%d  gallocr buffer = %.2f GB\n",
                ggml_graph_n_nodes(g), ex.buffer_size() / 1e9);
    for (auto& [t, d] : ins) ggml_backend_tensor_set(t, d, 0, ggml_nbytes(t));
    if (ex.compute(g, "shape_decode") != GGML_STATUS_SUCCESS) throw std::runtime_error("shape_dec compute");
    mem_probe("run1 computed (pre-readback)");
    std::vector<float> r = tensor_to_f32(out);
    mem_probe("run1 freed (post-readback)");
    return r;
}

static ggml_context* mkctx() {
    size_t meta = ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false) + (1 << 20);
    return ggml_init({ meta, nullptr, true });
}

// Per-voxel linear (optional per-row LN first), chunked so a single mul_mat
// never exceeds backend dispatch-grid limits (Vulkan caps grid dim 1 at 65535
// workgroups = ~2.1M rows; the res-1024 cascade reaches 2.6M voxels).
static constexpr int64_t kLinearRowChunk = 1000000;

static std::vector<float> linear_rows(const Model& m, const std::vector<float>& in, int Cin,
                                      int64_t N, const std::string& prefix, int Cout, bool pre_norm) {
    std::vector<float> out((size_t)Cout * N);
    for (int64_t r0 = 0; r0 < N; r0 += kLinearRowChunk) {
        const int64_t n = std::min(kLinearRowChunk, N - r0);
        ggml_context* c = mkctx();
        T* gh = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cin, n); ggml_set_input(gh);
        T* x = pre_norm ? ggml_norm(c, gh, 1e-5f) : gh;
        x = ggml_add(c, ggml_mul_mat(c, m.get(prefix + ".weight"), x), m.get(prefix + ".bias"));
        std::vector<float> r = run1(m, c, x, { {gh, in.data() + (size_t)Cin * r0} });
        std::copy(r.begin(), r.end(), out.begin() + (size_t)Cout * r0);
        ggml_free(c);
    }
    return out;
}

// Generic SparseUnetVaeDecoder: from_latent -> 4×(ConvNeXt stage + C2S) -> final LN -> output_layer.
// guide_subs!=null -> tex (C2S driven by provided masks); subs_out!=null -> shape (collect masks).
static std::vector<float> decode_unet(const Model& m, const std::vector<float>& latent, int out_ch,
                                      std::vector<std::array<int,3>>& coords,
                                      const std::vector<std::vector<uint8_t>>* guide_subs,
                                      std::vector<std::vector<uint8_t>>* subs_out,
                                      bool coords_only = false) {
    int N = (int)coords.size();
    // from_latent [32,N] -> [1024,N]
    std::vector<float> h = linear_rows(m, latent, 32, N, "from_latent", 1024, false);
    struct Stage { int C, nblk, Cout, c2si; const char* s; };
    const Stage stages[4] = { {1024,4,512,4,"0"}, {512,16,256,16,"1"}, {256,8,128,8,"2"}, {128,4,64,4,"3"} };
    for (int si = 0; si < 4; ++si) {
        const Stage& st = stages[si];
        std::vector<int32_t> nbr = build_neighbor_table(coords);
        if (getenv("TRELLIS_DBG_MEM"))
            fprintf(stderr, "    == stage %d: C=%d nblk=%d N=%d | host h=%.2f GB nbr=%.2f GB\n",
                    si, st.C, st.nblk, N, (double)h.size() * 4 / 1e9, (double)nbr.size() * 4 / 1e9);
        {   // ConvNeXt stage. Once the full stage input itself reaches ~2 GiB, the old
            // sparse_convnext concat chain multiplies that residency several-fold (the real
            // stage-3 failure reserved 31.36 GB). Go straight to true chunk-local streaming
            // for that case instead of deliberately provoking an OOM first. Smaller stages
            // keep the fast whole-stage graph, with an allocation-failure fallback as a safety net.
            size_t stream_at = 2ull * 1024 * 1024 * 1024;
            if (const char* e = getenv("TRELLIS_CONVNEXT_STREAM_GB")) {
                const double gb = atof(e);
                if (gb > 0.0) stream_at = (size_t)(gb * 1024.0 * 1024.0 * 1024.0);
            }
            const size_t feature_bytes = (size_t)st.C * (size_t)N * sizeof(float);
            const bool force_stream = getenv("TRELLIS_CONVNEXT_STREAM_ALL") != nullptr;
            bool stream_stage = force_stream || feature_bytes >= stream_at;

            if (!stream_stage) {
                // The host vector `h` is only assigned after a successful run1(), so it remains
                // the stage input if the fast-path allocation fails.
                ggml_context* c = mkctx();
                try {
                    T* gh = ggml_new_tensor_2d(c, GGML_TYPE_F32, st.C, N); ggml_set_input(gh);
                    T* gn = ggml_new_tensor_2d(c, GGML_TYPE_I32, N, 27);   ggml_set_input(gn);
                    T* x = gh;
                    for (int j = 0; j < st.nblk; ++j)
                        x = sparse_convnext(c, m, std::string("blocks.") + st.s + "." + std::to_string(j), x, gn, N);
                    std::vector<float> fast = run1(m, c, x, { {gh, h.data()}, {gn, nbr.data()} });
                    ggml_free(c); c = nullptr;
                    h = std::move(fast);
                } catch (const std::runtime_error& e) {
                    if (c) ggml_free(c);
                    if (std::string(e.what()) != "shape_dec alloc") throw;
                    stream_stage = true;
                    fprintf(stderr,
                            "      [shape-dec-stream] stage %d C=%d N=%d whole-stage alloc failed; "
                            "retrying streamed\n", si, st.C, N);
                }
            } else if (getenv("TRELLIS_DBG_ALLOC")) {
                fprintf(stderr,
                        "      [shape-dec-stream] stage %d C=%d N=%d feature-table=%.2f GiB "
                        ">= %.2f GiB; streaming\n",
                        si, st.C, N, feature_bytes / 1073741824.0, stream_at / 1073741824.0);
            }

            if (stream_stage) {
                for (int j = 0; j < st.nblk; ++j) {
                    const std::string bp = std::string("blocks.") + st.s + "." + std::to_string(j);
                    h = sparse_convnext_streamed(m, bp, h, st.C, nbr, N);
                }
            }
        }
        mem_probe("after ConvNeXt stage");
        const std::vector<uint8_t>* ext = guide_subs ? &(*guide_subs)[si] : nullptr;
        C2SResult r = sparse_c2s(m, std::string("blocks.") + st.s + "." + std::to_string(st.c2si), h, st.C, coords, st.Cout, ext);
        if (subs_out) subs_out->push_back(r.subdiv);
        h = std::move(r.feats); coords = std::move(r.coords); N = (int)coords.size();
        mem_probe("after c2s");
    }
    if (coords_only) return {};   // cascade upsample: just need the grown coords
    // final LN(no affine) + output_layer -> [out_ch, M]
    return linear_rows(m, h, 64, N, "output_layer", out_ch, true);
}

ShapeOut shape_decode(const Model& m, const std::vector<float>& latent,
                      const std::vector<std::array<int,3>>& coords0, int resolution) {
    ShapeOut so; so.coords = coords0; so.res = resolution;
    so.feats7 = decode_unet(m, latent, 7, so.coords, nullptr, &so.subs);
    return so;
}

std::vector<std::array<int,3>> shape_upsample(const Model& m, const std::vector<float>& latent,
                                              const std::vector<std::array<int,3>>& coords0) {
    std::vector<std::array<int,3>> coords = coords0;
    decode_unet(m, latent, 7, coords, nullptr, nullptr, /*coords_only=*/true);
    return coords;
}

std::vector<float> tex_decode(const Model& m, const std::vector<float>& tex_latent,
                              const std::vector<std::array<int,3>>& coords0,
                              const std::vector<std::vector<uint8_t>>& subs) {
    std::vector<std::array<int,3>> coords = coords0;
    return decode_unet(m, tex_latent, 6, coords, &subs, nullptr);   // 6-ch PBR at the same final coords
}

DecodeTree decode_tree_from_fine(const std::vector<std::array<int,3>>& fine,
                                 const std::vector<std::array<int,3>>& latent) {
    // 21 bits per axis, as sparse_s2c's parent key.
    auto key = [](int x, int y, int z) {
        return ((uint64_t)(uint32_t)x << 42) | ((uint64_t)(uint32_t)y << 21) | (uint64_t)(uint32_t)z;
    };
    // levels[0] = fine; levels[k] = unique(levels[k-1] >> 1), each a sorted key set. These are
    // the coordinate sets the encoder's four S2C blocks produce.
    std::vector<std::vector<uint64_t>> levels(5);
    levels[0].reserve(fine.size());
    for (const auto& c : fine) {
        if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[0] >= (1 << 21) || c[1] >= (1 << 21) || c[2] >= (1 << 21))
            throw std::runtime_error("decode_tree_from_fine: coordinate out of range");
        levels[0].push_back(key(c[0], c[1], c[2]));
    }
    std::sort(levels[0].begin(), levels[0].end());
    if (std::adjacent_find(levels[0].begin(), levels[0].end()) != levels[0].end())
        throw std::runtime_error("decode_tree_from_fine: duplicate fine voxel");
    for (int k = 1; k <= 4; ++k) {
        auto& L = levels[k];
        L.reserve(levels[k - 1].size() / 2 + 1);
        for (uint64_t v : levels[k - 1])
            L.push_back(key((int)(v >> 43), (int)((v >> 22) & 0xfffff), (int)((v >> 1) & 0xfffff)));
        std::sort(L.begin(), L.end());
        L.erase(std::unique(L.begin(), L.end()), L.end());
    }
    // The encoder's latent must be exactly the top level, or the tree belongs to another mesh.
    {
        std::vector<uint64_t> top;
        top.reserve(latent.size());
        for (const auto& c : latent) top.push_back(key(c[0], c[1], c[2]));
        std::sort(top.begin(), top.end());
        if (top != levels[4])
            throw std::runtime_error("decode_tree_from_fine: latent coords (" + std::to_string(latent.size()) +
                                     ") are not the fine voxels' level-4 parents (" +
                                     std::to_string(levels[4].size()) + ")");
    }
    // Walk top-down in sparse_c2s's order: parents in their current order, children by slot.
    DecodeTree t;
    std::vector<std::array<int,3>> cur = latent;
    for (int k = 3; k >= 0; --k) {
        const auto& L = levels[k];
        std::vector<uint8_t> mask((size_t)8 * cur.size(), 0);
        std::vector<std::array<int,3>> next;
        next.reserve(L.size());
        for (size_t i = 0; i < cur.size(); ++i) {
            const int x = cur[i][0], y = cur[i][1], z = cur[i][2];
            for (int o = 0; o < 8; ++o) {
                const int cx = 2 * x + (o & 1), cy = 2 * y + ((o >> 1) & 1), cz = 2 * z + ((o >> 2) & 1);
                if (!std::binary_search(L.begin(), L.end(), key(cx, cy, cz))) continue;
                mask[(size_t)o + 8 * i] = 1;
                next.push_back({ cx, cy, cz });
            }
        }
        if (next.size() != L.size())
            throw std::runtime_error("decode_tree_from_fine: stage " + std::to_string(3 - k) + " grew " +
                                     std::to_string(next.size()) + " voxels, the encoder level has " +
                                     std::to_string(L.size()));
        t.subs.push_back(std::move(mask));
        cur = std::move(next);
    }
    t.coords = std::move(cur);
    return t;
}

} // namespace trellis
