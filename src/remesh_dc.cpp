#include "remesh_dc.h"
#include "tri_bvh.h"
#include "mesh_print.h"
#include <algorithm>
#include <atomic>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trellis {

namespace {

inline int ctz64(uint64_t v) {
#ifdef _MSC_VER
    unsigned long i;
    _BitScanForward64(&i, v);
    return (int)i;
#else
    return __builtin_ctzll(v);
#endif
}

inline uint64_t key3(int x, int y, int z) {
    return ((uint64_t)(uint32_t)x << 42) | ((uint64_t)(uint32_t)y << 21) | (uint32_t)z;
}

void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    const int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> ts;
    const int64_t chunk = (n + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int64_t b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        ts.emplace_back(fn, b, e);
    }
    for (auto& t : ts) t.join();
}

// Exact squared Euclidean distance transform (Felzenszwalb & Huttenlocher), in place on a
// dx*dy*dz box (z fastest): on entry 0 at the sources and kEdtFar elsewhere, on return the
// squared distance in voxels to the nearest source.
constexpr float kEdtFar = 1e18f;

void edt_sq(std::vector<float>& g, int64_t dx, int64_t dy, int64_t dz) {
    const int64_t dims[3] = {dx, dy, dz};
    const int64_t stride[3] = {dy * dz, dz, 1};
    for (int axis = 2; axis >= 0; --axis) {
        const int64_t len = dims[axis], st = stride[axis];
        const int a = (axis + 1) % 3, b = (axis + 2) % 3;
        const int64_t lines = dims[a] * dims[b];
        parallel_for(lines, [&](int64_t lb, int64_t le) {
            std::vector<float> f((size_t)len), d((size_t)len), z((size_t)len + 1);
            std::vector<int64_t> v((size_t)len);
            for (int64_t line = lb; line < le; ++line) {
                const int64_t base = (line / dims[b]) * stride[a] + (line % dims[b]) * stride[b];
                bool any = false;
                for (int64_t q = 0; q < len; ++q) { f[(size_t)q] = g[(size_t)(base + q * st)]; any |= f[(size_t)q] < kEdtFar; }
                if (!any) continue;
                int64_t k = 0;
                v[0] = 0; z[0] = -kEdtFar; z[1] = kEdtFar;
                auto meet = [&](int64_t q, int64_t p) {   // where parabolas q and p cross
                    return ((f[(size_t)q] + (float)(q * q)) - (f[(size_t)p] + (float)(p * p))) / (float)(2 * (q - p));
                };
                for (int64_t q = 1; q < len; ++q) {
                    float s = meet(q, v[(size_t)k]);
                    while (s <= z[(size_t)k]) { --k; s = meet(q, v[(size_t)k]); }   // z[0] = -far stops it
                    ++k;
                    v[(size_t)k] = q; z[(size_t)k] = s; z[(size_t)k + 1] = kEdtFar;
                }
                k = 0;
                for (int64_t q = 0; q < len; ++q) {
                    while (z[(size_t)k + 1] < (float)q) ++k;
                    const int64_t p = v[(size_t)k];
                    d[(size_t)q] = std::min(kEdtFar, (float)((q - p) * (q - p)) + f[(size_t)p]);
                }
                for (int64_t q = 0; q < len; ++q) g[(size_t)(base + q * st)] = d[(size_t)q];
            }
        });
    }
}

}  // namespace

Mesh remesh_narrow_band_dc(const float* iverts, int64_t iV, const int32_t* ifaces, int64_t iF,
                           const TriBvh& bvh, int res, int band, float project_back) {
    RemeshOptions options;
    options.band = band;
    options.project_back = project_back;
    return remesh_narrow_band_dc(iverts, iV, ifaces, iF, bvh, res, options);
}

Mesh remesh_narrow_band_dc(const float* iverts, int64_t iV, const int32_t* ifaces, int64_t iF,
                           const TriBvh& bvh, int res, const RemeshOptions& options) {
    (void)iV;
    Mesh out;
    if (iF == 0 || bvh.empty() || res <= 0) return out;
    const int band = std::max(1, options.band);
    const float project_back = options.project_back;
    const bool solid = options.solid;

    // Reference domain: the world cube is inflated by (res+3·band)/res so the
    // offset shell never touches the boundary; eps is the offset distance.
    const float scale = (float)(res + 3 * band) / (float)res;
    const float cell = scale / (float)res;
    const float eps = (float)band * cell;
    const float keep = 0.87f * cell;

    // Candidate cells: conservative dilation of every triangle's AABB by the
    // band-plus-crossing radius, marked in a res^3 bitset.
    const int64_t nbits = (int64_t)res * res * res;
    std::vector<uint64_t> cand((size_t)((nbits + 63) / 64), 0);
    auto bit_set = [&cand, res](int x, int y, int z) {
        const int64_t i = ((int64_t)x * res + y) * res + z;
        cand[(size_t)(i >> 6)] |= 1ull << (i & 63);
    };
    auto bit_get = [&cand, res](int x, int y, int z) -> bool {
        const int64_t i = ((int64_t)x * res + y) * res + z;
        return (cand[(size_t)(i >> 6)] >> (i & 63)) & 1;
    };
    // A cell is active iff UDF(center) < (band+0.87)·cell, and its closest
    // surface point lies inside some triangle-marked cell, so the per-axis
    // index distance is < band+1.37, i.e. ≤ band+1.
    const int dil = band + 1;
    {
        const int F = (int)iF;
        std::vector<std::vector<uint64_t>> parts;
        const int hw = (int)std::max(1u, std::thread::hardware_concurrency());
        // Each worker previously allocated a full res^3 candidate bitset.  At
        // res=1024 that is 128 MiB per worker, so a 32-thread CPU consumed ~4 GiB
        // here before any geometry/BVH memory.  Cap only the replication memory;
        // the OR result and therefore the remesh are bit-for-bit equivalent.
        const size_t bytes_per_part = cand.size() * sizeof(uint64_t);
        const size_t parts_budget = (size_t)1024 * 1024 * 1024; // 1 GiB
        const int mem_workers = bytes_per_part ? (int)std::max<size_t>(1, parts_budget / bytes_per_part) : hw;
        const int nt = std::max(1, std::min(hw, mem_workers));
        if (nt < hw) {
            printf("  [remesh-mem] candidate bitset %.1f MiB/worker, workers %d->%d\n",
                   bytes_per_part / (1024.0*1024.0), hw, nt);
            fflush(stdout);
        }
        parts.assign(nt, {});
        std::vector<std::thread> ts;
        const int chunk = (F + nt - 1) / nt;
        for (int t = 0; t < nt; ++t) {
            const int b = t * chunk, e = std::min(F, b + chunk);
            if (b >= e) break;
            parts[t].assign(cand.size(), 0);
            ts.emplace_back([&, t, b, e]() {
                auto& bits = parts[t];
                auto setb = [&bits, res](int x, int y, int z) {
                    const int64_t i = ((int64_t)x * res + y) * res + z;
                    bits[(size_t)(i >> 6)] |= 1ull << (i & 63);
                };
                for (int f = b; f < e; ++f) {
                    float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
                    for (int j = 0; j < 3; ++j) {
                        const float* p = &iverts[3 * ifaces[3*f+j]];
                        for (int k = 0; k < 3; ++k) {
                            bmin[k] = std::min(bmin[k], p[k]);
                            bmax[k] = std::max(bmax[k], p[k]);
                        }
                    }
                    int c0[3], c1[3];
                    for (int k = 0; k < 3; ++k) {
                        c0[k] = std::max(0, (int)std::floor((bmin[k] / scale + 0.5f) * res) - dil);
                        c1[k] = std::min(res - 1, (int)std::floor((bmax[k] / scale + 0.5f) * res) + dil);
                    }
                    for (int x = c0[0]; x <= c1[0]; ++x)
                        for (int y = c0[1]; y <= c1[1]; ++y)
                            for (int z = c0[2]; z <= c1[2]; ++z) setb(x, y, z);
                }
            });
        }
        for (auto& th : ts) th.join();
        for (auto& bits : parts)
            if (!bits.empty())
                for (size_t i = 0; i < cand.size(); ++i) cand[i] |= bits[i];
    }

    // Solid mode: one bit per grid VERTEX (n = res+1 per axis). `wall` marks vertices
    // closer than eps to the surface; it later becomes "not exterior" (see below).
    const int64_t n = (int64_t)res + 1;
    std::vector<uint64_t> wall;
    auto vbit = [n](int x, int y, int z) -> int64_t { return ((int64_t)x * n + y) * n + z; };
    if (solid) wall.assign((size_t)((n * n * n + 63) / 64), 0);

    // Active voxels: |UDF(center) - eps| < 0.87*cell (spec 27 §4.1).
    std::vector<int> acoord;
    {
        std::vector<int64_t> cand_cells;
        for (int64_t w = 0; w < (int64_t)cand.size(); ++w) {
            uint64_t bits = cand[w];
            while (bits) {
                const int b = ctz64(bits);
                bits &= bits - 1;
                cand_cells.push_back((w << 6) | b);
            }
        }
        std::vector<uint8_t> act(cand_cells.size(), 0);
        parallel_for((int64_t)cand_cells.size(), [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                const int64_t c = cand_cells[i];
                const int x = (int)(c / ((int64_t)res * res)), y = (int)((c / res) % res), z = (int)(c % res);
                if (solid) {
                    // Every vertex within eps of the surface is the min corner of a
                    // candidate cell (its closest point sits in a triangle-marked cell at
                    // most `band` away per axis, and candidates are dilated by band+1).
                    const float q[3] = { ((float)x / res - 0.5f) * scale,
                                         ((float)y / res - 0.5f) * scale,
                                         ((float)z / res - 0.5f) * scale };
                    const TriBvh::Hit w = bvh.closest(q, eps);
                    if (w.face >= 0 && w.dist2 < eps * eps) act[i] |= 2;
                }
                const float p[3] = { ((x + 0.5f) / res - 0.5f) * scale,
                                     ((y + 0.5f) / res - 0.5f) * scale,
                                     ((z + 0.5f) / res - 0.5f) * scale };
                const TriBvh::Hit h = bvh.closest(p, eps + keep);
                if (h.face < 0) continue;
                const float f = std::sqrt(h.dist2) - eps;
                if (std::fabs(f) < keep) act[i] |= 1;
            }
        });
        for (size_t i = 0; i < cand_cells.size(); ++i) {
            const int64_t c = cand_cells[i];
            if (act[i] & 2) {
                const int64_t b = vbit((int)(c / ((int64_t)res * res)), (int)((c / res) % res), (int)(c % res));
                wall[(size_t)(b >> 6)] |= 1ull << (b & 63);
            }
            if (!(act[i] & 1)) continue;
            acoord.push_back((int)(c / ((int64_t)res * res)));
            acoord.push_back((int)((c / res) % res));
            acoord.push_back((int)(c % res));
        }
        cand.clear(); cand.shrink_to_fit();
    }
    const int64_t Na = (int64_t)acoord.size() / 3;
    if (Na < 100) {
        fprintf(stderr, "  remesh: only %lld active voxels; skipping remesh\n", (long long)Na);
        return out;
    }

    std::unordered_map<uint64_t, int> vox;
    vox.reserve((size_t)Na * 2);
    for (int64_t i = 0; i < Na; ++i) vox.emplace(key3(acoord[3*i], acoord[3*i+1], acoord[3*i+2]), (int)i);

    // f = UDF - eps at the grid VERTICES (corner mapping v/res, spec 27 §4.3).
    std::vector<int> vcoord;
    std::unordered_map<uint64_t, int> vmap;
    vmap.reserve((size_t)Na * 3);
    for (int64_t i = 0; i < Na; ++i)
        for (int dx = 0; dx < 2; ++dx) for (int dy = 0; dy < 2; ++dy) for (int dz = 0; dz < 2; ++dz) {
            const int x = acoord[3*i] + dx, y = acoord[3*i+1] + dy, z = acoord[3*i+2] + dz;
            if (vmap.emplace(key3(x, y, z), (int)(vcoord.size() / 3)).second) {
                vcoord.push_back(x); vcoord.push_back(y); vcoord.push_back(z);
            }
        }
    const int64_t Nv = (int64_t)vcoord.size() / 3;
    std::vector<float> fvert((size_t)Nv);
    parallel_for(Nv, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const float p[3] = { ((float)vcoord[3*i]   / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+1] / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+2] / res - 0.5f) * scale };
            // Crossing edges always have their far endpoint under eps+cell
            // (f changes at most one cell-length per edge), so this bound
            // never clips a value that feeds the crossing interpolation.
            const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
            fvert[i] = (h.face >= 0 ? std::sqrt(h.dist2) : eps + 2 * cell) - eps;
        }
    });
    auto fval = [&](int x, int y, int z) -> float {
        auto it = vmap.find(key3(x, y, z));
        return it == vmap.end() ? 1e9f : fvert[it->second];
    };

    // ---- solid mode: signed field from an exterior flood fill ------------------------
    int base_axis = -1, base_layer = 0;   // vertices with index < base_layer are cut away
    // Grid vertices whose field value is invented rather than measured: re-signed by the
    // fill, or outside the narrow band altogether. A crossing next to one is a cap vertex.
    std::vector<uint8_t> synthetic;
    if (solid) {
        // The contoured field decides the sign wherever it is known, so the flood and the
        // crossings cannot disagree by a rounding ulp.
        for (int64_t i = 0; i < Nv; ++i) {
            const int64_t b = vbit(vcoord[3*i], vcoord[3*i+1], vcoord[3*i+2]);
            if (fvert[i] < 0) wall[(size_t)(b >> 6)] |= 1ull << (b & 63);
            else              wall[(size_t)(b >> 6)] &= ~(1ull << (b & 63));
        }

        // Openness vote. The decoded surface is torn (thousands of holes, some tens of
        // cells wide), and a flood alone walks in through any tear wider than the band.
        // A vertex that sees the surface along nearly every lattice direction is interior
        // no matter how it could be reached: on a coarse copy of the wall (about 256^3),
        // count for each vertex the rays, out of the 26 axis and diagonal directions, that
        // leave the grid without meeting the wall; vertices with fewer than `seal` open
        // rays are shut to the flood. Under a tear only the ray through it is open, so the
        // tear is capped; a real opening (a cup, the gap between two legs) shows a cone of
        // sky and stays open.
        int64_t shut_coarse = 0, released_coarse = 0;
        if (options.seal > 0) {
            const int step = std::max(1, res / 256);
            const int m = (int)((n + step - 1) / step);
            const size_t m3 = (size_t)m * m * m;
            std::vector<uint8_t> coarse(m3, 0);
            for (size_t w = 0; w < wall.size(); ++w) {
                uint64_t bits = wall[w];
                while (bits) {
                    const int64_t b = (int64_t)(w << 6) | ctz64(bits);
                    bits &= bits - 1;
                    const int64_t x = b / (n * n), y = (b / n) % n, z = b % n;
                    coarse[((size_t)(x / step) * m + (size_t)(y / step)) * m + (size_t)(z / step)] = 1;
                }
            }
            std::vector<uint8_t> open(m3, 0);
            {
                int dirs[26][3], nd = 0;
                for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz)
                    if (dx || dy || dz) { dirs[nd][0] = dx; dirs[nd][1] = dy; dirs[nd][2] = dz; ++nd; }
                const int nt = (int)std::min<unsigned>(26, std::max(1u, std::thread::hardware_concurrency()));
                std::vector<std::vector<uint8_t>> counts((size_t)nt);
                std::vector<std::thread> workers;
                for (int t = 0; t < nt; ++t)
                    workers.emplace_back([&, t]() {
                        std::vector<uint8_t>& count = counts[(size_t)t];
                        count.assign(m3, 0);
                        std::vector<uint8_t> blocked(m3);
                        for (int k = t; k < 26; k += nt) {
                            const int* d = dirs[k];
                            // blocked(v) = the ray from v toward -d meets the wall; the
                            // predecessor v - d is always visited first.
                            for (int i = 0; i < m; ++i) {
                                const int x = d[0] >= 0 ? i : m - 1 - i, px = x - d[0];
                                for (int j = 0; j < m; ++j) {
                                    const int y = d[1] >= 0 ? j : m - 1 - j, py = y - d[1];
                                    for (int l = 0; l < m; ++l) {
                                        const int z = d[2] >= 0 ? l : m - 1 - l, pz = z - d[2];
                                        const size_t v = ((size_t)x * m + y) * m + z;
                                        if (px < 0 || px >= m || py < 0 || py >= m || pz < 0 || pz >= m) { blocked[v] = 0; continue; }
                                        const size_t q = ((size_t)px * m + py) * m + pz;
                                        blocked[v] = coarse[q] | blocked[q];
                                    }
                                }
                            }
                            for (size_t v = 0; v < m3; ++v) count[v] += !blocked[v];
                        }
                    });
                for (auto& w : workers) w.join();
                for (const auto& count : counts)
                    for (size_t v = 0; v < m3; ++v) open[v] += count[v];
            }
            std::vector<uint8_t> occupied;   // the coarse wall, before it becomes the shut mask
            if (options.open_ball_cells > 0) occupied = coarse;
            for (size_t v = 0; v < m3; ++v) {
                coarse[v] = open[v] < options.seal;
                shut_coarse += coarse[v];
            }
            std::vector<uint8_t>().swap(open);

            // Vessels. The floor of a deep cup sees the sky through one narrow cone, like the
            // inside of a figurine behind a tear, and the vote shuts both. What tells them
            // apart is the width of the opening: free space that a ball of radius R can reach
            // from outside without touching the wall is exterior, so a mouth wider than 2R
            // lets the ball in and a tear narrower than that does not. On the coarse grid,
            // padded so a ball fits above a model that fills the grid: ball centres are the
            // free voxels at least R from the wall, flooded from the padding; the space they
            // sweep (centres dilated by R) is released from the shut mask. Inner corners keep
            // a fillet of radius R that the ball cannot reach.
            if (options.open_ball_cells > 0) {
                const float R = options.open_ball_cells / (float)step;   // coarse cells
                const int64_t P = (int64_t)std::ceil(R) + 1, M = m + 2 * P;
                const size_t M3 = (size_t)(M * M * M);
                auto at = [M](int64_t x, int64_t y, int64_t z) { return (size_t)((x * M + y) * M + z); };
                std::vector<float> d(M3, kEdtFar);
                for (int64_t x = 0; x < m; ++x)
                    for (int64_t y = 0; y < m; ++y)
                        for (int64_t z = 0; z < m; ++z)
                            if (occupied[((size_t)x * m + (size_t)y) * m + (size_t)z]) d[at(x + P, y + P, z + P)] = 0.0f;
                edt_sq(d, M, M, M);
                // Flood the centres from the padding's corner (free, and R+1 from everything).
                const float R2 = R * R;
                std::vector<uint8_t> reached(M3, 0);
                std::vector<int64_t> stack{0};
                reached[0] = 1;
                while (!stack.empty()) {
                    const int64_t s = stack.back();
                    stack.pop_back();
                    const int64_t x = s / (M * M), y = (s / M) % M, z = s % M;
                    const int64_t nb[6][3] = {{x-1,y,z},{x+1,y,z},{x,y-1,z},{x,y+1,z},{x,y,z-1},{x,y,z+1}};
                    for (const auto& q : nb) {
                        if (q[0] < 0 || q[0] >= M || q[1] < 0 || q[1] >= M || q[2] < 0 || q[2] >= M) continue;
                        const size_t t = at(q[0], q[1], q[2]);
                        if (reached[t] || d[t] < R2) continue;
                        reached[t] = 1;
                        stack.push_back((int64_t)t);
                    }
                }
                for (size_t v = 0; v < M3; ++v) d[v] = reached[v] ? 0.0f : kEdtFar;
                std::vector<uint8_t>().swap(reached);
                edt_sq(d, M, M, M);
                int64_t released = 0;
                for (int64_t x = 0; x < m; ++x)
                    for (int64_t y = 0; y < m; ++y)
                        for (int64_t z = 0; z < m; ++z) {
                            const size_t v = ((size_t)x * m + (size_t)y) * m + (size_t)z;
                            if (coarse[v] && !occupied[v] && d[at(x + P, y + P, z + P)] <= R2) { coarse[v] = 0; ++released; }
                        }
                shut_coarse -= released;
                released_coarse = released;
            }

            // wall := wall | shut
            for (int64_t x = 0; x < n; ++x)
                for (int64_t y = 0; y < n; ++y) {
                    const uint8_t* row = &coarse[((size_t)(x / step) * m + (size_t)(y / step)) * m];
                    const int64_t b0 = (x * n + y) * n;
                    for (int64_t z = 0; z < n; ++z)
                        if (row[z / step]) wall[(size_t)((b0 + z) >> 6)] |= 1ull << ((b0 + z) & 63);
                }
        }

        // Exterior = open vertices 6-connected to the domain boundary (the inflated domain
        // keeps every boundary vertex free). Scanline flood along z, the bitset's fast axis.
        // `seen` = wall | shut | visited, so one lookup answers "may the flood enter here".
        // Called again after anything that adds solid, so no pocket of exterior is left
        // enclosed (an enclosed pocket would be an inside-out shell in the output).
        auto fill_unreachable = [&]() {
        std::vector<uint64_t> seen = wall;
        auto is_seen = [&seen](int64_t b) -> bool { return (seen[(size_t)(b >> 6)] >> (b & 63)) & 1; };
        std::vector<int64_t> stack;
        stack.push_back(0);
        while (!stack.empty()) {
            const int64_t s = stack.back();
            stack.pop_back();
            if (is_seen(s)) continue;
            const int64_t row = s / n;
            int64_t lo = s % n, hi = lo;
            while (lo > 0 && !is_seen(row * n + lo - 1)) --lo;
            while (hi < n - 1 && !is_seen(row * n + hi + 1)) ++hi;
            for (int64_t z = lo; z <= hi; ++z) {
                const int64_t b = row * n + z;
                seen[(size_t)(b >> 6)] |= 1ull << (b & 63);
            }
            const int64_t x = row / n, y = row % n;
            const int64_t nb[4][2] = {{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}};
            for (const auto& q : nb) {
                if (q[0] < 0 || q[0] >= n || q[1] < 0 || q[1] >= n) continue;
                const int64_t r2 = (q[0] * n + q[1]) * n;
                bool run = false;
                for (int64_t z = lo; z <= hi; ++z) {
                    const bool open = !is_seen(r2 + z);
                    if (open && !run) stack.push_back(r2 + z);
                    run = open;
                }
            }
        }
        // wall := solid = everything the flood did not reach (band, shut, cavities).
        for (size_t w = 0; w < wall.size(); ++w) wall[w] |= ~seen[w];
        };
        fill_unreachable();
        auto is_solid = [&wall, &vbit](int x, int y, int z) -> bool {
            const int64_t b = vbit(x, y, z);
            return (wall[(size_t)(b >> 6)] >> (b & 63)) & 1;
        };

        // Thin features. The band guarantees 2*band cells of material; a fin, a blade edge
        // or a strand thinner than `min_wall_cells` is grown, and only it. One cell per
        // side per pass:
        //   depth  = chamfer distance of each solid vertex to the exterior (near the surface);
        //   core   = solid deeper than half the wall: material that is already thick enough;
        //   thin   = shallow solid farther than 1.6*r from any core. (A flat thick surface is r
        //            from its core; the extra 0.6*r keeps the
        //            sliver that an opening shaves off an ordinary convex edge out of it.)
        // and thin vertices take their exterior 6-neighbours. The new surface lies outside
        // the narrow band, so it is extracted and smoothed with the caps.
        int64_t thickened = 0;
        int thicken_passes = 0;
        if (options.min_wall_cells > 2.0f * (float)band + 0.5f) {
            if (n * n * n > 1400000000ll) {
                printf("  remesh_solid: grid too large for wall thickening (res %d); skipped\n", res);
            } else {
                const float r = 0.5f * options.min_wall_cells;          // cells
                const int deep = (int)std::floor(3.0f * r);             // chamfer units (3 = one cell)
                const int cap = deep + 5;
                const int near_core = (int)std::ceil(1.6f * 3.0f * r);
                std::vector<uint8_t> depth((size_t)(n * n * n));
                auto solid_at = [&wall](int64_t b) -> bool { return (wall[(size_t)(b >> 6)] >> (b & 63)) & 1; };
                const int max_passes = (int)std::ceil(r) - band + 1;
                for (int pass = 0; pass < max_passes; ++pass) {
                    std::fill(depth.begin(), depth.end(), (uint8_t)0);
                    // Surface layer: solid vertices with an exterior 6-neighbour.
                    std::vector<std::vector<int64_t>> rim((size_t)std::max(1u, std::thread::hardware_concurrency()) + 1);
                    {
                        std::atomic<int> next_part{0};
                        parallel_for(n, [&](int64_t xb, int64_t xe) {
                            std::vector<int64_t>& part = rim[(size_t)next_part.fetch_add(1)];
                            for (int64_t x = xb; x < xe; ++x)
                                for (int64_t y = 0; y < n; ++y)
                                    for (int64_t z = 0; z < n; ++z) {
                                        const int64_t b = (x * n + y) * n + z;
                                        if (!solid_at(b)) continue;
                                        if ((x > 0 && !solid_at(b - n * n)) || (x < n - 1 && !solid_at(b + n * n)) ||
                                            (y > 0 && !solid_at(b - n)) || (y < n - 1 && !solid_at(b + n)) ||
                                            (z > 0 && !solid_at(b - 1)) || (z < n - 1 && !solid_at(b + 1)))
                                            part.push_back(b);
                                    }
                        });
                    }
                    // Chamfer flood inward; first pop of a vertex is its shortest distance.
                    std::vector<std::vector<int64_t>> bucket((size_t)cap + 6);
                    for (auto& part : rim) { bucket[3].insert(bucket[3].end(), part.begin(), part.end()); std::vector<int64_t>().swap(part); }
                    std::vector<int64_t> shallow;
                    auto neighbours = [&](int64_t b, const std::function<void(int64_t, int)>& visit) {
                        const int64_t x = b / (n * n), y = (b / n) % n, z = b % n;
                        for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
                            const int steps = (dx != 0) + (dy != 0) + (dz != 0);
                            if (!steps) continue;
                            const int64_t X = x + dx, Y = y + dy, Z = z + dz;
                            if (X < 0 || X >= n || Y < 0 || Y >= n || Z < 0 || Z >= n) continue;
                            visit((X * n + Y) * n + Z, 2 + steps);
                        }
                    };
                    for (int d = 3; d <= cap; ++d) {
                        for (size_t k = 0; k < bucket[(size_t)d].size(); ++k) {
                            const int64_t b = bucket[(size_t)d][k];
                            if (depth[(size_t)b]) continue;
                            depth[(size_t)b] = (uint8_t)d;
                            if (d <= deep) shallow.push_back(b);
                            neighbours(b, [&](int64_t q, int w) {
                                if (d + w <= cap && solid_at(q) && !depth[(size_t)q]) bucket[(size_t)(d + w)].push_back(q);
                            });
                        }
                        std::vector<int64_t>().swap(bucket[(size_t)d]);
                    }
                    // Shallow vertices within reach of a core vertex (depth > deep, or never
                    // reached = deeper than the cap) are marked with bit 7.
                    bucket.assign((size_t)near_core + 6, {});
                    for (const int64_t b : shallow)
                        neighbours(b, [&](int64_t q, int w) {
                            if (solid_at(q) && (depth[(size_t)q] == 0 || (depth[(size_t)q] & 0x7f) > deep) && w <= near_core)
                                bucket[(size_t)w].push_back(b);
                        });
                    for (int d = 3; d <= near_core; ++d) {
                        for (size_t k = 0; k < bucket[(size_t)d].size(); ++k) {
                            const int64_t b = bucket[(size_t)d][k];
                            if (depth[(size_t)b] & 0x80) continue;
                            depth[(size_t)b] |= 0x80;
                            neighbours(b, [&](int64_t q, int w) {
                                if (d + w > near_core || !solid_at(q)) return;
                                const uint8_t dq = depth[(size_t)q];
                                if (dq == 0 || (dq & 0x80) || (dq & 0x7f) > deep) return;
                                bucket[(size_t)(d + w)].push_back(q);
                            });
                        }
                        std::vector<int64_t>().swap(bucket[(size_t)d]);
                    }
                    int64_t grown = 0;
                    for (const int64_t b : shallow) {
                        if (depth[(size_t)b] & 0x80) continue;   // supported by thick material
                        const int64_t x = b / (n * n), y = (b / n) % n, z = b % n;
                        const int64_t q[6] = {x > 1 ? b - n * n : -1, x < n - 2 ? b + n * n : -1, y > 1 ? b - n : -1,
                                              y < n - 2 ? b + n : -1, z > 1 ? b - 1 : -1, z < n - 2 ? b + 1 : -1};
                        for (const int64_t t : q)
                            if (t >= 0 && !solid_at(t)) { wall[(size_t)(t >> 6)] |= 1ull << (t & 63); ++grown; }
                    }
                    if (!grown) break;
                    thickened += grown;
                    ++thicken_passes;
                }
                if (thickened) fill_unreachable();
            }
        }

        // Necks. The thin-wall rule leaves alone anything near thick material, so a short
        // pinch between two thick parts (a wrist, an ankle, a head on a thin neck) is never
        // grown, and it is where a print breaks. With R = half the target diameter:
        //   core    = solid deeper than R (exact distance to the exterior);
        //   parts   = connected components of the core, ignoring those smaller than a ball
        //             of radius 2R minus the erosion (a part at least twice the neck);
        //   joints  = where the geodesic fronts grown from different parts through the
        //             shallow solid meet; a joint whose cross-section is under a disc of
        //             radius R is a neck (a wide plate between two blocks is not);
        //   fix     = a tube of radius R along the shortest path from one part, through the
        //             joint, to the other.
        // On a grid of step max(1, res/512), restricted to the solid's bounding box.
        int necks_found = 0, necks_fixed = 0;
        if (options.min_neck_cells > 0) {
            const int s = std::max(1, res / 512);
            const float R = 0.5f * options.min_neck_cells / (float)s;   // sub-grid cells
            int64_t lo[3] = {n, n, n}, hi[3] = {-1, -1, -1};
            for (size_t w = 0; w < wall.size(); ++w) {
                uint64_t bits = wall[w];
                while (bits) {
                    const int64_t b = (int64_t)(w << 6) | ctz64(bits);
                    bits &= bits - 1;
                    const int64_t c[3] = {b / (n * n), (b / n) % n, b % n};
                    for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], c[k]); hi[k] = std::max(hi[k], c[k]); }
                }
            }
            int64_t o[3], dim[3];
            for (int k = 0; k < 3; ++k) {   // sub-grid box with one exterior voxel all round
                o[k] = std::max<int64_t>(0, lo[k] / s - 1);
                dim[k] = std::min<int64_t>((n - 1) / s, hi[k] / s + 1) - o[k] + 1;
            }
            const size_t N3 = (size_t)(dim[0] * dim[1] * dim[2]);
            auto sid = [&](int64_t x, int64_t y, int64_t z) { return (size_t)((x * dim[1] + y) * dim[2] + z); };
            auto fine_solid = [&](int64_t x, int64_t y, int64_t z) {
                return is_solid((int)((x + o[0]) * s), (int)((y + o[1]) * s), (int)((z + o[2]) * s));
            };
            std::vector<int32_t> label(N3);   // -1 exterior, 0 shallow, k > 0 part k
            {
                std::vector<float> d(N3);
                parallel_for(dim[0], [&](int64_t xb, int64_t xe) {
                    for (int64_t x = xb; x < xe; ++x)
                        for (int64_t y = 0; y < dim[1]; ++y)
                            for (int64_t z = 0; z < dim[2]; ++z)
                                d[sid(x, y, z)] = fine_solid(x, y, z) ? kEdtFar : 0.0f;
                });
                edt_sq(d, dim[0], dim[1], dim[2]);
                const float R2 = R * R;
                for (size_t v = 0; v < N3; ++v) label[v] = d[v] == 0.0f ? -1 : d[v] > R2 ? -2 : 0;   // -2: core, unlabelled
            }
            // Parts: 26-connected core components.
            const double min_part = 4.0 / 3.0 * 3.14159265 * std::pow(std::max(1.0f, R), 3.0);   // ball of radius 2R, eroded by R
            std::vector<int64_t> part_size{0};
            std::vector<int64_t> stack;
            auto for_nb26 = [&](int64_t v, const std::function<void(int64_t)>& f) {
                const int64_t x = v / (dim[1] * dim[2]), y = (v / dim[2]) % dim[1], z = v % dim[2];
                for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
                    if (!dx && !dy && !dz) continue;
                    const int64_t X = x + dx, Y = y + dy, Z = z + dz;
                    if (X < 0 || X >= dim[0] || Y < 0 || Y >= dim[1] || Z < 0 || Z >= dim[2]) continue;
                    f((int64_t)sid(X, Y, Z));
                }
            };
            for (size_t v0 = 0; v0 < N3; ++v0) {
                if (label[v0] != -2) continue;
                const int32_t k = (int32_t)part_size.size();
                int64_t count = 0;
                label[v0] = k;
                stack.assign(1, (int64_t)v0);
                while (!stack.empty()) {
                    const int64_t v = stack.back();
                    stack.pop_back();
                    ++count;
                    for_nb26(v, [&](int64_t q) { if (label[(size_t)q] == -2) { label[(size_t)q] = k; stack.push_back(q); } });
                }
                part_size.push_back(count);
            }
            // Small parts count as shallow solid. Fronts: breadth-first from every part at
            // once through the shallow solid (6-connected); owner and distance per voxel.
            std::vector<int32_t> owner(N3, 0);
            std::vector<uint16_t> dist(N3, 0);
            std::vector<int64_t> front, next;
            int parts = 0;
            for (size_t k = 1; k < part_size.size(); ++k) parts += (double)part_size[k] >= min_part;
            for (size_t v = 0; v < N3; ++v) {
                if (label[v] <= 0) continue;
                if ((double)part_size[(size_t)label[v]] < min_part) { label[v] = 0; continue; }
                owner[v] = label[v];
            }
            for (size_t v = 0; v < N3; ++v) {   // the parts' surfaces start the fronts
                if (owner[v] <= 0) continue;
                bool edge = false;
                for_nb26((int64_t)v, [&](int64_t q) { edge |= label[(size_t)q] == 0; });
                if (edge) front.push_back((int64_t)v);
            }
            const int64_t step6[6] = {dim[1] * dim[2], -dim[1] * dim[2], dim[2], -dim[2], 1, -1};
            for (uint16_t dd = 1; !front.empty() && dd < 65535; ++dd) {
                next.clear();
                for (const int64_t v : front)
                    for (const int64_t st : step6) {
                        const int64_t q = v + st;
                        if (q < 0 || q >= (int64_t)N3 || label[(size_t)q] != 0 || owner[(size_t)q]) continue;
                        owner[(size_t)q] = owner[(size_t)v];
                        dist[(size_t)q] = dd;
                        next.push_back(q);
                    }
                front.swap(next);
            }
            // Joints: 6-adjacent shallow voxels owned by different parts. Per pair, the
            // number of such faces (the joint's cross-section) and the shortest crossing.
            struct Joint { int64_t faces = 0; int64_t a = -1, b = -1; int best = 1 << 30; };
            std::unordered_map<uint64_t, Joint> joints;
            for (size_t v = 0; v < N3; ++v) {
                if (label[v] != 0 || owner[v] <= 0) continue;
                for (int k = 0; k < 6; k += 2) {   // + directions only: each face once
                    const int64_t q = (int64_t)v + step6[k];
                    if (q >= (int64_t)N3 || label[(size_t)q] != 0 || owner[(size_t)q] <= 0 || owner[(size_t)q] == owner[v]) continue;
                    int64_t a = (int64_t)v, b = q;
                    if (owner[(size_t)a] > owner[(size_t)b]) std::swap(a, b);
                    Joint& j = joints[((uint64_t)(uint32_t)owner[(size_t)a] << 32) | (uint32_t)owner[(size_t)b]];
                    ++j.faces;
                    const int len = (int)dist[(size_t)a] + (int)dist[(size_t)b];
                    if (len < j.best) { j.best = len; j.a = a; j.b = b; }
                }
            }
            const double max_faces = 3.14159265 * R * R;
            const float Rf = 0.5f * options.min_neck_cells;   // tube radius, fine cells
            const int ri = (int)std::ceil(Rf);
            int64_t stamped = 0;
            auto stamp = [&](int64_t v) {   // a ball of radius Rf at sub-grid voxel v
                const int64_t cx = (v / (dim[1] * dim[2]) + o[0]) * s, cy = ((v / dim[2]) % dim[1] + o[1]) * s, cz = (v % dim[2] + o[2]) * s;
                for (int dx = -ri; dx <= ri; ++dx) for (int dy = -ri; dy <= ri; ++dy) for (int dz = -ri; dz <= ri; ++dz) {
                    if ((float)(dx * dx + dy * dy + dz * dz) > Rf * Rf) continue;
                    const int64_t x = cx + dx, y = cy + dy, z = cz + dz;
                    if (x < 1 || x > n - 2 || y < 1 || y > n - 2 || z < 1 || z > n - 2) continue;   // keep the boundary free
                    const int64_t b = vbit((int)x, (int)y, (int)z);
                    if (!((wall[(size_t)(b >> 6)] >> (b & 63)) & 1)) { wall[(size_t)(b >> 6)] |= 1ull << (b & 63); ++stamped; }
                }
            };
            auto walk_home = [&](int64_t v) {   // down the distance field to the owning part
                for (;;) {
                    stamp(v);
                    if (label[(size_t)v] > 0) return;
                    int64_t best = -1;
                    for (const int64_t st : step6) {
                        const int64_t q = v + st;
                        if (q < 0 || q >= (int64_t)N3 || owner[(size_t)q] != owner[(size_t)v]) continue;
                        if (label[(size_t)q] > 0) { best = q; break; }
                        if (label[(size_t)q] == 0 && dist[(size_t)q] + 1 == dist[(size_t)v]) best = q;
                    }
                    if (best < 0) return;
                    v = best;
                }
            };
            for (const auto& kv : joints) {
                ++necks_found;
                if ((double)kv.second.faces >= max_faces) continue;   // a plate, not a neck
                walk_home(kv.second.a);
                walk_home(kv.second.b);
                ++necks_fixed;
            }
            if (stamped) fill_unreachable();
            printf("  remesh_solid: necks under %.1f cells: %d parts, %d joints, %d thickened (%lld vertices)\n",
                   options.min_neck_cells, parts, necks_found, necks_fixed, (long long)stamped);
        }

        // Flat base: the solid was classified on the whole object first (cutting before
        // the flood would let it in through the cut and hollow the model out); now every
        // vertex below the plane becomes exterior.
        if (options.base_axis >= 0 && options.base_axis < 3) {
            base_axis = options.base_axis;
            const float g = (options.base_plane / scale + 0.5f) * (float)res;   // plane, grid units
            base_layer = std::min(res - 1, std::max(1, (int)std::lround(g + 0.5f)));
            // Stability (base_auto): the lowest cut, between base_plane and base_plane_max,
            // whose footprint (convex hull of the first kept layer) holds the centre of mass
            // with `base_tip_deg` of tilt to spare. If none does, the lowest cut is kept and
            // a pedestal is added under it: the convex hull of the footprint and of a circle
            // around the centre of mass wide enough for that tilt.
            std::vector<std::pair<int, int>> pedestal;   // hull, counter-clockwise, closed
            int pedestal_layers = 0;
            if (options.base_auto) {
                const float gmax = (options.base_plane_max / scale + 0.5f) * (float)res;
                const int last = std::min(res - 1, std::max(base_layer, (int)std::lround(gmax + 0.5f)));
                const int u = (base_axis + 1) % 3, v = (base_axis + 2) % 3;
                std::vector<double> count((size_t)n, 0), sum_u((size_t)n, 0), sum_v((size_t)n, 0);
                std::vector<std::vector<std::pair<int, int>>> layer((size_t)(last - base_layer + 1));
                for (size_t w = 0; w < wall.size(); ++w) {
                    uint64_t bits = wall[w];
                    while (bits) {
                        const int64_t bit = (int64_t)(w << 6) | ctz64(bits);
                        bits &= bits - 1;
                        if (bit >= n * n * n) break;
                        const int c[3] = {(int)(bit / (n * n)), (int)((bit / n) % n), (int)(bit % n)};
                        const int l = c[base_axis];
                        count[(size_t)l] += 1; sum_u[(size_t)l] += c[u]; sum_v[(size_t)l] += c[v];
                        if (l >= base_layer && l <= last) layer[(size_t)(l - base_layer)].emplace_back(c[u], c[v]);
                    }
                }
                typedef std::pair<int, int> P2;
                auto convex_hull = [](std::vector<P2> pts) {
                    std::sort(pts.begin(), pts.end());
                    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
                    std::vector<P2> hull(2 * pts.size() + 1);
                    auto cross = [](const P2& o, const P2& p, const P2& q) {
                        return (int64_t)(p.first - o.first) * (q.second - o.second) - (int64_t)(p.second - o.second) * (q.first - o.first);
                    };
                    size_t h = 0;
                    if (pts.size() < 3) return std::vector<P2>();
                    for (size_t i = 0; i < pts.size(); ++i) { while (h >= 2 && cross(hull[h-2], hull[h-1], pts[i]) <= 0) --h; hull[h++] = pts[i]; }
                    for (size_t i = pts.size() - 1, lower = h + 1; i-- > 0;) { while (h >= lower && cross(hull[h-2], hull[h-1], pts[i]) <= 0) --h; hull[h++] = pts[i]; }
                    hull.resize(h);
                    return h >= 4 ? hull : std::vector<P2>();
                };
                struct Stance { float angle = -90.0f; double mu = 0, mv = 0, ml = 0; };
                auto stance = [&](int k) {
                    Stance st;
                    double m = 0;
                    for (int l = k; l < n; ++l) { m += count[(size_t)l]; st.mu += sum_u[(size_t)l]; st.mv += sum_v[(size_t)l]; st.ml += count[(size_t)l] * l; }
                    if (m <= 0) return st;
                    st.mu /= m; st.mv /= m; st.ml /= m;
                    const std::vector<P2> hull = convex_hull(layer[(size_t)(k - base_layer)]);
                    if (hull.empty()) return st;
                    double margin = 1e30;   // inside a counter-clockwise hull: distance to the nearest edge
                    for (size_t i = 0; i + 1 < hull.size(); ++i) {
                        const double ex = hull[i+1].first - hull[i].first, ey = hull[i+1].second - hull[i].second;
                        const double len = std::sqrt(ex * ex + ey * ey);
                        if (len > 0) margin = std::min(margin, (ex * (st.mv - hull[i].second) - ey * (st.mu - hull[i].first)) / len);
                    }
                    st.angle = (float)(std::atan2(margin, std::max(0.5, st.ml - ((double)k - 0.5))) * 57.29577951308232);
                    return st;
                };
                const Stance lowest = stance(base_layer);
                int chosen = -1;
                float chosen_angle = lowest.angle;
                for (int k = base_layer; k <= last && chosen < 0; ++k) {
                    const float angle = k == base_layer ? lowest.angle : stance(k).angle;
                    if (angle >= options.base_tip_deg) { chosen = k; chosen_angle = angle; }
                }
                if (chosen >= 0) {
                    printf("  remesh_solid: stable base at layer %d (lowest %d): tips over at %.1f deg\n", chosen, base_layer, chosen_angle);
                    base_layer = chosen;
                } else {
                    // Pedestal under the lowest cut: about 1.5 % of the grid thick.
                    pedestal_layers = std::min(base_layer - 1, std::max(3, (int)std::lround(0.015f * res)));
                    const double height = lowest.ml - ((double)(base_layer - pedestal_layers) - 0.5);
                    const double radius = height * std::tan(options.base_tip_deg * 0.017453292519943295) + 1.0;
                    std::vector<P2> pts = layer[0];
                    for (int i = 0; i < 64; ++i) {
                        const double t = 6.283185307179586 * i / 64.0;
                        pts.emplace_back(std::min((int)n - 2, std::max(1, (int)std::lround(lowest.mu + radius * std::cos(t)))),
                                         std::min((int)n - 2, std::max(1, (int)std::lround(lowest.mv + radius * std::sin(t)))));
                    }
                    pedestal = convex_hull(pts);
                    if (pedestal.empty()) pedestal_layers = 0;
                    printf("  remesh_solid: base at layer %d tips over at %.1f deg (target %.0f) and no cut up to layer %d fixes it;"
                           " %s\n", base_layer, lowest.angle, options.base_tip_deg, last,
                           pedestal_layers ? "adding a pedestal" : "no pedestal possible");
                }
            }
            const int u = (base_axis + 1) % 3, v = (base_axis + 2) % 3;
            int p[3];
            for (p[base_axis] = 0; p[base_axis] < base_layer; ++p[base_axis])
                for (p[u] = 0; p[u] < n; ++p[u])
                    for (p[v] = 0; p[v] < n; ++p[v]) {
                        const int64_t b = vbit(p[0], p[1], p[2]);
                        wall[(size_t)(b >> 6)] &= ~(1ull << (b & 63));
                    }
            if (pedestal_layers > 0) {
                int lo[2] = {(int)n, (int)n}, hi[2] = {0, 0};
                for (const auto& q : pedestal) {
                    lo[0] = std::min(lo[0], q.first); hi[0] = std::max(hi[0], q.first);
                    lo[1] = std::min(lo[1], q.second); hi[1] = std::max(hi[1], q.second);
                }
                for (p[u] = lo[0]; p[u] <= hi[0]; ++p[u])
                    for (p[v] = lo[1]; p[v] <= hi[1]; ++p[v]) {
                        bool inside = true;
                        for (size_t i = 0; i + 1 < pedestal.size() && inside; ++i)
                            inside = (int64_t)(pedestal[i+1].first - pedestal[i].first) * (p[v] - pedestal[i].second) -
                                     (int64_t)(pedestal[i+1].second - pedestal[i].second) * (p[u] - pedestal[i].first) >= 0;
                        if (!inside) continue;
                        for (p[base_axis] = base_layer - pedestal_layers; p[base_axis] < base_layer; ++p[base_axis]) {
                            const int64_t b = vbit(p[0], p[1], p[2]);
                            wall[(size_t)(b >> 6)] |= 1ull << (b & 63);
                        }
                    }
                base_layer -= pedestal_layers;   // the print now stands on the pedestal's underside
                fill_unreachable();              // e.g. the hollow under an arched sole
            }
        }

        // Re-sign the field. A filled vertex (f >= 0 but not exterior) goes negative, a cut
        // vertex goes positive; magnitudes are kept away from 0 so crossings stay interior.
        const float half = 0.5f * cell;
        synthetic.assign((size_t)Nv, 0);
        int64_t filled = 0, cut_away = 0;
        for (int64_t i = 0; i < Nv; ++i) {
            const bool in = is_solid(vcoord[3*i], vcoord[3*i+1], vcoord[3*i+2]);
            if (in && fvert[i] >= 0)      { fvert[i] = -std::max(fvert[i], half); ++filled; synthetic[(size_t)i] = 1; }
            else if (!in && fvert[i] < 0) { fvert[i] = std::max(-fvert[i], half); ++cut_away; }
        }

        // The solid's boundary now also runs where the narrow band never was: across the
        // caps of sealed tears and along the base cut. Activate every cell with both solid
        // and exterior corners; vertices the band did not sample sit half a cell from the
        // boundary, so those faces land mid-edge.
        std::vector<std::vector<int64_t>> mixed_parts((size_t)std::max(1u, std::thread::hardware_concurrency()) + 1);
        {
            std::atomic<int> next_part{0};
            parallel_for(res, [&](int64_t xb, int64_t xe) {
                std::vector<int64_t>& part = mixed_parts[(size_t)next_part.fetch_add(1)];
                for (int64_t x = xb; x < xe; ++x)
                    for (int64_t y = 0; y < res; ++y) {
                        const int64_t rows[4] = {(x * n + y) * n, (x * n + y + 1) * n, ((x + 1) * n + y) * n, ((x + 1) * n + y + 1) * n};
                        auto bit = [&](int r, int64_t z) -> int { const int64_t b = rows[r] + z; return (int)((wall[(size_t)(b >> 6)] >> (b & 63)) & 1); };
                        int prev = bit(0, 0) + bit(1, 0) + bit(2, 0) + bit(3, 0);
                        for (int64_t z = 0; z < res; ++z) {
                            const int cur = bit(0, z + 1) + bit(1, z + 1) + bit(2, z + 1) + bit(3, z + 1);
                            const int sum = prev + cur;
                            if (sum != 0 && sum != 8) part.push_back((x * res + y) * res + z);
                            prev = cur;
                        }
                    }
            });
        }
        int64_t extra_cells = 0;
        for (const auto& part : mixed_parts)
            for (const int64_t c : part) {
                const int cx = (int)(c / ((int64_t)res * res)), cy = (int)((c / res) % res), cz = (int)(c % res);
                if (!vox.emplace(key3(cx, cy, cz), (int)(acoord.size() / 3)).second) continue;
                acoord.push_back(cx); acoord.push_back(cy); acoord.push_back(cz);
                ++extra_cells;
                for (int k = 0; k < 8; ++k) {
                    const int x = cx + (k & 1), y = cy + ((k >> 1) & 1), z = cz + ((k >> 2) & 1);
                    if (!vmap.emplace(key3(x, y, z), (int)(vcoord.size() / 3)).second) continue;
                    vcoord.push_back(x); vcoord.push_back(y); vcoord.push_back(z);
                    fvert.push_back(is_solid(x, y, z) ? -half : half);
                    synthetic.push_back(1);
                }
            }
        std::vector<uint64_t>().swap(wall);
        printf("  remesh_solid: %lld vertices filled, %lld cells outside the band (caps%s)",
               (long long)filled, (long long)extra_cells, base_axis >= 0 ? ", base" : "");
        if (options.seal > 0) printf("; openness < %d of 26 shut (%lld coarse vertices)", options.seal, (long long)shut_coarse);
        if (options.seal > 0 && options.open_ball_cells > 0)
            printf("; %lld released by a %.1f-cell open ball", (long long)released_coarse, options.open_ball_cells);
        if (thickened) printf("; thin walls grown by %lld vertices in %d passes", (long long)thickened, thicken_passes);
        if (base_axis >= 0)
            printf("; flat base on axis %d at %.4f (%lld vertices cut)", base_axis,
                   (((float)base_layer - 0.5f) / res - 0.5f) * scale, (long long)cut_away);
        printf("\n");
        fflush(stdout);
    }

    // ---- solid mode: one polygon per surface cycle (marching cubes by construction) ----
    // On each cube face the crossing edges are linked in pairs (an ambiguous face joins the
    // inside corners, the same way from both cells that share it), so the crossings of a
    // cell fall into closed cycles; each cycle becomes one polygon over vertices that live
    // on grid edges and are shared by the four cells around them. Every polygon side lies
    // in a cube face and is used by exactly the two cells on either side of it: a closed
    // 2-manifold with a third of the triangles marching tetrahedra need. The result is
    // audited and the tetrahedra below remain the fallback.
    if (solid && !std::getenv("TRELLIS_SOLID_TETS")) {
        Mesh mc;
        const float min_fraction = 0.05f;
        std::unordered_map<uint64_t, int32_t> cuts;
        cuts.reserve(acoord.size());
        std::vector<uint8_t> cap;   // per output vertex: placed by guesswork, free to smooth
        auto cut = [&](int a, int b) -> int32_t {
            const uint32_t lo = (uint32_t)std::min(a, b), hi = (uint32_t)std::max(a, b);
            const uint64_t key = (uint64_t(lo) << 32) | hi;
            auto it = cuts.find(key);
            if (it != cuts.end()) return it->second;
            const bool through_base = base_axis >= 0 &&
                (vcoord[3*a+base_axis] < base_layer) != (vcoord[3*b+base_axis] < base_layer);
            const float fa = fvert[a], fb = fvert[b];
            const float t = through_base ? 0.5f : std::clamp(fa / (fa - fb), min_fraction, 1.0f - min_fraction);
            const int32_t index = (int32_t)(mc.verts.size() / 3);
            for (int k = 0; k < 3; ++k) {
                const float coord = vcoord[3*a+k] + t * (vcoord[3*b+k] - vcoord[3*a+k]);
                mc.verts.push_back((coord / res - 0.5f) * scale);
            }
            cuts.emplace(key, index);
            cap.push_back(!through_base && (synthetic[(size_t)a] || synthetic[(size_t)b]));
            return index;
        };
        // Cube tables: corner bit0/1/2 = +x/+y/+z; 12 edges; 6 faces as corner cycles.
        int edge_a[12], edge_b[12], edge_of[8][8], face[6][4];
        {
            int ne = 0;
            for (int axis = 0; axis < 3; ++axis)
                for (int c = 0; c < 8; ++c) {
                    if ((c >> axis) & 1) continue;
                    edge_a[ne] = c; edge_b[ne] = c | (1 << axis);
                    edge_of[c][c | (1 << axis)] = edge_of[c | (1 << axis)][c] = ne++;
                }
            for (int axis = 0; axis < 3; ++axis)
                for (int side = 0; side < 2; ++side) {
                    const int u = 1 << ((axis + 1) % 3), v = 1 << ((axis + 2) % 3), base = side << axis;
                    const int cyc[4] = {base, base | u, base | u | v, base | v};
                    for (int k = 0; k < 4; ++k) face[2 * axis + side][k] = cyc[k];
                }
        }
        const int64_t Nc = (int64_t)acoord.size() / 3;
        for (int64_t i = 0; i < Nc; ++i) {
            const int x = acoord[3*i], y = acoord[3*i+1], z = acoord[3*i+2];
            int corner[8];
            int mask = 0;
            for (int m = 0; m < 8; ++m) {
                corner[m] = vmap.at(key3(x + (m & 1), y + ((m >> 1) & 1), z + ((m >> 2) & 1)));
                if (fvert[corner[m]] < 0) mask |= 1 << m;
            }
            if (mask == 0 || mask == 255) continue;
            int link[12][2], nlink[12] = {0};
            auto join = [&](int a, int b) { link[a][nlink[a]++] = b; link[b][nlink[b]++] = a; };
            for (const auto& cyc : face) {
                int e[4], crossing[4], count = 0;
                for (int k = 0; k < 4; ++k) {
                    e[k] = edge_of[cyc[k]][cyc[(k + 1) % 4]];
                    crossing[k] = ((mask >> cyc[k]) & 1) != ((mask >> cyc[(k + 1) % 4]) & 1);
                    count += crossing[k];
                }
                if (count == 2) {
                    int first = -1;
                    for (int k = 0; k < 4; ++k) {
                        if (!crossing[k]) continue;
                        if (first < 0) first = e[k]; else join(first, e[k]);
                    }
                } else if (count == 4) {
                    for (int k = 0; k < 4; ++k)   // around each OUTSIDE corner: inside stays joined
                        if (!((mask >> cyc[k]) & 1)) join(e[(k + 3) % 4], e[k]);
                }
            }
            bool visited[12] = {false};
            for (int start = 0; start < 12; ++start) {
                if (nlink[start] != 2 || visited[start]) continue;
                int poly[12], n = 0, previous = -1, current = start;
                do {
                    visited[current] = true;
                    poly[n++] = current;
                    const int next = link[current][0] != previous ? link[current][0] : link[current][1];
                    previous = current;
                    current = next;
                } while (current != start && n < 12);
                int32_t pv[12];
                float direction[3] = {0, 0, 0};   // inside -> outside, summed over the cycle
                for (int k = 0; k < n; ++k) {
                    const int a = edge_a[poly[k]], b = edge_b[poly[k]];
                    pv[k] = cut(corner[a], corner[b]);
                    const int in = (mask >> a) & 1 ? a : b, outc = in == a ? b : a;
                    for (int d = 0; d < 3; ++d) direction[d] += (float)(((outc >> d) & 1) - ((in >> d) & 1));
                }
                float normal[3] = {0, 0, 0};      // Newell
                for (int k = 0; k < n; ++k) {
                    const float* p = &mc.verts[3 * (size_t)pv[k]];
                    const float* q = &mc.verts[3 * (size_t)pv[(k + 1) % n]];
                    normal[0] += (p[1] - q[1]) * (p[2] + q[2]);
                    normal[1] += (p[2] - q[2]) * (p[0] + q[0]);
                    normal[2] += (p[0] - q[0]) * (p[1] + q[1]);
                }
                if (normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2] < 0)
                    std::reverse(pv, pv + n);
                if (n <= 4) {
                    for (int k = 1; k + 1 < n; ++k) { mc.faces.push_back(pv[0]); mc.faces.push_back(pv[k]); mc.faces.push_back(pv[k + 1]); }
                } else {   // a centre vertex keeps every diagonal private to this polygon
                    float centre[3] = {0, 0, 0};
                    for (int k = 0; k < n; ++k)
                        for (int d = 0; d < 3; ++d) centre[d] += mc.verts[3 * (size_t)pv[k] + d] / n;
                    const int32_t c = (int32_t)(mc.verts.size() / 3);
                    mc.verts.insert(mc.verts.end(), centre, centre + 3);
                    uint8_t any_cap = 0;
                    for (int k = 0; k < n; ++k) any_cap |= cap[(size_t)pv[k]];
                    cap.push_back(any_cap);
                    for (int k = 0; k < n; ++k) { mc.faces.push_back(c); mc.faces.push_back(pv[k]); mc.faces.push_back(pv[(k + 1) % n]); }
                }
            }
        }
        // Cap smoothing. Where the solid's boundary was decided by the openness vote there
        // is no distance to interpolate, so cap vertices sit mid-edge on the vote's coarse
        // grid: stair-steps `vote_step` cells high on the visible surface. Relax them as a
        // membrane -- umbrella averaging with every measured vertex (and the base plane)
        // pinned -- for enough passes to diffuse a step, not enough to pull a large cap
        // flat. Connectivity is untouched, so the surface stays the same closed manifold.
        int64_t smoothed = 0;
        if (options.smooth_caps) {
            const int32_t MV = mc.V();
            std::vector<int32_t> slot((size_t)MV, -1), free_list;
            for (int32_t v = 0; v < MV; ++v)
                if (cap[(size_t)v]) { slot[(size_t)v] = (int32_t)free_list.size(); free_list.push_back(v); }
            smoothed = (int64_t)free_list.size();
            if (smoothed) {
                // Neighbours of the free vertices (each edge is seen from its two faces, so
                // every neighbour appears twice: a uniform weight).
                std::vector<int32_t> offset((size_t)smoothed + 1, 0);
                for (size_t f = 0; f + 2 < mc.faces.size(); f += 3)
                    for (int k = 0; k < 3; ++k) {
                        const int32_t s = slot[(size_t)mc.faces[f + k]];
                        if (s >= 0) offset[(size_t)s + 1] += 2;
                    }
                for (int64_t i = 0; i < smoothed; ++i) offset[(size_t)i + 1] += offset[(size_t)i];
                std::vector<int32_t> ring((size_t)offset[(size_t)smoothed]);
                {
                    std::vector<int32_t> cursor(offset.begin(), offset.end() - 1);
                    for (size_t f = 0; f + 2 < mc.faces.size(); f += 3)
                        for (int k = 0; k < 3; ++k) {
                            const int32_t s = slot[(size_t)mc.faces[f + k]];
                            if (s < 0) continue;
                            ring[(size_t)cursor[(size_t)s]++] = mc.faces[f + (k + 1) % 3];
                            ring[(size_t)cursor[(size_t)s]++] = mc.faces[f + (k + 2) % 3];
                        }
                }
                const int vote_step = std::max(1, res / 256);
                const int passes = std::min(96, 6 * vote_step * vote_step);
                std::vector<float> next((size_t)smoothed * 3);
                for (int pass = 0; pass < passes; ++pass) {
                    parallel_for(smoothed, [&](int64_t b, int64_t e) {
                        for (int64_t i = b; i < e; ++i) {
                            const int32_t begin = offset[(size_t)i], end = offset[(size_t)i + 1];
                            const float* p = &mc.verts[3 * (size_t)free_list[(size_t)i]];
                            float mean[3] = {0, 0, 0};
                            for (int32_t j = begin; j < end; ++j)
                                for (int d = 0; d < 3; ++d) mean[d] += mc.verts[3 * (size_t)ring[(size_t)j] + d];
                            for (int d = 0; d < 3; ++d)
                                next[3 * (size_t)i + d] = end > begin ? 0.5f * p[d] + 0.5f * mean[d] / (float)(end - begin) : p[d];
                        }
                    });
                    for (int64_t i = 0; i < smoothed; ++i)
                        for (int d = 0; d < 3; ++d) mc.verts[3 * (size_t)free_list[(size_t)i] + d] = next[3 * (size_t)i + d];
                }
                printf("  remesh_solid: smoothed %lld cap vertices (%d passes)\n", (long long)smoothed, passes);
            }
        }

        const MeshAudit check = audit_mesh(mc.verts.data(), mc.V(), mc.faces.data(), mc.F());
        if (check.boundary_edges == 0 && check.nonmanifold_edges == 0 && check.flipped_edges == 0 &&
            check.nonmanifold_vertices == 0 && check.inverted_components == 0 && mc.F() > 0) {
            printf("  remesh_solid: %lld active voxels -> V=%d F=%d (eps=%.4g), closed manifold\n",
                   (long long)Nc, mc.V(), mc.F(), eps);
            fflush(stdout);
            return mc;
        }
        printf("  remesh_solid: cycle extraction not manifold (%s); using marching tetrahedra\n",
               audit_summary(check).c_str());
        fflush(stdout);
    }

    if (solid || std::getenv("TRELLIS_REMESH_TETS")) {
        // Solid mode keeps crossings 5 % of an edge away from the grid vertices: no
        // zero-area slivers where the field is ~0 at a vertex.
        float min_fraction = solid ? 0.05f : 0.0f;
        if (const char* setting = std::getenv("TRELLIS_TETS_MIN_EDGE_FRACTION")) {
            char* end = nullptr;
            min_fraction = std::strtof(setting, &end);
            if (end == setting || *end || !std::isfinite(min_fraction)
                || min_fraction < 0.0f || min_fraction >= 0.5f)
                throw std::invalid_argument("TRELLIS_TETS_MIN_EDGE_FRACTION must be in [0, 0.5)");
        }
        static const int PERM[6][3] = {
            {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}
        };
        std::unordered_map<uint64_t, int32_t> intersections;
        intersections.reserve((size_t)Na * 2);
        size_t clamped_intersections = 0;
        auto cut = [&](int a, int b) -> int32_t {
            const uint32_t lo = (uint32_t)std::min(a, b), hi = (uint32_t)std::max(a, b);
            const uint64_t key = (uint64_t(lo) << 32) | hi;
            auto it = intersections.find(key);
            if (it != intersections.end()) return it->second;
            const float fa = fvert[a], fb = fvert[b];
            // An edge through the base plane crosses it exactly mid-layer: planar cap.
            const bool through_base = base_axis >= 0 &&
                (vcoord[3*a+base_axis] < base_layer) != (vcoord[3*b+base_axis] < base_layer);
            const float raw_t = through_base ? 0.5f : fa / (fa - fb);
            const float t = through_base ? 0.5f : std::clamp(raw_t, min_fraction, 1.0f - min_fraction);
            clamped_intersections += t != raw_t;
            const int32_t index = (int32_t)(out.verts.size() / 3);
            for (int k = 0; k < 3; ++k) {
                const float coord = vcoord[3*a+k] + t * (vcoord[3*b+k] - vcoord[3*a+k]);
                out.verts.push_back((coord / res - 0.5f) * scale);
            }
            intersections.emplace(key, index);
            return index;
        };
        auto emit = [&](int32_t a, int32_t b, int32_t c, const float* direction) {
            const float* pa = out.verts.data() + 3*a;
            const float* pb = out.verts.data() + 3*b;
            const float* pc = out.verts.data() + 3*c;
            const float u[3] = {pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]};
            const float v[3] = {pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2]};
            const float dot = (u[1]*v[2]-u[2]*v[1])*direction[0]
                            + (u[2]*v[0]-u[0]*v[2])*direction[1]
                            + (u[0]*v[1]-u[1]*v[0])*direction[2];
            if (dot < 0) std::swap(b, c);
            out.faces.push_back(a); out.faces.push_back(b); out.faces.push_back(c);
        };
        const int64_t Nt = (int64_t)acoord.size() / 3;   // Na plus any base-cut cells
        for (int64_t i = 0; i < Nt; ++i) {
            const int x = acoord[3*i], y = acoord[3*i+1], z = acoord[3*i+2];
            int corner[8];
            for (int mask = 0; mask < 8; ++mask)
                corner[mask] = vmap.at(key3(x + (mask & 1), y + ((mask >> 1) & 1), z + ((mask >> 2) & 1)));
            for (const auto& perm : PERM) {
                const int tet[4] = {corner[0], corner[1 << perm[0]],
                                    corner[(1 << perm[0]) | (1 << perm[1])], corner[7]};
                int inside[3], outside[3], ni = 0, no = 0;
                for (int t = 0; t < 4; ++t)
                    (fvert[tet[t]] < 0 ? inside[ni++] : outside[no++]) = tet[t];
                if (!ni || !no) continue;
                float direction[3] = {};
                for (int j = 0; j < no; ++j)
                    for (int k = 0; k < 3; ++k) direction[k] += float(vcoord[3*outside[j]+k]) / no;
                for (int j = 0; j < ni; ++j)
                    for (int k = 0; k < 3; ++k) direction[k] -= float(vcoord[3*inside[j]+k]) / ni;
                if (ni == 1) {
                    emit(cut(inside[0], outside[0]), cut(inside[0], outside[1]),
                         cut(inside[0], outside[2]), direction);
                } else if (no == 1) {
                    emit(cut(outside[0], inside[0]), cut(outside[0], inside[1]),
                         cut(outside[0], inside[2]), direction);
                } else {
                    const int32_t a = cut(inside[0], outside[0]);
                    const int32_t b = cut(inside[0], outside[1]);
                    const int32_t c = cut(inside[1], outside[1]);
                    const int32_t d = cut(inside[1], outside[0]);
                    emit(a, b, c, direction);
                    emit(a, c, d, direction);
                }
            }
        }
        printf("  %s: %lld active voxels -> V=%d F=%d (eps=%.4g, edge_fraction=%.4g, clamped=%zu)\n",
               solid ? "remesh_solid" : "remesh_tets", (long long)Nt, out.V(), out.F(), eps,
               min_fraction, clamped_intersections);
        fflush(stdout);
        return out;
    }

    // Dual vertices: plain mean of edge crossings, cell-center fallback; per
    // voxel, ownership of the 3 "far" edges records crossing direction
    // (spec 27 §4.4).
    std::vector<float> dual((size_t)Na * 3);
    std::vector<int8_t> owned((size_t)Na * 3, 0);
    parallel_for(Na, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
            double sum[3] = {0, 0, 0};
            int cnt = 0;
            for (int axis = 0; axis < 3; ++axis)
                for (int u = 0; u < 2; ++u) for (int v = 0; v < 2; ++v) {
                    int a0[3] = {vx, vy, vz}, a1[3];
                    a0[(axis + 1) % 3] += u;
                    a0[(axis + 2) % 3] += v;
                    a1[0] = a0[0]; a1[1] = a0[1]; a1[2] = a0[2];
                    a1[axis] += 1;
                    const float v1 = fval(a0[0], a0[1], a0[2]);
                    const float v2 = fval(a1[0], a1[1], a1[2]);
                    const bool c12 = v1 < 0 && v2 >= 0, c21 = v1 >= 0 && v2 < 0;
                    if (c12 || c21) {
                        const float t = -v1 / (v2 - v1);
                        double pt[3] = {(double)a0[0], (double)a0[1], (double)a0[2]};
                        pt[axis] += t;
                        for (int k = 0; k < 3; ++k) sum[k] += pt[k];
                        ++cnt;
                    }
                    if (u == 1 && v == 1) owned[3*i + axis] = c12 ? 1 : (c21 ? -1 : 0);
                }
            if (cnt) for (int k = 0; k < 3; ++k) dual[3*i+k] = (float)(sum[k] / cnt);
            else { dual[3*i] = vx + 0.5f; dual[3*i+1] = vy + 0.5f; dual[3*i+2] = vz + 0.5f; }
        }
    });

    // Quad assembly per owned crossing edge (spec 27 §4.5); winding from the
    // crossing direction. The reference's "planar diagonal" selection is a
    // latent no-op upstream (always diagonal q0-q2 unless triangle q0q1q2 is
    // exactly degenerate) — reproduced faithfully here as split 1 always.
    static const int OFF[3][4][3] = {
        {{0,0,0}, {0,0,1}, {0,1,1}, {0,1,0}},
        {{0,0,0}, {1,0,0}, {1,0,1}, {0,0,1}},
        {{0,0,0}, {0,1,0}, {1,1,0}, {1,0,0}},
    };
    std::vector<int32_t> qfaces;
    qfaces.reserve((size_t)Na * 6);
    std::vector<uint8_t> used((size_t)Na, 0);
    for (int64_t i = 0; i < Na; ++i) {
        const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
        for (int axis = 0; axis < 3; ++axis) {
            const int dir = owned[3*i + axis];
            if (!dir) continue;
            int q[4];
            bool ok = true;
            for (int k = 0; k < 4 && ok; ++k) {
                auto it = vox.find(key3(vx + OFF[axis][k][0], vy + OFF[axis][k][1], vz + OFF[axis][k][2]));
                if (it == vox.end()) ok = false;
                else q[k] = it->second;
            }
            if (!ok) continue;
            static const int S1N[6] = {0, 1, 2, 0, 2, 3};
            static const int S1P[6] = {0, 2, 1, 0, 3, 2};
            const int* sp = dir > 0 ? S1P : S1N;
            for (int k = 0; k < 6; ++k) qfaces.push_back(q[sp[k]]);
            for (int k = 0; k < 4; ++k) used[q[k]] = 1;
        }
    }
    if (qfaces.empty()) return out;

    // Compact used dual vertices; map back to world coordinates.
    std::vector<int32_t> remap((size_t)Na, -1);
    int nv2 = 0;
    for (int64_t i = 0; i < Na; ++i)
        if (used[i]) {
            remap[i] = nv2++;
            out.verts.push_back((dual[3*i]   / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+1] / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+2] / res - 0.5f) * scale);
        }
    out.faces.resize(qfaces.size());
    for (size_t k = 0; k < qfaces.size(); ++k) out.faces[k] = remap[qfaces[k]];

    if (std::getenv("TRELLIS_DIAG_DC_AMBIGUITY")) {
        int64_t ambiguous = 0;
        for (int64_t i = 0; i < Na; ++i) {
            const int base[3] = {acoord[3*i], acoord[3*i+1], acoord[3*i+2]};
            for (int axis = 0; axis < 3; ++axis) {
                int corner[3] = {base[0], base[1], base[2]};
                ++corner[axis];
                if (vox.find(key3(corner[0], corner[1], corner[2])) == vox.end()) continue;
                const int u = (axis + 1) % 3, v = (axis + 2) % 3;
                auto sign = [&](int du, int dv) {
                    int p[3] = {corner[0], corner[1], corner[2]};
                    p[u] += du; p[v] += dv;
                    return fval(p[0], p[1], p[2]) < 0;
                };
                const bool s00 = sign(0, 0), s10 = sign(1, 0);
                const bool s01 = sign(0, 1), s11 = sign(1, 1);
                ambiguous += s00 == s11 && s10 == s01 && s00 != s10;
            }
        }
        printf("  remesh_dc: ambiguous shared voxel faces=%lld\n", (long long)ambiguous);
    }

    // Project the dual vertices back onto the input surface (reference:
    // remesh_project=0.9, o_voxel/postprocess.py::to_glb -> remeshing.py §8).
    // Dual contouring places each vertex at the plain MEAN of its cell's edge
    // crossings, on the eps-offset shell — so sharp features come out rounded and
    // the shell keeps its own offset noise. Lerping each vertex back toward its
    // closest point on the original surface restores those features. The BVH's
    // closest point is exactly the reference's barycentric interpolation of the
    // hit triangle, so no uvw round-trip is needed.
    if (project_back > 0.0f) {
        const int64_t NV = (int64_t)out.verts.size() / 3;
        std::atomic<int64_t> missed{0};
        parallel_for(NV, [&](int64_t b, int64_t e) {
            int64_t local = 0;
            for (int64_t i = b; i < e; ++i) {
                float* v = &out.verts[3 * i];
                const float p[3] = {v[0], v[1], v[2]};
                // Vertices sit on the eps shell, so the surface is ~eps away; the
                // same bound the field pass uses is comfortably sufficient.
                const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
                if (h.face < 0) { ++local; continue; }   // no hit in range: leave as-is
                for (int k = 0; k < 3; ++k) v[k] = p[k] - project_back * (p[k] - h.point[k]);
            }
            if (local) missed.fetch_add(local, std::memory_order_relaxed);
        });
        if (missed.load())
            printf("  remesh_dc: project_back %.2f (%lld/%lld vertices had no hit in range)\n",
                   project_back, (long long)missed.load(), (long long)NV);
    }

    printf("  remesh_dc: %lld active voxels -> V=%d F=%d (eps=%.4g, project_back=%.2f)\n",
           (long long)Na, out.V(), out.F(), eps, project_back);
    fflush(stdout);
    return out;
}

}  // namespace trellis
