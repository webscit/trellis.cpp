// Filament colours for multi-material printers: the base-colour texture reduced to a small
// palette, one colour per triangle (or per sub-triangle of a large one, written as Orca split
// states), with specks merged away. See mesh_print.h, docs/spec/32-print-ready.md §2.7 and
// docs/spec/33-texture-only.md.
#include "mesh_print.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>

namespace trellis {

namespace {

void for_chunks(int64_t n, const std::function<void(int64_t, int64_t, int)>& fn) {
    const int nt = (int)std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    const int64_t chunk = (n + nt - 1) / nt;
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t) {
        const int64_t b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        ts.emplace_back(fn, b, e, t);
    }
    for (auto& t : ts) t.join();
}

struct Vec3 { double x = 0, y = 0, z = 0; };

double srgb_to_linear(double c) { return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4); }
double linear_to_srgb(double c) {
    c = std::min(1.0, std::max(0.0, c));
    return c <= 0.0031308 ? c * 12.92 : 1.055 * std::pow(c, 1 / 2.4) - 0.055;
}

// CIELAB (D65) from linear sRGB and back.
constexpr double kWhite[3] = {0.95047, 1.0, 1.08883};
Vec3 linear_to_lab(const Vec3& c) {
    const double X = (0.4124 * c.x + 0.3576 * c.y + 0.1805 * c.z) / kWhite[0];
    const double Y = (0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z) / kWhite[1];
    const double Z = (0.0193 * c.x + 0.1192 * c.y + 0.9505 * c.z) / kWhite[2];
    auto f = [](double t) { return t > 216.0 / 24389 ? std::cbrt(t) : (24389.0 / 27 * t + 16) / 116; };
    const double fx = f(X), fy = f(Y), fz = f(Z);
    return {116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)};
}
Vec3 lab_to_linear(const Vec3& lab) {
    const double fy = (lab.x + 16) / 116, fx = fy + lab.y / 500, fz = fy - lab.z / 200;
    auto g = [](double f) { const double t = f * f * f; return t > 216.0 / 24389 ? t : (116 * f - 16) / (24389.0 / 27); };
    const double X = g(fx) * kWhite[0], Y = g(fy) * kWhite[1], Z = g(fz) * kWhite[2];
    auto clamp01 = [](double v) { return std::min(1.0, std::max(0.0, v)); };
    return {clamp01(3.2406 * X - 1.5372 * Y - 0.4986 * Z), clamp01(-0.9689 * X + 1.8758 * Y + 0.0415 * Z),
            clamp01(0.0557 * X - 0.2040 * Y + 1.0570 * Z)};
}
double dist2(const Vec3& a, const Vec3& b) {
    const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

// Area-weighted value below which `q` of the total weight lies.
double weighted_quantile(const std::vector<double>& value, const std::vector<double>& weight, double q) {
    std::vector<int64_t> order(value.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return value[a] < value[b]; });
    const double total = std::accumulate(weight.begin(), weight.end(), 0.0);
    double sum = 0;
    for (int64_t i : order) {
        sum += weight[i];
        if (sum >= q * total) return value[i];
    }
    return order.empty() ? 0 : value[order.back()];
}

struct Dsu {
    std::vector<int32_t> parent;
    explicit Dsu(int64_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    int32_t find(int32_t a) {
        while (parent[a] != a) a = parent[a] = parent[parent[a]];
        return a;
    }
    void unite(int32_t a, int32_t b) { a = find(a); b = find(b); if (a != b) parent[std::max(a, b)] = std::min(a, b); }
};

}  // namespace

std::string paint_color_code(int slot) {
    char buffer[16];
    if (slot < 3) {
        std::snprintf(buffer, sizeof buffer, "%X", slot << 2);
        return buffer;
    }
    // A leaf whose state does not fit the two state bits is written as 0b11 in them, then
    // (state - 3) in following nibbles, 0xF meaning "15 and more follows". The string is read
    // from its end, so the nibbles appear in reverse.
    int rest = slot - 3;
    std::string code = "C";
    while (rest >= 15) { code = "F" + code; rest -= 15; }
    std::snprintf(buffer, sizeof buffer, "%X", rest);
    return buffer + code;
}

namespace {

// Orca's TriangleSelector bitstream (libslic3r/TriangleSelector.cpp, serialize): per node a
// nibble of 2 bits "split sides" then 2 bits (leaf: state; split: special side); a leaf state
// of 3 and above is 0b11 there plus a nibble of (state - 3), or 0b1111 and a nibble of
// (state - 18) from 18 on; a three-side split is followed by its children 3, 2, 1, 0. The 3MF
// string packs the bits in nibbles (bit j of a nibble = 2^j) written last nibble first.
void push_bits(std::vector<bool>& bits, int value, int n) {
    for (int j = 0; j < n; ++j) bits.push_back((value >> j) & 1);
}
void push_leaf(std::vector<bool>& bits, int state) {
    push_bits(bits, 0, 2);
    if (state < 3) { push_bits(bits, state, 2); return; }
    push_bits(bits, 3, 2);
    if (state <= 17) push_bits(bits, state - 3, 4);
    else { push_bits(bits, 15, 4); push_bits(bits, state - 18, 4); }
}
void push_tree(std::vector<bool>& bits, const uint8_t* slots, int64_t n) {
    bool uniform = true;
    for (int64_t i = 1; i < n && uniform; ++i) uniform = slots[i] == slots[0];
    if (uniform) { push_leaf(bits, slots[0] + 1); return; }
    push_bits(bits, 3, 2);   // three split sides
    push_bits(bits, 0, 2);   // special side (ignored for three)
    const int64_t q = n / 4;
    for (int child = 3; child >= 0; --child) push_tree(bits, slots + child * q, q);
}
std::string bits_to_code(const std::vector<bool>& bits) {
    static const char kHex[] = "0123456789ABCDEF";
    std::string code;
    for (size_t i = 0; i < bits.size(); i += 4) {
        int v = 0;
        for (int j = 0; j < 4 && i + j < bits.size(); ++j) v |= (int)bits[i + j] << j;
        code.push_back(kHex[v]);
    }
    std::reverse(code.begin(), code.end());
    return code;
}

struct FacePair { int32_t f, g; float length; };   // faces sharing a border of `length` mm
PrintColors quantize_faces(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                           const PrintTexture& texture, float size_mm, const ColorOptions& options,
                           const std::vector<FacePair>* extra_pairs = nullptr);

}  // namespace

std::string paint_tree_code(const uint8_t* leaf_slots, int depth) {
    std::vector<bool> bits;
    push_tree(bits, leaf_slots, (int64_t)1 << (2 * depth));
    return bits_to_code(bits);
}

PrintColors quantize_print_colors(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                                  const PrintTexture& texture, float size_mm, const ColorOptions& options) {
    if (options.cell_mm <= 0 || V <= 0 || F <= 0 || !texture.ok())
        return quantize_faces(verts, V, faces, F, texture, size_mm, options);
    // Millimetres per mesh unit, exactly as write_3mf scales.
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (int64_t v = 0; v < V; ++v)
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], verts[3 * v + k]);
            hi[k] = std::max(hi[k], verts[3 * v + k]);
        }
    const float extent = std::max(hi[0] - lo[0], std::max(hi[1] - lo[1], hi[2] - lo[2]));
    const double mm = size_mm > 0 && extent > 0 ? size_mm / extent : 1.0;
    std::vector<double> edge(F);
    for (int64_t f = 0; f < F; ++f) {
        double m = 0;
        for (int k = 0; k < 3; ++k) {
            const float* p = verts + 3 * faces[3 * f + k], *q = verts + 3 * faces[3 * f + (k + 1) % 3];
            m = std::max(m, std::sqrt((double)(p[0] - q[0]) * (p[0] - q[0]) + (double)(p[1] - q[1]) * (p[1] - q[1]) +
                                      (double)(p[2] - q[2]) * (p[2] - q[2])) * mm);
        }
        edge[f] = m;
    }
    // Depth per face: halve the edges until they are at most the cell. A leaf budget keeps the
    // temporary mesh bounded; past it the cell grows until the leaves fit.
    constexpr int64_t kMaxLeaves = 8'000'000;
    std::vector<uint8_t> depth(F, 0);
    double cell = options.cell_mm;
    for (;;) {
        int64_t leaves = 0;
        for (int64_t f = 0; f < F; ++f) {
            int d = 0;
            while (d < kPaintMaxDepth && edge[f] / (double)(1 << d) > cell) ++d;
            depth[f] = (uint8_t)d;
            leaves += (int64_t)1 << (2 * d);
        }
        if (leaves <= std::max<int64_t>(kMaxLeaves, F)) break;
        cell *= 1.25;
    }
    if (std::all_of(depth.begin(), depth.end(), [](uint8_t d) { return d == 0; }))
        return quantize_faces(verts, V, faces, F, texture, size_mm, options);

    // The leaf mesh: midpoints shared per (vertex, vertex) pair, so a UV seam keeps its two
    // sides' UVs while their positions stay bit-identical (adjacency welds them).
    std::vector<float> lv(verts, verts + 3 * V), luv(texture.uv, texture.uv + 2 * V);
    std::vector<int32_t> lf;
    std::unordered_map<uint64_t, int32_t> mids;
    auto mid = [&](int32_t a, int32_t b) {
        const uint64_t key = (uint64_t)(uint32_t)std::min(a, b) << 32 | (uint32_t)std::max(a, b);
        auto it = mids.find(key);
        if (it != mids.end()) return it->second;
        const int32_t id = (int32_t)(lv.size() / 3);
        for (int k = 0; k < 3; ++k) lv.push_back(0.5f * (lv[3 * (size_t)a + k] + lv[3 * (size_t)b + k]));
        for (int k = 0; k < 2; ++k) luv.push_back(0.5f * (luv[2 * (size_t)a + k] + luv[2 * (size_t)b + k]));
        mids.emplace(key, id);
        return id;
    };
    // Where two faces of different depth meet, their leaves do not share vertices (a
    // T-junction), so the speck merge would not see that border. Each leaf side lying on an
    // original side is recorded with its interval along that side; overlapping intervals of
    // faces with different depths become extra adjacency pairs.
    std::vector<float> wpos(verts, verts + 3 * V);
    std::vector<int32_t> wid(faces, faces + 3 * F);
    weld_exact(wpos, wid);   // welded vertex ids of the original corners
    struct Side { int k; double t0, t1; };   // original side k of the face (-1: interior), interval
    struct OnSide { uint64_t key; double lo, hi; int32_t leaf; int32_t face; };
    std::vector<OnSide> on_side;
    int64_t cur_face = 0;
    std::function<void(int32_t, int32_t, int32_t, Side, Side, Side, int)> emit =
        [&](int32_t a, int32_t b, int32_t c, Side sab, Side sbc, Side sca, int d) {
            if (d == 0) {
                const int32_t leaf = (int32_t)(lf.size() / 3);
                lf.push_back(a); lf.push_back(b); lf.push_back(c);
                for (const Side& sd : {sab, sbc, sca}) {
                    if (sd.k < 0) continue;
                    const int32_t p = wid[3 * cur_face + sd.k], q = wid[3 * cur_face + (sd.k + 1) % 3];
                    if (p == q) continue;
                    // the interval along p -> q, re-expressed along min -> max of the welded ids
                    double lo = std::min(sd.t0, sd.t1), hi = std::max(sd.t0, sd.t1);
                    if (p > q) { const double l2 = 1 - hi; hi = 1 - lo; lo = l2; }
                    on_side.push_back({(uint64_t)(uint32_t)std::min(p, q) << 32 | (uint32_t)std::max(p, q), lo, hi, leaf,
                                       (int32_t)cur_face});
                }
                return;
            }
            const int32_t ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
            auto half = [](const Side& sd, bool second) {
                if (sd.k < 0) return sd;
                const double m = 0.5 * (sd.t0 + sd.t1);
                return second ? Side{sd.k, m, sd.t1} : Side{sd.k, sd.t0, m};
            };
            const Side in{-1, 0, 0};
            emit(a, ab, ca, half(sab, false), in, half(sca, true), d - 1);
            emit(ab, b, bc, half(sab, true), half(sbc, false), in, d - 1);
            emit(bc, c, ca, half(sbc, true), half(sca, false), in, d - 1);
            emit(ab, bc, ca, in, in, in, d - 1);
        };
    std::vector<int64_t> first(F + 1, 0);
    for (int64_t f = 0; f < F; ++f) {
        first[f] = (int64_t)lf.size() / 3;
        cur_face = f;
        emit(faces[3 * f], faces[3 * f + 1], faces[3 * f + 2], Side{0, 0, 1}, Side{1, 0, 1}, Side{2, 0, 1}, depth[f]);
    }
    first[F] = (int64_t)lf.size() / 3;
    mids.clear();
    std::vector<FacePair> tj;
    std::sort(on_side.begin(), on_side.end(), [](const OnSide& x, const OnSide& y) {
        return x.key != y.key ? x.key < y.key : x.lo < y.lo;
    });
    for (size_t i = 0; i < on_side.size();) {
        size_t j = i;
        while (j < on_side.size() && on_side[j].key == on_side[i].key) ++j;
        const int32_t p = (int32_t)(on_side[i].key >> 32), q = (int32_t)(on_side[i].key & 0xffffffffu);
        const float* P = &wpos[3 * (size_t)p], *Q = &wpos[3 * (size_t)q];
        const double len = std::sqrt((double)(P[0] - Q[0]) * (P[0] - Q[0]) + (double)(P[1] - Q[1]) * (P[1] - Q[1]) +
                                     (double)(P[2] - Q[2]) * (P[2] - Q[2])) * mm;
        for (size_t x = i; x < j; ++x)
            for (size_t y = x + 1; y < j && on_side[y].lo < on_side[x].hi; ++y) {
                const OnSide& u = on_side[x], &w = on_side[y];
                // same face, or two faces of the same depth: they share vertices already
                if (u.face == w.face || depth[u.face] == depth[w.face]) continue;
                const double overlap = std::min(u.hi, w.hi) - std::max(u.lo, w.lo);
                if (overlap > 1e-12) tj.push_back({u.leaf, w.leaf, (float)(overlap * len)});
            }
        i = j;
    }
    on_side.clear(); on_side.shrink_to_fit();
    const PrintTexture ltex{luv.data(), texture.rgba, texture.size};
    PrintColors leaf = quantize_faces(lv.data(), (int64_t)lv.size() / 3, lf.data(), (int64_t)lf.size() / 3, ltex,
                                      size_mm, options, &tj);
    if (!leaf.ok()) return leaf;
    PrintColors out;
    out.palette = std::move(leaf.palette);
    out.share = std::move(leaf.share);
    out.patches = leaf.patches;
    out.face_slot.resize(F);
    out.face_code.resize(F);
    for (int64_t f = 0; f < F; ++f) {
        const uint8_t* s = leaf.face_slot.data() + first[f];
        const int64_t n = first[f + 1] - first[f];
        int count[kMaxPrintColors] = {0};
        for (int64_t i = 0; i < n; ++i) ++count[s[i]];
        out.face_slot[f] = (uint8_t)(std::max_element(count, count + kMaxPrintColors) - count);
        out.face_code[f] = paint_tree_code(s, depth[f]);
    }
    return out;
}

namespace {

PrintColors quantize_faces(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                           const PrintTexture& texture, float size_mm, const ColorOptions& options,
                           const std::vector<FacePair>* extra_pairs) {
    PrintColors out;
    const int K = std::min(options.count, kMaxPrintColors);
    if (K < 2 || V <= 0 || F <= 0 || !texture.ok()) return out;

    // Millimetres per mesh unit, exactly as write_3mf scales.
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (int64_t v = 0; v < V; ++v)
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], verts[3 * v + k]);
            hi[k] = std::max(hi[k], verts[3 * v + k]);
        }
    const float extent = std::max(hi[0] - lo[0], std::max(hi[1] - lo[1], hi[2] - lo[2]));
    const double mm = size_mm > 0 && extent > 0 ? size_mm / extent : 1.0;

    // One colour per triangle: the mean of ten texture samples, in linear light.
    static const double kBary[10][3] = {{1 / 3., 1 / 3., 1 / 3.}, {2 / 3., 1 / 6., 1 / 6.}, {1 / 6., 2 / 3., 1 / 6.},
                                        {1 / 6., 1 / 6., 2 / 3.}, {.5, .5, 0}, {0, .5, .5}, {.5, 0, .5},
                                        {.8, .1, .1}, {.1, .8, .1}, {.1, .1, .8}};
    double lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = srgb_to_linear(i / 255.0);
    std::vector<Vec3> lin(F), lab(F);
    std::vector<double> area(F);
    const int T = texture.size;
    for_chunks(F, [&](int64_t b, int64_t e, int) {
        for (int64_t f = b; f < e; ++f) {
            const int32_t* t = faces + 3 * f;
            Vec3 sum;
            for (const auto& w : kBary) {
                const double u = w[0] * texture.uv[2 * t[0]] + w[1] * texture.uv[2 * t[1]] + w[2] * texture.uv[2 * t[2]];
                const double v = w[0] * texture.uv[2 * t[0] + 1] + w[1] * texture.uv[2 * t[1] + 1] + w[2] * texture.uv[2 * t[2] + 1];
                const int x = std::min(T - 1, std::max(0, (int)(u * T)));
                const int y = std::min(T - 1, std::max(0, (int)(v * T)));   // glTF: v down from the top
                const unsigned char* p = texture.rgba + 4 * ((size_t)y * T + x);
                sum.x += lut[p[0]]; sum.y += lut[p[1]]; sum.z += lut[p[2]];
            }
            lin[f] = {sum.x / 10, sum.y / 10, sum.z / 10};
            lab[f] = linear_to_lab(lin[f]);
            const float* a = verts + 3 * t[0], *b2 = verts + 3 * t[1], *c = verts + 3 * t[2];
            const double e1[3] = {b2[0] - a[0], b2[1] - a[1], b2[2] - a[2]}, e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
            const double cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2], cz = e1[0] * e2[1] - e1[1] * e2[0];
            area[f] = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz) * mm * mm;
        }
    });
    const double total_area = std::accumulate(area.begin(), area.end(), 0.0);
    if (total_area <= 0) return out;

    // Shadow lift: generated textures carry baked-in shading, which would otherwise spend
    // most of the palette on near-blacks. Stretch L* from its 1st..99th area percentile to
    // [that 1st percentile, max(99th, 85)], lift the shadows with a gamma, and scale a*/b*
    // with L* (capped at 1.5x) so lifted colours keep their hue and saturation.
    if (options.lift > 0) {
        std::vector<double> L(F);
        for (int64_t f = 0; f < F; ++f) L[f] = lab[f].x;
        const double l0 = weighted_quantile(L, area, 0.01), l1 = weighted_quantile(L, area, 0.99);
        const double top = std::max(l1, 85.0), gamma = 1.0 / (1.0 + options.lift);
        for (int64_t f = 0; f < F; ++f) {
            const double t = std::min(1.0, std::max(0.0, (lab[f].x - l0) / std::max(l1 - l0, 1e-3)));
            const double L2 = std::min(98.0, std::max(2.0, l0 + (top - l0) * std::pow(t, gamma)));
            const double k = lab[f].x > 1 ? std::min(1.5, L2 / lab[f].x) : 1.0;
            lab[f] = {L2, lab[f].y * k, lab[f].z * k};
            lin[f] = lab_to_linear(lab[f]);
        }
    }

    // Palette: weighted k-means in Lab, k-means++ seeding with a fixed seed. Weights are the
    // square root of the area so small but distinct colours (eyes, buckles) win a slot
    // against large areas of near-identical shades.
    std::vector<double> wk(F);
    for (int64_t f = 0; f < F; ++f) wk[f] = std::sqrt(area[f] / total_area);
    std::mt19937_64 rng(0);
    auto pick = [&](const std::vector<double>& weight) {
        std::discrete_distribution<int64_t> d(weight.begin(), weight.end());
        return d(rng);
    };
    std::vector<Vec3> C;
    C.push_back(lab[pick(wk)]);
    std::vector<double> d2(F, 1e300), sw(F);
    while ((int)C.size() < K) {
        for (int64_t f = 0; f < F; ++f) { d2[f] = std::min(d2[f], dist2(lab[f], C.back())); sw[f] = wk[f] * d2[f]; }
        if (std::accumulate(sw.begin(), sw.end(), 0.0) <= 0) break;   // fewer distinct colours than K
        C.push_back(lab[pick(sw)]);
    }
    const int k_used = (int)C.size();
    std::vector<uint8_t> label(F, 0);
    const int nt = (int)std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    for (int it = 0; it < 60; ++it) {
        std::vector<std::vector<double>> acc(nt, std::vector<double>(4 * k_used, 0.0));
        for_chunks(F, [&](int64_t b, int64_t e, int t) {
            std::vector<double>& a = acc[t];
            for (int64_t f = b; f < e; ++f) {
                int best = 0;
                double bd = 1e300;
                for (int j = 0; j < k_used; ++j) {
                    const double d = dist2(lab[f], C[j]);
                    if (d < bd) { bd = d; best = j; }
                }
                label[f] = (uint8_t)best;
                a[4 * best] += wk[f] * lab[f].x; a[4 * best + 1] += wk[f] * lab[f].y;
                a[4 * best + 2] += wk[f] * lab[f].z; a[4 * best + 3] += wk[f];
            }
        });
        double moved = 0;
        for (int j = 0; j < k_used; ++j) {
            double s[4] = {0, 0, 0, 0};
            for (const auto& a : acc) for (int c = 0; c < 4; ++c) s[c] += a[4 * j + c];
            if (s[3] <= 0) continue;
            const Vec3 n{s[0] / s[3], s[1] / s[3], s[2] / s[3]};
            moved = std::max(moved, dist2(n, C[j]));
            C[j] = n;
        }
        if (moved < 1e-6) break;
    }

    // Adjacency over welded positions (the UV unwrap splits vertices along seams).
    std::vector<float> wv(verts, verts + 3 * V);
    std::vector<int32_t> wf(faces, faces + 3 * F);
    weld_exact(wv, wf);
    struct Edge { int32_t a, b; int32_t face; };
    std::vector<Edge> edges;
    edges.reserve(3 * F);
    for (int64_t f = 0; f < F; ++f)
        for (int k = 0; k < 3; ++k) {
            const int32_t a = wf[3 * f + k], b = wf[3 * f + (k + 1) % 3];
            if (a != b) edges.push_back({std::min(a, b), std::max(a, b), (int32_t)f});
        }
    std::sort(edges.begin(), edges.end(), [](const Edge& x, const Edge& y) {
        return x.a != y.a ? x.a < y.a : x.b != y.b ? x.b < y.b : x.face < y.face;
    });
    using Pair = FacePair;
    std::vector<Pair> pairs;   // faces sharing an edge (every pair on a non-manifold edge)
    for (size_t i = 0; i < edges.size();) {
        size_t j = i;
        while (j < edges.size() && edges[j].a == edges[i].a && edges[j].b == edges[i].b) ++j;
        const float* p = &wv[3 * (size_t)edges[i].a], *q = &wv[3 * (size_t)edges[i].b];
        const float len = (float)(std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) +
                                            (p[2] - q[2]) * (p[2] - q[2])) * mm);
        for (size_t x = i; x < j; ++x)
            for (size_t y = x + 1; y < j; ++y) pairs.push_back({edges[x].face, edges[y].face, len});
        i = j;
    }
    edges.clear(); edges.shrink_to_fit();
    if (extra_pairs) pairs.insert(pairs.end(), extra_pairs->begin(), extra_pairs->end());

    // Speck merge: a same-colour patch under speck_mm2 takes the colour it shares the longest
    // border with. Smallest first; a patch that moves freezes its neighbours for the round, so
    // two adjacent specks cannot keep swapping colours.
    auto components = [&](std::vector<int32_t>& comp) {
        Dsu dsu(F);
        for (const Pair& p : pairs)
            if (label[p.f] == label[p.g]) dsu.unite(p.f, p.g);
        std::vector<int32_t> id(F, -1);
        int32_t n = 0;
        for (int64_t f = 0; f < F; ++f) {
            const int32_t r = dsu.find((int32_t)f);
            if (id[r] < 0) id[r] = n++;
            comp[f] = id[r];
        }
        return n;
    };
    const std::vector<uint8_t> clustered = label;   // before the merge: what each triangle looks like
    std::vector<int32_t> comp(F);
    int32_t ncomp = components(comp);
    if (options.speck_mm2 > 0) {
        for (int round = 0; round < 32; ++round) {
            std::vector<double> carea(ncomp, 0.0);
            for (int64_t f = 0; f < F; ++f) carea[comp[f]] += area[f];
            std::vector<int32_t> small_id(ncomp, -1);
            std::vector<int32_t> small;
            for (int32_t c = 0; c < ncomp; ++c)
                if (carea[c] < options.speck_mm2) { small_id[c] = (int32_t)small.size(); small.push_back(c); }
            if (small.empty()) break;
            std::vector<double> border(small.size() * k_used, 0.0);
            std::vector<std::pair<int32_t, int32_t>> nbr;   // (small comp, neighbouring comp)
            for (const Pair& p : pairs) {
                const int32_t a = comp[p.f], b = comp[p.g];
                if (a == b) continue;
                if (small_id[a] >= 0) { border[(size_t)small_id[a] * k_used + label[p.g]] += p.length; nbr.push_back({a, b}); }
                if (small_id[b] >= 0) { border[(size_t)small_id[b] * k_used + label[p.f]] += p.length; nbr.push_back({b, a}); }
            }
            std::sort(nbr.begin(), nbr.end());
            nbr.erase(std::unique(nbr.begin(), nbr.end()), nbr.end());
            std::sort(small.begin(), small.end(), [&](int32_t a, int32_t b) { return carea[a] < carea[b]; });
            std::vector<char> frozen(ncomp, 0);
            std::vector<int> target(ncomp, -1);
            int moves = 0;
            for (int32_t c : small) {
                if (frozen[c]) continue;
                const double* row = &border[(size_t)small_id[c] * k_used];
                const int best = (int)(std::max_element(row, row + k_used) - row);
                if (row[best] <= 0) continue;   // an isolated shell: nothing to merge into
                target[c] = best;
                ++moves;
                auto it = std::lower_bound(nbr.begin(), nbr.end(), std::make_pair(c, (int32_t)-1));
                for (; it != nbr.end() && it->first == c; ++it) frozen[it->second] = 1;
            }
            if (!moves) break;
            for (int64_t f = 0; f < F; ++f)
                if (target[comp[f]] >= 0) label[f] = (uint8_t)target[comp[f]];
            ncomp = components(comp);
        }
    }

    // Slots by surface share; each slot's colour is the area-weighted mean, in linear light
    // (what the eye averages over a patch), of the triangles that were that colour before the
    // merge: a merged speck takes the colour, it does not tint it.
    std::vector<double> share(k_used, 0.0), own(k_used, 0.0), all_w(k_used, 0.0);
    std::vector<Vec3> mean(k_used), all(k_used);
    for (int64_t f = 0; f < F; ++f) {
        const int j = label[f];
        share[j] += area[f];
        all_w[j] += area[f];
        all[j].x += area[f] * lin[f].x; all[j].y += area[f] * lin[f].y; all[j].z += area[f] * lin[f].z;
        if (clustered[f] != j) continue;
        own[j] += area[f];
        mean[j].x += area[f] * lin[f].x; mean[j].y += area[f] * lin[f].y; mean[j].z += area[f] * lin[f].z;
    }
    for (int j = 0; j < k_used; ++j)
        if (own[j] <= 0) { mean[j] = all[j]; own[j] = all_w[j]; }   // only merged specks left: their mean
    std::vector<int> order;
    for (int j = 0; j < k_used; ++j) if (share[j] > 0) order.push_back(j);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return share[a] > share[b]; });
    std::vector<int> slot_of(k_used, 0);
    for (size_t s = 0; s < order.size(); ++s) {
        const int j = order[s];
        slot_of[j] = (int)s;
        auto byte = [&](double c) { return (uint32_t)std::lround(255 * linear_to_srgb(c / own[j])); };
        out.palette.push_back(byte(mean[j].x) << 16 | byte(mean[j].y) << 8 | byte(mean[j].z));
        out.share.push_back(share[j] / total_area);
    }
    out.face_slot.resize(F);
    for (int64_t f = 0; f < F; ++f) out.face_slot[f] = (uint8_t)slot_of[label[f]];
    out.patches = ncomp;
    return out;
}

}  // namespace

void set_report_colors(PrintReport& report, const PrintColors& colors) {
    report.palette = colors.palette;
    report.palette_share = colors.share;
    report.color_patches = colors.patches;
}

}  // namespace trellis
