#include "mesh_print.h"
#include "remesh_dc.h"
#include "tri_bvh.h"
#include "uv_bake.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <unordered_map>

namespace trellis {

namespace {

struct UnionFind {
    std::vector<int32_t> parent;
    explicit UnionFind(int64_t n) : parent((size_t)n) { std::iota(parent.begin(), parent.end(), 0); }
    int32_t find(int32_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    void unite(int32_t a, int32_t b) {
        a = find(a); b = find(b);
        if (a != b) parent[a] = b;
    }
};

// Undirected edge (lo, hi) in the high 62 bits, traversal direction in bit 0 (1 = hi -> lo),
// so sorting groups an edge's uses together. Vertex ids are < 2^31.
inline uint64_t edge_use(int32_t a, int32_t b) {
    return a < b ? ((uint64_t)a << 32 | (uint64_t)b << 1) : ((uint64_t)b << 32 | (uint64_t)a << 1 | 1u);
}

inline double signed_volume6(const float* a, const float* b, const float* c) {
    return (double)a[0] * ((double)b[1] * c[2] - (double)b[2] * c[1])
         - (double)a[1] * ((double)b[0] * c[2] - (double)b[2] * c[0])
         + (double)a[2] * ((double)b[0] * c[1] - (double)b[1] * c[0]);
}

}  // namespace

MeshAudit audit_mesh(const float* verts, int64_t V, const int32_t* faces, int64_t F) {
    MeshAudit a;
    a.vertices = V;
    a.faces = F;
    if (V <= 0 || F <= 0) return a;

    for (int k = 0; k < 3; ++k) { a.bbox_min[k] = 1e30f; a.bbox_max[k] = -1e30f; }
    std::vector<uint64_t> uses;
    uses.reserve((size_t)F * 3);
    UnionFind shells(V);
    std::vector<uint8_t> referenced((size_t)V, 0);
    for (int64_t f = 0; f < F; ++f) {
        const int32_t* t = faces + 3 * f;
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) { ++a.degenerate_faces; continue; }
        for (int j = 0; j < 3; ++j) {
            uses.push_back(edge_use(t[j], t[(j + 1) % 3]));
            referenced[t[j]] = 1;
            const float* p = verts + 3 * t[j];
            for (int k = 0; k < 3; ++k) {
                a.bbox_min[k] = std::min(a.bbox_min[k], p[k]);
                a.bbox_max[k] = std::max(a.bbox_max[k], p[k]);
            }
        }
        shells.unite(t[0], t[1]);
        shells.unite(t[1], t[2]);
    }
    std::sort(uses.begin(), uses.end());
    for (size_t i = 0; i < uses.size();) {
        size_t j = i;
        int forward = 0;
        while (j < uses.size() && (uses[j] >> 1) == (uses[i] >> 1)) { forward += !(uses[j] & 1); ++j; }
        const size_t count = j - i;
        if (count == 1) ++a.boundary_edges;
        else if (count > 2) ++a.nonmanifold_edges;
        else if (forward != 1) ++a.flipped_edges;
        i = j;
    }
    std::vector<uint64_t>().swap(uses);

    // Volume and area per shell; a negative shell is a cavity or an inside-out part.
    std::unordered_map<int32_t, double> shell_volume;
    for (int64_t f = 0; f < F; ++f) {
        const int32_t* t = faces + 3 * f;
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) continue;
        const float *p0 = verts + 3 * t[0], *p1 = verts + 3 * t[1], *p2 = verts + 3 * t[2];
        shell_volume[shells.find(t[0])] += signed_volume6(p0, p1, p2) / 6.0;
        const double u[3] = {(double)p1[0] - p0[0], (double)p1[1] - p0[1], (double)p1[2] - p0[2]};
        const double v[3] = {(double)p2[0] - p0[0], (double)p2[1] - p0[1], (double)p2[2] - p0[2]};
        const double c[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
        a.area += 0.5 * std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    }
    a.components = (int)shell_volume.size();
    for (const auto& kv : shell_volume) {
        a.volume += kv.second;
        if (kv.second < 0) ++a.inverted_components;
    }

    // Pinched vertices: the faces around a vertex should form ONE fan. Each face contributes
    // the link edge between its two other vertices; count the link's connected pieces.
    std::vector<int32_t> offset((size_t)V + 1, 0);
    for (int64_t f = 0; f < F; ++f) {
        const int32_t* t = faces + 3 * f;
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) continue;
        for (int j = 0; j < 3; ++j) ++offset[(size_t)t[j] + 1];
    }
    for (int64_t v = 0; v < V; ++v) offset[(size_t)v + 1] += offset[(size_t)v];
    std::vector<int32_t> incident((size_t)offset[(size_t)V]);
    {
        std::vector<int32_t> cursor(offset.begin(), offset.end() - 1);
        for (int64_t f = 0; f < F; ++f) {
            const int32_t* t = faces + 3 * f;
            if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) continue;
            for (int j = 0; j < 3; ++j) incident[(size_t)cursor[t[j]]++] = (int32_t)f;
        }
    }
    std::vector<std::pair<int32_t, int32_t>> link;
    std::vector<int32_t> ids;
    for (int64_t v = 0; v < V; ++v) {
        const int degree = offset[(size_t)v + 1] - offset[(size_t)v];
        if (degree < 2) continue;
        link.clear();
        ids.clear();
        for (int j = offset[(size_t)v]; j < offset[(size_t)v + 1]; ++j) {
            const int32_t* t = faces + 3 * (int64_t)incident[(size_t)j];
            const int at = t[0] == v ? 0 : (t[1] == v ? 1 : 2);
            link.emplace_back(t[(at + 1) % 3], t[(at + 2) % 3]);
            ids.push_back(link.back().first);
            ids.push_back(link.back().second);
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        UnionFind fan((int64_t)ids.size());
        int pieces = (int)ids.size();
        for (const auto& e : link) {
            const int32_t x = fan.find((int32_t)(std::lower_bound(ids.begin(), ids.end(), e.first) - ids.begin()));
            const int32_t y = fan.find((int32_t)(std::lower_bound(ids.begin(), ids.end(), e.second) - ids.begin()));
            if (x != y) { fan.parent[x] = y; --pieces; }
        }
        if (pieces > 1) ++a.nonmanifold_vertices;
    }
    return a;
}

std::string audit_summary(const MeshAudit& a) {
    char line[512];
    std::snprintf(line, sizeof line,
                  "watertight: %s  V=%lld F=%lld shells=%d  holes(boundary edges)=%lld "
                  "non-manifold edges=%lld flipped=%lld degenerate=%lld inverted shells=%d "
                  "pinched vertices=%lld  volume=%.6g area=%.6g",
                  a.watertight() ? "yes" : "NO", (long long)a.vertices, (long long)a.faces, a.components,
                  (long long)a.boundary_edges, (long long)a.nonmanifold_edges, (long long)a.flipped_edges,
                  (long long)a.degenerate_faces, a.inverted_components, (long long)a.nonmanifold_vertices,
                  a.volume, a.area);
    return line;
}

int weld_exact(std::vector<float>& verts, std::vector<int32_t>& faces) {
    const size_t V = verts.size() / 3;
    struct Key {
        uint32_t x, y, z;
        bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct Hash {
        size_t operator()(const Key& k) const {
            uint64_t h = k.x * 0x9E3779B97F4A7C15ull;
            h ^= (h >> 29) + k.y * 0xBF58476D1CE4E5B9ull;
            h ^= (h >> 31) + k.z * 0x94D049BB133111EBull;
            return (size_t)(h ^ (h >> 32));
        }
    };
    std::unordered_map<Key, int32_t, Hash> seen;
    seen.reserve(V);
    std::vector<int32_t> remap(V);
    std::vector<float> kept;
    kept.reserve(verts.size());
    for (size_t v = 0; v < V; ++v) {
        float p[3] = {verts[3 * v], verts[3 * v + 1], verts[3 * v + 2]};
        for (float& c : p) if (c == 0.0f) c = 0.0f;   // -0 and +0 are the same position
        Key k;
        std::memcpy(&k, p, sizeof k);
        const auto it = seen.emplace(k, (int32_t)(kept.size() / 3));
        if (it.second) kept.insert(kept.end(), p, p + 3);
        remap[v] = it.first->second;
    }
    for (int32_t& f : faces) f = remap[(size_t)f];
    const int merged = (int)(V - kept.size() / 3);
    verts.swap(kept);
    return merged;
}

int64_t repair_for_print(std::vector<float>& verts, std::vector<int32_t>& faces) {
    if (audit_mesh(verts.data(), (int64_t)verts.size() / 3, faces.data(), (int64_t)faces.size() / 3).watertight())
        return 0;
    int64_t changes = 0;

    // 1. Degenerate and duplicate faces. Faces over the same three vertices cancel by
    //    orientation: an opposite pair is a zero-thickness fin (both go), same-wound copies
    //    collapse to one.
    {
        struct Entry { int32_t v[3]; int32_t parity; int32_t face; };
        std::vector<Entry> entries;
        entries.reserve(faces.size() / 3);
        for (size_t f = 0; f + 2 < faces.size(); f += 3) {
            int32_t a = faces[f], b = faces[f + 1], c = faces[f + 2];
            if (a == b || b == c || a == c) continue;
            int32_t parity = 1;
            if (a > b) { std::swap(a, b); parity = -parity; }
            if (b > c) { std::swap(b, c); parity = -parity; }
            if (a > b) { std::swap(a, b); parity = -parity; }
            entries.push_back({{a, b, c}, parity, (int32_t)(f / 3)});
        }
        std::sort(entries.begin(), entries.end(), [](const Entry& l, const Entry& r) {
            return std::lexicographical_compare(l.v, l.v + 3, r.v, r.v + 3);
        });
        std::vector<int32_t> kept;
        kept.reserve(faces.size());
        for (size_t i = 0; i < entries.size();) {
            size_t j = i;
            int net = 0;
            while (j < entries.size() && std::equal(entries[i].v, entries[i].v + 3, entries[j].v)) net += entries[j++].parity;
            if (net != 0) {
                const int32_t* v = entries[i].v;
                kept.push_back(v[0]);
                kept.push_back(net > 0 ? v[1] : v[2]);
                kept.push_back(net > 0 ? v[2] : v[1]);
            }
            i = j;
        }
        changes += (int64_t)faces.size() / 3 - (int64_t)kept.size() / 3;
        faces.swap(kept);
    }

    // 2. One winding per shell, then close whatever is still open.
    clean_mesh((int)(verts.size() / 3), faces);
    const size_t before_fill = faces.size();
    fill_small_holes(faces, 1 << 20);
    changes += (int64_t)(faces.size() - before_fill) / 3;

    // 3. Outward normals: the solid remesh leaves no cavities, so a shell with negative
    //    volume is inside-out.
    const int64_t V = (int64_t)verts.size() / 3, F = (int64_t)faces.size() / 3;
    UnionFind shells(V);
    for (int64_t f = 0; f < F; ++f) { shells.unite(faces[3 * f], faces[3 * f + 1]); shells.unite(faces[3 * f + 1], faces[3 * f + 2]); }
    std::unordered_map<int32_t, double> volume;
    for (int64_t f = 0; f < F; ++f)
        volume[shells.find(faces[3 * f])] += signed_volume6(&verts[3 * (size_t)faces[3 * f]], &verts[3 * (size_t)faces[3 * f + 1]],
                                                            &verts[3 * (size_t)faces[3 * f + 2]]);
    for (int64_t f = 0; f < F; ++f)
        if (volume[shells.find(faces[3 * f])] < 0) { std::swap(faces[3 * f + 1], faces[3 * f + 2]); ++changes; }

    // 4. Drop vertices nothing references any more.
    std::vector<int32_t> remap((size_t)V, -1);
    std::vector<float> kept;
    kept.reserve(verts.size());
    for (int32_t& v : faces) {
        if (remap[(size_t)v] < 0) {
            remap[(size_t)v] = (int32_t)(kept.size() / 3);
            kept.insert(kept.end(), &verts[3 * (size_t)v], &verts[3 * (size_t)v] + 3);
        }
        v = remap[(size_t)v];
    }
    verts.swap(kept);
    return changes;
}

float wall_cells_for(float wall_mm, float size_mm, int res, float model_extent) {
    if (!(wall_mm > 0) || !(size_mm > 0) || res <= 0 || !(model_extent > 0)) return 0.0f;
    return wall_mm / size_mm * model_extent * (float)res;
}

void set_print_base(RemeshOptions& remesh, const float* verts, int64_t V, int up_axis, float base) {
    if (base == 0) return;
    remesh.base_axis = up_axis;
    remesh.base_auto = base < 0;
    remesh.base_plane = base_plane_of(verts, V, up_axis, base < 0 ? 0.01f : base);
    remesh.base_plane_max = base_plane_of(verts, V, up_axis, 0.04f);
}

float base_plane_of(const float* verts, int64_t V, int axis, float fraction) {
    float lo = 1e30f, hi = -1e30f;
    for (int64_t v = 0; v < V; ++v) {
        lo = std::min(lo, verts[3 * v + axis]);
        hi = std::max(hi, verts[3 * v + axis]);
    }
    return V > 0 ? lo + fraction * (hi - lo) : 0.0f;
}

Mesh make_printable(const Mesh& in, const PrintOptions& options) {
    Mesh source = in;
    weld_vertices(source.verts, source.faces, nullptr, 1.0f / ((float)options.res * 8.0f));
    fill_small_holes(source.faces);
    const TriBvh bvh = TriBvh::build(source.verts.data(), source.V(), source.faces.data(), source.F());

    RemeshOptions remesh;
    remesh.band = std::max(1, options.band);
    remesh.solid = true;
    remesh.seal = options.seal;
    remesh.min_wall_cells = options.wall_cells;
    set_print_base(remesh, source.verts.data(), source.V(), options.up_axis, options.base);
    const auto t0 = std::chrono::steady_clock::now();
    auto lap = [&t0]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    Mesh solid = remesh_narrow_band_dc(source.verts.data(), source.V(), source.faces.data(), source.F(),
                                       bvh, options.res, remesh);
    printf("  [print] solid remesh %.1fs\n", lap());
    fflush(stdout);
    if (solid.F() == 0) return solid;
    drop_small_components(solid.verts, solid.faces, 0.02f);

    Mesh out;
    if (options.faces > 0 && solid.F() > options.faces)
        decimate_qem(solid.verts, solid.V(), solid.faces, solid.F(), options.faces, out.verts, out.faces,
                     /*keep_topology=*/true);
    else
        out = std::move(solid);
    repair_for_print(out.verts, out.faces);
    printf("  [print] decimate + repair done at %.1fs\n", lap());
    return out;
}

}  // namespace trellis
