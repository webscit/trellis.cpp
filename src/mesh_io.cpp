#include "mesh_io.h"
#include "mesh_print.h"   // weld_exact

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>

namespace trellis {

namespace {

std::string lower_ext(const std::string& path) {
    const size_t d = path.find_last_of('.');
    if (d == std::string::npos) return "";
    std::string e = path.substr(d + 1);
    for (char& c : e) c = (char)std::tolower((unsigned char)c);
    return e;
}

bool load_gltf(const std::string& path, Mesh& out, std::string& err) {
    cgltf_options opt{};
    cgltf_data* data = nullptr;
    cgltf_result r = cgltf_parse_file(&opt, path.c_str(), &data);
    if (r != cgltf_result_success) { err = "cannot parse glTF: " + path; return false; }
    struct Guard { cgltf_data* d; ~Guard() { cgltf_free(d); } } guard{data};
    r = cgltf_load_buffers(&opt, data, path.c_str());
    if (r != cgltf_result_success) { err = "cannot load glTF buffers: " + path; return false; }
    if (cgltf_validate(data) != cgltf_result_success) { err = "invalid glTF: " + path; return false; }

    std::vector<float> verts;
    std::vector<int32_t> faces;
    for (cgltf_size ni = 0; ni < data->nodes_count; ++ni) {
        const cgltf_node* node = &data->nodes[ni];
        if (!node->mesh) continue;
        float M[16];
        cgltf_node_transform_world(node, M);   // column-major
        for (cgltf_size pi = 0; pi < node->mesh->primitives_count; ++pi) {
            const cgltf_primitive* prim = &node->mesh->primitives[pi];
            if (prim->type != cgltf_primitive_type_triangles) continue;
            const cgltf_accessor* pos = nullptr;
            for (cgltf_size a = 0; a < prim->attributes_count; ++a)
                if (prim->attributes[a].type == cgltf_attribute_type_position) pos = prim->attributes[a].data;
            if (!pos || pos->count == 0) continue;
            const int64_t base = (int64_t)verts.size() / 3;
            if (base + (int64_t)pos->count > INT32_MAX) { err = "mesh too large"; return false; }
            std::vector<float> p(pos->count * 3);
            if (cgltf_accessor_unpack_floats(pos, p.data(), p.size()) != p.size()) {
                err = "cannot read POSITION accessor"; return false;
            }
            for (cgltf_size v = 0; v < pos->count; ++v) {
                const float x = p[v * 3], y = p[v * 3 + 1], z = p[v * 3 + 2];
                verts.push_back(M[0] * x + M[4] * y + M[8]  * z + M[12]);
                verts.push_back(M[1] * x + M[5] * y + M[9]  * z + M[13]);
                verts.push_back(M[2] * x + M[6] * y + M[10] * z + M[14]);
            }
            // A negative-determinant transform mirrors the mesh; flip winding to keep it outward.
            const float det = M[0] * (M[5] * M[10] - M[9] * M[6]) - M[4] * (M[1] * M[10] - M[9] * M[2])
                            + M[8] * (M[1] * M[6] - M[5] * M[2]);
            const bool flip = det < 0;
            auto push_tri = [&](cgltf_size a, cgltf_size b, cgltf_size c) -> bool {
                if (a >= pos->count || b >= pos->count || c >= pos->count) return false;
                faces.push_back((int32_t)(base + a));
                faces.push_back((int32_t)(base + (flip ? c : b)));
                faces.push_back((int32_t)(base + (flip ? b : c)));
                return true;
            };
            if (prim->indices) {
                const cgltf_size n = prim->indices->count / 3 * 3;
                for (cgltf_size i = 0; i < n; i += 3)
                    if (!push_tri(cgltf_accessor_read_index(prim->indices, i),
                                  cgltf_accessor_read_index(prim->indices, i + 1),
                                  cgltf_accessor_read_index(prim->indices, i + 2))) {
                        err = "index out of range in " + path; return false;
                    }
            } else {
                for (cgltf_size i = 0; i + 2 < pos->count; i += 3) push_tri(i, i + 1, i + 2);
            }
        }
    }
    if (faces.empty()) { err = "no triangle primitives in " + path; return false; }
    out.verts = std::move(verts);
    out.faces = std::move(faces);
    return true;
}

bool load_obj(const std::string& path, Mesh& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<float> verts;
    std::vector<int32_t> faces;
    std::string line;
    std::vector<int64_t> poly;
    size_t lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (line.size() < 2) continue;
        if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t')) {
            float x, y, z;
            if (std::sscanf(line.c_str() + 2, "%f %f %f", &x, &y, &z) != 3 ||
                !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                err = "bad vertex at " + path + ":" + std::to_string(lineno); return false;
            }
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        } else if (line[0] == 'f' && (line[1] == ' ' || line[1] == '\t')) {
            poly.clear();
            std::istringstream is(line.substr(2));
            std::string tok;
            const int64_t nv = (int64_t)verts.size() / 3;
            while (is >> tok) {
                char* end = nullptr;
                const long long i = std::strtoll(tok.c_str(), &end, 10);   // "i", "i/t", "i//n", "i/t/n"
                if (end == tok.c_str() || i == 0) { err = "bad face at " + path + ":" + std::to_string(lineno); return false; }
                const int64_t idx = i > 0 ? i - 1 : nv + i;
                if (idx < 0 || idx >= nv) { err = "face index out of range at " + path + ":" + std::to_string(lineno); return false; }
                poly.push_back(idx);
            }
            for (size_t k = 2; k < poly.size(); ++k) {     // fan triangulation
                faces.push_back((int32_t)poly[0]);
                faces.push_back((int32_t)poly[k - 1]);
                faces.push_back((int32_t)poly[k]);
            }
        }
    }
    if (faces.empty()) { err = "no faces in " + path; return false; }
    if (verts.size() / 3 > (size_t)INT32_MAX) { err = "mesh too large"; return false; }
    out.verts = std::move(verts);
    out.faces = std::move(faces);
    return true;
}


// Binary STL: 80-byte header, uint32 facet count, then 50 bytes per facet (normal, 3 vertices,
// uint16 attribute). Detected by the exact size, because many binary files begin with "solid".
// ASCII STL: "solid ... facet normal ... outer loop / vertex x y z ... endloop endfacet ...".
bool load_stl(const std::string& path, Mesh& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<float> verts;
    std::vector<int32_t> faces;
    bool binary = false;
    uint32_t n = 0;
    if (data.size() >= 84) {
        std::memcpy(&n, data.data() + 80, 4);
        binary = data.size() == 84 + 50ull * n;
    }
    if (binary) {
        if ((uint64_t)n * 3 > (uint64_t)INT32_MAX) { err = "mesh too large: " + path; return false; }
        verts.resize((size_t)n * 9);
        faces.resize((size_t)n * 3);
        for (uint32_t i = 0; i < n; ++i) {
            std::memcpy(verts.data() + (size_t)i * 9, data.data() + 84 + 50ull * i + 12, 36);
            for (int k = 0; k < 3; ++k) faces[(size_t)i * 3 + k] = (int32_t)(i * 3 + k);
        }
    } else {
        size_t p = 0;
        while (p < data.size() && std::isspace((unsigned char)data[p])) ++p;
        if (data.compare(p, 5, "solid") != 0) {
            err = "not an STL file (neither binary of the right size nor ASCII): " + path; return false;
        }
        std::istringstream is(data);
        std::string tok;
        std::vector<float> loop;
        while (is >> tok) {
            if (tok == "vertex") {
                float x, y, z;
                if (!(is >> x >> y >> z)) { err = "bad vertex in " + path; return false; }
                loop.push_back(x); loop.push_back(y); loop.push_back(z);
            } else if (tok == "endloop" || tok == "endfacet") {
                const int nv = (int)loop.size() / 3;
                if (tok == "endloop" && nv < 3) { err = "facet with fewer than 3 vertices in " + path; return false; }
                if (nv >= 3) {
                    const int32_t base = (int32_t)(verts.size() / 3);
                    verts.insert(verts.end(), loop.begin(), loop.end());
                    for (int k = 2; k < nv; ++k) {   // fan, for the rare polygon facet
                        faces.push_back(base); faces.push_back(base + k - 1); faces.push_back(base + k);
                    }
                    if (verts.size() / 3 > (size_t)INT32_MAX) { err = "mesh too large: " + path; return false; }
                }
                loop.clear();
            }
        }
    }
    for (float v : verts)
        if (!std::isfinite(v)) { err = "non-finite vertex in " + path; return false; }
    if (faces.empty()) { err = "no facets in " + path; return false; }
    out.verts = std::move(verts);
    out.faces = std::move(faces);
    weld_exact(out.verts, out.faces);   // mesh_print.h: bit-identical positions, -0 == +0
    std::vector<int32_t> kept;
    kept.reserve(out.faces.size());
    for (size_t t = 0; t + 2 < out.faces.size(); t += 3) {
        const int32_t a = out.faces[t], b = out.faces[t + 1], c = out.faces[t + 2];
        if (a == b || b == c || a == c) continue;   // a facet with a repeated corner
        kept.push_back(a); kept.push_back(b); kept.push_back(c);
    }
    out.faces.swap(kept);
    if (out.faces.empty()) { err = "every facet of " + path + " is degenerate"; return false; }
    return true;
}

}  // namespace

bool is_stl_path(const std::string& path) { return lower_ext(path) == "stl"; }

bool parse_mesh_up(const std::string& s, MeshUp& out) {
    if      (s == "auto")               out = MeshUp::Auto;
    else if (s == "x"  || s == "+x")    out = MeshUp::X;
    else if (s == "-x")                 out = MeshUp::NX;
    else if (s == "y"  || s == "+y")    out = MeshUp::Y;
    else if (s == "-y")                 out = MeshUp::NY;
    else if (s == "z"  || s == "+z")    out = MeshUp::Z;
    else if (s == "-z")                 out = MeshUp::NZ;
    else return false;
    return true;
}

bool AxisMap::identity() const {
    for (int i = 0; i < 3; ++i) if (axis[i] != i || sign[i] != 1) return false;
    return true;
}

AxisMap AxisMap::inverse() const {
    AxisMap r;
    for (int i = 0; i < 3; ++i) { r.axis[axis[i]] = i; r.sign[axis[i]] = sign[i]; }
    return r;
}

AxisMap AxisMap::after(const AxisMap& first) const {
    AxisMap r;
    for (int i = 0; i < 3; ++i) { r.axis[i] = first.axis[axis[i]]; r.sign[i] = sign[i] * first.sign[axis[i]]; }
    return r;
}

void AxisMap::apply(float* v, size_t count, size_t stride) const {
    for (size_t k = 0; k < count; ++k, v += stride) {
        const float in[3] = {v[0], v[1], v[2]};
        for (int i = 0; i < 3; ++i) v[i] = sign[i] < 0 ? -in[axis[i]] : in[axis[i]];
    }
}

namespace {

AxisMap make_map(int a0, int s0, int a1, int s1, int a2, int s2) {
    AxisMap m;
    m.axis[0] = a0; m.sign[0] = s0; m.axis[1] = a1; m.sign[1] = s1; m.axis[2] = a2; m.sign[2] = s2;
    return m;
}

// File -> glTF: the rotation that takes the file axis `up` to +y (Auto resolved by the caller).
AxisMap up_map(MeshUp up) {
    switch (up) {
        case MeshUp::Z:  return make_map(0, 1, 2, 1, 1, -1);   // (x, y, z) -> (x, z, -y)
        case MeshUp::NZ: return make_map(0, 1, 2, -1, 1, 1);   // (x, y, z) -> (x, -z, y)
        case MeshUp::NY: return make_map(0, 1, 1, -1, 2, -1);  // (x, y, z) -> (x, -y, -z)
        case MeshUp::X:  return make_map(1, -1, 0, 1, 2, 1);   // (x, y, z) -> (-y, x, z)
        case MeshUp::NX: return make_map(1, 1, 0, -1, 2, 1);   // (x, y, z) -> (y, -x, z)
        default:         return AxisMap{};                     // y up already
    }
}

// glTF: a turn of `yaw` degrees about +y, counter-clockwise seen from above (right-handed).
AxisMap yaw_map(int yaw) {
    switch (((yaw % 360) + 360) % 360) {
        case 90:  return make_map(2, 1, 1, 1, 0, -1);    // (x, y, z) -> (z, y, -x)
        case 180: return make_map(0, -1, 1, 1, 2, -1);   // (x, y, z) -> (-x, y, -z)
        case 270: return make_map(2, -1, 1, 1, 0, 1);    // (x, y, z) -> (-z, y, x)
        default:  return AxisMap{};
    }
}

MeshUp resolve_up(const std::string& path, MeshUp up) {
    return up != MeshUp::Auto ? up : is_stl_path(path) ? MeshUp::Z : MeshUp::Y;
}

}  // namespace

AxisMap mesh_load_map(const std::string& path, MeshOrient o) {
    return yaw_map(o.yaw).after(up_map(resolve_up(path, o.up)));
}

AxisMap mesh_orient_undo_grid(const std::string& path, MeshOrient o) {
    const AxisMap to_grid = make_map(0, 1, 2, -1, 1, 1);   // glTF (x, y, z) -> grid (x, -z, y)
    const AxisMap native = up_map(resolve_up(path, MeshUp::Auto));
    // grid -> glTF -> (undo the load) file -> the file's own convention -> grid
    return to_grid.after(native.after(mesh_load_map(path, o).inverse().after(to_grid.inverse())));
}

std::string mesh_front_axis(const std::string& path, MeshOrient o) {
    // The file direction the load sends to glTF +z: out.z = sign[2] * in[axis[2]].
    const AxisMap m = mesh_load_map(path, o);
    return std::string(m.sign[2] < 0 ? "-" : "+") + "xyz"[m.axis[2]];
}

bool load_mesh(const std::string& path, Mesh& out, std::string& err, MeshOrient orient) {
    char magic[4] = {0, 0, 0, 0};
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "cannot open " + path; return false; }
        const size_t n = std::fread(magic, 1, 4, f);
        std::fclose(f);
        if (n < 4) { err = "file too short: " + path; return false; }
    }
    const std::string ext = lower_ext(path);
    bool ok = false;
    try {
        if (std::memcmp(magic, "glTF", 4) == 0 || ext == "glb" || ext == "gltf") ok = load_gltf(path, out, err);
        else if (ext == "obj") ok = load_obj(path, out, err);
        else if (ext == "stl") ok = load_stl(path, out, err);
        else { err = "unsupported mesh format (expected .glb, .gltf, .obj or .stl): " + path; return false; }
    } catch (const std::exception& e) {
        err = std::string("failed to load ") + path + ": " + e.what();
        return false;
    }
    if (!ok) return false;
    if (orient.yaw % 90 != 0) { err = "the mesh turn must be 0, 90, 180 or 270 degrees"; return false; }
    // File -> glTF y-up, then the turn: a proper rotation, so the winding is unchanged.
    mesh_load_map(path, orient).apply(out.verts);
    return true;
}

bool normalize_mesh_trellis(Mesh& m, MeshNorm& norm) {
    if (m.verts.size() < 9) return false;
    double lo[3] = { 1e300, 1e300, 1e300 }, hi[3] = { -1e300, -1e300, -1e300 };
    for (size_t i = 0; i < m.verts.size(); i += 3)
        for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], (double)m.verts[i + a]); hi[a] = std::max(hi[a], (double)m.verts[i + a]); }
    const double ext = std::max({ hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2] });
    if (!(ext > 0) || !std::isfinite(ext)) return false;
    for (int a = 0; a < 3; ++a) norm.center[a] = (float)((lo[a] + hi[a]) / 2);
    norm.scale = (float)(0.99999 / ext);
    for (size_t i = 0; i < m.verts.size(); i += 3) {
        const float x = (m.verts[i]     - norm.center[0]) * norm.scale;
        const float y = (m.verts[i + 1] - norm.center[1]) * norm.scale;
        const float z = (m.verts[i + 2] - norm.center[2]) * norm.scale;
        m.verts[i] = x; m.verts[i + 1] = -z; m.verts[i + 2] = y;
        for (int a = 0; a < 3; ++a) m.verts[i + a] = std::min(0.5f, std::max(-0.5f, m.verts[i + a]));
    }
    return true;
}

void restore_mesh_transform(std::vector<float>& v, const MeshNorm& norm) {
    // grid (x, y, z) = (gx, -gz_glTF, gy_glTF)  =>  glTF centre (cx, cy, cz) is (cx, -cz, cy) in grid axes.
    const float c[3] = { norm.center[0], -norm.center[2], norm.center[1] };
    const float inv = 1.0f / norm.scale;
    for (size_t i = 0; i + 2 < v.size(); i += 3)
        for (int a = 0; a < 3; ++a) v[i + a] = v[i + a] * inv + c[a];
}

}  // namespace trellis
