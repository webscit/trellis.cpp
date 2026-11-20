// Mesh input tests for --refine-mesh (docs/spec/31-mesh-refine.md). CPU only.
//
//   trellis-test-mesh-io <mesh.glb> [expect_V expect_F]
//
// Loads the mesh, checks V/F against the expected counts (tools/glb_metrics.py), normalises it
// (normalize_mesh_trellis) and checks the bounds, the axis swap and the inverse transform. Then
// writes small OBJ / glTF files to the temp directory: fan triangulation, negative indices, and
// malformed inputs that must fail with an error rather than crash.
#include "mesh_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

using namespace trellis;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static std::string tmpfile(const std::string& name, const std::string& body) {
    const auto p = std::filesystem::temp_directory_path() / ("trellis-test-mesh-io-" + name);
    std::ofstream(p, std::ios::binary) << body;
    return p.string();
}

int main(int argc, char** argv) {
    if (argc > 1) {
        Mesh m; std::string err;
        const bool ok = load_mesh(argv[1], m, err);
        CHECK(ok, "load %s: %s", argv[1], err.c_str());
        if (ok) {
            printf("%s: V=%d F=%d\n", argv[1], m.V(), m.F());
            if (argc > 3) CHECK(m.V() == atoi(argv[2]) && m.F() == atoi(argv[3]),
                                "counts %d/%d, expected %s/%s", m.V(), m.F(), argv[2], argv[3]);
            float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
            for (int i = 0; i < m.V(); ++i) for (int a = 0; a < 3; ++a) {
                lo[a] = std::fmin(lo[a], m.verts[i * 3 + a]); hi[a] = std::fmax(hi[a], m.verts[i * 3 + a]); }
            Mesh n = m; MeshNorm nm;
            CHECK(normalize_mesh_trellis(n, nm), "normalize");
            float nlo[3] = { 1e30f, 1e30f, 1e30f }, nhi[3] = { -1e30f, -1e30f, -1e30f };
            for (int i = 0; i < n.V(); ++i) for (int a = 0; a < 3; ++a) {
                nlo[a] = std::fmin(nlo[a], n.verts[i * 3 + a]); nhi[a] = std::fmax(nhi[a], n.verts[i * 3 + a]); }
            float ext = 0;
            for (int a = 0; a < 3; ++a) {
                CHECK(nlo[a] >= -0.5f && nhi[a] <= 0.5f, "axis %d out of [-0.5,0.5]: %g..%g", a, nlo[a], nhi[a]);
                ext = std::fmax(ext, nhi[a] - nlo[a]);
            }
            CHECK(std::fabs(ext - 0.99999f) < 1e-4f, "max extent %g", ext);
            // (x, y, z) -> (x, -z, y): grid y extent = input z extent, grid z = input y.
            CHECK(std::fabs((nhi[1] - nlo[1]) - (hi[2] - lo[2]) * nm.scale) < 1e-4f, "grid y != -input z");
            CHECK(std::fabs((nhi[2] - nlo[2]) - (hi[1] - lo[1]) * nm.scale) < 1e-4f, "grid z != input y");
            printf("normalised bounds: [%.4f %.4f] [%.4f %.4f] [%.4f %.4f], scale %.6f\n",
                   nlo[0], nhi[0], nlo[1], nhi[1], nlo[2], nhi[2], nm.scale);
            // Inverse: restore in grid axes, then the writers' (x, y, z) -> (x, z, -y) gives the input.
            std::vector<float> r = n.verts;
            restore_mesh_transform(r, nm);
            double md = 0;
            for (int i = 0; i < n.V(); ++i) {
                const float gx = r[i * 3], gy = r[i * 3 + 1], gz = r[i * 3 + 2];
                const float out[3] = { gx, gz, -gy };
                for (int a = 0; a < 3; ++a) md = std::fmax(md, std::fabs(out[a] - m.verts[i * 3 + a]));
            }
            const double span = std::fmax(hi[0] - lo[0], std::fmax(hi[1] - lo[1], hi[2] - lo[2]));
            printf("restore round trip: max |d| = %.3e (%.3e of the extent)\n", md, md / span);
            CHECK(md / span < 1e-5, "restore round trip %g", md);
        }
    }

    // OBJ: a quad (fan-split), negative indices, v/t/n tokens.
    {
        const std::string p = tmpfile("quad.obj",
            "# test\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1 4/1\nv 0 0 1\nf -5//1 -4//1 -1//1\n");
        Mesh m; std::string err;
        CHECK(load_mesh(p, m, err) && m.V() == 5 && m.F() == 3, "obj quad: %s V=%d F=%d", err.c_str(), m.V(), m.F());
        if (m.F() == 3) CHECK(m.faces[0] == 0 && m.faces[1] == 1 && m.faces[2] == 2 &&
                              m.faces[3] == 0 && m.faces[4] == 2 && m.faces[5] == 3 &&
                              m.faces[6] == 0 && m.faces[7] == 1 && m.faces[8] == 4, "obj fan/negative indices");
    }
    // Minimal glTF (embedded base64 buffer) with a translated node.
    {
        // 3 float3 positions (36 bytes) + 3 uint16 indices (6 bytes) + pad 2.
        const unsigned char bytes[44] = {
            0,0,0,0, 0,0,0,0, 0,0,0,0,   0,0,128,63, 0,0,0,0, 0,0,0,0,   0,0,0,0, 0,0,128,63, 0,0,0,0,
            0,0, 1,0, 2,0, 0,0 };
        static const char* b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string enc;
        for (int i = 0; i < 44; i += 3) {
            const int n = (bytes[i] << 16) | ((i + 1 < 44 ? bytes[i + 1] : 0) << 8) | (i + 2 < 44 ? bytes[i + 2] : 0);
            enc += b64[(n >> 18) & 63]; enc += b64[(n >> 12) & 63];
            enc += i + 1 < 44 ? b64[(n >> 6) & 63] : '='; enc += i + 2 < 44 ? b64[n & 63] : '=';
        }
        const std::string js =
            "{\"asset\":{\"version\":\"2.0\"},\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
            "\"nodes\":[{\"mesh\":0,\"translation\":[10,0,0]}],"
            "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},\"indices\":1}]}],"
            "\"buffers\":[{\"byteLength\":44,\"uri\":\"data:application/octet-stream;base64," + enc + "\"}],"
            "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},{\"buffer\":0,\"byteOffset\":36,\"byteLength\":6}],"
            "\"accessors\":[{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\",\"min\":[0,0,0],\"max\":[1,1,0]},"
            "{\"bufferView\":1,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"}]}";
        const std::string p = tmpfile("tri.gltf", js);
        Mesh m; std::string err;
        const bool ok = load_mesh(p, m, err);
        CHECK(ok && m.V() == 3 && m.F() == 1, "gltf: %s", err.c_str());
        if (ok && m.V() == 3) CHECK(m.verts[0] == 10.0f && m.verts[3] == 11.0f, "gltf node translation not applied");
    }
    // STL: a unit cube as a triangle soup (12 facets, 36 corners), binary and ASCII.
    {
        static const float P[8][3] = { {0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1} };
        static const int T[12][3] = { {0,2,1},{0,3,2},{4,5,6},{4,6,7},{0,1,5},{0,5,4},
                                      {1,2,6},{1,6,5},{2,3,7},{2,7,6},{3,0,4},{3,4,7} };
        // Binary, with a header that starts with "solid" (many exporters do this).
        std::string bin(80, ' ');
        bin.replace(0, 11, "solid cube ");
        const uint32_t n = 13;   // 12 facets + one degenerate (a repeated corner) that must be dropped
        bin.append((const char*)&n, 4);
        auto facet = [&](const float* a, const float* b, const float* c) {
            const float zero[3] = { 0, 0, 0 };
            bin.append((const char*)zero, 12);
            bin.append((const char*)a, 12); bin.append((const char*)b, 12); bin.append((const char*)c, 12);
            bin.append(2, '\0');
        };
        for (const auto& t : T) facet(P[t[0]], P[t[1]], P[t[2]]);
        facet(P[0], P[0], P[1]);
        // ASCII, same cube; -0 must weld with +0.
        std::string asc = "solid cube\n";
        for (const auto& t : T) {
            asc += "  facet normal 0 0 0\n    outer loop\n";
            for (int k = 0; k < 3; ++k) {
                char line[96];
                const float* p = P[t[k]];
                snprintf(line, sizeof line, "      vertex %s %g %g\n", p[0] == 0 && t[k] == 3 ? "-0" : (p[0] ? "1" : "0"), p[1], p[2]);
                asc += line;
            }
            asc += "    endloop\n  endfacet\n";
        }
        asc += "endsolid cube\n";
        for (const auto& c : { std::make_pair("cube_bin.stl", bin), std::make_pair("cube_ascii.stl", asc) }) {
            const std::string p = tmpfile(c.first, c.second);
            Mesh m; std::string err;
            const bool ok = load_mesh(p, m, err, MeshUp::Y);
            CHECK(ok && m.V() == 8 && m.F() == 12, "%s: %s V=%d F=%d (want 8/12)", c.first, err.c_str(), m.V(), m.F());
            // Closed after welding: every undirected edge is used by exactly two faces.
            std::map<std::pair<int,int>, int> edges;
            for (int f = 0; f < m.F(); ++f)
                for (int k = 0; k < 3; ++k) {
                    int a = m.faces[f * 3 + k], b = m.faces[f * 3 + (k + 1) % 3];
                    if (a > b) std::swap(a, b);
                    ++edges[{ a, b }];
                }
            bool closed = !edges.empty();
            for (const auto& e : edges) closed = closed && e.second == 2;
            CHECK(closed, "%s: welded cube is not closed", c.first);
            // Up axis: Auto on an .stl is z-up, rotated to glTF y-up as (x, y, z) -> (x, z, -y).
            Mesh z; std::string zerr;
            CHECK(load_mesh(p, z, zerr) && z.V() == m.V(), "%s auto: %s", c.first, zerr.c_str());
            bool rot = z.V() == m.V();
            for (int i = 0; rot && i < m.V(); ++i)
                rot = z.verts[i * 3] == m.verts[i * 3] && z.verts[i * 3 + 1] == m.verts[i * 3 + 2] &&
                      z.verts[i * 3 + 2] == -m.verts[i * 3 + 1];
            CHECK(rot, "%s: auto up axis is not the z-up rotation", c.first);
            // ... which TRELLIS normalisation then maps back: the grid frame is z-up.
            if (rot) {
                MeshNorm nm; Mesh g = z;
                CHECK(normalize_mesh_trellis(g, nm), "normalize stl");
                bool same = true;
                for (int i = 0; i < g.V(); ++i)
                    for (int a = 0; a < 3; ++a)
                        same = same && std::fabs(g.verts[i * 3 + a] - (m.verts[i * 3 + a] - 0.5f) * nm.scale) < 1e-6f;
                CHECK(same, "%s: z-up file does not land unrotated in the grid frame", c.first);
            }
        }
        // A binary file whose size disagrees with its facet count, not starting with "solid".
        std::string trunc(84, '\0');
        trunc[80] = 5;
        { Mesh m; std::string err;
          CHECK(!load_mesh(tmpfile("trunc.stl", trunc), m, err), "truncated binary STL accepted"); }
    }
    // Malformed inputs: an error, never a crash.
    const std::pair<const char*, const char*> bad[] = {
        { "empty.obj", "" },
        { "noface.obj", "v 0 0 0\nv 1 0 0\n" },
        { "badidx.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n" },
        { "zeroidx.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n" },
        { "trunc.glb", "glTF\x02" },
        { "junk.glb", "glTFjunkjunkjunkjunk" },
        { "bad.gltf", "{\"asset\":{\"version\":\"2.0\"},\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":7}}]}]}" },
        { "x.stl", "solid x\n" },
        { "short.stl", "solid x\n facet normal 0 0 1\n outer loop\n vertex 0 0 0\n vertex 1 0 0\n endloop\n endfacet\n" },
        { "nan.stl", "solid x\n facet normal 0 0 1\n outer loop\n vertex 0 0 0\n vertex nan 0 0\n vertex 0 1 0\n endloop\n endfacet\n" },
    };
    for (const auto& b : bad) {
        const std::string p = tmpfile(b.first, b.second);
        Mesh m; std::string err;
        const bool ok = load_mesh(p, m, err);
        CHECK(!ok && !err.empty(), "malformed %s accepted", b.first);
        if (!ok) printf("malformed %-12s -> %s\n", b.first, err.c_str());
    }
    { Mesh m; std::string err; CHECK(!load_mesh("/nonexistent/x.glb", m, err), "missing file accepted"); }
    { Mesh m; m.verts = { 1, 1, 1, 1, 1, 1, 1, 1, 1 }; m.faces = { 0, 1, 2 }; MeshNorm nm;
      CHECK(!normalize_mesh_trellis(m, nm), "degenerate mesh normalised"); }

    // Orientation (--mesh-up / --mesh-yaw, docs/spec/34-studio-texture-mode.md). A marker mesh:
    // vertex 0 at an offset origin, then one vertex 1 unit along each signed file axis
    // (+x, -x, +y, -y, +z, -z), with uneven extents so the bounding box is not symmetric.
    {
        const float O[3] = { 3.0f, -2.0f, 7.5f };
        const float ext[3] = { 1.0f, 1.25f, 1.5f };
        std::string obj, stl = "solid m\n";
        float P[7][3];
        for (int a = 0; a < 3; ++a) P[0][a] = O[a];
        for (int k = 0; k < 6; ++k) {
            for (int a = 0; a < 3; ++a) P[1 + k][a] = O[a];
            P[1 + k][k / 2] += (k % 2 ? -1.0f : 1.0f) * ext[k / 2];
        }
        for (const auto& p : P) { char l[96]; snprintf(l, sizeof l, "v %g %g %g\n", p[0], p[1], p[2]); obj += l; }
        const int T[4][3] = { { 0, 1, 3 }, { 0, 3, 2 }, { 0, 5, 4 }, { 0, 6, 1 } };   // 0-based
        for (const auto& t : T) {
            char l[64]; snprintf(l, sizeof l, "f %d %d %d\n", t[0] + 1, t[1] + 1, t[2] + 1); obj += l;
            stl += "facet normal 0 0 0\nouter loop\n";
            for (int k = 0; k < 3; ++k) {
                char v[96]; snprintf(v, sizeof v, "vertex %g %g %g\n", P[t[k]][0], P[t[k]][1], P[t[k]][2]); stl += v;
            }
            stl += "endloop\nendfacet\n";
        }
        stl += "endsolid m\n";
        const char* names[] = { "+x", "-x", "+y", "-y", "+z", "-z" };
        // The glTF direction of a vertex relative to the origin vertex, as a signed axis name.
        auto dir_of = [&](const Mesh& m, int v) -> std::string {
            for (int a = 0; a < 3; ++a) {
                const float d = m.verts[3 * v + a] - m.verts[a];
                if (std::fabs(d) > 0.5f) return std::string(d > 0 ? "+" : "-") + "xyz"[a];
            }
            return "?";
        };
        // Index of the marker vertex at a signed file axis name, found by position.
        auto vert_at = [&](const Mesh& m, const std::string& axis) -> int {
            const int k = (int)(std::find(names, names + 6, axis) - names);
            for (int v = 0; v < m.V(); ++v) {
                bool eq = true;
                for (int a = 0; a < 3; ++a) eq = eq && m.verts[3 * v + a] == P[1 + k][a];
                if (eq) return v;
            }
            return -1;
        };
        const std::string up_names[] = { "auto", "+x", "-x", "+y", "-y", "+z", "-z" };
        for (const auto& file : { tmpfile("orient.obj", obj), tmpfile("orient.stl", stl) }) {
            const bool is_stl = file.size() > 4 && file.substr(file.size() - 4) == ".stl";
            Mesh raw; std::string err;
            // the raw file positions: load with the identity map (y up, no turn) unless STL ...
            CHECK(load_mesh(file, raw, err, MeshOrient{ MeshUp::Y, 0 }), "orient raw: %s", err.c_str());
            // ... the default output (grid axes after restore), which every orientation must reproduce
            Mesh def; MeshNorm dn;
            CHECK(load_mesh(file, def, err) && normalize_mesh_trellis(def, dn), "orient default: %s", err.c_str());
            restore_mesh_transform(def.verts, dn);
            for (const std::string& un : up_names) {
                MeshUp up; CHECK(parse_mesh_up(un, up), "parse %s", un.c_str());
                const std::string up_axis = un == "auto" ? (is_stl ? "+z" : "+y") : un;
                for (int yaw : { 0, 90, 180, 270 }) {
                    const MeshOrient o{ up, yaw };
                    const AxisMap L = mesh_load_map(file, o);
                    int det = 0;   // of the signed permutation: sign of the permutation times the signs
                    {
                        const int* a = L.axis;
                        const int par = (a[0] == 0 ? (a[1] == 1 ? 1 : -1) : a[0] == 1 ? (a[1] == 0 ? -1 : 1) : (a[1] == 0 ? 1 : -1));
                        det = par * L.sign[0] * L.sign[1] * L.sign[2];
                    }
                    CHECK(det == 1, "%s up %s yaw %d: not a proper rotation", file.c_str(), un.c_str(), yaw);
                    Mesh m;
                    CHECK(load_mesh(file, m, err, o), "%s up %s yaw %d: %s", file.c_str(), un.c_str(), yaw, err.c_str());
                    if (m.V() != raw.V()) continue;
                    // the up axis points up, the reported front axis faces glTF +z
                    CHECK(dir_of(m, vert_at(raw, up_axis)) == "+y", "%s up %s yaw %d: up goes to %s", file.c_str(),
                          un.c_str(), yaw, dir_of(m, vert_at(raw, up_axis)).c_str());
                    const std::string front = mesh_front_axis(file, o);
                    CHECK(dir_of(m, vert_at(raw, front)) == "+z", "%s up %s yaw %d: front %s goes to %s",
                          file.c_str(), un.c_str(), yaw, front.c_str(), dir_of(m, vert_at(raw, front)).c_str());
                    // the turn is counter-clockwise seen from above: what faced +x at yaw 0 faces -z at 90
                    if (yaw == 90) {
                        Mesh m0; load_mesh(file, m0, err, MeshOrient{ up, 0 });
                        int vx = -1;
                        for (int v = 1; v < m0.V(); ++v) if (dir_of(m0, v) == "+x") vx = v;
                        CHECK(vx >= 0 && dir_of(m, vx) == "-z", "%s up %s: yaw 90 is not counter-clockwise", file.c_str(), un.c_str());
                    }
                    // normalise, restore, undo: bit-identical to the default orientation's output
                    MeshNorm nm;
                    CHECK(normalize_mesh_trellis(m, nm), "normalize");
                    restore_mesh_transform(m.verts, nm);
                    const AxisMap undo = mesh_orient_undo_grid(file, o);
                    undo.apply(m.verts);
                    CHECK(m.verts == def.verts, "%s up %s yaw %d: output differs from the default orientation",
                          file.c_str(), un.c_str(), yaw);
                    const bool native = yaw == 0 && (un == "auto" || un == (is_stl ? "+z" : "+y"));
                    CHECK(undo.identity() == native, "%s up %s yaw %d: undo identity %d", file.c_str(), un.c_str(),
                          yaw, (int)undo.identity());
                }
            }
            CHECK(mesh_front_axis(file, MeshOrient{}) == (is_stl ? "-y" : "+z"), "%s: default front %s", file.c_str(),
                  mesh_front_axis(file, MeshOrient{}).c_str());
            Mesh bad;
            CHECK(!load_mesh(file, bad, err, MeshOrient{ MeshUp::Auto, 45 }), "yaw 45 accepted");
        }
        MeshUp up;
        CHECK(!parse_mesh_up("w", up) && !parse_mesh_up("", up) && !parse_mesh_up("--z", up), "bad mesh_up accepted");
    }

    printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL OK", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
