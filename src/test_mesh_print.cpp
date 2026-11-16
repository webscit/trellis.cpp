// 3D-print readiness tests (docs/spec/32-print-ready.md). CPU only, no models.
//
//   trellis-test-mesh-print [mesh.glb]
//
// Synthetic cases: the audit's defect counts, the solid remesh against the shapes that break
// the plain one (an inner shell, internal geometry, a small hole, an open sheet), the flat
// base, the topology-preserving decimation, the repair pass and the 3MF package. With a mesh
// argument it also runs the full path on that file and requires a watertight result.
#include "mesh_io.h"
#include "mesh_print.h"
#include "remesh_dc.h"
#include "tri_bvh.h"
#include "uv_bake.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace trellis;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// Axis-aligned box, outward wound. `skip` drops one of the 6 sides (0 = -x .. 5 = +z).
static void add_box(Mesh& m, const float lo[3], const float hi[3], int skip = -1) {
    const int base = m.V();
    for (int i = 0; i < 8; ++i) {
        m.verts.push_back(i & 1 ? hi[0] : lo[0]);
        m.verts.push_back(i & 2 ? hi[1] : lo[1]);
        m.verts.push_back(i & 4 ? hi[2] : lo[2]);
    }
    static const int quads[6][4] = {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    for (int q = 0; q < 6; ++q) {
        if (q == skip) continue;
        const int* v = quads[q];
        for (int k : {0, 1, 2, 0, 2, 3}) m.faces.push_back(base + v[k]);
    }
}

static Mesh box(float half, int skip = -1) {
    Mesh m;
    const float lo[3] = {-half, -half, -half}, hi[3] = {half, half, half};
    add_box(m, lo, hi, skip);
    return m;
}

static MeshAudit audit(const Mesh& m) { return audit_mesh(m.verts.data(), m.V(), m.faces.data(), m.F()); }

static Mesh remesh(const Mesh& m, int res, const RemeshOptions& options) {
    const TriBvh bvh = TriBvh::build(m.verts.data(), m.V(), m.faces.data(), m.F());
    return remesh_narrow_band_dc(m.verts.data(), m.V(), m.faces.data(), m.F(), bvh, res, options);
}

int main(int argc, char** argv) {
    const int res = 64;
    const float cell = (res + 3.0f) / res / res;   // band 1
    RemeshOptions solid;
    solid.solid = true;

    {   // audit: a closed box is watertight; without its lid it has a 4-edge hole.
        const MeshAudit closed = audit(box(0.3f));
        CHECK(closed.watertight() && closed.components == 1, "closed box: %s", audit_summary(closed).c_str());
        CHECK(std::fabs(closed.volume - 0.216) < 1e-5 && std::fabs(closed.area - 2.16) < 1e-5, "box volume %g area %g", closed.volume, closed.area);
        const MeshAudit open = audit(box(0.3f, 5));
        CHECK(!open.watertight() && open.boundary_edges == 4, "open box boundary %lld", (long long)open.boundary_edges);
        Mesh inside_out = box(0.3f);
        for (size_t f = 0; f < inside_out.faces.size(); f += 3) std::swap(inside_out.faces[f + 1], inside_out.faces[f + 2]);
        CHECK(audit(inside_out).inverted_components == 1 && !audit(inside_out).watertight(), "inside-out box not flagged");
        Mesh pinch = box(0.1f);   // two boxes sharing one corner vertex
        const float lo[3] = {0.1f, 0.1f, 0.1f}, hi[3] = {0.3f, 0.3f, 0.3f};
        add_box(pinch, lo, hi);
        weld_exact(pinch.verts, pinch.faces);
        CHECK(audit(pinch).nonmanifold_vertices == 1, "pinch vertices %lld", (long long)audit(pinch).nonmanifold_vertices);
    }

    {   // The plain remesh of a closed box is two shells (outer skin + inner cavity wall);
        // the solid remesh is one, and its volume is the box grown by eps.
        const Mesh b = box(0.3f);
        RemeshOptions plain;
        Mesh shell = remesh(b, res, plain);
        clean_mesh(shell.V(), shell.faces);
        CHECK(audit(shell).components == 2, "plain remesh shells %d", audit(shell).components);
        const Mesh s = remesh(b, res, solid);
        const MeshAudit a = audit(s);
        CHECK(a.watertight() && a.components == 1 && a.nonmanifold_vertices == 0, "solid box: %s", audit_summary(a).c_str());
        const double grown = std::pow(0.6 + 2.0 * cell, 3.0);   // upper bound: rounded edges cut a little
        CHECK(a.volume > 0.216 && a.volume < grown * 1.001, "solid box volume %g (box 0.216, grown %g)", a.volume, grown);
        CHECK(a.volume > 0.9 * grown, "solid box volume %g is hollow (grown %g)", a.volume, grown);
    }

    {   // Internal geometry and a hole narrower than 2*eps both vanish into the solid.
        Mesh m = box(0.3f);
        const float lo[3] = {-0.1f, -0.1f, -0.1f}, hi[3] = {0.1f, 0.1f, 0.1f};
        add_box(m, lo, hi);   // a box floating inside
        const float tiny = 0.6f * cell;   // pinhole in the lid: a missing sliver quad
        // rebuild the lid as a frame around a tiny square hole
        Mesh holed;
        add_box(holed, std::array<float, 3>{-0.3f, -0.3f, -0.3f}.data(), std::array<float, 3>{0.3f, 0.3f, 0.3f}.data(), 5);
        const float z = 0.3f, o = 0.3f;
        const float ring[8][2] = {{-o, -o}, {o, -o}, {o, o}, {-o, o}, {-tiny, -tiny}, {tiny, -tiny}, {tiny, tiny}, {-tiny, tiny}};
        const int base = holed.V();
        for (const auto& p : ring) { holed.verts.push_back(p[0]); holed.verts.push_back(p[1]); holed.verts.push_back(z); }
        for (int i = 0; i < 4; ++i) {
            const int j = (i + 1) % 4;
            for (int k : {i, j, 4 + j, i, 4 + j, 4 + i}) holed.faces.push_back(base + k);
        }
        add_box(holed, lo, hi);
        weld_exact(holed.verts, holed.faces);
        CHECK(audit(holed).boundary_edges == 4, "pinhole fixture boundary %lld", (long long)audit(holed).boundary_edges);
        for (const Mesh* in : {&m, &holed}) {
            const MeshAudit a = audit(remesh(*in, res, solid));
            CHECK(a.watertight() && a.components == 1, "internal/pinhole: %s", audit_summary(a).c_str());
            CHECK(a.volume > 0.216, "internal/pinhole volume %g: the cavity was not filled", a.volume);
        }
    }

    {   // A box without its lid is a real opening: the flood enters and the result is a closed
        // thin-walled cup, about 2*eps thick.
        const MeshAudit a = audit(remesh(box(0.3f, 5), res, solid));
        CHECK(a.watertight() && a.components == 1, "open cup: %s", audit_summary(a).c_str());
        const double wall = 5 * 0.36 * 2.0 * cell;
        CHECK(a.volume > 0.5 * wall && a.volume < 2.0 * wall, "cup volume %g, expected about %g", a.volume, wall);
    }

    {   // A torn surface: a third of the lid is missing. The flood alone walks in (a cup);
        // the openness vote caps the tear and the box is solid again. A shallow tray, whose
        // floor sees the sky in every upward direction, must stay a tray.
        Mesh torn;
        add_box(torn, std::array<float, 3>{-0.3f, -0.3f, -0.3f}.data(), std::array<float, 3>{0.3f, 0.3f, 0.3f}.data(), 5);
        const int base = torn.V();   // lid with a 0.2 x 0.2 hole in the middle
        const float o = 0.3f, h = 0.1f;
        const float ring[8][2] = {{-o, -o}, {o, -o}, {o, o}, {-o, o}, {-h, -h}, {h, -h}, {h, h}, {-h, h}};
        for (const auto& p : ring) { torn.verts.push_back(p[0]); torn.verts.push_back(p[1]); torn.verts.push_back(0.3f); }
        for (int i = 0; i < 4; ++i) {
            const int j = (i + 1) % 4;
            for (int k : {i, j, 4 + j, i, 4 + j, 4 + i}) torn.faces.push_back(base + k);
        }
        weld_exact(torn.verts, torn.faces);
        RemeshOptions sealing = solid;
        sealing.seal = 3;
        const MeshAudit leaky = audit(remesh(torn, res, solid));
        const MeshAudit capped = audit(remesh(torn, res, sealing));
        CHECK(leaky.watertight() && leaky.volume < 0.08, "flood-only torn box: %s", audit_summary(leaky).c_str());
        CHECK(capped.watertight() && capped.components == 1 && capped.volume > 0.9 * 0.216,
              "sealed torn box: %s", audit_summary(capped).c_str());

        Mesh tray;
        add_box(tray, std::array<float, 3>{-0.3f, -0.3f, -0.05f}.data(), std::array<float, 3>{0.3f, 0.3f, 0.05f}.data(), 5);
        const MeshAudit open_tray = audit(remesh(tray, res, sealing));
        const double slab = 0.6 * 0.6 * 0.1;
        CHECK(open_tray.watertight() && open_tray.volume < 0.6 * slab, "tray was filled: volume %g of slab %g", open_tray.volume, slab);
    }

    // Flat base: the bottom 25 % goes, the cut is one exact plane, the solid stays closed.
    // On every axis: TRELLIS grids are z-up, Pixal3D's are y-up until the final rotation.
    for (int axis = 0; axis < 3; ++axis) {
        RemeshOptions cut = solid;
        cut.base_axis = axis;
        cut.base_plane = -0.3f + 0.25f * 0.6f;
        const Mesh s = remesh(box(0.3f), res, cut);
        const MeshAudit a = audit(s);
        CHECK(a.watertight() && a.components == 1, "flat base, axis %d: %s", axis, audit_summary(a).c_str());
        CHECK(std::fabs(a.bbox_min[axis] - cut.base_plane) < cell, "axis %d: base at %g, plane %g", axis, a.bbox_min[axis], cut.base_plane);
        for (int other = 0; other < 3; ++other)
            if (other != axis) CHECK(a.bbox_min[other] < -0.3f, "axis %d: side %d was cut too", axis, other);
        int on_plane = 0;
        for (int v = 0; v < s.V(); ++v) on_plane += s.verts[3 * v + axis] == a.bbox_min[axis];
        CHECK(on_plane > 1000, "axis %d: only %d vertices on the base plane", axis, on_plane);
        const double expect = std::pow(0.6 + 2.0 * cell, 2.0) * (0.3 + cell - a.bbox_min[axis]);
        CHECK(a.volume > 0.95 * expect && a.volume < 1.001 * expect, "axis %d: cut volume %g, expected %g", axis, a.volume, expect);
    }

    {   // The whole path, decimated hard: still one closed manifold.
        PrintOptions options;
        options.res = res;
        options.faces = 2000;
        Mesh m = box(0.3f);
        const float lo[3] = {-0.05f, -0.05f, 0.3f}, hi[3] = {0.05f, 0.05f, 0.45f};
        add_box(m, lo, hi);   // a post on top, overlapping surfaces
        const Mesh out = make_printable(m, options);
        const MeshAudit a = audit(out);
        CHECK(a.watertight() && a.components == 1 && a.nonmanifold_vertices == 0, "make_printable: %s", audit_summary(a).c_str());
        CHECK(out.F() <= 2400 && out.F() > 100, "decimated to %d faces", out.F());

        // 3MF: a ZIP whose model part lists every vertex and triangle; textured adds a PNG.
        const auto dir = std::filesystem::temp_directory_path();
        const std::string plain = (dir / "trellis-test-print.3mf").string();
        CHECK(write_3mf(plain.c_str(), out.verts.data(), out.V(), out.faces.data(), out.F(), 80.0f), "write_3mf");
        std::ifstream file(plain, std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        CHECK(bytes.size() > 200 && bytes.compare(0, 4, "PK\x03\x04") == 0, "3mf is not a zip (%zu bytes)", bytes.size());
        CHECK(bytes.find("3D/3dmodel.model") != std::string::npos && bytes.find("[Content_Types].xml") != std::string::npos, "3mf parts missing");
        CHECK(bytes.compare(bytes.size() - 22, 4, "PK\x05\x06") == 0, "3mf end-of-central-directory missing");

        std::vector<float> uv((size_t)out.V() * 2, 0.5f);
        const std::vector<unsigned char> rgba(8 * 8 * 4, 200);
        const PrintTexture texture{uv.data(), rgba.data(), 8};
        const std::string coloured = (dir / "trellis-test-print-tex.3mf").string();
        CHECK(write_3mf(coloured.c_str(), out.verts.data(), out.V(), out.faces.data(), out.F(), 80.0f, &texture), "write_3mf textured");
        std::ifstream tfile(coloured, std::ios::binary);
        const std::string tbytes((std::istreambuf_iterator<char>(tfile)), std::istreambuf_iterator<char>());
        CHECK(tbytes.find("3D/Textures/base_color.png") != std::string::npos, "3mf texture part missing");
        std::filesystem::remove(plain);
        std::filesystem::remove(coloured);
    }

    {   // Repair: a missing face, a zero-thickness fin and an inside-out winding.
        Mesh m = box(0.3f);
        m.faces.resize(m.faces.size() - 3);                                   // open a triangle
        for (int k : {0, 1, 2, 0, 2, 1}) m.faces.push_back(m.faces[(size_t)k]);   // fin: same face twice, opposite
        for (size_t f = 0; f < m.faces.size(); f += 3) std::swap(m.faces[f + 1], m.faces[f + 2]);
        CHECK(!audit(m).watertight(), "broken fixture is watertight");
        CHECK(repair_for_print(m.verts, m.faces) > 0, "repair changed nothing");
        const MeshAudit a = audit(m);
        CHECK(a.watertight() && std::fabs(a.volume - 0.216) < 1e-5, "repaired: %s", audit_summary(a).c_str());
        CHECK(repair_for_print(m.verts, m.faces) == 0, "repair is not idempotent");
    }

    CHECK(band_for_wall(0, 100, 512, 1) == 1 && band_for_wall(0.39f, 100, 512, 1) == 1 &&
          band_for_wall(0.8f, 100, 512, 1) == 3 && band_for_wall(50, 100, 512, 1) == 8, "band_for_wall");

    if (argc > 1) {
        Mesh m;
        std::string error;
        MeshNorm norm;
        CHECK(load_mesh(argv[1], m, error) && normalize_mesh_trellis(m, norm), "load %s: %s", argv[1], error.c_str());
        PrintOptions options;
        options.res = 256;
        options.faces = 50000;
        options.base = 0.02f;
        const MeshAudit a = audit(make_printable(m, options));
        printf("%s -> %s\n", argv[1], audit_summary(a).c_str());
        CHECK(a.watertight(), "%s is not watertight after make_printable", argv[1]);
    }

    if (g_fail) { fprintf(stderr, "%d check(s) failed\n", g_fail); return 1; }
    printf("mesh_print: all checks passed\n");
    return 0;
}
