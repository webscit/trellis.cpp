// trellis-print — make an existing mesh 3D-printable. CPU only, no models needed.
//
//   trellis-print <in.glb|in.gltf|in.obj> <out.3mf> [--res N] [--faces N] [--size MM]
//                 [--base FRACTION] [--wall MM] [--band N] [--interior M] [--glb out.glb]
//
// Loads the mesh, rebuilds it as one closed solid (exterior flood fill + marching tetrahedra,
// docs/spec/32-print-ready.md), decimates without changing the topology, and writes a 3MF in
// millimetres, z-up, sitting on the build plate. Geometry only: the input's texture is not
// carried over (trellis-cli --watertight --3mf bakes one). The same pipeline runs inside
// trellis-cli; this tool is for meshes that already exist.
#include "mesh_glb.h"
#include "mesh_io.h"
#include "mesh_print.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace trellis;

static void usage() {
    fprintf(stderr,
        "usage: trellis-print <in.glb|in.gltf|in.obj> <out.3mf> [options]\n"
        "      --res N         remesh grid resolution, 64..1536 (default 512)\n"
        "      --faces N       triangle budget; 0 keeps the full remesh (default 300000)\n"
        "      --size MM       longest side of the print in millimetres (default 100)\n"
        "      --base F|auto   cut this fraction of the height off the bottom for a flat base,\n"
        "                      0..0.5 (default 0 = off); auto = the lowest cut the model stands\n"
        "                      on, else a pedestal under it\n"
        "      --wall MM       minimum wall thickness at --size (default 0.8): thinner fins and\n"
        "                      edges are grown, and joints between thick parts narrower than\n"
        "                      twice it are thickened. 0 = off\n"
        "      --band N        remesh band width in cells (default 1)\n"
        "      --interior M    auto (default): close tears, keep wide openings (cups, vases)\n"
        "                      open; solid: fill those too; hollow: close nothing\n"
        "      --glb PATH      also write the solid as an untextured GLB\n");
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 1; }
    const std::string input = argv[1], output = argv[2];
    PrintOptions options;
    float size_mm = 100.0f, wall_mm = 0.8f;
    int band = 0;
    std::string glb;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
        auto number = [&](float lo, float hi, float& target) {
            char* end = nullptr;
            const float n = v ? std::strtof(v, &end) : NAN;
            if (!v || end == v || *end || !(n >= lo && n <= hi)) {
                fprintf(stderr, "[trellis-print] %s expects a number in %g..%g\n", a.c_str(), lo, hi);
                return false;
            }
            target = n;
            ++i;
            return true;
        };
        float n = 0;
        if      (a == "--res")   { if (!number(64, 1536, n)) return 1; options.res = (int)n; }
        else if (a == "--faces") { if (!number(0, 2e9f, n)) return 1; options.faces = (int)n; }
        else if (a == "--size")  { if (!number(1e-3f, 1e6f, size_mm)) return 1; }
        else if (a == "--base" && v && std::string(v) == "auto") { options.base = -1.0f; ++i; }
        else if (a == "--base")  { if (!number(0, 0.5f, options.base)) return 1; }
        else if (a == "--wall")  { if (!number(0, 1e6f, wall_mm)) return 1; }
        else if (a == "--band")  { if (!number(1, 8, n)) return 1; band = (int)n; }
        else if (a == "--interior" && v && parse_print_interior(v, options.interior)) { ++i; }
        else if (a == "--glb" && v) { glb = v; ++i; }
        else { fprintf(stderr, "[trellis-print] unknown or incomplete option: %s\n", a.c_str()); usage(); return 1; }
    }

    const auto t0 = std::chrono::steady_clock::now();
    Mesh mesh;
    std::string error;
    if (!load_mesh(input, mesh, error)) { fprintf(stderr, "[trellis-print] %s\n", error.c_str()); return 1; }
    MeshNorm norm;
    if (!normalize_mesh_trellis(mesh, norm)) { fprintf(stderr, "[trellis-print] empty or degenerate mesh\n"); return 1; }
    {
        Mesh welded = mesh;
        weld_exact(welded.verts, welded.faces);
        printf("input   %s\n", audit_summary(audit_mesh(welded.verts.data(), welded.V(), welded.faces.data(), welded.F())).c_str());
    }
    // normalize_mesh_trellis scales the longest side to ~1, so the model extent is 1.
    options.band = band > 0 ? band : 1;
    options.wall_cells = wall_cells_for(wall_mm, size_mm, options.res, 1.0f);
    printf("solid remesh @res %d, band %d, minimum wall %.2f mm (%.1f cells), neck %.2f mm, interior %s, at %.0f mm\n",
           options.res, options.band, wall_mm, options.wall_cells, 2.0f * wall_mm, print_interior_name(options.interior), size_mm);
    fflush(stdout);

    Mesh solid = make_printable(mesh, options);
    if (solid.F() == 0) { fprintf(stderr, "[trellis-print] remesh produced no surface\n"); return 1; }
    const MeshAudit audit = audit_mesh(solid.verts.data(), solid.V(), solid.faces.data(), solid.F());
    printf("output  %s\n", audit_summary(audit).c_str());
    fputs(print_report_text(print_report(solid.verts.data(), solid.V(), solid.faces.data(), solid.F(), size_mm, 2, wall_mm)).c_str(), stdout);

    if (!write_3mf(output.c_str(), solid.verts.data(), solid.V(), solid.faces.data(), solid.F(), size_mm)) {
        fprintf(stderr, "[trellis-print] cannot write %s\n", output.c_str());
        return 1;
    }
    if (!glb.empty() && !write_glb(glb.c_str(), solid.verts.data(), solid.V(), solid.faces.data(), solid.F())) {
        fprintf(stderr, "[trellis-print] cannot write %s\n", glb.c_str());
        return 1;
    }
    const float extent = std::max(audit.bbox_max[0] - audit.bbox_min[0],
                         std::max(audit.bbox_max[1] - audit.bbox_min[1], audit.bbox_max[2] - audit.bbox_min[2]));
    const double mm = size_mm / extent;
    printf("wrote %s: %.1f x %.1f x %.1f mm, %.1f cm3, %lld triangles in %.1fs\n", output.c_str(),
           (audit.bbox_max[0] - audit.bbox_min[0]) * mm, (audit.bbox_max[1] - audit.bbox_min[1]) * mm,
           (audit.bbox_max[2] - audit.bbox_min[2]) * mm, audit.volume * mm * mm * mm / 1000.0,
           (long long)audit.faces,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return audit.watertight() ? 0 : 2;
}
