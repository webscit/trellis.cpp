// 3D-print readiness: watertight audit, last-resort repair, slicer placement and STL export.
// See docs/spec/32-print-ready.md. The watertight surface itself comes from the solid remesh
// (remesh_dc.h, RemeshOptions::solid) followed by a topology-preserving decimation; this
// module checks the result and writes it in the form slicers expect.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "dual_grid.h"   // trellis::Mesh

namespace trellis {

struct RemeshOptions;

class TriBvh;

// What a slicer needs to be true of a mesh. Edges are counted over the index buffer, so
// run it on a welded mesh (weld_exact for a UV-split GLB buffer).
struct MeshAudit {
    int64_t vertices = 0, faces = 0;
    int64_t boundary_edges = 0;      // used by one face: a hole
    int64_t nonmanifold_edges = 0;   // used by more than two faces
    int64_t flipped_edges = 0;       // used by two faces that disagree on orientation
    int64_t degenerate_faces = 0;    // a repeated vertex index
    int64_t nonmanifold_vertices = 0;// more than one face fan around the vertex (a pinch)
    int     components = 0;          // edge-connected shells
    int     inverted_components = 0; // shells with negative volume (cavities / inside-out)
    double  volume = 0;              // signed, mesh units^3 (meaningful when closed)
    double  area = 0;
    float   bbox_min[3] = {0, 0, 0}, bbox_max[3] = {0, 0, 0};

    // Closed, edge-manifold, consistently oriented, no cavity: every slicer accepts it.
    bool watertight() const {
        return faces > 0 && boundary_edges == 0 && nonmanifold_edges == 0 && flipped_edges == 0 &&
               degenerate_faces == 0 && inverted_components == 0;
    }
};

MeshAudit audit_mesh(const float* verts, int64_t V, const int32_t* faces, int64_t F);
// One line, e.g. "watertight: yes  V=.. F=.. shells=1 volume=..".
std::string audit_summary(const MeshAudit& a);

// Merge vertices with bit-identical positions (the UV unwrap duplicates vertices along chart
// seams). In place; returns the number merged.
int weld_exact(std::vector<float>& verts, std::vector<int32_t>& faces);

// Safety net behind the by-construction pipeline: drop degenerate and duplicate faces (an
// opposite-wound pair is a zero-thickness fin and goes entirely), unify the winding, close
// every remaining boundary loop, and turn each shell outward. A watertight input is returned
// unchanged. Returns the number of faces removed + added + flipped.
int64_t repair_for_print(std::vector<float>& verts, std::vector<int32_t>& faces);

// A wall of wall_mm on a model printed size_mm long, in remesh grid cells
// (RemeshOptions::min_wall_cells). model_extent is the model's longest side in mesh units.
float wall_cells_for(float wall_mm, float size_mm, int res, float model_extent);

// RemeshOptions::open_ball_cells for a grid of `res`: the radius of the ball that tells a
// real opening (the mouth of a cup) from a tear in the decoded surface, which is narrower.
float open_ball_cells_for(int res);

// What happens inside the model (--print-interior):
//   Auto   tears are capped, wide openings (cups, vases) stay open -- the default;
//   Solid  tears are capped and every deep concavity is filled, vessels included;
//   Hollow nothing is capped: only what the band closes by itself is solid.
enum class PrintInterior { Auto, Solid, Hollow };
bool parse_print_interior(const char* text, PrintInterior& out);
const char* print_interior_name(PrintInterior interior);
void set_print_interior(RemeshOptions& remesh, PrintInterior interior, int res);

// Minimum wall and minimum neck from the wall in cells: features thinner than the wall are
// grown, and joints between thick parts narrower than twice the wall are thickened.
void set_print_walls(RemeshOptions& remesh, float wall_cells);

// Complete geometry-only path for an arbitrary triangle mesh in the TRELLIS frame
// ([-0.5, 0.5]^3, as normalize_mesh_trellis leaves it): weld, solid remesh at `res`,
// topology-preserving QEM to `faces`, repair. Used by trellis-print and the tests; trellis-cli
// runs the same steps interleaved with its texture bake.
struct PrintOptions {
    int   res = 512;            // remesh grid resolution
    int   band = 1;             // remesh band
    int   faces = 300000;       // decimation target; <= 0 keeps the full remesh
    float base = 0.0f;          // fraction of the height cut off for a flat base; 0 = none;
                                //   < 0 = automatic: the lowest stable cut, else a pedestal
    float wall_cells = 0.0f;    // RemeshOptions::min_wall_cells (see wall_cells_for); 0 = off
    int   up_axis = 2;          // axis the base is perpendicular to (TRELLIS: z)
    PrintInterior interior = PrintInterior::Auto;   // see set_print_interior
};
Mesh make_printable(const Mesh& in, const PrintOptions& options);

// Fills RemeshOptions' base fields from a PrintOptions-style `base` value (see above):
// a fixed cut at that fraction of the height, or, when negative, the stability search
// between 1 % and 4 % of the height with a pedestal as the fallback. 0 leaves them off.
struct RemeshOptions;
void set_print_base(RemeshOptions& remesh, const float* verts, int64_t V, int up_axis, float base);

// What will matter on the printer, measured on the final mesh at its printed size.
struct PrintReport {
    MeshAudit audit;
    float  size_mm[3] = {0, 0, 0};   // footprint x, footprint y, height
    double volume_cm3 = 0, area_cm2 = 0;
    float  wall_target_mm = 0;       // the wall the model was asked to have (0.8 if unset)
    float  wall_p1_mm = 0;           // wall thickness under the thinnest 1 % of the surface
    bool   wall_capped = false;      // wall_p1_mm is a lower bound (thicker than measured range)
    double thin_area_pct = 0;        // surface share thinner than the target
    bool   flat_base = false;        // touches the plate on an area, not on points
    float  base_area_mm2 = 0;        // convex hull of the contact
    bool   stands = false;           // centre of mass over the contact hull
    float  tip_angle_deg = 0;        // tilt needed to topple it
    double overhang_pct = 0;         // surface share facing down steeper than 45 degrees
    std::vector<uint32_t> palette;   // filament colours (set_report_colors); empty = none
    std::vector<double>   palette_share;
    int    color_patches = 0;
    std::vector<std::string> warnings;
    bool ok() const { return warnings.empty(); }
};
// `verts` in any frame with `up_axis` pointing up; size_mm as for write_3mf; wall_mm is the
// target the thin-area share is measured against (<= 0: 0.8 mm).
PrintReport print_report(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                         float size_mm, int up_axis = 2, float wall_mm = 0.0f);
std::string print_report_text(const PrintReport& report);   // indented log block
std::string print_report_json(const PrintReport& report);   // one JSON object

// Base-cut plane for RemeshOptions: `fraction` of the way up the mesh's extent on `axis`.
float base_plane_of(const float* verts, int64_t V, int axis, float fraction);

// Optional base-colour texture for write_3mf: per-vertex UVs in the glTF convention (origin
// top-left) over the same vertex buffer, and a square RGBA image (uv_bake's BakedMesh).
struct PrintTexture {
    const float*         uv = nullptr;     // [V*2]
    const unsigned char* rgba = nullptr;   // [size*size*4]
    int                  size = 0;
    bool ok() const { return uv && rgba && size > 0; }
};

// Filament colours for multi-material printers (--print-colors): the texture reduced to a
// few colours, one per triangle. See docs/spec/32-print-ready.md §2.7.
struct ColorOptions {
    int   count = 0;          // palette size, 2..32; 0 = off
    float speck_mm2 = 4.0f;   // colour patches smaller than this merge into a neighbour
    float lift = 0.3f;        // shadow lift: 0 keeps the texture's tones; the levels are
                              //   stretched and a gamma of 1/(1+lift) lifts baked-in shading
};
struct PrintColors {
    std::vector<uint32_t> palette;   // 0xRRGGBB (sRGB); slot s+1 is palette[s], largest share first
    std::vector<double>   share;     // surface share of each slot, 0..1
    std::vector<uint8_t>  face_slot; // per input face, 0-based slot
    int patches = 0;                 // connected same-colour regions after the speck merge
    bool ok() const { return !palette.empty(); }
};
constexpr int kMaxPrintColors = 32;
// verts/faces as for write_3mf (a UV-split buffer is fine: adjacency is taken over welded
// positions); `size_mm` as for write_3mf, so speck_mm2 is on the printed part.
PrintColors quantize_print_colors(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                                  const PrintTexture& texture, float size_mm, const ColorOptions& options);
// Orca / Prusa / Bambu triangle-selector code for a whole triangle painted with filament
// `slot` (1-based): 1 -> "4", 2 -> "8", 3 -> "0C", 5 -> "2C", 20 -> "2FC".
std::string paint_color_code(int slot);
// Adds the palette (hex, share, patch count) to a report's text and JSON.
void set_report_colors(PrintReport& report, const PrintColors& colors);

// 3MF package (core spec + materials extension), laid out the way a slicer wants it:
// millimetres, z-up, the lowest point on z = 0, centred on the origin in x/y, uniformly
// scaled so the longest bounding-box side is size_mm (<= 0 keeps the mesh's own units).
// `verts` are in the TRELLIS frame (z-up).
//
// Vertices with identical positions are merged, so a UV-split buffer is written as the
// closed manifold it is; with a texture the UVs go per triangle corner into a
// texture2dgroup (PNG inside the package), which is how 3MF keeps a textured mesh
// watertight. Unlike STL the file carries its unit and its colour.
//
// With `colors` (one slot per face, from quantize_print_colors) the texture is replaced by
// filament colours: a basematerials group with a pid/p1 per triangle (the standard form),
// Orca's per-triangle paint_color (what Orca-family slicers read), and the filament colours
// in Metadata/project_settings.config so the slots open already coloured.
bool write_3mf(const char* path, const float* verts, int64_t V, const int32_t* faces, int64_t F,
               float size_mm = 100.0f, const PrintTexture* texture = nullptr,
               const PrintColors* colors = nullptr);

}  // namespace trellis
