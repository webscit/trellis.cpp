// Mesh input for --refine-mesh / --texture-mesh: GLB / glTF (cgltf), OBJ and STL loading, and
// the TRELLIS.2 normalisation into the latent grid's frame (docs/spec/31-mesh-refine.md,
// docs/spec/33-texture-only.md).
#pragma once
#include <string>
#include "dual_grid.h"   // trellis::Mesh

namespace trellis {

// Up axis of the file. The loader returns glTF y-up vertices: a z-up file is rotated by
// (x, y, z) -> (x, z, -y). Auto means z for STL (CAD / slicer convention), y for the others.
enum class MeshUp { Auto, Y, Z };

// True when `path` would be read as STL (by extension).
bool is_stl_path(const std::string& path);

// Loads every triangle primitive of every node (world transforms applied) of a .glb / .gltf,
// the v/f records of an .obj (fan-triangulated polygons, negative indices), or the facets of
// a binary / ASCII .stl, merged into one mesh. The format is detected by the "glTF" magic,
// else by extension. STL facets are welded on exact position (STL stores each triangle
// separately) and faces left with a repeated vertex are dropped. Returns false with `err` on
// any failure; never throws on malformed input.
bool load_mesh(const std::string& path, Mesh& out, std::string& err, MeshUp up = MeshUp::Auto);

// The transform normalize_mesh_trellis applied: v_norm = R * (v - center) * scale, where R maps
// glTF (x, y, z) to the latent grid's (x, -z, y).
struct MeshNorm { float center[3] = {0, 0, 0}; float scale = 1.0f; };

// The reference pipeline's preprocess_mesh: bbox centre, scale = 0.99999 / max extent, then
// (x, y, z) -> (x, -z, y). The GLB writers rotate (x, y, z) -> (x, z, -y), the inverse, so a
// y-up input keeps its orientation on output. Returns false for an empty / degenerate mesh.
bool normalize_mesh_trellis(Mesh& m, MeshNorm& norm);

// Inverse of the scale/centre part, applied to vertices already rotated back to glTF axes by a
// writer's (x, y, z) -> (x, z, -y): v_out = v / scale + center. In-place on [V*3] floats that are
// still in grid axes, i.e. before the writer's rotation: grid (x, y, z) -> glTF (x, z, -y) means
// the glTF-frame centre is added as (cx, -cz, cy) in grid axes.
void restore_mesh_transform(std::vector<float>& verts_grid_axes, const MeshNorm& norm);

} // namespace trellis
