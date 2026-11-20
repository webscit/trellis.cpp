// Mesh input for --refine-mesh / --texture-mesh: GLB / glTF (cgltf), OBJ and STL loading, and
// the TRELLIS.2 normalisation into the latent grid's frame (docs/spec/31-mesh-refine.md,
// docs/spec/33-texture-only.md).
#pragma once
#include <string>
#include <vector>
#include "dual_grid.h"   // trellis::Mesh

namespace trellis {

// The file axis that points up. The loader returns glTF y-up vertices: a z-up file is rotated
// by (x, y, z) -> (x, z, -y). Auto means z for STL (CAD / slicer convention), y for the others.
enum class MeshUp { Auto, Y, Z, X, NX, NY, NZ };

// "auto", "x", "-x", "y", "-y", "z", "-z" ("+x" etc. too). False for anything else.
bool parse_mesh_up(const std::string& s, MeshUp& out);

// How the input mesh is turned before painting (docs/spec/34-studio-texture-mode.md): the file
// axis `up` points up, then a turn of `yaw` degrees (0, 90, 180, 270) about the up axis,
// counter-clockwise seen from above. The canonical front (the side one image is painted on) is
// glTF +z, i.e. grid -y.
struct MeshOrient { MeshUp up = MeshUp::Auto; int yaw = 0; };

// A signed axis permutation: out[i] = sign[i] * in[axis[i]]. Every map built here is a proper
// rotation (det +1), so it keeps the triangle winding, and it is exact on floats.
struct AxisMap {
    int axis[3] = {0, 1, 2};
    int sign[3] = {1, 1, 1};
    bool identity() const;
    AxisMap inverse() const;
    AxisMap after(const AxisMap& first) const;   // this o first
    void apply(float* xyz, size_t count, size_t stride = 3) const;   // `count` vectors
    void apply(std::vector<float>& xyz) const { apply(xyz.data(), xyz.size() / 3); }
};

// The rotation load_mesh applies to `path`'s vertices (file -> glTF y-up) for `o`.
AxisMap mesh_load_map(const std::string& path, MeshOrient o);

// Undoes the turn `o` adds to the file's own convention, as a map of GRID-axis vertices (the
// frame restore_mesh_transform works in): applied after it, the output keeps the file's
// orientation (a z-up STL still comes back y-up in a GLB, as without `o`). Identity for the
// default orientation.
AxisMap mesh_orient_undo_grid(const std::string& path, MeshOrient o);

// The file axis that faces the canonical front after `o`, e.g. "-y" (for the log).
std::string mesh_front_axis(const std::string& path, MeshOrient o);

// True when `path` would be read as STL (by extension).
bool is_stl_path(const std::string& path);

// Loads every triangle primitive of every node (world transforms applied) of a .glb / .gltf,
// the v/f records of an .obj (fan-triangulated polygons, negative indices), or the facets of
// a binary / ASCII .stl, merged into one mesh. The format is detected by the "glTF" magic,
// else by extension. STL facets are welded on exact position (STL stores each triangle
// separately) and faces left with a repeated vertex are dropped. Returns false with `err` on
// any failure; never throws on malformed input.
bool load_mesh(const std::string& path, Mesh& out, std::string& err, MeshOrient orient = {});
inline bool load_mesh(const std::string& path, Mesh& out, std::string& err, MeshUp up) {
    return load_mesh(path, out, err, MeshOrient{up, 0});
}

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
