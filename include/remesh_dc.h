// Narrow-band UDF dual-contouring remesh, ported from the reference
// (CuMesh remeshing.py::remesh_narrow_band_dc + simple_dual_contour.cu; see
// docs/spec/27-reference-postprocess.md §4). Rebuilds a noisy voxel-derived
// mesh as the closed, consistently-oriented offset surface at distance
// eps = band·scale/res around it — one watertight manifold that downstream
// simplification and UV unwrapping can digest.
#pragma once
#include "dual_grid.h"

namespace trellis {

class TriBvh;

// `bvh` must be built over (verts, faces). Returns an empty mesh on failure
// (caller keeps the un-remeshed path).
// `project_back` lerps each dual vertex toward its closest point on the input
// surface (o_voxel postprocess `remesh_project`; 0 disables).
//
// Default 0 to match the HF space — the config that produced the reference GLBs
// users compare against — which passes remesh_project=0 (trellis2-space/app.py).
// o_voxel's own to_glb default is 0.9 and TRELLIS.2's app.py inherits it, so the
// knob is kept: measured on the issue #1 goblin, 0.9 improves the surface
// (dihedral 8.08->7.41deg, >60deg spikes 1.12%->0.64%) but quadruples our UV
// merge clusters (2519->10248), so it is NOT a free win in this pipeline.
Mesh remesh_narrow_band_dc(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                           const TriBvh& bvh, int res, int band = 1,
                           float project_back = 0.0f);

// Solid (watertight) variant for 3D printing — docs/spec/32-print-ready.md.
//
// The plain remesh contours |UDF| = eps, which around a closed input is TWO shells: the
// outer skin and a mirror-image inner one eps inside it. A slicer reads that as a hollow
// object with a 2*eps wall (0.4 % of the model at res 512). `solid` turns the unsigned
// field into a signed one before contouring: grid vertices farther than eps from the
// surface are flood-filled from the domain boundary, and whatever the flood cannot reach
// (the band itself, the cavity behind it, any internal geometry) is solid. The band is
// >= 2 cells thick, so the 6-connected flood cannot leak through a closed surface, and
// holes narrower than 2*eps are sealed for free. The surface is then extracted by
// marching tetrahedra on the Kuhn subdivision, which is a closed, consistently oriented
// 2-manifold by construction (no ambiguous cases, unlike dual contouring) with outward
// normals. `project_back` is ignored.
struct RemeshOptions {
    int   band = 1;
    float project_back = 0.0f;
    bool  solid = false;
    // Solid only: seal tears wider than the band. A vertex with fewer than `seal` of its 26
    // lattice rays leaving the grid unobstructed counts as interior even if the flood could
    // reach it (see remesh_dc.cpp). 0 = flood only; higher fills deeper concavities.
    int   seal = 0;
    // Flat base (solid only): everything below `base_plane` (input-mesh coordinate along
    // `base_axis`, 0/1/2) is cut away and capped with an exactly planar face, so the print
    // has a bed-contact surface. The plane snaps to the nearest half grid layer. -1 = off.
    int   base_axis = -1;
    float base_plane = 0.0f;
};
Mesh remesh_narrow_band_dc(const float* verts, int64_t V, const int32_t* faces, int64_t F,
                           const TriBvh& bvh, int res, const RemeshOptions& options);

}  // namespace trellis
