# 32 — Print-ready output (`--watertight`, `--3mf`, `trellis-print`)

How a generated asset becomes something a slicer accepts: one closed, manifold,
outward-oriented solid, at a real size, sitting on the build plate.

## 1. What was wrong with the default GLB

The default postprocess is a faithful port of the reference `to_glb` (spec 27). It is
built for rendering, and three of its properties defeat a slicer. Measured on the goblin
at res 512 (`trellis-print` prints this audit for any input):

| | default GLB | `--watertight` |
|---|---|---|
| boundary edges (holes) | 0 | 0 |
| non-manifold edges | 639 | 0 |
| pinched vertices | 234 | 0 |
| enclosed volume (unit cube) | 0.0102 | 0.0701 |
| surface area | 4.95 | 2.31 |

1. **It is hollow.** The narrow-band remesh contours `|UDF| = eps`. Around a closed
   surface that is two shells: the outer skin and a mirror image `eps` inside it. A
   slicer sees a shell with a `2·eps` wall — 0.39 mm on a 100 mm print at res 512 — and
   the volume above is that wall, not the object.
2. **It is not manifold.** Dual contouring puts one vertex in each cell, so two sheets
   passing through the same cell are pinched together; the QEM collapse then adds fins,
   and the post-decimation weld + fan fill adds more.
3. **The surface underneath is torn.** The decoded mesh (before the remesh) has 27,755
   boundary edges in 6,384 loops on this model; 30 of the holes are more than 16 cells
   across and the largest about 50. The double shell hides them — its inner and outer
   skins simply join around each rim — but any inside/outside test has to survive them.
4. **It has no size.** The mesh lives in a unit cube, y-up; a slicer reads that as a
   1 mm object lying on its back.

## 2. Strategy

Repairing a mesh after the fact (close holes, split non-manifold edges, MeshFix-style
delete-and-refill) is fragile, and position-based formats undo vertex splits anyway.
Instead every stage is made to produce a closed manifold **by construction**, and the
result is audited rather than trusted.

### 2.1 Signed field from an exterior flood fill (`RemeshOptions::solid`)

The unsigned distance is already sampled on the grid. Vertices farther than `eps` from
the surface are free; the flood starts at the domain boundary and spreads through free
vertices, 6-connected. Whatever it cannot reach is solid: the band, the cavity behind
it, any internal geometry. This is the same idea as ManifoldPlus / voxel-remesh
"make solid", applied to the field the remesh computes anyway.

- **No leaks through a closed surface.** A unit grid step changes the distance by at
  most one cell; crossing the surface means passing a point at distance 0, so one end
  of the step is closer than half a cell, which is inside the band (`eps >= 1` cell).
- **Small holes seal themselves.** A hole narrower than `2·eps` is covered by the band.
- Cost: two bitsets of `(res+1)^3` bits and a scanline flood.

The flood alone is not enough: it walks in through every tear wider than the band, and
the goblin comes out as a closed but hollow shell (volume 0.0100). A closing radius was
tried and rejected — sealing a 50-cell tear needs a radius that also fills the gap
between two legs.

### 2.1b Openness vote (`RemeshOptions::seal`, `--print-seal N`, default 3)

A point deep inside a torn shell sees surface in nearly every direction; a point in a
real concavity sees a cone of open space. On a coarse copy of the wall (step `res/256`)
each vertex counts the rays, out of the 26 axis and diagonal lattice directions, that
leave the grid without meeting the wall — 26 dynamic-programming sweeps, each vertex
taking its predecessor's answer. Vertices with fewer than `seal` open rays are shut to
the flood. Under a tear only the ray through it is open, so the tear is capped a little
below the surface; isolated open vertices inside do not matter because the flood still
has to reach them from outside.

The solid's boundary now also runs where the narrow band never was (across the caps,
along the base cut), so after classification every cell with both solid and exterior
corners is activated; vertices the band did not sample are placed half a cell from the
boundary. Caps therefore come out stair-stepped at the vote's grid (2 cells at res 512,
4 at res 1024) and are smoothed afterwards (§2.2b).

Volume against the threshold on the goblin: 0 → 0.0100 (hollow), 2 → 0.0694,
3 → 0.0702, 5 → 0.0727. The plateau is the filled figure; the slow rise is concave
detail being bridged.

**Limit.** 26 rays cannot tell a deep vase from a torn figurine: below roughly one
opening-radius of depth, a container's interior sees only the vertical ray and is
filled. A shallow tray stays open (tested). Use `--print-seal 0` for hollow objects —
then tears are not capped either, and the part is a thin shell wherever the flood got in.

### 2.2 Manifold extraction

Dual contouring cannot be used on the signed field for the reason in §1.2. The solid is
extracted as marching cubes *by cycles*: on every cube face the crossing edges are
linked in pairs (an ambiguous face always joins the inside corners, so both cells that
share the face agree), the links close into cycles, and each cycle is one polygon over
vertices that sit on grid edges. A polygon side lies in a cube face and is used by
exactly the two cells either side of it, so the surface is a closed 2-manifold with
outward normals. Polygons of five or more sides get a centre vertex so no diagonal can
coincide with a neighbour's. Crossings are clamped 5 % away from grid vertices (no
zero-area faces). The extraction is audited, and marching tetrahedra on the Kuhn
subdivision (`TRELLIS_SOLID_TETS=1` forces it; three times the triangles) is the
fallback.

### 2.2b Cap smoothing (`RemeshOptions::smooth_caps`, on by default)

A crossing is a *cap vertex* when one end of its grid edge has an invented field value
(re-signed by the fill, or outside the narrow band). After extraction those vertices are
relaxed as a membrane: umbrella averaging, `6·step²` passes (24 at res 512), with every
measured vertex and the base plane pinned. Connectivity is untouched, so the mesh is
the same closed manifold. On the goblin at res 512 (undecimated, 155 K cap vertices of
1.09 M): mean dihedral 6.67° → 4.38°, edges over 30° 6.75 % → 2.27 %, area 2.50 → 2.38,
volume 0.0743 → 0.0742. `post-replay --no-cap-smooth` gives the unsmoothed surface.

### 2.3 Flat base (`--print-base F`)

The lowest fraction `F` of the height is cut away and capped. The cut happens in the
field, after the flood — cutting first would let the flood in through the cut and hollow
the model. Vertices below the plane become exterior, the cells along the cut are
activated with the caps (§2.1b), and every edge through the plane is crossed at exactly
mid-layer, so the cap is one exact plane.

### 2.4 Topology-preserving decimation

`decimate_qem(..., keep_topology = true)` rejects every collapse that fails the link
condition (the edge has two faces and its endpoints share exactly two neighbours), so a
closed manifold stays one. The check only exists on the CPU path; the cost loop is
threaded. The post-decimation weld and fan fill are skipped — they are what pinched the
default mesh.

### 2.5 Audit and repair

`audit_mesh` counts boundary, non-manifold and inconsistently wound edges, degenerate
faces, pinched vertices, shells, and shells with negative volume; `watertight()` is all
of them at zero. `trellis-cli` prints it for the decimated mesh and again for the
vertex buffer that went into the GLB (welded by position: the UV unwrap splits vertices
along chart seams). `repair_for_print` is the safety net behind the construction
(duplicate/fin removal, winding, hole fill, outward orientation); on a watertight mesh
it does nothing.

## 3. Output formats

- **GLB** — with `--watertight` the GLB *is* the solid: same texture bake, closed
  geometry. Blender and Cura (with its plugin) read it; it has no unit.
- **3MF** (`--3mf PATH`) — the slicer format. Millimetres, z-up, lowest point on z = 0,
  centred, longest side `--print-size` (default 100). Vertices are shared by index, so
  the UV-seam duplicates are merged back and the solid is closed *in the file*; the
  base-colour texture goes in as a `texture2d` + `texture2dgroup` (materials extension)
  with UVs per triangle corner, PNG inside the package. STL was considered and dropped:
  no unit, no colour, no shared vertices.
  Slicer support for the texture varies — the geometry always loads; full-colour
  workflows read `texture2dgroup`, most FDM slicers ignore it and use their own paint data.

## 4. Interfaces

| CLI | server field | |
|---|---|---|
| `--watertight` | `watertight=on` | solid remesh + topology-preserving QEM + audit |
| `--3mf PATH` | `3mf=on`, then `GET /jobs/{id}/3mf` | also write the 3MF (implies watertight) |
| `--print-size MM` | `print_size` | longest side of the 3MF, default 100 |
| `--print-base F` | `print_base` | flat base, fraction of height, 0..0.5 (implies watertight) |
| `--print-seal N` | `print_seal` | openness threshold for capping tears, 0..26, default 3 (0 = flood only) |
| `--print-wall MM` | `print_wall` | minimum wall: band = ceil(wall / size · extent · res / 2), capped at 8 |

`--print-wall` works by widening the remesh band, which also grows the whole model by
`eps` and rounds detail of that size; an explicit `--band` wins. Not combinable with
`--decim GRID` (cluster decimation tears the surface) or `--retopo`.

`trellis-print in.glb out.3mf [--res --faces --size --base --wall --band --glb]` runs
the geometry path on any existing GLB / glTF / OBJ, CPU only, no models. It exits 2 if
the result is not watertight.

## 5. Cost and limits

- Goblin, res 512, Jetson Thor: solid remesh 5 s (2.0 M faces), decimation to 300 K
  about 30 s on the CPU. The default path decimates on the GPU, so `--watertight` is
  slower. At res 1024 (band 2) the remesh yields 8.4 M faces and the whole run, neural
  stages included, took 1211 s; the result audits watertight with the same volume
  (0.074) as at res 512.
- Also run end to end, all auditing watertight with a textured 3MF: `--refine-mesh` at
  res 512 (158 s) and `--model pixal3d` at its 1024 cascade (1020 s; the base is cut on
  the grid's y axis there, and the 3MF still comes out upright).
- Before cap smoothing existed, the caps showed as rough patches at res 1024 (the vote
  runs on a 4-cell grid there). Smoothing is measured at res 512 only so far.
- Self-intersections are not checked. The link condition and the flip test keep them
  rare and slicers tolerate them, but the mesh is not guaranteed intersection-free
  (`TRELLIS_QEM_COLLISION_GUARD` exists and is slow).
- Floating parts: components under 2 % of the largest are dropped; anything larger that
  is not attached to the body is kept and reported as `shells > 1`.
- Overhangs, supports and orientation are left to the slicer.
- Not done: finer openness sampling, wall thickening of thin parts only (needs a medial-axis estimate),
  hollowing with drain holes, per-vertex colour 3MF for slicers that ignore textures.

## 6. Tests

`trellis-test-mesh-print [mesh.glb]` — audit counts on fixtures (open box, inside-out,
pinch); solid remesh of a closed box (plain remesh gives two shells, solid one, volume
within 10 % of the grown box), of a box with internal geometry and with a pinhole, of an
open cup; the openness vote (a box with a third of its lid torn out is hollow with the
flood alone and solid with `seal = 3`; a shallow tray stays a tray); the flat base (exact plane, expected volume); `make_printable` decimated to
2 K faces; repair of a hole + fin + inverted winding; the 3MF package; `band_for_wall`.
With a mesh argument it runs the full path on it and requires a watertight result.
