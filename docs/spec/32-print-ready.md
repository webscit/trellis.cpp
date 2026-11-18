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

### 2.1b Openness vote (`RemeshOptions::seal`, default 3)

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

26 rays cannot tell a deep vase from a torn figurine: below roughly one opening-radius
of depth, a container's interior sees only the vertical ray and is shut. Finer angular
sampling would not help much — the floor of a deep cup and the inside of a figurine behind
a wide tear subtend similar solid angles. §2.1c separates them by the width of the opening.

### 2.1c Open ball (`RemeshOptions::open_ball_cells`, `--print-interior auto`)

What tells a cup from a torn figurine is how wide the opening is. Free space that a ball
of radius `R` can reach from outside without touching the wall is exterior: a mouth wider
than `2R` lets the ball in, a tear narrower than that does not. On the vote's coarse grid,
padded by `R` so a ball fits above a model that fills the grid: an exact distance
transform of the wall, a flood from the padding through the voxels at least `R` from the
wall (the ball's possible centres), a second distance transform from what the flood
reached (the space the ball sweeps), and every shut voxel within `R` of it is released.
Releasing is all it does: it never fills anything, so the gap between two legs and other
concavities the vote already leaves open are untouched.

`R = res / 12` (43 cells at res 512): an opening must be about a sixth of the grid wide to
count. On the goblin's replay at res 512 a ball of radius 16 (32 cells across) already
does not reach the interior, although its widest tears measure 40–50 cells: the band and
the coarse wall (a coarse voxel is wall if any fine vertex in it is) narrow the effective
opening. At 43 cells the result is identical to no ball (volume 0.07547, nothing
released). Inner corners of a vessel keep a fillet of radius `R`
the ball cannot reach. A vase whose neck is narrower than `2R` is still filled.

`--print-interior` picks the behaviour: `auto` (default) = vote + open ball; `solid` =
vote only, every deep concavity filled, vessels included; `hollow` = no vote, only what
the band closes by itself is solid (tears are not capped either, and the part is a thin
shell wherever the flood got in).

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

### 2.3b Stable base (`--print-base auto`)

A flat base is not enough if the centre of mass is not over it. On the solid's voxels the
search takes, for each candidate cut from 1 % to 4 % of the height, the convex hull of the
first kept layer and the centre of mass of everything above it, and computes the tilt at
which the model would topple (`atan(margin / height of the centre of mass)`). The lowest
cut that reaches 15° wins. If none does, the 1 % cut is kept and a pedestal about 1.5 % of
the grid thick is added under it: the filled convex hull of the footprint and of a circle
around the centre of mass wide enough for 15°. The flood runs once more afterwards, so a
hollow closed off by the pedestal (an arched sole) is filled rather than left as a bubble.
Goblin, res 512: 12.5° on a plain 1 % cut, no cut up to 4 % reaches 15°, pedestal → 15.5°.

### 2.3c Thin walls (`--print-wall MM`, default 0.8)

The band guarantees `2·band` cells of material and no more: a cape, a blade edge or a
strand prints as a 0.4 mm film. Fattening the whole model (a wider band) trades that for
lost detail everywhere, so only the thin parts are grown, on the solid's voxels, one cell
per side per pass:

- `depth`: chamfer distance (3-4-5) of each solid vertex to the exterior, near the surface;
- `core`: solid deeper than half the wall — already thick enough;
- `thin`: shallow solid farther than `1.6·r` from any core (`r` = half the wall). A flat
  thick surface is exactly `r` from its core; the extra `0.6·r` keeps out the sliver an
  opening would shave off an ordinary convex edge.

Thin vertices take their exterior 6-neighbours; passes stop when nothing is thin. The new
surface lies outside the narrow band, so it is extracted and smoothed with the caps, and
the flood runs again so that growth cannot enclose a pocket. Relief that is attached to
thick material (a strap, a rivet) is within reach of a core and is left alone.

Measured: a sheet with no thickness comes out 2 cells thick, and about 6 with
`min_wall_cells = 6`; a thick box is unchanged (tests). On the goblin at res 512 with a
1.6 mm target, the wall under the thinnest 1 % of the surface goes from 0.81 mm to
1.29 mm — better, not a guarantee: the membrane smoothing takes some of the growth back.
At the default 0.8 mm the goblin has almost nothing to grow (13 K grid vertices).
Needs one byte per grid vertex (1.1 GB at res 1024); skipped above that. At res 1024 the
whole solid remesh — vote, flood, 4 thickening passes, base search, extraction and 96
smoothing passes — takes 68 s on the goblin (5–10 s at res 512).

### 2.3d Necks (twice `--print-wall`)

The thin-wall rule leaves alone anything within reach of thick material, so a short pinch
between two thick parts — a wrist, an ankle, a head on a thin neck — is never grown, and
it is where a print breaks. With `R` = half the target diameter (the target is twice the
wall: 1.6 mm at the default 0.8), on a grid of step `max(1, res/512)` restricted to the
solid's bounding box:

- `core`: solid farther than `R` from the exterior (exact distance transform);
- `parts`: 26-connected components of the core, ignoring those smaller than a ball of
  radius `R` (a part has to be at least twice the neck to count);
- `joints`: geodesic fronts grown breadth-first from every part at once through the
  shallow solid; where fronts from two parts meet is the joint between them, and its
  face count is its cross-section. Under `π·R²` (a disc of the target diameter) it is a
  neck; a plate as wide as the blocks it joins is not;
- `fix`: a tube of radius `R` stamped along the shortest path from one part, through the
  joint, to the other; the flood runs again. The path runs on the sub-grid, so at res 1024
  the tube can sit one sub-grid cell (2 cells) off the joint's centre: the dumbbell at
  res 1024 with a 120-cell target comes out 119 cells thick, centred to 2 cells.

Tests: two blocks joined by a short bar 4 cells thick come out joined by one 12.2 × 12.2
cells thick against a 12-cell target, where the thin-wall rule alone left it at 5.7 (the
cap smoothing does not take it back); joined by a plate as wide as the blocks they are
unchanged, as is a thick box. On the goblin at res 512
with the default wall: the core is one part, no joints, nothing stamped — also printed at 40 mm, where the target is 20 cells. With a 30–40-cell target the core splits into two parts whose joint is wider than the disc, so nothing is stamped either. The pass costs about 5 s and 640 MB at res 512. Res 1024: one part, no joints; the whole replay (vote, open ball, walls, necks, base, extraction, decimation) took 264 s with a 4.8 GB peak.

Memory: one int32 label, one int32 owner, one uint16 distance and (transiently) one
float per sub-grid voxel of the bounding box.

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

### 2.6 Printability report

Printed after every watertight run, written as JSON by `--print-report PATH`, served by
`GET /jobs/{id}/report`. Measured on the final mesh at its printed size:

| item | how | warning when |
|---|---|---|
| size, volume, surface | bounding box, signed volume | — |
| watertight, shells | the audit (§2.5) | not watertight; more than one shell |
| walls | a ray from every face straight into the material; the first hit counts only if that face looks back (normals more than 120° apart), so an ordinary edge is not a thin wall. Reported: the wall under the thinnest 1 % of the surface, and the share under the target | thinnest 1 % under 75 % of the target, or more than 10 % of the surface under it |
| base | vertices within 0.1 mm of the plate; convex hull area; centre of mass against the hull | no flat base; centre of mass outside; topples under 10° |
| overhangs | surface share facing down steeper than 45°, base excluded | above 25 % |

The wall share includes fine relief (a 1 mm strap on a thick body measures 1 mm), which is
why the warning keys on the thinnest percentile rather than on the share alone. Goblin,
res 512, defaults plus `--print-base auto`: 81 × 50 × 100 mm, 75 cm³, thinnest 1 % at
0.81 mm, stands at 15.5°, 12 % overhang, no warnings.

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
| `--print-base F\|auto` | `print_base` | flat base, fraction of height, 0..0.5; `auto` = lowest stable cut, else a pedestal (implies watertight) |
| `--print-report PATH` | `GET /jobs/{id}/report` | the printability report as JSON |
| `--print-interior M` | `print_interior` | `auto` (default): cap tears, keep wide openings open; `solid`: fill those too; `hollow`: cap nothing |
| `--print-wall MM` | `print_wall` | minimum wall at `--print-size`, default 0.8; thinner features are grown locally, joints between thick parts narrower than twice it are thickened (0 = off) |

Not combinable with
`--decim GRID` (cluster decimation tears the surface) or `--retopo`.

`trellis-print in.glb out.3mf [--res --faces --size --base --wall --band --interior --glb]` runs
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
- Necks are found and fixed but not reported: the log says how many joints were
  thickened; the report (§2.6) measures walls, not joints.
- Showcase models through `trellis-print`, `--interior solid` against `auto`: six of seven (axe, chest, drone, golem, knight, racer) are identical. The cottage loses 35 % of its volume: its mesh has no floor, so the flood enters from below and the walls are hollow either way; `solid` then fills the attic as a plug floating half-way up, `auto` releases it and leaves a consistent open-bottom shell with fillets in the roof corners. No necks at 1.6 mm on any of them. A lathed vase (neck 28 % of its height) with five 0.06 tears in its wall: `solid` fills it, `auto` keeps it hollow with the tears capped (one shell, genus 0) — the ball sweeps up to a tear from both sides but cannot pass through it.
- Not done: hollowing with drain holes, per-vertex colour 3MF for slicers that ignore textures.

## 6. Tests

`trellis-test-mesh-print [mesh.glb]` — audit counts on fixtures (open box, inside-out,
pinch); solid remesh of a closed box (plain remesh gives two shells, solid one, volume
within 10 % of the grown box), of a box with internal geometry and with a pinhole, of an
open cup; the openness vote (a box with a 0.12 hole in its lid is hollow with the
flood alone and solid with `seal = 3`; a shallow tray stays a tray); the open ball (a cup
2.5 times as deep as it is wide, filled by the vote alone, stays hollow, also with a tear
in its side; the torn box and the tray do not change); necks (§2.3d); the flat base (exact plane, expected volume); `make_printable` decimated to
2 K faces; repair of a hole + fin + inverted winding; the 3MF package; thin-wall growth (sheet grown,
thick box unchanged); the report and the stance (a top-heavy block falls over on a plain
cut and stands on the automatic pedestal; a thin sheet is reported thin).
With a mesh argument it runs the full path on it and requires a watertight result.
