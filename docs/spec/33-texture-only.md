# 33 — Texture-only mode (`--texture-mesh`)

Status: phases 1–5 done (STL input, decoder tree, `--texture-mesh` / `POST /texture`, the bake
onto the input, a 3MF of the input triangles with split painting and a printability report).
Split painting is confirmed in Snapmaker Orca 2.4.1.

Paints an existing mesh from one or more images, **without changing its geometry**. This is
upstream TRELLIS.2's `Trellis2TexturingPipeline` (`trellis2/pipelines/trellis2_texturing.py`),
listed in ROADMAP M7 as `texture_mesh_multiview`. It differs from `--refine-mesh` (spec 31),
which re-samples the shape on the mesh's latent cells and outputs a new surface.

```
mesh (STL / OBJ / GLB) → weld (STL) → up axis → normalize_mesh_trellis
  → mesh_to_flexible_dual_grid @R (dual vertices) → shape_encode (keeps every level's coords)
  → normalise with SHAPE_MEAN / SHAPE_STD
  → texture SLAT flow (tex_flow_512 when R == 512, else tex_flow_1024; image cond at the same
    resolution), multi-view blend as in spec 30
  → tex_decode driven by subs built from the encoder's levels → PBR on exactly the input voxels
  → uv_bake onto the ORIGINAL triangles (VoxelPbr @R) → restore transform → GLB / 3MF
```

No sparse-structure flow, shape flow, shape decode, hole fill, remesh or decimation runs.

## Reference behaviour (verified against upstream sources)

- `run(mesh, image, seed=42, resolution=1024, texture_size=2048)`: `preprocess_mesh`, then
  `encode_shape_slat(mesh, resolution)`, then `sample_tex_slat(cond, tex_model, shape_slat)`,
  then `decode_tex_slat`, then `postprocess_mesh(mesh, pbr_voxel, resolution, texture_size)`.
  `resolution == 512` selects `tex_slat_flow_model_512` and a 512 image cond; anything else uses
  1024.
- `sample_tex_slat` normalises the shape SLAT with `shape_slat_normalization` and passes it as
  `concat_cond`. The noise is `shape_slat.replace(feats=randn(...))`, so the texture SLAT sits on
  the encoder's latent coords.
- **The texture decoder's voxel tree.** `SparseSpatial2Channel.forward` registers
  `channel2spatial_2 = (x.coords, idx, subidx)` on its output. `SparseTensor.replace()` shares
  the spatial-cache dict by reference. In the decoder, `SparseChannel2Spatial.forward(x, None)`
  reads that cache and gathers `x_feats[idx * 8 + subidx]` onto the stored **original fine
  coords**. The texture decoder (no `to_subdiv`; `decode_tex_slat` passes no subdivision)
  therefore rebuilds the exact voxel set of the mesh's dual grid at R, level by level. It does
  not predict anything. `shape_upsample` (used by refine) predicts subdivision and may prune
  cells (spec 31: 2 cells on the chest at 1024), so it must not be used here.
- `subidx = Σ (coord_i mod 2) · 2^i` over the spatial columns, i.e. slot `lx + 2·ly + 4·lz`.
  This is the same convention as our `sparse_s2c` (`sparse.cpp`) and `sparse_c2s` mask
  (`o + 8·i`, child `(2x + (o&1), 2y + (o>>1&1), 2z + (o>>2&1))`).
- `postprocess_mesh` bakes onto the preprocessed input mesh (same faces), UV-unwraps it with
  cumesh, rasterises at `texture_size`, and samples
  `grid_sample_3d(pbr, grid=(pos + 0.5) · resolution, trilinear)`. Holes in the atlas are
  filled by TELEA inpaint.

## Coordinate tree (`decode_tree_from_fine`, `shape_decoder.h`)

Each S2C block's output coords are `unique(c >> 1)` of its input, so the encoder's levels are
a function of the fine set alone: `levels[0]` = the dual-grid coords @R,
`levels[k] = unique(levels[k−1] >> 1)`. The tree is built from those, and the encoder is not
changed. The function checks that `levels[4]` equals the encoder's latent coord set and throws
otherwise (a tree for another mesh, or an encoder that pruned). For decoder stage k (k = 0..3,
coarse to fine), walk the current parents in the order `sparse_c2s` processes them. For each parent `i`,
set mask bit `o + 8·i` if child `(2x + (o&1), 2y + (o>>1&1), 2z + (o>>2&1))` is in
`levels[3 - k]`. The emitted children, in mask order, are the next stage's parents. This
reproduces `sparse_c2s`'s output order, so the final list is the order of `tex_decode`'s output
rows.

Invariant (test): the final coordinate list is a permutation of `levels[0]` (equal size, equal
set), and every stage's emitted set equals the encoder level it mirrors.

Measured (`trellis-test-shape-enc texsubs`, NVIDIA Thor), on a generated z-up STL (a torus on a
block, 6,156 facets → 3,080 welded vertices):

| grid | backend | latent | stages | fine voxels | tree vs dual grid | tex_decode |
|---|---|---|---|---|---|---|
| 128 | CPU | — | — | 53,616 | identical | 53,616 finite rows, 1.4 s |
| 256 | Vulkan | 852 @16 | 852 → 3,208 → 13,472 → 53,616 → 217,580 | 217,580 | identical | 217,580 finite rows |
| 1024 | Vulkan | 13,472 @64 | 13,472 → 53,616 → 217,580 → 866,560 → 3,476,292 | 3,476,292 | identical | 3,476,292 finite rows, 47 s total (encode 27 s) |

A latent with one cell removed is rejected. Note the token count: this small part already
gives 13,472 texture-flow tokens at 1024, so phase 3's warning threshold matters.

## Latent normalisation

The texture flow's concat input is the normalised shape latent (`slat_norm` in
`trellis_cli.cpp`). `ShapeLatent.feats` is the raw posterior mean, so it gets
`(raw − SHAPE_MEAN[c]) / SHAPE_STD[c]` first.

## Phases

1. **STL input, welding, up axis** (`mesh_io`).
   - Binary STL is detected by size (`84 + 50·n`), because binary files often start with
     `solid` too. Otherwise ASCII (`solid … facet … vertex … endsolid`).
   - STL vertices are welded on exact position. STL is a triangle soup: unwelded, every edge
     is a boundary to the voxeliser's boundary pass, and xatlas makes one chart per triangle.
     Faces that collapse to a repeated index are dropped.
   - `MeshUp`: `Auto` (z for STL, y otherwise), `Y`, `Z`. A z-up input is rotated into glTF y-up
     on load, `(x, y, z) → (x, z, −y)`, so the rest of the pipeline is unchanged. Composed with
     `normalize_mesh_trellis`'s `(x, y, z) → (x, −z, y)` this is the identity: TRELLIS's grid
     frame is z-up.
   - CLI: `--mesh-up auto|y|z` for `--refine-mesh` (and `--texture-mesh` later). `trellis-print`
     reads STL too: the generated STL's 3MF keeps its 60 × 40 × 56 mm z-up box.
   - Tests: `trellis-test-mesh-io` (binary cube with a `solid` header and a degenerate facet,
     ASCII cube with `-0`, closed after welding, the up-axis rotation, z-up lands unrotated in
     the grid frame, malformed STL files).
2. **Decoder tree** (`shape_decoder`): `DecodeTree decode_tree_from_fine(fine, latent)`, see
   above. Test: `trellis-test-shape-enc texsubs <shape_enc.gguf> <tex_dec.gguf> <mesh> <grid>
   [gpu]` checks the invariant and the rejection, then runs `tex_decode` on the tree and checks
   for one finite row per fine voxel. Phase 3 feeds `tex_decode` the flow's texture latent and
   pairs its rows with `DecodeTree::coords` for the `VoxelPbr`.
3. **CLI / server** (done). `--texture-mesh PATH`, `POST /texture`, `"task"=texture` on
   `/jobs`, `mesh_up` field. In `trellis_run` it is a third branch next to refine:
   - [3/6] voxelise with duals, `shape_encode`, `decode_tree_from_fine`; the pose check runs on
     the mesh's voxels.
   - [4/7] no shape flow: `slat_norm = (raw − SHAPE_MEAN) / SHAPE_STD` on the latent coords.
   - [5/7] no shape decode: `mesh` is the input, `so.coords` / `so.subs` come from the tree, so
     the texture stage runs unchanged (`tex_flow_512` + cond 512 at R = 512, else
     `tex_flow_1024` + cond 1024; never the mixed-resolution path).
   - Needs `shape_enc.gguf` (no voxel fallback). Past `--max-tokens` the run stops and suggests
     `--res 512`; above 16,384 tokens at 1024 it warns.
   - `apply_texture_mesh_rules` (CLI and server) refuses `--refine-mesh`, `--model pixal3d`,
     `--no-texture`, `--retopo`, explicit `--watertight`, `--print-base`, `--faces` /
     `--decim`, `--normal-map`, `--dump-post`, and drops the watertight rebuild that `--3mf`
     implies. `--tex-res` is ignored with a notice.
4. **Bake onto the input mesh** (done). The [7/7] weld / hole fill / remesh / decimation are
   skipped; the input triangles are unwrapped (`--xatlas`, the default, or `--box-uv`) and baked
   from `VoxelPbr{tree coords, PBR, R}` with no BVH snap. The output is restored to the input's
   position and size.
5. **Printing** (done). `--3mf` writes the input triangles as they are, at the file's size
   unless `--print-size` (STL / OBJ units read as millimetres, glTF as metres), and the
   printability report audits those triangles (welded over the UV seams). `--print-colors`
   paints large triangles in parts with Orca split states; see below.

## Split painting (`--print-color-res`, all modes)

One colour per triangle is coarse on a CAD part, whose faces are a few large triangles. Orca
stores a painted triangle as a split tree (`libslic3r/TriangleSelector.cpp`, `serialize` /
`deserialize`; the string form in `Model.cpp`, `get_triangle_as_string` /
`set_triangle_from_string`), so a triangle can carry several colours while the mesh stays as
it is. Read from OrcaSlicer's sources:

- Per node a nibble: bits 0–1 the number of split sides, bits 2–3 the leaf state or the
  special side. A leaf state of 3 and above is `0b11` there, then a nibble of (state − 3), or
  `0xF` and a nibble of (state − 18) from 18 on. A split node is followed by its children in
  the order 3, 2, 1, 0.
- A three-side split of (v0, v1, v2) with midpoints m01, m12, m20 has children
  0 = (v0, m01, m20), 1 = (m01, v1, m12), 2 = (m12, v2, m20), 3 = (m01, m12, m20)
  (`perform_split`, special side 0).
- The 3MF string: each hex digit is 4 bits, 2^j first, and the string is read from its end.
  The leaf-only codes this gives are the ones Snapmaker Orca 2.4.1 writes (`4`, `8`, `0C`,
  `2C`, `2FC`).

`quantize_print_colors` with `ColorOptions::cell_mm > 0`:

1. Per triangle, a depth d (≤ `kPaintMaxDepth` = 8) so that its longest edge / 2^d ≤ cell (at
   the 3MF's size). Leaves are capped at 8M; past that the cell grows.
2. A temporary leaf mesh: uniform three-side splits in Orca's child order, midpoints shared
   per vertex pair (UVs interpolated; a UV seam's two sides keep bit-identical positions).
3. The unchanged quantiser (palette, speck merge) runs on the leaves. Two faces of different
   depth meet in T-junctions, where the leaves share no vertices: each leaf side on an
   original side is recorded with its interval along it, and overlapping intervals give the
   speck merge the missing borders (without them, the dense goblin at 100 mm kept 6,584
   patches instead of 226).
4. Each triangle's leaves fold into a split tree, equal-coloured subtrees collapsing to a leaf
   (`paint_tree_code`), so a uniform triangle keeps its one-leaf code. `face_slot` is the
   majority slot (for the 3MF's `basematerials`).

`write_3mf` writes the code as is (they can be thousands of digits).

Tests (`trellis-test-mesh-print`): every slot's one-leaf tree equals `paint_color_code`;
random trees of depth 1–4 with 2–32 colours round-trip through a decoder written from Orca's
reader; a red disc on a blue 100 mm square of two triangles decodes, with Orca's child
geometry, to the right colour on every leaf more than 0.02 from the rim (131,072 leaves) and
the disc's share within 1e-4 of its area; a 0.3 mm² sliver at a T-junction merges (the test
fails without the interval borders). Measured:

| input | triangles painted in parts | leaves | patches (whole triangles) | 3MF size |
|---|---|---|---|---|
| torus-on-block STL at 60 mm | 1,175 of 6,156 | 19,923 | 47 (49) | +7% |
| generated goblin at 100 mm | 8,350 of 141,168 | 181,809 | 226 (231) | — |

On the STL the block's faces, two triangles each, go from one flat colour to the texture's
detail. Confirmed in Snapmaker Orca 2.4.1: the split colours load and show on the block.

## Validation

Measured (NVIDIA Thor, Vulkan):

| input | res | tokens | fine voxels | wall | geometry check |
|---|---|---|---|---|---|
| generated torus-on-block STL, 6,156 facets | 512 | 3,208 | 866,560 | 46 s | same 6,156 triangles, vertices within 4e-8 of the extent |
| same | 1024 | 13,472 | 3,476,292 | 319 s (texture flow 258 s) | 3MF: the 6,156 input triangles, 60 × 40 × 56 mm z-up, 4 filament colours |
| goblin GLB generated at 512 (seed 42), repainted from `goblin.png` (seed 7) | 512 | 3,402 | 2,148,266 | 62 s | same 141,168 triangles, vertices within 4.5e-8 |

The repainted goblin matches the generated one in a four-view render: helmet, armour, belt,
buckle, loincloth and skin land in the same places, a little more saturated. The torus,
painted from the goblin image, takes coherent armour / skin patches and stands upright (z-up
handled). `POST /texture` with the STL returns the GLB; `watertight=on`, `faces` and a missing
mesh answer 400. The renderer-based colour-error comparison with `--refine-mesh` (item 2 below)
is not done yet.

1. Coordinate-set identity (phase 2), CPU and Vulkan.
2. Strip the texture from a generated GLB, retexture it from its source image, and score it with
   the spec-30 renderer on conditioned and held-out views, against `--refine-mesh` on the same
   inputs: colour error close, geometry identical.
3. Orientation: a z-up STL box with one distinctly coloured face in the reference image; that
   face lands on the right side in the GLB and in the slicer.
4. STL → `--3mf --print-colors 4` opens painted in Snapmaker Orca with the input's triangles and
   mm size.

## Open questions

- Default R: 1024 matches upstream; 512 is faster and often enough for a print.
- `--print-colors`: subdivision first, or straight to `paint_color` split states.
