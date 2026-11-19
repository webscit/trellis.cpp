# 34 — Studio: paint a mesh

Status: phases 1–3 done (`--mesh-up` signed axes, `--mesh-yaw`, measured below; the Studio
mode switch, mesh slot and turned preview; the paint request, records, print options, 512
default and "Retry at 512"). Phase 4 (user docs) open.

Brings texture-only mode (spec 33, `POST /jobs` with `task=texture`) to Trellis Studio. The
user drops an STL / OBJ / GLB and a photo, turns the mesh until the photo's side faces the
front, and gets the mesh back painted, as a GLB and optionally as a coloured 3MF of its own
triangles. Refine mode (`task=refine`) stays out of Studio for now.

Decisions (from the brainstorm):

- **A.** A mode switch at the top of the left panel; dropping a mesh file switches to it.
- **B.** The bare mesh is previewed before generating, with snapped orientation controls
  (up axis + quarter turns) and a server-side `mesh_yaw`.
- **D.** The print options become "3MF of this mesh": size from the file, colours, colour
  detail (`print_color_res`). Colour detail is also added to the normal print options.
- **E.** Resolution 512 (default) or 1024; a too-dense error offers "Retry at 512".

## Why orientation matters

With one image the pipeline conditions on the photo globally: the azimuth is ignored
(`V > 1` gates every per-view stage in `trellis_run`), and the texture flow paints the mesh as
if the photo shows its canonical front. Which side of the input is the front therefore comes
only from the file's axes. A CAD part exported lying down, or facing +x, gets the photo's
front on the wrong face. Extra views have azimuths relative to that same front.

The canonical front is grid −y (`front_axis` default `-y`, z up). A z-up STL maps to the grid
with the identity (spec 33), so the front is the STL's −y side. In glTF (y up) that is +z,
which is where `<model-viewer>`'s default camera sits: **the preview's default view is the
front the photo is matched to.**

Measured in phase 1 (`--res 512`, seed 1, one front photo: a red face with a white disc, a
yellow top; the share of white texels per side of the output, named by the STL's axes):

| mesh | turn | front (log) | white on the photo's face | elsewhere |
|---|---|---|---|---|
| cube 30 mm | 0°, 90° | −y, −x | disc on 5 of the 6 sides | — |
| box 40 × 20 × 30 | 0° | −y (wide side) | 10.5% on −y | 10.6% on +y (the back) |
| box 40 × 20 × 30 | 90° | −x (narrow side) | 0% | 0% everywhere |
| wedge, tall face −y | 0° | −y | 10.7% on −y | 0% |
| wedge, tall face −y | 180° | +y | 10.5% on −y | 0% |

So the front is where the spec says (a z-up STL's −y, at turn 0), and the top lands on the up
axis every time. But the texture model also matches the photo to the shape: a symmetric cube
gets the disc everywhere, the box's back mirrors its front, and the wedge's distinctive tall
face gets the disc even turned to the back. Where the shape at the front does not match the
photo (the box at 90°), the photo's details are lost. The turn therefore matters, but as
"help the model", not as "the photo lands on this face": Studio's wording says to turn the
mesh so the side in the photo faces you.

## Server / CLI: orientation of the input mesh

`--mesh-up` today takes `auto|y|z`. It becomes the full set of snapped orientations:

- `--mesh-up auto|x|-x|y|-y|z|-z`: the axis of the file that points up. `auto` stays z for
  `.stl`, y otherwise. Server field `mesh_up`, same values.
- `--mesh-yaw 0|90|180|270`: then a turn about the up axis, counter-clockwise seen from
  above, in degrees. Server field `mesh_yaw`. Other values are refused (a quarter turn is an
  exact axis permutation with sign changes, so the geometry stays bit-identical).

Both apply to `--texture-mesh` and `--refine-mesh`. Implementation:

1. `load_mesh` already maps the file's up axis to glTF y. It gets a general
   `MeshOrient { int up; int yaw; }` and applies one signed axis permutation `P` to the
   vertices (with det `P` = +1, so the winding is unchanged).
2. Everything downstream runs as now. After `restore_mesh_transform` (the two call sites,
   the 3MF mesh `bm` and the GLB `mesh`), the inverse permutation `Pᵀ` is applied, so the
   output keeps the file's orientation, as today with `--mesh-up`: a z-up STL comes back as a
   y-up GLB and a z-up 3MF, the input triangles unchanged.
3. A `[mesh]` log line prints the orientation and which file axis faces the front
   (`mesh_front_axis`).

Tests (`trellis-test-mesh-io`): for every up axis (auto and the six signed axes) and turn, on
an OBJ and an STL marker mesh: the map is a proper rotation, the up axis lands on glTF +y, the
reported front on glTF +z, turn 90 is counter-clockwise from above, and load → normalise →
restore → undo gives the default orientation's output bit for bit (the undo is the identity
exactly for the default). `trellis-test-multiview`: the argument rules (bad axis, bad turn,
`--mesh-up` / `--mesh-yaw` without a mesh, refine accepts them).

## Studio

### A. Mode switch

A segmented control at the top of the left panel: **Generate from image | Paint a mesh**.

- In "Paint a mesh", a **Mesh** drop zone sits above the photo drop zone (STL, OBJ, GLB,
  glTF binary). Below it one line: `part.stl · 6,156 triangles · 60 × 40 × 56 mm` (STL / OBJ
  read as millimetres, GLB as metres, as the server does).
- Dropping a mesh file on either drop zone, or anywhere on the window, switches to "Paint a
  mesh" and loads it. Dropping an image keeps the mode.
- The photo zone and "Extra views" stay as they are. The button reads **Paint mesh**; it is
  enabled with a mesh and a photo.
- The last mode and each mode's own settings (resolution, print) are remembered in
  `localStorage` (wrapped in try/catch), so switching back restores them.

### B. Mesh preview and orientation

Before generating, the viewer shows the mesh untextured:

- **Conversion in the browser** (`meshpreview.ts`, new, no dependency): STL (binary by size
  `84 + 50n`, else ASCII) and OBJ (`v` / `f` lines, polygons fanned, negative indices) are
  turned into a minimal untextured GLB (positions + flat normals, one grey material). GLB files
  are shown as they are. Parse errors are shown before anything is sent to the server.
- **Orientation** under the viewer, two snapped controls:
  - **Up:** Auto · +Z · +Y · +X · −Z · −Y · −X (Auto = +Z for STL, +Y otherwise).
  - **Turn:** ⟲ 90° and ⟳ 90° buttons, and the current value (0°, 90°, 180°, 270°).
  The preview applies the same rotation through `<model-viewer>`'s `orientation` attribute
  (no re-conversion), and resets the camera to the front.
- **Front marker:** a `<model-viewer>` hotspot at the front of the bounding box with the
  photo's thumbnail and the label "photo side". The caption reads "Turn the mesh so the side
  in the photo faces you." (Phase 1: the model matches the photo to this front, and to the
  shape; a wrong turn loses the photo's details.)
- The values go to the server as `mesh_up` and `mesh_yaw` (Auto sends nothing).
- When the result arrives, the viewer shows the painted GLB in the file's orientation; the
  orientation controls are hidden until the user goes back to editing (any input change).

### D. Print options

In "Paint a mesh" the "3D print" select becomes **3MF** with `off` / `3MF of this mesh`:

- **Size (mm):** a checkbox "from file" (checked by default, field disabled and showing the
  file's size); unchecking it enables the number field and sends `print_size`.
- **3MF colours:** as today.
- **Colour detail** (new, both modes): `0.5 mm` · `1 mm · default` · `2 mm` · `whole
  triangles`. Sent as `print_color_res` (0.5, 1, 2, 0) when colours are on. Tooltip: "Large
  triangles are painted in parts down to this size (Orca split painting); the triangles
  themselves are not changed."
- Hidden in this mode: flat base, minimum wall, interior, and the Texture on/off select. The
  request sends `3mf=on` without `watertight`; the server's rules refuse the rest anyway.
- The printability report panel works unchanged (the server returns one for texture jobs
  with a 3MF). Its wording for an open mesh says "your mesh", not "the model".

### E. Resolution and limits

- In "Paint a mesh" the resolution select has `512 · fast` (default) and `1024 · detailed`;
  1536 is hidden.
- When a job fails with the token limit (the server's error contains `--res 512`), the error
  toast has a **Retry at 512** button that sets 512 and resubmits.
- The stage text uses the server's stages as today; the encode stage reads "Reading your
  mesh".

### Requests

`api.ts`: `toForm` gains the mesh branch. With a mesh it sends `task=texture`, the `mesh`
part (named with the file's extension, so the server's sniffing and the log are right),
`mesh_up` / `mesh_yaw` when set, and the print fields above. Image and view fields are as for
generation. `waitForJob` is unchanged (result, 3MF, report).

### Records and gallery

- `GenParams` gains `mode: "image" | "mesh"` (absent = image), `meshUp`, `meshYaw`,
  `printSizeFromFile`, `printColorRes`.
- `GenRecord` gains `mesh?: Blob` and `meshName?: string`; resuming a pending job and opening
  a gallery item restore the mesh and its orientation.
- Gallery thumbnails of painted meshes get a small "mesh" badge; the caption shows the file
  name.

## Phases

1. **Orientation in the CLI / server** (`mesh_up` signed axes, `mesh_yaw`, the inverse on
   output, tests). Verify the front: a z-up STL box with one red face in the photo, at each
   of the four turns, comes back with red on the face the turn put at the front (rendered
   with the spec-30 renderer and in `<model-viewer>`).
2. **Studio mode switch, mesh drop zone, browser conversion, preview with orientation
   controls and the front marker** (done). `app/src/orient.ts` mirrors `mesh_io`'s maps (the
   preview shows `load ∘ native⁻¹`, so it is drawn in the file's own convention like the
   result); `app/src/meshpreview.ts` reads the files and writes the turn as a root node's
   rotation in the GLB's JSON (exact, and no reliance on `<model-viewer>`'s Euler order).
   `npm test` (Node's runner, 9 tests): every orientation's up / front / turn direction and its
   quaternion, the fronts trellis-cli logged in phase 1, STL binary (with a "solid" header) /
   ASCII and OBJ (quads, `v/vt/vn`, negative indices) giving the same box, GLB pass-through and
   the root node, and nine malformed files failing with a message. In headless Chromium: the
   wedge STL reads `40 × 20 × 30 mm` with `-y` at the front, ⟲ gives `-x` and `20 × 40 × 30`,
   up −Z shows it upside down; the marker sits on the front face; an OBJ dropped on the
   gallery switches modes; a junk STL keeps the previous mesh with an error.
3. **Requests, records, print options (D, both modes' colour detail), resolution and the
   retry** (done). The server now passes texture-mode failures on: an unreadable or flat mesh,
   a missing `shape_enc.gguf` and the token limit are thrown with their message rather than
   returned as an exit code (which the server reports as "3D reconstruction failed"); Studio
   recognises the token limit by its `use --res 512` and offers the retry, which repaints the
   failed record at 512 whatever is on screen by then. `MeshFile.size` (mm, from the parsed
   file or a GLB's POSITION min / max) gives the mesh line and "from file" without the viewer.
   Each mode keeps its own resolution and print settings in `localStorage`.
   End to end in headless Chromium against a `trellis-server` started with `--max-tokens 6000`:
   the wedge turned 180° with "3MF of this mesh" and 4 colours, first at 1024, is refused
   ("it needs 12,024 texture tokens, over this server's limit of 6,000") with **Retry at
   512**, which paints it: the server logs `turn 180 deg`, a 3MF at the file's 40 mm, 4
   filaments with 1 mm split painting, and the report says "Ready to print"; the gallery item
   (`mesh · 512`) reopens in "Paint a mesh" with the wedge and its 180° turn. The torus-on-block
   STL and a photo dropped together on the photo zone fill both slots; with "from file"
   unticked at 80 mm the 3MF is 80 mm and the report says "your mesh is not watertight" and
   flags its two shells. Image mode keeps 1024 and its own print settings throughout.
4. **Docs:** `docs/getting-started.md` gets a "Paint a mesh" section; the README's Studio
   lines and `app/README.md` mention it.

## Open questions

- Whether "Auto" should try to guess a front for CAD parts (for example, the largest face
  turned away from the build plate). Not planned; the snapped controls are the answer for now.
- A "Repaint" action on generated gallery items (brainstorm idea C) reuses all of this and
  is left for later.
