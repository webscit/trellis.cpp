# Improvement ideas

Follow-ups to the multi-view conditioning and mesh-refinement work (branch `multiview`).
Items 1–4 are backed by measurements in [`docs/spec/30-multiview.md`](docs/spec/30-multiview.md)
and [`docs/spec/31-mesh-refine.md`](docs/spec/31-mesh-refine.md); the rest are ideas or gaps left open.

Suggested order: a concave test object for **5** and **4**, **7a** (a 5×+ headroom measured in the DiT forward; item 7 was measured and gives nothing), then **1** and **4** (they decide whether
multi-view is worth using), then **9** and **12** (Studio usability).

## Multi-view quality

1. **Outlines come out slimmer — addressed on branch `mv-quality` (`--mv-ss union`, per-stage
   `--blend-temp`; opt-in).** Measured on CUDA (knight, 2 seeds): the SS blend causes it; union
   brings held-out IoU from 0.728 to 0.820 (single view 0.809) and keeps the best back colour.
   Follow-up `--mv-ss fill` (blend + front cells only where the front outline has holes) also
   keeps the blended side views: cond. IoU .851 vs .825 union / .806 blend. Visual-hull carving
   was tried and dropped (unsafe when proportions differ). Open: pick a default; tune τ. Original note: Blending the views' guidance shrinks the shape:
   at N=2 the front silhouette IoU dropped from 0.94 to 0.85. Ideas to test:
   - condition the sparse-structure stage (which sets the overall shape) on the front view only;
     `--mv-stages` already allows this, but the comparison hasn't been run;
   - merge the views' occupancy predictions (union) instead of averaging their guidance;
   - use a different blend temperature per stage.
2. **The model's canonical pose overrides the views — addressed on `mv-quality`
   (`--yaw-offset auto`; opt-in).** The chest gets yaw 270, so the views land on the right faces
   (domed ends kept, back colour 29.9 → 20.8). Silhouette limits: front/back ties, proportions.
   Original note: TRELLIS rotated the chest into
   its own preferred pose despite explicit side views. Fix: try a few yaw offsets and keep the one
   whose renders best match the input silhouettes.
3. **Front axis only partly settled (R1).** `-y` beats the ComfyUI conventions (`ref-z`/`ref-x`),
   but the yaw-180 control on the knight was inconclusive. Re-run on an object with a clearly
   asymmetric silhouette.
4. **Better evaluation.** The A5 back-silhouette criterion is meaningless with orthographic renders
   (the back is a mirror of the front). Switch to perspective renders and held-out real photos;
   keep a small fixed test set and a scoring script (`tools/render_views.py`) in the repo.
5. **Occlusion-aware texture blending — done on `mv-quality` (`--mv-visibility`; opt-in).** No
   measurable effect on knight/chest (convex, views 90° apart); needs a concave test object.
   Original note: Weights depend only on the view direction, so a view can
   colour regions it never saw. Weight by visibility from each camera, at least in the texture stage.
6. **Unknown camera angles — addressed on `mv-quality` (`--view PATH:auto`, `--check-views`,
   server `azimuths=…,auto`, Studio "auto" direction).** Knight az 90/180/270 recovered exactly;
   front/back is ambiguous on symmetric objects (reported). Original note: Users must supply azimuths; estimate them automatically, or at least
   sanity-check the supplied ones.

## Speed

7. **Batch the views — tried, no gain (measured 2026-11-18, CUDA on Thor).** Each guided step runs
   V+1 DiT forwards one after another. A prototype (since discarded) stacked the entries on the
   token axis with per-entry attention and timed it on the real weights. Output was bit-exact, but
   it was **slower**:

   | flow, tokens | batch 3 | batch 5 |
   |---|---|---|
   | SS dense, 4096 | 0.94× | 0.93× |
   | shape 512, 3110 | 0.93× | 0.88× |

   Forward time is linear in tokens, so nothing per-forward is left to amortise. Skipping
   near-zero views would rarely fire at τ=2 (mean weights ~0.5/0.5 at N=2). Not worth integrating;
   see item 7a instead.

7a. **Make each DiT forward faster (new, highest-value speed item).** A 4096-token SS forward takes
   1.46 s (~8 TFLOPS effective) whatever the attention mode (FA default 1.46 s, FA fast 1.46 s,
   exact SDPA 1.59 s), so the linears dominate. cuBLAS on the same GPU does the DiT's MLP GEMM
   (8192×1536 · 1536×4096) at **51 TFLOPS** (f32 accumulate) / **193 TFLOPS** (f16 accumulate),
   which would put all linears at ~0.2 s per forward. Next step: per-op timing of one forward
   (which kernels ggml-cuda picks for f16 weights × f32 activations on sm_110, and how much time the
   glue ops — casts, conts, RoPE `set_rows`, concats — take). This speeds up single-view too.
8. **Batch the per-view preprocessing.** BiRefNet background removal and DINOv3 encoding run once
   per view, sequentially.

## Mesh refine

9. **Refine in Studio.** The server's `/refine` endpoint exists, but the UI has no mesh input.
10. **Refine with retopo.** `--retopo` with the original transform restored is rejected rather than
    supported.
11. **Pixal3D.** Multi-view and refine are rejected for `--model pixal3d`.

## Server and UI

12. **Live progress in the browser.** Only the desktop app streams progress (server log events); the
    browser shows a timer. Add server-sent progress events.
13. **Serve the UI from `trellis-server`.** A `--web-dir` flag would put UI and API on one port; the
    Docker image could then ship `app/dist`, and the browser would default to the page's origin.
14. **Remote servers in the desktop app.** Needs a Host field, a wider `connect-src` CSP, and skipping
    the local server launch when the host isn't local (today: use an SSH tunnel).
15. **Missing options.** `--mv-stages` isn't exposed by the server or the UI; the Studio form-layout
    question (plan open question 4) is still open.

## Testing and packaging

16. **Unrun checks:**
    - A1/A2 oracles against the real reference code (`MV_REF=<ComfyUI-Trellis2 clone>
      python tools/ref_multiview_weights.py` and `tools/ref_multiview_step.py`);
    - the T10 legacy parity tests;
    - the Windows/MSVC build;
    - the Pixal3D path after the shared driver refactor;
    - the Tauri desktop build with the multi-view UI.
17. **Docker:** a Vulkan variant, x86 builds, publishing to a registry, building the image in CI.
18. **Keep the browser test.** Add the Playwright end-to-end test of the Studio multi-view flow to the
    repo, with a request check that works for multipart bodies (Playwright can't read them; inspect
    server-side or route the request).
