# Getting started with Trellis Studio

**Trellis Studio** is a desktop app that runs Microsoft TRELLIS.2 image→3D
reconstruction locally (via `trellis.cpp`) and previews the result in an
interactive 3D viewer. This guide gets a brand-new machine from zero to a
generated `.glb`.

## One-command install

The installer auto-detects your GPU runtime (**CUDA → ROCm → Vulkan** fallback),
downloads the matching `trellis-server` build plus the ~16.5 GB of model weights,
installs the desktop app, and writes the config the app reads on launch.

### Linux (x86-64)

```bash
curl -fsSL https://raw.githubusercontent.com/pwilkin/trellis.cpp/main/install/install.sh | bash
```

### Windows (x64), in PowerShell

```powershell
irm https://raw.githubusercontent.com/pwilkin/trellis.cpp/main/install/install.ps1 | iex
```

That's it — launch **Trellis Studio**, drop in an image, and click **Generate 3D**.

## What gets installed where

| what | Linux | Windows |
|------|-------|---------|
| server + runtime libs | `~/.local/share/trellis-studio/runtime/` | `%LOCALAPPDATA%\trellis-studio\runtime\` |
| model weights (~16.5 GB) | `~/.local/share/trellis-studio/models/` | `%LOCALAPPDATA%\trellis-studio\models\` |
| app config | `~/.config/trellis-studio/config.json` | `%APPDATA%\trellis-studio\config.json` |
| desktop app | `.AppImage` in the install dir | installed via the setup .exe (Start menu) |

## Installer options

Both scripts accept the same flags (`--flag value` on Linux, `-Flag value` on
Windows):

| flag | effect |
|------|--------|
| `--backend cuda\|rocm\|vulkan` | force a runtime instead of auto-detecting |
| `--gpu N` | GPU index (default `0`; `<0` = CPU) |
| `--port P` | server port (default `8080`) |
| `--dest DIR` | install location |
| `--models-dir DIR` | where to store weights (e.g. a bigger drive) |
| `--quant q8\|q4` | download quantized weights: `q8` ~9.5 GB (near-lossless), `q4` ~6 GB (smaller, slight quality loss). Default is f16 (~16.5 GB). |
| `--skip-models` | don't download weights (set the folder later in Settings) |
| `--skip-app` | don't download the desktop app |
| `-y` / `-Yes` | don't prompt for confirmation |

Examples:

```bash
# reuse weights already on a fast drive; force Vulkan
./install/install.sh --backend vulkan --models-dir /mnt/ssd/trellis --skip-models
```

## Backend detection

- **NVIDIA** → CUDA (the bundle ships the CUDA runtime; nothing else needed).
- **AMD / Intel / everything else** → **Vulkan**, which is self-contained and, on
  the validated Strix Halo iGPU, actually the fastest backend. The installer
  notes when an AMD card is ROCm-capable.
- **ROCm** is opt-in with `--backend rocm`. The published ROCm bundle needs a
  matching TheRock ROCm 7.x runtime on your library path (`LD_LIBRARY_PATH` /
  `PATH`); if the server won't start, re-run with `--backend vulkan`.

## Using the app

1. **Drop or pick an image** (or paste from the clipboard).
2. Adjust **Resolution** (512 light / 1024 cascade / 1536 high), **seed**,
   **background removal**, and **UV unwrap** if you like.
3. **Generate 3D** — this takes a few minutes; the live stage line shows progress.
4. **Rotate/zoom** in the preview; **Reset view** re-frames; **Export to GLB** saves the model; the arrow beside it picks the format (GLB, or 3MF when "3D print" was on). Turning **3D print** on reveals the print size, flat base, minimum wall and interior options, and adds a print report under the preview.
5. Every result is saved to a local **gallery** (IndexedDB) — click a thumbnail to
   reload it, even after restarting the app.

## Paint a mesh

**Paint a mesh** (the second tab above the drop zone) colours a model you already have, for
example a CAD part, from a photo of the real object. The mesh keeps its triangles, its size
and its orientation: nothing is regenerated, so the result fits whatever it was designed for.

1. **One-time setup.** Reading a mesh needs one more model file, `shape_enc.gguf`, that the
   installer doesn't download. Convert it from Microsoft's checkpoint (about 700 MB) with
   the converter in this repository, writing it into your models folder:

   ```bash
   git clone https://github.com/pwilkin/trellis.cpp && cd trellis.cpp
   python3 -m venv .venv && .venv/bin/pip install numpy gguf huggingface_hub
   .venv/bin/hf download microsoft/TRELLIS.2-4B ckpts/shape_enc_next_dc_f16c32_fp16.safetensors \
       ckpts/shape_enc_next_dc_f16c32_fp16.json --local-dir trellis2
   TRELLIS_MODELS=trellis2 TRELLIS_GGUF_OUT=~/.local/share/trellis-studio/models \
       .venv/bin/python tools/convert.py shape_enc
   ```

   On Windows, use `.venv\Scripts\` and set `TRELLIS_GGUF_OUT` to
   `%LOCALAPPDATA%\trellis-studio\models`. The server looks for the file on each job, so
   there is nothing to restart.
2. **Drop a mesh** (STL, OBJ or GLB) on the mesh slot, or anywhere on the window: Studio
   switches to *Paint a mesh* by itself. Then add a photo of the object as usual. STL and
   OBJ files are read in millimetres, GLB in metres.
3. **Turn the mesh so the side in the photo faces you.** The preview shows the mesh in grey
   with a marker on the side the photo will paint. **Up** picks the file's axis that points
   up (*auto* is z for STL, y for OBJ and GLB); **⟲ 90° / ⟳ 90°** turn it about that axis.
   The photo is matched by shape as well as by side, but a front that doesn't match the
   photo can lose the photo's details. The painted model keeps the file's own orientation
   whatever you pick here.
4. **Resolution** is *512 · fast* by default; *1024 · detailed* gives sharper colours but
   is slower. Meshes with many small triangles (or a large CAD part) can be too much for 1024:
   Studio then says so and offers **Retry at 512**.
5. **3MF** set to *3MF of this mesh* also writes a 3MF of the painted mesh. Its size comes
   from the file (untick **from file** to scale it), and **3MF colours** works as in image
   mode: the texture, or 4 to 32 filament colours. With filament colours, **Colour detail**
   (0.5 mm, 1 mm, 2 mm or *whole triangles*) paints the large flat triangles of a CAD part in
   smaller pieces, so a face made of two triangles still gets the photo's detail. The print
   report under the preview checks the mesh as it is: Studio doesn't add a base or thicken
   walls, so fix those in your CAD tool.
6. **Paint mesh**. The result is named `<your mesh>_painted` and goes to the gallery with a
   *mesh* badge; clicking it brings back the mesh and its orientation, ready to paint again.

Each tab keeps its own resolution and print settings.

## No app? Use it in a browser

The UI is a plain web bundle, so you can skip the desktop app entirely:

```bash
# start the server the installer downloaded
~/.local/share/trellis-studio/runtime/trellis-server \
  --models ~/.local/share/trellis-studio/models --port 8080
```

then open the built UI (or `npm run dev` in `app/`) and point it at
`127.0.0.1:8080` in **Settings**.

## Portable (no-install)

Prefer not to install anything? Grab the portable archive from the releases page
instead of running an installer:

- Windows: `trellis-studio-windows-x64-portable.zip`
- Linux: `trellis-studio-linux-x86_64-portable.tar.gz`

Unzip it anywhere and run the app in place — it keeps **everything inside that
folder** (config and generated GLBs go to `./data/`, and it auto-detects a
`./runtime/` server and `./models/` weights next to it), so nothing is written to
your system. Drop the `trellis-<backend>-<os>-x64` runtime into a `runtime/`
folder and the GGUFs into `models/` (or point it at existing ones in Settings),
and you're set. Delete the folder to uninstall. (Linux still needs system
`webkit2gtk-4.1`.)

## Where are the logs?

Every server launch is written to a timestamped log file (the last 20 are kept):

| | Linux | Windows |
|--|-------|---------|
| installed | `~/.local/share/trellis-studio/logs/` | `%LOCALAPPDATA%\trellis-studio\logs\` |
| portable | `./data/logs/` next to the app | `.\data\logs\` next to the app |

Open the folder straight from **Settings → Server logs → Open**. The log records
the exact `--models` path, GPU index and backend the server was launched with,
plus its full stdout/stderr — attach the newest file to a bug report.

## Troubleshooting

- **"Server is offline"** right after generating — the pipeline is still loading
  the models; large weights take a moment on the first request.
- **Blank white window / the desktop flickers on launch** (common on NVIDIA and
  some Wayland setups) — the app now disables WebKit's DMA-BUF renderer on Linux
  automatically. If you still hit it, launch with
  `WEBKIT_DISABLE_DMABUF_RENDERER=1`; to opt back in, set it to `0`.
- **"Unable to establish IDB database file" / empty gallery** — some WebKitGTK
  builds can't open IndexedDB. Generations are unaffected: every result is still
  written to your output folder; only the in-app gallery won't persist across
  restarts.
- **Settings changes (e.g. models directory) seem ignored** — an older build
  could reuse a server left running by a previous crash. Fully quit the app (or
  reboot) once after updating; the current build kills the server with the app
  and no longer reuses a stale one when you hit **Save & restart**.
- **The app can't read the server response / a CORS error appears** — the app
  needs a `trellis-server` build that sends CORS headers (v0.4.4+). The installer
  pulls the *latest* release, so update if you're on an older server bundle.
- **"setup needed" banner** — no `config.json` was found. Re-run the installer, or
  open **Settings** and point it at your models directory.
- **ROCm server won't start** — install the gfx-matched TheRock ROCm runtime, or
  switch to Vulkan (`--backend vulkan`).
- **"painting a mesh needs …/shape_enc.gguf"** — do the one-time setup in
  [Paint a mesh](#paint-a-mesh); the path in the message is the models folder the
  server uses.
- **"This trellis-server can't paint meshes yet"** — the server predates *Paint a mesh*;
  re-run the installer to update it.
