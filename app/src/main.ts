import "./ui.css";
import { generate, health, waitForJob, type GenerateHooks, type GenerateResult, type JobStatus } from "./api";
import { loadConfig } from "./config";
import { renderSettings } from "./settings";
import { all, clear as clearStore, del as removeRecord, get as getRecord, isEphemeral, newId, put } from "./store";
import { isTauri, listen, saveBytes, saveToOutputDir } from "./tauri";
import { Viewer } from "./viewer";
import { MESH_EXT, orientGlb, readMesh, type MeshFile } from "./meshpreview";
import { MESH_UPS, applyMap, frontAxis, nativeUp, previewMap, quaternion, type MeshUp, type MeshYaw } from "./orient";
import { AUTO_AZ, DEFAULT_PARAMS, MAX_VIEWS, type ExtraView, type GenParams, type GenRecord, type PrintInterior, type PrintReport } from "./types";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---- element refs ----
const dropzone = $("dropzone");
const fileInput = $<HTMLInputElement>("file-input");
const inputPreview = $<HTMLImageElement>("input-preview");
const dropHint = $("dropzone-hint");
const generateBtn = $<HTMLButtonElement>("generate-btn");
const progress = $("progress");
const progressStage = $("progress-stage");
const progressElapsed = $("progress-elapsed");
const cancelBtn = $<HTMLButtonElement>("cancel-btn");
const resetViewBtn = $<HTMLButtonElement>("reset-view");
const exportBtn = $<HTMLButtonElement>("export-btn");
const exportMenuBtn = $<HTMLButtonElement>("export-menu-btn");
const exportMenu = $("export-menu");
const viewerCaption = $("viewer-caption");
const galleryEl = $("gallery");
const clearGalleryBtn = $("clear-gallery");
const backendBadge = $("backend-badge");
const serverDot = $("server-dot");
const serverLabel = $("server-label");
const setupBanner = $("setup-banner");
const frontTag = $("front-tag");
const viewsEl = $("views");
const viewsList = $("views-list");
const addViewBtn = $<HTMLButtonElement>("add-view");
const viewInput = $<HTMLInputElement>("view-input");
const clearViewsBtn = $("clear-views");
const mvAdv = $("mv-adv");
const modeBtns = Array.from(document.querySelectorAll<HTMLButtonElement>(".mode-switch button[data-mode]"));
const meshZone = $("mesh-zone");
const meshInput = $<HTMLInputElement>("mesh-input");
const meshHint = $("mesh-hint");
const meshInfo = $("mesh-info");
const orientBar = $("orient-bar");
const orientUp = $<HTMLSelectElement>("orient-up");

// ---- state ----
const viewer = new Viewer($("viewer-mount"));
let inputImage: Blob | null = null;
let inputName = "input.png";
let currentGlb: Blob | null = null;
let current3mf: Blob | null = null; // the viewer model as a 3MF, when the run made one
let currentName = "model"; // file name stem for exports of the model in the viewer
// Set when the user opens a gallery item during a run: the result then goes to the
// gallery without taking over the viewer.
let browsedAway = false;
let activeId: string | null = null;
let serverOnline = false;
let generating = false;
let abort: AbortController | null = null;
let elapsedTimer: number | null = null;
let galleryUrls: string[] = [];
let warnedEphemeral = false;
let extraViews: ExtraView[] = [];
// Camera direction of the main image (sent only with extra views).
let mainAz = 0;
let mainEl = 0;
let viewUrls: string[] = [];
// "Paint a mesh" (docs/spec/34-studio-texture-mode.md): the mesh, its orientation, and
// whether the viewer shows its untextured preview (rather than a result).
type Mode = "image" | "mesh";
let mode: Mode = "image";
try {
  if (localStorage.getItem("studio-mode") === "mesh") mode = "mesh";
} catch { /* storage unavailable: start in image mode */ }
interface InputMesh extends MeshFile {
  blob: Blob;
  name: string;
}
let inputMesh: InputMesh | null = null;
let meshUp: MeshUp = "auto";
let meshYaw: MeshYaw = 0;
let previewing = false;
let previewSeq = 0; // a newer preview (or leaving it) makes an older load's result stale
let markerUrl: string | null = null;
let resultCaption = "";
let runningMode: Mode = "image"; // the mode of the job in progress (for its stage text) // the caption of the result in the viewer, restored after a preview

// ---- controls -> params ----
function readParams(): GenParams {
  const res = parseInt(($("ctl-res") as HTMLSelectElement).value, 10);
  const seed = parseInt(($("ctl-seed") as HTMLInputElement).value, 10);
  return {
    resolution: (res === 512 || res === 1536 ? res : 1024) as GenParams["resolution"],
    seed: isNaN(seed) ? DEFAULT_PARAMS.seed : seed,
    bgRemoval: ($("ctl-bg") as HTMLSelectElement).value as GenParams["bgRemoval"],
    uv: ($("ctl-uv") as HTMLSelectElement).value as GenParams["uv"],
    texture: ($("ctl-tex") as HTMLSelectElement).value !== "off",
    watertight: ($("ctl-print") as HTMLSelectElement).value === "on",
    printSize: readNumber("ctl-print-size", DEFAULT_PARAMS.printSize!, 5, 2000),
    printBase: ($("ctl-print-base") as HTMLSelectElement).value,
    printWall: readNumber("ctl-print-wall", DEFAULT_PARAMS.printWall!, 0, 20),
    printInterior: ($("ctl-print-interior") as HTMLSelectElement).value as PrintInterior,
    printColors: parseInt(($("ctl-print-colors") as HTMLSelectElement).value, 10) || 0,
    printColorRes: parseFloat(($("ctl-print-color-res") as HTMLSelectElement).value) || 0,
    printSizeFromFile: ($("ctl-print-size-file") as HTMLInputElement).checked,
    mode,
    meshUp: mode === "mesh" ? meshUp : undefined,
    meshYaw: mode === "mesh" ? meshYaw : undefined,
    blendTemp: readBlendTemp(),
    frontAxis: ($("ctl-front") as HTMLSelectElement).value as GenParams["frontAxis"],
    autoYaw: ($("ctl-yaw") as HTMLSelectElement).value === "auto",
    ssMode: ($("ctl-ss") as HTMLSelectElement).value as GenParams["ssMode"],
    visibility: ($("ctl-vis") as HTMLSelectElement).value === "on",
    mainAz,
    mainEl,
  };
}

/** A number input's value, clamped; the fallback when it is empty or not a number. */
function readNumber(id: string, fallback: number, lo: number, hi: number): number {
  const n = parseFloat(($(id) as HTMLInputElement).value);
  return isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
}

function readBlendTemp(): number {
  const t = parseFloat(($("ctl-blend") as HTMLInputElement).value);
  return isFinite(t) && t > 0 ? t : DEFAULT_PARAMS.blendTemp;
}

function applyParams(p: GenParams): void {
  ($("ctl-res") as HTMLSelectElement).value = String(p.resolution);
  ($("ctl-seed") as HTMLInputElement).value = String(p.seed);
  ($("ctl-bg") as HTMLSelectElement).value = p.bgRemoval;
  ($("ctl-uv") as HTMLSelectElement).value = p.uv;
  // records saved before this option existed have no field: treat as textured
  ($("ctl-tex") as HTMLSelectElement).value = p.texture === false ? "off" : "on";
  ($("ctl-print") as HTMLSelectElement).value = p.watertight ? "on" : "off";
  // records saved before the print options existed have none of these fields
  ($("ctl-print-size") as HTMLInputElement).value = String(p.printSize ?? DEFAULT_PARAMS.printSize);
  ($("ctl-print-base") as HTMLSelectElement).value = p.printBase ?? DEFAULT_PARAMS.printBase!;
  ($("ctl-print-wall") as HTMLInputElement).value = String(p.printWall ?? DEFAULT_PARAMS.printWall);
  ($("ctl-print-interior") as HTMLSelectElement).value = p.printInterior ?? (p.printSeal === 0 ? "hollow" : DEFAULT_PARAMS.printInterior!);
  ($("ctl-print-colors") as HTMLSelectElement).value = String(p.printColors ?? DEFAULT_PARAMS.printColors);
  ($("ctl-print-color-res") as HTMLSelectElement).value = String(p.printColorRes ?? DEFAULT_PARAMS.printColorRes);
  ($("ctl-print-size-file") as HTMLInputElement).checked = p.printSizeFromFile ?? true;
  syncPrintOptions();
  // records saved before multi-view existed have neither field
  ($("ctl-blend") as HTMLInputElement).value = String(p.blendTemp ?? DEFAULT_PARAMS.blendTemp);
  ($("ctl-front") as HTMLSelectElement).value = p.frontAxis ?? DEFAULT_PARAMS.frontAxis;
  ($("ctl-yaw") as HTMLSelectElement).value = p.autoYaw ? "auto" : "given";
  ($("ctl-ss") as HTMLSelectElement).value = p.ssMode ?? "fill";
  ($("ctl-vis") as HTMLSelectElement).value = p.visibility ? "on" : "off";
  mainAz = p.mainAz ?? 0;
  mainEl = p.mainEl ?? 0;
}

// ---- toasts ----
function toast(msg: string, kind: "" | "ok" | "err" = "", action?: { label: string; run: () => void }): void {
  const el = document.createElement("div");
  el.className = "toast" + (kind ? " " + kind : "");
  el.textContent = msg;
  if (action) {
    const b = document.createElement("button");
    b.className = "toast-action";
    b.type = "button";
    b.textContent = action.label;
    b.addEventListener("click", () => {
      el.remove();
      action.run();
    });
    el.append(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), action ? 20000 : kind === "err" ? 8000 : 4000);
}

// ---- input image ----
function setInput(blob: Blob, name: string): void {
  inputImage = blob;
  inputName = name || "input.png";
  inputPreview.src = URL.createObjectURL(blob);
  inputPreview.classList.remove("hidden");
  dropHint.classList.add("hidden");
  renderViews(); // the main-image row shows the new thumbnail
  if (previewing) viewer.setFrontMarker(frontMarker()); // the marker shows the photo
  updateGenerateEnabled();
}

// ---- extra views (multi-view conditioning) ----
// The main image is always the front view (azimuth 0, elevation 0); each extra
// view is another photo of the same object with its own camera direction.
const AZIMUTHS: [number, string][] = [
  [0, "front"], [45, "front-right"], [90, "right"], [135, "back-right"],
  [180, "back"], [225, "back-left"], [270, "left"], [315, "front-left"],
];

/** Guess a view's direction from its file name, else take the next unused one. */
function guessDirection(name: string): { az: number; el: number } {
  const n = name.toLowerCase();
  if (/\btop\b|_top|top_/.test(n)) return { az: 0, el: 90 };
  if (/bottom/.test(n)) return { az: 0, el: -90 };
  if (/back/.test(n)) return { az: 180, el: 0 };
  if (/left/.test(n)) return { az: 270, el: 0 };
  if (/right/.test(n)) return { az: 90, el: 0 };
  const used = new Set([mainAz, ...extraViews.map((v) => v.az)]);
  const az = [180, 90, 270, 45, 135, 225, 315].find((a) => !used.has(a)) ?? 180;
  return { az, el: 0 };
}

function addViews(files: File[]): void {
  for (const f of files) {
    if (!isImage(f)) {
      toast(`"${f.name}" isn't an image — skipped.`, "err");
      continue;
    }
    if (extraViews.length + 1 >= MAX_VIEWS) {
      toast(`At most ${MAX_VIEWS} images (front + ${MAX_VIEWS - 1} views) per generation.`, "err");
      break;
    }
    extraViews.push({ image: f, name: f.name || "view.png", ...guessDirection(f.name || "") });
  }
  renderViews();
}

/** Direction select + elevation input for one image; calls back with clamped values. */
function directionControls(
  az0: number,
  el0: number,
  onAz: (az: number) => void,
  onEl: (el: number) => void,
  allowAuto = false,
): [HTMLSelectElement, HTMLInputElement] {
  const az = document.createElement("select");
  az.title = "Azimuth: where the camera was, going around the object";
  for (const [deg, label] of AZIMUTHS) az.add(new Option(`${label} · ${deg}°`, String(deg)));
  // Estimated by the server from the silhouettes (front/back can be ambiguous on symmetric objects).
  if (allowAuto) az.add(new Option("auto (estimate)", String(AUTO_AZ)));
  if (!AZIMUTHS.some(([deg]) => deg === az0)) az.add(new Option(`${az0}°`, String(az0)));
  az.value = String(az0);
  az.addEventListener("change", () => onAz(parseFloat(az.value)));

  const el = document.createElement("input");
  el.type = "number";
  el.min = "-90";
  el.max = "90";
  el.step = "15";
  el.value = String(el0);
  el.title = "Elevation in degrees (90 = from above, -90 = from below)";
  el.addEventListener("change", () => {
    const e = Math.max(-90, Math.min(90, parseFloat(el.value) || 0));
    el.value = String(e);
    onEl(e);
  });
  return [az, el];
}

function azLabel(az: number): string {
  if (az === AUTO_AZ) return "auto";
  return `${AZIMUTHS.find(([deg]) => deg === az)?.[1] ?? "view"} · ${az}°`;
}

function updateFrontTag(): void {
  frontTag.textContent = `main · ${azLabel(mainAz)}${mainEl ? `, ${mainEl}° up` : ""}`;
  frontTag.classList.toggle("hidden", extraViews.length === 0 || !inputImage);
}

function renderViews(): void {
  viewUrls.forEach((u) => URL.revokeObjectURL(u));
  viewUrls = [];
  viewsList.innerHTML = "";
  if (extraViews.length) {
    viewsList.innerHTML = `<div class="vhead"><span></span><span>direction</span><span>elev °</span><span></span></div>`;
    // The main image's own direction (default front, level): the other views are
    // placed relative to the object the same way.
    const row = document.createElement("div");
    row.className = "vrow vmain";
    row.title = `${inputName} (main image)`;
    const img = document.createElement("img");
    if (inputImage) {
      const url = URL.createObjectURL(inputImage);
      viewUrls.push(url);
      img.src = url;
    }
    img.alt = "main image";
    row.appendChild(img);
    const [az, el] = directionControls(mainAz, mainEl,
      (a) => { mainAz = a; updateFrontTag(); },
      (e) => { mainEl = e; updateFrontTag(); });
    row.append(az, el);
    const tag = document.createElement("span");
    tag.className = "vmain-tag";
    tag.textContent = "main";
    row.appendChild(tag);
    viewsList.appendChild(row);
  }
  extraViews.forEach((v, i) => {
    const row = document.createElement("div");
    row.className = "vrow";
    row.title = v.name;

    const img = document.createElement("img");
    const url = URL.createObjectURL(v.image);
    viewUrls.push(url);
    img.src = url;
    img.alt = v.name;
    row.appendChild(img);

    const [az, el] = directionControls(v.az, v.el,
      (a) => (extraViews[i].az = a),
      (e) => (extraViews[i].el = e),
      true);
    row.append(az, el);

    const del = document.createElement("button");
    del.className = "vdel";
    del.type = "button";
    del.textContent = "✕";
    del.title = "Remove this view";
    del.addEventListener("click", () => {
      extraViews.splice(i, 1);
      renderViews();
    });
    row.appendChild(del);

    viewsList.appendChild(row);
  });
  const multi = extraViews.length > 0;
  updateFrontTag();
  clearViewsBtn.classList.toggle("hidden", !multi);
  mvAdv.classList.toggle("hidden", !multi);
  addViewBtn.disabled = extraViews.length + 1 >= MAX_VIEWS;
  const verb = mode === "mesh" ? "Paint mesh" : "Generate 3D";
  generateBtn.textContent = multi ? `${verb} · ${extraViews.length + 1} views` : verb;
}

addViewBtn.addEventListener("click", () => viewInput.click());
viewInput.addEventListener("change", () => {
  addViews(Array.from(viewInput.files ?? []));
  viewInput.value = "";
});
clearViewsBtn.addEventListener("click", () => {
  extraViews = [];
  renderViews();
});
["dragenter", "dragover"].forEach((ev) =>
  viewsEl.addEventListener(ev, (e) => {
    e.preventDefault();
    viewsEl.classList.add("drag");
  }),
);
viewsEl.addEventListener("dragleave", (e) => {
  if (!viewsEl.contains((e as DragEvent).relatedTarget as Node | null)) viewsEl.classList.remove("drag");
});
viewsEl.addEventListener("drop", (e) => {
  e.preventDefault();
  viewsEl.classList.remove("drag");
  takeFiles(Array.from((e as DragEvent).dataTransfer?.files ?? []), "views");
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if ((e as KeyboardEvent).key === "Enter") fileInput.click();
});
fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files ?? []);
  if (files[0]) setInput(files[0], files[0].name);
  // several files picked at once: the first is the front, the rest extra views
  if (files.length > 1) addViews(files.slice(1));
  fileInput.value = "";
});
// HTML5 drag-and-drop. In Tauri this needs `dragDropEnabled: false` on the
// window (tauri.conf.json): otherwise the webview's native file-drop handler
// swallows the drop on Windows/WebView2 and the page never sees the files.

// A file dropped anywhere outside the drop zone must not make the webview
// navigate to it (which would replace the whole app with the image). A mesh dropped
// anywhere is taken (and switches to "Paint a mesh"); the drop zones handle their own.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  if (e.defaultPrevented) return; // a drop zone took it
  e.preventDefault();
  const files = Array.from(e.dataTransfer?.files ?? []);
  if (files.some(isMesh)) takeFiles(files, "main");
});

const IMAGE_EXT = /\.(png|jpe?g|webp|bmp|gif|tiff?)$/i;
const isImage = (f: File) => f.type.startsWith("image/") || IMAGE_EXT.test(f.name);
const isMesh = (f: File) => MESH_EXT.test(f.name);

/**
 * Dropped or picked files: a mesh becomes the mesh to paint (switching to "Paint a mesh"),
 * images become the main image and extra views ("views": all of them extra views).
 */
function takeFiles(files: File[], where: "main" | "views"): void {
  const mesh = files.find(isMesh);
  const images = files.filter((f) => !isMesh(f) && isImage(f));
  for (const f of files)
    if (!isMesh(f) && !isImage(f)) toast(`"${f.name}" isn't an image or a mesh (STL, OBJ, GLB) — skipped.`, "err");
  if (mesh) void setMesh(mesh);
  if (!images.length) return;
  if (where === "views") return addViews(images);
  setInput(images[0], images[0].name);
  // several images at once: the first is the front, the rest extra views
  if (images.length > 1) addViews(images.slice(1));
}

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  }),
);
// dragleave also fires when the pointer moves onto a child (the preview <img>),
// which made the highlight flicker; only clear it when actually leaving the zone.
dropzone.addEventListener("dragleave", (e) => {
  if (!dropzone.contains((e as DragEvent).relatedTarget as Node | null)) dropzone.classList.remove("drag");
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  takeFiles(Array.from((e as DragEvent).dataTransfer?.files ?? []), "main");
});
window.addEventListener("paste", async (e: ClipboardEvent) => {
  for (const item of e.clipboardData?.items ?? []) {
    const image = item.type.startsWith("image/") && item.getAsFile();
    if (image) return setInput(image, image.name || "pasted.png");
  }
  // WebKitGTK can omit image data from ClipboardEvent. Since paste was
  // user-triggered, try reading the clipboard item list for an image.
  const items = await navigator.clipboard.read().catch(() => []);
  for (const item of items) {
    const type = item.types.find((type) => type.startsWith("image/"));
    if (type) {
      const image = await item.getType(type).catch(() => null);
      if (image) return setInput(image, "pasted.png");
    }
  }
});

// ---- "Paint a mesh": mode, mesh, orientation preview (docs/spec/34-studio-texture-mode.md) ----

/** The panel for `m`, without touching the viewer. */
function applyModeUi(m: Mode): void {
  if (m !== mode) {
    saveModeSettings();
    mode = m;
    restoreModeSettings(m);
  }
  syncPrintOptions();
  try { localStorage.setItem("studio-mode", m); } catch { /* ignore */ }
  for (const b of modeBtns) b.setAttribute("aria-selected", String(b.dataset.mode === m));
  meshZone.classList.toggle("hidden", m !== "mesh");
  $("dropzone-text").textContent = m === "mesh" ? "Drop a photo of the object" : "Drop an image here";
  renderViews(); // the button's verb
  updateGenerateEnabled();
}

function setMode(m: Mode): void {
  applyModeUi(m);
  if (m === "mesh" && inputMesh) void showMeshPreview();
  else leavePreview(true);
}
for (const b of modeBtns) b.addEventListener("click", () => setMode(b.dataset.mode as Mode));

async function setMesh(file: File): Promise<void> {
  let m: MeshFile;
  try {
    m = await readMesh(await file.arrayBuffer(), file.name);
  } catch (e) {
    toast((e as Error).message, "err");
    return;
  }
  inputMesh = { ...m, blob: file, name: file.name };
  meshUp = "auto";
  meshYaw = 0;
  renderMeshInfo();
  if (mode !== "mesh") applyModeUi("mesh");
  await showMeshPreview();
}

function clearMesh(): void {
  inputMesh = null;
  renderMeshInfo();
  leavePreview(true);
  updateGenerateEnabled();
}

/** The mesh line: name, triangles, and the size as turned (width × depth × height). */
function renderMeshInfo(): void {
  meshHint.classList.toggle("hidden", !!inputMesh);
  meshInfo.classList.toggle("hidden", !inputMesh);
  syncPrintOptions(); // "from file" shows the mesh's size
  if (!inputMesh) return;
  $("mesh-name").textContent = inputMesh.name;
  const mm = (m: number) => (m < 0.05 ? "0" : m.toFixed(m < 10 ? 1 : 0));
  const d = applyMap(previewMap(inputMesh.kind, meshUp, meshYaw), inputMesh.size).map(Math.abs);
  // the size kept on one line (non-breaking spaces)
  $("mesh-stats").textContent = `${inputMesh.triangles.toLocaleString()} triangles · ` +
    `${[mm(d[0]), mm(d[2]), mm(d[1])].join(" × ")} mm`;
  $("mesh-stats").title = "width × depth × height, as turned (STL / OBJ units read as millimetres, GLB as metres)";
}

/** The pin on the preview's front face: the photo's thumbnail, labelled. */
function frontMarker(): HTMLElement {
  const el = document.createElement("div");
  el.className = "front-marker";
  if (markerUrl) URL.revokeObjectURL(markerUrl);
  markerUrl = null;
  if (inputImage) {
    markerUrl = URL.createObjectURL(inputImage);
    const img = document.createElement("img");
    img.src = markerUrl;
    img.alt = "";
    el.append(img);
  }
  const label = document.createElement("span");
  label.textContent = inputImage ? "photo side" : "front";
  el.append(label);
  return el;
}

/** Show the mesh untextured, turned as set, with the orientation controls and the marker. */
async function showMeshPreview(): Promise<void> {
  if (!inputMesh) return;
  const seq = ++previewSeq;
  previewing = true;
  orientBar.classList.remove("hidden");
  showPrintReport(null);
  setViewerTools(false);
  resetViewBtn.disabled = false;
  syncOrientControls();
  const { kind, preview } = inputMesh;
  const glb = orientGlb(preview, quaternion(previewMap(kind, meshUp, meshYaw)));
  try {
    await viewer.load(new Blob([glb], { type: "model/gltf-binary" }));
  } catch (e) {
    if (seq === previewSeq) toast(`The mesh preview couldn't render: ${(e as Error).message}`, "err");
    return;
  }
  if (seq !== previewSeq) return;
  viewer.resetView();
  renderMeshInfo();
  viewer.setFrontMarker(frontMarker());
  viewerCaption.textContent = `${inputMesh.name} · its ${frontAxis(kind, meshUp, meshYaw)} side faces you`;
}

/** Back from the preview; `restore` puts the last result (or nothing) back in the viewer. */
function leavePreview(restore: boolean): void {
  if (!previewing) return;
  previewing = false;
  ++previewSeq;
  orientBar.classList.add("hidden");
  viewer.setFrontMarker(null);
  if (!restore) return;
  if (currentGlb) {
    viewer.load(currentGlb).catch(() => {});
    setViewerTools(true);
    viewerCaption.textContent = resultCaption;
  } else {
    viewer.clear();
    setViewerTools(false);
    viewerCaption.textContent = "";
  }
}

function setResultCaption(text: string): void {
  resultCaption = text;
  viewerCaption.textContent = text;
}

function syncOrientControls(): void {
  if (!inputMesh) return;
  const native = nativeUp(inputMesh.kind).toUpperCase();
  orientUp.replaceChildren(...MESH_UPS.map((u) =>
    new Option(u === "auto" ? `Auto · ${native}` : u.toUpperCase().replace("-", "−"), u)));
  orientUp.value = meshUp;
  $("turn-value").textContent = `${meshYaw}°`;
}

orientUp.addEventListener("change", () => {
  meshUp = orientUp.value as MeshUp;
  void showMeshPreview();
});
const turn = (deg: number) => {
  meshYaw = (((meshYaw + deg) % 360) + 360) % 360 as MeshYaw;
  void showMeshPreview();
};
$("turn-ccw").addEventListener("click", () => turn(90));
$("turn-cw").addEventListener("click", () => turn(-90));

meshZone.addEventListener("click", (e) => {
  if ((e.target as HTMLElement).closest("#mesh-clear, #mesh-change")) return;
  if (inputMesh) void showMeshPreview(); // back to the orientation
  else meshInput.click();
});
meshZone.addEventListener("keydown", (e) => {
  if ((e as KeyboardEvent).key === "Enter") meshInput.click();
});
$("mesh-change").addEventListener("click", () => meshInput.click());
$("mesh-clear").addEventListener("click", clearMesh);
meshInput.addEventListener("change", () => {
  const f = meshInput.files?.[0];
  if (f) void setMesh(f);
  meshInput.value = "";
});
["dragenter", "dragover"].forEach((ev) =>
  meshZone.addEventListener(ev, (e) => {
    e.preventDefault();
    meshZone.classList.add("drag");
  }),
);
meshZone.addEventListener("dragleave", (e) => {
  if (!meshZone.contains((e as DragEvent).relatedTarget as Node | null)) meshZone.classList.remove("drag");
});
meshZone.addEventListener("drop", (e) => {
  e.preventDefault();
  meshZone.classList.remove("drag");
  takeFiles(Array.from((e as DragEvent).dataTransfer?.files ?? []), "main");
});

// ---- generate ----
function fmtElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function updateGenerateEnabled(): void {
  generateBtn.disabled = !(serverOnline && inputImage && !generating && (mode === "image" || inputMesh));
  generateBtn.title = mode === "mesh" && !inputMesh ? "Drop the mesh to paint first" : "";
}

function describeJob(s: JobStatus | null): string {
  if (!s) return "server unreachable — retrying (the job keeps running on the server)…";
  if (s.status === "queued") {
    const ahead = (s.position ?? 1) - 1;
    return ahead > 0 ? `queued — ${ahead} job${ahead > 1 ? "s" : ""} ahead` : "queued — starts next";
  }
  if (s.status === "running") return "running on the server…";
  return s.status;
}

/**
 * Run one generation (new, or resumed after a reload) with the progress UI.
 * `rec` holds everything captured when it started, so browsing the gallery or
 * changing inputs meanwhile can't leak into the saved result.
 */
async function runJob(
  rec: GenRecord,
  work: (signal: AbortSignal, hooks: GenerateHooks) => Promise<GenerateResult>,
): Promise<void> {
  generating = true;
  runningMode = rec.params.mode ?? "image";
  browsedAway = false;
  updateGenerateEnabled();
  progress.classList.remove("hidden");
  progressStage.textContent = "sending…";
  const started = Date.now();
  progressElapsed.textContent = "0:00";
  elapsedTimer = window.setInterval(() => {
    progressElapsed.textContent = fmtElapsed(Date.now() - started);
  }, 1000);
  abort = new AbortController();
  let running = false;
  try {
    const { glb, model3mf, report } = await work(abort.signal, {
      onJob: async (id) => {
        // A pending gallery entry: it survives a reload, which resumes the job.
        rec.jobId = id;
        rec.pending = true;
        await put(rec);
        await refreshGallery();
      },
      onStatus: (st) => {
        if (st?.status === "running") running = true;
        progressStage.textContent = describeJob(st);
      },
    });
    await finishRecord(rec, glb, model3mf, report);
  } catch (e) {
    await removeRecord(rec.id);
    await refreshGallery();
    if (abort?.signal.aborted) {
      toast(running
        ? "Cancelled — the server finishes the run already in progress, but its result is discarded."
        : "Generation cancelled");
    } else {
      const msg = (e as Error).message || "generation failed";
      // A mesh too dense for the 1024 texture flow: offer the 512 one.
      if (rec.params.mode === "mesh" && /--res 512/.test(msg)) {
        const m = /(\d+) texture tokens .*--max-tokens (\d+)/.exec(msg);
        const n = (s: string) => Number(s).toLocaleString();
        toast("This mesh is too detailed to paint at 1024" +
          (m ? `: it needs ${n(m[1])} texture tokens, over this server's limit of ${n(m[2])}.` : "."), "err", {
          label: "Retry at 512",
          run: () => retryAt512(rec),
        });
      } else toast(msg, "err");
    }
  } finally {
    generating = false;
    if (elapsedTimer) window.clearInterval(elapsedTimer);
    progress.classList.add("hidden");
    updateGenerateEnabled();
  }
}

/** Paint the same mesh, photo and settings again with the 512 texture flow. */
function retryAt512(failed: GenRecord): void {
  if (generating) return toast("Wait for the current run to finish first.", "err");
  if (!failed.mesh || !failed.meshName) return;
  if (mode === "mesh") {
    ($("ctl-res") as HTMLSelectElement).value = "512";
    saveModeSettings();
  }
  const rec: GenRecord = {
    ...failed,
    id: newId(),
    ts: Date.now(),
    params: { ...failed.params, resolution: 512 },
    glb: null,
    thumb: null,
    pending: undefined,
    jobId: undefined,
  };
  const mesh = { blob: failed.mesh, name: failed.meshName };
  void runJob(rec, (signal, hooks) => generate(rec.input, rec.params, rec.views ?? [], signal, hooks, mesh));
}

/** Store a finished generation; show it unless the user is browsing another item. */
async function finishRecord(rec: GenRecord, glb: Blob, model3mf?: Blob, report?: PrintReport): Promise<void> {
  rec.glb = glb;
  rec.model3mf = model3mf ?? null;
  rec.report = report ?? null;
  rec.pending = false;
  delete rec.jobId;
  const nViews = rec.views?.length ?? 0;
  const stem = rec.name.replace(/\.[^.]+$/, "") || "model";

  // 1) Write the GLB to the output folder FIRST. This on-disk file is the real
  // deliverable and must survive any later failure — a dead gallery DB (some
  // WebKitGTK builds can't open IndexedDB) or a WebGL/model-viewer crash must
  // never cost the user a successful generation.
  let savedPath: string | null = null;
  if (isTauri()) {
    try {
      const bytes = new Uint8Array(await glb.arrayBuffer());
      const mv = nViews ? `_mv${nViews + 1}` : "";
      const painted = rec.params.mode === "mesh" ? "_painted" : "";
      const fname = `${stem}${painted}_${rec.params.resolution}${mv}_seed${rec.params.seed}_${rec.id}.glb`;
      savedPath = await saveToOutputDir(fname, bytes);
      if (model3mf)
        await saveToOutputDir(fname.replace(/\.glb$/, ".3mf"), new Uint8Array(await model3mf.arrayBuffer()));
    } catch (e) {
      toast(`Auto-save to output folder failed: ${(e as Error).message}`, "err");
    }
  }

  // 2) Add it to the gallery. The store falls back to in-memory if IndexedDB
  // is unavailable, so this never throws and never blocks the save above.
  await put(rec);
  if (isEphemeral() && !warnedEphemeral) {
    warnedEphemeral = true;
    toast(
      "Gallery won't persist across restarts (IndexedDB unavailable on this system) — " +
        "but every generation is still saved to your output folder.",
      "err",
    );
  }
  if (browsedAway) {
    toast(`"${rec.name}" finished — it's in the gallery${savedPath ? ` and saved to ${savedPath}` : ""}.`, "ok");
    await refreshGallery();
    return;
  }
  toast(savedPath ? `Saved to ${savedPath}` : rec.params.mode === "mesh" ? "Mesh painted" : "Generation complete", "ok");
  leavePreview(false);
  currentGlb = glb;
  current3mf = model3mf ?? null;
  showPrintReport(rec.report, rec.params.mode === "mesh");
  currentName = stem + (rec.params.mode === "mesh" ? "_painted" : "");
  activeId = rec.id;
  setViewerTools(true);
  const size = glb.size < 1e6 ? `${Math.max(1, Math.round(glb.size / 1e3))} KB` : `${(glb.size / 1e6).toFixed(1)} MB`;
  setResultCaption((rec.params.mode === "mesh" ? `${rec.name} · painted at ` : "") +
    `${rec.params.resolution}${viewsLabel(nViews)} · seed ${rec.params.seed} · ${size}`);
  await refreshGallery();

  // 3) Best-effort 3D preview + gallery thumbnail — never blocks the save above.
  try {
    await viewer.load(glb);
    const thumb = await viewer.thumbnail();
    if (thumb) {
      rec.thumb = thumb;
      await put(rec);
      await refreshGallery();
    }
  } catch (e) {
    toast(`3D preview couldn't render (your result is still saved): ${(e as Error).message}`, "err");
  }
}

async function doGenerate(): Promise<void> {
  if (!inputImage || generating) return;
  const paint = mode === "mesh";
  if (paint && !inputMesh) return;
  const params = readParams();
  const views = extraViews.map((v) => ({ ...v }));
  const rec: GenRecord = {
    id: newId(),
    ts: Date.now(),
    name: paint ? inputMesh!.name : inputName,
    params,
    input: inputImage,
    views: views.length ? views : undefined,
    mesh: paint ? inputMesh!.blob : undefined,
    meshName: paint ? inputMesh!.name : undefined,
    glb: null,
    thumb: null,
  };
  const mesh = rec.mesh ? { blob: rec.mesh, name: rec.meshName! } : undefined;
  await runJob(rec, (signal, hooks) => generate(rec.input, params, views, signal, hooks, mesh));
}

/** Pick up jobs that were still running on the server when the page was closed. */
async function resumePending(): Promise<void> {
  for (const r of (await all()).filter((x) => x.pending)) {
    if (!r.jobId || generating) {
      if (!r.jobId) await removeRecord(r.id);
      continue;
    }
    toast(`Resuming "${r.name}" — still on the server`);
    const id = r.jobId;
    await runJob(r, (signal, hooks) => waitForJob(id, signal, hooks));
  }
  await refreshGallery();
}

generateBtn.addEventListener("click", doGenerate);
cancelBtn.addEventListener("click", () => abort?.abort());

// ---- viewer tools ----
function setViewerTools(on: boolean): void {
  resetViewBtn.disabled = !on;
  exportBtn.disabled = !on;
  exportMenuBtn.disabled = !on;
  if (!on) closeExportMenu();
}
resetViewBtn.addEventListener("click", () => viewer.resetView());

// ---- 3D print: options follow the toggle; the report follows the model in the viewer ----
const printOpts = $("print-opts");
const sizeInput = $<HTMLInputElement>("ctl-print-size");
const sizeFromFile = $<HTMLInputElement>("ctl-print-size-file");

/** The mesh's longest side in mm (what the 3MF gets "from file"), or null without a mesh. */
function meshLongestMm(): number | null {
  return inputMesh ? Math.max(...inputMesh.size) : null;
}

/**
 * The print and resolution controls for the mode: "Paint a mesh" writes a 3MF of the mesh as
 * it is (no flat base, wall or interior: nothing is rebuilt), at the file's size by default,
 * and paints at 512 or 1024 only.
 */
function syncPrintOptions(): void {
  const mesh = mode === "mesh";
  printOpts.classList.toggle("hidden", ($("ctl-print") as HTMLSelectElement).value !== "on");
  $("ctl-print-label").textContent = mesh ? "3MF" : "3D print";
  $("ctl-print-on").textContent = mesh ? "3MF of this mesh" : "watertight solid";
  $("ctl-print-wrap").title = mesh
    ? "Also write a 3MF of your mesh's own triangles, painted. Nothing is rebuilt: what you dropped is what prints."
    : "Rebuild the mesh as one closed, manifold solid a slicer accepts. Slower.";
  for (const id of ["print-base-wrap", "print-wall-wrap", "print-interior-wrap", "ctl-tex-wrap"])
    $(id).classList.toggle("hidden", mesh);
  $("print-size-file-wrap").classList.toggle("hidden", !mesh);
  const fromFile = mesh && sizeFromFile.checked;
  if (fromFile) {
    if (!sizeInput.disabled) sizeInput.dataset.manual = sizeInput.value;
    sizeInput.disabled = true;
    const mm = meshLongestMm();
    sizeInput.value = mm ? mm.toFixed(mm < 100 ? 1 : 0) : "";
    sizeInput.placeholder = "the mesh's size";
  } else if (sizeInput.disabled) {
    sizeInput.disabled = false;
    sizeInput.value = sizeInput.dataset.manual || String(DEFAULT_PARAMS.printSize);
  }
  const colors = parseInt(($("ctl-print-colors") as HTMLSelectElement).value, 10) || 0;
  $("print-color-res-wrap").classList.toggle("hidden", colors === 0);
  const res = $<HTMLSelectElement>("ctl-res");
  ($("res-1536") as HTMLOptionElement).hidden = mesh;
  if (mesh && res.value === "1536") res.value = "1024";
  $("res-512").textContent = mesh ? "512 · fast (default)" : "512 · light";
  $("res-1024").textContent = mesh ? "1024 · detailed, slower" : "1024 · cascade";
}

// Each mode keeps its own resolution and print settings (a per-viewer convenience).
interface ModeSettings {
  res: string;
  print: string;
  size: string;
  sizeFromFile: boolean;
  colors: string;
  colorRes: string;
}
const MODE_DEFAULTS: Record<Mode, ModeSettings> = {
  image: { res: "1024", print: "off", size: "100", sizeFromFile: true, colors: "0", colorRes: "1" },
  mesh: { res: "512", print: "off", size: "100", sizeFromFile: true, colors: "0", colorRes: "1" },
};
let modeSettings: Partial<Record<Mode, ModeSettings>> = {};
try {
  modeSettings = JSON.parse(localStorage.getItem("studio-mode-settings") || "{}") || {};
} catch { /* storage unavailable or corrupt: defaults */ }

function saveModeSettings(): void {
  modeSettings[mode] = {
    res: ($("ctl-res") as HTMLSelectElement).value,
    print: ($("ctl-print") as HTMLSelectElement).value,
    size: sizeInput.disabled ? sizeInput.dataset.manual || String(DEFAULT_PARAMS.printSize) : sizeInput.value,
    sizeFromFile: sizeFromFile.checked,
    colors: ($("ctl-print-colors") as HTMLSelectElement).value,
    colorRes: ($("ctl-print-color-res") as HTMLSelectElement).value,
  };
  try { localStorage.setItem("studio-mode-settings", JSON.stringify(modeSettings)); } catch { /* ignore */ }
}

function restoreModeSettings(m: Mode): void {
  const s = { ...MODE_DEFAULTS[m], ...modeSettings[m] };
  ($("ctl-res") as HTMLSelectElement).value = s.res;
  ($("ctl-print") as HTMLSelectElement).value = s.print;
  sizeInput.disabled = false;
  sizeInput.value = s.size;
  sizeFromFile.checked = s.sizeFromFile;
  ($("ctl-print-colors") as HTMLSelectElement).value = s.colors;
  ($("ctl-print-color-res") as HTMLSelectElement).value = s.colorRes;
}

for (const id of ["ctl-res", "ctl-print", "ctl-print-size", "ctl-print-size-file", "ctl-print-colors", "ctl-print-color-res"])
  $(id).addEventListener("change", () => {
    syncPrintOptions();
    saveModeSettings();
  });
syncPrintOptions();

const reportEl = $<HTMLDetailsElement>("print-report");
/** The server words its advice as CLI flags; say it with this app's control names. */
function studioWording(warning: string, mesh: boolean): string {
  if (mesh)
    // A painted mesh is printed as it is: the advice is about the mesh itself.
    return warning
      .replace(/^not watertight/, "your mesh is not watertight")
      .replace("raise --print-wall or --print-size",
               'thicken them in your CAD tool, or set a larger "Print size"')
      .replace(/use --print-base( auto)?( for a wider base)?$/,
               "the mesh prints as it is: turn it in your slicer or add a base in your CAD tool");
  return warning
    .replace("raise --print-wall or --print-size", 'raise "Minimum wall" or "Print size"')
    .replace("use --print-base auto for a wider base", 'set "Flat base" to auto for a wider base')
    .replace("use --print-base auto", 'set "Flat base" to auto')
    .replace("use --print-base", 'choose a "Flat base"');
}
function showPrintReport(report: PrintReport | null | undefined, mesh = false): void {
  reportEl.classList.toggle("hidden", !report);
  if (!report) return;
  const badge = $("print-report-badge");
  badge.textContent = report.ok ? "Ready to print" : `${report.warnings.length} warning${report.warnings.length === 1 ? "" : "s"}`;
  badge.classList.toggle("warn", !report.ok);
  const [w, d, h] = report.size_mm;
  $("print-report-line").textContent =
    `${w.toFixed(0)} × ${d.toFixed(0)} × ${h.toFixed(0)} mm · ${report.volume_cm3.toFixed(1)} cm³`;

  const rows = $("print-report-rows");
  rows.replaceChildren();
  const row = (label: string, value: string, bad = false) => {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    dd.classList.toggle("bad", bad);
    rows.append(dt, dd);
  };
  row("Watertight", report.watertight ? `yes · ${report.shells} shell${report.shells === 1 ? "" : "s"}` : "no",
      !report.watertight || report.shells > 1);
  row("Walls",
      `thinnest 1% of the surface is ${report.wall.at_least ? "at least " : ""}${report.wall.thinnest_1pct_mm.toFixed(2)} mm · ` +
      `${report.wall.thin_area_pct.toFixed(1)}% under ${report.wall.target_mm.toFixed(2)} mm (fine relief included)`,
      report.wall.thinnest_1pct_mm < 0.75 * report.wall.target_mm);
  row("Base",
      !report.base.flat ? "not flat" :
      `flat · ${report.base.contact_mm2.toFixed(0)} mm² · ` +
      (report.base.stands ? `tips over at ${report.base.tip_angle_deg.toFixed(0)}°` : "falls over"),
      !report.base.flat || !report.base.stands || report.base.tip_angle_deg < 10);
  row("Overhangs", `${report.overhang_pct.toFixed(0)}% of the surface steeper than 45°`, report.overhang_pct > 25);
  row("Triangles", report.triangles.toLocaleString());
  if (report.colors && report.colors.palette.length) {
    // One swatch per filament slot, in slot order, with its share of the surface on hover.
    const dt = document.createElement("dt");
    dt.textContent = "Colours";
    const dd = document.createElement("dd");
    const note = document.createElement("div");
    note.textContent = `${report.colors.palette.length} filaments · ${report.colors.patches.toLocaleString()} patches`;
    const strip = document.createElement("div");
    strip.className = "swatches";
    for (const c of report.colors.palette) {
      const s = document.createElement("span");
      s.className = "swatch";
      s.style.background = c.hex;
      s.title = `slot ${c.slot} · ${c.hex} · ${c.share_pct.toFixed(1)}%`;
      strip.append(s);
    }
    dd.append(note, strip);
    rows.append(dt, dd);
  }

  const list = $("print-report-warnings");
  list.replaceChildren(...report.warnings.map((text) => {
    const li = document.createElement("li");
    li.textContent = studioWording(text, mesh);
    return li;
  }));
  reportEl.open = !report.ok; // warnings open by themselves; a clean report stays one line
}

// ---- export: | Export to GLB | v | ----
// The main button exports in the chosen format; the arrow opens the format menu. The choice
// is a per-user convenience, so it lives in localStorage (and works without it).
type ExportFormat = "glb" | "3mf";
const EXPORT_LABEL: Record<ExportFormat, string> = { glb: "GLB", "3mf": "3MF" };
let exportFormat: ExportFormat = "glb";
try {
  if (localStorage.getItem("export-format") === "3mf") exportFormat = "3mf";
} catch { /* storage unavailable: keep the default */ }

function exportItems(): HTMLButtonElement[] {
  return Array.from(exportMenu.querySelectorAll<HTMLButtonElement>("button[data-format]"));
}
function setExportFormat(f: ExportFormat): void {
  exportFormat = f;
  exportBtn.textContent = `Export to ${EXPORT_LABEL[f]}`;
  for (const item of exportItems()) item.setAttribute("aria-checked", String(item.dataset.format === f));
  try { localStorage.setItem("export-format", f); } catch { /* ignore */ }
}
function closeExportMenu(): void {
  exportMenu.classList.add("hidden");
  exportMenuBtn.setAttribute("aria-expanded", "false");
}
function openExportMenu(): void {
  exportMenu.classList.remove("hidden");
  exportMenuBtn.setAttribute("aria-expanded", "true");
  (exportItems().find((i) => i.dataset.format === exportFormat) ?? exportItems()[0])?.focus();
}
setExportFormat(exportFormat);

exportMenuBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  if (exportMenu.classList.contains("hidden")) openExportMenu();
  else closeExportMenu();
});
exportMenu.addEventListener("click", (e) => {
  const item = (e.target as HTMLElement).closest<HTMLButtonElement>("button[data-format]");
  if (!item) return;
  setExportFormat(item.dataset.format as ExportFormat);
  closeExportMenu();
  exportBtn.focus();
});
exportMenu.addEventListener("keydown", (e) => {
  const items = exportItems();
  const at = items.indexOf(document.activeElement as HTMLButtonElement);
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    items[(at + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length]?.focus();
  } else if (e.key === "Escape") {
    closeExportMenu();
    exportMenuBtn.focus();
  }
});
document.addEventListener("click", (e) => {
  if (!exportMenu.contains(e.target as Node)) closeExportMenu();
});

exportBtn.addEventListener("click", async () => {
  if (!currentGlb) return;
  if (exportFormat === "3mf" && !current3mf) {
    toast(mode === "mesh"
      ? 'This model has no 3MF. Set "3MF" to "3MF of this mesh" and paint again.'
      : 'This model has no 3MF. Set "3D print" to "watertight solid" and generate again.', "err");
    return;
  }
  const blob = exportFormat === "3mf" ? current3mf! : currentGlb;
  const ok = await saveBytes(`${currentName}.${exportFormat}`, new Uint8Array(await blob.arrayBuffer()));
  if (ok) toast("Saved", "ok");
});

/** " · N views" for a generation with extra views, "" for a single image. */
function viewsLabel(extra: number): string {
  return extra ? ` · ${extra + 1} views` : "";
}

// ---- gallery ----
async function loadRecord(id: string): Promise<void> {
  const rec = await getRecord(id);
  if (!rec) return;
  if (rec.pending || !rec.glb) {
    toast("Still generating on the server — it opens here when it's done.");
    return;
  }
  if (generating) browsedAway = true;
  leavePreview(false);
  try {
    await viewer.load(rec.glb);
  } catch (e) {
    toast((e as Error).message, "err");
    return;
  }
  const recMode = rec.params.mode ?? "image";
  applyModeUi(recMode);
  if (recMode === "mesh" && rec.mesh && rec.meshName) {
    // The painted mesh's own file and orientation, ready to paint again.
    try {
      const m = await readMesh(await rec.mesh.arrayBuffer(), rec.meshName);
      inputMesh = { ...m, blob: rec.mesh, name: rec.meshName };
      meshUp = rec.params.meshUp ?? "auto";
      meshYaw = rec.params.meshYaw ?? 0;
    } catch {
      inputMesh = null;
    }
    renderMeshInfo();
  }
  inputImage = rec.input;
  inputName = recMode === "mesh" ? "photo.png" : rec.name;
  inputPreview.src = URL.createObjectURL(rec.input);
  inputPreview.classList.remove("hidden");
  dropHint.classList.add("hidden");
  applyParams(rec.params);
  extraViews = (rec.views ?? []).map((v) => ({ ...v }));
  renderViews();
  currentGlb = rec.glb;
  current3mf = rec.model3mf ?? null;
  showPrintReport(rec.report, recMode === "mesh");
  currentName = (rec.name.replace(/\.[^.]+$/, "") || "model") + (recMode === "mesh" ? "_painted" : "");
  activeId = rec.id;
  setViewerTools(true);
  setResultCaption(`${rec.name} · ${recMode === "mesh" ? "painted at " : ""}${rec.params.resolution}${viewsLabel(rec.views?.length ?? 0)} · seed ${rec.params.seed}`);
  updateGenerateEnabled();
  await refreshGallery();
}

async function refreshGallery(): Promise<void> {
  galleryUrls.forEach((u) => URL.revokeObjectURL(u));
  galleryUrls = [];
  const recs = await all();
  galleryEl.innerHTML = "";
  if (!recs.length) {
    galleryEl.innerHTML = `<div class="gallery-empty">No generations yet.</div>`;
    return;
  }
  for (const r of recs) {
    const item = document.createElement("div");
    item.className = "gitem" + (r.id === activeId ? " active" : "") + (r.pending ? " pending" : "");
    item.title = `${r.name} · ${new Date(r.ts).toLocaleString()}`;

    const img = document.createElement("img");
    const src = r.thumb ?? r.input;
    const url = URL.createObjectURL(src);
    galleryUrls.push(url);
    img.src = url;
    item.appendChild(img);

    const meta = document.createElement("div");
    meta.className = "gmeta";
    meta.textContent = r.pending
      ? "generating…"
      : `${r.params.mode === "mesh" ? "mesh · " : ""}${r.params.resolution}${r.views?.length ? ` · ${r.views.length + 1}v` : ""}`;
    item.appendChild(meta);

    const delBtn = document.createElement("button");
    delBtn.className = "gdel";
    delBtn.textContent = "✕";
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      await removeRecord(r.id);
      if (activeId === r.id) activeId = null;
      await refreshGallery();
    });
    if (!r.pending) item.appendChild(delBtn);

    item.addEventListener("click", () => loadRecord(r.id));
    galleryEl.appendChild(item);
  }
}

clearGalleryBtn.addEventListener("click", async () => {
  if (!confirm("Delete all saved generations?")) return;
  await clearStore();
  activeId = null;
  await refreshGallery();
});

// ---- settings modal ----
const modal = $("settings-modal");
async function openSettings(): Promise<void> {
  await renderSettings($("settings-body"), () => {
    pollHealth();
    modal.classList.add("hidden");
    toast("Settings applied");
  });
  modal.classList.remove("hidden");
}
$("settings-btn").addEventListener("click", openSettings);
$("banner-settings").addEventListener("click", openSettings);
$("settings-close").addEventListener("click", () => modal.classList.add("hidden"));
modal.addEventListener("click", (e) => {
  if (e.target === modal) modal.classList.add("hidden");
});

// ---- server status ----
async function pollHealth(): Promise<void> {
  const cfg = await loadConfig(true);
  backendBadge.textContent = cfg.backend !== "unknown" ? cfg.backend : "—";
  const ok = await health();
  serverOnline = ok;
  serverDot.className = "dot " + (ok ? "ok" : "err");
  serverLabel.textContent = ok ? "ready" : cfg.configured ? "offline" : "setup needed";
  const needSetup = !ok && !cfg.configured;
  setupBanner.classList.toggle("hidden", !needSetup);
  if (needSetup) {
    (setupBanner.querySelector("span") as HTMLElement).textContent =
      "Trellis Studio isn't set up yet — run the installer or point it at your models directory.";
  } else if (!ok && cfg.configured) {
    setupBanner.classList.remove("hidden");
    (setupBanner.querySelector("span") as HTMLElement).textContent =
      "Server is offline — it may still be loading, or check the models directory in settings.";
  }
  updateGenerateEnabled();
}

// ---- server log -> progress (Tauri only) ----
listen<string>("server-log", (line) => {
  const t = String(line).trim();
  if (!generating || !t) return;
  // Painting a mesh: its encode stage is said in words; the rest as the server logs it.
  if (runningMode === "mesh" && /^\[(mesh|texture-mesh)\]|^\[3\/6\]|shape SLAT encode/.test(t))
    progressStage.textContent = "Reading your mesh…";
  else progressStage.textContent = t;
});

// ---- boot ----
async function boot(): Promise<void> {
  setViewerTools(false);
  restoreModeSettings(mode);
  applyModeUi(mode);
  renderMeshInfo();
  await refreshGallery();
  await pollHealth();
  window.setInterval(pollHealth, 4000);
  if (!isTauri()) {
    // Browser mode: no shell to report a backend.
    backendBadge.textContent = "browser";
  }
  void resumePending();
}
boot();
