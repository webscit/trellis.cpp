// Shared types for Trellis Studio.

export type Resolution = 512 | 1024 | 1536;
export type BgRemoval = "auto" | "birefnet" | "threshold";
export type Uv = "xatlas" | "box";
export type FrontAxis = "-y" | "+y" | "+x" | "-x" | "ref-z" | "ref-x";

/** The server accepts at most this many images per request (front + extra views). */
export const MAX_VIEWS = 12;

/** An additional photo of the same object, taken from another direction.
 *  az: 0 front, 90 right, 180 back, 270 left; el: +90 top, -90 bottom. */
export interface ExtraView {
  image: Blob;
  name: string;
  az: number;
  el: number;
}

/** Generation knobs mapped onto trellis-server's POST /generate form fields. */
/** --print-interior: what is sealed inside the model. */
export type PrintInterior = "auto" | "solid" | "hollow";

export interface GenParams {
  resolution: Resolution;
  seed: number;
  bgRemoval: BgRemoval;
  uv: Uv;
  /** false => geometry only (server field texture=off). */
  texture: boolean;
  /** true => one closed, manifold solid for 3D printing (server field watertight=on). */
  watertight?: boolean;
  /** 3D print options, sent only with watertight (server fields print_*). */
  printSize?: number; // mm, longest side
  printBase?: string; // "auto" | "0" | fraction of the height
  printWall?: number; // mm, 0 = off
  printInterior?: PrintInterior;
  /** Filament colours in the 3MF (server field print_colors): 0 = textured 3MF, else 2..32. */
  printColors?: number;
  /** Older records: 0 = keep hollow, anything else = the then default. */
  printSeal?: number;
  /** Multi-view blend sharpness (server field blend_temp; only sent with extra views). */
  blendTemp: number;
  /** Multi-view front axis (server field front_axis; only sent with extra views). */
  frontAxis: FrontAxis;
  /** Camera direction of the main image (only sent with extra views; default 0, 0). */
  mainAz?: number;
  mainEl?: number;
}

export const DEFAULT_PARAMS: GenParams = {
  resolution: 1024,
  seed: 42,
  bgRemoval: "auto",
  uv: "xatlas",
  texture: true,
  watertight: false,
  printSize: 100,
  printBase: "auto",
  printWall: 0.8,
  printInterior: "auto",
  printColors: 0,
  blendTemp: 2,
  frontAxis: "-y",
};

/** Written by the installer, read by the Tauri shell + surfaced to the UI. */
export interface AppConfig {
  serverBin: string;
  modelsDir: string;
  backend: string;
  gpu: number;
  host: string;
  port: number;
  /** where generated GLBs are auto-saved (Tauri only). */
  outputDir: string;
  /** true once the shell has a usable config.json; false => "setup needed". */
  configured: boolean;
}

/** trellis-server's printability report (GET /jobs/{id}/report). */
export interface PrintReport {
  ok: boolean;
  size_mm: [number, number, number];
  volume_cm3: number;
  surface_cm2: number;
  watertight: boolean;
  shells: number;
  triangles: number;
  wall: { target_mm: number; thinnest_1pct_mm: number; at_least: boolean; thin_area_pct: number };
  base: { flat: boolean; contact_mm2: number; stands: boolean; tip_angle_deg: number };
  overhang_pct: number;
  /** Present when the 3MF carries filament colours: slot 1.. by surface share. */
  colors?: { patches: number; palette: { slot: number; hex: string; share_pct: number }[] };
  warnings: string[];
}

/** One persisted generation (IndexedDB record). */
export interface GenRecord {
  id: string;
  ts: number;
  name: string;
  params: GenParams;
  input: Blob; // source image (the front view)
  views?: ExtraView[]; // extra views (multi-view); absent on single-image records
  glb: Blob | null; // resulting model/gltf-binary; null while pending
  /** The same model as a 3MF (made when "3D print" is on); absent otherwise. */
  model3mf?: Blob | null;
  /** Printability report of a "3D print" run; absent otherwise. */
  report?: PrintReport | null;
  /** true while the server is still working on it (the entry survives reloads). */
  pending?: boolean;
  /** trellis-server job id of a pending entry, to resume polling after a reload. */
  jobId?: string;
  thumb: Blob | null; // model-viewer snapshot for the gallery
}
