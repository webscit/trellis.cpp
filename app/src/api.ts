// HTTP client for the resident trellis-server (see src/trellis-server.cpp):
//   GET  /health   -> "ok"
//   POST /generate  multipart: image file + seed/resolution/bg_removal/uv/texture fields;
//                   multi-view: one "image" part per view (front first) + "azimuths" and
//                   "elevations" (comma lists, one entry per image) + blend_temp/front_axis
//                   -> model/gltf-binary, or JSON {"error": "..."} on failure.

import { apiBase, loadConfig } from "./config";
import { DEFAULT_PARAMS, type ExtraView, type GenParams } from "./types";

async function base(): Promise<string> {
  return apiBase(await loadConfig());
}

export async function health(timeoutMs = 2000): Promise<boolean> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${await base()}/health`, { signal: ctrl.signal });
    return res.ok && (await res.text()).trim() === "ok";
  } catch {
    return false;
  } finally {
    clearTimeout(t);
  }
}

/**
 * EXIF orientation of a JPEG (1 = upright; 0 = no tag / not parseable). Only the
 * APP1 segment near the file start is read.
 */
async function jpegOrientation(image: Blob): Promise<number> {
  const b = new DataView(await image.slice(0, 128 * 1024).arrayBuffer());
  let off = 2;
  while (off + 4 <= b.byteLength && b.getUint8(off) === 0xff) {
    const marker = b.getUint8(off + 1);
    const len = b.getUint16(off + 2);
    if (marker === 0xda) break; // start of scan: no more metadata
    if (marker === 0xe1 && off + 10 <= b.byteLength && b.getUint32(off + 4) === 0x45786966) {
      const tiff = off + 10; // "Exif\0\0" then the TIFF header
      if (tiff + 8 > b.byteLength) return 0;
      const le = b.getUint16(tiff) === 0x4949;
      const ifd = tiff + b.getUint32(tiff + 4, le);
      if (ifd + 2 > b.byteLength) return 0;
      const n = b.getUint16(ifd, le);
      for (let i = 0; i < n; i++) {
        const e = ifd + 2 + i * 12;
        if (e + 12 > b.byteLength) return 0;
        if (b.getUint16(e, le) === 0x0112) return b.getUint16(e + 8, le);
      }
      return 0;
    }
    off += 2 + len;
  }
  return 0;
}

/**
 * Can trellis-server read these bytes as-is? It decodes with stb_image (PNG,
 * JPEG, BMP, GIF, TGA, PSD, HDR, PNM). Those are sent untouched: re-encoding
 * would premultiply alpha on cut-out edges, apply ICC profiles the CLI ignores,
 * and inflate uploads; TGA/PSD/HDR/PNM can't even be decoded by a browser.
 */
async function serverReadable(image: Blob): Promise<boolean> {
  const h = new Uint8Array(await image.slice(0, 16).arrayBuffer());
  const ascii = String.fromCharCode(...h);
  if (h[0] === 0x89 && ascii.slice(1, 4) === "PNG") return true;
  if (h[0] === 0xff && h[1] === 0xd8 && h[2] === 0xff) {
    // stb_image ignores EXIF rotation: let the browser apply it to rotated photos.
    return (await jpegOrientation(image)) <= 1;
  }
  if (ascii.startsWith("GIF8") || ascii.startsWith("BM") || ascii.startsWith("8BPS")) return true;
  if (ascii.startsWith("#?RADIANCE") || ascii.startsWith("#?RGBE")) return true;
  if (/^P[56]\s/.test(ascii)) return true;
  // TGA has no magic number: trust the file name / type.
  const name = image instanceof File ? image.name : "";
  return /\.(tga|tpic)$/i.test(name) || /tga|targa/i.test(image.type);
}

/** The image as the server should receive it: untouched when readable, else PNG. */
export async function prepareImage(image: Blob): Promise<Blob> {
  return (await serverReadable(image)) ? image : toPng(image);
}

/**
 * Re-encode an image as PNG in the browser, for formats stb_image can't read
 * (WebP, AVIF, SVG...) and for EXIF-rotated JPEGs (<img> applies the rotation).
 */
export async function toPng(image: Blob): Promise<Blob> {
  const url = URL.createObjectURL(image);
  try {
    const img = new Image();
    img.src = url;
    try {
      await img.decode();
    } catch {
      throw new Error("this image format can't be decoded — use a PNG, JPEG or WebP file");
    }
    // An SVG without width/height has no intrinsic size.
    const w = img.naturalWidth || 1024;
    const h = img.naturalHeight || 1024;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("canvas unavailable");
    ctx.drawImage(img, 0, 0, w, h);
    const png = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!png) throw new Error("could not convert the image to PNG");
    return png;
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** trellis-server maps bg_removal: "auto" keeps the server default (no field). */
function toForm(image: Blob, p: GenParams, views: ExtraView[]): FormData {
  const fd = new FormData();
  fd.append("image", image, "input.png");
  if (views.length) {
    views.forEach((v, i) => fd.append("image", v.image, `view${i + 1}.png`));
    fd.append("azimuths", ["0", ...views.map((v) => String(v.az))].join(","));
    fd.append("elevations", ["0", ...views.map((v) => String(v.el))].join(","));
    // Only override the server's launch defaults when the user changed them.
    if (p.blendTemp !== DEFAULT_PARAMS.blendTemp) fd.append("blend_temp", String(p.blendTemp));
    if (p.frontAxis !== DEFAULT_PARAMS.frontAxis) fd.append("front_axis", p.frontAxis);
  }
  fd.append("seed", String(p.seed));
  fd.append("resolution", String(p.resolution));
  if (p.bgRemoval !== "auto") fd.append("bg_removal", p.bgRemoval);
  fd.append("uv", p.uv);
  if (!p.texture) fd.append("texture", "off");
  return fd;
}

export interface GenerateResult {
  glb: Blob;
}

export async function generate(
  image: Blob,
  params: GenParams,
  views: ExtraView[] = [],
  signal?: AbortSignal,
): Promise<GenerateResult> {
  const pngs = await Promise.all([image, ...views.map((v) => v.image)].map(prepareImage));
  const res = await fetch(`${await base()}/generate`, {
    method: "POST",
    body: toForm(pngs[0], params, views.map((v, i) => ({ ...v, image: pngs[i + 1] }))),
    signal,
  });
  if (!res.ok) {
    let msg = `generation failed (HTTP ${res.status})`;
    try {
      const j = await res.json();
      if (j && typeof j.error === "string") msg = j.error;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(msg);
  }
  const glb = await res.blob();
  if (glb.size === 0) throw new Error("server returned an empty model");
  return { glb };
}
