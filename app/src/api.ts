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
  const res = await fetch(`${await base()}/generate`, {
    method: "POST",
    body: toForm(image, params, views),
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
