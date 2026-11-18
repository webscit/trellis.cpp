// Thin bridge over the Tauri v2 runtime so the same UI also runs in a plain
// browser (vite dev / served by trellis-server). Every Tauri call is guarded by
// `isTauri()`; in the browser we fall back to sensible web behaviour.

import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { listen as tauriListen, type UnlistenFn } from "@tauri-apps/api/event";

export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/** Invoke a Rust command; throws in the browser (callers must guard). */
export function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  return tauriInvoke<T>(cmd, args);
}

/** Subscribe to a Tauri event; no-op (returns a noop unlisten) in the browser. */
export async function listen<T>(
  event: string,
  handler: (payload: T) => void,
): Promise<UnlistenFn> {
  if (!isTauri()) return () => {};
  return tauriListen<T>(event, (e) => handler(e.payload as T));
}

/**
 * Save bytes to disk. In Tauri, open a native "Save as" dialog then write via
 * the fs plugin. In the browser, trigger a normal download.
 */
export async function saveBytes(defaultName: string, bytes: Uint8Array): Promise<boolean> {
  // The format follows the file name's extension (GLB unless it says 3MF).
  const is3mf = /\.3mf$/i.test(defaultName);
  const kind = is3mf
    ? { name: "3MF", extensions: ["3mf"], mime: "model/3mf" }
    : { name: "glTF binary", extensions: ["glb"], mime: "model/gltf-binary" };
  if (isTauri()) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const { writeFile } = await import("@tauri-apps/plugin-fs");
    const path = await save({
      defaultPath: defaultName,
      filters: [{ name: kind.name, extensions: kind.extensions }],
    });
    if (!path) return false;
    await writeFile(path, bytes);
    return true;
  }
  const blob = new Blob([bytes as BlobPart], { type: kind.mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = defaultName;
  a.click();
  URL.revokeObjectURL(url);
  return true;
}

/** Native folder picker (Tauri only); returns the chosen path or null. */
export async function pickDirectory(current?: string): Promise<string | null> {
  if (!isTauri()) return null;
  const { open } = await import("@tauri-apps/plugin-dialog");
  const res = await open({ directory: true, defaultPath: current || undefined });
  return typeof res === "string" ? res : null;
}

/** Open the configured output directory in the OS file browser (Tauri only). */
export async function openOutputDir(): Promise<void> {
  if (isTauri()) await invoke("open_output_dir");
}

/** Open the server logs directory in the OS file browser (Tauri only). */
export async function openLogsDir(): Promise<void> {
  if (isTauri()) await invoke("open_logs_dir");
}

/** The logs directory path (Tauri only); empty string in the browser. */
export async function logsDir(): Promise<string> {
  if (!isTauri()) return "";
  try {
    return await invoke<string>("logs_dir");
  } catch {
    return "";
  }
}

/**
 * Auto-save GLB bytes into the configured output directory (Tauri only).
 * The Rust side resolves + creates the dir and returns the full path; the fs
 * plugin then writes the bytes there. Returns the path, or null in the browser.
 */
export async function saveToOutputDir(name: string, bytes: Uint8Array): Promise<string | null> {
  if (!isTauri()) return null;
  const { writeFile } = await import("@tauri-apps/plugin-fs");
  const path = await invoke<string>("output_path", { name });
  await writeFile(path, bytes);
  return path;
}
