// Untextured previews of the meshes "Paint a mesh" sends to the server
// (docs/spec/34-studio-texture-mode.md). STL (binary or ASCII) and OBJ are turned into a
// minimal GLB in the browser; a GLB is shown as it is. The orientation is a root node's
// rotation added to the GLB's JSON, so the vertices are never touched. Parsing follows the
// server's loader (src/mesh_io.cpp): binary STL by size, ASCII by "solid", OBJ "v" / "f"
// with fanned polygons and negative indices. No DOM: tests run it with `node --test`.

import { nativeMap, applyMap, type MeshKind } from "./orient.ts";

export interface MeshFile {
  kind: MeshKind;
  /** Triangles in the file (OBJ polygons fanned; GLB: each mesh counted once). */
  triangles: number;
  /** A GLB of the file in its own convention (STL z-up turned to glTF y-up), in metres. */
  preview: ArrayBuffer;
}

export const MESH_EXT = /\.(stl|obj|glb)$/i;

export function meshKind(name: string): MeshKind | null {
  const m = MESH_EXT.exec(name);
  return m ? (m[1].toLowerCase() as MeshKind) : null;
}

/** Read a mesh file; throws an Error with a message for the user on a malformed file. */
export async function readMesh(data: ArrayBuffer, name: string): Promise<MeshFile> {
  const bytes = new Uint8Array(data);
  const isGlb = bytes.length >= 12 && String.fromCharCode(...bytes.subarray(0, 4)) === "glTF";
  const kind: MeshKind | null = isGlb ? "glb" : meshKind(name);
  if (!kind) throw new Error(`"${name}" isn't a mesh — use an STL, OBJ or GLB file.`);
  if (kind === "glb") return { kind, triangles: glbTriangles(data, name), preview: data };
  const { pos, idx } = kind === "stl" ? parseStl(data, name) : parseObj(data, name);
  const tris = idx ? idx.length / 3 : pos.length / 9;
  if (!tris) throw new Error(`"${name}" has no triangles.`);
  // The file's own convention, as the result GLB will be: STL z-up -> glTF y-up. STL / OBJ
  // carry no unit and are read as millimetres; glTF is in metres.
  const m = nativeMap(kind);
  for (let i = 0; i < pos.length; i += 3) {
    const v = applyMap(m, [pos[i], pos[i + 1], pos[i + 2]]);
    pos[i] = v[0] / 1000; pos[i + 1] = v[1] / 1000; pos[i + 2] = v[2] / 1000;
  }
  return { kind, triangles: tris, preview: writeGlb(pos, idx) };
}

function bad(name: string, what: string): Error {
  return new Error(`"${name}" can't be read: ${what}.`);
}

function parseStl(data: ArrayBuffer, name: string): { pos: Float32Array; idx: null } {
  const dv = new DataView(data);
  if (data.byteLength >= 84) {
    const n = dv.getUint32(80, true);
    if (n > 0 && 84 + 50 * n === data.byteLength) {
      const pos = new Float32Array(n * 9);
      for (let f = 0; f < n; f++)
        for (let k = 0; k < 9; k++) {
          const v = dv.getFloat32(84 + 50 * f + 12 + 4 * k, true);
          if (!isFinite(v)) throw bad(name, `a vertex of facet ${f + 1} is not a number`);
          pos[9 * f + k] = v;
        }
      return { pos, idx: null };
    }
  }
  const text = new TextDecoder().decode(data);
  if (!/^\s*solid/.test(text)) throw bad(name, "not a binary STL (wrong size) and not an ASCII one");
  const vals: number[] = [];
  const re = /^\s*vertex\s+(\S+)\s+(\S+)\s+(\S+)/gm;
  for (let m; (m = re.exec(text)); ) {
    const v = [parseFloat(m[1]), parseFloat(m[2]), parseFloat(m[3])];
    if (!v.every(isFinite)) throw bad(name, `bad vertex "${m[0].trim()}"`);
    vals.push(...v);
  }
  if (!vals.length) throw bad(name, "no facets");
  if (vals.length % 9) throw bad(name, "a facet with fewer than 3 vertices");
  return { pos: new Float32Array(vals), idx: null };
}

function parseObj(data: ArrayBuffer, name: string): { pos: Float32Array; idx: Uint32Array } {
  const text = new TextDecoder().decode(data);
  const pos: number[] = [];
  const idx: number[] = [];
  const lines = text.split(/\r?\n/);
  for (let ln = 0; ln < lines.length; ln++) {
    const t = lines[ln].trim().split(/\s+/);
    if (t[0] === "v") {
      const v = [parseFloat(t[1]), parseFloat(t[2]), parseFloat(t[3])];
      if (!v.every(isFinite)) throw bad(name, `bad vertex at line ${ln + 1}`);
      pos.push(...v);
    } else if (t[0] === "f") {
      const nv = pos.length / 3;
      const ids = t.slice(1).map((tok) => {
        const i = parseInt(tok.split("/")[0], 10);
        const r = i < 0 ? nv + i : i - 1;
        if (!Number.isInteger(i) || i === 0 || r < 0 || r >= nv) throw bad(name, `bad face at line ${ln + 1}`);
        return r;
      });
      if (ids.length < 3) throw bad(name, `bad face at line ${ln + 1}`);
      for (let k = 1; k + 1 < ids.length; k++) idx.push(ids[0], ids[k], ids[k + 1]);
    }
  }
  if (!idx.length) throw bad(name, "no faces");
  return { pos: new Float32Array(pos), idx: new Uint32Array(idx) };
}

interface GltfJson {
  scene?: number;
  scenes?: { nodes?: number[] }[];
  nodes?: Record<string, unknown>[];
  meshes?: { primitives: { mode?: number; indices?: number; attributes: Record<string, number> }[] }[];
  accessors?: { count: number }[];
  [k: string]: unknown;
}

/** The JSON and BIN chunks of a GLB; throws on anything that isn't one. */
function splitGlb(data: ArrayBuffer, name: string): { json: GltfJson; bin: Uint8Array | null } {
  const dv = new DataView(data);
  if (data.byteLength < 20 || dv.getUint32(0, true) !== 0x46546c67 || dv.getUint32(4, true) !== 2)
    throw bad(name, "not a glTF 2.0 binary");
  const jlen = dv.getUint32(12, true);
  if (dv.getUint32(16, true) !== 0x4e4f534a || 20 + jlen > data.byteLength) throw bad(name, "no JSON chunk");
  let json: GltfJson;
  try {
    json = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 20, jlen)));
  } catch {
    throw bad(name, "its JSON chunk doesn't parse");
  }
  let bin: Uint8Array | null = null;
  const off = 20 + jlen;
  if (off + 8 <= data.byteLength && dv.getUint32(off + 4, true) === 0x004e4942) {
    const blen = dv.getUint32(off, true);
    if (off + 8 + blen > data.byteLength) throw bad(name, "its binary chunk is truncated");
    bin = new Uint8Array(data, off + 8, blen);
  }
  return { json, bin };
}

function glbTriangles(data: ArrayBuffer, name: string): number {
  const { json } = splitGlb(data, name);
  let n = 0;
  for (const mesh of json.meshes ?? [])
    for (const p of mesh.primitives ?? []) {
      if ((p.mode ?? 4) !== 4) continue;
      const acc = json.accessors?.[p.indices ?? p.attributes?.POSITION ?? -1];
      if (acc) n += Math.floor(acc.count / 3);
    }
  if (!n) throw bad(name, "no triangles");
  return n;
}

const pad4 = (n: number) => (n + 3) & ~3;

function packGlb(json: object, bin: Uint8Array | null): ArrayBuffer {
  const js = new TextEncoder().encode(JSON.stringify(json));
  const jlen = pad4(js.length);
  const blen = bin ? pad4(bin.length) : 0;
  const total = 12 + 8 + jlen + (bin ? 8 + blen : 0);
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546c67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jlen, true);
  dv.setUint32(16, 0x4e4f534a, true);
  out.set(js, 20);
  out.fill(0x20, 20 + js.length, 20 + jlen);
  if (bin) {
    const o = 20 + jlen;
    dv.setUint32(o, blen, true);
    dv.setUint32(o + 4, 0x004e4942, true);
    out.set(bin, o + 8);
  }
  return out.buffer;
}

/** A minimal GLB: one grey, double-sided primitive (no normals: viewers shade it flat). */
export function writeGlb(pos: Float32Array, idx: Uint32Array | null): ArrayBuffer {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i += 3)
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], pos[i + a]);
      hi[a] = Math.max(hi[a], pos[i + a]);
    }
  const pbytes = pos.byteLength, ibytes = idx ? idx.byteLength : 0;
  const bin = new Uint8Array(pbytes + ibytes);
  bin.set(new Uint8Array(pos.buffer, pos.byteOffset, pbytes), 0);
  if (idx) bin.set(new Uint8Array(idx.buffer, idx.byteOffset, ibytes), pbytes);
  const prim: Record<string, unknown> = { attributes: { POSITION: 0 }, material: 0 };
  const bufferViews: object[] = [{ buffer: 0, byteOffset: 0, byteLength: pbytes, target: 34962 }];
  const accessors: object[] = [
    { bufferView: 0, componentType: 5126, count: pos.length / 3, type: "VEC3", min: lo, max: hi },
  ];
  if (idx) {
    bufferViews.push({ buffer: 0, byteOffset: pbytes, byteLength: ibytes, target: 34963 });
    accessors.push({ bufferView: 1, componentType: 5125, count: idx.length, type: "SCALAR" });
    prim.indices = 1;
  }
  const json = {
    asset: { version: "2.0", generator: "Trellis Studio mesh preview" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [prim] }],
    materials: [{
      pbrMetallicRoughness: { baseColorFactor: [0.42, 0.45, 0.5, 1], metallicFactor: 0, roughnessFactor: 0.85 },
      doubleSided: true,
    }],
    buffers: [{ byteLength: bin.length }],
    bufferViews,
    accessors,
  };
  return packGlb(json, bin);
}

/** The same GLB with its scene under one root node turned by quaternion `q` ([x, y, z, w]). */
export function orientGlb(data: ArrayBuffer, q: [number, number, number, number]): ArrayBuffer {
  const { json, bin } = splitGlb(data, "mesh");
  const si = json.scene ?? 0;
  const scenes = json.scenes ?? [{ nodes: [] }];
  const scene = scenes[si] ?? (scenes[si] = { nodes: [] });
  const nodes = json.nodes ?? (json.nodes = []);
  nodes.push({ name: "studio-orientation", rotation: q, children: scene.nodes ?? [] });
  scene.nodes = [nodes.length - 1];
  json.scenes = scenes;
  json.scene = si;
  return packGlb(json, bin);
}
