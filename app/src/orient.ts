// Mesh orientation for "Paint a mesh" (docs/spec/34-studio-texture-mode.md). Mirrors the
// server's mesh_io (MeshOrient / AxisMap): the file axis `up` points up, then a quarter turn
// `yaw` about it, counter-clockwise seen from above. Every map is a signed axis permutation
// with determinant +1, so the preview is turned exactly as the server will turn the mesh.
// No imports: tests run this file directly with `node --test`.

export type MeshUp = "auto" | "+x" | "-x" | "+y" | "-y" | "+z" | "-z";
export type MeshYaw = 0 | 90 | 180 | 270;
export type MeshKind = "stl" | "obj" | "glb";

export const MESH_UPS: MeshUp[] = ["auto", "+z", "+y", "+x", "-z", "-y", "-x"];

/** out[i] = sign[i] * in[axis[i]] */
export interface AxisMap {
  axis: [number, number, number];
  sign: [number, number, number];
}

const map = (a0: number, s0: number, a1: number, s1: number, a2: number, s2: number): AxisMap => ({
  axis: [a0, a1, a2],
  sign: [s0, s1, s2],
});
const IDENTITY = map(0, 1, 1, 1, 2, 1);

/** this ∘ first: apply `first`, then `m`. */
export function after(m: AxisMap, first: AxisMap): AxisMap {
  const r = map(0, 1, 1, 1, 2, 1);
  for (let i = 0; i < 3; i++) {
    r.axis[i] = first.axis[m.axis[i]];
    r.sign[i] = m.sign[i] * first.sign[m.axis[i]];
  }
  return r;
}

export function inverse(m: AxisMap): AxisMap {
  const r = map(0, 1, 1, 1, 2, 1);
  for (let i = 0; i < 3; i++) {
    r.axis[m.axis[i]] = i;
    r.sign[m.axis[i]] = m.sign[i];
  }
  return r;
}

export function applyMap(m: AxisMap, v: [number, number, number]): [number, number, number] {
  return [m.sign[0] * v[m.axis[0]], m.sign[1] * v[m.axis[1]], m.sign[2] * v[m.axis[2]]];
}

/** The up axis "auto" stands for: z for STL (CAD / slicer convention), y otherwise. */
export function nativeUp(kind: MeshKind): Exclude<MeshUp, "auto"> {
  return kind === "stl" ? "+z" : "+y";
}

/** File -> glTF: the rotation taking the file axis `up` to +y. */
function upMap(up: Exclude<MeshUp, "auto">): AxisMap {
  switch (up) {
    case "+z": return map(0, 1, 2, 1, 1, -1);  // (x, y, z) -> (x, z, -y)
    case "-z": return map(0, 1, 2, -1, 1, 1);  // (x, y, z) -> (x, -z, y)
    case "-y": return map(0, 1, 1, -1, 2, -1); // (x, y, z) -> (x, -y, -z)
    case "+x": return map(1, -1, 0, 1, 2, 1);  // (x, y, z) -> (-y, x, z)
    case "-x": return map(1, 1, 0, -1, 2, 1);  // (x, y, z) -> (y, -x, z)
    default:   return IDENTITY;
  }
}

/** glTF: a turn about +y, counter-clockwise seen from above. */
function yawMap(yaw: MeshYaw): AxisMap {
  switch (yaw) {
    case 90:  return map(2, 1, 1, 1, 0, -1);  // (x, y, z) -> (z, y, -x)
    case 180: return map(0, -1, 1, 1, 2, -1); // (x, y, z) -> (-x, y, -z)
    case 270: return map(2, -1, 1, 1, 0, 1);  // (x, y, z) -> (-z, y, x)
    default:  return IDENTITY;
  }
}

const resolve = (kind: MeshKind, up: MeshUp) => (up === "auto" ? nativeUp(kind) : up);

/** What the server does to the file's vertices (file -> glTF y-up), as mesh_load_map. */
export function loadMap(kind: MeshKind, up: MeshUp, yaw: MeshYaw): AxisMap {
  return after(yawMap(yaw), upMap(resolve(kind, up)));
}

/** The file's own convention (file -> glTF), as the result GLB is written. */
export function nativeMap(kind: MeshKind): AxisMap {
  return upMap(nativeUp(kind));
}

/**
 * The turn to show on the preview, which is drawn in the file's own convention (like the
 * result): native glTF -> turned glTF.
 */
export function previewMap(kind: MeshKind, up: MeshUp, yaw: MeshYaw): AxisMap {
  return after(loadMap(kind, up, yaw), inverse(nativeMap(kind)));
}

/** The file axis that faces the front (glTF +z) after the turn, e.g. "-y" (mesh_front_axis). */
export function frontAxis(kind: MeshKind, up: MeshUp, yaw: MeshYaw): string {
  const m = loadMap(kind, up, yaw);
  return (m.sign[2] < 0 ? "-" : "+") + "xyz"[m.axis[2]];
}

/** The unit quaternion [x, y, z, w] of a rotation map (glTF node.rotation). */
export function quaternion(m: AxisMap): [number, number, number, number] {
  const M = [0, 1, 2].map((i) => [0, 1, 2].map((j) => (m.axis[i] === j ? m.sign[i] : 0)));
  const t = M[0][0] + M[1][1] + M[2][2];
  let x: number, y: number, z: number, w: number;
  if (t > 0) {
    const s = Math.sqrt(t + 1) * 2;
    w = s / 4; x = (M[2][1] - M[1][2]) / s; y = (M[0][2] - M[2][0]) / s; z = (M[1][0] - M[0][1]) / s;
  } else if (M[0][0] > M[1][1] && M[0][0] > M[2][2]) {
    const s = Math.sqrt(1 + M[0][0] - M[1][1] - M[2][2]) * 2;
    w = (M[2][1] - M[1][2]) / s; x = s / 4; y = (M[0][1] + M[1][0]) / s; z = (M[0][2] + M[2][0]) / s;
  } else if (M[1][1] > M[2][2]) {
    const s = Math.sqrt(1 + M[1][1] - M[0][0] - M[2][2]) * 2;
    w = (M[0][2] - M[2][0]) / s; x = (M[0][1] + M[1][0]) / s; y = s / 4; z = (M[1][2] + M[2][1]) / s;
  } else {
    const s = Math.sqrt(1 + M[2][2] - M[0][0] - M[1][1]) * 2;
    w = (M[1][0] - M[0][1]) / s; x = (M[0][2] + M[2][0]) / s; y = (M[1][2] + M[2][1]) / s; z = s / 4;
  }
  return [x, y, z, w];
}

/** The server field values: "auto" is not sent; the turn only when not 0. */
export function orientFields(up: MeshUp, yaw: MeshYaw): Record<string, string> {
  const f: Record<string, string> = {};
  if (up !== "auto") f.mesh_up = up.replace("+", "");
  if (yaw) f.mesh_yaw = String(yaw);
  return f;
}
