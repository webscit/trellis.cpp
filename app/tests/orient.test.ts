// node --test tests/ — the orientation maps match the server's (src/mesh_io.cpp).
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  MESH_UPS, applyMap, frontAxis, inverse, after, loadMap, orientFields, previewMap, quaternion,
  type MeshKind, type MeshYaw,
} from "../src/orient.ts";

const YAWS: MeshYaw[] = [0, 90, 180, 270];
const KINDS: MeshKind[] = ["stl", "obj", "glb"];
const z0 = (v: number[]) => v.map((c) => c + 0); // -0 -> 0
const axisVec = (s: string): [number, number, number] => {
  const v: [number, number, number] = [0, 0, 0];
  v["xyz".indexOf(s[1])] = s[0] === "-" ? -1 : 1;
  return v;
};

function rotate(q: number[], v: number[]): number[] {
  const [x, y, z, w] = q;
  // v' = q v q*, expanded
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}

test("every orientation is a proper rotation; up goes up, the front goes to +z", () => {
  for (const kind of KINDS)
    for (const up of MESH_UPS)
      for (const yaw of YAWS) {
        const m = loadMap(kind, up, yaw);
        const upAxis = up === "auto" ? (kind === "stl" ? "+z" : "+y") : up;
        assert.deepEqual(z0(applyMap(m, axisVec(upAxis))), [0, 1, 0], `${kind} ${up} ${yaw}: up`);
        assert.deepEqual(z0(applyMap(m, axisVec(frontAxis(kind, up, yaw)))), [0, 0, 1], `${kind} ${up} ${yaw}: front`);
        const q = quaternion(m);
        assert.ok(Math.abs(Math.hypot(...q) - 1) < 1e-12, "unit quaternion");
        for (const a of ["+x", "+y", "+z"]) {
          const r = rotate(q, axisVec(a)), e = applyMap(m, axisVec(a));
          assert.ok(r.every((c, i) => Math.abs(c - e[i]) < 1e-12), `${kind} ${up} ${yaw}: quaternion vs map on ${a}`);
        }
        assert.deepEqual(after(inverse(m), m), loadMap("glb", "+y", 0), "inverse");
      }
});

test("turns are counter-clockwise seen from above", () => {
  for (const kind of KINDS)
    for (const up of MESH_UPS) {
      const m0 = loadMap(kind, up, 0), m90 = loadMap(kind, up, 90);
      // the file direction that faced +x at turn 0 faces -z at 90
      const d = applyMap(inverse(m0), [1, 0, 0]);
      assert.deepEqual(applyMap(m90, d), [0, 0, -1], `${kind} ${up}`);
    }
});

test("the fronts trellis-cli logged in phase 1", () => {
  assert.deepEqual(YAWS.map((y) => frontAxis("stl", "auto", y)), ["-y", "-x", "+y", "+x"]);
  assert.equal(frontAxis("stl", "-z", 180), "-y");
  assert.equal(frontAxis("obj", "auto", 0), "+z");
});

test("the preview turn is the identity for the file's own convention", () => {
  for (const kind of KINDS) assert.deepEqual(quaternion(previewMap(kind, "auto", 0)), [0, 0, 0, 1]);
  assert.deepEqual(quaternion(previewMap("stl", "+z", 0)), [0, 0, 0, 1]);
  assert.deepEqual(quaternion(previewMap("obj", "+y", 0)), [0, 0, 0, 1]);
  // an STL: up -z turns it over (180 degrees about x in glTF)
  const q = quaternion(previewMap("stl", "-z", 0));
  assert.ok(Math.abs(Math.abs(q[0]) - 1) < 1e-12, `upside down: ${q}`);
});

test("server fields", () => {
  assert.deepEqual(orientFields("auto", 0), {});
  assert.deepEqual(orientFields("-x", 270), { mesh_up: "-x", mesh_yaw: "270" });
  assert.deepEqual(orientFields("+z", 0), { mesh_up: "z" });
});
