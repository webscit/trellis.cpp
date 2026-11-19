// node --test tests/ — mesh files read like the server reads them, previews are valid GLBs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { meshKind, orientGlb, readMesh, writeGlb } from "../src/meshpreview.ts";

const enc = (s: string) => new TextEncoder().encode(s).buffer as ArrayBuffer;

// A 40 x 20 x 30 box, corner at the origin, as file vertices and quads.
const P = [[0, 0, 0], [40, 0, 0], [0, 20, 0], [40, 20, 0], [0, 0, 30], [40, 0, 30], [0, 20, 30], [40, 20, 30]];
const Q = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
const TRIS = Q.flatMap(([a, b, c, d]) => [[a, b, c], [a, c, d]]);

function binaryStl(): ArrayBuffer {
  const b = new ArrayBuffer(84 + 50 * TRIS.length);
  const dv = new DataView(b);
  new Uint8Array(b).set(new TextEncoder().encode("solid but binary"), 0); // a "solid" header must not fool it
  dv.setUint32(80, TRIS.length, true);
  TRIS.forEach((t, f) => t.forEach((v, k) => P[v].forEach((c, a) => dv.setFloat32(84 + 50 * f + 12 + 12 * k + 4 * a, c, true))));
  return b;
}
const asciiStl = () =>
  "solid box\n" + TRIS.map((t) => "facet normal 0 0 0\nouter loop\n" +
    t.map((v) => `vertex ${P[v].join(" ")}\n`).join("") + "endloop\nendfacet\n").join("") + "endsolid box\n";
// quads, with negative (relative) indices on the last face and v/vt/vn tokens
const obj = () =>
  P.map((p) => `v ${p.join(" ")}`).join("\n") + "\n" +
  Q.slice(0, 5).map((q) => "f " + q.map((i) => `${i + 1}/1/1`).join(" ")).join("\n") + "\n" +
  "f " + Q[5].map((i) => String(i - 8)).join(" ") + "\n";

/** The preview's JSON, its POSITION values and the root rotation (if any). */
function inspect(glb: ArrayBuffer) {
  const dv = new DataView(glb);
  assert.equal(dv.getUint32(0, true), 0x46546c67, "magic");
  assert.equal(dv.getUint32(8, true), glb.byteLength, "length");
  const jlen = dv.getUint32(12, true);
  assert.equal(jlen % 4, 0, "JSON chunk padded");
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(glb, 20, jlen)));
  const binOff = 20 + jlen + 8;
  const acc = json.accessors[json.meshes[0].primitives[0].attributes.POSITION];
  const bv = json.bufferViews[acc.bufferView];
  const pos = new Float32Array(glb.slice(binOff + (bv.byteOffset ?? 0), binOff + (bv.byteOffset ?? 0) + acc.count * 12));
  return { json, pos };
}

test("STL binary / ASCII and OBJ give the same box, z-up STL turned to y-up, in metres", async () => {
  const cases: [string, ArrayBuffer][] = [["box.stl", binaryStl()], ["box.stl", enc(asciiStl())], ["box.obj", enc(obj())]];
  for (const [name, data] of cases) {
    const m = await readMesh(data, name);
    assert.equal(m.triangles, 12, name);
    const { json, pos } = inspect(m.preview);
    const acc = json.accessors[0];
    if (m.kind === "stl") {
      // file (x, y, z) mm -> glTF (x, z, -y) m: height 30 mm is glTF y
      assert.deepEqual(acc.min.map((v: number) => +v.toFixed(6)), [0, 0, -0.02], name);
      assert.deepEqual(acc.max.map((v: number) => +v.toFixed(6)), [0.04, 0.03, 0], name);
    } else {
      assert.deepEqual(acc.max.map((v: number) => +v.toFixed(6)), [0.04, 0.02, 0.03], name);
    }
    assert.ok(pos.every(Number.isFinite));
  }
});

test("GLB: read as is, triangle count from the accessors, orientation as a root node", async () => {
  const pos = new Float32Array(TRIS.flatMap((t) => t.flatMap((v) => P[v].map((c) => c / 1000))));
  const glb = writeGlb(pos, null);
  const m = await readMesh(glb, "x.glb");
  assert.equal(m.kind, "glb");
  assert.equal(m.triangles, 12);
  assert.equal(m.preview, glb);
  const turned = orientGlb(glb, [0, 0.7071, 0, 0.7071]);
  const { json, pos: p2 } = inspect(turned);
  assert.equal(json.scenes[0].nodes.length, 1);
  const root = json.nodes[json.scenes[0].nodes[0]];
  assert.deepEqual(root.rotation, [0, 0.7071, 0, 0.7071]);
  assert.deepEqual(root.children, [0]);
  assert.deepEqual(Array.from(p2), Array.from(pos), "vertices untouched");
  // a GLB is recognised by its magic whatever the name
  assert.equal((await readMesh(turned, "upload.bin")).kind, "glb");
});

test("extensions", () => {
  assert.equal(meshKind("Part.STL"), "stl");
  assert.equal(meshKind("a.obj"), "obj");
  assert.equal(meshKind("a.glb"), "glb");
  assert.equal(meshKind("a.gltf"), null);
  assert.equal(meshKind("a.png"), null);
});

test("malformed files fail with a message, never a crash", async () => {
  const nan = binaryStl();
  new DataView(nan).setFloat32(84 + 12, NaN, true);
  const truncGlb = writeGlb(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), null).slice(0, 30);
  const cases: [string, ArrayBuffer][] = [
    ["x.stl", enc("hello")],
    ["short.stl", enc("solid s\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nendloop\nendfacet\n")],
    ["empty.stl", enc("solid e\nendsolid e\n")],
    ["nan.stl", nan],
    ["bad.obj", enc("v 0 0 0\nv 1 0 0\nf 1 2 5\n")],
    ["zero.obj", enc("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n")],
    ["nofaces.obj", enc("v 0 0 0\n")],
    ["trunc.glb", truncGlb],
    ["photo.png", enc("\x89PNG....")],
  ];
  for (const [name, data] of cases)
    await assert.rejects(readMesh(data, name), (e: Error) => e.message.includes(name), name);
});
