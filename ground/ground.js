// Near-field ground: streams the HiRISE 25 cm orthoimage tiles written by walk/pipeline/tiles.py around the walker
// into two toroidal clipmap textures, and the 1 m DTM tiles into a height cache.
//
// Level 0: 2048 x 2048 texels at 0.25 m = 512 m, 4 x 4 tiles of 128 m (o0/i_j.jpg, 512 px).
// Level 1: 2048 x 2048 texels at 0.5 m = 1024 m, 4 x 4 tiles of 256 m (o1/i_j.jpg, 512 px).
// World tile (i, j) of a level always sits in texture slot (i mod 4, j mod 4), so the texture coordinate of world point
// (x, z) is (x, z) / (level size), wrapped by REPEAT. When the walker moves, only the slots whose tile changes are
// rewritten; texture contents never move. The coordinate is not passed through fract() in the shader, because the jump
// at the wrap would make the derivatives (and so the mip level) wrong along one line.
//
// Textures are RG8 with 10 mip levels (2048 down to 4): G = coverage (0 = no data), R = albedo code times coverage.
// Coverage ramps from 0 to 1 over RAMP metres away from no-data pixels and from tile sides that have no neighbour tile,
// so filtering never mixes no-data (code 0) into the albedo, and the data edge fades instead of ending in a hard line.
// Tiles are 512 px and slot-aligned, so mip level k of a tile occupies its own (512 >> k)^2 block at slot * (512 >> k);
// the levels are box-filtered in a worker (coverage-weighted, in linear albedo) and uploaded with the tile, so the GPU
// never generates mipmaps.
//
// JPEG fetch, decode and mip building run in two workers. The main thread only uploads: at most 2 tiles per update()
// and stops early once the upload time this frame exceeds budgetMs.
//
// Brightness: the tile JPEGs and the page's l0_a.jpg are both sqrt-encoded albedo, but from different ortho products
// (RED_A 0.25 m vs RED_C 1 m), different de-lighting (1 m DTM, fitted sky term vs 2 m smoothed DTM, sky term 0.22) and
// different normalisation (tile p99.5 median vs p99.7). Measured on the Melas start, the relation between the two at 2 m
// is not a pure scale (linear least squares: L0 = 0.109 + 0.987 tile). So the module fits a mapping at load time:
// quantile matching of 2 m block values of the o2 tiles against l0_a.jpg over the same pixels, stored as 17 knots of a
// piecewise-linear curve from tile code to l0_a code (groundLut). groundAlbedo() returns values on l0_a's scale, i.e.
// the same number the page reads from texture2D(map, vUv).r.

const T0 = 128;                              // level-0 tile size in metres (tiles.json tile_m)
const N = 4;                                 // tiles per clipmap side
const PX = 512;                              // tile size in pixels
const MIPS = 10;                             // 2048 .. 4
const RAMP = 6;                              // metres over which coverage rises at a data edge
const KEEP = 1.375;                          // keep the window while the walker is at least this many tiles from its edge

const WORKER_SRC = `
const cv = new OffscreenCanvas(512, 512); const cx = cv.getContext("2d", { willReadFrequently: true });
function readBitmap(bm) {
  if (cv.width !== bm.width || cv.height !== bm.height) { cv.width = bm.width; cv.height = bm.height; }
  cx.drawImage(bm, 0, 0); const px = cx.getImageData(0, 0, bm.width, bm.height).data, out = new Uint8Array(bm.width * bm.height);
  for (let k = 0; k < out.length; k++) out[k] = px[4 * k];
  bm.close(); return out;
}
async function decode(url) {
  const r = await fetch(url); if (!r.ok) return null;
  return readBitmap(await createImageBitmap(await r.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" }));
}
// Coverage: distance (pixels, chamfer 1 / 1.414) to the nearest no-data pixel or open side, divided by ramp, smoothstepped.
// open = [west, east, north, south]: true where the neighbouring tile does not exist.
function coverage(c, n, open, ramp) {
  if (!open.some(Boolean) && c.every((v) => v > 0)) return new Float32Array(n * n).fill(1);   // no edge in this tile
  const d = new Float32Array(n * n), D = 1.4142;
  for (let k = 0; k < d.length; k++) d[k] = c[k] ? 1e9 : 0;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const o = j * n + i; let v = d[o]; if (!v) continue;
    if (i === 0) { if (open[0]) v = Math.min(v, 1); } else v = Math.min(v, d[o - 1] + 1);
    if (j === 0) { if (open[2]) v = Math.min(v, 1); } else { v = Math.min(v, d[o - n] + 1); if (i > 0) v = Math.min(v, d[o - n - 1] + D); if (i < n - 1) v = Math.min(v, d[o - n + 1] + D); }
    d[o] = v;
  }
  for (let j = n - 1; j >= 0; j--) for (let i = n - 1; i >= 0; i--) {
    const o = j * n + i; let v = d[o]; if (!v) continue;
    if (i === n - 1) { if (open[1]) v = Math.min(v, 1); } else v = Math.min(v, d[o + 1] + 1);
    if (j === n - 1) { if (open[3]) v = Math.min(v, 1); } else { v = Math.min(v, d[o + n] + 1); if (i < n - 1) v = Math.min(v, d[o + n + 1] + D); if (i > 0) v = Math.min(v, d[o + n - 1] + D); }
    d[o] = v;
  }
  const w = new Float32Array(n * n);
  for (let k = 0; k < w.length; k++) { const x = Math.min(1, d[k] / ramp); w[k] = x * x * (3 - 2 * x); }
  return w;
}
// Mip levels as interleaved RG: coverage-weighted 2 x 2 averages in linear albedo (code^2), re-encoded as sqrt.
function mips(c0, n, w0) {
  const out = []; let lin = new Float32Array(n * n), w = w0;
  for (let k = 0; k < lin.length; k++) { const v = c0[k] / 255; lin[k] = v * v; }
  for (;;) {
    const rg = new Uint8Array(n * n * 2);
    for (let k = 0; k < n * n; k++) { rg[2 * k] = Math.round(Math.sqrt(lin[k]) * 255 * w[k]); rg[2 * k + 1] = Math.round(w[k] * 255); }
    out.push(rg); if (n === 1) break;
    const m = n >> 1, nl = new Float32Array(m * m), nw = new Float32Array(m * m);
    for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) {
      const o = 2 * j * n + 2 * i, a = w[o], b = w[o + 1], c = w[o + n], d = w[o + n + 1], ws = a + b + c + d;
      nl[j * m + i] = ws > 0 ? (lin[o] * a + lin[o + 1] * b + lin[o + n] * c + lin[o + n + 1] * d) / ws : 0; nw[j * m + i] = ws / 4;
    }
    lin = nl; w = nw; n = m;
  }
  return out;
}
async function heights(url) {
  const r = await fetch(url); if (!r.ok) return null;
  const st = r.body.pipeThrough(new DecompressionStream("deflate")), raw = new Uint8Array(await new Response(st).arrayBuffer());
  const d = new Int16Array(raw.buffer, 0, raw.byteLength >> 1), n = 129, h = new Float32Array(n * n);
  for (let j = 0; j < n; j++) { let acc = 0; for (let i = 0; i < n; i++) { acc += d[j * n + i]; const q = acc & 0xffff; h[j * n + i] = q === 65535 ? NaN : q / 100; } }
  return h;
}
// Quantile matching of tile codes (o2, 1 m) against l0_a codes (2 m) over the L0 pixels both cover.
async function fit(m) {
  const a = readBitmap(m.bitmap), W = m.crop.w, r = Math.max(1, Math.round(m.res)), hx = new Float64Array(1024), hy = new Float64Array(1024);
  let n = 0;
  for (const t of m.tiles) {
    const c = await decode(t.url); if (!c) continue;
    const s = Math.round(Math.sqrt(c.length));
    for (let q = 0; q < m.crop.h; q++) for (let p = 0; p < W; p++) {
      const y = a[q * W + p]; if (y < 16) continue;                     // l0_a no data or nearly black
      // L0 pixel (crop.p0 + p, crop.q0 + q) covers x0 + res (p0 + p) - res / 2 .. + res metres; o2 pixel k covers tile start + k .. + 1 m
      const ox = Math.round(m.x0 + r * (m.crop.p0 + p) - r / 2 - t.x), oz = Math.round(m.z0 + r * (m.crop.q0 + q) - r / 2 - t.z);
      if (ox < 0 || oz < 0 || ox + r > s || oz + r > s) continue;
      let sum = 0, ok = true;
      for (let b = 0; b < r && ok; b++) for (let a2 = 0; a2 < r; a2++) { const v = c[(oz + b) * s + ox + a2]; if (!v) { ok = false; break; } sum += v * v; }
      if (!ok) continue;
      const x = Math.sqrt(sum / (r * r)) / 255;
      hx[Math.min(1023, Math.floor(x * 1024))]++; hy[Math.min(1023, Math.floor(y / 255 * 1024))]++; n++;
    }
  }
  if (n < 5000) return { n, lut: null };
  const cdf = (h) => { const c = new Float64Array(1025); for (let k = 0; k < 1024; k++) c[k + 1] = c[k] + h[k] / n; return c; };
  const Cx = cdf(hx), Cy = cdf(hy);
  const inv = (C, u) => { let k = 0; while (k < 1023 && C[k + 1] < u) k++; const d = C[k + 1] - C[k]; return (k + (d > 0 ? (u - C[k]) / d : 0.5)) / 1024; };
  const at = (C, v) => { const f = Math.min(1024, Math.max(0, v * 1024)), k = Math.min(1023, Math.floor(f)); return C[k] + (C[k + 1] - C[k]) * (f - k); };
  // knots inside the 0.5..99.5 % range of the tile values are matched directly; outside it the curve continues with
  // the slope between the 5 % and 95 % quantiles
  const lo = inv(Cx, 0.005), hi = inv(Cx, 0.995), x5 = inv(Cx, 0.05), x95 = inv(Cx, 0.95);
  const slope = (inv(Cy, 0.95) - inv(Cy, 0.05)) / Math.max(1e-3, x95 - x5);
  const map = (t) => t < lo ? inv(Cy, at(Cx, lo)) - (lo - t) * slope : t > hi ? inv(Cy, at(Cx, hi)) + (t - hi) * slope : inv(Cy, at(Cx, t));
  const lut = []; for (let k = 0; k <= 16; k++) lut.push(Math.min(1, Math.max(0, map(k / 16))));
  for (let k = 1; k <= 16; k++) lut[k] = Math.max(lut[k], lut[k - 1]);
  return { n, lut, quantiles: [0.05, 0.25, 0.5, 0.75, 0.95].map((u) => [u, +inv(Cx, u).toFixed(4), +inv(Cy, u).toFixed(4)]) };
}
onmessage = async (e) => {
  const m = e.data, t0 = performance.now();
  try {
    if (m.type === "tile") {
      const c = await decode(m.url), n = c ? Math.round(Math.sqrt(c.length)) : 0, lv = c ? mips(c, n, coverage(c, n, m.open, m.ramp)) : null;
      postMessage({ id: m.id, levels: lv, ms: performance.now() - t0 }, lv ? lv.map((a) => a.buffer) : []);
    } else if (m.type === "h") {
      const h = await heights(m.url); postMessage({ id: m.id, h, ms: performance.now() - t0 }, h ? [h.buffer] : []);
    } else if (m.type === "fit") {
      postMessage({ id: m.id, fit: await fit(m), ms: performance.now() - t0 });
    }
  } catch (err) { postMessage({ id: m.id, error: String(err) }); }
};`;

export const glsl = /* glsl */`
uniform sampler2D groundTex0; uniform sampler2D groundTex1;
uniform vec3 groundKey0[16]; uniform vec3 groundKey1[16];   // tile (i, j) held by each slot, and its fade-in (0..1)
uniform vec4 groundWin0; uniform vec4 groundWin1;           // window of each level: x0, z0, x1, z1 (metres)
uniform vec4 groundFade;                                     // camera distance where level 0 fades out (x..y) and level 1 (z..w)
uniform float groundLut[17];                                 // tile code -> l0_a code, knots at 0, 1/16, .. 1
float groundWeight(vec3 key, vec2 tile, vec4 win, vec2 p) {
  float k = abs(key.x - tile.x) + abs(key.y - tile.y) < 0.5 ? key.z : 0.0;
  vec2 e = min(p - win.xy, win.zw - p);
  return k * smoothstep(0.0, 16.0, min(e.x, e.y));
}
// Albedo of the 25 cm (near) and 0.5 m (further) clipmaps at world point p, on the scale of the page's l0_a.jpg.
// valid = 0 where neither level has data; the caller blends with its own albedo: mix(own, groundAlbedo(..), valid).
float groundAlbedo(vec2 p, float dist, out float valid) {
  vec2 i0 = floor(p / 128.0), s0 = mod(i0, 4.0), i1 = floor(p / 256.0), s1 = mod(i1, 4.0);
  vec2 e0 = texture2D(groundTex0, p / 512.0).rg, e1 = texture2D(groundTex1, p / 1024.0).rg;   // r = code * coverage, g = coverage
  float t0 = e0.r / max(e0.g, 0.004), t1 = e1.r / max(e1.g, 0.004);
  float w0 = groundWeight(groundKey0[int(s0.y * 4.0 + s0.x)], i0, groundWin0, p) * (1.0 - smoothstep(groundFade.x, groundFade.y, dist)) * e0.g;
  float w1 = groundWeight(groundKey1[int(s1.y * 4.0 + s1.x)], i1, groundWin1, p) * (1.0 - smoothstep(groundFade.z, groundFade.w, dist)) * e1.g;
  valid = w0 + (1.0 - w0) * w1;
  float t = (w0 * t0 + (1.0 - w0) * w1 * t1) / max(valid, 1e-4);
  float f = clamp(t, 0.0, 1.0) * 16.0; int k = int(min(f, 15.0));
  return mix(groundLut[k], groundLut[k + 1], f - float(k));
}
`;

export function createGround({ THREE, renderer, baseUrl, tilesJson, meta, albedo2m = null, heightFallback = null, budgetMs = 1, fitAt = null }) {
  const base = new URL(baseUrl.endsWith("/") ? baseUrl : baseUrl + "/", location.href).href;   // workers made from a blob URL cannot resolve relative URLs
  const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }));
  const workers = [new Worker(url), new Worker(url)];
  let nextId = 1, rr = 0; const waiting = new Map();
  for (const w of workers) w.onmessage = (e) => { const cb = waiting.get(e.data.id); waiting.delete(e.data.id); if (cb) cb(e.data); };
  const ask = (msg, transfer = []) => new Promise((res) => { const id = nextId++; waiting.set(id, res); workers[rr++ % workers.length].postMessage({ ...msg, id }, transfer); });

  const stats = { uploads: [], decodes: [], heights: 0, fit: null, updateMs: [] };
  const zero = new Uint8Array(PX * N * PX * N * 2);
  const zeroLevels = []; for (let k = 0; k < MIPS; k++) zeroLevels.push(zero.subarray(0, (PX >> k) * (PX >> k) * 2));
  const src = new THREE.DataTexture(null, 1, 1, THREE.RGFormat, THREE.UnsignedByteType);   // never uploaded; carries data into copyTextureToTexture
  const dstPos = new THREE.Vector2();

  function makeLevel(dir, tileM, keys, fade) {
    const size = PX * N, mipmaps = [];
    for (let k = 0; k < MIPS; k++) mipmaps.push({ data: zero.subarray(0, (size >> k) * (size >> k) * 2), width: size >> k, height: size >> k });
    const tex = new THREE.DataTexture(mipmaps[0].data, size, size, THREE.RGFormat, THREE.UnsignedByteType);
    tex.mipmaps = mipmaps; tex.generateMipmaps = false; tex.flipY = false; tex.unpackAlignment = 1; tex.colorSpace = THREE.NoColorSpace;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.anisotropy = Math.min(16, renderer.capabilities.getMaxAnisotropy()); tex.needsUpdate = true;
    renderer.initTexture(tex);
    const keyU = []; for (let s = 0; s < N * N; s++) keyU.push(new THREE.Vector3(1e6, 1e6, 0));
    return { dir, tileM, have: new Set(keys), fade, tex, keyU, win: new THREE.Vector4(), i0: null, j0: null,
      slots: Array.from({ length: N * N }, () => ({ key: null, i: 0, j: 0, t: -1 })), cache: new Map(), loading: new Set() };
  }
  const levels = [makeLevel("o0", T0, Object.keys(tilesJson.h), [150, 175]), makeLevel("o1", 2 * T0, tilesJson.o1 || [], [300, 340])];
  const lut = []; for (let k = 0; k <= 16; k++) lut.push(k / 16);    // identity until the fit arrives
  const uniforms = {
    groundTex0: { value: levels[0].tex }, groundTex1: { value: levels[1].tex },
    groundKey0: { value: levels[0].keyU }, groundKey1: { value: levels[1].keyU },
    groundWin0: { value: levels[0].win }, groundWin1: { value: levels[1].win },
    groundFade: { value: new THREE.Vector4(...levels[0].fade, ...levels[1].fade) }, groundLut: { value: lut },
  };

  // ---------- brightness fit against the page's l0_a.jpg ----------
  const fitDone = (async () => {
    if (!albedo2m || !tilesJson.o2 || !meta) { console.warn("ground: no 2 m albedo given, tile brightness left unmapped"); return null; }
    const L0 = meta.L0, [fx, fz] = fitAt || [meta.pois[0].x, meta.pois[0].z], S = 4 * T0;
    const near = tilesJson.o2.map((k) => k.split("_").map(Number)).sort((a, b) =>
      Math.hypot((a[0] + 0.5) * S - fx, (a[1] + 0.5) * S - fz) - Math.hypot((b[0] + 0.5) * S - fx, (b[1] + 0.5) * S - fz)).slice(0, 9);
    if (!near.length) return null;
    const xa = Math.min(...near.map((t) => t[0])) * S, xb = (Math.max(...near.map((t) => t[0])) + 1) * S;
    const za = Math.min(...near.map((t) => t[1])) * S, zb = (Math.max(...near.map((t) => t[1])) + 1) * S;
    const p0 = Math.max(0, Math.floor((xa - L0.x0) / L0.res)), p1 = Math.min(L0.w, Math.ceil((xb - L0.x0) / L0.res) + 1);
    const q0 = Math.max(0, Math.floor((za - L0.z0) / L0.res)), q1 = Math.min(L0.h, Math.ceil((zb - L0.z0) / L0.res) + 1);
    if (p1 <= p0 || q1 <= q0) return null;
    const bitmap = await createImageBitmap(albedo2m, p0, q0, p1 - p0, q1 - q0);
    const r = await ask({ type: "fit", bitmap, crop: { p0, q0, w: p1 - p0, h: q1 - q0 }, x0: L0.x0, z0: L0.z0, res: L0.res,
      tiles: near.map(([i, j]) => ({ url: `${base}o2/${i}_${j}.jpg`, x: i * S, z: j * S })) }, [bitmap]);
    stats.fit = r.fit ? { n: r.fit.n, lut: r.fit.lut && r.fit.lut.map((v) => +v.toFixed(4)), quantiles: r.fit.quantiles, ms: Math.round(r.ms) } : { error: r.error };
    if (r.fit && r.fit.lut) for (let k = 0; k <= 16; k++) lut[k] = r.fit.lut[k];
    else console.warn("ground: brightness fit failed", r);
    return stats.fit;
  })();

  // ---------- streaming ----------
  const mod = (a) => ((a % N) + N) % N;
  function windowFor(L, i0, j0, x, z) {             // keep (i0, j0) while the walker is far enough from its edges
    const m = L.tileM, k = KEEP * m;
    const ok = (a, v) => a !== null && v - a * m >= k && (a + N) * m - v >= k;
    return [ok(i0, x) ? i0 : Math.round(x / m) - 2, ok(j0, z) ? j0 : Math.round(z / m) - 2];
  }
  function request(L, key) {
    if (L.cache.has(key) || L.loading.has(key)) return;
    if (!L.have.has(key)) { L.cache.set(key, null); return; }        // outside the data: an empty tile
    L.loading.add(key);
    const [i, j] = key.split("_").map(Number), open = [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]].map(([a, b]) => !L.have.has(`${a}_${b}`));
    ask({ type: "tile", url: `${base}${L.dir}/${key}.jpg`, open, ramp: RAMP * PX / L.tileM }).then((r) => {
      L.loading.delete(key); if (r.ms) stats.decodes.push(r.ms);
      L.cache.set(key, r.levels || null);
      for (const old of L.cache.keys()) {                              // drop the least recently used, except tiles still to upload
        if (L.cache.size <= 28) break;
        if (!L.slots.some((s) => s.key === old && s.t < 0)) L.cache.delete(old);
      }
    });
  }
  function upload(L, s, levelsData) {
    const sl = L.slots[s], sx = mod(sl.i), sz = mod(sl.j);
    for (let k = 0; k < MIPS; k++) {
      const n = PX >> k; src.image = { data: levelsData ? levelsData[k] : zeroLevels[k], width: n, height: n };
      dstPos.set(sx * n, sz * n); renderer.copyTextureToTexture(src, L.tex, null, dstPos, 0, k);
    }
  }
  const hcache = new Map(), hloading = new Set();
  function requestH(i, j) {
    const key = `${i}_${j}`;
    if (hcache.has(key) || hloading.has(key)) return;
    if (!(key in tilesJson.h)) { hcache.set(key, null); return; }
    hloading.add(key);
    ask({ type: "h", url: `${base}h/${key}.bin` }).then((r) => {
      hloading.delete(key); stats.heights++;
      if (r.h) { const lo = tilesJson.h[key]; for (let k = 0; k < r.h.length; k++) r.h[k] += lo; }
      hcache.set(key, r.h || null);
      while (hcache.size > 36) hcache.delete(hcache.keys().next().value);
    });
  }

  let px = null, pz = null, pt = 0, vx = 0, vz = 0;
  function update(x, z) {
    const now = performance.now(), t0 = now;
    if (px !== null) {                               // smoothed velocity, for prefetching; jumps (teleports) reset it
      const dt = Math.max(1e-3, (now - pt) / 1000), dx = x - px, dz = z - pz;
      if (Math.hypot(dx, dz) > 50) { vx = vz = 0; } else { const a = Math.min(1, dt * 2); vx += (dx / dt - vx) * a; vz += (dz / dt - vz) * a; }
    }
    px = x; pz = z; pt = now;
    const pending = [];
    for (const L of levels) {
      const [i0, j0] = windowFor(L, L.i0, L.j0, x, z);
      if (i0 !== L.i0 || j0 !== L.j0) {
        L.i0 = i0; L.j0 = j0; L.win.set(i0 * L.tileM, j0 * L.tileM, (i0 + N) * L.tileM, (j0 + N) * L.tileM);
        for (let j = j0; j < j0 + N; j++) for (let i = i0; i < i0 + N; i++) {
          const s = mod(j) * N + mod(i), key = `${i}_${j}`;
          if (L.slots[s].key !== key) { L.slots[s] = { key, i, j, t: -1 }; L.keyU[s].set(i, j, 0); }
        }
      }
      // decode the window, nearest tiles first, then the window the walker is heading for
      const want = [];
      for (const sl of L.slots) if (sl.t < 0) want.push([Math.hypot((sl.i + 0.5) * L.tileM - x, (sl.j + 0.5) * L.tileM - z), sl.key]);
      want.sort((a, b) => a[0] - b[0]); for (const [, key] of want) request(L, key);
      const ahead = Math.min(L.tileM, Math.hypot(vx, vz) * 8), sp = Math.hypot(vx, vz) || 1;
      if (ahead > 2) {
        const ax = x + vx / sp * ahead, az = z + vz / sp * ahead, [a0, b0] = windowFor(L, L.i0, L.j0, ax, az);
        for (let j = b0; j < b0 + N; j++) for (let i = a0; i < a0 + N; i++) request(L, `${i}_${j}`);
      }
      for (let s = 0; s < N * N; s++) {
        const sl = L.slots[s];
        if (sl.t < 0 && L.cache.has(sl.key)) pending.push([Math.hypot((sl.i + 0.5) * L.tileM - x, (sl.j + 0.5) * L.tileM - z) * (L === levels[0] ? 1 : 1.5), L, s]);
        else if (sl.t >= 0) L.keyU[s].z = Math.min(1, (now - sl.t) / 250);   // fade a new tile in over 250 ms
      }
    }
    pending.sort((a, b) => a[0] - b[0]);
    let n = 0;
    for (const [, L, s] of pending) {
      if (n >= 2 || performance.now() - t0 > budgetMs) break;
      const sl = L.slots[s], u0 = performance.now(), data = L.cache.get(sl.key);
      upload(L, s, data); sl.t = now; n++;
      L.cache.delete(sl.key); L.cache.set(sl.key, data);                 // most recently used last
      stats.uploads.push(performance.now() - u0);
    }
    // 1 m heights: the 3 x 3 tiles around the walker and the tile ahead
    const ti = Math.floor(x / T0), tj = Math.floor(z / T0);
    for (let j = tj - 1; j <= tj + 1; j++) for (let i = ti - 1; i <= ti + 1; i++) requestH(i, j);
    const sp = Math.hypot(vx, vz);
    if (sp > 0.3) requestH(Math.floor((x + vx / sp * 96) / T0), Math.floor((z + vz / sp * 96) / T0));
    stats.updateMs.push(performance.now() - t0);
    if (stats.updateMs.length > 4000) stats.updateMs.splice(0, 2000);
  }

  // Ground height (metres above meta.site.base) from the 1 m DTM tiles, bilinear; heightFallback(x, z) (or NaN) where
  // the tile is not loaded or has no data.
  function heightAt(x, z) {
    const i = Math.floor(x / T0), j = Math.floor(z / T0), h = hcache.get(`${i}_${j}`);
    if (!h) return heightFallback ? heightFallback(x, z) : NaN;
    const u = x - i * T0, v = z - j * T0, a = Math.min(T0 - 1, Math.floor(u)), b = Math.min(T0 - 1, Math.floor(v)), fu = u - a, fv = v - b, o = b * 129 + a;
    const h00 = h[o], h10 = h[o + 1], h01 = h[o + 129], h11 = h[o + 130];
    const r = (h00 * (1 - fu) + h10 * fu) * (1 - fv) + (h01 * (1 - fu) + h11 * fu) * fv;
    return Number.isFinite(r) ? r : (heightFallback ? heightFallback(x, z) : NaN);
  }
  // number of window slots (both levels) not yet uploaded
  const pendingCount = () => levels.reduce((n, L) => n + L.slots.filter((s) => s.t < 0).length, 0);
  function dispose() { for (const w of workers) w.terminate(); URL.revokeObjectURL(url); for (const L of levels) L.tex.dispose(); }

  return { update, uniforms, glsl, heightAt, fitDone, pendingCount, stats, dispose };
}
