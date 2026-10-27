// Candor Chasma on foot: first-person walk on HiRISE terrain.
// World frame: x east, z south, y up, metres. Heights are relative to meta.site.base.
import * as THREE from "three";
import { stepWalker, newWalker, stepLength, WALK, LEG } from "./physics/player.js";
import { createAudio } from "./sound/audio.js";
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  if (document.documentElement.classList.contains("no-walk")) return;   // phones and tablets: index.html shows #desktop-only
  const params = new URLSearchParams(location.search);
  const DEBUG = params.has("debug");
  // ?site=candor|noctis|gale loads ../sites/<name>/; without it, the Melas data in data/
  const SITE = (params.get("site") || "").replace(/[^a-z0-9_-]/gi, "");
  const DATA = "./sites/" + (SITE || "candor") + "/";       // Candor is the site of the published walk
  const isTouch = window.matchMedia("(pointer: coarse)").matches;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (isTouch) { document.body.classList.add("touch"); $("controls-desk").hidden = true; $("controls-touch").hidden = false; }

  // ---------------- loading ----------------
  const progress = { got: {}, total: {} };
  const stepOf = (url) => /l0_(h|aux)\./.test(url) ? "dtm" : /l0_a\.|tile/.test(url) ? "photo" : "around";
  const mb = (b) => (b / 1e6).toFixed(1);
  function setStep(k, state, text) {
    const li = document.querySelector('#load-steps li[data-step="' + k + '"]'); if (!li) return;
    li.dataset.state = state; if (text != null) li.querySelector(".n").textContent = text;
  }
  function showProgress() {
    let g = 0, t = 0; const per = {};
    for (const k in progress.total) {
      const s = per[stepOf(k)] || (per[stepOf(k)] = { g: 0, t: 0 }), got = progress.got[k] || 0;
      s.g += got; s.t += progress.total[k]; g += got; t += progress.total[k];
    }
    if (t) $("bar").firstElementChild.style.width = Math.min(100, (g / t) * 100).toFixed(1) + "%";
    for (const k in per) { const s = per[k]; setStep(k, s.g >= s.t ? "done" : "active", s.g >= s.t ? mb(s.t) + " MB" : mb(s.g) + " of " + mb(s.t) + " MB"); }
  }
  async function fetchBytes(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Could not load " + url + " (HTTP " + res.status + ").");
    progress.total[url] = +res.headers.get("Content-Length") || 1;
    const reader = res.body.getReader(), chunks = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      chunks.push(value); got += value.length; progress.got[url] = got; showProgress();
    }
    const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }
  async function inflate(u8) {
    const st = new Blob([u8]).stream().pipeThrough(new DecompressionStream("deflate"));
    return new Uint8Array(await new Response(st).arrayBuffer());
  }
  async function loadHeights(L) {
    const raw = await inflate(await fetchBytes(DATA + L.file));
    const d = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength >> 1);
    const w = L.w, h = L.h, out = new Float32Array(w * h);
    for (let j = 0; j < h; j++) {
      let acc = 0; const o = j * w;
      for (let i = 0; i < w; i++) { acc += d[o + i]; out[o + i] = ((acc & 0xffff) + L.offset) * L.unit; }
    }
    return out;
  }
  async function loadBitmap(file) {
    const bytes = await fetchBytes(DATA + file);
    return createImageBitmap(new Blob([bytes], { type: "image/jpeg" }));
  }

  // ---------------- terrain sampling (CPU) ----------------
  function bilinear(g, arr, x, z) {
    let fx = (x - g.x0) / g.res, fz = (z - g.z0) / g.res;
    fx = fx < 0 ? 0 : fx > g.w - 1.0001 ? g.w - 1.0001 : fx;
    fz = fz < 0 ? 0 : fz > g.h - 1.0001 ? g.h - 1.0001 : fz;
    const i = fx | 0, j = fz | 0, tx = fx - i, tz = fz - j, o = j * g.w + i;
    return (arr[o] * (1 - tx) + arr[o + 1] * tx) * (1 - tz) + (arr[o + g.w] * (1 - tx) + arr[o + g.w + 1] * tx) * tz;
  }
  const smooth = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
  let G0, G1, G2, H0, H1, H2, MASK0, ROCKK, BLEND0;
  let L1band = 1500;
  function inGrid(g, x, z) { return x >= g.x0 && z >= g.z0 && x <= g.x0 + (g.w - 1) * g.res && z <= g.z0 + (g.h - 1) * g.res; }
  function weight1(x, z) {
    const ex = Math.min(x - G1.x0, G1.x0 + (G1.w - 1) * G1.res - x), ez = Math.min(z - G1.z0, G1.z0 + (G1.h - 1) * G1.res - z);
    return smooth(Math.min(ex, ez) / L1band);
  }
  function mask0(x, z) { return inGrid(G0, x, z) ? bilinear(G0, MASK0, x, z) : 0; }
  function height1(x, z) {             // L1 blended over L2
    const h2 = bilinear(G2, H2, x, z), w = weight1(x, z);
    return w > 0 ? h2 + (bilinear(G1, H1, x, z) - h2) * w : h2;
  }
  function heightAt(x, z) {            // the surface the player stands on (matches the L0 mesh)
    return inGrid(G0, x, z) ? bilinear(G0, BLEND0, x, z) : height1(x, z);
  }
  // Height of relief finer than the elevation data, the two coarsest octaves of microSlope in the terrain shader.
  const fract = (v) => v - Math.floor(v);
  function h12(x, y) {                 // same hash as the shader's h12
    let a = fract(x * 0.1031), b = fract(y * 0.1031), c = a;
    const q = a * (b + 33.33) + b * (c + 33.33) + c * (a + 33.33); a += q; b += q; c += q; return fract((a + b) * c);
  }
  function vnoise2(x, y) {
    const i = Math.floor(x), j = Math.floor(y), f = x - i, g = y - j, u = f * f * (3 - 2 * f), v = g * g * (3 - 2 * g);
    const a = h12(i, j), b = h12(i + 1, j), c = h12(i, j + 1), d = h12(i + 1, j + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  }
  function microHeight(x, z) {
    if (!inGrid(G0, x, z)) return 0;
    const rough = 0.55 + 0.85 * Math.min(1, bilinear(G0, ROCKK, x, z) / 0.12);
    const h0 = vnoise2(x / 2.4, z / 2.4), L = 2.4 * 0.42, xr = 0.8 * x - 0.6 * z, zr = 0.6 * x + 0.8 * z;
    const h1 = vnoise2(xr / L + 17.31, zr / L + 17.31);
    return rough * 0.09 * (2.4 * (h0 - 0.5) + L * (h1 - 0.5));
  }
  function slopeAt(x, z) {
    const e = 1.5, gx = (heightAt(x + e, z) - heightAt(x - e, z)) / (2 * e), gz = (heightAt(x, z + e) - heightAt(x, z - e)) / (2 * e);
    return { gx, gz, deg: Math.atan(Math.hypot(gx, gz)) * 57.2958 };
  }

  // ---------------- terrain meshes ----------------
  let R = 3396190;
  const curve = (x, z) => (x * x + z * z) / (2 * R);
  // Builds the vertex and index arrays of one terrain chunk. It uses no variables from outside, so the same code runs on the
  // main thread and (as source text) in the chunk worker.
  // g = {w, h, res, x0, z0}; hv = Float32Array of heights (index j * w + i); c = the chunk's {i0, j0, i1, j1, cx, cz};
  // uvO = [originX, originZ, width, height] of the albedo map or null; skipR = [x0, x1, z0, z1, strict] or null: a cell or
  // skirt segment is left out where its bounds lie inside that rectangle; R = planet radius for the curvature drop.
  function buildChunkArrays(g, hv, c, step, uvO, skipR, skirt, noNormals, R) {
    const is = [], js = [];
    for (let i = c.i0; i < c.i1; i += step) is.push(i); is.push(c.i1);
    for (let j = c.j0; j < c.j1; j += step) js.push(j); js.push(c.j1);
    const nx = is.length, nz = js.length, vmax = nx * nz + 2 * (nx + nz);
    const pos = new Float32Array(vmax * 3), nor = noNormals ? null : new Float32Array(vmax * 3), uv = uvO ? new Float32Array(vmax * 2) : null;
    const idx = new (vmax > 65535 ? Uint32Array : Uint16Array)(((nx - 1) * (nz - 1) + 2 * (nx + nz)) * 6);
    const clampI = (i) => (i < 0 ? 0 : i > g.w - 1 ? g.w - 1 : i), clampJ = (j) => (j < 0 ? 0 : j > g.h - 1 ? g.h - 1 : j);
    const X = (i) => g.x0 + i * g.res, Z = (j) => g.z0 + j * g.res, HV = (i, j) => hv[j * g.w + i];
    let skip = null;
    if (skipR) {
      const [sx0, sx1, sz0, sz1, strict] = skipR;
      skip = strict ? (xa, za, xb, zb) => xa > sx0 && xb < sx1 && za > sz0 && zb < sz1 : (xa, za, xb, zb) => xa >= sx0 && xb <= sx1 && za >= sz0 && zb <= sz1;
    }
    let v = 0, n = 0;
    function put(i, j, drop) {
      const x = X(i), z = Z(j);
      pos[v * 3] = x - c.cx; pos[v * 3 + 1] = HV(i, j) - (x * x + z * z) / (2 * R) - drop; pos[v * 3 + 2] = z - c.cz;
      if (nor) {
        const H = (a, b) => HV(clampI(a), clampJ(b));
        const dx = H(i - step, j) - H(i + step, j), dz = H(i, j - step) - H(i, j + step), dy = 2 * step * g.res, l = Math.hypot(dx, dy, dz);
        nor[v * 3] = dx / l; nor[v * 3 + 1] = dy / l; nor[v * 3 + 2] = dz / l;
      }
      if (uv) { uv[v * 2] = (x - uvO[0]) / uvO[2]; uv[v * 2 + 1] = (z - uvO[1]) / uvO[3]; }
      return v++;
    }
    for (let b = 0; b < nz; b++) for (let a = 0; a < nx; a++) put(is[a], js[b], 0);
    for (let b = 0; b < nz - 1; b++) for (let a = 0; a < nx - 1; a++) {
      if (skip && skip(X(is[a]), Z(js[b]), X(is[a + 1]), Z(js[b + 1]))) continue;
      const p = b * nx + a, q = p + 1, r = p + nx, t = r + 1;
      idx[n++] = p; idx[n++] = r; idx[n++] = q; idx[n++] = q; idx[n++] = r; idx[n++] = t;
    }
    if (skirt) {                                              // vertical strips hanging from the chunk border hide cracks between levels
      const d = skirt * step * g.res;
      for (const [b, horiz] of [[0, true], [nz - 1, true], [0, false], [nx - 1, false]]) {
        const list = horiz ? is : js, s0 = v;
        for (let k = 0; k < list.length; k++) horiz ? put(list[k], js[b], d) : put(is[b], list[k], d);
        for (let k = 0; k < list.length - 1; k++) {
          if (skip && (horiz ? skip(X(list[k]), Z(js[b]), X(list[k + 1]), Z(js[b])) : skip(X(is[b]), Z(list[k]), X(is[b]), Z(list[k + 1])))) continue;
          const top = horiz ? b * nx + k : k * nx + b, top2 = horiz ? b * nx + k + 1 : (k + 1) * nx + b;
          idx[n++] = top; idx[n++] = top2; idx[n++] = s0 + k; idx[n++] = top2; idx[n++] = s0 + k + 1; idx[n++] = s0 + k;
        }
      }
    }
    if (!n) return null;
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;   // bounding sphere, as three computes it
    for (let k = 0; k < v * 3; k += 3) {
      if (pos[k] < x0) x0 = pos[k]; if (pos[k] > x1) x1 = pos[k];
      if (pos[k + 1] < y0) y0 = pos[k + 1]; if (pos[k + 1] > y1) y1 = pos[k + 1];
      if (pos[k + 2] < z0) z0 = pos[k + 2]; if (pos[k + 2] > z1) z1 = pos[k + 2];
    }
    const sx = (x0 + x1) / 2, sy = (y0 + y1) / 2, sz = (z0 + z1) / 2; let r2 = 0;
    for (let k = 0; k < v * 3; k += 3) { const dx = pos[k] - sx, dy = pos[k + 1] - sy, dz = pos[k + 2] - sz, d2 = dx * dx + dy * dy + dz * dz; if (d2 > r2) r2 = d2; }
    return { pos: pos.slice(0, v * 3), nor: nor ? nor.slice(0, v * 3) : null, uv: uv ? uv.slice(0, v * 2) : null, idx: idx.slice(0, n), sphere: [sx, sy, sz, Math.sqrt(r2)] };
  }
  function geometryFromArrays(r) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(r.pos, 3));
    if (r.nor) geo.setAttribute("normal", new THREE.BufferAttribute(r.nor, 3));
    if (r.uv) geo.setAttribute("uv", new THREE.BufferAttribute(r.uv, 2));
    geo.setIndex(new THREE.BufferAttribute(r.idx, 1));
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(r.sphere[0], r.sphere[1], r.sphere[2]), r.sphere[3]);
    return geo;
  }
  // Chunks are built in one Web Worker so a build never stalls a frame. The worker gets each layer's height array once,
  // then builds on request and sends the arrays back as transferable buffers. Without a working Worker the same builds
  // run on the main thread, a few per frame.
  const chunkBuilder = (() => {
    function workerMain(build) {
      const grids = [];
      self.onmessage = (e) => {
        const m = e.data;
        if (m.type === "init") { m.layers.forEach((L, k) => { grids[k] = L; }); return; }
        const t = performance.now(), r = build(grids[m.layer].g, grids[m.layer].hv, m.c, m.step, m.uvO, m.skipR, m.skirt, m.noNormals, m.R);
        const tr = r ? [r.pos.buffer, r.idx.buffer].concat(r.nor ? [r.nor.buffer] : [], r.uv ? [r.uv.buffer] : []) : [];
        self.postMessage({ id: m.id, r, ms: performance.now() - t }, tr);
      };
    }
    let worker = null, grids = null; const jobs = new Map(); let nextId = 1;
    const MAX_INFLIGHT = 3;
    let rest = [], restPos = 0;                               // the latest priority queue, worked through as jobs finish
    function finish(job, r, ms) { jobs.delete(job.id); job.done(r, ms); fill(); }
    function runSync(job) {
      const L = grids[job.msg.layer], m = job.msg, t = performance.now();
      finish(job, buildChunkArrays(L.g, L.hv, m.c, m.step, m.uvO, m.skipR, m.skirt, m.noNormals, m.R), performance.now() - t);
    }
    function submit(q) {
      const job = { id: nextId++, msg: { layer: q.li, c: q.layer.chunkInfo(q.c), step: q.step, uvO: q.layer.o.uvO || null, skipR: q.layer.o.skipR || null,
        skirt: q.layer.o.skirt, noNormals: !!q.layer.o.noNormals, R }, done: (r, ms) => q.layer.built(q.c, q.step, r, ms) };
      q.c.pending[q.step] = true; jobs.set(job.id, job);
      if (worker) { job.msg.id = job.id; worker.postMessage(job.msg); } else runSync(job);
    }
    function fill() {
      while (worker && jobs.size < MAX_INFLIGHT && restPos < rest.length) {
        const q = rest[restPos++];
        if (!q.c.pending[q.step] && !q.c.meshes[q.step] && !q.c.empty[q.step]) submit(q);
      }
    }
    function fail(why) {                                      // the worker cannot run here: do its pending jobs on the main thread
      console.warn("Chunk worker unavailable, building terrain on the main thread:", why);
      if (worker) worker.terminate(); worker = null;
      for (const job of [...jobs.values()]) runSync(job);
    }
    return {
      start(list) {                                           // list = [{g, hv}] per layer
        grids = list;
        if (params.has("noworker")) return;                      // test switch for the main-thread fallback
        try {
          const src = "const build = " + buildChunkArrays.toString() + ";\n(" + workerMain.toString() + ")(build);";
          worker = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
          worker.onmessage = (e) => { const job = jobs.get(e.data.id); if (job) finish(job, e.data.r, e.data.ms); };
          worker.onerror = (e) => fail(e.message || "worker error");
          worker.postMessage({ type: "init", layers: list });
        } catch (err) { worker = null; }
      },
      busy: () => jobs.size,
      // queue = [{layer, li, c, step, d}] sorted by priority. With a worker, at most MAX_INFLIGHT jobs run at once and the
      // rest of the queue is sent as they finish; without one, builds run here until the deadline passes.
      pump(queue, deadline) {
        if (worker) { rest = queue; restPos = 0; fill(); return; }
        for (const q of queue) {
          if (performance.now() >= deadline) break;
          if (!q.c.meshes[q.step] && !q.c.empty[q.step]) submit(q);
        }
      },
    };
  })();

  // A layer is a grid cut into chunks; each chunk shows one level of detail chosen by distance.
  // Missing levels are built by the chunk worker and a coarser level is shown meanwhile.
  class Layer {
    constructor(scene, g, opts) {
      this.g = g; this.o = opts; this.chunks = []; this.pinned = false;   // pinned: keep fine levels that were built ahead for a drop
      const C = opts.chunk;
      for (let j0 = 0; j0 < g.h - 1; j0 += C) for (let i0 = 0; i0 < g.w - 1; i0 += C) {
        const i1 = Math.min(i0 + C, g.w - 1), j1 = Math.min(j0 + C, g.h - 1);
        const minx = g.x0 + i0 * g.res, maxx = g.x0 + i1 * g.res, minz = g.z0 + j0 * g.res, maxz = g.z0 + j1 * g.res;
        const group = new THREE.Group(); group.position.set((minx + maxx) / 2, 0, (minz + maxz) / 2); scene.add(group);
        this.chunks.push({ i0, j0, i1, j1, minx, maxx, minz, maxz, cx: group.position.x, cz: group.position.z, group, meshes: {}, empty: {}, pending: {} });
      }
      this.coarsest = opts.levels[opts.levels.length - 1][0];
      this.builds = [];                                       // [step, ms, time, cells wide, cells high] per build, for the debug harness
    }
    chunkInfo(c) { return { i0: c.i0, j0: c.j0, i1: c.i1, j1: c.j1, cx: c.cx, cz: c.cz }; }
    want(c, px, pz) {
      const dx = Math.max(c.minx - px, 0, px - c.maxx), dz = Math.max(c.minz - pz, 0, pz - c.maxz), d = Math.hypot(dx, dz);
      for (const [step, dist] of this.o.levels) if (d < dist) return [step, d];
      return [this.coarsest, d];
    }
    // Called when the worker (or the main-thread fallback) has finished a chunk.
    built(c, step, r, ms) {
      const t = performance.now();
      delete c.pending[step];
      if (DEBUG) this.builds.push([step, ms, Math.round(t), c.i1 - c.i0, c.j1 - c.j0]);
      if (!r) { c.empty[step] = true; return; }
      const mesh = new THREE.Mesh(geometryFromArrays(r), this.o.material); mesh.visible = false; c.group.add(mesh); c.meshes[step] = mesh;
      if (DEBUG) stats.asm.push(performance.now() - t);
    }
    // Chunks whose wanted level is missing are added to `queue`, nearest first once sorted. Returns true if all wanted levels exist.
    update(px, pz, queue, li) {
      let complete = true;
      for (const c of this.chunks) {
        const [want, d] = this.want(c, px, pz);
        if (!c.meshes[want] && !c.empty[want]) {
          complete = false;
          if (!c.pending[want]) queue.push({ layer: this, li, c, step: want, d });
        }
        let show = want;
        if (!c.meshes[want]) {   // fall back to the nearest existing level, preferring coarser
          show = null;
          for (const [s] of this.o.levels) if (s > want && c.meshes[s]) { show = s; break; }
          if (show === null) for (const [s] of this.o.levels.slice().reverse()) if (s < want && c.meshes[s]) { show = s; break; }
        }
        for (const s in c.meshes) c.meshes[s].visible = Number(s) === show;
        const finest = this.o.levels[0][0];
        if (!this.pinned && want !== finest && c.meshes[finest] && d > this.o.levels[0][1] * 2) {
          c.meshes[finest].geometry.dispose(); c.group.remove(c.meshes[finest]); delete c.meshes[finest];
        }
      }
      return complete;
    }
    // Like update, but only asks for the levels wanted at (px, pz): nothing shown or removed. Used while fading out for a drop.
    prefetch(px, pz, queue, li) {
      let complete = true;
      for (const c of this.chunks) {
        const [want, d] = this.want(c, px, pz);
        if (!c.meshes[want] && !c.empty[want]) { complete = false; if (!c.pending[want]) queue.push({ layer: this, li, c, step: want, d }); }
      }
      return complete;
    }
  }
  const byPriority = (a, b) => a.li - b.li || a.d - b.d;
  // Add the missing chunk levels around (px, pz) to `queue`; returns true if every wanted level exists.
  function collectLayers(px, pz, ahead, queue) {
    let complete = true;
    layers.forEach((l, li) => { if (!(ahead ? l.prefetch(px, pz, queue, li) : l.update(px, pz, queue, li))) complete = false; });
    return complete;
  }
  function updateLayers(px, pz, deadline) {
    const queue = [], complete = collectLayers(px, pz, false, queue);
    if (queue.length) { queue.sort(byPriority); chunkBuilder.pump(queue, deadline); }
    return complete && !chunkBuilder.busy();
  }

  // ---------------- shaders ----------------
  const terrainVS = `
    #include <common>
    #include <logdepthbuf_pars_vertex>
    varying vec3 vWorld; varying vec3 vNormal; varying vec2 vUv;
    void main() {
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorld = wp.xyz; vNormal = normal;
      #ifdef USE_MAPUV
      vUv = uv;
      #else
      vUv = vec2(0.0);
      #endif
      gl_Position = projectionMatrix * viewMatrix * wp;
      #include <logdepthbuf_vertex>
    }`;
  const terrainFS = `
    #include <common>
    #include <logdepthbuf_pars_fragment>
    uniform sampler2D map; uniform float hasMap; uniform float detail; uniform sampler2D gdet; uniform sampler2D nmap; uniform float hasNmap;
    uniform sampler2D map2; uniform float hasMap2; uniform vec4 map2X;
    uniform vec3 sunDir; uniform float sinEl; uniform float amb; uniform vec3 sunCol; uniform vec3 skyCol; uniform vec3 gndCol;
    uniform vec3 fogCol; uniform float fogDen; uniform vec3 camPos; uniform float pxAng;
    uniform sampler2DShadow shMap; uniform mat4 shMat; uniform float shOn; uniform float shTexel; uniform vec3 shZ;
    uniform vec3 rampA; uniform vec3 rampB; uniform vec3 rampC;
    varying vec3 vWorld; varying vec3 vNormal; varying vec2 vUv;
    // ground.glsl (replaced by terrainMaterial: the streamed 25 cm albedo of ../ground/ground.js, or nothing)
    float h12(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
    vec2 vnoiseGrad(vec2 p) {                               // gradient of value noise
      vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f), du = 6.0 * f * (1.0 - f);
      float a = h12(i), b = h12(i + vec2(1.0, 0.0)), c = h12(i + vec2(0.0, 1.0)), d = h12(i + vec2(1.0, 1.0));
      return du * (vec2(b - a, c - a) + (a - b - c + d) * u.yx);
    }
    // Slope of ground relief finer than the 2 m elevation data: five octaves, 2.4 m down to 7.5 cm,
    // each dropped once its wavelength covers fewer than ~3 pixels.
    vec2 microSlope(vec2 p, float dist) {
      vec2 g = vec2(0.0); float L = 2.4, px = dist * pxAng; mat2 M = mat2(1.0, 0.0, 0.0, 1.0); const mat2 Rot = mat2(0.8, 0.6, -0.6, 0.8);
      for (int k = 0; k < 5; k++) {
        float fade = clamp((L / px - 2.0) / 3.0, 0.0, 1.0);
        if (fade <= 0.0) break;                              // L only shrinks, so every finer octave is faded out too
        g += (vnoiseGrad(M * p / L + float(k) * 17.31) * M) * (0.09 * fade);
        M = Rot * M; L *= 0.42;
      }
      return g;
    }
    float mapShadow(sampler2DShadow m, mat4 M, float on, float texel, vec3 Z, vec3 w) {   // 1 = lit
      vec3 c = (M * vec4(w, 1.0)).xyz;
      vec2 uv = c.xy * 0.5 + 0.5;
      if (on < 0.5 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;
      float z = c.z * Z.x + Z.y + Z.z;                      // depth in 0..1 with a bias toward the sun
      vec2 o = vec2(0.5 * texel);                           // four hardware-filtered compares, each averaging 2 x 2 texels
      float lit = texture(m, vec3(uv + vec2(-o.x, -o.y), z)) + texture(m, vec3(uv + vec2(o.x, -o.y), z))
                + texture(m, vec3(uv + vec2(-o.x, o.y), z)) + texture(m, vec3(uv + vec2(o.x, o.y), z));
      vec2 e = abs(uv - 0.5) * 2.0;
      return mix(lit * 0.25, 1.0, smoothstep(0.8, 1.0, max(e.x, e.y)));
    }
    float rockShadow(vec3 w) { return mapShadow(shMap, shMat, shOn, shTexel, shZ, w); }   // rocks and pebbles near the player
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    float vnoise(vec2 p) {
      vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
      return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
    }
    void main() {
      #include <logdepthbuf_fragment>
      vec3 n; float strip = 1.0, rocky = 0.3;
      if (hasNmap > 0.5) { vec4 e = texture2D(nmap, vUv); vec2 q = e.rg * 2.0 - 1.0; n = normalize(vec3(q.x, sqrt(max(0.0, 1.0 - dot(q, q))), q.y)); strip = e.b; rocky = e.a; }
      else n = normalize(vNormal);
      float dist = length(vWorld - camPos);
      float slopeShade = 0.55 - (1.0 - n.y) * 0.8 + 0.08 * (vnoise(vWorld.xz * 0.0011) - 0.5) + 0.05 * (vnoise(vWorld.xz * 0.013) - 0.5);
      float g = slopeShade;
      if (hasMap > 0.5) {
        float t = texture2D(map, vUv).r, valid = smoothstep(0.02, 0.06, t);
        if (hasMap2 > 0.5) {                                 // outside the HiRISE strip, use the coarser layer's albedo
          float t2 = texture2D(map2, (vWorld.xz - map2X.xy) * map2X.zw).r;
          t = mix(t2, t, strip); valid = mix(smoothstep(0.02, 0.06, t2), valid, strip);
        }
        #ifdef GROUND
        float gv; float tg = groundAlbedo(vWorld.xz, dist, gv);   // streamed 25 cm / 0.5 m albedo near the walker
        t = mix(t, tg, gv); valid = mix(valid, 1.0, gv);
        #endif
        g = mix(slopeShade, t, valid);
      }
      g = clamp(g, 0.0, 1.0);
      vec3 alb = g < 0.5 ? mix(rampA, rampB, g * 2.0) : mix(rampB, rampC, g * 2.0 - 1.0);
      // Ground finer than the 25 cm photo: real Mars ground from InSight's arm camera (IDC workspace orthomosaic, sol 12,
      // 2 mm/px; Elysium Planitia, not Candor), kept only below about 8 cm and stored as a brightness ratio around 1 (see
      // tex/README.txt). Sampled at its true scale twice, the second rotated and shifted, so the 2 m tile does not repeat
      // visibly; the average of two has 1/sqrt(2) of the contrast, so it is scaled back by 1.41. Mipmaps fade it with distance.
      float near = (1.0 - smoothstep(30.0, 60.0, dist)) * step(0.001, detail);
      if (near > 0.0) {
        vec2 p = vWorld.xz * 0.5;
        float r = 0.5 * (texture2D(gdet, p).r + texture2D(gdet, mat2(0.8, -0.6, 0.6, 0.8) * p + vec2(0.37, 0.71)).r) + 0.5;
        alb *= mix(1.0, 1.0 + (r - 1.0) * 1.41, near);
      }
      vec2 ms = microSlope(vWorld.xz, dist) * mix(0.55, 1.4, rocky);   // rougher where the data shows more rocks
      n = normalize(n - vec3(ms.x, 0.0, ms.y));
      float direct = max(dot(n, sunDir), 0.0) / sinEl * rockShadow(vWorld);
      vec3 sky = mix(gndCol, skyCol, 0.5 + 0.5 * n.y);
      vec3 col = alb * (sunCol * (1.0 - amb) * direct + sky * amb);
      float f = 1.0 - exp(-dist * fogDen);
      col = mix(col, fogCol, clamp(f, 0.0, 0.97));
      gl_FragColor = vec4(col, 1.0);
    }`;
  const skyVS = `
    #include <common>
    #include <logdepthbuf_pars_vertex>
    varying vec3 vDir;
    void main() { vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      #include <logdepthbuf_vertex>
    }`;
  const skyFS = `
    #include <common>
    #include <logdepthbuf_pars_fragment>
    uniform vec3 sunDir; uniform vec3 horizon; uniform vec3 zenith; uniform vec3 halo; uniform vec3 below;
    varying vec3 vDir;
    void main() {
      #include <logdepthbuf_fragment>
      vec3 d = normalize(vDir);
      vec3 col = mix(horizon, zenith, pow(clamp(d.y, 0.0, 1.0), 0.55));
      col = mix(col, below, smoothstep(0.0, -0.08, d.y));
      float ang = acos(clamp(dot(d, sunDir), -1.0, 1.0));
      col += halo * (exp(-ang * 9.0) * 0.5 + exp(-ang * 2.2) * 0.12);
      col = mix(col, vec3(1.0, 0.99, 0.96), smoothstep(0.0058, 0.0040, ang));
      gl_FragColor = vec4(col, 1.0);
    }`;

  // ---------------- renderer and scene ----------------
  const canvas = $("scene");
  // three r163+ supports only WebGL 2 and throws here when it is missing, so the message is set here instead of in boot().
  // Depth: a reversed float depth buffer (needs EXT_clip_control) keeps precision from 0.1 m to 1.5e6 m. A logarithmic depth buffer
  // gives that too, but its shaders write gl_FragDepth, which turns off early-Z on Apple GPUs. Where the extension is missing,
  // or with ?logz, the logarithmic depth buffer is used as a fallback.
  const REVERSE_Z = !params.has("logz") && (() => {
    try {
      const gl = document.createElement("canvas").getContext("webgl2"), ok = !!(gl && gl.getExtension("EXT_clip_control"));
      if (gl) { const lose = gl.getExtension("WEBGL_lose_context"); if (lose) lose.loseContext(); }
      return ok;
    } catch (e) { return false; }
  })();
  // antialias is off because the scene is smoothed by camRT's 4x MSAA; the canvas only gets the lens pass's full-screen quad.
  const renderer = (() => {
    try { return new THREE.WebGLRenderer({ canvas, antialias: false, logarithmicDepthBuffer: !REVERSE_Z, reversedDepthBuffer: REVERSE_Z, powerPreference: "high-performance" }); }
    catch (e) { $("status").textContent = "This page needs WebGL 2. Open it in a current version of Chrome, Edge, Firefox or Safari."; throw e; }
  })();
  // Colour: every shader here does its lighting on sRGB-encoded numbers and writes the result unconverted, as under r128.
  // The scene is drawn into camRT, a render target, and three never converts colours written to a render target.
  // The lens pass then draws to the screen with a ShaderMaterial that has no <colorspace_fragment>, so outputColorSpace
  // is not applied either. It is set to sRGB (the default) so that adding <colorspace_fragment> to the lens pass is the
  // only change needed when the scene moves to linear lighting.
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const CANVAS_RATIO = Math.min(window.devicePixelRatio || 1, 1.5);
  renderer.setPixelRatio(CANVAS_RATIO);
  // Dynamic resolution: the scene is drawn at one of these pixel ratios (see adaptResolution in the frame loop).
  const dyn = { steps: CANVAS_RATIO > 1 ? [1, 1.25, 1.5].filter((r) => r < CANVAS_RATIO - 1e-3).concat(CANVAS_RATIO) : [CANVAS_RATIO], i: 0, frames: [], n: 0,
    lastChange: 0, lastSlow: 0, lastRaise: 0, holdUp: 8000, fixed: params.has("fixedres") };
  dyn.i = dyn.steps.length - 1;
  const scene = new THREE.Scene();
  // The walker's own eyes: 60 deg tall (91 deg wide at 16:9), a straight-line view.
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1.5e6);
  camera.rotation.order = "YXZ";
  // The canvas keeps its size; dynamic resolution changes only camRT, which the lens pass resamples onto the canvas.
  function resizeTarget() {
    const db = renderer.getDrawingBufferSize(new THREE.Vector2()), k = dyn.steps[dyn.i] / CANVAS_RATIO, rw = Math.max(1, Math.round(db.x * k)), rh = Math.max(1, Math.round(db.y * k));
    camRT.setSize(rw, rh);
    common.pxAng.value = 2 * Math.tan(camera.fov * Math.PI / 360) / rh;
  }
  function resize() { const w = window.innerWidth, h = window.innerHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    lensMat.uniforms.aspect.value = w / h; resizeTarget(); }
  // The scene is drawn into camRT (4x MSAA, sized by dynamic resolution) and copied to the canvas by a full-screen pass.
  const camRT = new THREE.WebGLRenderTarget(1, 1, Object.assign({ samples: 4 }, REVERSE_Z ? { depthTexture: new THREE.DepthTexture(1, 1, THREE.FloatType), resolveDepthBuffer: false } : {}));   // 4x MSAA; replaces r128's WebGLMultisampleRenderTarget
  const lensMat = new THREE.ShaderMaterial({
    uniforms: { src: { value: camRT.texture }, aspect: { value: 1 }, tanH: { value: 0.577 }, sunCam: { value: new THREE.Vector3(0, 0, 1) }, glare: { value: 0 } },
    vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
    // Glare in the eye: light from the sun scattered inside the eye adds a veil over the view around it. Stiles-Holladay:
    // L_veil = 10 E / theta^2 (cd/m2, E = sun illuminance at the eye in lux, theta = angle from the sun in degrees, valid
    // ~1-30 deg). On Mars E ~ 43,000 lux (about 0.43 of Earth's ~100,000 lux at 1.52 AU, before dust); sunlit ground of
    // albedo 0.25 under a 35 deg sun is ~0.25 E sin(35)/pi ~ 1,960 cd/m2, which this page draws at about 0.6. So the veil in
    // screen units is 0.6 * 10 * 43000 / 1960 / theta^2 ~ 130 / theta^2.
    fragmentShader: `uniform sampler2D src; uniform float aspect; uniform float tanH; uniform vec3 sunCam; uniform float glare; varying vec2 vUv;
      void main() {
        vec3 c = texture2D(src, vUv).rgb;
        if (glare > 0.0) {
          vec3 v = normalize(vec3((vUv * 2.0 - 1.0) * vec2(aspect, 1.0) * tanH, -1.0));   // this pixel's view direction, camera space
          float th = max(1.0, degrees(acos(clamp(dot(v, sunCam), -1.0, 1.0))));
          float g = glare * 130.0 / (th * th);
          c += (vec3(1.0, 0.95, 0.88) - c) * (1.0 - exp(-g));   // the veil, compressed toward white so it fades smoothly instead of clipping (display choice)
        }
        gl_FragColor = vec4(c, 1.0);
      }`,
    depthTest: false, depthWrite: false,
  });
  // Whether terrain hides the sun: checked every 0.25 s by stepping along its direction out to 40 km, eased over ~0.3 s.
  let sunSeen = 1, sunSeenT = 1, sunCheckT = 0;
  function sunGlare(dt) {
    const d = common.sunDir.value, u = lensMat.uniforms;
    sunCheckT -= dt;
    if (sunCheckT <= 0) {
      sunCheckT = 0.25; sunSeenT = 1;
      const tanEl = d.y / Math.max(1e-3, Math.hypot(d.x, d.z)), ex = camera.position.x, ez = camera.position.z, ey = camera.position.y + curve(ex, ez);
      for (let s = 2; s < 40000; s *= 1.15) {
        const x = ex + d.x / Math.hypot(d.x, d.z) * s, z = ez + d.z / Math.hypot(d.x, d.z) * s;
        if (heightAt(x, z) - curve(x, z) + curve(ex, ez) > ey + s * tanEl) { sunSeenT = 0; break; }
      }
    }
    sunSeen += (sunSeenT - sunSeen) * (1 - Math.exp(-dt / 0.3));
    u.glare.value = sunSeen;                                              // 0 when the sun is behind terrain
    u.tanH.value = Math.tan(camera.fov * Math.PI / 360);
    u.sunCam.value.copy(d).transformDirection(camera.matrixWorldInverse);   // the sun's direction in camera space
  }
  const lensScene = new THREE.Scene(), lensCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const lensQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), lensMat); lensQuad.frustumCulled = false; lensScene.add(lensQuad);
  window.addEventListener("resize", resize);
  // With ColorManagement on (r152+), new THREE.Color(hex) converts the hex from sRGB to linear. The shaders here expect the
  // sRGB-encoded numbers themselves (see the colour note at the renderer), so the hex goes in unconverted: declaring it
  // LinearSRGBColorSpace tells three to store hex / 255 as it is. Drop the second argument when lighting becomes linear.
  const C = (hex) => new THREE.Color().setHex(hex, THREE.LinearSRGBColorSpace);
  const common = {
    sunDir: { value: new THREE.Vector3(0, 1, 0) }, sinEl: { value: 0.56 }, amb: { value: 0.24 },
    sunCol: { value: C(0xfff0dc) }, skyCol: { value: C(0xdcb994) }, gndCol: { value: C(0x6e4a33) },
    fogCol: { value: C(0xc9a582) }, fogDen: { value: 1.7e-5 }, camPos: { value: new THREE.Vector3() }, pxAng: { value: 1e-3 },
    shMap: { value: null }, shMat: { value: new THREE.Matrix4() }, shOn: { value: 0 }, shTexel: { value: 1 / 2048 }, shZ: { value: new THREE.Vector3() },
    rampA: { value: C(0x34211a) }, rampB: { value: C(0x956042) }, rampC: { value: C(0xe0bd96) },
    gdet: { value: null },
  };
  {                                                          // fine ground texture (see the terrain shader)
    const t = new THREE.TextureLoader().load("tex/ground_detail.jpg");
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.NoColorSpace; t.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    common.gdet.value = t;
  }
  resize();
  // gnd: the object from createGround (ground.js) to add the streamed 25 cm albedo to this material, or nothing
  function terrainMaterial(tex, detail, nmap, tex2, tex2X, gnd) {
    const m = new THREE.ShaderMaterial({
      uniforms: Object.assign({ map: { value: tex }, hasMap: { value: tex ? 1 : 0 }, detail: { value: detail }, nmap: { value: nmap || null }, hasNmap: { value: nmap ? 1 : 0 },
        map2: { value: tex2 || null }, hasMap2: { value: tex2 ? 1 : 0 }, map2X: { value: tex2X || new THREE.Vector4() } }, common, gnd ? gnd.uniforms : {}),
      vertexShader: terrainVS, fragmentShader: terrainFS.replace("// ground.glsl", (gnd ? gnd.glsl : "") + "//"),
    });
    m.defines = {};
    if (tex) m.defines.USE_MAPUV = "";
    if (gnd) m.defines.GROUND = "";
    return m;
  }
  const skyMat = new THREE.ShaderMaterial({
    uniforms: { sunDir: common.sunDir, horizon: { value: C(0xd9b38d) }, zenith: { value: C(0x9c6d4c) }, halo: { value: C(0xbfd4e2) }, below: { value: C(0xb08a68) } },
    vertexShader: skyVS, fragmentShader: skyFS, side: THREE.BackSide, depthWrite: false,
  });
  // The sky dome only needs to be inside the far plane: it is drawn first and writes no depth, so it never hides terrain.
  // At 1.2e6 m it sat so close to the far plane (1.5e6) that float rounding clipped pieces of it for single frames (black holes).
  const sky = new THREE.Mesh(new THREE.SphereGeometry(5e4, 48, 24), skyMat); sky.renderOrder = -1; sky.frustumCulled = false; scene.add(sky);
  // These lights only reach the rocks (MeshStandardMaterial). r128 multiplied light intensity by PI inside the shader
  // (legacy lights); r155+ does not, so the r128 intensities 0.5 and 0.9 are multiplied by PI here.
  const LEGACY_LIGHT = Math.PI;
  scene.add(new THREE.HemisphereLight(C(0xdcb994), C(0x5a3a26), 0.5 * LEGACY_LIGHT));
  const sunLight = new THREE.DirectionalLight(C(0xfff0dc), 0.9 * LEGACY_LIGHT); scene.add(sunLight);

  function normalMap(g, hv, mask, rock) {
    const data = new Uint8Array(g.w * g.h * 4), dy = 2 * g.res;   // RGBA8: r = nx, g = nz, b = strip mask, a = rockiness (RG8 is not supported by three r128)
    for (let j = 0, o = 0; j < g.h; j++) {
      const jm = j > 0 ? j - 1 : 0, jp = j < g.h - 1 ? j + 1 : j;
      for (let i = 0; i < g.w; i++, o++) {
        const im = i > 0 ? i - 1 : 0, ip = i < g.w - 1 ? i + 1 : i;
        const dx = hv(im, j) - hv(ip, j), dz = hv(i, jm) - hv(i, jp), l = Math.sqrt(dx * dx + dy * dy + dz * dz);
        data[4 * o] = Math.round((dx / l * 0.5 + 0.5) * 255); data[4 * o + 1] = Math.round((dz / l * 0.5 + 0.5) * 255);
        data[4 * o + 2] = mask ? Math.round(mask[o] * 255) : 255; data[4 * o + 3] = rock ? Math.min(255, Math.round(rock[o] / 0.12 * 255)) : 77;
      }
    }
    const t = new THREE.DataTexture(data, g.w, g.h, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.unpackAlignment = 1; t.flipY = false; t.generateMipmaps = true; t.colorSpace = THREE.NoColorSpace;   // data, not colour
    t.minFilter = THREE.LinearMipmapLinearFilter; t.magFilter = THREE.LinearFilter; t.anisotropy = 4; t.needsUpdate = true;
    return t;
  }

  // ---------------- rocks ----------------
  // A rounded, irregular rock: an icosphere pushed in and out by a few smooth waves over its surface, with two or three
  // flattened sides (rocks break along planes, then wind and dust round the edges). The offset is a function of direction
  // only, so the copies of a vertex shared by neighbouring faces move together, and normals are averaged over those copies
  // so the surface shades smoothly. Shape is a design choice; sizes and numbers come from the rock fields below.
  function rockGeometry(seed, detail = 1) {
    const geo = new THREE.IcosahedronGeometry(0.5, detail), p = geo.attributes.position; let s = seed;
    const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
    const dir = () => new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
    const waves = [0, 1, 2, 3].map((m) => ({ d: dir(), f: 1.4 + m * 0.9 + rnd(), ph: rnd() * 6.28, a: 0.1 / (1 + m * 0.6) }));
    const cuts = [0, 1, 2].slice(0, 2 + (rnd() < 0.5 ? 1 : 0)).map(() => ({ d: dir(), at: 0.35 + rnd() * 0.3 }));
    const v = new THREE.Vector3(), n = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) {
      n.fromBufferAttribute(p, i).normalize();
      let k = 1; for (const w of waves) k += w.a * Math.sin(w.f * n.dot(w.d) * 3.14 + w.ph);
      for (const c of cuts) { const e = n.dot(c.d) - c.at; if (e > 0) k -= e * 0.8; }   // the flattened sides
      v.copy(n).multiplyScalar(0.5 * k); p.setXYZ(i, v.x, v.y, v.z);
    }
    geo.computeVertexNormals();
    const nr = geo.attributes.normal, sum = new Map(), key = (i) => [p.getX(i), p.getY(i), p.getZ(i)].map((c) => Math.round(c * 1e4)).join();
    for (let i = 0; i < p.count; i++) { const kk = key(i), a = sum.get(kk) || [0, 0, 0]; a[0] += nr.getX(i); a[1] += nr.getY(i); a[2] += nr.getZ(i); sum.set(kk, a); }
    for (let i = 0; i < p.count; i++) { const a = sum.get(key(i)); n.set(a[0], a[1], a[2]).normalize(); nr.setXYZ(i, n.x, n.y, n.z); }
    return geo;
  }
  const rockMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, metalness: 0 });
  // Rock surface: the InSight ground texture (see the terrain shader) projected from the three axes at twice its true scale
  // for grain, and settled dust on the sides that face up, in the ground's mid tone (design choice: Mars rocks in rover
  // images are dark where clean and dust-coloured on top).
  rockMat.onBeforeCompile = (sh) => {
    sh.uniforms.gdet = common.gdet; sh.uniforms.dustCol = { value: common.rampB.value.clone().lerp(common.rampC.value, 0.45) };
    sh.vertexShader = sh.vertexShader.replace("#include <common>", "#include <common>\nvarying vec3 vRW; varying vec3 vRN;")
      .replace("#include <project_vertex>", `#include <project_vertex>
        mat4 rw = modelMatrix;
        #ifdef USE_INSTANCING
        rw = rw * instanceMatrix;
        #endif
        vRW = (rw * vec4(transformed, 1.0)).xyz; vRN = normalize(mat3(rw) * objectNormal);`);
    sh.fragmentShader = sh.fragmentShader.replace("#include <common>", "#include <common>\nuniform sampler2D gdet; uniform vec3 dustCol; varying vec3 vRW; varying vec3 vRN;\n" +
        "float rn3(vec3 p) { vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f); float n = dot(i, vec3(1.0, 57.0, 113.0));" +
        " vec4 a = fract(sin(vec4(n, n + 1.0, n + 57.0, n + 58.0)) * 43758.5453), b = fract(sin(vec4(n + 113.0, n + 114.0, n + 170.0, n + 171.0)) * 43758.5453);" +
        " vec4 m = mix(a, b, f.z); vec2 q = mix(m.xz, m.yw, f.x); return mix(q.x, q.y, f.y); }\n" +
        "vec3 rgrad(vec3 p) { const float e = 0.15; return vec3(rn3(p + vec3(e, 0, 0)) - rn3(p - vec3(e, 0, 0)), rn3(p + vec3(0, e, 0)) - rn3(p - vec3(0, e, 0)), rn3(p + vec3(0, 0, e)) - rn3(p - vec3(0, 0, e))) / (2.0 * e); }")
      .replace("#include <color_fragment>", `#include <color_fragment>
        vec3 bw = pow(abs(vRN), vec3(4.0)); bw /= bw.x + bw.y + bw.z;
        float r = texture2D(gdet, vRW.yz).r * bw.x + texture2D(gdet, vRW.xz).r * bw.y + texture2D(gdet, vRW.xy).r * bw.z + 0.5;
        diffuseColor.rgb *= 1.0 + (r - 1.0) * 1.6;
        diffuseColor.rgb = mix(diffuseColor.rgb, dustCol, smoothstep(0.45, 0.95, vRN.y) * 0.5);`)
      .replace("#include <normal_fragment_maps>", `#include <normal_fragment_maps>
        {                                    // pits and knobs a few cm across: two octaves of noise slope, tangent to the surface
          vec3 g = rgrad(vRW * 7.0) * 0.12 + rgrad(vRW * 19.0) * 0.06, N = normalize(vRN);
          normal = normalize(normal - mat3(viewMatrix) * (g - dot(g, N) * N));
        }`);
  };
  function hash2(a, b) { let h = (Math.imul(a, 374761393) + Math.imul(b, 668265263)) | 0; h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; }
  // A rock field is instanced meshes filled from square tiles around the player; tiles are generated from a hash, so they come back the same.
  // Each tile owns a range of slots in every kind's instance buffer, so adding or removing a tile rewrites only that range.
  function rockField(tile, radius, kinds, cap, detail, gen) {
    const meshes = [], free = [], dirtyM = [], dirtyC = [];
    for (let k = 0; k < kinds; k++) {
      const m = new THREE.InstancedMesh(rockGeometry(11 + k * 17 + detail * 5, detail + 1), rockMat, cap);
      m.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(cap * 3), 3);
      m.instanceColor.setUsage(THREE.DynamicDrawUsage);
      m.count = 0; m.frustumCulled = false; scene.add(m); meshes.push(m);
      free.push([[0, cap]]); dirtyM.push([]); dirtyC.push([]);   // free = [start, length] slot ranges, sorted by start
    }
    return { tile, radius, kinds, cap, gen, meshes, tiles: new Map(), stage: new Map(), free, dirtyM, dirtyC, dirty: false, unplaced: false, version: 0 };
  }
  const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpE = new THREE.Euler(), tmpS = new THREE.Vector3(), tmpP = new THREE.Vector3();
  const tmpC = new THREE.Color();
  let ALB = null, albW = 0, albH = 0;                         // ground brightness at 1/4 resolution, for rock colours
  function groundColour(x, z, out) {
    let g = 0.5;
    if (ALB) {
      const i = Math.max(0, Math.min(albW - 1, Math.round((x - G0.x0) / (G0.res * 4)))), j = Math.max(0, Math.min(albH - 1, Math.round((z - G0.z0) / (G0.res * 4))));
      g = ALB[j * albW + i] / 255;
    }
    g *= 0.8;                                                // rocks read a little darker than the dust around them
    const a = common.rampA.value, b = common.rampB.value, c = common.rampC.value;
    if (g < 0.5) out.copy(a).lerp(b, g * 2); else out.copy(b).lerp(c, g * 2 - 1);
    return out;
  }
  // A tile is generated into shared scratch arrays (grown to the tile's rock count), then copied out at its exact size.
  const tileScratch = { mats: [], cols: [] };
  function beginTile(kinds, n) {
    const out = { mats: [], cols: [], cnt: new Array(kinds).fill(0), solid: [] };
    for (let k = 0; k < kinds; k++) {
      if (!tileScratch.mats[k] || tileScratch.mats[k].length < n * 16) { tileScratch.mats[k] = new Float32Array(n * 16); tileScratch.cols[k] = new Float32Array(n * 3); }
      out.mats.push(tileScratch.mats[k]); out.cols.push(tileScratch.cols[k]);
    }
    return out;
  }
  function finishTile(out) {
    for (let k = 0; k < out.cnt.length; k++) { out.mats[k] = out.mats[k].slice(0, out.cnt[k] * 16); out.cols[k] = out.cols[k].slice(0, out.cnt[k] * 3); }
    return out;
  }
  function placeRock(out, kind, x, z, size, a, b, c, sink, shade) {
    tmpE.set(a * 6.28, b * 6.28, c * 6.28); tmpQ.setFromEuler(tmpE);
    tmpS.set(size, size * (0.5 + b * 0.35), size * (0.7 + a * 0.45));
    tmpP.set(x, heightAt(x, z) - curve(x, z) - size * sink, z);
    tmpM.compose(tmpP, tmpQ, tmpS);
    const k = out.cnt[kind]++;
    tmpM.toArray(out.mats[kind], k * 16);
    groundColour(x, z, tmpC); const cols = out.cols[kind]; cols[k * 3] = tmpC.r * shade; cols[k * 3 + 1] = tmpC.g * shade * (0.96 + 0.08 * a); cols[k * 3 + 2] = tmpC.b * shade;
  }
  // Rocks from 8 cm to 1.4 m, as many as the rock abundance measured from the HiRISE image says.
  const ROCKS = rockField(16, 110, 4, 3000, 1, (ti, tj, T) => {
    const cx = (ti + 0.5) * T, cz = (tj + 0.5) * T;
    if (mask0(cx, cz) < 0.3) return finishTile(beginTile(4, 0));
    const kAb = bilinear(G0, ROCKK, cx, cz);                       // rock area fraction, 0.01..0.2
    const n = Math.round(kAb * T * T * 0.5 * (0.6 + 0.8 * hash2(ti * 3 + 1, tj * 7 - 2))), out = beginTile(4, n);
    for (let r = 0; r < n; r++) {
      const a = hash2(ti * 92821 + r, tj * 68917 - r), b = hash2(ti * 31337 - r * 7, tj * 1013 + r * 11), c = hash2(ti + r * 131, tj - r * 977);
      const x = ti * T + a * T, z = tj * T + b * T;
      const size = Math.min(1.4, 0.08 + 0.3 * Math.pow(-Math.log(Math.max(1e-4, hash2(ti * 7 + r * 3, tj * 5 - r))), 1.4));
      // generated rocks are placed at random; keep big ones out of 15 m around each start spot so one does not block the
      // arrival view (design choice)
      if (size > 0.45 && meta.pois.some((q) => (q.x - x) ** 2 + (q.z - z) ** 2 < 225)) continue;
      placeRock(out, (c * 4) | 0, x, z, size, a, b, c, 0.2, 0.7 + 0.35 * hash2(ti - r * 5, tj + r * 3));
      if (size > 0.45) out.solid.push(x, z, 0.5 * size * Math.max(1, 0.7 + a * 0.45));   // big enough to walk around, not over
    }
    return finishTile(out);
  });
  // Pebbles from 2.5 to 13 cm near the player: below what HiRISE resolves, scaled with the rock abundance around them.
  const PEBBLES = rockField(6, 24, 2, 9000, 0, (ti, tj, T) => {
    const cx = (ti + 0.5) * T, cz = (tj + 0.5) * T;
    if (mask0(cx, cz) < 0.3) return finishTile(beginTile(2, 0));
    const kAb = bilinear(G0, ROCKK, cx, cz);
    const n = Math.round(T * T * (0.8 + kAb * 25) * (0.5 + hash2(ti * 5 - 3, tj * 11 + 1))), out = beginTile(2, n);
    for (let r = 0; r < n; r++) {
      const a = hash2(ti * 7919 + r * 3, tj * 104729 - r), b = hash2(ti * 6007 - r * 13, tj * 2003 + r * 5), c = hash2(ti * 3 + r * 211, tj - r * 457);
      const size = 0.025 + 0.03 * -Math.log(Math.max(0.03, hash2(ti * 13 - r, tj * 17 + r * 7)));
      placeRock(out, (c * 2) | 0, ti * T + a * T, tj * T + b * T, size, a, b, c, 0.3, 0.6 + 0.5 * hash2(ti + r * 9, tj - r * 3));
    }
    return finishTile(out);
  });
  const rockMeshes = ROCKS.meshes;
  // Push a point at (x, z) out of any big rock; small rocks are stepped over.
  const BODY = 0.35;
  function pushOutOfRocks(x, z) {
    const T = ROCKS.tile, ti0 = Math.floor(x / T), tj0 = Math.floor(z / T);
    for (let ti = ti0 - 1; ti <= ti0 + 1; ti++) for (let tj = tj0 - 1; tj <= tj0 + 1; tj++) {
      const t = ROCKS.tiles.get(ti + "," + tj); if (!t) continue;
      for (let k = 0; k < t.solid.length; k += 3) {
        const dx = x - t.solid[k], dz = z - t.solid[k + 1], min = t.solid[k + 2] + BODY, d = Math.hypot(dx, dz);
        if (d < min && d > 1e-6) { x = t.solid[k] + dx / d * min; z = t.solid[k + 1] + dz / d * min; }
      }
    }
    return [x, z];
  }
  // Slot ranges of one kind's instance buffer: first-fit allocation from a sorted free list, merged again on release.
  function allocSlots(f, k, n) {
    const list = f.free[k];
    for (let i = 0; i < list.length; i++) {
      if (list[i][1] < n) continue;
      const start = list[i][0]; list[i][0] += n; list[i][1] -= n; if (!list[i][1]) list.splice(i, 1);
      return start;
    }
    return -1;
  }
  function releaseSlots(f, k, start, n) {
    const list = f.free[k]; let i = 0; while (i < list.length && list[i][0] < start) i++;
    list.splice(i, 0, [start, n]);
    if (i + 1 < list.length && list[i][0] + list[i][1] === list[i + 1][0]) { list[i][1] += list[i + 1][1]; list.splice(i + 1, 1); }
    if (i > 0 && list[i - 1][0] + list[i - 1][1] === list[i][0]) { list[i - 1][1] += list[i][1]; list.splice(i, 1); }
  }
  // Copy a tile into its slots (allocating them if it has none yet for a kind) and note the changed float ranges.
  function placeTile(f, t) {
    for (let k = 0; k < f.kinds; k++) {
      if (t.slot[k] !== undefined) continue;
      const n = t.cnt[k], start = n ? allocSlots(f, k, n) : 0;
      if (start < 0) { f.unplaced = true; continue; }          // the buffer is full; retried when slots are freed
      t.slot[k] = start;
      if (!n) continue;
      const m = f.meshes[k];
      m.instanceMatrix.array.set(t.mats[k], start * 16); m.instanceColor.array.set(t.cols[k], start * 3);
      f.dirtyM[k].push([start * 16, n * 16]); f.dirtyC[k].push([start * 3, n * 3]); f.dirty = true;
    }
  }
  function removeTile(f, key, t) {
    for (let k = 0; k < f.kinds; k++) {
      if (t.slot[k] === undefined || !t.cnt[k]) continue;
      f.meshes[k].instanceMatrix.array.fill(0, t.slot[k] * 16, (t.slot[k] + t.cnt[k]) * 16);   // an all-zero matrix draws nothing
      f.dirtyM[k].push([t.slot[k] * 16, t.cnt[k] * 16]); releaseSlots(f, k, t.slot[k], t.cnt[k]);
    }
    f.tiles.delete(key); f.dirty = true; f.version++;
  }
  function addTile(f, key, t) { t.slot = []; f.tiles.set(key, t); placeTile(f, t); f.dirty = true; f.version++; }
  function uploadRanges(attr, list) {
    if (!list.length) return;
    for (const [start, len] of list) attr.addUpdateRange(start, len);   // three merges touching ranges before uploading
    list.length = 0; attr.needsUpdate = true;
  }
  // Brings the tiles around (px, pz) up to date, generating until the deadline. With `ahead` the tiles go to a side store
  // (f.stage) and nothing else changes; they are picked up for free when the player arrives. Returns true if no tile is missing.
  function updateField(f, px, pz, deadline, ahead) {
    const T = f.tile, R = f.radius; let complete = true;
    const t0i = Math.floor((px - R) / T), t1i = Math.floor((px + R) / T), t0j = Math.floor((pz - R) / T), t1j = Math.floor((pz + R) / T);
    if (!ahead) for (const [key, t] of f.tiles) if (Math.hypot(t.cx - px, t.cz - pz) > R + 2 * T) removeTile(f, key, t);
    outer: for (let ti = t0i; ti <= t1i; ti++) for (let tj = t0j; tj <= t1j; tj++) {
      const cx = (ti + 0.5) * T, cz = (tj + 0.5) * T;
      if (Math.hypot(cx - px, cz - pz) > R) continue;
      const key = ti + "," + tj; if (f.tiles.has(key)) continue;
      if (f.stage.has(key)) { if (!ahead) { addTile(f, key, f.stage.get(key)); f.stage.delete(key); } continue; }
      if (performance.now() > deadline) { complete = false; break outer; }
      const t = f.gen(ti, tj, T); t.cx = cx; t.cz = cz;
      if (ahead) f.stage.set(key, t); else addTile(f, key, t);
    }
    if (ahead) return complete;
    if (f.unplaced) { f.unplaced = false; for (const t of f.tiles.values()) placeTile(f, t); }
    if (!f.dirty) return complete;
    f.dirty = false;
    f.meshes.forEach((m, k) => {
      let hw = 0; for (const t of f.tiles.values()) if (t.slot[k] !== undefined && t.cnt[k]) hw = Math.max(hw, t.slot[k] + t.cnt[k]);
      m.count = hw; uploadRanges(m.instanceMatrix, f.dirtyM[k]); uploadRanges(m.instanceColor, f.dirtyC[k]);
    });
    return complete;
  }
  function updateRocks(px, pz, deadline, ahead) {
    const a = updateField(ROCKS, px, pz, deadline, ahead), b = updateField(PEBBLES, px, pz, deadline, ahead); return a && b;
  }

  // ---------------- rock shadows ----------------
  // Rocks and pebbles are drawn from the sun into a depth map that follows the player; the terrain shader reads it with a
  // hardware depth compare. The map is redrawn only when its position or the set of rocks changes.
  const SH_SIZE = 2048, SH_SPAN = 70;
  const shadowDepth = new THREE.DepthTexture(SH_SIZE, SH_SIZE, THREE.UnsignedIntType, undefined, undefined, undefined, THREE.LinearFilter, THREE.LinearFilter);
  shadowDepth.compareFunction = REVERSE_Z ? THREE.GreaterEqualCompare : THREE.LessEqualCompare;
  const shadowRT = new THREE.WebGLRenderTarget(SH_SIZE, SH_SIZE, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false, depthTexture: shadowDepth });
  const sunCam = new THREE.OrthographicCamera(-SH_SPAN / 2, SH_SPAN / 2, SH_SPAN / 2, -SH_SPAN / 2, 1, 800);
  sunCam.layers.set(1);
  const depthMat = new THREE.ShaderMaterial({
    vertexShader: `void main() { vec4 p = vec4(position, 1.0);
      #ifdef USE_INSTANCING
      p = instanceMatrix * p;
      #endif
      gl_Position = projectionMatrix * modelViewMatrix * p; }`,
    fragmentShader: `void main() {}`,
    colorWrite: false,
  });
  [...ROCKS.meshes, ...PEBBLES.meshes].forEach((m) => m.layers.enable(1));
  common.shMap.value = shadowDepth; common.shTexel.value = 1 / SH_SIZE;
  // depth in 0..1 from the clip-space z of the sun camera: reversed depth maps clip z straight to 0..1 (1 = near), standard maps -1..1
  common.shZ.value.set(REVERSE_Z ? 1 : 0.5, REVERSE_Z ? 0 : 0.5, REVERSE_Z ? 0.00006 : -0.00006);
  const shRight = new THREE.Vector3(), shUp = new THREE.Vector3(), shC = new THREE.Vector3();
  const shSnap = { a: NaN, b: NaN, c: NaN, version: -1 };
  const SH_STEP = (SH_SPAN / SH_SIZE) * 64;                   // the map moves in whole texels, 64 at a time, so shadow edges don't crawl and it is redrawn rarely
  const SH_DEPTH_STEP = 32;                                  // metres along the sun direction; the sun camera sits 400 m back, its range is 1..800 m
  function renderShadows(px, py, pz) {
    const d = common.sunDir.value;
    shRight.set(0, 1, 0).cross(d).normalize(); shUp.copy(d).cross(shRight).normalize();
    tmpP.set(px, py, pz);
    const a = Math.round(shRight.dot(tmpP) / SH_STEP) * SH_STEP, b = Math.round(shUp.dot(tmpP) / SH_STEP) * SH_STEP, c = Math.round(d.dot(tmpP) / SH_DEPTH_STEP) * SH_DEPTH_STEP;
    const version = ROCKS.version + PEBBLES.version;
    if (a === shSnap.a && b === shSnap.b && c === shSnap.c && version === shSnap.version) return;
    shSnap.a = a; shSnap.b = b; shSnap.c = c; shSnap.version = version; if (DEBUG) stats.shadowDraws++;
    shC.copy(shRight).multiplyScalar(a).addScaledVector(shUp, b).addScaledVector(d, c);
    sunCam.position.copy(shC).addScaledVector(d, 400); sunCam.up.copy(shUp); sunCam.lookAt(shC); sunCam.updateMatrixWorld();
    const prevTarget = renderer.getRenderTarget(), prevAutoColour = renderer.autoClearColor;
    scene.overrideMaterial = depthMat; renderer.setRenderTarget(shadowRT); renderer.autoClearColor = false;
    renderer.render(scene, sunCam);
    scene.overrideMaterial = null; renderer.setRenderTarget(prevTarget); renderer.autoClearColor = prevAutoColour;
    common.shMat.value.multiplyMatrices(sunCam.projectionMatrix, sunCam.matrixWorldInverse);   // after the render: three flips the projection when it first draws with reversed depth
    common.shOn.value = 1;
  }

  // ---------------- state ----------------
  let meta = null, ready = false, layers = [];
  const P = { x: 0, z: 0, y: 0, vx: 0, vz: 0, vy: 0, yaw: 0, pitch: 0, air: false, walked: 0, climbed: 0, phase: 0, lastH: null, camY: null, land: 0 };
  const keys = {}; let started = false, paused = true, soundOn = true, hudHidden = true;   // the walk starts with an empty screen; M shows the map
  const touchMove = { x: 0, y: 0 }; let touchRun = false;

  // ---------------- HUD ----------------
  const mapBase = $("mapbase"), mapDot = $("mapdot"); let mapScale = 1, mapW = 0, mapH = 0;
  function buildMap(bitmap) {
    const maxH = isTouch ? 150 : 220; mapH = maxH; mapW = Math.round(maxH * G0.w / G0.h); mapScale = mapH / (G0.h * G0.res);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    for (const c of [mapBase, mapDot]) { c.width = mapW * dpr; c.height = mapH * dpr; c.style.width = mapW + "px"; c.style.height = mapH + "px"; c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0); }
    const ctx = mapBase.getContext("2d");
    ctx.fillStyle = "#2a1f18"; ctx.fillRect(0, 0, mapW, mapH);
    ctx.globalAlpha = 1; ctx.drawImage(bitmap, 0, 0, mapW, mapH);
    ctx.globalCompositeOperation = "multiply"; ctx.fillStyle = "#d39a6a"; ctx.fillRect(0, 0, mapW, mapH); ctx.globalCompositeOperation = "source-over";
    for (const d of meta.pois) {                             // the numbered start-point markers (buttons in #drops) sit on the map
      const [x, y] = mapXY(d.x, d.z), el = $("drop-" + d.id); if (el) { el.style.left = x + "px"; el.style.top = y + "px"; }
    }
    const kmPx = 1000 * mapScale; ctx.fillStyle = "#f0e7dc"; ctx.fillRect(6, mapH - 9, kmPx, 2);
    ctx.font = "500 10px 'IBM Plex Mono', monospace"; ctx.fillText("1 km", 6, mapH - 13);
  }
  function mapXY(x, z) { return [(x - (G0.x0 - G0.res / 2)) * mapScale, (z - (G0.z0 - G0.res / 2)) * mapScale]; }
  function drawDot() {
    const ctx = mapDot.getContext("2d"); ctx.clearRect(0, 0, mapW, mapH);
    const [x, y] = mapXY(P.x, P.z);
    ctx.save(); ctx.translate(x, y); ctx.rotate(-P.yaw);
    ctx.fillStyle = "#a3cbe8"; ctx.strokeStyle = "#10161b"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, -9); ctx.lineTo(6, 7); ctx.lineTo(0, 3); ctx.lineTo(-6, 7); ctx.closePath(); ctx.stroke(); ctx.fill(); ctx.restore();
  }
  const clockT0 = performance.now();
  const marsHours = () => meta.site.sun.localTime + (performance.now() - clockT0) / 1000 / 1.0274912 / 3600;   // Mars seconds are 1.0275 Earth seconds
  // The sun moves with local true solar time t (hours). Declination from Ls (Allison & McEwen 2000, PSS 48:215):
  // dec = asin(0.42565 sin Ls) + 0.25 deg sin Ls; then the usual hour-angle formulas at the site's latitude. At the image's
  // time this gives the label's sun to within 0.15 deg (Candor: el 34.84, az 287.48 vs 287.33); the small difference is
  // carried as an offset so t = image time reproduces the label exactly. In a 20 minute visit the sun moves about 5 deg,
  // little enough that the light already in the 25 cm photo still agrees.
  function sunAngles(t) {
    const Ls = (meta.site.sun.Ls || 0) * Math.PI / 180, p = meta.site.lat0 * Math.PI / 180;
    const d = Math.asin(0.42565 * Math.sin(Ls)) + 0.25 * Math.PI / 180 * Math.sin(Ls), H = (t - 12) * 15 * Math.PI / 180;
    const el = Math.asin(Math.sin(p) * Math.sin(d) + Math.cos(p) * Math.cos(d) * Math.cos(H));
    const az = Math.atan2(-Math.sin(H) * Math.cos(d), Math.cos(p) * Math.sin(d) - Math.sin(p) * Math.cos(d) * Math.cos(H));
    return [az, el];
  }
  let sunOff = null;
  function setSun(t) {
    const s = meta.site.sun, [az0, el0] = sunAngles(s.localTime);
    if (!sunOff) sunOff = [s.az * Math.PI / 180 - az0, s.el * Math.PI / 180 - el0];
    const [a1, e1] = sunAngles(t), az = a1 + sunOff[0], el = e1 + sunOff[1];
    common.sunDir.value.set(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)).normalize();
    // direct light is divided by the image's sin(el) (the de-lit photo is the ground under that sun), so as the sun
    // lowers, flat ground darkens as sin(el) does
    common.sinEl.value = Math.sin(s.el * Math.PI / 180); sunLight.position.copy(common.sunDir.value);
  }
  function updateHud() {
    drawDot();
  }
  let toastTimer = 0;
  // Tips, each shown once at the moment it is useful and only after the opening help line has gone: the map once you have
  // walked a while, sharing when you stop and stand at a view. Using the key first counts as knowing it.
  const tips = { map: false, share: false, still: 0, t0: 0 };
  function updateTips(dt, now) {
    if (!tips.t0) tips.t0 = now;
    if (now - tips.t0 < 13000 || drop) return;
    tips.still = Math.hypot(P.vx, P.vz) < 0.05 ? tips.still + dt : 0;
    if (!tips.map && P.walked > 150) { tips.map = true; toast("M shows the map", 4000); }
    else if (!tips.share && P.walked > 40 && tips.still > 4) { tips.share = true; toast("C copies a link to this view", 5000); }
  }
  // Share: a link that opens the walk at this exact spot, facing the same way.
  function shareView() {
    const u = new URL(location.href); u.search = ""; if (SITE) u.searchParams.set("site", SITE);
    u.searchParams.set("at", [P.x, P.z, P.yaw, P.pitch].map((v, k) => v.toFixed(k < 2 ? 1 : 3)).join(","));
    const shown = () => toast("Link to this spot copied", 3000), fallback = () => toast(u.href, 10000);
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(u.href).then(shown, fallback); else fallback();
  }
  function toast(msg, ms) { const t = $("toast"); t.textContent = msg; t.classList.remove("off"); clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add("off"), ms || 3500); }
  function showHud(on) {
    const vis = on && !hudHidden;
    $("mapbox").hidden = !vis; $("hint").hidden = !on;
    $("reticle").hidden = true;  $("touch").hidden = !(on && isTouch); $("debug").hidden = !(on && DEBUG);
  }

  // ---------------- audio ----------------
  // Sound: sound/audio.js. Mars wind recorded by Perseverance's SuperCam microphone, as heard standing there with no suit.
  let audio = null;
  function startAudio() { if (!audio) { audio = createAudio(); audio.start(); } audio.setPaused(!soundOn); }
  $("sound").addEventListener("click", () => {
    soundOn = !soundOn; $("sound").setAttribute("aria-pressed", String(soundOn));
    if (audio) audio.setPaused(!soundOn || paused);
  });

  // ---------------- input ----------------
  function jump() { if (!P.air) jumpReq = true; }
  window.addEventListener("keydown", (e) => {
    keys[e.code] = true;
    if ((!started || paused) && ready && /^(Key[WASD]|Arrow(Up|Down|Left|Right))$/.test(e.code)) begin();   // a walk key starts (or resumes) the walk
    else if (started && !paused && e.code !== "Escape" && !document.pointerLockElement) engage();   // first key after a mouse start
    if (!started || paused) return;
    if (e.code === "Space") { e.preventDefault(); jump(); }
    if (e.code === "KeyM") { tips.map = true; hudHidden = !hudHidden; showHud(true); }
    if (e.code === "KeyC" && started && !paused) { tips.share = true; shareView(); }
    const k = /^Digit([1-9])$/.exec(e.code);
    if (k && meta.pois[k[1] - 1]) dropTo(meta.pois[k[1] - 1]);
    if (e.code === "Escape" && !document.pointerLockElement) pause();
  });
  window.addEventListener("keyup", (e) => { keys[e.code] = false; });
  window.addEventListener("blur", () => { for (const k in keys) keys[k] = false; });
  // Mouse and drag move a target; the head turns toward it with a 70 ms time constant, as a head has weight (design choice).
  let lookT = null;
  function look(dx, dy, k) {
    dx = Math.max(-120, Math.min(120, dx)); dy = Math.max(-120, Math.min(120, dy));
    if (!lookT) lookT = { yaw: P.yaw, pitch: P.pitch };
    lookT.yaw -= dx * k; lookT.pitch = Math.max(-1.45, Math.min(1.45, lookT.pitch - dy * k));
  }
  function lockPointer() {
    if (isTouch) return;
    try {
      const r = canvas.requestPointerLock({ unadjustedMovement: true });
      if (r && r.catch) r.catch(() => { try { const r2 = canvas.requestPointerLock(); if (r2 && r2.catch) r2.catch(() => {}); } catch (e) { /* drag fallback */ } });
    } catch (e) { try { const r2 = canvas.requestPointerLock(); if (r2 && r2.catch) r2.catch(() => {}); } catch (e2) { /* drag fallback */ } }
  }
  let lookDrag = null, lockedOnce = false;
  document.addEventListener("mousemove", (e) => {
    if (document.pointerLockElement === canvas) look(e.movementX, e.movementY, 0.0021);
    // Moving the mouse starts the walk. Until the first click or key the mouse cannot be captured, so it turns the view
    // directly from its movement.
    else if (ready && !isTouch && (!started || !paused) && document.pointerLockElement !== canvas) {
      if (!started) begin(false);
      look(e.movementX, e.movementY, 0.0021);
    }
  });
  canvas.addEventListener("pointerdown", (e) => {
    if (!started && ready) { begin(); return; }             // the first click on the scene starts the walk (mouse capture and sound)
    if (!started || paused) return;
    if (!document.pointerLockElement) engage();
    lookDrag = { id: e.pointerId, x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!lookDrag || lookDrag.id !== e.pointerId || document.pointerLockElement === canvas) return;
    look(e.clientX - lookDrag.x, e.clientY - lookDrag.y, isTouch ? 0.005 : 0.004); lookDrag.x = e.clientX; lookDrag.y = e.clientY;
  });
  canvas.addEventListener("pointerup", () => { lookDrag = null; });
  canvas.addEventListener("pointercancel", () => { lookDrag = null; });
  document.addEventListener("pointerlockchange", () => {
    const locked = document.pointerLockElement === canvas; canvas.classList.toggle("locked", locked);
    if (locked) lockedOnce = true; else if (started && !paused && !isTouch && lockedOnce) pause();
  });
  const stick = $("stick"), knob = $("knob"); let stickId = null;
  function moveStick(e) {
    const r = stick.getBoundingClientRect(); let dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
    const m = Math.hypot(dx, dy), max = r.width / 2 - 10; if (m > max) { dx *= max / m; dy *= max / m; }
    touchMove.x = dx / max; touchMove.y = dy / max; knob.style.transform = "translate(" + dx + "px," + dy + "px)";
  }
  stick.addEventListener("pointerdown", (e) => { stickId = e.pointerId; stick.setPointerCapture(e.pointerId); moveStick(e); });
  stick.addEventListener("pointermove", (e) => { if (e.pointerId === stickId) moveStick(e); });
  const endStick = () => { stickId = null; touchMove.x = touchMove.y = 0; knob.style.transform = ""; };
  stick.addEventListener("pointerup", endStick); stick.addEventListener("pointercancel", endStick);
  $("t-run").addEventListener("click", () => { touchRun = !touchRun; $("t-run").setAttribute("aria-pressed", String(touchRun)); });
  $("t-jump").addEventListener("click", jump);
  $("t-pause").addEventListener("click", () => pause());

  // ---------------- places ----------------
  // Moving the player is cheap: terrain and rocks around the new spot are built in the frames that follow.
  function placeAt(d) {
    lookT = null; P.x = d.x; P.z = d.z; P.yaw = d.yaw; P.pitch = d.pitch || 0; P.vx = P.vz = P.vy = 0; P.air = false; P.lastH = null; P.camY = null;
    P.y = heightAt(P.x, P.z);
    foot.h = foot.hs = null; P.w = null; P.lean = 0; P.land = 0; P.jolt = 0;   // the body starts fresh on the new ground, not at the old spot's height
  }
  // A drop fades the screen out for 300 ms. During the fade, terrain chunks and rock tiles for the new spot are prepared in
  // the background; then the player moves and the screen stays dark until the new surroundings are complete (at most 2 s).
  let drop = null;
  function dropTo(d) {
    if (drop) return;
    if (reduceMotion) { placeAt(d); return; }
    $("fade").classList.add("on");
    drop = { d, t0: performance.now(), placed: false, tPlaced: 0 };
    layers.forEach((l) => { l.pinned = true; });
  }
  function stepDrop(now, complete) {
    if (!drop.placed) {
      if (now - drop.t0 >= 300) { placeAt(drop.d); drop.placed = true; drop.tPlaced = now; return; }
      updateRocks(drop.d.x, drop.d.z, now + 3, true);
      return;
    }
    if (complete || now - drop.tPlaced > 2000) {
      $("fade").classList.remove("on");
      layers.forEach((l) => { l.pinned = false; }); ROCKS.stage.clear(); PEBBLES.stage.clear(); drop = null;
    }
  }

  // ---------------- start / pause ----------------
  // gesture: false when the start comes from moving the mouse, which browsers do not count as a user action, so the mouse
  // capture and the sound wait for the first click or key (see engage).
  function begin(gesture = true) {
    if (!ready) return;
    const first = !started;
    started = true; paused = false; $("brief").hidden = true; showHud(true); canvas.classList.remove("idle");
    if (gesture) engage();
    if (first) { $("hint").classList.remove("off"); setTimeout(() => $("hint").classList.add("off"), 12000); }
  }
  function engage() { if (!isTouch && !document.pointerLockElement) lockPointer(); startAudio(); }
  function pause() {
    paused = true; $("brief").hidden = false; showHud(false); canvas.classList.add("idle"); $("start").textContent = "resume walking";
    if (document.pointerLockElement) document.exitPointerLock();
    if (audio) audio.setPaused(true);
  }
  $("start").addEventListener("click", begin);

  // ---------------- simulation ----------------
  // The walker is physics/player.js: an unsuited walker at 3/8 g (NASA/TP-2009-214796), walking pace 0.94 m/s (Cavagna et al.
  // 1998), Tobler's grade factor, Coulomb traction (loose ground slides past its 31 deg angle of repose). Here: the strip
  // edge, pushing out of big rocks, and the footfall events for the body and sound.
  const EYE = 1.65;                                           // m, the eyes above the feet (a 1.80 m person)
  // Ground type for traction: "rock" where the rock map is above 7% cover (steep faces and dark rocky ground), else loose
  // ground. Design choice: the map is derived from slope and darkness, not from counted rocks.
  const terrainIO = { heightAt, slopeAt, groundAt: (x, z) => inGrid(G0, x, z) && bilinear(G0, ROCKK, x, z) > 0.07 ? "rock" : "sand" };
  let edgeT = 0, jumpReq = false;
  function stepPlayer(dt, now) {
    let mx = 0, mz = 0;
    if (keys.KeyW || keys.ArrowUp) mz -= 1; if (keys.KeyS || keys.ArrowDown) mz += 1;
    if (keys.KeyA || keys.ArrowLeft) mx -= 1; if (keys.KeyD || keys.ArrowRight) mx += 1;
    mx += touchMove.x; mz += touchMove.y;
    const m = Math.hypot(mx, mz); if (m > 1) { mx /= m; mz /= m; }
    const sy = Math.sin(P.yaw), cy = Math.cos(P.yaw);
    const input = { moveX: mx * cy + mz * sy, moveZ: -mx * sy + mz * cy, run: !!(keys.ShiftLeft || keys.ShiftRight || touchRun), jump: jumpReq };
    jumpReq = false;
    // a teleport (start spot, drop) moved P outside the walker: start a new walker there
    if (!P.w || P.w.x !== P.x || P.w.z !== P.z) P.w = { ...newWalker(P.x, P.z, terrainIO), phase: P.phase };
    const s = stepWalker(P.w, input, terrainIO, dt);
    let nx = s.x, nz = s.z;
    if (mask0(nx, nz) < 0.4) {                                // edge of the HiRISE strip: stay on measured ground
      if (mask0(nx, P.z) >= 0.4) nz = P.z; else if (mask0(P.x, nz) >= 0.4) nx = P.x; else { nx = P.x; nz = P.z; }
      if (now - edgeT > 6000) { edgeT = now; toast("This is the edge of the measured ground. Press M for the map."); }
    }
    [nx, nz] = pushOutOfRocks(nx, nz);
    if (nx !== s.x || nz !== s.z) {                           // blocked: lose the speed into the obstacle
      if (dt > 0) { s.vx = (nx - P.x) / dt; s.vz = (nz - P.z) / dt; }
      s.x = nx; s.z = nz; if (!s.air) s.y = heightAt(nx, nz);
    }
    const dist = Math.hypot(s.x - P.x, s.z - P.z);
    P.w = s; P.x = s.x; P.z = s.z; P.y = s.y; P.vx = s.vx; P.vz = s.vz; P.vy = s.vy; P.air = s.air; P.phase = s.phase;
    if (s.events.step) plantFoot();   // no footstep sound: synthesised steps did not sound real (no recordings of footsteps on Mars exist)
    if (s.events.land) { P.land = Math.min(0.12, s.events.land * 0.04); P.jolt = 1.2; plantFoot(); }
    P.walked += dist;
    const h = heightAt(P.x, P.z);
    if (P.lastH !== null && h > P.lastH) P.climbed += h - P.lastH;
    P.lastH = h;
  }

  // ---------------- body ----------------
  // The eyes ride on the body, and the body vaults over the stance leg (inverted pendulum): between footfalls the
  // hip is at foot + sqrt(LEG^2 - d^2), d = how far the body is past the foot along the walk. Each footfall lands where the
  // foot actually is, on the 1 m terrain plus the fine relief, so the eye steps up and down with the ground under each
  // foot. Weight shifts to the new foot over about 0.1 s (double support). A rigid leg overstates the rise and fall since
  // the knee flexes in stance, so the arc is scaled by ARC (design choice).
  const ARC = 0.6, UPPER = EYE - LEG;   // UPPER: hip to eyes
  const foot = { x: 0, z: 0, h: null, hs: null };
  const groundH = (x, z) => heightAt(x, z) + microHeight(x, z);
  function plantFoot() {
    const v = Math.hypot(P.vx, P.vz), ux = v > 1e-3 ? P.vx / v : 0, uz = v > 1e-3 ? P.vz / v : 0, half = 0.5 * stepLength(v);
    foot.x = P.x + ux * half; foot.z = P.z + uz * half; foot.h = groundH(foot.x, foot.z);
    if (foot.hs === null) foot.hs = foot.h;
  }
  // ---------------- dynamic resolution ----------------
  // Frames are timed against the 60 Hz frame time. If the last second averages over 18 ms the scene is drawn at the next lower
  // pixel ratio. It goes back up only after a quiet spell (holdUp) with a frame time at 60 Hz; if that brings the slowdown back
  // within 15 s, the wait before the next try doubles (up to 60 s), so the ratio does not flicker.
  function setResolutionStep(i, now) { dyn.i = i; dyn.lastChange = now; dyn.n = 0; resizeTarget(); }
  function adaptResolution(now, ms) {
    if (dyn.fixed || dyn.steps.length < 2 || ms > 100) return;   // a frame over 100 ms is a tab switch or a stall, not load
    dyn.frames[dyn.n++ % 60] = ms;
    if (dyn.n < 60 || now - dyn.lastChange < 1500) return;
    let sum = 0; for (let k = 0; k < 60; k++) sum += dyn.frames[k];
    const mean = sum / 60;
    if (mean > 18) {
      dyn.lastSlow = now;
      if (dyn.i > 0) { if (now - dyn.lastRaise < 15000) dyn.holdUp = Math.min(60000, dyn.holdUp * 2); setResolutionStep(dyn.i - 1, now); }
    } else if (mean > 17.4) dyn.lastSlow = now;
    else if (dyn.i < dyn.steps.length - 1 && now - Math.max(dyn.lastChange, dyn.lastSlow) > dyn.holdUp) { dyn.lastRaise = now; setResolutionStep(dyn.i + 1, now); }
  }

  // ---------------- frame loop ----------------
  const stats = { frames: [], asm: [], shadowDraws: 0 };
  let lastT = performance.now(), hudT = 0, dbgT = 0, frameCount = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    const rawDt = (now - lastT) / 1000, dt = Math.min(0.05, rawDt); lastT = now;
    if (!ready) return;
    if (started && !paused && !drop) adaptResolution(now, rawDt * 1000);
    const t0 = performance.now();
    if (lookT) {
      const a = reduceMotion ? 1 : 1 - Math.exp(-dt / 0.07);
      P.yaw += (lookT.yaw - P.yaw) * a; P.pitch += (lookT.pitch - P.pitch) * a;
    }
    if (started && !paused) stepPlayer(dt, now);
    const speed = Math.hypot(P.vx, P.vz), gait = reduceMotion || P.air ? 0 : Math.min(1, speed / WALK);
    let target;
    if (P.air || foot.h === null) { target = P.y + EYE; if (!P.air) plantFoot(); }
    else {
      if (speed < 0.05) { foot.x += (P.x - foot.x) * Math.min(1, dt * 4); foot.z += (P.z - foot.z) * Math.min(1, dt * 4); foot.h = groundH(foot.x, foot.z); }  // standing: weight over both feet
      foot.hs += (foot.h - foot.hs) * (1 - Math.exp(-dt * 20));
      const ux = speed > 1e-3 ? P.vx / speed : 0, uz = speed > 1e-3 ? P.vz / speed : 0;
      const d = Math.max(-0.9 * LEG, Math.min(0.9 * LEG, (P.x - foot.x) * ux + (P.z - foot.z) * uz));
      target = foot.hs + LEG - ARC * (LEG - Math.sqrt(LEG * LEG - d * d)) + UPPER;
    }
    P.camY = P.camY === null ? target : P.camY + (target - P.camY) * (1 - Math.exp(-dt * (P.air ? 40 : 30)));
    P.land *= Math.exp(-dt * 6);
    // Lean: walkers lean into a climb and back on a descent, half the slope angle uphill and a third downhill along the
    // direction of travel, settling over about 0.5 s (design choice). The head follows half of it.
    let grade = 0;
    if (speed > 0.1 && !P.air) { const sl = slopeAt(P.x, P.z); grade = (sl.gx * P.vx + sl.gz * P.vz) / speed; }
    const leanT = Math.atan(grade) * (grade > 0 ? 0.5 : 0.33) * Math.min(1, speed / WALK);
    P.lean = (P.lean || 0) + (leanT - (P.lean || 0)) * (1 - Math.exp(-dt * 2));
    // The head sways a little over the stance foot and gets a short shake at each footfall (design choice).
    P.jolt = (P.jolt || 0) * Math.exp(-dt * 14);
    const sway = gait * 0.02 * Math.sin(P.phase);
    const shake = P.jolt * 0.006 * Math.sin(now * 0.075);
    camera.position.set(P.x + Math.cos(P.yaw) * sway, P.camY - P.land + shake - curve(P.x, P.z), P.z - Math.sin(P.yaw) * sway);   // the meshes drop with the planet's curvature
    camera.rotation.set(P.pitch - 0.5 * P.lean + shake, P.yaw, 0);   // eyes and neck hold the gaze steady against the torso's sway and twist
    common.camPos.value.copy(camera.position); sky.position.copy(camera.position);
    const t1 = performance.now();
    const deadline = t1 + (drop && drop.placed ? 6 : 2.5);      // per-frame budget for rock tiles and main-thread terrain builds (the screen is dark after a drop)
    const queue = [], terrainDone = collectLayers(P.x, P.z, false, queue) && !chunkBuilder.busy();
    if (drop && !drop.placed) collectLayers(drop.d.x, drop.d.z, true, queue);
    if (queue.length) { queue.sort(byPriority); chunkBuilder.pump(queue, deadline); }
    if (ground) ground.update(P.x, P.z);                        // at most 2 tile uploads, about 1 ms
    const t2 = performance.now();
    const rocksDone = updateRocks(P.x, P.z, Math.max(deadline, t2 + 0.8));
    if (drop) stepDrop(t2, terrainDone && rocksDone);
    const t3 = performance.now();
    if (started && !paused) updateTips(dt, now);
    hudT += dt; if (started && !paused && hudT > 0.2) { hudT = 0; updateHud(); setSun(marsHours()); }
    if (audio && started && !paused) audio.update();
    const t4 = performance.now();
    if (!window.__walkNoShadow) renderShadows(P.x, P.y - curve(P.x, P.z), P.z); else common.shOn.value = 0;
    renderer.setRenderTarget(camRT); renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    const sceneCalls = renderer.info.render.calls, sceneTris = renderer.info.render.triangles;
    sunGlare(dt); renderer.render(lensScene, lensCam);
    if (DEBUG && window.__rec && window.__rec.on) {         // debug: keep small copies of frames to find one-frame artefacts
      const r = window.__rec; r.ctx = r.ctx || Object.assign(document.createElement("canvas"), { width: 640, height: 360 }).getContext("2d", { willReadFrequently: true });
      r.ctx.drawImage(canvas, 0, 0, 640, 360); const px = r.ctx.getImageData(0, 0, 640, 360).data; let dark = 0;
      for (let i = 0; i < px.length; i += 4) if (px[i] > 200 && px[i + 1] < 60 && px[i + 2] > 200) dark++;   // magenta = nothing drawn (set the clear colour to magenta when testing)
      r.dark.push(dark); r.pos.push([P.x, P.z, P.yaw, P.pitch, camera.position.x, camera.position.y, camera.position.z]); if (r.keep) r.urls.push(r.ctx.canvas.toDataURL("image/jpeg", 0.8));
    }
    const t5 = performance.now();
    if (DEBUG) {
      stats.frames.push([rawDt * 1000, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]);
      frameCount++; dbgT += rawDt;
      if (dbgT > 0.5) {
        $("debug").textContent = (frameCount / dbgT).toFixed(0) + " fps · " + sceneCalls + " draws · " + (sceneTris / 1e6).toFixed(2) + " M tris · " + dyn.steps[dyn.i] + "x";
        frameCount = 0; dbgT = 0;
      }
    }
  }

  // ---------------- boot ----------------
  let ground = null;                                     // streamed 25 cm albedo (../ground/ground.js), when the site has tiles/
  async function boot() {
    const status = $("status");
    try {
      if (typeof DecompressionStream === "undefined") throw new Error("This browser cannot unpack the terrain. Open the page in a current version of Chrome, Edge, Firefox or Safari.");
      meta = await (await fetch(DATA + "meta.json")).json();
      R = meta.site.R;
      const L0 = meta.L0, L1 = meta.L1, L2 = meta.L2;
      [G0, G1, G2] = [L0, L1, L2].map((L) => ({ w: L.w, h: L.h, res: L.res, x0: L.x0, z0: L.z0 }));
      const [h0, h1, h2, auxRaw, a0, a1] = await Promise.all([
        loadHeights(L0), loadHeights(L1), loadHeights(L2), fetchBytes(DATA + L0.aux).then(inflate), loadBitmap(L0.albedo), loadBitmap(L1.albedo),
      ]);
      H0 = h0; H1 = h1; H2 = h2;
      status.textContent = "preparing terrain…"; setStep("build", "active");
      await new Promise((r) => setTimeout(r, 0));
      const n0 = G0.w * G0.h; MASK0 = new Float32Array(n0); ROCKK = new Float32Array(n0); BLEND0 = new Float32Array(n0);
      for (let i = 0; i < n0; i++) { MASK0[i] = auxRaw[i] / 255; ROCKK[i] = auxRaw[n0 + i] / 255 * 0.25; }
      const EDGE = 80;                                       // fade the strip out at the border of the L0 grid too
      for (let j = 0, o = 0; j < G0.h; j++) for (let i = 0; i < G0.w; i++, o++) {
        const e = Math.min(i, G0.w - 1 - i, j, G0.h - 1 - j) * G0.res; if (e < EDGE) MASK0[o] *= smooth(e / EDGE);
      }
      for (let j = 0, o = 0; j < G0.h; j++) for (let i = 0; i < G0.w; i++, o++) {
        const m = MASK0[o];
        BLEND0[o] = m >= 0.999 ? H0[o] : H0[o] * m + height1(G0.x0 + i * G0.res, G0.z0 + j * G0.res) * (1 - m);
      }
      setSun(meta.site.sun.localTime);                         // the sun as in the image, then it moves with the clock
      // The albedo JPEGs are read as brightness (.r) and mapped through the colour ramp, so they are sampled unconverted.
      const tex = (bm) => { const t = new THREE.Texture(bm); t.flipY = false; t.colorSpace = THREE.NoColorSpace; t.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy()); t.needsUpdate = true; return t; };
      const uvFor = (g) => [g.x0 - g.res / 2, g.z0 - g.res / 2, g.w * g.res, g.h * g.res];   // albedo map origin and size
      {
        albW = Math.ceil(G0.w / 4); albH = Math.ceil(G0.h / 4);
        const cv = document.createElement("canvas"); cv.width = albW; cv.height = albH; const cx = cv.getContext("2d");
        cx.drawImage(a0, 0, 0, albW, albH); const px = cx.getImageData(0, 0, albW, albH).data;
        ALB = new Uint8Array(albW * albH); for (let k = 0; k < ALB.length; k++) ALB[k] = px[4 * k];
      }
      // Vertex heights for L1 and L2, computed once. L0 covers its whole grid rectangle (outside the strip its heights
      // already equal L1's), so L1 has a plain hole there and sinks 2 m under the hole's edge so the two never fight.
      const X1 = G0.x0 + (G0.w - 1) * G0.res, Z1 = G0.z0 + (G0.h - 1) * G0.res;
      const inRect0 = (x, z) => x >= G0.x0 && x <= X1 && z >= G0.z0 && z <= Z1;
      const V1 = new Float32Array(G1.w * G1.h);
      for (let j = 0, o = 0; j < G1.h; j++) for (let i = 0; i < G1.w; i++, o++) {
        const x = G1.x0 + i * G1.res, z = G1.z0 + j * G1.res; V1[o] = height1(x, z) - (inRect0(x, z) ? 2 : 0);
      }
      const half1 = (G1.w - 1) * G1.res / 2, V2 = new Float32Array(G2.w * G2.h);
      for (let j = 0, o = 0; j < G2.h; j++) for (let i = 0; i < G2.w; i++, o++) V2[o] = H2[o] - 60 * weight1(G2.x0 + i * G2.res, G2.z0 + j * G2.res);
      status.textContent = "preparing lighting…"; await new Promise((r) => setTimeout(r, 0));
      const n0map = normalMap(G0, (i, j) => BLEND0[j * G0.w + i], MASK0, ROCKK);
      const n1map = normalMap(G1, (i, j) => H1[j * G1.w + i]);
      const t1 = tex(a1), t1X = new THREE.Vector4(G1.x0 - G1.res / 2, G1.z0 - G1.res / 2, 1 / (G1.w * G1.res), 1 / (G1.h * G1.res));
      // The 25 cm tiles (walk/pipeline/tiles.py) are optional: without DATA/tiles/tiles.json the page runs without them.
      // a0 goes in to fit the tiles' brightness to l0_a.jpg (see ground.js); createGround reads it before buildMap runs.
      try {
        const r = await fetch(DATA + "tiles/tiles.json");
        if (r.ok) {
          const tilesJson = await r.json(), { createGround } = await import("./ground/ground.js");
          ground = createGround({ THREE, renderer, baseUrl: DATA + "tiles/", tilesJson, meta, albedo2m: a0 });
        }
      } catch (e) { console.warn("25 cm ground tiles not used:", e); ground = null; }
      if (DEBUG) window.__ground = ground;
      layers.push(new Layer(scene, G0, {
        name: "L0", chunk: 128, levels: [[1, 200], [2, 550], [4, 1500], [8, 1e9]], skirt: 1.2, material: terrainMaterial(tex(a0), 0.18, n0map, t1, t1X, ground), noNormals: true,
        uvO: uvFor(G0),
      }));
      layers.push(new Layer(scene, G1, {
        name: "L1", chunk: 160, levels: [[1, 3000], [2, 9000], [4, 1e9]], skirt: 0.6, material: terrainMaterial(t1, 0.0, n1map), noNormals: true,
        uvO: uvFor(G1), skipR: [G0.x0, X1, G0.z0, Z1, false],
      }));
      // L2: 400 m, sunk below L1 inside the L1 square
      const in2 = half1 - 2000;
      layers.push(new Layer(scene, G2, {
        name: "L2", chunk: 200, levels: [[1, 40000], [2, 1e9]], skirt: 0.4, material: terrainMaterial(null, 0.0),
        skipR: [-in2, in2, -in2, in2, true],
      }));
      chunkBuilder.start([{ g: G0, hv: BLEND0 }, { g: G1, hv: V1 }, { g: G2, hv: V2 }]);
      // places
      for (const d of meta.pois) {
        const b = document.createElement("button"); b.type = "button"; b.id = "drop-" + d.id;   // a numbered marker on the map
        b.textContent = meta.pois.indexOf(d) + 1; b.title = d.label; b.setAttribute("aria-label", d.label);
        b.addEventListener("click", () => { dropTo(d); if (started && !paused) lockPointer(); });
        $("drops").appendChild(b);
      }
      buildMap(a0);
      const at = params.get("at");
      // ?at=x,z,yaw,pitch: a shared view (see shareView); ignored if malformed or off the measured ground
      const sh = at ? at.split(",").map(Number) : [];
      const start = sh.length >= 2 && sh.every(Number.isFinite) && mask0(sh[0], sh[1]) >= 0.4
        ? { x: sh[0], z: sh[1], yaw: sh[2] || 0, pitch: Math.max(-1.4, Math.min(1.4, sh[3] || 0)) } : meta.pois[0];
      placeAt(start);
      for (;;) {                                             // build everything around the start before showing the page as ready
        const rocks = updateRocks(P.x, P.z, performance.now() + 1e9), terrain = updateLayers(P.x, P.z, performance.now() + 1e9);
        if (ground) ground.update(P.x, P.z);
        if (rocks && terrain && (!ground || ground.pendingCount() === 0)) break;
        await new Promise((r) => setTimeout(r, 10));
      }
      camera.position.set(P.x, P.y + EYE, P.z); camera.rotation.set(P.pitch, P.yaw, 0);
      renderer.compile(scene, camera);
      ready = true; status.textContent = ""; $("start").disabled = false; setStep("build", "done");
    } catch (err) {
      status.textContent = (err && err.message) ? err.message : "The terrain could not be built in this browser.";
      console.error(err);
    }
  }
  if (DEBUG) window.__walk = {
    P, keys, stats, renderer, layers, dyn, REVERSE_Z, ROCKS, scene, rockMeshes, shadowRT, common, PEBBLES, sky, camera, begin: () => begin(), isReady: () => ready,
    view: (x, z, yaw, pitch) => { placeAt({ x, z, yaw, pitch }); }, heightAt: (x, z) => heightAt(x, z), meta: () => meta,
  };
  requestAnimationFrame(frame);
  boot();
})();
