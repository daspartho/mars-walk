// An unsuited walker on Mars (no suit, by choice of the experience): one pure step function, stepWalker(state, input, terrain, dt) -> new state.
// Every constant below is either cited or marked "design choice". Big rocks are the caller's job (push out after this).
//
// Frame: x, z horizontal (m), y up (m, height of the feet = ground height when standing). Speeds are horizontal
// (map) speeds, as in Tobler's hiking function.
// terrain = { heightAt(x, z) -> m, slopeAt(x, z) -> { gx, gz } (rise per metre, as main.js), groundAt(x, z) -> "sand" | "rock" }
// input   = { moveX, moveZ: wished direction in the world frame, length 0..1; run: bool; jump: bool (jumps when on the ground) }
// state   = { x, y, z, vx, vy, vz, air, phase, effort } (newWalker() makes one). The returned state also carries
//   sliding (bool), gait ("walk" | "lope" | "air"), effortNow (0-1, instantaneous) and
//   events = { step: 0 or footstep strength 0-1, land: 0 or vertical impact speed m/s, blocked: bool }.

export const G_MARS = 3.71;                 // m/s^2, NASA Mars fact sheet (3.71; EWT used 3.68)
const G_EARTH = 9.81;

// Body. The walker is the subject group of NASA/TP-2009-214796 (Norcross et al., EVA Walkback Test) without a suit (the
// experience leaves the suit out). That study also measured its subjects unsuited at 3/8 g.
export const BODY_MASS = 81.4;              // kg, EWT subject mean (Table 1)
export const SUIT_MASS = 0;                 // kg, no suit (the EWT's MKIII suit was 120 kg)
const HEIGHT = 1.80;                        // m, EWT subject mean 180.3 cm
export const LEG = 0.530 * HEIGHT;          // m, hip (greater trochanter) height = 0.530 x stature (Winter, Biomechanics and
                                            // Motor Control of Human Movement, segment-length table)
const MASS = BODY_MASS + SUIT_MASS;

// Speeds.
export const WALK = 3.4 / 3.6;              // 0.94 m/s: optimal walking speed at Mars gravity, Cavagna et al. 1998 (Nature
                                            // 393:636, 3.4 km/h vs 5.5 km/h on Earth). Froude number v^2/(g L) = 0.25.
export const LOPE = 2.2;                    // m/s, design choice: Froude number 1.37, the same as a 3.6 m/s jog on Earth
                                            // (Froude scaling, Kram et al. 1997 J Exp Biol 200:821). EWT subjects in the MKIII
                                            // were near peak VO2 by 1.5 m/s, so this is a short-burst speed.
export const GAIT_CHANGE = 1.74;            // m/s, measured unsuited walk-run transition at 3/8 g (EWT, Froude 0.89, well above
                                            // Kram's Froude 0.5). Suited it was 1.54 +/- 0.16 (Table 4).

// Ground. Traction is Coulomb friction: the foot can push along the surface with at most mu x normal force.
// Sand: mu = tan(angle of repose); slopes steeper than that fail under the foot and you slide.
export const REPOSE_SAND = 31;              // deg, Atwood-Stone & McEwen 2013 (GRL 40:2929): dune slip faces 30-31 deg at Gale.
                                            // Rover-measured soil friction angles are 30-37 deg (Sullivan et al. 2011).
export const MAX_ROCK = 42;                 // deg, design choice (brief: hold on rock up to about 40-45 deg), mu = tan(42) = 0.90
export const MU = { sand: Math.tan(REPOSE_SAND * Math.PI / 180), rock: Math.tan(MAX_ROCK * Math.PI / 180) };
export const STEP_MAX = 0.6;                // m, design choice: highest ledge you can step onto unsuited at 3/8 g (no data found)
const PROBE = 0.3;                          // m, design choice: ledge test this far ahead, about one boot length
const LIFT_MARGIN = 0.05;                   // m, design choice: leave the ground only when it falls 5 cm below a free-fall path

// Jump, design choice from a force model: a 0.40 m countermovement jump on Earth (a typical fit adult; design value) with
// 0.40 m of leg push-off gives an average push force F = m_body (g_E + g_E h / d) = 2 body weights. The same force on the
// body plus suit at Mars gravity gives v0^2 = 2 d (F / (m_body + m_suit) - g_M). The suit's limit on knee bend is ignored.
const JUMP_EARTH = 0.40, PUSH = 0.40;
const JUMP_FORCE = BODY_MASS * (G_EARTH + G_EARTH * JUMP_EARTH / PUSH);
export const JUMP_V = Math.sqrt(Math.max(0, 2 * PUSH * (JUMP_FORCE / MASS - G_MARS)));   // unsuited 3.57 m/s, apex 1.72 m, 1.9 s in the air

// Grade: Tobler 1993 (NCGIA Technical Report 93-1) hiking function, speed proportional to exp(-3.5 |s + 0.05|), s = rise per
// metre along the path, normalised to 1 on flat ground. Tobler is fitted on Earth; the uphill part is mostly the power to
// lift the weight, so s is scaled by (weight on Mars) / (body weight on Earth) = 0.94 for this walker (extrapolation).
const GRADE_SCALE = MASS * G_MARS / (BODY_MASS * G_EARTH);
export const tobler = (s) => Math.exp(-3.5 * (Math.abs(s * GRADE_SCALE + 0.05) - 0.05));

// Step length: relative stride = 2.3 Fr^0.3 (Alexander & Jayes 1983, J Zool 201:135; dynamic similarity). One step = half a
// stride. Not measured at 0.38 g. Walk 0.72 m, lope 1.21 m.
export const stepLength = (v) => 0.5 * LEG * 2.3 * Math.pow(Math.max(v, 0.05) ** 2 / (G_MARS * LEG), 0.3);

// Effort (0-1) = (VO2 - rest) / (VO2peak - rest), a metabolic proxy for audio.
// Level walking, suited, 3/8 g: EWT Table 5 gives 27.6, 32.5, 35.7 ml/kg/min at about 1.25, 1.38, 1.52 m/s (PTS - 0.49,
// -0.36, -0.22 m/s, mean unsuited PTS 1.74). Fitted here as rest + 17.3 v^1.48 (within 1.2 ml/kg/min of the three points);
// below 1.07 m/s and above 1.53 m/s this is extrapolation (no suited running data at 3/8 g).
// Terrain multiplies the locomotion part: Soule & Goldman 1972 (J Appl Physiol 32:706) loose sand 2.1; rock 1.2 (their
// "light brush" as a stand-in for uneven ground; design choice). The EWT treadmill is 1.0.
// Grade: Pandolf et al. 1977 (J Appl Physiol 43:577) grade term eta (W + L)(0.35 V G) watts, G in %, scaled by g_M / g_E
// (lifting work is proportional to weight; extrapolation). Downhill (G < 0) adds nothing: Pandolf is for level and uphill.
// 1 ml O2 = 20.1 J. Rest 3.5 ml/kg/min (1 MET). VO2peak 48.7 ml/kg/min (EWT subject mean).
const VO2_REST = 3.5, VO2_PEAK = 48.7, J_PER_ML = 20.1;
export const TERRAIN_COST = { sand: 2.1, rock: 1.2 };
export function vo2(v, grade, ground) {
  const eta = TERRAIN_COST[ground] || 1;
  const level = 17.3 * Math.pow(Math.max(v, 0), 1.48);
  const liftW = eta * MASS * 0.35 * v * Math.max(0, grade * 100) * (G_MARS / G_EARTH);
  return VO2_REST + eta * level + liftW / J_PER_ML * 60 / BODY_MASS;
}
export const effortOf = (v, grade, ground) => Math.min(1, Math.max(0, (vo2(v, grade, ground) - VO2_REST) / (VO2_PEAK - VO2_REST)));
const EFFORT_TAU = 30;                      // s, design choice: typical VO2 on-response time constant of a fit adult

export function newWalker(x, z, terrain) {
  return { x, y: terrain.heightAt(x, z), z, vx: 0, vy: 0, vz: 0, air: false, phase: 0, effort: 0 };
}

// Ledge test: can the walker move from (x, z) in direction (dx, dz)? Blocked when the ground one boot length ahead is more
// than STEP_MAX above the feet.
function blockedAt(terrain, x, z, y, dx, dz) {
  const l = Math.hypot(dx, dz); if (l < 1e-9) return false;
  return terrain.heightAt(x + dx / l * PROBE, z + dz / l * PROBE) - y > STEP_MAX;
}

// Move horizontally by (vx dt, vz dt); if that is blocked by a ledge, keep the free axis (slide along it), else stop.
function moveH(terrain, s, dt, ev) {
  const tries = [[s.vx, s.vz], [s.vx, 0], [0, s.vz]];
  for (const [vx, vz] of tries) {
    if (vx === 0 && vz === 0) break;
    if (!blockedAt(terrain, s.x, s.z, s.y, vx, vz)) {
      if (vx !== s.vx || vz !== s.vz) ev.blocked = true;
      s.vx = vx; s.vz = vz; s.x += vx * dt; s.z += vz * dt; return;
    }
  }
  if (s.vx !== 0 || s.vz !== 0) ev.blocked = true;
  s.vx = 0; s.vz = 0;
}

// Vertical speed of the ground under feet moving at (vx, vz): slope of the ground from one boot length behind to the feet.
// One-sided on purpose: at an edge it gives the slope you were walking on, not a difference across the drop.
function groundVy(terrain, x, z, vx, vz) {
  const v = Math.hypot(vx, vz); if (v < 1e-9) return 0;
  return v * (terrain.heightAt(x, z) - terrain.heightAt(x - vx / v * PROBE, z - vz / v * PROBE)) / PROBE;
}

function substep(s, input, terrain, dt, ev) {
  const g = G_MARS;
  let wx = input.moveX || 0, wz = input.moveZ || 0;
  const wl = Math.hypot(wx, wz); if (wl > 1) { wx /= wl; wz /= wl; }
  const want = Math.min(1, wl);
  let grade = 0;

  if (!s.air) {
    const sl = terrain.slopeAt(s.x, s.z), gx = sl.gx, gz = sl.gz, gm = Math.hypot(gx, gz);
    const ground = terrain.groundAt(s.x, s.z) === "rock" ? "rock" : "sand";
    const th = Math.atan(gm), c2 = Math.cos(th) ** 2;
    // Target velocity: base speed times Tobler's factor for the grade along the wished direction.
    let tx = 0, tz = 0;
    if (want > 0.01) {
      const dx = wx / wl, dz = wz / wl;
      grade = gx * dx + gz * dz;
      const sp = (input.run ? LOPE : WALK) * want * tobler(grade);
      tx = dx * sp; tz = dz * sp;
    }
    // Friction circle, in horizontal projection: gravity pulls downslope with g sin(th) cos(th); the foot can push with
    // at most mu g cos^2(th) (both projected by cos th, so you slip exactly when tan th > mu).
    const pull = gm > 1e-9 ? g * Math.sin(th) * Math.cos(th) / gm : 0;
    const pgx = -gx * pull, pgz = -gz * pull;                 // gravity's horizontal pull, downslope
    const R = MU[ground] * g * c2;
    let fx = (tx - s.vx) / dt - pgx, fz = (tz - s.vz) / dt - pgz;   // foot force per mass needed to reach the target now
    const fl = Math.hypot(fx, fz);
    if (fl > R) { fx *= R / fl; fz *= R / fl; }
    s.vx += (fx + pgx) * dt; s.vz += (fz + pgz) * dt;
    s.sliding = Math.hypot(pgx, pgz) > R;
    if (Math.abs(s.vx) < 1e-6 && Math.abs(s.vz) < 1e-6) { s.vx = 0; s.vz = 0; }

    const vySlope = groundVy(terrain, s.x, s.z, s.vx, s.vz);
    if (input.jump) {
      s.air = true; s.vy = vySlope + JUMP_V;
    } else {
      const x0 = s.x, z0 = s.z;
      moveH(terrain, s, dt, ev);
      const h = terrain.heightAt(s.x, s.z), yFree = s.y + vySlope * dt - 0.5 * g * dt * dt;
      if (h < yFree - LIFT_MARGIN) {                         // ground falls away faster than gravity: leave it
        s.air = true; s.y = yFree; s.vy = vySlope - g * dt;
      } else {
        s.y = h; s.vy = vySlope;
      }
      const dist = Math.hypot(s.x - x0, s.z - z0), v = dist / dt;
      if (dist > 0 && !s.air) {                               // one footstep per pi of phase
        const prev = s.phase; s.phase += dist / stepLength(v) * Math.PI;
        if (Math.floor(prev / Math.PI) !== Math.floor(s.phase / Math.PI)) {
          // Strength, design choice: louder with speed and effort, and on steep steps either way (push-off uphill,
          // braking downhill).
          const e = effortOf(v, grade, ground);
          ev.step = Math.max(ev.step, Math.min(1, 0.35 + 0.3 * Math.min(1, v / LOPE) + 0.25 * e + 0.5 * Math.abs(grade)));
        }
      }
      s.effortNow = effortOf(v, grade, ground);
      s.gait = v > GAIT_CHANGE ? "lope" : "walk";
      return;
    }
  }

  // In the air: no traction, no drag worth counting (air density about 0.015 kg/m^3). Exact constant-g step.
  if (blockedAt(terrain, s.x, s.z, s.y, s.vx, s.vz)) { s.vx = 0; s.vz = 0; ev.blocked = true; }
  s.x += s.vx * dt; s.z += s.vz * dt;
  s.y += s.vy * dt - 0.5 * g * dt * dt; s.vy -= g * dt;
  const h = terrain.heightAt(s.x, s.z);
  if (s.y <= h && s.vy <= 0) {
    ev.land = Math.max(ev.land, -s.vy);
    s.air = false; s.y = h; s.vy = groundVy(terrain, s.x, s.z, s.vx, s.vz);
  } else if (s.y < h) { s.y = h; }                           // rising through a slope: stay on it
  s.effortNow = s.effortNow || 0;
  s.gait = s.air ? "air" : "walk";
}

export function stepWalker(state, input, terrain, dt) {
  const s = { ...state, events: undefined };
  const ev = { step: 0, land: 0, blocked: false };
  const n = Math.max(1, Math.ceil(dt / 0.02)), h = dt / n;   // sub-steps of at most 20 ms
  for (let k = 0; k < n; k++) substep(s, k === 0 ? input : { ...input, jump: false }, terrain, h, ev);
  s.effort = (state.effort || 0) + ((s.effortNow || 0) - (state.effort || 0)) * (1 - Math.exp(-dt / EFFORT_TAU));
  s.events = ev;
  return s;
}
