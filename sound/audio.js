// Sound for the Mars walk: the wind of Mars, recorded by the SuperCam microphone on Perseverance (Jezero crater), heard
// as if you stood there with no suit. Nothing is synthesised. Web Audio only: buffers and native nodes.
// Levels are marked MEASURED (with source) or DESIGN (a choice). Sources and reasoning: NOTES.md. Asset levels:
// sounds.json (written by analyse.py).
//
//   const audio = createAudio();
//   startButton.onclick = () => audio.start();      // needs a user gesture
//   each frame: audio.update({ dt });

const P_REF = 20e-6; // Pa, 0 dB SPL

// MEASURED: the recordings are calibrated in pascals at the microphone, 2.1 m above the ground (Maurice et al. 2022,
// Nature 605:653). Their Leq is 28 to 36 dB SPL in 10-300 Hz, about as loud as a quiet room. Thin air (about 20 dB
// quieter than on Earth) and the loss of high frequencies are already in them.
// DESIGN: playback calibration. The browser cannot know the listener's playback level, so a digital 1.0 is set to
// 62 dB SPL: the loudest recording (36 dB SPL) then plays at about -26 dBFS, audible at a normal volume. The three
// recordings keep their measured levels relative to each other.
const FULL_SCALE_DB_SPL = 62;

// Wind loop switching. DESIGN.
const WIND_HOLD_S = [80, 160]; // play one recording for a random time in this range
const WIND_XFADE_S = 10;

const dbToGain = (db) => Math.pow(10, db / 20);
const paToGain = (pa) => pa / (P_REF * dbToGain(FULL_SCALE_DB_SPL));

export function createAudio({ base = new URL('./', import.meta.url).href } = {}) {
  let ctx = null, out = null, loading = null;
  let wind = [], windIdx = 0, windNext = 0;

  async function loadAssets() {
    const meta = await (await fetch(new URL('sounds.json', base))).json();
    // Safari may not decode Ogg Opus: fall back to an AAC copy of the same sound (made with afconvert, same timeline)
    const get = async (f) => ctx.decodeAudioData(await (await fetch(new URL(f, base))).arrayBuffer());
    const decode = async (a) => { try { return await get(a.file); } catch (e) { return get(a.file.replace(/\.opus$/, '.m4a')); } };
    const c = ctx;
    const bufs = await Promise.all(meta.wind.map(decode));
    if (ctx !== c) return; // stopped while loading
    wind = bufs.map((buf, i) => {
      const a = meta.wind[i], src = ctx.createBufferSource(); src.buffer = buf; src.loop = true;
      if (a.loop_end_s) { src.loopStart = a.loop_start_s; src.loopEnd = a.loop_end_s; }
      const g = ctx.createGain(); g.gain.value = 0;
      src.connect(g).connect(out); src.start(0, a.loop_start_s || 0);
      return { g, unit: paToGain(a.pa_per_full_scale) };   // gain for this recording's measured level
    });
    windIdx = Math.floor(Math.random() * wind.length);
    wind[windIdx].g.gain.setTargetAtTime(wind[windIdx].unit, ctx.currentTime, 1);
    windNext = ctx.currentTime + hold();
  }

  const hold = () => WIND_HOLD_S[0] + Math.random() * (WIND_HOLD_S[1] - WIND_HOLD_S[0]);

  function switchWind() {
    const t = ctx.currentTime, next = (windIdx + 1 + Math.floor(Math.random() * (wind.length - 1))) % wind.length;
    wind[windIdx].g.gain.setTargetAtTime(0, t, WIND_XFADE_S / 3);
    wind[next].g.gain.setTargetAtTime(wind[next].unit, t, WIND_XFADE_S / 3);
    windIdx = next; windNext = t + hold();
  }

  return {
    async start() {
      if (ctx) { await ctx.resume(); return loading; }
      ctx = new AudioContext({ sampleRate: 48000 });
      await ctx.resume();
      out = ctx.createGain(); out.connect(ctx.destination);
      loading = loadAssets();
      return loading;
    },
    stop() {
      if (ctx) ctx.close().catch(() => {});
      ctx = null; wind = [];
    },
    setPaused(p) {
      if (ctx) (p ? ctx.suspend() : ctx.resume()).catch(() => {});
    },
    update() {
      if (!ctx || ctx.state !== 'running') return;
      if (wind.length > 1 && ctx.currentTime > windNext) switchWind();
    },
    info() { return { wind: wind.length ? windIdx : null }; },
  };
}
