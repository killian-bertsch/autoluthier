// Waveform + loop editor. wavesurfer.js renders the waveform from server-precomputed peaks
// (`/api/peaks/{id}`) — decoding a whole render client-side is documented as far too slow here,
// so wavesurfer never gets a real audio URL, only `peaks` + `duration`. Regions plugin draws
// the loop region; dragging its handles updates a per-sample override in `ctx.config.overrides`
// and calls `ctx.markDirty()`, the same dirty/Save flow every other config edit in this app uses
// — there is no separate auto-persist path for loop points.
//
// Loop *audition* is deliberately not wavesurfer's own region-loop playback (documented as not
// sample-accurate): a hand-wired `AudioBufferSourceNode` with `loop`/`loopStart`/`loopEnd` set
// directly is the only way to hear the exact seam a sampler would produce.

const ZOOM_LEVELS = [
  { label: "Overview", index: 0 },
  { label: "Mid", index: 1 },
  { label: "Detail", index: 2 },
];
const SEAM_INSET_FRAMES = 120; // frames shown on each side of the seam in the inset canvas

let audioCtx = null;
function sharedAudioContext() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

let viewTokenCounter = 0;

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/**
 * @param {HTMLElement} container Emptied and filled with the waveform/loop editor.
 * @param {{
 *   api: any,
 *   config: any,
 *   selectedSampleId: string | null,
 *   onSelectSample: (id: string) => void,
 *   markDirty: () => void,
 * }} ctx
 */
export function renderWaveformView(container, ctx) {
  if (container._teardown) container._teardown();
  container.innerHTML = '<div class="view-loading">Loading samples…</div>';
  const token = String(++viewTokenCounter);
  container.dataset.viewToken = token;

  ctx.api
    .listSamples()
    .then((samples) => {
      if (container.dataset.viewToken !== token) return; // superseded by a later call
      buildView(container, samples, ctx);
    })
    .catch((e) => {
      container.innerHTML = `<div class="empty-state"><p>${escapeHtml(e.message)}</p></div>`;
    });
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function buildView(container, samples, ctx) {
  container.innerHTML = "";
  const state = {
    samples,
    sample: null, // the selected SampleSummary
    wavesurfer: null,
    regions: null,
    buffers: { raw: null, preview: null }, // decoded AudioBuffers, cached per source
    playingSource: null,
    sourceNode: null,
    zoomIndex: 1,
    snapToZero: false,
    zeroCrossings: null, // fetched lazily, cached per sample id
  };
  container._teardown = () => teardown(state);

  const root = document.createElement("div");
  root.className = "waveform-view";
  root.innerHTML = `
    <div class="wv-toolbar">
      <select class="field-select wv-sample-picker"></select>
      <select class="field-select wv-zoom-picker">
        ${ZOOM_LEVELS.map((z) => `<option value="${z.index}">${z.label}</option>`).join("")}
      </select>
      <label class="wv-snap-toggle">
        <input type="checkbox" class="wv-snap-checkbox" />
        Snap to zero crossing
      </label>
    </div>
    <div class="wv-waveform"></div>
    <div class="wv-panel" style="display:none">
      <div class="wv-audition">
        <button class="btn wv-play">▶ Play loop</button>
        <div class="wv-ab">
          <button class="btn btn-ghost wv-ab-raw is-active" data-source="raw">Dry</button>
          <button class="btn btn-ghost wv-ab-preview" data-source="preview">Processed</button>
        </div>
        <button class="btn btn-ghost wv-run-preview" style="display:none">Run preview</button>
        <span class="wv-audition-message"></span>
      </div>
      <div class="wv-loop-info">
        <span class="wv-loop-label">No loop detected.</span>
        <div class="wv-loop-actions">
          <button class="btn btn-ghost wv-reset-auto">Reset to auto-detect</button>
          <button class="btn btn-ghost wv-disable-loop">Disable loop</button>
        </div>
      </div>
      <div class="wv-seam">
        <div class="group-header">Loop seam inset</div>
        <canvas class="wv-seam-canvas" width="480" height="90"></canvas>
      </div>
    </div>
  `;
  container.appendChild(root);

  const picker = root.querySelector(".wv-sample-picker");
  const sustain = samples.filter((s) => s.kind === "sustain");
  const release = samples.filter((s) => s.kind === "release");
  picker.innerHTML =
    `<optgroup label="Sustain">${sustain
      .map((s) => `<option value="${s.id}">${s.note_name} v${s.velocity}</option>`)
      .join("")}</optgroup>` +
    (release.length
      ? `<optgroup label="Release">${release
          .map((s) => `<option value="${s.id}">${s.note_name} v${s.velocity} (release)</option>`)
          .join("")}</optgroup>`
      : "");

  const initialId = ctx.selectedSampleId && samples.some((s) => s.id === ctx.selectedSampleId)
    ? ctx.selectedSampleId
    : samples[0]?.id;
  if (initialId) picker.value = initialId;

  picker.addEventListener("change", () => {
    ctx.onSelectSample(picker.value);
    loadSample(root, state, picker.value, ctx);
  });

  root.querySelector(".wv-zoom-picker").value = String(state.zoomIndex);
  root.querySelector(".wv-zoom-picker").addEventListener("change", (e) => {
    state.zoomIndex = Number(e.target.value);
    if (state.sample) loadWaveform(root, state, ctx);
  });

  root.querySelector(".wv-snap-checkbox").addEventListener("change", (e) => {
    state.snapToZero = e.target.checked;
  });

  root.querySelector(".wv-play").addEventListener("click", () => togglePlay(root, state, ctx));
  root.querySelectorAll(".wv-ab button").forEach((btn) => {
    btn.addEventListener("click", () => setAuditionSource(root, state, btn.dataset.source));
  });
  root.querySelector(".wv-run-preview").addEventListener("click", () => runPreviewAndRetry(root, state, ctx));
  root.querySelector(".wv-reset-auto").addEventListener("click", () => applyOverride(root, state, ctx, "reset"));
  root.querySelector(".wv-disable-loop").addEventListener("click", () => applyOverride(root, state, ctx, "disable"));

  if (initialId) loadSample(root, state, initialId, ctx);
}

function loadSample(root, state, sampleId, ctx) {
  state.sample = state.samples.find((s) => s.id === sampleId) ?? null;
  state.buffers = { raw: null, preview: null };
  state.zeroCrossings = null;
  stopPlayback(state);
  if (state.sample) loadWaveform(root, state, ctx);
}

function loadWaveform(root, state, ctx) {
  const sample = state.sample;
  const waveformEl = root.querySelector(".wv-waveform");
  const panel = root.querySelector(".wv-panel");

  ctx.api.getPeaks(sample.id).then((levels) => {
    if (state.sample !== sample) return; // selection changed mid-fetch
    const level = levels[Math.min(state.zoomIndex, levels.length - 1)];
    const peaksArray = level.mins[0].map((min, i) => Math.max(Math.abs(min), Math.abs(level.maxs[0][i])));

    if (state.wavesurfer) {
      state.wavesurfer.destroy();
      state.wavesurfer = null;
    }
    waveformEl.innerHTML = "";

    const ws = window.WaveSurfer.create({
      container: waveformEl,
      height: 120,
      waveColor: cssVar("--border-strong") || "#3a3d43",
      progressColor: cssVar("--accent-dim") || "#8a6428",
      cursorColor: cssVar("--accent-bright") || "#f0b357",
      peaks: [peaksArray],
      duration: sample.duration_s,
      interact: false,
      normalize: true,
    });
    const regions = ws.registerPlugin(window.WaveSurfer.Regions.create());
    state.wavesurfer = ws;
    state.regions = regions;

    panel.style.display = "";
    drawLoopRegion(root, state, ctx);
  });
}

function currentOverride(ctx, sample) {
  return (ctx.config.overrides ?? []).find((o) => o.note === sample.note && o.velocity === sample.velocity);
}

function effectiveLoopPoints(ctx, sample) {
  const override = sample.kind === "sustain" ? currentOverride(ctx, sample) : null;
  if (override?.loop_disabled) return null;
  if (override?.loop_start != null && override?.loop_end != null) {
    return { start: override.loop_start, end: override.loop_end, isOverride: true };
  }
  if (sample.has_loop) return { fromDetection: true };
  return null;
}

function drawLoopRegion(root, state, ctx) {
  const sample = state.sample;
  state.regions.clearRegions();
  const label = root.querySelector(".wv-loop-label");
  const canEdit = sample.kind === "sustain";

  const points = effectiveLoopPoints(ctx, sample);
  if (!points) {
    label.textContent = "No loop.";
    return;
  }
  if (points.fromDetection && !points.isOverride) {
    // Auto-detected: we don't have the exact detected frames without processing the chain,
    // so the region is only drawn once a manual override supplies concrete points. Detection
    // itself is reported by /api/samples' has_loop flag only.
    label.textContent = "Loop auto-detected by the pipeline (exact points shown after an edit here).";
  }

  const startFrac = points.isOverride ? points.start / sample.n_frames : null;
  const endFrac = points.isOverride ? points.end / sample.n_frames : null;
  if (startFrac == null) return;

  const region = state.regions.addRegion({
    start: startFrac * sample.duration_s,
    end: endFrac * sample.duration_s,
    color: "rgba(217, 154, 60, 0.15)",
    drag: canEdit,
    resize: canEdit,
    id: "loop",
  });
  updateLoopLabel(label, points.start, points.end, sample);
  drawSeamInset(root, state, points.start, points.end);

  if (!canEdit) return;
  region.on("update-end", () => {
    let startFrame = Math.round((region.start / sample.duration_s) * sample.n_frames);
    let endFrame = Math.round((region.end / sample.duration_s) * sample.n_frames);
    if (state.snapToZero) {
      ensureZeroCrossings(state, ctx).then((crossings) => {
        if (!crossings) return finalizeRegion(root, state, ctx, startFrame, endFrame);
        finalizeRegion(root, state, ctx, snapTo(crossings, startFrame), snapTo(crossings, endFrame));
      });
    } else {
      finalizeRegion(root, state, ctx, startFrame, endFrame);
    }
  });
}

function snapTo(crossings, frame) {
  let best = crossings[0] ?? frame;
  let bestDist = Infinity;
  for (const c of crossings) {
    const dist = Math.abs(c - frame);
    if (dist < bestDist) {
      bestDist = dist;
      best = c;
    }
  }
  return best;
}

function ensureZeroCrossings(state, ctx) {
  if (state.zeroCrossings) return Promise.resolve(state.zeroCrossings);
  return ctx.api
    .getZeroCrossings(state.sample.id)
    .then((crossings) => {
      state.zeroCrossings = crossings;
      return crossings;
    })
    .catch(() => null);
}

function finalizeRegion(root, state, ctx, startFrame, endFrame) {
  if (endFrame <= startFrame) endFrame = startFrame + 1;
  applyLoopOverride(ctx, state.sample, startFrame, endFrame);
  updateLoopLabel(root.querySelector(".wv-loop-label"), startFrame, endFrame, state.sample);
  drawSeamInset(root, state, startFrame, endFrame);
}

function updateLoopLabel(el, start, end, sample) {
  el.textContent = `Loop ${start} → ${end} (${((end - start) / sample.sample_rate).toFixed(3)}s)`;
}

/** Write (or replace) this sample's loop override in `ctx.config.overrides` and mark dirty. */
function applyLoopOverride(ctx, sample, loopStart, loopEnd) {
  const overrides = (ctx.config.overrides ?? []).filter(
    (o) => !(o.note === sample.note && o.velocity === sample.velocity)
  );
  overrides.push({
    note: sample.note,
    velocity: sample.velocity,
    loop_start: loopStart,
    loop_end: loopEnd,
    loop_disabled: false,
  });
  ctx.config.overrides = overrides;
  ctx.markDirty();
}

function applyOverride(root, state, ctx, action) {
  const sample = state.sample;
  const overrides = (ctx.config.overrides ?? []).filter(
    (o) => !(o.note === sample.note && o.velocity === sample.velocity)
  );
  if (action === "disable") {
    overrides.push({ note: sample.note, velocity: sample.velocity, loop_disabled: true });
  }
  // "reset" just drops any override — auto-detection takes over again.
  ctx.config.overrides = overrides;
  ctx.markDirty();
  drawLoopRegion(root, state, ctx);
}

function drawSeamInset(root, state, loopStart, loopEnd) {
  const canvas = root.querySelector(".wv-seam-canvas");
  const ctx2d = canvas.getContext("2d");
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);
  const buffer = state.buffers[state.playingSource ?? "raw"];
  if (!buffer) {
    ctx2d.fillStyle = cssVar("--text-faint") || "#6b6c70";
    ctx2d.font = "11px monospace";
    ctx2d.fillText("Press Play to decode audio for the seam inset.", 8, canvas.height / 2);
    return;
  }
  const data = buffer.getChannelData(0);
  const before = sliceAround(data, loopEnd, -SEAM_INSET_FRAMES, SEAM_INSET_FRAMES);
  const after = sliceAround(data, loopStart, -SEAM_INSET_FRAMES, SEAM_INSET_FRAMES);
  const combined = [...before, ...after];
  drawWaveformLine(ctx2d, canvas, combined, before.length);
}

function sliceAround(data, center, from, to) {
  const out = [];
  for (let i = from; i < to; i++) {
    const idx = center + i;
    out.push(idx >= 0 && idx < data.length ? data[idx] : 0);
  }
  return out;
}

function drawWaveformLine(ctx2d, canvas, values, seamIndex) {
  const w = canvas.width;
  const h = canvas.height;
  const midY = h / 2;
  ctx2d.strokeStyle = cssVar("--accent") || "#d99a3c";
  ctx2d.beginPath();
  values.forEach((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = midY - v * midY * 0.9;
    if (i === 0) ctx2d.moveTo(x, y);
    else ctx2d.lineTo(x, y);
  });
  ctx2d.stroke();

  const seamX = (seamIndex / (values.length - 1)) * w;
  ctx2d.strokeStyle = cssVar("--danger") || "#c25b4a";
  ctx2d.beginPath();
  ctx2d.moveTo(seamX, 0);
  ctx2d.lineTo(seamX, h);
  ctx2d.stroke();
}

function setAuditionSource(root, state, source) {
  root.querySelectorAll(".wv-ab button").forEach((b) => b.classList.toggle("is-active", b.dataset.source === source));
  state.desiredSource = source;
  const msg = root.querySelector(".wv-audition-message");
  const runBtn = root.querySelector(".wv-run-preview");
  msg.textContent = "";
  runBtn.style.display = "none";
}

function decodeSource(state, ctx, source) {
  if (state.buffers[source]) return Promise.resolve(state.buffers[source]);
  return fetch(ctx.api.audioUrl(state.sample.id, source))
    .then((r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.arrayBuffer();
    })
    .then((buf) => sharedAudioContext().decodeAudioData(buf))
    .then((decoded) => {
      state.buffers[source] = decoded;
      return decoded;
    });
}

function togglePlay(root, state, ctx) {
  const playBtn = root.querySelector(".wv-play");
  if (state.sourceNode) {
    stopPlayback(state);
    playBtn.textContent = "▶ Play loop";
    return;
  }

  const source = state.desiredSource ?? "raw";
  const msg = root.querySelector(".wv-audition-message");
  const runBtn = root.querySelector(".wv-run-preview");
  msg.textContent = "Decoding…";
  runBtn.style.display = "none";

  decodeSource(state, ctx, source)
    .then((buffer) => startPlayback(root, state, buffer, source))
    .then(() => {
      msg.textContent = "";
      playBtn.textContent = "■ Stop";
      drawSeamInset(root, state, effectiveStart(state, ctx), effectiveEnd(state, ctx));
    })
    .catch((e) => {
      if (source === "preview" && String(e.message) === "404") {
        msg.textContent = "No preview covers this sample yet.";
        runBtn.style.display = "";
      } else {
        msg.textContent = `Couldn't load audio: ${e.message}`;
      }
    });
}

function effectiveStart(state, ctx) {
  const points = effectiveLoopPoints(ctx, state.sample);
  return points?.start ?? 0;
}

function effectiveEnd(state, ctx) {
  const points = effectiveLoopPoints(ctx, state.sample);
  return points?.end ?? state.sample.n_frames;
}

function startPlayback(root, state, buffer, source) {
  const ac = sharedAudioContext();
  const node = ac.createBufferSource();
  node.buffer = buffer;
  const sr = buffer.sampleRate;
  // Release samples are never looped (dsp.loop only ever runs on sustain) — audition just
  // plays them once.
  const points = state.sample.kind === "sustain" ? currentLoopFrames(state) : null;
  if (points) {
    node.loop = true;
    node.loopStart = points.start / sr;
    node.loopEnd = points.end / sr;
  }
  node.connect(ac.destination);
  node.start();
  node.onended = () => {
    if (state.sourceNode === node) {
      state.sourceNode = null;
      root.querySelector(".wv-play").textContent = "▶ Play loop";
    }
  };
  state.sourceNode = node;
  state.playingSource = source;
  return Promise.resolve();
}

function currentLoopFrames(state) {
  const region = state.regions?.getRegions?.().find((r) => r.id === "loop");
  if (!region) return null;
  return {
    start: Math.round((region.start / state.sample.duration_s) * state.sample.n_frames),
    end: Math.round((region.end / state.sample.duration_s) * state.sample.n_frames),
  };
}

function runPreviewAndRetry(root, state, ctx) {
  const msg = root.querySelector(".wv-audition-message");
  msg.textContent = "Running preview…";
  ctx.api
    .runPreview({})
    .then(() => {
      state.buffers.preview = null;
      msg.textContent = "";
      togglePlay(root, state, ctx);
    })
    .catch((e) => {
      msg.textContent = `Preview failed: ${e.message}`;
    });
}

function stopPlayback(state) {
  if (state.sourceNode) {
    try {
      state.sourceNode.stop();
    } catch {
      // already stopped
    }
    state.sourceNode.disconnect();
    state.sourceNode = null;
  }
  state.playingSource = null;
}

function teardown(state) {
  stopPlayback(state);
  if (state.wavesurfer) {
    state.wavesurfer.destroy();
    state.wavesurfer = null;
  }
}
