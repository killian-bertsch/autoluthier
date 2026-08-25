// Sample matrix: one canvas, not one DOM element per cell — an 18-layer x 32-note instrument is
// already 576 cells and this is meant to scale past that. Cells are colored by a selectable
// metric from `/api/matrix`; note names for the tooltip/click-through come from `/api/samples`
// rather than reimplementing MIDI note naming in JS (`domain/notes.py` is the one place that
// logic lives).

const CELL_SIZE = 16;
const SILENT_PEAK_DB = -60.0; // flagged as an anomaly regardless of which metric is displayed

const METRICS = [
  { key: "peak_db", label: "Peak (dB)" },
  { key: "rms_db", label: "RMS (dB)" },
  { key: "duration_s", label: "Length (s)" },
];

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  const n = parseInt(clean, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

let viewTokenCounter = 0;

function lerpColor(a, b, t) {
  const ca = hexToRgb(a);
  const cb = hexToRgb(b);
  const r = Math.round(ca[0] + (cb[0] - ca[0]) * t);
  const g = Math.round(ca[1] + (cb[1] - ca[1]) * t);
  const bch = Math.round(ca[2] + (cb[2] - ca[2]) * t);
  return `rgb(${r}, ${g}, ${bch})`;
}

/**
 * @param {HTMLElement} container Emptied and filled with the matrix.
 * @param {{ api: any, openSample: (id: string) => void }} ctx
 */
export function renderMatrixView(container, ctx) {
  if (container._teardown) container._teardown();
  container.innerHTML = '<div class="view-loading">Loading matrix…</div>';
  const token = String(++viewTokenCounter);
  container.dataset.viewToken = token;

  Promise.all([ctx.api.getMatrix(), ctx.api.listSamples()])
    .then(([rows, samples]) => {
      if (container.dataset.viewToken !== token) return; // superseded by a later call
      buildView(container, rows, samples, ctx);
    })
    .catch((e) => {
      if (container.dataset.viewToken !== token) return;
      container.innerHTML = `<div class="empty-state"><p>${e.message}</p></div>`;
    });
}

function buildView(container, rows, samples, ctx) {
  container.innerHTML = "";
  if (!rows.length) {
    container.innerHTML = '<div class="empty-state"><p>No samples loaded — nothing to show in the matrix.</p></div>';
    return;
  }
  const idByKey = new Map(samples.map((s) => [`${s.kind}-${s.note}-${s.velocity}`, s]));

  const state = { kind: "sustain", metric: "peak_db" };

  const root = document.createElement("div");
  root.className = "matrix-view";
  root.innerHTML = `
    <div class="wv-toolbar">
      <select class="field-select mv-kind-picker">
        <option value="sustain">Sustain</option>
        <option value="release">Release</option>
      </select>
      <select class="field-select mv-metric-picker">
        ${METRICS.map((m) => `<option value="${m.key}">${m.label}</option>`).join("")}
      </select>
      <span class="mv-legend"></span>
    </div>
    <div class="mv-canvas-wrap">
      <canvas class="mv-canvas"></canvas>
    </div>
    <div class="mv-tooltip" style="display:none"></div>
  `;
  container.appendChild(root);

  const canvas = root.querySelector(".mv-canvas");
  const tooltip = root.querySelector(".mv-tooltip");
  const kindPicker = root.querySelector(".mv-kind-picker");
  const metricPicker = root.querySelector(".mv-metric-picker");
  kindPicker.value = state.kind;
  metricPicker.value = state.metric;

  let layout = null; // computed by draw(): notes[], velocities[], cellFor(note, vel)

  function draw() {
    const filtered = rows.filter((r) => r.kind === state.kind);
    const notes = [...new Set(filtered.map((r) => r.note))].sort((a, b) => a - b);
    const velocities = [...new Set(filtered.map((r) => r.velocity))].sort((a, b) => b - a);
    const byKey = new Map(filtered.map((r) => [`${r.note}-${r.velocity}`, r]));

    canvas.width = notes.length * CELL_SIZE;
    canvas.height = velocities.length * CELL_SIZE;
    const c2d = canvas.getContext("2d");
    c2d.fillStyle = cssVar("--bg-1") || "#131417";
    c2d.fillRect(0, 0, canvas.width, canvas.height);

    const values = filtered.map((r) => r[state.metric]).filter((v) => Number.isFinite(v));
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const low = cssVar("--bg-3") || "#232529";
    const high = cssVar("--accent-bright") || "#f0b357";
    const danger = cssVar("--danger") || "#c25b4a";

    notes.forEach((note, xi) => {
      velocities.forEach((vel, yi) => {
        const row = byKey.get(`${note}-${vel}`);
        const x = xi * CELL_SIZE;
        const y = yi * CELL_SIZE;
        if (!row) {
          c2d.strokeStyle = cssVar("--border") || "#2c2e33";
          c2d.strokeRect(x + 0.5, y + 0.5, CELL_SIZE - 1, CELL_SIZE - 1);
          return;
        }
        const t = hi > lo ? (row[state.metric] - lo) / (hi - lo) : 0.5;
        c2d.fillStyle = lerpColor(low, high, Math.max(0, Math.min(1, t)));
        c2d.fillRect(x, y, CELL_SIZE, CELL_SIZE);

        if (row.peak_db < SILENT_PEAK_DB) {
          c2d.strokeStyle = danger;
          c2d.lineWidth = 2;
          c2d.beginPath();
          c2d.moveTo(x + 3, y + 3);
          c2d.lineTo(x + CELL_SIZE - 3, y + CELL_SIZE - 3);
          c2d.moveTo(x + CELL_SIZE - 3, y + 3);
          c2d.lineTo(x + 3, y + CELL_SIZE - 3);
          c2d.stroke();
          c2d.lineWidth = 1;
        }
      });
    });

    root.querySelector(".mv-legend").textContent =
      values.length ? `${lo.toFixed(1)} … ${hi.toFixed(1)}  (× = below ${SILENT_PEAK_DB} dB peak)` : "";

    layout = {
      notes,
      velocities,
      cellFor(px, py) {
        const xi = Math.floor(px / CELL_SIZE);
        const yi = Math.floor(py / CELL_SIZE);
        if (xi < 0 || xi >= notes.length || yi < 0 || yi >= velocities.length) return null;
        const note = notes[xi];
        const vel = velocities[yi];
        return { note, vel, row: byKey.get(`${note}-${vel}`) };
      },
    };
  }

  kindPicker.addEventListener("change", () => {
    state.kind = kindPicker.value;
    draw();
  });
  metricPicker.addEventListener("change", () => {
    state.metric = metricPicker.value;
    draw();
  });

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const cell = layout?.cellFor(e.clientX - rect.left, e.clientY - rect.top);
    if (!cell || !cell.row) {
      tooltip.style.display = "none";
      return;
    }
    const sample = idByKey.get(`${state.kind}-${cell.note}-${cell.vel}`);
    tooltip.style.display = "";
    tooltip.style.left = `${e.clientX - rect.left + 12}px`;
    tooltip.style.top = `${e.clientY - rect.top + 12}px`;
    tooltip.textContent = `${sample ? sample.note_name : cell.note} v${cell.vel} — peak ${cell.row.peak_db.toFixed(
      1
    )} dB, rms ${cell.row.rms_db.toFixed(1)} dB, ${cell.row.duration_s.toFixed(2)}s`;
  });
  canvas.addEventListener("mouseleave", () => {
    tooltip.style.display = "none";
  });
  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const cell = layout?.cellFor(e.clientX - rect.left, e.clientY - rect.top);
    if (!cell) return;
    const sample = idByKey.get(`${state.kind}-${cell.note}-${cell.vel}`);
    if (sample) ctx.openSample(sample.id);
  });

  draw();
}
