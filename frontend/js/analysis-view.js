// Analysis view: hand-rolled SVG charts (no charting library, per CLAUDE.md's step 11 plan).
// Two charts, chosen to match what the backend actually exposes rather than inventing fields:
// `/api/analysis` gives one rt_decay estimate per release sample (a scatter by note, colored by
// velocity), and `/api/matrix` gives peak/RMS/length per sample, from which a peak-level
// histogram over the sustain set is built client-side.

const CHART_WIDTH = 640;
const CHART_HEIGHT = 220;
const MARGIN = { top: 16, right: 16, bottom: 32, left: 44 };
let viewTokenCounter = 0;

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function plotArea() {
  return {
    x0: MARGIN.left,
    y0: MARGIN.top,
    x1: CHART_WIDTH - MARGIN.right,
    y1: CHART_HEIGHT - MARGIN.bottom,
  };
}

function axisLine(area) {
  const g = svgEl("g", { class: "chart-axes" });
  g.appendChild(
    svgEl("line", {
      x1: area.x0,
      y1: area.y1,
      x2: area.x1,
      y2: area.y1,
      style: "stroke:var(--border-strong)",
    })
  );
  g.appendChild(
    svgEl("line", {
      x1: area.x0,
      y1: area.y0,
      x2: area.x0,
      y2: area.y1,
      style: "stroke:var(--border-strong)",
    })
  );
  return g;
}

function scaleLinear(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}

/**
 * @param {HTMLElement} container Emptied and filled with the analysis charts.
 * @param {{ api: any }} ctx
 */
export function renderAnalysisView(container, ctx) {
  if (container._teardown) container._teardown();
  container.innerHTML = '<div class="view-loading">Loading analysis…</div>';
  const token = String(++viewTokenCounter);
  container.dataset.viewToken = token;

  Promise.all([ctx.api.getAnalysis(), ctx.api.getMatrix()])
    .then(([decay, matrix]) => {
      if (container.dataset.viewToken !== token) return; // superseded by a later call
      buildView(container, decay, matrix);
    })
    .catch((e) => {
      if (container.dataset.viewToken !== token) return;
      container.innerHTML = `<div class="empty-state"><p>${e.message}</p></div>`;
    });
}

function buildView(container, decay, matrix) {
  container.innerHTML = "";
  const root = document.createElement("div");
  root.className = "analysis-view";

  root.appendChild(sectionHeader("Release decay rate (rt_decay) by note"));
  root.appendChild(decay.length ? decayScatter(decay) : emptyNote("No release samples."));

  root.appendChild(sectionHeader("Sustain peak level distribution"));
  const sustainPeaks = matrix.filter((r) => r.kind === "sustain").map((r) => r.peak_db);
  root.appendChild(sustainPeaks.length ? peakHistogram(sustainPeaks) : emptyNote("No sustain samples."));

  container.appendChild(root);
}

function sectionHeader(text) {
  const h = document.createElement("div");
  h.className = "group-header";
  h.textContent = text;
  return h;
}

function emptyNote(text) {
  const p = document.createElement("p");
  p.className = "chart-empty";
  p.textContent = text;
  return p;
}

function decayScatter(decay) {
  const area = plotArea();
  const notes = decay.map((d) => d.note);
  const values = decay.map((d) => d.rt_decay_db_per_s);
  const velocities = decay.map((d) => d.velocity);

  const yMax = Math.max(24, ...values);
  const xScale = scaleLinear([Math.min(...notes) - 1, Math.max(...notes) + 1], [area.x0, area.x1]);
  const yScale = scaleLinear([0, yMax], [area.y1, area.y0]);
  const vMin = Math.min(...velocities);
  const vMax = Math.max(...velocities);

  const svg = svgEl("svg", { viewBox: `0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`, class: "chart" });
  svg.appendChild(axisLine(area));

  for (const gridY of [0, 6, 12, 18, 24]) {
    if (gridY > yMax) continue;
    const y = yScale(gridY);
    svg.appendChild(svgEl("line", { x1: area.x0, y1: y, x2: area.x1, y2: y, style: "stroke:var(--border)" }));
    svg.appendChild(textLabel(area.x0 - 6, y + 3, String(gridY), "end"));
  }
  svg.appendChild(textLabel((area.x0 + area.x1) / 2, CHART_HEIGHT - 6, "MIDI note"));
  svg.appendChild(textLabel(12, (area.y0 + area.y1) / 2, "dB/s", "middle", -90));

  decay.forEach((d) => {
    const t = vMax > vMin ? (d.velocity - vMin) / (vMax - vMin) : 0.5;
    const radius = 3 + t * 3;
    const circle = svgEl("circle", {
      cx: xScale(d.note),
      cy: yScale(d.rt_decay_db_per_s),
      r: radius,
      style: `fill:var(--accent); fill-opacity:${0.35 + t * 0.5}`,
    });
    circle.appendChild(svgEl("title", {})).textContent = `note ${d.note} vel ${d.velocity}: ${d.rt_decay_db_per_s.toFixed(2)} dB/s`;
    svg.appendChild(circle);
  });

  return svg;
}

function peakHistogram(values) {
  const area = plotArea();
  const binCount = 16;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const bins = new Array(binCount).fill(0);
  values.forEach((v) => {
    const idx = Math.min(binCount - 1, Math.floor(((v - lo) / span) * binCount));
    bins[idx] += 1;
  });
  const maxCount = Math.max(...bins, 1);

  const xScale = scaleLinear([0, binCount], [area.x0, area.x1]);
  const yScale = scaleLinear([0, maxCount], [area.y1, area.y0]);
  const barWidth = (area.x1 - area.x0) / binCount;

  const svg = svgEl("svg", { viewBox: `0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`, class: "chart" });
  svg.appendChild(axisLine(area));
  svg.appendChild(textLabel((area.x0 + area.x1) / 2, CHART_HEIGHT - 6, "Peak level (dB)"));
  svg.appendChild(textLabel(12, (area.y0 + area.y1) / 2, "count", "middle", -90));

  bins.forEach((count, i) => {
    const x = xScale(i);
    const y = yScale(count);
    const rect = svgEl("rect", {
      x: x + 1,
      y,
      width: Math.max(0, barWidth - 2),
      height: area.y1 - y,
      style: "fill:var(--accent-dim)",
    });
    rect.appendChild(svgEl("title", {})).textContent = `${(lo + (i / binCount) * span).toFixed(1)} dB: ${count}`;
    svg.appendChild(rect);
  });

  [lo, lo + span / 2, hi].forEach((v, i) => {
    svg.appendChild(textLabel(xScale(i * (binCount / 2)), area.y1 + 16, v.toFixed(0)));
  });

  return svg;
}

function textLabel(x, y, text, anchor = "middle", rotate = 0) {
  const attrs = { x, y, "text-anchor": anchor, class: "chart-label" };
  if (rotate) attrs.transform = `rotate(${rotate} ${x} ${y})`;
  const el = svgEl("text", attrs);
  el.textContent = text;
  return el;
}
