// Overrides: a dedicated list editor for `ProjectConfig.overrides`, the per-sample manual loop
// decisions the waveform view's drag-to-set/reset flow writes into `ctx.config.overrides`.
// `param-form.js` deliberately skips array-of-object fields (`stages` gets its own editor;
// `overrides` gets this one, per CLAUDE.md's step-12 placement) since a flat row-per-property
// form is the wrong shape for either.
//
// Each row edits one override's loop points / disabled flag directly (reusing `numeric-field`
// and the same switch markup `param-form.js` uses) and can reset it outright — the same
// "Reset to auto-detect" action the waveform view offers per-sample, available here for every
// override at once without opening each sample individually.
//
// An override naming a (note, velocity) that isn't in the loaded sustain set is flagged rather
// than hidden or silently dropped: step 6 named this a known gap ("silently ignored... step 9
// can validate overrides against the loaded set and surface it in the UI") and this is where
// that surfacing finally happens.

import "./numeric-field.js";

let viewTokenCounter = 0;

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

/**
 * @param {HTMLElement} container Emptied and filled with the overrides list.
 * @param {{ api: any, config: any, markDirty: () => void, openSample: (id: string) => void }} ctx
 */
export function renderOverridesView(container, ctx) {
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
      if (container.dataset.viewToken !== token) return;
      container.innerHTML = `<div class="empty-state"><p>${escapeHtml(e.message)}</p></div>`;
    });
}

function buildView(container, samples, ctx) {
  container.innerHTML = "";
  const sustainByKey = new Map(
    samples.filter((s) => s.kind === "sustain").map((s) => [`${s.note}-${s.velocity}`, s])
  );

  const root = document.createElement("div");
  root.className = "overrides-view";
  container.appendChild(root);

  const overrides = () => ctx.config.overrides ?? [];

  function updateOverride(index, patch) {
    ctx.config.overrides = overrides().map((o, i) => (i === index ? { ...o, ...patch } : o));
    ctx.markDirty();
  }

  function removeOverride(index) {
    ctx.config.overrides = overrides().filter((_, i) => i !== index);
    ctx.markDirty();
    render();
  }

  function render() {
    root.innerHTML = "";
    const list = overrides();
    if (!list.length) {
      const p = document.createElement("p");
      p.className = "chart-empty";
      p.textContent =
        "No overrides yet. Drag a loop handle in the Waveform view to create one — an override always beats auto-detection.";
      root.appendChild(p);
      return;
    }

    const missing = list.filter((o) => !sustainByKey.has(`${o.note}-${o.velocity}`));
    if (missing.length) {
      const warn = document.createElement("p");
      warn.className = "override-warning";
      warn.textContent = `${missing.length} override${missing.length === 1 ? "" : "s"} name a sample outside the loaded set — flagged below, and has no effect until one matches.`;
      root.appendChild(warn);
    }

    const list_el = document.createElement("div");
    list_el.className = "overrides-list";
    list.forEach((override, index) => list_el.appendChild(renderRow(override, index)));
    root.appendChild(list_el);
  }

  function renderRow(override, index) {
    const sample = sustainByKey.get(`${override.note}-${override.velocity}`);
    const row = document.createElement("div");
    row.className = "override-row";

    const label = document.createElement("div");
    label.className = "override-label";
    const name = sample
      ? `${sample.note_name} v${override.velocity}`
      : `note ${override.note} v${override.velocity}`;
    label.innerHTML = `<span class="mono">${escapeHtml(name)}</span>`;
    if (!sample) {
      const flag = document.createElement("span");
      flag.className = "override-missing";
      flag.textContent = "not in loaded set";
      flag.title = "No loaded sustain sample matches this override — it has no effect until one does.";
      label.appendChild(flag);
    } else {
      label.classList.add("clickable");
      label.title = "Open this sample in the Waveform view";
      label.addEventListener("click", () => ctx.openSample(sample.id));
    }
    row.appendChild(label);

    const fields = document.createElement("div");
    fields.className = "override-fields";

    if (override.loop_disabled) {
      fields.appendChild(tag("loop disabled"));
    } else if (override.loop_start != null && override.loop_end != null) {
      fields.appendChild(
        numericField("start", override.loop_start, sample, (value) =>
          updateOverride(index, { loop_start: Math.round(value) })
        )
      );
      fields.appendChild(
        numericField("end", override.loop_end, sample, (value) =>
          updateOverride(index, { loop_end: Math.round(value) })
        )
      );
    } else {
      fields.appendChild(tag("no loop points set"));
    }
    row.appendChild(fields);

    const resetBtn = document.createElement("button");
    resetBtn.className = "btn btn-ghost";
    resetBtn.textContent = "Reset to auto";
    resetBtn.addEventListener("click", () => removeOverride(index));
    row.appendChild(resetBtn);

    return row;
  }

  function tag(text) {
    const el = document.createElement("span");
    el.className = "override-tag";
    el.textContent = text;
    return el;
  }

  function numericField(labelText, value, sample, onCommit) {
    const wrap = document.createElement("div");
    wrap.className = "override-field";
    const label = document.createElement("span");
    label.className = "override-field-label";
    label.textContent = labelText;
    wrap.appendChild(label);

    const el = document.createElement("numeric-field");
    el.min = 0;
    el.max = sample ? Math.max(0, sample.n_frames - 1) : Number.POSITIVE_INFINITY;
    el.step = 1;
    el.fineStep = 1;
    el.unit = "fr";
    el.value = value;
    el.addEventListener("numeric-change", (e) => onCommit(e.detail.value));
    wrap.appendChild(el);
    return wrap;
  }

  render();
}
