// Auto-renders a form from a JSON Schema object + its `$defs`, one `.param-row` per
// property, grouped under `.group-header`s by the `group` hint and ordered by `order`.
// This is what makes "add a field once in `config/schema.py`" reach the UI for free —
// nothing here knows about `Recording`, `Output`, or any other specific model by name.
//
// Only scalar fields (string/number/integer/boolean, plain or enum) are rendered.
// Array-of-object fields (`stages`, `overrides`) are structurally different editors
// (reorderable lists, not a flat row each) and are out of scope here — `stage-chain.js`
// handles `stages`; `overrides` gets its editor in step 12 per the plan.

import { groupProperties, labelFor, listProperties, resolveRef } from "./schema-utils.js";
import "./numeric-field.js";

/**
 * @param {HTMLElement} container Emptied and filled with the rendered form.
 * @param {any} schema The (possibly $ref-wrapped) object schema to render.
 * @param {Record<string, any>} defs The document's `$defs`.
 * @param {Record<string, any>} data Current values, read by property name.
 * @param {(name: string, value: any) => void} onChange Called with the new value on commit.
 * @param {(() => void) | undefined} onLivePreview If given, wired to every numeric field's
 *   continuous `numeric-input` event (mid-drag, not just on commit) — the caller's own debounced
 *   preview trigger. Omitted entirely for sections whose fields don't feed the DSP chain (e.g.
 *   Recording, Selection), where a live audio preview would be misleading.
 * @returns {string[]} Names of the properties actually rendered (skips array-of-object fields).
 */
export function renderParamForm(container, schema, defs, data, onChange, onLivePreview) {
  container.innerHTML = "";
  const properties = listProperties(schema, defs).filter((p) => isScalarField(p.node, defs));
  const grouped = groupProperties(properties);
  const rendered = [];

  for (const [group, fields] of grouped) {
    if (group) {
      const header = document.createElement("div");
      header.className = "group-header";
      header.textContent = group;
      container.appendChild(header);
    }
    for (const field of fields) {
      container.appendChild(renderRow(field, data, onChange, defs, onLivePreview));
      rendered.push(field.name);
    }
  }
  return rendered;
}

/** Array/object fields need a dedicated editor; everything else is one row. */
function isScalarField(node, defs) {
  const resolved = resolveRef(node, defs);
  return resolved.type !== "array" && resolved.type !== "object";
}

function renderRow(field, data, onChange, defs, onLivePreview) {
  const { name, node, nullable } = field;
  const row = document.createElement("div");
  row.className = "param-row";

  const label = document.createElement("span");
  label.className = "param-label";
  label.textContent = labelFor(name, node);
  if (node.help) label.title = node.help;
  row.appendChild(label);

  const control = document.createElement("div");
  control.className = "param-control";

  const currentValue = data[name] ?? node.default ?? null;

  if (nullable) {
    control.appendChild(renderNullableControl(field, data, onChange, defs, currentValue, onLivePreview));
  } else {
    control.appendChild(
      buildControl(node, currentValue, (value) => onChange(name, value), onLivePreview)
    );
  }

  row.appendChild(control);
  return row;
}

/** A nullable field gets an "auto" pill that swaps between `null` and a live control. */
function renderNullableControl(field, data, onChange, defs, currentValue, onLivePreview) {
  const { name, node } = field;
  const wrap = document.createElement("div");
  wrap.className = "param-control";

  const isAuto = currentValue === null || currentValue === undefined;
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = `param-auto-toggle${isAuto ? " is-auto" : ""}`;
  toggle.textContent = isAuto ? "auto" : "set";

  let inner = null;
  if (!isAuto) {
    inner = buildControl(node, currentValue, (value) => onChange(name, value), onLivePreview);
  }

  toggle.addEventListener("click", () => {
    if (isAuto) {
      onChange(name, defaultForType(node));
    } else {
      onChange(name, null);
    }
  });

  wrap.appendChild(toggle);
  if (inner) wrap.appendChild(inner);
  return wrap;
}

function defaultForType(node) {
  if (node.type === "boolean") return false;
  if (node.type === "integer" || node.type === "number") return node.minimum ?? node.exclusiveMinimum ?? 0;
  return "";
}

/** Build the actual input control for one resolved (non-array/object) schema node. */
function buildControl(node, value, commit, onLivePreview) {
  if (Array.isArray(node.enum)) {
    return buildSelect(node, value, commit, onLivePreview);
  }
  if (node.type === "boolean") {
    return buildSwitch(value, commit, onLivePreview);
  }
  if (node.type === "integer" || node.type === "number") {
    return buildNumericField(node, value, commit, onLivePreview);
  }
  return buildText(value, commit);
}

function buildSelect(node, value, commit, onLivePreview) {
  const select = document.createElement("select");
  select.className = "field-select";
  for (const option of node.enum) {
    const opt = document.createElement("option");
    opt.value = option;
    opt.textContent = option;
    if (option === value) opt.selected = true;
    select.appendChild(opt);
  }
  select.addEventListener("change", () => {
    commit(select.value);
    onLivePreview?.();
  });
  return select;
}

function buildSwitch(value, commit, onLivePreview) {
  const el = document.createElement("div");
  el.className = `switch${value ? " on" : ""}`;
  el.setAttribute("role", "switch");
  el.setAttribute("tabindex", "0");
  el.setAttribute("aria-checked", String(Boolean(value)));
  const knob = document.createElement("div");
  knob.className = "knob";
  el.appendChild(knob);

  const toggle = () => {
    const next = !el.classList.contains("on");
    el.classList.toggle("on", next);
    el.setAttribute("aria-checked", String(next));
    commit(next);
    onLivePreview?.();
  };
  el.addEventListener("click", toggle);
  el.addEventListener("keydown", (e) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      toggle();
    }
  });
  return el;
}

function buildNumericField(node, value, commit, onLivePreview) {
  const el = document.createElement("numeric-field");
  const min = node.minimum ?? (Number.isFinite(node.exclusiveMinimum) ? node.exclusiveMinimum + epsilonFor(node) : Number.NEGATIVE_INFINITY);
  const max = node.maximum ?? (Number.isFinite(node.exclusiveMaximum) ? node.exclusiveMaximum - epsilonFor(node) : Number.POSITIVE_INFINITY);
  el.min = min;
  el.max = max;
  el.step = node.step ?? (node.type === "integer" ? 1 : defaultStep(min, max));
  if (node.fine_step) el.fineStep = node.fine_step;
  el.log = Boolean(node.log);
  el.unit = node.unit ?? "";
  el.value = Number(value ?? node.default ?? min ?? 0);
  el.addEventListener("numeric-change", (e) => {
    const raw = e.detail.value;
    commit(node.type === "integer" ? Math.round(raw) : raw);
  });
  if (onLivePreview) el.addEventListener("numeric-input", onLivePreview);
  return el;
}

function epsilonFor(node) {
  return node.type === "integer" ? 1 : 1e-6;
}

function defaultStep(min, max) {
  if (Number.isFinite(min) && Number.isFinite(max) && max > min) {
    return (max - min) / 200;
  }
  return 1;
}

function buildText(value, commit) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "field-text";
  input.value = value ?? "";
  input.addEventListener("change", () => commit(input.value));
  return input;
}
