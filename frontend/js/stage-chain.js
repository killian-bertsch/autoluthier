// Renders `ProjectConfig.stages` as a reorderable, toggleable list. Each row expands to
// that stage's own params form (rendered via `param-form.js` against `/api/schema/stages`).
//
// Reorder uses the native HTML5 drag-and-drop API — no library, since this is the one
// interaction vanilla `draggable="true"` + `dragstart`/`dragover`/`drop` covers completely
// and pulling in a dependency for it would contradict the "no build step, hand-written ESM"
// decision for something this small.

import { renderParamForm } from "./param-form.js";

const expandedIds = new Set();

/**
 * @param {HTMLElement} container Emptied and filled with the stage list.
 * @param {Array<{id: string, enabled: boolean, params: Record<string, any>}>} stages
 * @param {Record<string, any>} stageSchemas Stage id -> that stage's params JSON Schema.
 * @param {(stages: Array<any>) => void} onChange Called with the full updated array on any edit.
 * @param {(() => void) | undefined} onLivePreview Wired to every stage param's live drag/edit —
 *   see `param-form.js`'s `onLivePreview`.
 */
export function renderStageChain(container, stages, stageSchemas, onChange, onLivePreview) {
  container.innerHTML = "";
  const list = document.createElement("div");
  list.className = "stage-chain";

  stages.forEach((stage, index) => {
    list.appendChild(renderStageRow(stage, index, stages, stageSchemas, onChange, onLivePreview));
  });

  container.appendChild(list);
}

function renderStageRow(stage, index, stages, stageSchemas, onChange, onLivePreview) {
  const row = document.createElement("div");
  row.className = `stage-row${expandedIds.has(rowKey(stage, index)) ? " expanded" : ""}`;
  row.draggable = true;
  row.dataset.index = String(index);

  const header = document.createElement("div");
  header.className = "stage-header";

  const handle = document.createElement("span");
  handle.className = "drag-handle";
  handle.textContent = "⠿";
  header.appendChild(handle);

  const chevron = document.createElement("span");
  chevron.className = "expand-chevron";
  chevron.textContent = expandedIds.has(rowKey(stage, index)) ? "▾" : "▸";
  header.appendChild(chevron);

  const name = document.createElement("span");
  name.className = "stage-name";
  name.textContent = stageLabel(stage.id);
  header.appendChild(name);

  const toggle = document.createElement("div");
  toggle.className = `switch${stage.enabled ? " on" : ""}`;
  toggle.setAttribute("role", "switch");
  toggle.setAttribute("aria-checked", String(stage.enabled));
  const knob = document.createElement("div");
  knob.className = "knob";
  toggle.appendChild(knob);
  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    const next = stages.map((s, i) => (i === index ? { ...s, enabled: !s.enabled } : s));
    onChange(next);
    onLivePreview?.();
  });
  header.appendChild(toggle);

  const toggleExpand = () => {
    const key = rowKey(stage, index);
    if (expandedIds.has(key)) expandedIds.delete(key);
    else expandedIds.add(key);
    renderStageChain(findChainContainer(row), stages, stageSchemas, onChange, onLivePreview);
  };
  chevron.addEventListener("click", toggleExpand);
  name.addEventListener("click", toggleExpand);

  row.appendChild(header);

  if (expandedIds.has(rowKey(stage, index))) {
    row.appendChild(renderStageParams(stage, index, stages, stageSchemas, onChange, onLivePreview));
  }

  attachDragHandlers(row, stages, onChange, onLivePreview);
  return row;
}

function renderStageParams(stage, index, stages, stageSchemas, onChange, onLivePreview) {
  const schema = stageSchemas[stage.id];
  const hasFields = schema && Object.keys(schema.properties ?? {}).length > 0;

  if (!hasFields) {
    const empty = document.createElement("div");
    empty.className = "stage-params-empty";
    empty.textContent = schema ? "No parameters." : "No schema registered for this stage.";
    return empty;
  }

  const paramsDiv = document.createElement("div");
  paramsDiv.className = "stage-params";
  renderParamForm(
    paramsDiv,
    schema,
    {},
    stage.params ?? {},
    (paramName, paramValue) => {
      const nextParams = { ...(stage.params ?? {}), [paramName]: paramValue };
      const next = stages.map((s, i) => (i === index ? { ...s, params: nextParams } : s));
      onChange(next);
    },
    onLivePreview
  );
  return paramsDiv;
}

function attachDragHandlers(row, stages, onChange, onLivePreview) {
  row.addEventListener("dragstart", (e) => {
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", row.dataset.index);
    row.classList.add("dragging");
  });
  row.addEventListener("dragend", () => {
    row.classList.remove("dragging");
    row.parentElement?.querySelectorAll(".stage-row").forEach((r) => r.classList.remove("drag-over"));
  });
  row.addEventListener("dragover", (e) => {
    e.preventDefault();
    row.classList.add("drag-over");
  });
  row.addEventListener("dragleave", () => row.classList.remove("drag-over"));
  row.addEventListener("drop", (e) => {
    e.preventDefault();
    row.classList.remove("drag-over");
    const from = Number(e.dataTransfer.getData("text/plain"));
    const to = Number(row.dataset.index);
    if (from === to || Number.isNaN(from)) return;
    const next = [...stages];
    const [moved] = next.splice(from, 1);
    next.splice(to, 0, moved);
    onChange(next);
    onLivePreview?.();
  });
}

function rowKey(stage, index) {
  return `${index}:${stage.id}`;
}

function stageLabel(id) {
  return id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** The stage list's container element, so a toggle-expand can re-render in place. */
function findChainContainer(rowElement) {
  return rowElement.closest(".stage-chain")?.parentElement ?? rowElement.parentElement;
}
