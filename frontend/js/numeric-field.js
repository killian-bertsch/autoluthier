// <numeric-field>: the draggable numeric control every DSP/config number renders through.
//
// Interactions: drag horizontally to scrub (shift+drag for the fine step), double-click to
// type a value directly, arrow keys to step by one unit while focused, a ghost bar showing
// position within [min, max] when both are finite. A custom element rather than an Alpine
// component because the pointer-capture drag state machine has no natural expression as
// declarative x-directives, and every config screen (this step) plus every stage's params
// (also this step) plus the future per-sample override editor (step 12) all need the same
// widget — one implementation, constructed with plain JS properties, no attribute parsing.
//
// Fires a `numeric-change` CustomEvent (bubbles, detail: { value }) once a drag/edit/arrow-key
// interaction commits. It never fires on every intermediate drag frame.
//
// A second event, `numeric-input` (same detail shape), fires on *every* value change including
// mid-drag frames — for step 12's live preview, which wants to reprocess audio continuously
// while a DSP parameter is being scrubbed, not only once the drag ends. `numeric-change` still
// fires exactly once per interaction, so consumers that persist to config (the normal case)
// don't need to change; only a consumer that opts into `numeric-input` pays for the extra
// events, and it owns its own debouncing (this element doesn't decide that for it).

const PIXELS_PER_STEP = 4;
const LOG_DRAG_SPAN_PX = 300; // px of drag to sweep the full [min, max] range in log mode

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

/** Decimal places to display, inferred from `step` (or the value's own magnitude if no step). */
function decimalsFor(step, value) {
  if (Number.isFinite(step) && step > 0) {
    const text = String(step);
    const dot = text.indexOf(".");
    return dot === -1 ? 0 : text.length - dot - 1;
  }
  const magnitude = Math.abs(value);
  if (magnitude === 0) return 2;
  if (magnitude < 10) return 2;
  if (magnitude < 100) return 1;
  return 0;
}

export class NumericField extends HTMLElement {
  constructor() {
    super();
    this._value = 0;
    this._min = Number.NEGATIVE_INFINITY;
    this._max = Number.POSITIVE_INFINITY;
    this._step = 1;
    this._fineStep = null;
    this._unit = "";
    this._log = false;
    this._decimals = null;
    this._dragging = false;
    this._editing = false;
  }

  connectedCallback() {
    this.innerHTML = `
      <div class="nf-body" tabindex="0">
        <div class="nf-ghost"></div>
        <span class="nf-value tabular-nums"></span>
      </div>
      <span class="nf-unit"></span>
    `;
    this._body = this.querySelector(".nf-body");
    this._ghost = this.querySelector(".nf-ghost");
    this._valueEl = this.querySelector(".nf-value");
    this._unitEl = this.querySelector(".nf-unit");

    this._body.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    this._body.addEventListener("dblclick", () => this._enterEdit());
    this._body.addEventListener("keydown", (e) => this._onKeyDown(e));
    this._render();
  }

  get value() {
    return this._value;
  }

  set value(v) {
    this._value = clamp(v, this._min, this._max);
    this._render();
  }

  get min() {
    return this._min;
  }

  set min(v) {
    this._min = v;
    this._render();
  }

  get max() {
    return this._max;
  }

  set max(v) {
    this._max = v;
    this._render();
  }

  get step() {
    return this._step;
  }

  set step(v) {
    this._step = v;
    this._render();
  }

  get fineStep() {
    return this._fineStep ?? this._step / 10;
  }

  set fineStep(v) {
    this._fineStep = v;
  }

  get unit() {
    return this._unit;
  }

  set unit(v) {
    this._unit = v ?? "";
    this._render();
  }

  get log() {
    return this._log;
  }

  set log(v) {
    this._log = Boolean(v);
  }

  set decimals(v) {
    this._decimals = v;
    this._render();
  }

  _effectiveMin() {
    return Number.isFinite(this._min) ? this._min : this._log ? 1e-3 : this._value - 100;
  }

  _effectiveMax() {
    return Number.isFinite(this._max) ? this._max : this._log ? this._value * 100 : this._value + 100;
  }

  _render() {
    if (!this._body) return;
    const decimals = this._decimals ?? decimalsFor(this._step, this._value);
    this._valueEl.textContent = this._value.toFixed(decimals);
    this._unitEl.textContent = this._unit;

    if (Number.isFinite(this._min) && Number.isFinite(this._max) && this._max > this._min) {
      const fraction = this._log
        ? (Math.log(Math.max(this._value, 1e-9)) - Math.log(Math.max(this._min, 1e-9))) /
          (Math.log(Math.max(this._max, 1e-9)) - Math.log(Math.max(this._min, 1e-9)))
        : (this._value - this._min) / (this._max - this._min);
      this._ghost.style.width = `${clamp(fraction, 0, 1) * 100}%`;
    } else {
      this._ghost.style.width = "0%";
    }
  }

  _fireInput() {
    this.dispatchEvent(new CustomEvent("numeric-input", { bubbles: true, detail: { value: this._value } }));
  }

  _commit(value) {
    this._value = clamp(value, this._min, this._max);
    this._render();
    this._fireInput();
    this.dispatchEvent(new CustomEvent("numeric-change", { bubbles: true, detail: { value: this._value } }));
  }

  _onPointerDown(event) {
    if (this._editing) return;
    event.preventDefault();
    this._body.setPointerCapture(event.pointerId);
    const startX = event.clientX;
    const startValue = this._value;
    let moved = false;

    const onMove = (moveEvent) => {
      const deltaPx = moveEvent.clientX - startX;
      if (!moved && Math.abs(deltaPx) > 2) {
        moved = true;
        this._dragging = true;
        this._body.classList.add("dragging");
      }
      if (!moved) return;
      const fine = moveEvent.shiftKey;
      let next;
      if (this._log) {
        const span = fine ? LOG_DRAG_SPAN_PX * 10 : LOG_DRAG_SPAN_PX;
        const logMin = Math.log(Math.max(this._effectiveMin(), 1e-9));
        const logMax = Math.log(Math.max(this._effectiveMax(), 1e-9));
        const startLog = Math.log(Math.max(startValue, 1e-9));
        next = Math.exp(startLog + (deltaPx / span) * (logMax - logMin));
      } else {
        const unit = fine ? this.fineStep : this._step;
        next = startValue + (deltaPx / PIXELS_PER_STEP) * unit;
      }
      this._value = clamp(next, this._min, this._max);
      this._render();
      this._fireInput();
    };

    const onUp = () => {
      this._body.removeEventListener("pointermove", onMove);
      this._body.removeEventListener("pointerup", onUp);
      this._body.classList.remove("dragging");
      this._dragging = false;
      if (moved) {
        this._commit(this._value);
      }
    };

    this._body.addEventListener("pointermove", onMove);
    this._body.addEventListener("pointerup", onUp);
  }

  _onKeyDown(event) {
    if (this._editing) return;
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
    event.preventDefault();
    const unit = event.shiftKey ? this.fineStep : this._step;
    const direction = event.key === "ArrowUp" ? 1 : -1;
    this._commit(this._value + unit * direction);
  }

  _enterEdit() {
    this._editing = true;
    this._body.classList.add("editing");
    const input = document.createElement("input");
    input.className = "nf-input";
    input.type = "text";
    input.inputMode = "decimal";
    input.value = String(this._value);
    this._valueEl.replaceWith(input);
    this._inputEl = input;
    input.focus();
    input.select();

    const finish = (commit) => {
      const parsed = Number.parseFloat(input.value);
      input.replaceWith(this._valueEl);
      this._body.classList.remove("editing");
      this._editing = false;
      this._inputEl = null;
      if (commit && Number.isFinite(parsed)) {
        this._commit(parsed);
      } else {
        this._render();
      }
    };

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") finish(true);
      if (e.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(true));
  }
}

customElements.define("numeric-field", NumericField);
