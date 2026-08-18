# Auto-sampler_CLI_2 — Rewrite Plan

## Context

`Auto-sampler_CLI` (V1) works but has structural problems that block growth: every config field must be
edited in **four** places, DSP stages are hardcoded in a fixed order, everything is `float64` in RAM
(~2–4 GB for an 18-layer instrument), the transient shaper and loop finder are pure-Python per-sample
loops, and there is no interactive surface — you hand-edit a TOML, run a batch, and hope. A full audit
of V1 also turned up 12 real bugs, several of which the current tests actively pin in place.

V2 is a **clean-break rewrite** in an adjacent, fully independent project: its own git repo, its own
`uv`-managed venv, no shared code or config with V1. It keeps V1's proven audio math but rebuilds the
structure around a single Pydantic-v2 schema that drives TOML parsing, validation, CLI flags, *and*
the UI form — so a field is defined once. On top of that sits a local web frontend for waveform and
loop-point editing, batch inspection, and live preview.

**Outcome:** the same SFZ instruments, produced faster and in a fraction of the memory, from an
interface where you can see and hear what the pipeline is doing before committing a full render.

---

## Decisions (all confirmed)

| Area | Decision |
|---|---|
| Package / command | **`autosampler`** |
| Export format | **SFZ only**, improved opcode model |
| Slicing | **Pure arithmetic**, as V1. No onset detection |
| New DSP | **EQ/filter**, **stereo width**, **baked loop crossfade**, **true-peak limiter** |
| Stage order | Shaper **before** normalize (fixes V1's defeated ceiling); chain **reorderable in UI** |
| Saturation | Shaper's `tanh` becomes an **optional** control, not always-on |
| V1 compat | **Clean break** — no importer |
| Helpers ported | MIDI session generator, source pre-normalizer, layer concatenator (Python, not bash) |
| Output audio | **Configurable** format (FLAC/WAV), depth (16/24/32/float), optional resample. Default 24-bit FLAC |
| Precision | **float32**, fully vectorized, parallel across cores |
| Project file | `project.toml` **inside the instrument folder** + app-level workspace index |
| Overrides | **Full per-sample overrides, persisted**, always beating auto-detection |
| Preview | **Live preview on a subset**, same code path as the full render |
| Rigor | **Strict**: ruff + `mypy --strict`, Pydantic v2, real DSP tests |
| Frontend | **Local web app** in browser |
| Views | Waveform+loop editor, sample matrix, analysis+metering, SFZ preview+job log |
| CLI model | Both: `run ./keybass` and `run --scan ./instruments` |
| UI launch | `autosampler ui` serves + auto-opens browser; `--dev` adds asset live-reload |
| Aesthetic | **Studio dark / Bitwig-like** — near-black low-chroma surfaces, hairline borders, single amber accent |
| Git | **Local only, no remote.** V1's repo and remote left untouched |

### Environment constraints that shaped this
- **No Linux Node** (`npm` is Windows npm via WSL interop) → **no bundler, no build step**. Frontend is
  hand-written ESM + vendored library files committed to the repo.
- `gh` not installed → no remote is created. `uv 0.12.3`, `ffmpeg 6.1.1`, Python 3.12.3 available.
- WSL2 → the page opens in the Windows browser. This is also *why* browser audio beats a TUI: loop-point
  editing needs pixel precision and real playback, which a terminal cannot deliver.

---

## Stack

**Backend:** Python 3.12 · `uv` · Pydantic v2 (config + JSON Schema) · NumPy · SciPy · soundfile ·
pyloudnorm · Typer (CLI) · FastAPI + uvicorn · pytest + ruff + mypy.

**Frontend (no build step):** hand-written ESM · Alpine.js ~10 KB (chrome/reactive state only) ·
vendored **wavesurfer.js 7.12.11** (zero prod deps, BSD-3) + RegionsPlugin for draggable loop markers ·
SSE for job progress · plain CSS custom properties for tokens · JetBrains Mono with `tabular-nums`
for every numeric readout.

**Two frontend specifics that matter:**
- wavesurfer decodes the *whole* file in the browser — far too slow here. **Peaks are precomputed
  server-side** in NumPy at a few zoom levels and handed to wavesurfer's `peaks` option.
- Neither wavesurfer's nor peaks.js's region-loop playback is sample-accurate. Genuine loop-click
  verification uses a **hand-wired Web Audio `AudioBufferSourceNode`** with `loop`/`loopStart`/`loopEnd`.

**Design language:** compact parameter rows (`dimmed label ····· mono value unit [↺]`, ~30 px) grouped
under small-caps headers, per Ableton Live device panels / Bitwig inspector. **Draggable numeric fields**
(drag to scrub, shift+drag fine, double-click to type, arrows to step, ghost bar showing position in
range) rather than knobs or sliders. Loop-marker conventions from Ableton Sampler / Kontakt / Decent
Sampler: shaded loop region, distinct start/end handles, a zoomed **loop-seam inset** showing the join,
and a snap-to-zero-crossing toggle.

---

## Audio math preserved verbatim from V1

Behavioral contracts — the rewrite must reproduce these exactly.

- **Velocity list:** `x==1 → [127]`; `x==2 → [1,127]`; else `v_0=1`, `v_{x-1}=127`, `v_i = round(1 + i*126/(x-1))`.
- **Slicing:** `event_start = round(event_index * (hold+release) * sr)`, note-outer/velocity-inner;
  sustain `[0, frames_hold)`, release `[frames_hold, frames_hold+frames_release)`; `event_index`
  advances even for notes outside `[min_note, max_note]`.
- **Key zones:** `lokey_i = min_note` if first else `(notes[i-1]+notes[i])//2 + 1`;
  `hikey_i = max_note` if last else `(notes[i]+notes[i+1])//2`.
- **Velocity zones:** `lo_nat = 1` if first else `(v[i-1]+v_c)//2 + 1`; `hi_nat = 127` if last else
  `(v_c+v[i+1])//2`; `xfade = int(pct/100 * (hi_nat-lo_nat+1))`; zones extend outward by `xfade`.
- **`amp_velcurve_1 = 10**(-range_db/20)`**.
- **Loop score** = `0.65*amp_similarity + 0.35*max(correlation, 0)` over upward zero crossings in
  85%–97% of the sample; 10 ms RMS window, 20 ms correlation window, 50 ms min gap, top-10 × top-10.
- **`rt_decay`:** 20 ms non-overlapping RMS windows, linear fit over the first 80%, clamp `[1, 24]` dB/s,
  default `6.0` below 2 windows.
- **Subset tie-breaks:** `select_notes` target 1 → `notes[n//2]`; `select_velocities` k=1 → highest.
- **Output layout:** `{out}/{name}/{name}_sustain.sfz`, `{name}_release.sfz`,
  `samples/{name}_{NoteName}_v{vel}.flac`, `samples/release/{name}_{NoteName}_v{vel}_rel.flac`;
  `name` = instrument folder name, `NoteName` uses `C#`-style sharps (60 → `C4`).

## V1 bugs fixed in V2

1. Velocity-mode dynamic range averaged `0.0` in for single-layer/silent notes, dragging it down.
2. Velocity-mode release normalization hardcoded `-18.0` dB, ignoring config.
3. Peak ceiling applied per-velocity-group in lufs/rms but globally in velocity mode.
4. `bool` is an `int` subclass, so TOML `true` silently became `1` for int fields.
5. `pre_trim_ms` longer than the sample → empty array → `mean()` of empty → NaN written into the FLAC.
6. Truncated-render fallback emitted a **mono** zeros array even for stereo instruments.
7. `loop_crossfade_ms` > 15% of sample length silently disabled looping for every sample.
8. Set-based index dedup in `select_notes`/`select_velocities` could silently return fewer entries.
9. Unknown config keys silently ignored → a typo fell back to the default with no warning.
10. Shaper ran *after* the peak ceiling, so `peak_ceiling_db` was not the final peak.
11. `instrument_name` was documented and parseable but unconditionally discarded.
12. `convert_samples.sh`'s `STEREO_INSTRUMENTS=("ag", "wg", "pearl")` kept the commas, so `ag`/`wg`
    never matched and were silently downmixed to mono.

---

## Target layout

```
Auto-sampler_CLI_2/
├── pyproject.toml uv.lock .gitignore README.md CLAUDE.md   # CLAUDE.md holds the ledger
├── .venv/                                                   # independent, uv-managed
├── src/autosampler/
│   ├── config/      schema.py  hints.py  toml_io.py  workspace.py
│   ├── domain/      notes.py  units.py  models.py           # single source of truth for note/vel math
│   ├── io/          reader.py  writer.py  slicer.py  peaks.py
│   ├── dsp/         base.py  registry.py  dc.py  trim.py  eq.py  stereo.py
│   │                transient.py  normalize.py  limiter.py  loop.py
│   ├── pipeline/    graph.py  executor.py  events.py  preview.py
│   ├── export/      zones.py  sfz.py  writer.py  reports.py
│   ├── analysis/    metrics.py  matrix.py
│   ├── helpers/     midi_session.py  prenormalize.py  concat_layers.py
│   ├── cli/         app.py  (+ one module per command)
│   └── server/      app.py  routes/  sse.py  static.py
├── frontend/        index.html  css/  js/  vendor/          # no build step
└── tests/           unit/  dsp/  export/  api/  e2e/  golden/
```

**Core rule:** the CLI and the server are both *thin clients* over `pipeline` + `export`. No domain
logic in either. This is what makes the UI cheap and keeps preview and full render on one code path.

**Config mechanism (the backbone):** Pydantic v2 models with `model_config = ConfigDict(extra="forbid",
strict=True)`. Per-field `json_schema_extra` carries UI hints — `unit`, `step`, `fine_step`, `log`,
`group`, `order`, `help`. `model_json_schema()` is served to the frontend, which renders the parameter
rows from it. Add a field once and it appears in TOML, validation, CLI, and UI simultaneously.

**Stage abstraction:** each stage is `(id, ParamsModel, apply(buf, params, ctx))` in a registry. The
chain is a list of `{id, enabled, params}` in config, so it is reorderable and toggleable from the UI
with no code change.

---

## Development ledger

`CLAUDE.md` in the V2 repo carries the ledger — the single source of truth across cleared contexts.
Step 0 creates it; **every later step updates its own row as the last action before committing.**

```markdown
## Development Ledger
Status: todo | in-progress | done | blocked

| # | Step | Status | Commit | Notes / gotchas for the next context |
|---|------|--------|--------|--------------------------------------|
| 0 | Scaffold, tooling, ledger | done | abc1234 | uv venv at .venv, py3.12 |
| 1 | Config schema + TOML IO   | todo | —       | |

### Current state
One paragraph: what works, what is stubbed, what the next step does first.

### Conventions established
Naming, error handling, test layout, DSP buffer contract — so later steps stay consistent.
```

**Per-step contract:** read `CLAUDE.md` first → do the work → `ruff check`, `mypy --strict`, `pytest`
all green → update ledger row + `Current state` + any new conventions → `git commit` naming the step.

---

## Steps

Each is one context. Every step ends green and committed.

**0 · Scaffold** — `mkdir Auto-sampler_CLI_2`, `git init` (no remote), `uv init` + `uv venv`,
`pyproject.toml` (src layout, deps, ruff/mypy-strict/pytest config), `.gitignore` (`.venv/`,
`__pycache__/`, `*.pyc` — V1's mistake not repeated), `CLAUDE.md` with the ledger skeleton, README.
*Verify:* `uv run pytest` collects, `ruff`/`mypy` run clean on an empty package.

**1 · Config schema + TOML IO** — `config/schema.py` (Recording, Selection, Crossfade, Output,
StageChain, ProjectConfig), `config/hints.py`, `config/toml_io.py`, `config/workspace.py`.
`extra="forbid"` and strict types fix bugs **4** and **9**. JSON Schema export.
*Verify:* round-trip a project.toml; unknown key raises; `true` for an int field raises.

**2 · Domain + slicer + audio IO** — `domain/notes.py` (velocity list, note grid, midi↔name — the one
place, no duplication), `domain/units.py`, `domain/models.py` (float32 `Sample`, `SampleSet` with
`(note,velocity)` dict index for O(1) lookup), `io/reader.py`, `io/writer.py` (FLAC/WAV, bit depth,
resample), `io/slicer.py`. Stereo-correct zero fill fixes bug **6**.
*Verify:* exact velocity lists for x=1,2,4,18; slice offsets against a synthesized marker render;
truncated input pads with correct channel count.

**3 · DSP framework + core stages** — `dsp/base.py` (Stage protocol), `dsp/registry.py`, `dc.py`,
`trim.py` (clamped → fixes **5**), `normalize.py` (lufs/rms/velocity, consistent ceiling, cached
`pyln.Meter` per sample rate). Fixes **1**, **2**, **3**. Vectorized float32 throughout.
*Verify:* analytic signals — known-RMS sine normalizes to target; DC offset removed to < 1e-6;
velocity mode's range on a synthetic ramp matches the closed form.

**4 · New DSP stages** — `eq.py` (scipy biquads: HP/LP/low-shelf/high-shelf/peaking, `sosfilt`),
`stereo.py` (width / mid-side), `transient.py` (**vectorized** dual envelope, saturation now optional),
`limiter.py` (lookahead true-peak). The recursive peak-hold `env[i]=max(amp[i], c*env[i-1])` does not
vectorize directly — implement via a decay-normalized cumulative maximum
(`env = c**i * maximum.accumulate(amp * c**-i)`) with block-wise rescaling to avoid overflow, and
assert equivalence against the naive loop in tests.
*Verify:* filter magnitude response at known frequencies vs `scipy.signal.sosfreqz`; vectorized
envelope matches the reference loop to 1e-6; limiter never exceeds ceiling on a clipping-hostile signal.

**5 · Loop detection + baked crossfade** — `dsp/loop.py`: vectorized zero crossings
(`np.flatnonzero((x[:-1]<0)&(x[1:]>=0))+1`), the 0.65/0.35 scoring, a guard so a large
`loop_crossfade_ms` cannot collapse the search window (fixes **7**), and the baked equal-power crossfade.
*Verify:* detected points land on upward zero crossings; a pure sine loops with no discontinuity
(measure the seam sample delta); crossfade at 50% of sample length still yields a valid loop.

**6 · Pipeline, events, parallelism** — `pipeline/graph.py` (build ordered chain from config),
`events.py` (structured progress), `executor.py` (`ProcessPoolExecutor` for per-sample CPU work since
NumPy only partly releases the GIL; threads for IO), `preview.py` (subset selection reusing the same
graph). Per-sample overrides applied here so they always beat auto-detection.
*Verify:* preview on 4 samples byte-identical to the same 4 from a full run; progress events monotonic;
parallel result identical to serial.

**7 · SFZ export** — `export/zones.py` (key/velocity zones with **order-preserving** dedup → fixes
**8**), `export/sfz.py` (opcode model → text), `export/writer.py` (file layout), `reports.py`.
*Verify:* golden-file SFZ comparison against V1 output for the checked-in `keybass` config, proving the
zone math and opcodes are unchanged; zones contiguous and gapless over `[min_note, max_note]`.

**8 · CLI + ported helpers** — Typer app: `init`, `run [--scan]`, `validate`, `preview`, `ui`, plus
`midi` (session generator), `prenorm`, `concat` (fixes **12**). Rich progress bound to the event bus.
*Verify:* end-to-end run on a synthesized instrument folder produces a loadable SFZ + FLACs; `--scan`
handles a mixed dir; malformed project fails with a clear message and non-zero exit.

**9 · Server** — FastAPI: `/api/schema`, `/api/project`, `/api/samples`, `/api/peaks/{id}`,
`/api/matrix`, `/api/analysis`, `/api/sfz/preview`, `/api/preview`, `/api/jobs`, `/api/audio/{id}`,
and `/events` (SSE). `io/peaks.py` precomputes min/max arrays at 3 zoom levels. Static mount + `--dev`
reload; `ui` picks a free port and opens the browser.
*Verify:* schema endpoint matches the Pydantic models; peaks payload sizes sane; SSE streams a real job;
audio route returns playable FLAC.

**10 · Frontend shell + schema-driven forms** — tokens (studio dark, amber accent, hairline borders),
app shell + nav, the **draggable numeric field** component, parameter rows auto-rendered from the JSON
Schema + hints, stage chain list with enable toggles and drag reorder. Vendor Alpine + wavesurfer.
Alpine owns chrome and forms only; the heavy views are vanilla ES modules — Alpine is not an SPA
framework and 576 samples plus a waveform editor is past its comfort zone.
*Verify:* every config field renders with correct unit/step/range; edits round-trip to `project.toml`;
looks right at 1280 and 1920 wide.

**11 · The four views** — waveform + loop editor (wavesurfer + Regions, loop-seam inset,
zero-crossing snap, `AudioBufferSourceNode` loop audition, A/B dry vs processed); sample matrix
(**canvas**, not DOM — 576+ cells, colored by peak/LUFS/length, click to open); analysis (hand-rolled
SVG charts, no charting library); SFZ preview + diff + job log over SSE.
*Verify:* drag a loop marker → value persists to `project.toml` → survives reload; matrix flags a
deliberately silenced slice; loop audition is audibly seamless on a sine.

**12 · Live preview, overrides, polish** — debounced + cancellable preview while dragging a control
(`AbortController`, trailing-edge debounce ~150 ms, one in-flight request), per-sample override
persistence and reset-to-auto, keyboard shortcuts, empty/error states, README + docs, final
`ruff`/`mypy`/`pytest` sweep and ledger close-out.
*Verify:* full manual pass on the real `keybass` instrument; dragging a numeric field never queues more
than one request; overrides survive a full re-run.

---

## Verification

**Per step:** `uv run ruff check .` · `uv run mypy --strict src` · `uv run pytest -q` — all green
before the commit.

**Cross-cutting, once step 7 lands:** a golden-file test renders the V1 `keybass` settings through V2
and diffs the SFZ against V1's output. This is the main guard that the rewrite did not change the audio
math, and it must stay green through steps 8–12.

**End-to-end, once step 11 lands:** point V2 at a real instrument folder, `autosampler ui`, then confirm
in the browser: the matrix shows all slices with no anomalies, a waveform loads and its loop auditions
seamlessly, dragging a loop marker persists, live preview responds while dragging a parameter, the SFZ
preview matches what `run` writes, and the written instrument loads in a sampler.

**Performance check:** the 18-layer `keybass` set should complete in materially less wall time and
well under half V1's peak RSS. Measure both on the first full run in step 8 and record the numbers in
the ledger.

## Risks

- **Vectorized peak-hold envelope** (step 4) is the only genuinely tricky numerics. Mitigated by
  keeping the naive loop as the test reference; if the cumulative-max form proves numerically fragile,
  fall back to a Cython-free `scipy.signal.lfilter` approximation or chunked loop — the tradeoff is
  speed, not correctness.
- **No-build frontend** means hand-writing what a framework would give free. Mitigated by keeping
  Alpine for forms and reserving vanilla modules for the two heavy views.
- **Golden-file comparison against V1** may reveal V1 behavior that is a bug rather than a contract.
  Where those collide, the fixed-bugs list above wins and the golden file is regenerated with a note.
- **576-sample matrix + live preview** is the most likely performance cliff in the UI; canvas rendering
  and single-in-flight preview requests are the mitigations, chosen up front.
