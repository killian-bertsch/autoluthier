# Autoluthier

Autoluthier turns a single long recording of an instrument into a fully-processed,
velocity-layered SFZ instrument. You render a generated MIDI "autosampler" session into your
instrument (hardware, plugin, whatever can receive MIDI and print audio) as one continuous
WAV/FLAC file, and Autosampler slices that render back into individual per-(note, velocity)
samples, runs them through a configurable DSP chain (trim, transient shaping, loudness
normalization, EQ, stereo width, limiting, loop detection), and writes out a ready-to-load
`.sfz` instrument plus its audio files.

It ships as both a scriptable CLI and a local single-page web app with waveform editing, a
sample matrix, and live preview — both are thin clients over the same processing pipeline, so
anything you can do in the browser you can also do (or automate) from the command line.

## Table of contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [End-to-end workflow](#end-to-end-workflow)
- [Project layout](#project-layout)
- [`project.toml` reference](#projecttoml-reference)
- [CLI reference](#cli-reference)
- [The DSP stage chain](#the-dsp-stage-chain)
- [SFZ export](#sfz-export)
- [The web UI](#the-web-ui)
- [HTTP API](#http-api)
- [Source layout](#source-layout)
- [Development](#development)

## How it works

1. **Record** — a generated MIDI file steps through every `(note, velocity)` combination you
   want, holding each note for a fixed time and leaving a fixed silence after note-off. You
   render this into your instrument to get one long `sustain.wav` (and, optionally, a second
   pass recording note-off release tails into `release.wav`).
2. **Slice** — because the recording is driven by exact, known timing, Autosampler locates each
   sample by pure arithmetic (`event_index * (hold_time + release_time) * sample_rate`) — no
   onset detection, no drift.
3. **Process** — every sliced sample runs through an ordered, user-configurable chain of DSP
   stages (DC removal, trim, transient shaping, loudness normalization, EQ, stereo width,
   limiting, loop detection), executed in parallel across CPU cores.
4. **Export** — samples are grouped into contiguous key/velocity zones, encoded to the
   configured container/bit depth, and written next to one or two `.sfz` files (sustain, and
   release if present) that describe them.

Everything is driven by a single Pydantic v2 schema (`ProjectConfig`, stored as `project.toml`)
that also generates the web UI's config forms and validation — a new field defined once appears
in TOML, the CLI, and the browser with no other code changes.

## Installation

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -e ".[dev]"
```

This installs the `autosampler` console script into the project's virtualenv (`uv run
autosampler ...`, or activate the venv directly).

## End-to-end workflow

```
autosampler midi   →  record in your DAW/sampler  →  autosampler run
     │                                                     ▲
     └─ generates .mid + a matching project.toml           │
                                                            │
        (optional) autosampler prenorm / concat ───────────┘
        (optional) autosampler validate / preview  (sanity-check before a full run)
```

1. **Generate a recording session:**

   ```bash
   autosampler midi -X 5 -N 3 -H 2.0 -R 1.0 -o piano --start-note 21 --end-note 108
   ```

   Writes `piano.mid` (one note-on/off event per note/velocity pair, at a fixed 120 BPM — only
   relative timing matters) and a `project.toml` in `./piano/` that exactly matches the MIDI
   file's timing. The recording parameters can never drift out of sync with the audio because
   both come from the same call.

2. **Render it.** Load `piano.mid` into your DAW, plugin host, or hardware sampler and print the
   output as one continuous file. Save it as `sustain.wav` (or `.flac`) directly inside the
   `piano/` folder created above, next to `project.toml`. Optionally do a second pass capturing
   note-off release tails and save it as `release.wav`/`.flac` in the same folder.

   If you recorded separate takes per velocity layer instead of one continuous MIDI-driven
   pass, use `autosampler concat` to stitch and normalize them into the same `sustain`/`release`
   layout, then `autosampler init` to attach a `project.toml` describing that recording.

3. **(Optional) Normalize raw renders** across many instruments before slicing:

   ```bash
   autosampler prenorm ./recordings --target-db -6.0
   ```

4. **(Optional) Sanity-check** the config and preview a few samples before committing to a full
   render:

   ```bash
   autosampler validate ./piano
   autosampler preview ./piano
   ```

5. **Process it:**

   ```bash
   autosampler run ./piano -o ./output
   ```

   Slices, processes, and exports `./output/piano/` containing the SFZ file(s) and audio
   samples. Use `--scan` to batch-process every instrument folder under a parent directory in
   one call.

Alternatively, run `autosampler ui` at any point after step 2 and do the same work — plus
interactive loop editing, live preview, and analysis views — in a browser.

## Project layout

An instrument is a single folder. Autosampler expects, at minimum:

```
piano/
├── project.toml     # required — recording settings, DSP chain, output config
├── sustain.wav       # required — the continuous multi-note/velocity render
└── release.wav       # optional — a second pass of note-off release tails
```

`autosampler run`/`--scan` writes its output to a separate directory (default `./output/`),
one subfolder per instrument:

```
output/piano/
├── piano_sustain.sfz
├── piano_release.sfz        # only if release samples exist
├── velocity_range.txt       # only when velocity-mode normalize measured a dynamic range
└── samples/
    ├── piano_C4_v127.flac
    ├── ...
    └── release/
        └── piano_C4_v127_rel.flac
```

`autosampler preview` writes its (smaller, auditioning-only) output to `<folder>/preview/` by
default.

## `project.toml` reference

Generated for you by `autosampler midi`/`init`, and editable by hand or through the web UI's
config panels. Every field below is validated with `extra="forbid"` — an unrecognized or
misspelled key is a hard error, not a silently-ignored default.

```toml
instrument_name = "piano"        # optional; defaults to the folder name

[recording]                      # must exactly match the MIDI session used to render sustain.wav
velocity_layers = 5
semitone_interval = 3
hold_time = 2.0
release_time = 1.0
start_note = 21
end_note = 108
# release_hold_time / release_release_time override hold_time/release_time for release.wav

[selection]                      # thins the recorded set down to what the SFZ ships
min_note = 21
max_note = 108
note_percentage = 100.0          # % of in-range notes to keep, evenly spaced
velocity_layers_out = 5          # <= recording.velocity_layers
# velocity_map = "0-63:1, 64-127:5"   # optional per-range layer counts, overrides velocity_layers_out

[crossfade]
crossfade_percent = 0.0          # neighboring-zone overlap (xfin/xfout)
loop_crossfade_ms = 20.0         # baked crossfade length at each loop point
loop_crossfade_mode = "baked"    # "baked" renders the fade into the audio; "sfz" emits a loop_crossfade opcode instead
loop_crossfade_shape = "linear"  # "linear" or "equal_power"

[output]
container = "flac"               # "flac" or "wav"
sample_format = "pcm24"          # "pcm16" | "pcm24" | "pcm32" | "float32" (FLAC only supports pcm16/pcm24)
sample_rate = 44100               # omit to keep the source rate
collapse_to_mono = false
ampeg_release = 0.5
velocity_dynamic_range_db = 40.0  # fallback amp_velcurve_1 range when normalize isn't in velocity mode

[[stages]]                       # ordered DSP chain — see below
id = "dc"
enabled = true

[[overrides]]                    # per-sample manual loop-point decisions; always beat auto-detection
note = 60
velocity = 127
loop_start = 88200
loop_end = 132300
```

## CLI reference

Every command is a thin wrapper over the same `pipeline`/`export`/`helpers` functions the web
UI calls — nothing here makes a processing decision the UI can't also make.

| Command | Purpose |
|---|---|
| `autosampler init <dir>` | Write a `project.toml` for an already-recorded instrument. |
| `autosampler run <dir>` | Slice, process, and export one instrument (or, with `--scan`, every instrument under a parent folder). |
| `autosampler validate <dir>` | Load and validate `project.toml` + build the DSP chain without touching audio. |
| `autosampler preview <dir>` | Process an evenly-spread subset of notes/velocities for a quick listen. |
| `autosampler midi` | Generate an autosampler MIDI session + a matching `project.toml`. |
| `autosampler prenorm <dir>` | Peak-normalize raw `sustain`/`release` renders across many instrument folders, in place. |
| `autosampler concat <dir> [out]` | Concatenate numbered per-layer raw takes into `sustain`/`release` files. |
| `autosampler ui` | Launch the local web UI (FastAPI + browser frontend). |

<details>
<summary><strong>autosampler init</strong> — scaffold a project.toml for existing audio</summary>

```
autosampler init <folder> -X <velocity_layers> -N <semitone_interval> -H <hold_time> -R <release_time>
    [--start-note 21] [--end-note 108] [--name INSTRUMENT] [--force]
```

Writes `project.toml` into `folder` describing recording settings you provide (matching audio
you've already placed there, e.g. via `concat`). Refuses to overwrite an existing
`project.toml` unless `--force`.

</details>

<details>
<summary><strong>autosampler run</strong> — slice, process, export</summary>

```
autosampler run <target> [--output-dir/-o output] [--scan] [--instrument NAME] [--workers N]
```

- Without `--scan`: `target` is one instrument folder.
- With `--scan`: `target` is a parent directory; every immediate subfolder containing a
  `project.toml` is processed. `--instrument` restricts this to one named subfolder.
- `--workers` sets the number of worker processes for parallel per-sample DSP (default: one per
  CPU).

Shows a live Rich progress bar per instrument. A failing instrument is reported and skipped
rather than aborting the batch; the command exits non-zero only if every instrument failed.

</details>

<details>
<summary><strong>autosampler validate</strong> — fast sanity check</summary>

```
autosampler validate <folder>
```

Loads `project.toml`, validates it against the schema, and builds the DSP stage chain — the
same failure points a real `run` would hit early (bad TOML, unknown/mistyped fields, invalid or
duplicate stages) — without loading or processing any audio. Prints `OK` and the stage count on
success.

</details>

<details>
<summary><strong>autosampler preview</strong> — audition the chain on a subset</summary>

```
autosampler preview <folder> [--notes N] [--velocities N] [--mode exact|subset] [--out dir] [--workers N]
```

Picks an evenly-spread grid of `(note, velocity)` samples, runs just those through the
configured chain, and writes them to `<folder>/preview/` as `<NoteName>_v<velocity>.<ext>`
(release files get a `_rel` suffix). `exact` mode is bit-identical to a full run through the
barrier stage (normalize) before subsetting; `subset` mode runs only the subset through
everything, which is cheap but gives approximate normalize statistics.

</details>

<details>
<summary><strong>autosampler midi</strong> — generate a recording session</summary>

```
autosampler midi -X <velocity_layers> -N <semitone_interval> -H <hold_time> -R <release_time> \
    -o <output_stem> [--start-note 21] [--end-note 108] [--out-dir ./<output_stem>/]
```

Writes `<output_stem>.mid` (systematic note-on/off events at 120 BPM, stepping notes by
`semitone_interval` and velocities across `velocity_layers`) and a `project.toml` in
`--out-dir` describing that exact layout. Prints a summary and reminds you of the next step:
render the `.mid`, save the result as `sustain.wav` next to `project.toml`, then `autosampler
run`.

</details>

<details>
<summary><strong>autosampler prenorm</strong> — batch peak-normalize raw renders</summary>

```
autosampler prenorm <source_dir> [--target-db -6.0] [--dry-run]
```

For every immediate subfolder of `source_dir`, finds its `sustain`/`release` render, measures
its true peak, and rewrites it in place (streamed, preserving format/bit depth) so its peak
sits at `--target-db`. `--dry-run` reports the gain that would be applied without writing.
Silent files are skipped.

</details>

<details>
<summary><strong>autosampler concat</strong> — stitch separately-recorded velocity layers</summary>

```
autosampler concat <input_dir> [output_dir] [--stereo NAME ...] [--target-db -6.0] [--layers 1,2,3]
```

Expects raw takes named `{n} <InstrumentName>.wav|flac` per sustain layer and `R
<InstrumentName>.wav|flac` for a release tail. Concatenates layers in the order given by
`--layers`, downmixes to mono unless the instrument name is passed to `--stereo`,
peak-normalizes sustain and release independently to `--target-db`, and writes
`<output_dir>/<InstrumentName>/sustain.flac` (+ `release.flac`) as 24-bit FLAC — ready for
`autosampler init`.

</details>

<details>
<summary><strong>autosampler ui</strong> — launch the web app</summary>

```
autosampler ui [--host 127.0.0.1] [--port 8000] [--no-open]
```

Starts the FastAPI/uvicorn server and opens it in your default browser (`--no-open` to
suppress). If no built frontend is found it opens `/docs` (FastAPI's auto-generated API docs)
instead — the `/api` and `/events` endpoints work regardless.

</details>

## The DSP stage chain

`project.toml`'s `[[stages]]` list is an ordered, reorderable, individually-toggleable chain.
The default chain (a fresh `init`/`midi` project) is:

```
dc → trim → transient (disabled) → normalize → eq (disabled) → stereo (disabled) → limiter (disabled) → loop
```

`eq`, `stereo`, and `limiter` are new in this version and default to disabled so a fresh
project reproduces the original tool's output. `normalize` and `loop` are not per-buffer
stages — they see the whole sample set at once and may each appear at most once in the chain.

| Stage | What it does | Key parameters |
|---|---|---|
| **dc** | Removes DC offset (per-channel mean subtraction, computed in float64). | — |
| **trim** | Cuts a fixed duration off the start of every sample. | `pre_trim_ms` (default 0) |
| **transient** | SPL-style attack/sustain split — independent gain on the transient vs. body of a sound, sustain-only. | `attack_db` (±12, default 6), `sustain_db` (±12, default 0), `speed_ms` (1–50, default 10), `saturate` (optional tanh soft-clip) |
| **normalize** | Loudness-matches samples across velocity layers or notes; always applies a peak ceiling afterward. | `mode` (`lufs` / `rms` / `velocity`), `target_db` (default −18), `peak_ceiling_db` (default −1, attenuation only) |
| **eq** | Single biquad filter section (Audio EQ Cookbook formulas). | `filter_type` (lowpass/highpass/low_shelf/high_shelf/peaking), `freq_hz`, `q`, `gain_db` |
| **stereo** | Mid-side width control. | `width` (0 = mono, 1 = unchanged, 2 = max wide) |
| **limiter** | Lookahead true-peak limiter (4x oversampled). | `ceiling_db` (default −0.3), `lookahead_ms` (default 5), `release_ms` (default 50) |
| **loop** | Detects a sustain loop region and manages its crossfade seam. | `region_start_frac` (default 0.85), `region_end_frac` (default 0.97), `min_loop_ms` (default 50) — crossfade length/mode/shape live in `[crossfade]`, not here |

**Normalize modes:**
- `lufs`/`rms` — groups samples by velocity layer and matches each group's measured level
  (integrated LUFS via `pyloudnorm`, with an RMS fallback, or plain RMS) to `target_db`.
- `velocity` — groups by note and fits a linear gain curve across that note's velocity layers,
  so peak level scales smoothly from the lowest to the highest recorded velocity. The measured
  spread becomes a per-note `dynamic_range_db`, which is what ends up in the exported SFZ's
  `amp_velcurve_1` opcode (falling back to `output.velocity_dynamic_range_db` when normalize
  isn't in velocity mode).

**Loop detection:** downmixes to mono, finds upward zero-crossings inside the
`region_start_frac`–`region_end_frac` window of the sample, and scores candidate start/end
pairs by a blend of local amplitude similarity and windowed correlation. The best-scoring pair
meeting `min_loop_ms` becomes the loop; `crossfade.loop_crossfade_ms` of audio at the seam is
either faded directly into the sample (`baked` mode — sounds identical in every sampler) or left
untouched with a `loop_crossfade` opcode emitted instead (`sfz` mode — lets the sampler do the
fade at playback, but only works if it implements the opcode). Per-sample manual overrides
(`[[overrides]]` in `project.toml`, or edited live in the web UI's Waveform view) replace
detection entirely for that sample and always win.

Execution is parallelized across CPU-bound worker processes wherever the chain has no
cross-sample dependency; `normalize` is the one synchronization point ("barrier") the chain
waits on. Results are always bit-identical to a fully serial run — parallelism only changes
wall-clock time.

## SFZ export

Samples are grouped into contiguous key zones (boundaries at the midpoint between adjacent
selected notes) and velocity zones (same idea, plus optional inward-growing crossfade regions
sized by `crossfade.crossfade_percent`). Two SFZ files are written per instrument:

- **`<name>_sustain.sfz`** — one `<region>` per sample with `lokey`/`hikey`/`pitch_keycenter`,
  `lovel`/`hivel` (+ `xfin`/`xfout` if configured), and — if the sample has a loop —
  `loop_mode=loop_continuous`, `loop_start`, `loop_end`, and (in `sfz` crossfade mode only)
  `loop_crossfade`. The `<group>` carries `ampeg_release` and `amp_velcurve_1`.
- **`<name>_release.sfz`** (only if `release.wav`/`.flac` was recorded) — `trigger=release`
  regions with an envelope tuned so the release tail fades in as the held note fades out
  (`ampeg_attack` set to the sustain patch's `ampeg_release`), plus a measured `rt_decay`
  opcode per sample (dB/s decay rate estimated from the release tail itself).

When velocity-mode normalize ran, a human-readable `velocity_range.txt` is also written,
explaining the measured dynamic range and what to set the sampler's velocity curve minimum to.

## The web UI

`autosampler ui` serves a local, single-session app (one browser tab, one instrument open at a
time) with:

- **Config panels** (General/Recording/Selection/Output/Crossfade) — every field auto-rendered
  from the project's JSON Schema, so a new config field appears in the UI with no frontend
  change.
- **Stage Chain** — the DSP chain as a reorderable, toggleable list; each stage expands to its
  own parameter form, also schema-driven.
- **Overrides** — every per-sample manual loop decision in one place: edit loop points
  directly, see one flagged if it names a sample outside the currently loaded set, or reset it
  back to auto-detection. The same overrides a dragged loop handle in the Waveform view writes.
- **Waveform & Loop** — wavesurfer-backed waveform + loop region editor, snap-to-zero-crossing,
  a zoomed loop-seam inset, and dry/processed A/B audition via a hand-wired Web Audio loop
  player (sample-accurate, unlike a plugin's own region-loop playback).
- **Sample Matrix** — a canvas grid of every sample colored by peak/RMS/length, flagging
  anomalously quiet slices; click a cell to jump to it in the Waveform view.
- **Analysis** — release decay-rate scatter and a sustain peak-level histogram.
- **SFZ & Log** — the exact SFZ text a real run would currently write, a diff against the last
  completed render's actual output, and a live job log over SSE while a full render runs.

**Live preview.** Dragging a Stage Chain or Crossfade control reprocesses a small sample subset
against your in-progress, *unsaved* edit — debounced (~150 ms after the last change) and
cancellable, so only one preview request is ever in flight no matter how fast you drag. Nothing
is written to disk until you press Save; the live preview only updates what
`GET /api/audio/{id}?source=preview` serves for auditioning in the Waveform view.

**Keyboard shortcuts:** `Ctrl`/`Cmd`+`S` saves from anywhere. In the Waveform view: `Space`
toggles loop audition, `[`/`]` step to the previous/next sample — all ignored while a text
field, select, or an editing numeric field has focus.

The frontend is hand-written ESM with no build step (vendored Alpine.js + wavesurfer.js under
`frontend/vendor/`) — open `frontend/index.html`'s served route directly, no `npm install`
required.

## HTTP API

The web UI is itself just a client of this API — anything below can be scripted directly.
Interactive docs are always available at `/docs` while `autosampler ui` is running.

| Method | Path | Purpose |
|---|---|---|
| GET/POST/PUT | `/api/project` | Get, open, or validate-and-save the current instrument's config. |
| GET/POST/DELETE | `/api/workspace` | List/add/remove tracked projects (recent-projects index). |
| GET | `/api/schema` | `ProjectConfig` JSON Schema (drives config forms). |
| GET | `/api/schema/stages` | Per-DSP-stage params JSON Schema (drives the stage-chain editor). |
| GET | `/api/samples` | List every raw sample with note/velocity/duration/loop info. |
| GET | `/api/peaks/{id}` | Multi-zoom-level min/max peak data for one sample (`?source=raw\|preview`). |
| GET | `/api/audio/{id}` | Encoded audio bytes for one sample (`?source=raw\|preview`). |
| POST | `/api/preview` | Process a sample subset through the (optionally unsaved) chain for auditioning. |
| GET | `/api/zero-crossings/{id}` | Upward zero-crossing positions, for loop-editor snapping. |
| GET | `/api/matrix` | Per-sample peak/RMS/length metrics for the Sample Matrix view. |
| GET | `/api/analysis` | Per-release-sample RT decay-rate estimates. |
| GET | `/api/sfz/preview` | The SFZ text a real export would currently write, computed live. |
| POST | `/api/jobs` | Start a full background render+export; returns a job id. |
| GET | `/api/jobs/{id}` | Poll a job's status/error/output/events. |
| GET | `/api/jobs/{id}/output` | Read a *completed* job's actual on-disk SFZ output. |
| GET | `/events` | SSE stream of pipeline progress events across all running jobs. |
| POST | `/api/helpers/midi-session` | CLI `midi` parity. |
| GET | `/api/helpers/midi-session/download` | Download a generated `.mid` session. |
| POST | `/api/helpers/prenormalize` | CLI `prenorm` parity. |
| POST | `/api/helpers/concat` | CLI `concat` parity. |

Full renders run in a background thread (never on the async event loop, since DSP is
CPU-bound); progress is available either by subscribing to `/events` or polling
`/api/jobs/{id}`.

## Source layout

```
src/autosampler/
├── cli/          # Typer commands — one thin module per subcommand, plus app.py wiring them up
├── config/       # ProjectConfig (Pydantic v2 schema), TOML read/write, workspace index
├── domain/       # Sample/SampleSet/InstrumentAudio data model, note/velocity math, unit conversions
├── io/           # Audio file reading/writing, slicing raw renders into samples, waveform peak data
├── dsp/          # Individual DSP stages + the registry mapping config stage ids to implementations
├── pipeline/     # Builds the stage chain into an executable graph and runs it (serial or parallel)
├── analysis/     # Sample matrix metrics and release decay-rate estimation, for the UI/reports
├── export/       # Key/velocity zone computation, SFZ document model, audio+SFZ writing, reports
├── helpers/      # MIDI session generation, prenormalize, and concat — shared by CLI and API
└── server/       # FastAPI app, routes, background job system, SSE broadcasting

frontend/         # Hand-written ESM frontend (no build step); vendored deps under frontend/vendor/
tests/            # Mirrors the src/ layout, plus e2e/ and golden/ (golden-file SFZ output tests)
```

## Development

```bash
uv run pytest -q            # test suite
uv run ruff check .         # lint
uv run mypy --strict src    # type-check
```

`mypy --strict` is enforced on `src/` only; the config in `pyproject.toml` also documents the
project's ruff rule set (pydocstyle, pylint, numpy-specific checks, etc.) and pytest markers
(`slow`, `e2e`).
