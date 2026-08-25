# Auto-sampler CLI 2

A clean-break rewrite of [Auto-sampler_CLI](../Auto-sampler_CLI) — a batch multisample
instrument builder. You record a virtual instrument by playing a generated autosampler MIDI
file into it and rendering one long WAV/FLAC; this tool slices that render back into
individual per-(note, velocity) samples, processes them through a configurable DSP chain, and
emits SFZ 1.0 instrument files.

V2 keeps V1's proven audio math (slicing arithmetic, key/velocity zone logic, loop scoring)
but rebuilds the structure around a single Pydantic v2 schema that drives TOML config, CLI
flags, and a local web UI — and adds a browser-based interactive frontend for waveform/loop
editing, a sample matrix, analysis views, and live preview.

This project is fully independent of V1: separate git repo, separate `uv`-managed virtualenv,
no shared code or config.

See [../CLAUDE.md](../CLAUDE.md) for the full rewrite plan and the development ledger tracking
progress across steps — it is the single source of truth for this project.

## Status

Feature-complete through step 12 of the ledger in [../CLAUDE.md](../CLAUDE.md); see that file
for exact scope and any open follow-ups.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
uv run ruff check .
uv run mypy --strict src
```

## Usage

```bash
autosampler init <dir>          # scaffold a new instrument project
autosampler run <dir>           # process one instrument
autosampler run --scan <dir>    # process every instrument subfolder found
autosampler validate <dir>      # check project.toml without processing
autosampler preview <dir>       # render a subset for quick listening
autosampler midi <dir>          # generate an autosampler MIDI session + metadata.toml
autosampler prenorm <dir>       # peak-normalize source sustain/release recordings
autosampler concat <dir>        # concatenate per-layer recordings into sustain/release.flac
autosampler ui                  # launch the local web interface (opens a browser tab)
```

Every one of these is also reachable through `autosampler ui`'s browser UI — the web app is
the primary surface; the CLI is a thin scripting-friendly alternative over the same
`pipeline`/`export`/helper functions (CLAUDE.md's "web-first" decision).

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
