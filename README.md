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

In development — see the ledger in [../CLAUDE.md](../CLAUDE.md) for the current step.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
uv run ruff check .
uv run mypy --strict src
```

## Usage (once implemented)

```bash
autosampler init <dir>          # scaffold a new instrument project
autosampler run <dir>           # process one instrument
autosampler run --scan <dir>    # process every instrument subfolder found
autosampler validate <dir>      # check project.toml without processing
autosampler preview <dir>       # render a subset for quick listening
autosampler ui                  # launch the local web interface
```
