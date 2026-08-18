# CLAUDE.md — Auto-sampler CLI 2 (V2)

Read this file first in every new context. It is the single source of truth across cleared
contexts for what is done, what is in progress, and what the next step should do first.

Full rewrite rationale, locked decisions, stack, preserved audio-math contracts, and the
V1-bugs-fixed list live in [docs/PLAN.md](docs/PLAN.md) — read it alongside this file before
starting a step that touches something you're unsure about.

This project is fully independent of `../Auto-sampler_CLI` (V1): separate git repo (local
only, no remote), separate `uv`-managed `.venv/`, no shared code or config, no importer.

## Per-step contract

1. Read this file (and `docs/PLAN.md` if the step needs the full rationale).
2. Do the work for that step only.
3. `uv run ruff check .` · `uv run mypy --strict src` · `uv run pytest -q` — all green.
4. Update the ledger row below + "Current state" + any new entries in "Conventions
   established" — as the last action before committing.
5. `git commit` naming the step.

## Development Ledger

Status: todo | in-progress | done | blocked

| # | Step | Status | Commit | Notes / gotchas for the next context |
|---|------|--------|--------|--------------------------------------|
| 0 | Scaffold, tooling, ledger | done | 9522899 | uv venv (py3.12.3, uv 0.12.3) + editable install verified; ruff/mypy --strict/pytest all green |
| 1 | Config schema + TOML IO | todo | — | |
| 2 | Domain + slicer + audio IO | todo | — | |
| 3 | DSP framework + core stages | todo | — | |
| 4 | New DSP stages (EQ, stereo, transient, limiter) | todo | — | |
| 5 | Loop detection + baked crossfade | todo | — | |
| 6 | Pipeline, events, parallelism | todo | — | |
| 7 | SFZ export | todo | — | |
| 8 | CLI + ported helpers | todo | — | |
| 9 | Server (FastAPI + SSE) | todo | — | |
| 10 | Frontend shell + schema-driven forms | todo | — | |
| 11 | The four views (waveform/loop, matrix, analysis, SFZ+log) | todo | — | |
| 12 | Live preview, overrides, polish | todo | — | |

### Current state

Step 0 complete: repo scaffolded, `uv venv` created and `.venv/` installed in editable+dev mode, ruff/mypy --strict/pytest all verified green on the empty package (pytest exit 5 = no tests collected yet, which is expected). `docs/PLAN.md` holds the approved plan. Nothing functional exists yet — all `src/autosampler` packages are empty stubs with only a docstring in `__init__.py`, except a placeholder Typer app in `cli/app.py`. Next: Step 1 — config schema + TOML IO (`config/schema.py`, `hints.py`, `toml_io.py`, `workspace.py`), fixing V1 bugs #4 (bool-as-int) and #9 (silently ignored unknown keys) via `extra="forbid"` + strict types.
(`.venv/` and `__pycache__/` both excluded — V1's mistake not repeated), the full
`src/autosampler/{config,domain,io,dsp,pipeline,export,analysis,helpers,cli,server}` package
skeleton with empty `__init__.py` files, `tests/{unit,dsp,export,api,e2e,golden}` skeleton,
`frontend/{css,js,vendor}` placeholders, a stub Typer entry point at
`src/autosampler/cli/app.py`, `README.md`, and this ledger. `docs/PLAN.md` holds the approved
plan. Nothing functional exists yet. Next: finish Step 0 by creating `.venv`, installing in
editable+dev mode, and confirming `ruff`/`mypy --strict`/`pytest` all run clean on the empty
package — then commit and move to Step 1 (config schema + TOML IO).

### Conventions established

- Source layout: `src/autosampler/...` (src layout, not flat) so the installed package can't
  accidentally import from the repo root.
- Every stage/module directory is a real package (`__init__.py`), even if currently empty.
- `ruff` config in `pyproject.toml` enables docstring (`D`, google convention), annotation
  (`ANN`), and pylint (`PL`) rule groups — write typed, docstringed code from the start rather
  than retrofitting for "strict rigor" later.
- Tests mirror the source layout under `tests/` (`unit/`, `dsp/`, `export/`, `api/`, `e2e/`,
  `golden/`) rather than being colocated with source.
- Always invoke tooling via `uv run ...` so the independent `.venv/` is used, never a bare
  `python`/`pytest` that might resolve to V1's environment or the system interpreter.
