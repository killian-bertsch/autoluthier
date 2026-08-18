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
| 1 | Config schema + TOML IO | done | _pending_ | Added `tomli-w` dep (stdlib `tomllib` only reads). `StageConfig.params` is a generic JSON dict for now — concrete per-stage param models arrive with the dsp registry in steps 3-5; the pipeline will validate `params` against those once they exist. `loop_crossfade_ms` > 15% of sample length (bug 7) can't be guarded here since it needs actual sample length — that guard belongs in `dsp/loop.py` (step 5). |
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

Step 1 complete: `config/schema.py` defines `RecordingConfig`, `SelectionConfig`,
`CrossfadeConfig`, `OutputConfig`, `StageConfig`/`default_stage_chain()`, and `ProjectConfig`,
all with `model_config = ConfigDict(extra="forbid", strict=True)` (fixes V1 bugs #4 and #9) and
`json_schema_extra=ui_hint(...)` metadata (unit/step/group/order/help) on every field, ready for
`ProjectConfig.model_json_schema()` to drive the frontend later. `config/hints.py` holds the
`ui_hint()` builder. `config/toml_io.py` has generic `read_toml`/`write_toml` plus typed
`load_project`/`save_project` (uses stdlib `tomllib` to read, the new `tomli-w` dep to write).
`config/workspace.py` adds `WorkspaceIndex`/`discover_projects` for an app-level "known
projects" list, independent of any one `project.toml`. `default_stage_chain()` encodes the
locked decision that the transient shaper runs before normalize (fixes bug #10), with the new
optional stages (eq/stereo/limiter) off by default so a fresh project matches V1's out-of-the-box
behavior. `StageConfig.params` is deliberately a generic JSON dict — no DSP stages exist yet
(steps 3-5), so their param models aren't defined; the pipeline builder will validate `params`
against each stage's registered model once `dsp/registry.py` exists. 36 tests in
`tests/unit/{test_config_schema,test_toml_io,test_workspace}.py` cover round-tripping, the
extra-key/strict-bool rejections, cross-field checks (note ranges, `velocity_layers_out` vs
`velocity_layers`), `velocity_map` parsing, and workspace discovery/dedup. Nothing reads real
audio or writes SFZ yet. Next: Step 2 — domain + slicer + audio IO (`domain/notes.py` for the
one true velocity-list/note-name/midi implementation, `domain/models.py` for float32
`Sample`/`SampleSet`, `io/reader.py`/`writer.py`/`slicer.py`), fixing bug #6 (stereo-correct zero
fill on truncated renders).

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
- Every Pydantic config model uses `model_config = ConfigDict(extra="forbid", strict=True)` —
  no exceptions — since that combination is what fixes V1 bugs #4 and #9. New config models
  added in later steps must follow the same pattern.
- Recursive JSON-like type aliases (`JSONValue` in `config/schema.py`) must use PEP 695 `type
  X = ...` syntax, not `X = Union[...]` — Pydantic's schema builder recurses infinitely on the
  old-style form and raises `RecursionError` at model-build time.
- Per-field UI metadata goes through `config/hints.py::ui_hint(...)` passed as
  `json_schema_extra=` — never write a raw `json_schema_extra={...}` dict inline, so the hint
  vocabulary (unit/step/fine_step/log/group/order/help) stays consistent across every field.
- DSP stage parameter models live with their stage module and are registered in
  `dsp/registry.py` (steps 3-5) — `config/schema.py::StageConfig.params` stays a generic
  `dict[str, JSONValue]` and does not grow stage-specific fields itself.
