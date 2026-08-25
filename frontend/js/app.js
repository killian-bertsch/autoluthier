// App shell root: workspace/open-project flow, nav section switching, and wiring the
// schema-driven renderers (`param-form.js`, `stage-chain.js`) to the loaded project's config.
// Alpine owns chrome (nav, header, the open-project form) via `x-data`/`x-model`/`x-show`;
// the config panels themselves are vanilla DOM built by the renderers, since their shape is
// driven by a JSON Schema fetched at runtime, not something a fixed Alpine template can express.

import { api } from "./api.js";
import { renderAnalysisView } from "./analysis-view.js";
import { createLivePreviewScheduler } from "./live-preview.js";
import { renderMatrixView } from "./matrix-view.js";
import { renderOverridesView } from "./overrides-view.js";
import { renderParamForm } from "./param-form.js";
import { renderSfzView } from "./sfz-view.js";
import { renderStageChain } from "./stage-chain.js";
import { renderWaveformView } from "./waveform-view.js";

// Nav section -> the ProjectConfig schema's $defs key holding that section's model.
// "general" is handled separately below: it renders the top-level ProjectConfig schema itself,
// which (once object/array fields are filtered out by `renderParamForm`) leaves just the one
// scalar field at that level, `instrument_name` — no special-casing needed for that field.
const SECTION_DEFS = {
  recording: "RecordingConfig",
  selection: "SelectionConfig",
  output: "OutputConfig",
  crossfade: "CrossfadeConfig",
};

document.addEventListener("alpine:init", () => {
  window.Alpine.data("app", () => ({
    workspace: [],
    schema: null,
    stageSchemas: null,
    project: null,
    config: null,
    section: "workspace",
    openFolderInput: "",
    saveState: "idle", // idle | dirty | saving | saved | error
    errorMessage: "",
    selectedSampleId: null,
    lastJobId: null,
    livePreviewStatus: "idle", // idle | previewing | error
    livePreviewError: "",

    async init() {
      this._livePreview = createLivePreviewScheduler(api, {
        onStart: () => {
          this.livePreviewStatus = "previewing";
        },
        onSuccess: () => {
          this.livePreviewStatus = "idle";
        },
        onError: (e) => {
          this.livePreviewStatus = "error";
          this.livePreviewError = e.message;
          setTimeout(() => {
            if (this.livePreviewStatus === "error") this.livePreviewStatus = "idle";
          }, 3000);
        },
      });
      document.addEventListener("keydown", (e) => this.onGlobalKeydown(e));

      this.workspace = await api.listWorkspace().catch(() => []);
      [this.schema, this.stageSchemas] = await Promise.all([
        api.getProjectSchema(),
        api.getStageSchemas(),
      ]);
      try {
        this._loadProject(await api.getProject());
      } catch {
        // No instrument open yet — the workspace/open-project screen is the landing view.
      }
    },

    /** Debounced (~150ms) + cancellable — see `live-preview.js`. Wired to any control whose
     *  edit feeds the DSP chain (stage params, crossfade), not to fields like Recording that
     *  would need a reload to actually take effect. */
    scheduleLivePreview() {
      this.livePreviewStatus = "previewing";
      this._livePreview.schedule(() => this.config);
    },

    onGlobalKeydown(e) {
      const isSaveCombo = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s";
      if (isSaveCombo && this.project) {
        e.preventDefault();
        if (this.saveState !== "saving") this.save();
        return;
      }
      const tag = (e.target?.tagName || "").toLowerCase();
      const typing = tag === "input" || tag === "textarea" || e.target?.isContentEditable;
      if (typing) return;
      if (e.key === "Escape" && this.errorMessage) {
        this.errorMessage = "";
      }
    },

    async openFolder(path) {
      if (!path) return;
      this.errorMessage = "";
      try {
        this._loadProject(await api.openProject(path));
      } catch (e) {
        this.errorMessage = e.message;
      }
    },

    _loadProject(summary) {
      this._livePreview?.cancel(); // any in-flight preview targeted the instrument being replaced
      this.livePreviewStatus = "idle";
      this.project = summary;
      this.config = summary.config;
      this.saveState = "idle";
      this.section = "general";
      this.$nextTick(() => this.renderActiveSection());
    },

    selectSection(name) {
      this.section = name;
      this.$nextTick(() => this.renderActiveSection());
    },

    openSample(id) {
      this.selectedSampleId = id;
      this.section = "waveform";
      this.$nextTick(() => this.renderActiveSection());
    },

    renderActiveSection() {
      const el = this.$refs.sectionBody;
      if (!el || !this.config) return;
      // Heavy views (waveform/matrix/analysis/sfz) hold live resources — a WaveSurfer
      // instance, an AudioContext playback node, an open EventSource — that a plain
      // `innerHTML = ""` would leak. Every renderer tears its own previous state down when
      // re-entered, but switching to a *different* section bypasses that renderer entirely,
      // so this is the one place that has to catch it regardless of where we're headed next.
      if (el._teardown) {
        el._teardown();
        el._teardown = null;
      }

      if (this.section === "stages") {
        renderStageChain(
          el,
          this.config.stages,
          this.stageSchemas,
          (stages) => {
            this.config.stages = stages;
            this.markDirty();
            this.renderActiveSection();
          },
          () => this.scheduleLivePreview()
        );
        return;
      }

      if (this.section === "overrides") {
        renderOverridesView(el, {
          api,
          config: this.config,
          markDirty: () => this.markDirty(),
          openSample: (id) => this.openSample(id),
        });
        return;
      }

      if (this.section === "waveform") {
        renderWaveformView(el, {
          api,
          config: this.config,
          selectedSampleId: this.selectedSampleId,
          onSelectSample: (id) => {
            this.selectedSampleId = id;
          },
          markDirty: () => this.markDirty(),
        });
        return;
      }

      if (this.section === "matrix") {
        renderMatrixView(el, { api, openSample: (id) => this.openSample(id) });
        return;
      }

      if (this.section === "analysis") {
        renderAnalysisView(el, { api });
        return;
      }

      if (this.section === "sfz") {
        renderSfzView(el, {
          api,
          lastJobId: this.lastJobId,
          setLastJobId: (id) => {
            this.lastJobId = id;
          },
        });
        return;
      }

      const isGeneral = this.section === "general";
      const sectionSchema = isGeneral ? this.schema : this.schema.$defs[SECTION_DEFS[this.section]];
      const sectionData = isGeneral ? this.config : this.config[this.section];
      if (!sectionSchema) return;

      // Only crossfade's fields feed the DSP chain build_chain() reads — Recording/Selection/
      // Output need a reload (Save) to actually take effect, so a live preview there would just
      // be misleading busywork against audio that hasn't changed.
      const onLivePreview = this.section === "crossfade" ? () => this.scheduleLivePreview() : undefined;
      renderParamForm(
        el,
        sectionSchema,
        this.schema.$defs,
        sectionData,
        (name, value) => {
          sectionData[name] = value;
          this.markDirty();
        },
        onLivePreview
      );
    },

    markDirty() {
      this.saveState = "dirty";
    },

    saveStatusLabel() {
      return { idle: "saved", dirty: "unsaved", saving: "saving…", saved: "saved", error: "save failed" }[
        this.saveState
      ];
    },

    async save() {
      this.saveState = "saving";
      this.errorMessage = "";
      try {
        const summary = await api.saveProject(this.config);
        const activeSection = this.section;
        this._loadProject(summary);
        this.section = activeSection;
        this.saveState = "saved";
        this.$nextTick(() => this.renderActiveSection());
        setTimeout(() => {
          if (this.saveState === "saved") this.saveState = "idle";
        }, 1500);
      } catch (e) {
        this.saveState = "error";
        this.errorMessage = e.message;
      }
    },
  }));
});
