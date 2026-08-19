// App shell root: workspace/open-project flow, nav section switching, and wiring the
// schema-driven renderers (`param-form.js`, `stage-chain.js`) to the loaded project's config.
// Alpine owns chrome (nav, header, the open-project form) via `x-data`/`x-model`/`x-show`;
// the config panels themselves are vanilla DOM built by the renderers, since their shape is
// driven by a JSON Schema fetched at runtime, not something a fixed Alpine template can express.

import { api } from "./api.js";
import { renderParamForm } from "./param-form.js";
import { renderStageChain } from "./stage-chain.js";

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

    async init() {
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

    renderActiveSection() {
      const el = this.$refs.sectionBody;
      if (!el || !this.config) return;

      if (this.section === "stages") {
        renderStageChain(el, this.config.stages, this.stageSchemas, (stages) => {
          this.config.stages = stages;
          this.markDirty();
          this.renderActiveSection();
        });
        return;
      }

      const isGeneral = this.section === "general";
      const sectionSchema = isGeneral ? this.schema : this.schema.$defs[SECTION_DEFS[this.section]];
      const sectionData = isGeneral ? this.config : this.config[this.section];
      if (!sectionSchema) return;

      renderParamForm(el, sectionSchema, this.schema.$defs, sectionData, (name, value) => {
        sectionData[name] = value;
        this.markDirty();
      });
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
