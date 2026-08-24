// SFZ preview + diff + job log. "Diff" here means: the live SFZ text `/api/sfz/preview` would
// currently produce, compared line-by-line against what the *last completed render actually
// wrote to disk* (`/api/jobs/{id}/output`) — the only prior version of the instrument this app
// keeps anywhere. There is no version history, so "diff against nothing yet" (before any job has
// run this session) is a real, labeled state rather than an empty diff pretending to be current.
//
// The job log subscribes to `/events` (SSE) and keeps only events tagged with the job it started,
// since that stream carries every job's events, not just this session's.

let viewTokenCounter = 0;

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

/** Classic O(n*m) LCS-based line diff — fine at SFZ-file sizes (tens to low hundreds of lines). */
function diffLines(a, b) {
  const n = a.length;
  const m = b.length;
  const lcs = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const out = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      out.push({ type: "same", text: a[i] });
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      out.push({ type: "removed", text: a[i] });
      i++;
    } else {
      out.push({ type: "added", text: b[j] });
      j++;
    }
  }
  while (i < n) out.push({ type: "removed", text: a[i++] });
  while (j < m) out.push({ type: "added", text: b[j++] });
  return out;
}

function renderDiff(oldText, newText) {
  const rows = diffLines((oldText ?? "").split("\n"), (newText ?? "").split("\n"));
  const pre = document.createElement("pre");
  pre.className = "sfz-diff mono";
  rows.forEach((row) => {
    const line = document.createElement("div");
    line.className = `diff-line diff-${row.type}`;
    const marker = { same: " ", added: "+", removed: "-" }[row.type];
    line.textContent = `${marker} ${row.text}`;
    pre.appendChild(line);
  });
  return pre;
}

/**
 * @param {HTMLElement} container Emptied and filled with the SFZ + log view.
 * @param {{ api: any, lastJobId: string | null, setLastJobId: (id: string) => void }} ctx
 */
export function renderSfzView(container, ctx) {
  if (container._teardown) container._teardown();
  container.innerHTML = '<div class="view-loading">Loading SFZ preview…</div>';
  const token = String(++viewTokenCounter);
  container.dataset.viewToken = token;

  ctx.api
    .getSfzPreview()
    .then((preview) => {
      if (container.dataset.viewToken !== token) return; // superseded by a later call
      buildView(container, preview, ctx);
    })
    .catch((e) => {
      if (container.dataset.viewToken !== token) return;
      container.innerHTML = `<div class="empty-state"><p>${escapeHtml(e.message)}</p></div>`;
    });
}

function buildView(container, preview, ctx) {
  container.innerHTML = "";
  const state = { doc: "sustain", eventSource: null, logLines: [] };

  const root = document.createElement("div");
  root.className = "sfz-view";
  root.innerHTML = `
    <div class="wv-toolbar">
      <select class="field-select sfz-doc-picker">
        <option value="sustain">Sustain SFZ</option>
        <option value="release" ${preview.release_sfz ? "" : "disabled"}>Release SFZ</option>
      </select>
      <button class="btn btn-ghost sfz-toggle-diff">Show diff vs. last render</button>
      <span class="sfz-range">${
        preview.measured_dynamic_range_db != null
          ? `Measured dynamic range: ${preview.measured_dynamic_range_db.toFixed(1)} dB`
          : ""
      }</span>
    </div>
    <div class="sfz-body"></div>

    <div class="group-header">Render</div>
    <div class="sfz-job-controls">
      <button class="btn btn-primary sfz-run-job">Run full render</button>
      <span class="sfz-job-status"></span>
    </div>
    <pre class="sfz-log mono"></pre>
  `;
  container.appendChild(root);
  container._teardown = () => {
    if (state.eventSource) state.eventSource.close();
  };

  const docPicker = root.querySelector(".sfz-doc-picker");
  const diffToggle = root.querySelector(".sfz-toggle-diff");
  const body = root.querySelector(".sfz-body");
  let showDiff = false;
  let lastOutput = null; // { sustain_sfz, release_sfz } from the last completed job, if fetched
  // `ctx.lastJobId` is only a snapshot from when this view was mounted; track our own copy so
  // starting a new render from this same view updates the diff target immediately, without
  // waiting for the app shell to re-render this view with a fresh ctx.
  let currentJobId = ctx.lastJobId;

  function currentText() {
    return state.doc === "sustain" ? preview.sustain_sfz : preview.release_sfz;
  }

  function render() {
    body.innerHTML = "";
    if (!showDiff) {
      const pre = document.createElement("pre");
      pre.className = "sfz-text mono";
      pre.textContent = currentText() ?? "";
      body.appendChild(pre);
      return;
    }
    if (!currentJobId) {
      body.innerHTML = '<p class="chart-empty">No render has completed yet this session — nothing to diff against.</p>';
      return;
    }
    const showLoading = () => {
      body.innerHTML = '<div class="view-loading">Loading last render output…</div>';
    };
    if (lastOutput) {
      const prior = state.doc === "sustain" ? lastOutput.sustain_sfz : lastOutput.release_sfz;
      body.appendChild(renderDiff(prior, currentText()));
      return;
    }
    showLoading();
    ctx.api
      .getJobOutput(currentJobId)
      .then((output) => {
        lastOutput = output;
        render();
      })
      .catch(() => {
        body.innerHTML = '<p class="chart-empty">Could not load the last render\'s output.</p>';
      });
  }

  docPicker.addEventListener("change", () => {
    state.doc = docPicker.value;
    render();
  });
  diffToggle.addEventListener("click", () => {
    showDiff = !showDiff;
    diffToggle.classList.toggle("is-active", showDiff);
    diffToggle.textContent = showDiff ? "Show plain preview" : "Show diff vs. last render";
    render();
  });

  render();

  const runBtn = root.querySelector(".sfz-run-job");
  const status = root.querySelector(".sfz-job-status");
  const log = root.querySelector(".sfz-log");

  function appendLog(line) {
    state.logLines.push(line);
    log.textContent = state.logLines.join("\n");
    log.scrollTop = log.scrollHeight;
  }

  function subscribe(jobId) {
    if (state.eventSource) state.eventSource.close();
    const es = new EventSource("/events");
    state.eventSource = es;
    es.onmessage = (msg) => {
      let payload;
      try {
        payload = JSON.parse(msg.data);
      } catch {
        return;
      }
      if (payload.job_id !== jobId) return;
      appendLog(formatEvent(payload));
      if (payload.event === "run_completed" || payload.event === "run_failed") {
        status.textContent = payload.event === "run_completed" ? "Completed." : `Failed: ${payload.message}`;
        es.close();
        state.eventSource = null;
        lastOutput = null; // stale now that a new render just wrote fresh output
      }
    };
    es.onerror = () => {
      appendLog("(stream error — polling GET /api/jobs instead)");
      es.close();
      state.eventSource = null;
      pollJob(jobId);
    };
  }

  function pollJob(jobId) {
    const poll = () => {
      ctx.api.getJob(jobId).then((job) => {
        status.textContent = job.status === "running" ? "Running…" : job.status;
        if (job.status === "running") setTimeout(poll, 500);
      });
    };
    poll();
  }

  runBtn.addEventListener("click", () => {
    state.logLines = [];
    log.textContent = "";
    status.textContent = "Starting…";
    ctx.api
      .startJob({})
      .then(({ job_id }) => {
        currentJobId = job_id;
        ctx.setLastJobId(job_id);
        status.textContent = "Running…";
        subscribe(job_id);
      })
      .catch((e) => {
        status.textContent = `Failed to start: ${e.message}`;
      });
  });
}

function formatEvent(payload) {
  switch (payload.event) {
    case "run_started":
      return `run started: ${payload.instrument} (${payload.total_steps} steps)`;
    case "step_started":
      return `step ${payload.index} started: ${payload.label}`;
    case "step_progress":
      return `step ${payload.index}: ${payload.completed_units}/${payload.total_units} (${(payload.fraction * 100).toFixed(0)}%)`;
    case "step_completed":
      return `step ${payload.index} done in ${payload.duration_s.toFixed(2)}s`;
    case "run_completed":
      return `run completed: ${payload.sample_count} samples in ${payload.duration_s.toFixed(2)}s`;
    case "run_failed":
      return `run failed: ${payload.error_type}: ${payload.message}`;
    default:
      return JSON.stringify(payload);
  }
}
