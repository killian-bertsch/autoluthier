// Thin fetch wrappers over the FastAPI backend. No caching, no retries — the
// server is a single local process on the same machine, and every route here
// is idempotent-enough that a failed request just surfaces to the UI as an error.

/**
 * @param {string} path
 * @param {RequestInit} [init]
 */
async function request(path, init) {
  const response = await fetch(path, {
    headers: init && init.body ? { "Content-Type": "application/json" } : undefined,
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      // Non-JSON error body; fall back to the status text already set above.
    }
    throw new Error(`${response.status} ${path}: ${detail}`);
  }
  if (response.status === 204) return null;
  return response.json();
}

export const api = {
  getProjectSchema: () => request("/api/schema"),
  getStageSchemas: () => request("/api/schema/stages"),

  listWorkspace: () => request("/api/workspace"),
  addToWorkspace: (path, name) =>
    request("/api/workspace", { method: "POST", body: JSON.stringify({ path, name }) }),
  removeFromWorkspace: (path) =>
    request(`/api/workspace?path=${encodeURIComponent(path)}`, { method: "DELETE" }),

  getProject: () => request("/api/project"),
  openProject: (folder) =>
    request("/api/project", { method: "POST", body: JSON.stringify({ folder }) }),
  saveProject: (config) =>
    request("/api/project", { method: "PUT", body: JSON.stringify(config) }),

  listSamples: () => request("/api/samples"),
  getPeaks: (id, source = "raw") =>
    request(`/api/peaks/${encodeURIComponent(id)}?source=${source}`),
  getZeroCrossings: (id, source = "raw") =>
    request(`/api/zero-crossings/${encodeURIComponent(id)}?source=${source}`),
  audioUrl: (id, source = "raw") => `/api/audio/${encodeURIComponent(id)}?source=${source}`,

  getMatrix: () => request("/api/matrix"),
  getAnalysis: () => request("/api/analysis"),

  getSfzPreview: () => request("/api/sfz/preview"),
  runPreview: (body) =>
    request("/api/preview", { method: "POST", body: JSON.stringify(body ?? {}) }),

  startJob: (body) => request("/api/jobs", { method: "POST", body: JSON.stringify(body ?? {}) }),
  getJob: (id) => request(`/api/jobs/${encodeURIComponent(id)}`),
  getJobOutput: (id) => request(`/api/jobs/${encodeURIComponent(id)}/output`),
};
