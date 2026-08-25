// Debounced, cancellable live preview: while a DSP-affecting control (a stage param, a
// crossfade setting) is being dragged, this reprocesses a small subset of samples against the
// in-progress, *unsaved* config and holds the result server-side (`POST /api/preview`), so the
// Waveform view's "Processed" audition reflects the edit without a Save round-trip first.
//
// A `numeric-field` fires `numeric-input` once per pointermove frame while dragging — far too
// often to hit the server on every one. This module is where that gets tamed to CLAUDE.md step
// 12's contract: trailing-edge debounce (~150 ms) and never more than one request in flight,
// via `AbortController` cancelling a still-pending request when a newer edit supersedes it.
//
// Runs in `"subset"` mode deliberately: `pipeline.preview`'s own docs call this "as cheap as a
// preview can be... useful while dragging; not something to judge final level by" — exactly the
// tradeoff a control still being scrubbed wants. The manual "Run preview" action elsewhere
// still defaults to `"exact"` for a trustworthy final listen.

const DEBOUNCE_MS = 150;

/**
 * @param {any} api
 * @param {{ onStart?: () => void, onSuccess?: (result: any) => void, onError?: (e: Error) => void }} handlers
 */
export function createLivePreviewScheduler(api, handlers = {}) {
  let timer = null;
  let controller = null;

  function schedule(getConfig) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      if (controller) controller.abort();
      controller = new AbortController();
      const { signal } = controller;
      handlers.onStart?.();
      api
        .runPreview({ config: getConfig(), mode: "subset" }, signal)
        .then((result) => {
          if (signal.aborted) return;
          controller = null;
          handlers.onSuccess?.(result);
        })
        .catch((e) => {
          if (signal.aborted || e.name === "AbortError") return;
          controller = null;
          handlers.onError?.(e);
        });
    }, DEBOUNCE_MS);
  }

  function cancel() {
    if (timer) clearTimeout(timer);
    timer = null;
    if (controller) {
      controller.abort();
      controller = null;
    }
  }

  return { schedule, cancel };
}
