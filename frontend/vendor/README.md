# Vendored libraries

No Linux Node on this machine (`npm` is Windows npm via WSL interop) → no bundler, no build
step. These are the unmodified UMD/IIFE production builds, fetched once and committed.

| File | Package | Version | Global |
|---|---|---|---|
| `alpine.min.js` | `alpinejs` | 3.14.9 | `window.Alpine` (auto-starts on load) |
| `wavesurfer.min.js` | `wavesurfer.js` | 7.12.11 | `window.WaveSurfer` |
| `wavesurfer.regions.min.js` | `wavesurfer.js` plugins/regions | 7.12.11 | `window.WaveSurfer.Regions` |
| `fonts/jetbrains-mono-var.woff2` | JetBrains Mono | v24 (Google Fonts build) | `@font-face` in `css/tokens.css` |

`fonts/jetbrains-mono-var.woff2` is a single **variable** font file (weight axis 100-800) — one
`@font-face` declaration with a `font-weight: 100 800` range covers every weight `tokens.css`
uses (the browser interpolates), instead of vendoring one static file per weight. License: SIL
OFL 1.1 (`fonts/OFL.txt`).

`wavesurfer.min.js`/`wavesurfer.regions.min.js` are vendored now but not yet loaded by
`index.html` — nothing in step 10's shell needs them; the waveform view (step 11) is what wires
them in. `alpine.min.js` **is** loaded, and must come **after** the app's own modules have
registered their `Alpine.data(...)` components via `alpine:init`, since Alpine auto-starts on
its own `queueMicrotask` as soon as it evaluates.

To update: re-fetch `https://unpkg.com/<pkg>@<version>/dist/...` and the matching
`LICENSE`/`LICENSE.md` from the package's GitHub tag, and bump the version in this table.
