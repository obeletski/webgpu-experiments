# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

A benchmark, not an app. Three pages run the **same MNIST model** three ways so
their cost can be compared; the digit classifier is the vehicle, not the subject.
The thesis the repo defends is that **the CPU beats the GPU** for a model this
small (~0.6 MFLOP), because per-call GPU overhead dwarfs the arithmetic.

| Page | Implementation |
| --- | --- |
| `index.html` | LiteRT.js, with a CPU (wasm) / GPU (webgpu) switcher |
| `webgpu.html` | the `.tflite` parsed in JS, hand-written WGSL, fused vs per-layer |
| `browser-model-api.html` | `navigator.digitclassifier`, a Web API in a custom Chromium build |

Static site, no build step. `node serve.js` → <http://localhost:8080>. A server
is required: from `file://` the page sits on the opaque null origin and can
fetch neither the ES module nor the model.

Published by GitHub Pages from `main` at
<https://obeletski.github.io/webgpu-experiments/>. **Pushing to `main` publishes.**

## Rules that protect the comparison

- **The drawing code and the MNIST preprocessing are duplicated across all three
  pages on purpose.** Bounding box → fit to 20×20 → centre of mass at (14,14) →
  NHWC ×3 channels. The pages are only comparable while those are byte-identical.
  Do **not** factor them into a shared module, and if you change one, change all
  three. The same goes for the timing harness.
- **Never put stock-Chrome and custom-Chromium figures in one table.** Two
  browsers are involved (below) and they are three milestones apart.
- **Label every measurement with its browser, device and protocol.** Raw data
  goes in `docs/measurements/`; the README summarises and links to it.
- Report what was measured, including the parts that undercut the thesis. The
  README already carries a finding that reverses one of its own conclusions.

## The test device and its two browsers

OnePlus 13 (`CPH2653`, Snapdragon 8 Elite, **Adreno 830**, Android 16 / API 36),
reached over adb. Both browsers are installed and their numbers are kept apart:

| | Package | Used by |
| --- | --- | --- |
| Stock Chrome `150.0.7871.188` | `com.android.chrome` | performance report, the finding, second experiment |
| Custom Chromium `153.0.8005.0` | `org.chromium.chrome` | third experiment only |

`browser-model-api.html` runs **only** on the custom build, launched with
`--enable-blink-features=DigitClassifier`, and needs a secure context.
`navigator.digitclassifier` does not exist in any shipping browser. The Chromium
source for it lives in a separate checkout, `~/chromium/src`.

**Build the custom Chromium as `is_debug = false`.** A debug build does not
merely run slower — it changed the *classification*, predicting `4` where release
predicts `1`, and it distorts timings unevenly enough to invert the ordering
between GPU paths. See the third experiment in the README.

## Measuring

Protocol, from the README's "Getting numbers that reproduce":

```sh
adb shell cmd power set-fixed-performance-mode-enabled true   # unrooted: no governor pinning
# ... 4 s warm-up burst of Classify taps, then N samples, report the median ...
adb shell cmd power set-fixed-performance-mode-enabled false  # always restore
```

Force-stop the browser between pages. Drive the UI with `input swipe` and
`input tap` so every page classifies an identical drawn digit, and read results
from `screencap` — page `console.log()` does not reach logcat.

**Mind the timing budget.** The harness repeats an inference until
`TIMING_MIN_MS` accumulates. At the default 5 ms a 2–3 ms GPU path averages only
1–3 internal runs, giving CVs of 45–55%; at 50 ms the CVs fall to 3–11% **and the
medians move**, so the small budget biases the estimate rather than only widening
its spread. All three pages accept `?timing_min_ms=50&timing_max_runs=200`.
Defaults are unchanged so older figures stay comparable — every number published
before 2026-08-24 was taken at 5 ms and carries that defect.

Fixed-performance mode **only fixes the CPU**; GPU noise is under-sampling, not
DVFS.

## Conventions

- Vanilla ES modules, no framework, no bundler. `node_modules` is gitignored, so
  `index.html` falls back to the `esm.run` CDN when served from Pages.
- Comments explain *why*. The existing ones carry hard-won reasoning — the
  black-fill-before-`drawImage` note, the centre-of-mass rationale, the
  `Promise.race` timeout — do not strip them.
- Commit messages: imperative, explain the reasoning a reader could not
  re-derive, wrap at 72 characters.
