# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

**The README is the deliverable.** It carries the findings, the tables and the
reasoning; the pages only produce the numbers. A measurement that changes a
conclusion has to change the README too, and its raw data has to land in
`docs/measurements/`.

## Commands

```sh
npm install    # optional: fetches @litertjs/core so index.html uses local files
npm start      # = node serve.js → http://localhost:8080
```

There is **no build step, no bundler, no test suite and no linter** — do not go
looking for them. Verification is manual: serve the site, draw a digit, classify.
A server is required: from `file://` the page sits on the opaque null origin and
can fetch neither the ES module nor the model. Without `npm install` the import
map's local paths 404 and `index.html` falls back to the `esm.run` / jsDelivr
CDN — which is exactly what happens on GitHub Pages, since `node_modules` is
gitignored.

Published by GitHub Pages from `main` at
<https://obeletski.github.io/webgpu-experiments/>. **Pushing to `main` publishes.**

## Architecture

Each page is a **single self-contained HTML file** — CSS in one `<style>`, all
logic in one inline `<script>`. There are no shared modules and no imports
between pages; `index.html` is the only one that imports anything at all
(LiteRT.js, via an import map with a CDN fallback). `serve.js` is a
zero-dependency static file server and is not part of the benchmark.

Every page has the same five stages, in this order:

1. **Draw** — pointer events (mouse/touch/stylus in one path) on a 280×280 canvas.
2. **Preprocess** — bounding box → fit to 20×20 → centre of mass at (14,14) →
   28×28, filled black *before* `drawImage`, grayscale, normalized to `[0,1]`,
   replicated across 3 channels.
3. **Infer** — the one part that differs per page.
4. **Time** — run the inference a fixed `TIMING_RUNS` times (default 100) after
   one untimed warm-up run that is discarded; report the mean *and the run
   count*.
5. **Report** — argmax over 10 probabilities.

Stages 1, 2, 4 and 5 are **byte-identical copies** across the three files. That
duplication is the design (see the rules below).

**The model contract**, shared by all three: input `[1,28,28,3]` float32 — 2352
values, *not* the 784 grayscale ones MNIST examples usually assume — output
`[1,10]` float32, **already softmaxed**. Layers are `2352 → 128` (ReLU) `→ 10`.
`index.html` reads the shape from `getInputDetails()` rather than hard-coding it.

Page-specific:

- `index.html` — lazily acquires one `GPUDevice` and reuses it; a backend switch
  recompiles from the model bytes fetched at startup and never touches the
  network; a failed switch falls back to the previously working backend.
- `webgpu.html` — a ~90-line hand-rolled flatbuffer reader pulls the weights out
  of `digit_classifier.tflite` in the browser, then runs them two ways: **fused**
  (one `dispatchWorkgroups(1)` at `@workgroup_size(128)`, hidden layer stays in
  `var<workgroup>`) and **per-layer** (three dispatches through storage buffers,
  how a general runtime must work). The difference between the two *is* the
  measurement of dispatch cost.
- `browser-model-api.html` — fetches nothing at all; hands 2352 floats to C++
  inside Blink. Detects which of {build, flag, secure context} is missing and
  says so rather than failing silently.

## Rules that protect the comparison

- **The drawing code and the MNIST preprocessing are duplicated across all three
  pages on purpose.** The pages are only comparable while those are
  byte-identical. Do **not** factor them into a shared module, and if you change
  one, change all three. The same goes for the timing harness.
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

Force-stop the browser between pages.

**Drive and read the pages over the DevTools protocol, not `screencap`.** Page
`console.log()` does not reach logcat, which is why the original protocol used
`input swipe` / `input tap` and OCR. Attaching to Chrome is both exact and
guarantees an identical stroke across pages rather than aiming at screen
coordinates:

```sh
adb forward tcp:9222 localabstract:chrome_devtools_remote
curl http://localhost:9222/json          # find the page target
# then Runtime.evaluate over its webSocketDebuggerUrl
```

This is a **readout** change only — the harness runs inside the page either way,
and nothing in the timed region involves the driver. Record the deviation in the
measurement file.

**Measure uncommitted code without pushing.** Pages only serves what is on
`main`, but the phone can reach a local server, and `http://localhost` is a
secure context there, so WebGPU still initialises:

```sh
node serve.js
adb reverse tcp:8080 tcp:8080            # phone loads http://localhost:8080/...
```

Use this for any A/B of a change that is not published yet. Serve both variants
side by side so the only difference is the code.

### Traps in the driver

- **Do not wait for the result text to change.** Two consecutive runs can produce
  a byte-identical line — same digit, same confidence, same rounded ms — and a
  wait-for-difference then hangs forever. Clear the element and wait for it to be
  refilled.
- **Overwrite any injected helper**, do not `window.__h = window.__h || {…}`. The
  page is not reloaded between driver runs, so a stale helper survives and silently
  runs the old code.
- **`performance.memory` is useless here** — it returned a quantised 10,000,000 on
  every configuration. Use CDP `Performance.getMetrics` → `JSHeapUsedSize`, after
  `HeapProfiler.collectGarbage`.
- **On-device `transferSize` is unreliable** — most resources come back as cache
  hits reporting 0 even after a force-stop. Take wire sizes from `curl` against
  the live site; only `decodedBodySize` is trustworthy from the device.

**One measurement is 100 inferences.** The harness runs a fixed `TIMING_RUNS`
(default 100) and reports the mean. The count is fixed rather than derived from a
time budget so that **every backend is sampled alike**: under the old budget a
0.15 ms CPU path bought ~33 internal runs and a 2–3 ms GPU path only 1–3, so the
reported spreads were never comparable quantities. Override with
`?timing_runs=400` on any of the three pages.

**100 is not enough for the CPU.** Measured on the OnePlus, the CPU median moves
0.140 → 0.090 ms between 100 and 1000 runs and then holds through 5000; the GPU
moves 1.600 → 1.610 ms, i.e. it is already settled at 100. **At the default the
CPU figure is 56% too high, so measure CPU paths with `?timing_runs=1000`.**

The cause is clock ramp, not quantisation: 100 runs of a 0.09 ms inference is only
~9 ms of work, too short a burst for the core to reach its clock, while 100 GPU
runs is already ~160 ms. Fixed-performance mode does not prevent it — it holds a
sustainable ceiling, not an instantaneous frequency. Quantisation is real but
small: `performance.now()` is clamped to ~1 ms, which bounds resolution at
`1 ms ÷ elapsed` and is unbiased.

The default stays at 100 because 1000 makes a Classify click take ~1.6 s on the
GPU path, which is a bad interactive page. **The stopping rule is what matters:
raise the count until the median stops moving, then stop.** It resolves this case
in three steps — see
[the run-count sweep](docs/measurements/2026-08-24-stock-chrome-cpu-baseline.md).

**Every published figure predates this harness** and was taken under the old time
budget, most of them at 5 ms where the GPU CVs ran 45–55%. Numbers from the
fixed-count harness cannot go in a table with them.

Fixed-performance mode **only fixes the CPU**; GPU noise is under-sampling, not
DVFS.

The driver page and POST collector the README describes (`adb reverse
tcp:8099 tcp:8099`) are **scratch files, deliberately not committed** — rebuilt
per experiment. The README documents the method, not a script to run.

## Where things live

- `docs/measurements/` — raw data behind every published number, one file per
  measurement session, each stating device, browser and protocol.
- `docs/architecture.md` — "Three Paths to One Inference": the API stack each
  page uses, diagrammed, plus an a-priori prediction of memory and time written
  *before* the measurements. It gets memory right and time backwards on purpose;
  that gap is the point.
- `docs/superpowers/specs/`, `docs/superpowers/plans/` — the design spec and
  implementation plan for the direct-WebGPU experiment.

## Conventions

- Vanilla ES modules, no framework, no bundler. `node_modules` is gitignored, so
  `index.html` falls back to the `esm.run` CDN when served from Pages.
- Comments explain *why*. The existing ones carry hard-won reasoning — the
  black-fill-before-`drawImage` note, the centre-of-mass rationale, the
  `Promise.race` timeout — do not strip them.
- Commit messages: imperative, explain the reasoning a reader could not
  re-derive, wrap at 72 characters.
