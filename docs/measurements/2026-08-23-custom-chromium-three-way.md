# Three implementations, one model, one device

**Date** 2026-08-23/24 · **Device** OnePlus 13 (`CPH2653`, Snapdragon 8 Elite,
Adreno 830, Android 16 / API 36) · **Browser** a locally built Chromium
**153.0.8005.0**, measured in **both** a release and a debug configuration ·
**Content** served from <https://obeletski.github.io/webgpu-experiments/> over
HTTPS.

All three pages in this repository run the same model. This measures them
against the same drawn digit with the same timing harness, under the fixed-clock
protocol the [README](../../README.md#getting-numbers-that-reproduce) already
uses.

> [!IMPORTANT]
> **The debug build did not merely run slower — it classified the digit
> differently.** LiteRT.js on WebGPU predicted `4` on the debug build and `1` on
> the release build, from identical page bytes and an identical input. See
> [The debug build changes the answer](#the-debug-build-changes-the-answer).
> Treat the release column as the real result and the debug column as a warning
> about benchmarking debug browsers.

---

## Contents

- [Results](#results)
- [The debug build changes the answer](#the-debug-build-changes-the-answer)
- [Debug vs release](#debug-vs-release)
- [One-time costs](#one-time-costs)
- [Protocol](#protocol)
- [Raw data](#raw-data)
- [Caveats](#caveats)

---

## Results

Release build. Steady-state cost of one `classify()`, median of 5:

| Implementation | Page | Median | Mean | CV | Min–Max | vs CPU |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| LiteRT.js — **CPU (wasm)** | `index.html` | **0.15 ms** | 0.16 | 12.5% | 0.14 – 0.19 | 1.00× |
| LiteRT.js — GPU (webgpu) | `index.html` | **1.43 ms** | 1.89 | 55.5% | 1.10 – 3.70 | 9.5× |
| Direct WebGPU — fused | `webgpu.html` | **2.50 ms** | 3.16 | 45.1% | 2.47 – 5.70 | 16.7× |
| `navigator.digitclassifier` | `browser-model-api.html` | **2.70 ms** | 2.72 | 8.1% | 2.50 – 3.05 | 18.0× |

**The CPU wins by 9.5× against the best GPU path.** This reproduces the
repository's existing finding — a ~0.6 MFLOP model is far too small for GPU
dispatch and readback to pay for themselves — and the margin is close to the 7.7×
measured previously on stock Chrome 150.

The CPU row is a follow-up, not part of the original three; it was added on the
debug run to locate a wrong prediction and kept because it is the only baseline
that makes the GPU numbers mean anything.

Two things worth noting about the GPU rows:

- **`navigator.digitclassifier` is the slowest of the three GPU paths but by far
  the steadiest** — CV 8.1% against 45% and 55%. Its five measurements span
  2.50–3.05 ms where the direct-WebGPU page spans 2.47–5.70 ms.
- **The ordering inverted between builds.** On debug the built-in API was the
  *fastest* GPU path (3.65 ms vs 7.30 and 8.90); on release it is the slowest.
  It gained the least from the release build (1.35×) because its cost is
  dominated by the GPU round-trip — dispatch, `MapAsync` readback, event-loop
  flush — which does not care how Chromium was compiled. The JavaScript paths
  gained 2.9× and 6.2× because their per-call overhead was what the debug build
  was inflating.

As the README already records, a single GPU inference exceeds the harness's 5 ms
`TIMING_MIN_MS` budget, so GPU figures average fewer internal runs than CPU
figures. `TIMING_MIN_MS` was deliberately **not** changed, because changing it
would break comparability with every number already published here.

---

## The debug build changes the answer

Three of four configurations predicted `1` on both builds. LiteRT.js on WebGPU
predicted **`4` at 100.0% confidence on the debug build, and `1` on release** —
five out of five measurements each way.

The first run served the pages from the device's own `127.0.0.1` with a local
`node_modules`; the second served them from GitHub Pages, where `index.html`
falls back to the CDN. So build and content source changed together. **That
confound was resolved by re-installing the debug APK and re-running it against
the same Pages content:**

| Build | Content | Prediction |
| --- | --- | --- |
| Debug | device `127.0.0.1`, local `node_modules` | `4` (100.0%) |
| **Debug** | **GitHub Pages, CDN LiteRT** | **`4` (100.0%)** |
| Release | GitHub Pages, CDN LiteRT | `1` (100.0%) |

The library is not the variable either: local `@litertjs/core` is `2.5.3` and the
CDN fallback is pinned to `2.5.3`. **The browser build is the variable.**

On the debug build the disagreement was internal to LiteRT.js and reproducible on
demand — toggling the accelerator inside one page load, without redrawing, flipped
the answer and flipped it back:

| Step | Backend | Result |
| --- | --- | --- |
| 1 | GPU (webgpu) | Predicted **4** — 15.70 ms |
| 2 | CPU (wasm) | Predicted **1** — 0.85 ms |
| 3 | GPU (webgpu) | Predicted **4** — 8.10 ms |

**What this establishes:** a `is_debug = true` Chromium can produce a *different
classification*, not just slower timings. Benchmarking or validating a GPU
inference path on a debug browser can therefore mislead about correctness as well
as speed.

**What it does not establish:** the cause. One input, one device, one driver, and
no isolation of whether it originates in Dawn, in the LiteRT delegate, or in a
Blink code path that only exists when DCHECKs are compiled in. It is not a claim
about LiteRT.js in shipping browsers — on the release build, LiteRT's WebGPU
backend agrees with everything else.

---

## Debug vs release

Same device, same protocol, same content, same 5-measurement medians:

| Implementation | Debug | Release | Speed-up |
| --- | ---: | ---: | ---: |
| LiteRT.js — CPU (wasm) | 0.88 ms | 0.15 ms | 5.9× |
| LiteRT.js — GPU (webgpu) | 8.90 ms | 1.43 ms | 6.2× |
| Direct WebGPU — fused | 7.30 ms | 2.50 ms | 2.9× |
| `navigator.digitclassifier` | 3.65 ms | 2.70 ms | 1.35× |

Build configuration, identical apart from the debug switches:

| | `out/Default` | `out/Release` |
| --- | --- | --- |
| `is_debug` | `true` | `false` |
| `dcheck_always_on` | `true` | `false` |
| `symbol_level` | 1 | 1 |
| APK size | 715,197,274 B | **389,907,305 B** |

---

## One-time costs

Release build. Paid once per page load, before any user-visible classification:

| Implementation | Setup | Cold start | What "setup" means |
| --- | ---: | ---: | --- |
| LiteRT.js — GPU | 131.3 ms | 124.9 ms | `loadAndCompile` for the webgpu accelerator |
| LiteRT.js — CPU | 8.0 ms | 2.5 ms | `loadAndCompile` for the wasm accelerator |
| Direct WebGPU | <1 ms | 31.7 ms | building the compute pipeline |
| `navigator.digitclassifier` | 23.4 ms | 39.3 ms | `getModel()` |

Neither LiteRT figure includes the network: the ~9 MB wasm runtime and the 1.2 MB
`.tflite` are fetched before compilation is timed. The built-in API has nothing to
fetch at all — model and weights ship inside the browser binary.

`getModel()` and its cold start do not overlap, but they are not attributed the
way the labels suggest. `getModel()` waits only for `RequestAdapter` and
`RequestDevice`; the shader module and pipeline it orders are Dawn wire calls that
are never flushed there, so the pipeline build is paid on the first `classify()`.
The two builds make this visible: `getModel()` fell 208 → 23 ms while its cold
start *rose* 11.9 → 39.3 ms. Time to first answer went from ~220 ms to ~63 ms.

---

## Protocol

Per the README's reproducibility section, identical for every configuration:

1. **Fixed clocks** — `adb shell cmd power set-fixed-performance-mode-enabled true`
   for the whole session, restored to `false` afterwards. The device is unrooted,
   so the `walt` governor cannot be pinned; this is Android's own benchmarking
   facility and holds a clock the device can sustain.
2. **Browser restarted between pages** — `am force-stop org.chromium.chrome`, then
   a fresh `am start` at the page URL.
3. **Warm-up** — 4 s of continuous Classify taps, untimed and discarded.
4. **5 measurements**, each a separate Classify tap, screenshotted and read off the
   device. Median reported.
5. Each measurement is itself the harness's mean: one untimed warm-up run, then
   repeat until 5 ms has accumulated (max 49 runs).

Every page classified **the same drawn digit**: a vertical bar produced by
`input swipe 541 700 541 1300 500`. On the debug run the rendered canvases were
compared pixel-wise and differ by a mean absolute value of 0.03/255 over a 59×188
region at the stroke's end — 0.17% of total ink, from swipe timing jitter.

Chromium was launched with `--enable-blink-features=DigitClassifier`, which
`browser-model-api.html` requires and the other two ignore.

Conditions: battery 92–97%, charging throughout; thermal zone 0 stayed between
34.7 °C and 38.6 °C, so nothing was thermally limited.

---

## Raw data

Milliseconds, in the order measured. Parentheses give the number of internal runs
the harness averaged for that measurement.

**Release** (content from GitHub Pages):

```
LiteRT.js — CPU (wasm)        0.19 (27)  0.15 (33)  0.15 (34)  0.14 (35)  0.15 (33)
LiteRT.js — GPU (webgpu)      1.10 (5)   1.43 (4)   1.87 (3)   3.70 (2)   1.35 (6)
Direct WebGPU — fused         2.50 (3)   2.47 (3)   2.65 (2)   2.47 (3)   5.70 (1)
navigator.digitclassifier     3.05 (2)   2.70 (2)   2.50 (2)   2.80 (2)   2.55 (2)
```

**Debug** (content from the device's own 127.0.0.1):

```
LiteRT.js — CPU (wasm)        0.90 (6)   0.71 (7)   0.80 (7)   1.06 (5)   0.88 (6)
LiteRT.js — GPU (webgpu)      8.60 (1)   8.90 (1)   8.90 (1)   9.50 (2)   7.20 (1)
Direct WebGPU — fused         8.30 (1)   2.95 (2)   3.70 (2)   7.30 (2)  10.20 (1)
navigator.digitclassifier     6.30 (1)   7.20 (1)   3.05 (2)   3.05 (2)   3.65 (2)
```

Predictions were `1` for every measurement except LiteRT.js — GPU (webgpu) on the
debug build, which returned `4` on all five.

---

## Caveats

1. **Not a shipping browser.** This is a local `is_official_build = false`
   Chromium 153. Comparisons within each column are sound; against stock Chrome
   they are indicative at best.
2. **Five measurements is a small sample** against CVs of 45–55% on two of the GPU
   rows. The medians order the implementations; they do not support quoting ratios
   to two significant figures.
3. **Fixed-performance mode only fixes the CPU.** The README established this
   previously — the GPU stayed at CV 49% under the same control — and these GPU
   CVs are consistent with that. The CPU row's 12.5% and the built-in API's 8.1%
   show what a well-sampled measurement looks like here.
4. **One input.** A vertical bar is an easy `1` and an unusual `4`. Nothing here
   samples the model's accuracy across digits.
5. **The GPU rows are under-sampled** by the 5 ms timing budget, as described
   above.
6. **The debug misclassification is uninvestigated.** Reproducible, but its cause
   is unknown.
