# Seven cases on the custom build: the fence poll A/B'd inside one page

**Date** 2026-08-25 · **Device** OnePlus 13 (`CPH2653`, Snapdragon 8 Elite,
Adreno 830, Android 16 / API 36) · **Browser** **custom Chromium 153.0.8005.0
release** (`org.chromium.chrome`), launched with
`--enable-blink-features=DigitClassifier` — *not* stock Chrome · **Content**
GitHub Pages over HTTPS · **Clocks** `set-fixed-performance-mode-enabled true`,
restored after · **Protocol** driven over the DevTools protocol
(`adb forward tcp:9222`, `Runtime.evaluate`); one identical injected vertical
stroke per page; six page loads per session with `am force-stop` between each;
`?timing_runs=1000`; 3 sessions × 5 samples per case; median reported. Raw data:
[`2026-08-25-custom-chromium-seven-case.json`](2026-08-25-custom-chromium-seven-case.json).

> [!IMPORTANT]
> **Custom-build figures. Never pool them with a stock-Chrome table**, including
> the README's results table. The comparisons *within* this file are sound —
> same browser, same session block, same harness, one drawn input — and that is
> the point of it: seven paths measured against each other rather than against
> history.

The first run since `webgpu.html` made its fence poll selectable, so the poll is
now an **in-page A/B** rather than a comparison between two builds of the page
served side by side ([the earlier method](2026-08-24-webgpu-mapasync-poll.md)).
Crossing the poll with the dispatch mode gives four cases on that page; with
LiteRT.js's two backends and the browser-native API, seven in total.

## Headline

| | Case | Page | Median | CV | vs CPU |
| --- | --- | --- | ---: | ---: | ---: |
| **C2** | LiteRT.js — **CPU (`wasm`)** | `index.html` | **0.090 ms** | **0.0%** | 1.00× |
| **C3** | Direct WebGPU — **fused · poll** | `webgpu.html` | **0.480 ms** | 7.9% | 5.3× |
| **C4** | Direct WebGPU — **per-layer · poll** | `webgpu.html` | **0.510 ms** | 43.1%¹ | 5.7× |
| **C1** | LiteRT.js — GPU (`webgpu`) | `index.html` | **1.410 ms** | 3.8% | 15.7× |
| **C5** | Direct WebGPU — **fused · no poll** | `webgpu.html` | **2.440 ms** | 1.7% | 27.1× |
| **C6** | Direct WebGPU — **per-layer · no poll** | `webgpu.html` | **2.540 ms** | 1.5% | 28.2× |
| **C7** | `navigator.digitclassifier` | `browser-model-api.html` | **3.710 ms** | 1.4% | 41.2× |

n = 15 for every row (3 sessions × 5). All seven predicted `1` at 100%
confidence from the same stroke, and each page printed the case it ran, so no
row can be a mislabelled neighbour.

¹ C4's CV is one sample: a single 1.46 ms reading among fourteen between 0.44 and
0.62. Drop it and the CV is 10.7%. The median is unaffected; see
[the tail](#the-polls-tail-does-not-average-out).

## 1. The fence poll is worth ~5×, measured inside one page

| Dispatch mode | no poll | poll | ratio |
| --- | ---: | ---: | ---: |
| fused | 2.440 ms | **0.480 ms** | **5.08×** |
| per-layer | 2.540 ms | **0.510 ms** | **4.98×** |

Both distributions are disjoint (U = 225 of 225, permutation p < 0.00001 over
200,000 resamples). Nothing differs between the two columns but how `run()`
waits: same submits, same buffers, same shaders, bit-identical output. This is
the cleanest form of the [lazy-fence finding](2026-08-24-webgpu-mapasync-poll.md)
so far — one page, one session, one URL parameter apart.

## 2. A dispatch costs ~0.05 ms — the tightest measurement of it yet

The unpolled pair prices dispatch better than anything before it, because at
CV 1.5–1.7% the two rows barely overlap:

| | fused (1 dispatch) | per-layer (3 dispatches) | Δ for two extra dispatches |
| --- | ---: | ---: | ---: |
| no poll | 2.440 ms | 2.540 ms | **+0.100 ms** (U = 14/225, p = 0.00001) |
| poll | 0.480 ms | 0.510 ms | +0.030 ms (U = 66/225, p = 0.031) |

**~0.05 ms per dispatch**, against the second experiment's
`0.03 – 0.51 ms` 95% CI — an order of magnitude tighter, and it lands near the
bottom of that interval. The polled pair agrees in sign but is noise-dominated;
the unpolled pair is the measurement. Either way the conclusion the second
experiment drew stands and hardens: **dispatch is minor**. Two extra dispatches
cost 4% of an unpolled inference and are not what makes the GPU slow here.

## 3. The C++ browser API is 1.27 ms slower than the *unpolled* JavaScript page

This is the finding this session adds, and it corrects an explanation the repo
currently gives.

| | Median | CV |
| --- | ---: | ---: |
| Direct WebGPU — fused · **no poll** (JS) | 2.440 ms | 1.7% |
| `navigator.digitclassifier` (C++ in Blink) | **3.710 ms** | 1.4% |

**+1.270 ms, 1.52×, disjoint** (C5's slowest sample 2.53 is below C7's fastest
3.65; U = 225/225, p < 0.00001).

The [existing account](../findings.md#re-measured-with-the-fence-poll-the-c-api-is-now-the-slowest-gpu-path)
says the C++ path's ~3.7 ms is "the signature of a path pinned to the lazy-fence
floor" — it does not poll, so it pays the full wait. That is half an
explanation. **The floor is now measured directly on the same device, browser
and session: 2.44 ms**, by a JavaScript page doing the same work with the poll
switched off. The C++ path pays 1.27 ms *beyond* it.

Two candidate explanations, **neither tested here**:

- **The submit split.** `webgpu.html` issues the compute and the readback copy
  as two submits, which was worth ~0.5 ms when it landed
  ([data](2026-08-24-webgpu-two-submit.md)). If the C++ path uses one, that
  accounts for part of the gap — and would mean the comparison above prices *two
  choices*, not one.
- **The Blink boundary.** `classify()` copies 2352 floats from a JS
  `Float32Array` into C++ and an integer back, per call. Not obviously ~0.7 ms,
  but not measured either.

What would settle it is reading the module's implementation — which this host
does not have (there is no `~/chromium/src` checkout on it) — or instrumenting
`classify()` the way `run()` was instrumented for
[the trace](2026-08-24-litert-vs-handwritten-tracing.md). Until then: the C++
path is slower than an ordinary unpolled WebGPU page by an amount the lazy fence
does not explain, and that is as far as this data goes.

## 4. LiteRT.js beats an ordinary WebGPU page, and loses to a polled one

| | Median |
| --- | ---: |
| Direct WebGPU — fused · poll | 0.480 ms |
| LiteRT.js — GPU | 1.410 ms |
| Direct WebGPU — fused · no poll | 2.440 ms |

LiteRT sits **between the two fence modes**, which is exactly what the
[trace](2026-08-24-litert-vs-handwritten-tracing.md) predicted: it never polls,
but its per-inference submits and buffer churn service the fence a few times per
call, so it lands between doing nothing about the fence (2.44 ms) and poking it
from every task (0.48 ms). The hand-written page beats LiteRT **only** with the
poll on; with the poll off, LiteRT is 1.73× faster than it.

## The poll's tail does not average out

C8, one session, 5 samples per cell:

| `?timing_runs=` | fused · poll | fused · no poll |
| ---: | ---: | ---: |
| 100 | 0.770 ms | 2.530 ms |
| 1000 | 0.530 ms | 2.500 ms |
| 5000 | 0.680 ms | 2.480 ms |

**The unpolled path satisfies the stopping rule; the polled path does not.**
No-poll moves 2% across a 50× change in run count — settled at 100. The polled
median wanders 0.77 → 0.53 → 0.68 with no sign of converging, and its samples
carry the same fat tail at every count (1.02 at 100 runs, 1.22 at 1000, 0.86 at
5000).

That is not under-sampling, and raising the count is not the fix. Each timed
measurement is already a mean over N runs; if the noise were per-run it would
average away as N grows. It does not, because the poll's latency rides
**event-loop scheduling**, which scales with the measurement rather than within
it. The right treatment is what this session did — many samples across
force-stopped sessions, medians reported — not a bigger N.

So the fence-poll figure reproduces as a **median** (0.480 here against 0.520 in
the [four-way](2026-08-25-custom-chromium-four-way.md) and ~0.46 in the
[3-session](2026-08-24-fence-poll-3session.md) run) while individual samples stay
unreliable. It remains the least settled number in the repo, and the one that
most depends on what else the page is doing.

## What this does not settle

- **Nothing about stock Chrome.** Custom build only. The README's results table
  still waits on a stock-Chrome re-measure of the fence poll served from Pages.
- **Cold start and pipeline build** were captured (n = 1 per session, in the
  JSON) but not analysed: they range 7.7 – 73.1 ms with no pattern at that
  sample size, and `getModel()` on the C++ page ranged 20.9 – 234.3 ms. Three
  points is too few for any of it to mean something.
- **Order effects.** Cases ran in a fixed order every session (the repo's
  convention), so a systematic drift within a session would not show up.
- **One device, one driver, one drawn digit.**
