# CPU vs GPU Inference in the Browser

**How much does it cost to ask a browser to run a tiny neural network?** One
hand-drawn MNIST digit classifier, implemented three ways — through
[LiteRT.js](https://ai.google.dev/edge/litert/web) on WebAssembly and WebGPU,
by hand against the WebGPU API in WGSL, and as
**`navigator.digitclassifier`: the model compiled into a custom Chromium build**,
weights linked into the binary, no JavaScript in the timed region at all.

Draw a digit, classify it, then run the same drawing through another
implementation. Every figure the UI shows — compile, cold start, steady state —
is part of the comparison.

The classifier is the vehicle, not the subject: it exists to give every path
identical work. **On every device tested the CPU wins**, because a 0.6 MFLOP
inference is far too small to pay for a GPU round-trip — and **putting the model
inside the browser does not rescue the GPU**. The C++ implementation with the
leanest stack in the repository is the *slowest* GPU path measured, for reasons
that took three experiments to pin down.

<p align="center">
  <img src="screenshot.png" width="360"
       alt="The app running on an Android phone: a hand-drawn 3 on a black canvas, predicted as 3 at 100.0% confidence in 3.80 ms on the webgpu backend. Above the result, a CPU/GPU segmented switch with GPU selected, showing compile 89.90 ms and cold start 12.70 ms.">
</p>

**Live demo:** <https://obeletski.github.io/webgpu-experiments/> ·
**How it was measured and what it means:** [`docs/findings.md`](docs/findings.md)

The repository holds three pages that run the **same model** three different
ways, so the cost of the machinery around the arithmetic can be isolated:

| Page | Implementation | Needs |
| --- | --- | --- |
| [`index.html`](index.html) | LiteRT.js with a CPU (`wasm`) / GPU (`webgpu`) switcher | any browser |
| [`webgpu.html`](webgpu.html) | the `.tflite` parsed in JS, hand-written WGSL; fused vs per-layer × fence-poll on vs off | WebGPU |
| [`browser-model-api.html`](browser-model-api.html) | `navigator.digitclassifier`, a Web API compiled into a custom Chromium build | that build |

## Results

Seven cases, one drawn digit, measured together so they can be compared with each
other. **OnePlus CPH2653**, **custom Chromium 153.0.8005.0 release**, content
from the Pages site over HTTPS, clocks fixed, browser force-stopped between
pages, `?timing_runs=1000`, **3 sessions × 5 samples** (n = 15 per row), median
reported.

| Case | Page | Median | CV | vs CPU |
| --- | --- | ---: | ---: | ---: |
| **LiteRT.js — CPU (`wasm`)** | `index.html` | **0.090 ms** | **0.0%** | 1.00× |
| Direct WebGPU — **fused · fence poll** | `webgpu.html` | **0.480 ms** | 7.9% | **5.3×** |
| Direct WebGPU — **per-layer · fence poll** | `webgpu.html` | **0.510 ms** | 43.1%¹ | 5.7× |
| LiteRT.js — GPU (`webgpu`) | `index.html` | **1.410 ms** | 3.8% | 15.7× |
| Direct WebGPU — **fused · no poll** | `webgpu.html` | **2.440 ms** | 1.7% | 27.1× |
| Direct WebGPU — **per-layer · no poll** | `webgpu.html` | **2.540 ms** | 1.5% | 28.2× |
| **`navigator.digitclassifier`** (C++ in Blink) | `browser-model-api.html` | **3.710 ms** | 1.4% | 41.2× |

¹ One sample of 1.46 ms among fourteen between 0.44 and 0.62 — the fence poll's
tail, which [does not average out](docs/findings.md#the-four-cases-and-what-the-floor-actually-is).
Without it, 10.7%. The median is unaffected.

**The CPU is 5.3× faster than the best GPU path** — the narrowest margin this
repository has measured, and the same verdict every earlier measurement reached.
Three things behind the ordering:

- **The browser-built-in model is the slowest GPU path**, at 3.710 ms. Removing
  JavaScript, the runtime and the download did not remove the cost; it removed
  the *workaround*. The page beside it reaches 0.480 ms only by hurrying Chrome's
  GPU process along — which the C++ path, for all its leanness, does not do.
- **Most of what looked like a GPU round-trip was Chrome checking its completion
  fence lazily.** Polling `onSubmittedWorkDone` while the readback is pending is
  worth ~5× on the identical computation, bit-identical output: 2.440 → 0.480 ms.
  Both fence modes are a switch on the page, so the table above is an in-page
  A/B rather than a comparison across builds.
- **A dispatch costs ~0.05 ms**, from the unpolled pair (+0.100 ms for two extra
  dispatches, at CV 1.5–1.7%). Dispatch was never what made the GPU slow here.

Raw data and the statistics:
[seven-case run](docs/measurements/2026-08-25-custom-chromium-seven-case.md).

> [!IMPORTANT]
> **This table is a custom Chromium build, and you cannot install it.** It is the
> only browser that has `navigator.digitclassifier` at all, so it is the only one
> that can measure all seven cases side by side. It is also three milestones
> newer than the stock Chrome on the same phone and is not an official build, so
> **none of these figures may be compared with a stock-Chrome figure.** The stock
> measurements — which is where the reproducible-on-your-own-device numbers live,
> and where most of the reasoning was done — are in
> [**Findings**](docs/findings.md), kept in their own tables and never pooled
> with these.

> [!NOTE]
> Much of [`docs/findings.md`](docs/findings.md) was measured with an **older,
> budget-driven harness**, most of it at a 5 ms budget where the GPU rows carried
> CVs of 45–55%. Those sections are kept as measured — they carry the reasoning
> that produced the findings — but figures from the two harnesses must not be
> pooled, and where an old number disagrees with a fixed-count one, the
> fixed-count one is current.

Three experiments produced the table above, and each reversed a conclusion of the
one before. The full account is in [**Findings**](docs/findings.md): the
[first measurements](docs/findings.md#performance-report), the
[reason the GPU loses](docs/findings.md#finding-the-gpu-is-slower-than-the-cpu),
the [model inside the browser](docs/findings.md#third-experiment-a-model-built-into-the-browser),
and the [method](docs/findings.md#the-experiment).

## Running it

It is a static site — no build step. Serve the repository root over HTTP by any
means; the two below are what this project uses.

**Locally:**

```bash
npm install      # fetches @litertjs/core - node_modules is not committed
npm start        # → http://localhost:8080
```

**From GitHub Pages:** push the repo, then *Settings → Pages → Source: Deploy
from a branch → `main` / `/ (root)`*. Every path in `index.html` is relative, so
it works unmodified under the `/<repo>/` subpath, and Pages serves it over HTTPS
— which WebGPU requires outside `localhost`. `node_modules` is not committed, so
the import map's local paths 404 and the page falls back to loading LiteRT.js and
its Wasm from jsDelivr; verified working, at the cost of ~1 s of runtime init
against ~0.15 s locally. `.nojekyll` keeps Pages from running the files through
Jekyll.

`npm install` is optional — `serve.js` itself needs no dependencies, and without
the local packages the page uses the same CDN fallback that GitHub Pages does.

A server is required. Opening `index.html` as a `file://` URL leaves the page on
the opaque `null` origin, where the browser blocks fetching both the ES module
from `node_modules` and the `.tflite` file. Drawing still works from `file://`;
classification cannot. `serve.js` is a zero-dependency static server (Node
built-ins only).

## How it works

| Stage | What happens |
| --- | --- |
| Draw | Pointer events on a 280×280 canvas (mouse, touch and stylus in one path) |
| Fit | The canvas is CSS-scaled to the viewport; pointer coordinates are mapped back to bitmap space |
| Downsample | Fill a 28×28 canvas black, then `drawImage` the 280×280 onto it |
| Encode | Grayscale, normalize to `[0,1]`, replicate per channel to match the model input |
| Infer | `new Tensor(buf, shape)` → `model.run(tensor)` → `outputs[0].data()` |
| Report | Argmax over 10 probabilities, plus per-run timing |

The model declares its input as **`[1, 28, 28, 3]`** — 2352 RGB floats, *not* the
784 grayscale values MNIST examples usually assume. The code reads the shape from
`getInputDetails()` rather than hard-coding it, so it adapts if the model changes.

| | |
| --- | --- |
| Signature | `serving_default` |
| Input | `flatten_input`, `[1,28,28,3]`, float32 |
| Output | `dense_1`, `[1,10]`, float32 (already softmaxed) |
| Layers | `2352 → 128` (ReLU) `→ 10` → softmax |
| Size | 1.2 MB, 302,474 trainable float32 parameters |

## Backend switcher

A segmented control picks which LiteRT.js accelerator runs inference:

- **CPU (wasm)** — XNNPack kernels compiled to WebAssembly. Not "no
  acceleration": relaxed-SIMD vector kernels.
- **GPU (webgpu)** — compute shaders via the browser's WebGPU API.

Switching recompiles immediately and discards the previous `CompiledModel`. The
model bytes are fetched once at startup and reused for every compile, so a switch
never touches the network. The `GPUDevice` is acquired lazily and only once, then
reused. Each compile is
followed by a throwaway inference, so the first timing shown is never dominated
by one-time warm-up. A failed switch recompiles the previously working backend
rather than leaving the page with no model; with no adapter present, the GPU
button is disabled with the reason in its `title`.

## Reproducing the numbers

Mobile clocks scale with load, so an unpinned device spans a 9.2× range on the
same workload. The protocol that makes measurements reproduce — fixed
performance mode over adb, a warm-up burst, medians rather than means, and a run
count raised until the median stops moving — is in
[Getting numbers that reproduce](docs/findings.md#getting-numbers-that-reproduce).
All three pages accept `?timing_runs=` to override the default of 100 runs per
measurement; **CPU paths need `?timing_runs=1000`** or they read ~56% too slow.
`webgpu.html` also takes `?mode=layered` and `?fence_poll=0`, which between them
address the [four cases](docs/findings.md#the-round-trip-that-wasnt-chromes-lazy-fence)
that page can be measured in — one URL each, nothing to click.

## Documentation

| Document | What it covers |
| --- | --- |
| [`docs/findings.md`](docs/findings.md) | **The findings**: three experiments, every measurement and the reasoning, including the corrections |
| [`docs/architecture.md`](docs/architecture.md) | The API stack each page uses, diagrammed, plus an a-priori prediction written *before* the measurements |
| [`wgsl-explainer.md`](wgsl-explainer.md) | How the hand-written WGSL kernels in `webgpu.html` were derived from the model, line by line |
| [`docs/measurements/`](docs/measurements/) | Raw data behind every published number, one file per session, each stating device, browser and protocol |
| [`docs/superpowers/`](docs/superpowers/) | The design spec and implementation plans for the direct-WebGPU experiment |
| [`CLAUDE.md`](CLAUDE.md) | Guidance for Claude Code: the rules that keep the three pages comparable |

## Repository layout

| File | Purpose |
| --- | --- |
| `index.html` | First experiment: LiteRT.js, drawing, preprocessing, backend switcher, inference |
| `webgpu.html` | Second experiment: the same model driven directly against the WebGPU API |
| `browser-model-api.html` | Third experiment: the same model behind `navigator.digitclassifier`, a browser-built-in API |
| `digit_classifier.tflite` | The model — 1.2 MB, 302,474 parameters |
| `serve.js` | Zero-dependency static server |
| `screenshot.png` | The image at the top, captured on the Android device |
| `.nojekyll` | Stops GitHub Pages running the site through Jekyll |

## License

[Apache-2.0](LICENSE). `digit_classifier.tflite` carries Apache-2.0 in its own
TFLite metadata, and [LiteRT.js](https://github.com/google-ai-edge/LiteRT) is
Apache-2.0 as well.
