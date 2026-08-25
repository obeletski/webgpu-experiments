# CPU vs GPU Inference in the Browser

**A [LiteRT.js](https://ai.google.dev/edge/litert/web) benchmark: WebAssembly
against WebGPU, measured on a hand-drawn MNIST digit classifier.** Draw a digit,
classify it, then flip the backend and run the same drawing through the other
one. Every figure the UI shows — compile, cold start, steady state — is part of
the comparison.

The classifier is the vehicle, not the subject: it exists to give both backends
identical work so their cost can be compared. **On every device tested the CPU
wins**, because a 0.6 MFLOP inference is far too small to pay for a GPU
round-trip.

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
| [`webgpu.html`](webgpu.html) | the `.tflite` parsed in JS, hand-written WGSL, fused vs per-layer | WebGPU |
| [`browser-model-api.html`](browser-model-api.html) | `navigator.digitclassifier`, a Web API compiled into a custom Chromium build | that build |

## Results

**OnePlus CPH2653**, **stock Chrome 150.0.7871.188**, content served from the
Pages site over HTTPS, clocks fixed, Chrome force-stopped between pages, 3
sessions × 5 samples each. Every figure below is taken with the **fixed-count
harness at a run count high enough that the median has stopped moving** — see
[how many runs a measurement needs](docs/findings.md#the-cpu-baseline-and-how-many-runs-a-measurement-needs).

| Implementation | Page | Median | CV | vs CPU |
| --- | --- | ---: | ---: | ---: |
| **LiteRT.js — CPU (`wasm`)** | `index.html` | **0.090 ms** | **0.0%** | 1.00× |
| LiteRT.js — GPU (`webgpu`) | `index.html` | **1.610 ms** | 2.1% | **17.9×** |
| Direct WebGPU — fused | `webgpu.html` | **3.270 ms** | 1.9% | **36.3×** |

**The CPU is 17.9× faster than the best GPU path.** Hand-writing the WebGPU path
does not help: it is 2.03× slower than the runtime it was written to beat.

> [!NOTE]
> The **3.270 ms** direct-WebGPU figure predates two fixes, and the second one
> reverses the sentence above. After
> [tracing why LiteRT is faster](docs/findings.md#traced-why-litertjs-is-faster),
> `webgpu.html`'s `run()` issues the compute and the readback as two separate
> submits (**~2.75 ms** fused,
> [data](docs/measurements/2026-08-24-webgpu-two-submit.md)) — and then the
> remaining gap turned out not to be the round-trip at all but **Chrome's lazy
> fence servicing**: polling `onSubmittedWorkDone` while the `mapAsync` is
> pending measured fused at **~0.38 ms** and per-layer at **~0.41 ms** across two
> sessions, past LiteRT by ~4×
> ([data](docs/measurements/2026-08-24-webgpu-mapasync-poll.md)). Output stays
> bit-identical; nothing is pipelined — each call still waits for its own result.
> The table keeps the rigorously-sampled 3.270 ms until the fix is re-measured
> from the Pages site with the full 3-session protocol; when it is, the
> hand-written page becomes the fastest GPU path and the CPU's margin over the
> GPU narrows from 17.9× to ~4×.

Raw data:
[CPU baseline and run-count sweep](docs/measurements/2026-08-24-stock-chrome-cpu-baseline.md)
·
[LiteRT vs direct WebGPU](docs/measurements/2026-08-24-stock-chrome-litert-vs-webgpu.md).

> [!NOTE]
> Much of [`docs/findings.md`](docs/findings.md) was measured with an **older,
> budget-driven harness**, most of it at a 5 ms budget where the GPU rows
> carried CVs of 45–55%. Those sections are kept as measured — they carry the
> reasoning that produced the findings — but where a number there disagrees with
> the table above, **the table above is the current one.** Figures from the two
> harnesses must not be pooled.

### Two browsers, deliberately kept apart

Both are installed on the same OnePlus CPH2653, and **their numbers are never
mixed in one table**:

| | Browser | Used by |
| --- | --- | --- |
| **Stock** | Chrome **150.0.7871.188** (`com.android.chrome`) | the [performance report](docs/findings.md#performance-report), the [finding](docs/findings.md#finding-the-gpu-is-slower-than-the-cpu), the [second experiment](docs/findings.md#second-experiment-direct-webgpu) |
| **Custom** | locally built Chromium **153.0.8005.0** release (`org.chromium.chrome`) | the [third experiment](docs/findings.md#third-experiment-a-model-built-into-the-browser) |

The custom build is the only one that can run `browser-model-api.html` at all —
`navigator.digitclassifier` does not exist in a shipping browser. It is three
milestones newer than the stock Chrome, and not an official build, so **a figure
from one section cannot be compared with a figure from the other.** Comparisons
within a section are sound; across sections they are not, and where the two
disagree that is called out rather than reconciled.

Three experiments went into that table, and two of them reversed a conclusion
of the one before — including the ~3 ms "GPU round-trip" that turned out to be
Chrome checking its completion fence lazily. The full account is in
[**Findings**](docs/findings.md): the
[numbers](docs/findings.md#performance-report), the
[reason](docs/findings.md#finding-the-gpu-is-slower-than-the-cpu) and the
[method](docs/findings.md#the-experiment).

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
