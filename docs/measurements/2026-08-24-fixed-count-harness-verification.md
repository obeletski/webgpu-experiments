# Fixed-count harness: functional verification

**Date** 2026-08-24 · **Host** desktop, 11th Gen Intel Core i7-11850H (16
threads), 16 GB, WSL2 on Windows · **Browser** stock Chrome **151.0.7922.170**
(Windows, `--headless=new`) · **GPU adapter** **SwiftShader — software** ·
**Content** served over plain HTTP from `localhost` out of the git working tree.

> [!CAUTION]
> **This is not a benchmark and none of these timings mean anything.** The GPU
> figures come from a *software* adapter, the host is a laptop with no clock
> pinning and no warm-up burst, and each configuration was sampled 2–3 times.
> They exist to show the rewritten harness runs and reports what it should.
>
> This is also a **third browser** — desktop Chrome 151 — distinct from both the
> OnePlus's stock Chrome 150 and the custom Chromium 153. Per the repository's
> rule, **nothing here may be placed in a table with figures from either of
> them.**

## What was being verified

The commit that replaced the harness's `TIMING_MIN_MS` time budget with a fixed
`TIMING_RUNS` count (default 100). Three claims:

1. All three pages still load, preprocess and classify.
2. Each measurement reports the mean of exactly 100 internal runs.
3. `?timing_runs=` overrides the count.

## Method

A scratch driver page loaded each target in a same-origin iframe, waited for the
model, synthesised pointer events to draw a vertical stroke (a `1`), clicked
*Classify*, and POSTed the result line to a collector on `localhost:8097` — the
method described in the [findings doc's harness](../findings.md#the-harness)
section.
The driver and collector were scratch files and are not committed.

`node_modules` was absent, so `index.html` took its **CDN fallback** for
LiteRT.js and its wasm (`esm.run` / `jsdelivr`), the same path GitHub Pages
uses. `webgpu.html` fetched nothing external. WebGPU initialised because
`localhost` is a secure context.

Chrome flags, per the [headless GPU
note](../findings.md#getting-a-gpu-in-headless-chrome):

```
--headless=new --no-sandbox --enable-unsafe-webgpu
--use-webgpu-adapter=swiftshader --enable-features=Vulkan,WebGPUService
```

## Raw results

Every line is the page's own result text, verbatim. No console errors and no
unhandled rejections were recorded on any run.

| Page | Path | `timing_runs` | Reported |
| --- | --- | ---: | --- |
| `index.html` | CPU (`wasm`) | 100 | `0.13` · `0.11` · `0.08` ms (mean of 100) |
| `index.html` | GPU (`webgpu`, SwiftShader) | 100 | `3.10` · `2.97` · `2.90` ms (mean of 100) |
| `webgpu.html` | fused, run A | 100 | `4.26` · `3.83` · `3.66` ms (mean of 100) |
| `webgpu.html` | fused, run B | 100 | `4.24` · `3.74` ms (mean of 100) |
| `webgpu.html` | fused, override | 7 | `6.19` · `5.27` ms (**mean of 7**) |

One-time costs, one observation each:

| Page | Path | Compile / pipeline | Cold start |
| --- | --- | --- | --- |
| `index.html` | CPU (`wasm`) | 3.90 ms | 1.30 ms |
| `index.html` | GPU (`webgpu`) | 70.60 ms | 139.80 ms |
| `webgpu.html` | fused | `<1 ms` | 44.30 / 48.70 / 85.10 ms |

Every run predicted **`1`** at **100.0%** match, which is the expected answer for
the synthesised stroke and matches what the same drawing produces elsewhere.

`browser-model-api.html` **could not be exercised** — this is stock Chrome, not
the custom build. It reported the expected explanation rather than failing
silently:

> navigator.digitclassifier is missing. It ships only in a Chromium build
> carrying the DigitClassifier module, and it is `status: "test"` in
> `runtime_enabled_features.json5`, so it stays off unless the browser is started
> with `--enable-blink-features=DigitClassifier`.

## The one observation worth keeping

The CPU means multiply back to **exact integers**: `0.13 × 100 = 13 ms`,
`0.11 × 100 = 11 ms`, `0.08 × 100 = 8 ms`. Elapsed time landing on whole
milliseconds across every sample is direct confirmation that `performance.now()`
is clamped to 1 ms steps here, which is the premise the whole harness is built on — and it puts a number on the cost: at ~10 ms of elapsed time the CPU figure
is quantised to roughly **8–12%**, coarser than the ~7% estimated for the phone,
because this host's CPU is faster than the OnePlus's and so buys less elapsed
time from the same 100 runs.

That is an argument for a higher count on fast CPU paths, and it is the thing to
re-check first when the pages are next measured properly.

## What this does *not* establish

- **No performance conclusion whatsoever.** Software rasterisation, unpinned
  clocks, n≤3. The CPU-beats-GPU comparison in particular cannot be drawn from
  these rows, because the "GPU" here is a CPU.
- **`browser-model-api.html`'s inference path is unverified.** Its timing block
  is byte-identical to the two pages that were exercised — the shared region
  hashes the same across all three files — but that is an argument, not a test.
- **The repository still has no figures from this harness.** Every number in the
  README was taken under the old time budget. A re-measurement on the OnePlus
  under the fixed-clock protocol is still outstanding, and should check whether
  the median moves again between 100 and 200 runs.
