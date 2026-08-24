# LiteRT.js GPU against direct WebGPU, stock Chrome, fixed-count harness

**Date** 2026-08-24 · **Device** OnePlus 13 (`CPH2653`, Snapdragon 8 Elite,
Adreno 830, Android 16 / API 36) · **Browser** **stock Chrome 150.0.7871.188**
(`com.android.chrome`) · **Content** served from
<https://obeletski.github.io/webgpu-experiments/> over HTTPS · **Harness** fixed
`TIMING_RUNS = 100` (no URL overrides) · **Clocks** `set-fixed-performance-mode-enabled true`
for the whole session, restored afterwards.

> [!NOTE]
> This is stock Chrome, the same browser as the [performance
> report](../../README.md#performance-report) and the [second
> experiment](../../README.md#second-experiment-direct-webgpu) — **not** the
> custom Chromium 153 of the
> [three-way comparison](2026-08-23-custom-chromium-three-way.md).
>
> It is however the **first** measurement taken with the fixed-count harness.
> Every earlier figure in this repository used the old time budget, so these
> numbers may be compared with each other but **not** pooled into a table with
> anything measured before 2026-08-24.

## Result

**LiteRT.js on WebGPU is 1.93× faster than the hand-written WebGPU page.**

| Implementation | Page | n | Median | Mean | SD | CV | Range | Session medians |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| **LiteRT.js — GPU (webgpu)** | `index.html` | 15 | **1.68 ms** | 1.67 | 0.123 | **7.4%** | 1.46 – 1.91 | 1.78 / 1.69 / 1.56 |
| **Direct WebGPU — fused** | `webgpu.html` | 15 | **3.24 ms** | 3.17 | 0.206 | **6.5%** | 2.80 – 3.42 | 3.27 / 3.17 / 3.32 |

**The distributions do not overlap.** LiteRT's slowest sample (1.91 ms) sits
below the hand-written page's fastest (2.80 ms). U = 225 of a possible 225;
permutation two-sided p = 0.00004 over 200,000 resamples.

Every one of the 30 samples predicted **`1`** at 100.0% match, and every one
reports **`mean of 100`** — the fixed count is in force on both pages.

### One-time costs

One observation per session, listed raw.

| Page | Compile / pipeline | Cold start |
| --- | --- | --- |
| `index.html` (GPU) | 93.50 / 94.70 / 87.10 ms | 62.30 / 12.30 / 13.40 ms |
| `webgpu.html` (fused) | `<1 ms` / `<1 ms` / 0.10 ms | 7.90 / 32.60 / 26.10 ms |

The hand-written page still builds its pipeline in under a millisecond against
LiteRT's ~90 ms compile. That advantage is unchanged and is not what this
measurement is about; it is paid once, while the 1.56 ms difference below is paid
per inference.

## What this settles

The README's second experiment reported that removing LiteRT.js **bought 1.34×**,
and the [three-way re-measurement](2026-08-23-custom-chromium-three-way.md) then
found the opposite ordering on custom Chromium 153, leaving the question open:

> It needs re-measuring at a 50 ms budget on stock Chrome before it can be
> trusted, and only then will it be clear whether the inversion is a browser
> difference or was never there.

**It was never there.** The inversion reproduces on the same stock Chrome that
produced the 1.34×, and by a wider margin than on the custom build (1.93× here
against 1.73× there). The original figure was an artefact of under-sampling: at
the old 5 ms budget a 3 ms path was averaged over one to three internal runs, and
LiteRT's GPU row carried a CV of 49.1%. At 100 runs both CVs fall to 6–7% and the
ordering is unambiguous.

**This does not disturb the repository's main thesis.** The CPU remains far ahead
of every GPU path; what changes is the ranking *within* the GPU paths. The
hand-written page is not faster than the runtime it was written to beat.

## Method

Per the README's [protocol](../../README.md#getting-numbers-that-reproduce):
fixed-performance mode on for the whole session and restored at the end, Chrome
**force-stopped between every page load**, a 4-second warm-up burst of Classify
activations before any sample, and the same stroke drawn on every page.

**3 sessions × 5 samples per page**, `webgpu.html` first in each session, giving
15 samples per page. Session medians are listed above so between-session drift
can be seen; it is small (1.56–1.78 and 3.17–3.32).

The warm-up burst is a fixed 4 seconds of wall time, so it buys a different number
of classifications per page — **~21 on `index.html` against ~12 on
`webgpu.html`** — because each classification is now 100 inferences and LiteRT's
are faster. Both are far past the point where warm-up matters.

### Deviation from the documented protocol

The README drives the UI with `input swipe` / `input tap` and reads results from
`screencap`, because `console.log()` does not reach logcat. This session instead
attached to Chrome's DevTools socket over `adb forward tcp:9222
localabstract:chrome_devtools_remote` and drove the page from there: pointer
events were synthesised onto the canvas and the result line was read out of the
DOM.

This is a **readout** change, not a timing change — the harness runs inside the
page either way, and nothing in the timed region involves the driver. It removes
OCR from the loop and guarantees the drawn stroke is identical across pages
rather than merely aiming at the same screen coordinates. The driver was a
scratch file and is not committed.

## Limitations

- **One device, one browser, one session block.** No claim beyond a OnePlus
  CPH2653 on stock Chrome 150.
- **No CPU baseline was taken in this session.** The two GPU paths can be
  compared with each other, but neither can be anchored against `wasm` without
  re-running `index.html` on its CPU backend under the same harness.
- **`webgpu.html`'s per-layer mode was not measured** — only fused, which is the
  faster of the two.
- **Why LiteRT.js is faster remains unexplained.** The
  [hypothesis](../../README.md#re-measured-the-ordering-really-does-invert) about
  per-readback buffer allocation is still untested; this measurement only
  establishes that the difference is real and reproducible on a second browser.
