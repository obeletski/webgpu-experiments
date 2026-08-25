# Fence-poll, three sessions: the median reproduces, the spread does not settle

**Date** 2026-08-24 · **Device** OnePlus 13 (`CPH2653`, Snapdragon 8 Elite,
Adreno 830, Android 16 / API 36) · **Browser** **custom Chromium 153.0.8005.0
release** (`org.chromium.chrome`) — *not* the stock Chrome 150 that most of
[`findings.md`](../findings.md) uses · **Harness** fixed-count, 3 sessions × 5
samples per path (n = 15 each), median reported · **Clocks**
`set-fixed-performance-mode-enabled true`, restored after. Raw samples:
[`2026-08-24-fence-poll-3session.json`](2026-08-24-fence-poll-3session.json).

> [!IMPORTANT]
> **These are custom-build figures and must never be pooled with a stock-Chrome
> table**, including the README's results table. The custom build is three
> milestones newer and is not an official build. The comparison *within* this
> table is sound; a row of it against a row of any stock-Chrome table is not.

> [!NOTE]
> **The run count and the content source were not recorded with the data**, and
> the JSON carries no header — this file was written afterwards from the raw
> samples, the commit that added them, and the session it was taken in. The
> sample count (15 per path) is the only part of the protocol the data itself
> establishes. Treat the medians as reproducible and the protocol line as
> partial; anything that hinges on the run count needs a fresh measurement.

Re-measures the [fence-poll fix](2026-08-24-webgpu-mapasync-poll.md) — `run()`
asking for a fresh `onSubmittedWorkDone()` signal from each new macrotask while
the `mapAsync` is pending — across three sessions rather than the two the
original probe used.

## Headline

| Path | Page | Median | Mean | SD | CV | Range |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **LiteRT.js — CPU (`wasm`)** | `index.html` | **0.06 ms** | 0.065 | 0.010 | 14.8% | 0.06 – 0.09 |
| **Direct WebGPU — per-layer (fence-poll)** | `webgpu.html` | **0.40 ms** | 0.551 | 0.308 | **56.0%** | 0.37 – 1.21 |
| **Direct WebGPU — fused (fence-poll)** | `webgpu.html` | **0.46 ms** | 0.515 | 0.232 | **45.0%** | 0.33 – 1.07 |
| LiteRT.js — GPU (`webgpu`) | `index.html` | **1.57 ms** | 1.575 | 0.054 | 3.4% | 1.49 – 1.71 |

**The median reproduces.** The two-session probe measured 0.38 ms fused and
0.41 ms per-layer; three sessions land at 0.46 and 0.40. The fence-poll page is
the fastest GPU path measured on this build — ~3.4× past LiteRT.js's 1.57 ms —
and the CPU is still ahead of it, by ~7×.

**The spread does not.** Fused CV 45%, per-layer CV 56%, and **every session
contributes a ~1 ms outlier** (1.07, 1.07 fused; 1.08, 1.20, 1.21 per-layer)
against a 0.33 – 0.47 body. That is the poll's own mechanism showing through: it
waits on task scheduling, so a sample lands late whenever the event loop is busy.
Compare LiteRT's GPU row at CV 3.4% — it pays a larger, steadier cost.

By median, per-layer (0.40 ms) reads *faster* than fused (0.46 ms), reversing the
dispatch-cost ordering every earlier measurement found. At these CVs that is
noise, not a finding: the two distributions overlap almost completely.

## What it does and does not establish

- **Establishes:** the fence-poll speed-up is real and repeats across sessions;
  it is not an artefact of the one session that first measured it.
- **Establishes:** the poll's cost distribution is heavy-tailed and
  scheduling-dependent, so a single sample from it means little. Report medians
  from many sessions, never a mean.
- **Does not establish** anything about stock Chrome. This is the custom build;
  the stock-Chrome fence-poll figures are in
  [`2026-08-24-webgpu-mapasync-poll.md`](2026-08-24-webgpu-mapasync-poll.md),
  taken over localhost, and the README's table still awaits a stock-Chrome
  re-measure served from the Pages site.
- **Is a different session** from
  [`2026-08-25-custom-chromium-four-way.md`](2026-08-25-custom-chromium-four-way.md),
  which measured the same build the next day at `timing_runs=1000` and put fused
  at 0.520 ms (CV 32.7%) with LiteRT GPU at 1.430 ms. Two sessions of the same
  build agreeing to within ~0.06 ms on a number this noisy is the reassuring
  part; the difference between them is inside the tail described above.

Cited from [`../architecture.md`](../architecture.md) §7, which diagrams why the
poll works.
