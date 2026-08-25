# Polling the fence: the ~3.3 ms "round-trip" was mostly Chrome being lazy

**Date** 2026-08-24 · **Device** OnePlus 13 (`CPH2653`, Snapdragon 8 Elite,
Adreno 830, Android 16 / API 36) · **Browser** **stock Chrome 150.0.7871.188**
(`com.android.chrome`) · **Content** local files over `adb reverse tcp:8080`
(uncommitted change) · **Clocks** `set-fixed-performance-mode-enabled true`,
restored after · **Protocol** driven over the DevTools protocol (`adb forward
tcp:9222`, `Runtime.evaluate`); identical injected "1" stroke; Chrome
force-stopped between pages; `?timing_runs=1000`; each page figure the median of
5 samples. Raw data:
[`2026-08-24-webgpu-mapasync-poll.json`](2026-08-24-webgpu-mapasync-poll.json).

Follows the [two-submit change](2026-08-24-webgpu-two-submit.md), which left the
hand-written page at ~2.75 ms fused against LiteRT's ~1.6 ms and called the rest
"one full CPU↔GPU round-trip" a self-contained `run()` cannot hide. This session
set out to close that gap and found the premise wrong: **most of the remaining
~2.5 ms was not the round-trip. It was Chrome not checking whether the GPU had
finished.**

## The mechanism

A pending `mapAsync` resolves only when the GPU process notices the copy's fence
has signalled, and on Android Chrome that servicing is lazy — left alone it takes
~2.5 ms to notice work that finished in microseconds. Nothing in the page keeps
the device ticking between submits, so the map's resolution waits on whatever
internal schedule Chrome has. The [pipelined 0.48 ms](2026-08-24-litert-vs-handwritten-tracing.md)
figure was the tell: submits arriving while a map is pending make it resolve
almost immediately, which cannot be true of a physical round-trip but is exactly
how a lazily-polled fence behaves.

The fix: after submitting compute and readback and calling `mapAsync`, poll —
from each fresh macrotask, ask for a new `onSubmittedWorkDone()` signal (a fence
query, forcing the GPU process to check completion) until the map settles. A
`MessageChannel` message is the yield; `setTimeout(0)` clamps to ≥1 ms, which
would put the poll cadence above the thing being polled.

## Probe: what forces the fence check (and what does not)

Scratch page (`bench-roundtrip.html`, deleted after) with the same fused
MNIST-shaped workload and interchangeable `run()` structures, every variant
returning *its own* input's result — no one-behind pipelining anywhere. 3 rounds
× 1000 runs each, medians:

| Variant | ms | |
| --- | ---: | --- |
| one submit, block on `mapAsync` (pre-fix `run()`) | 3.31 | matches the published 3.26 |
| two submits, block (`run()` after the two-submit fix) | 2.69 | matches the published ~2.75 |
| two submits + task boundary between them | 2.61 | the LiteRT-style task split alone buys little |
| two submits + **`onSubmittedWorkDone` poll while map pending** | **0.39** | **the fix** |
| one submit + the same poll | 0.63 | works too; two submits are still slightly better |
| two submits + a single `onSubmittedWorkDone` next to the submits | 2.60 | rides the same flush, changes nothing |
| two submits + `await` the workDone promise alongside the map | 2.72 | same story |
| two submits + poll with **empty `queue.submit([])`** instead | **24.5** | a submit is heavyweight; 9× *worse* |
| two submits + task churn with **no** GPU pokes (control) | 5.92 | yielding alone makes it worse, not better |
| two submits, no input `writeBuffer` | 2.55 | upload costs ~0.1 ms; not the lever |

So the poke must be (a) a fence query, not a submit, and (b) issued from later
tasks, repeatedly, while the map is pending. All probe variants returned
bit-identical vectors on the device (`check()` compares all against the serial
baseline).

## The real page, paired A/B

`webgpu.html` with the poll in `run()` against the committed two-submit build
served side by side, same session block, Chrome force-stopped between pages,
5 × 1000-run samples per mode:

| | fused | per-layer |
| --- | ---: | ---: |
| two-submit baseline (committed) | **2.72 ms** | **2.81 ms** |
| with `onSubmittedWorkDone` poll | **0.38 ms** | **0.41 ms** |

A second fresh session of the fixed page repeated fused **0.39** / per-layer
**0.38 ms**. Against the published table: fused **3.270 → ~0.4 ms** (~8×), and
**past LiteRT's 1.610 ms by ~4×**. The dispatch-count delta the page exists to
measure survives, shrunk to ~0.03 ms — consistent with the trace's finding that
dispatch was never the cost.

**Run-count check**: fused at `?timing_runs=5000` gave 0.49, 0.50, 0.36 ms —
the same band as 1000, so 1000 is on the plateau. **Spread caveat**: samples now
range 0.33–0.76 ms where the old figures held ±2%; the poll's latency depends on
task scheduling, so the tail is fatter. The medians are stable across sessions.

## Correctness — bit-identical

- **Desktop** (headless Chrome 135, SwiftShader WebGPU): the fixed synthetic
  vector (`((i*2654435761)>>>0)%1000/1000`) through `window.__engine.run` returns
  the **exact same float32 vector** for the fixed and baseline builds, in **both**
  modes, and fused equals per-layer within a build.
- **Phone**: the identical injected "1" stroke predicts **1 at 100%** on both
  builds, both modes, every sample.

The poll cannot change the result: `mapAsync` still resolves only after the copy
completes; polling changes *when Chrome notices*, not what is in the buffer. The
loop exits on device loss because `mapAsync` settles either way.

## What this revises

- "The ~3.3 ms is the cost of waiting for the GPU to answer" — true only in the
  sense that the *waiting* was the cost; the waiting itself was mostly slack. The
  irreducible submit + compute + copy + map-and-notice cost on this device is
  **~0.4 ms**.
- The trace's claim that a single one-shot inference must pay ~3.3 ms falls with
  it: the poll is not pipelining — it accelerates each call *individually*, one-shot
  included. `onSubmittedWorkDone ≈ mapAsync` at ~3.3 ms remains true, but both were
  measuring the servicing schedule, not the round-trip.
- The CPU (0.090 ms) still beats the best GPU path — by ~4×, no longer 17.9×.
  The thesis narrows; it does not flip.

## Method note

Localhost serving is the documented uncommitted-code flow; the published-site
3-session protocol should re-measure before the README's [results
table](../../README.md#results) is rewritten. The probe page and the CDP drivers were scratch files, deleted
after the run.
