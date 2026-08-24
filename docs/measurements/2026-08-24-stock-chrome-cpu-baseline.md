# CPU baseline, and how many runs a measurement actually needs

**Date** 2026-08-24 · **Device** OnePlus 13 (`CPH2653`, Snapdragon 8 Elite,
Adreno 830, Android 16 / API 36) · **Browser** **stock Chrome 150.0.7871.188**
(`com.android.chrome`) · **Content** <https://obeletski.github.io/webgpu-experiments/index.html>
over HTTPS · **Clocks** `set-fixed-performance-mode-enabled true` for every
session, restored after each · **Protocol** Chrome force-stopped between
sessions, 4 s warm-up burst, 3 sessions × 5 samples per configuration, **GPU
measured before CPU in every session** as in the README's original protocol.

Companion to the [LiteRT-vs-direct-WebGPU
measurement](2026-08-24-stock-chrome-litert-vs-webgpu.md), which had no CPU
baseline. Same device, browser, harness and protocol; a separate session block.

## Headline

| Backend | Settled median | n | CV | Runs needed to settle |
| --- | ---: | ---: | ---: | --- |
| **CPU (`wasm`)** | **0.090 ms** | 15 | **0.0%** | **1000** |
| **GPU (`webgpu`)** | **1.610 ms** | 15 | **2.1%** | 100 |

**The GPU costs 17.9× the CPU** on a properly sampled measurement — a wider
margin than the 7.7× the README reports from the old 5 ms harness.

## The count matters more than expected

Each configuration was measured at 100, 1000 and (for the CPU) 5000 internal runs
per sample, by loading the page with `?timing_runs=`.

| Backend | `timing_runs` | Median | Mean | SD | CV | Range | Session medians |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| GPU (`webgpu`) | 100 | 1.600 ms | 1.604 | 0.0990 | 6.2% | 1.38 – 1.73 | 1.580 / 1.590 / 1.600 |
| GPU (`webgpu`) | 1000 | **1.610 ms** | 1.611 | 0.0344 | **2.1%** | 1.54 – 1.66 | 1.640 / 1.610 / 1.580 |
| CPU (`wasm`) | 100 | 0.140 ms | 0.141 | 0.0409 | 28.9% | 0.09 – 0.24 | 0.140 / 0.110 / 0.180 |
| CPU (`wasm`) | 1000 | **0.090 ms** | 0.090 | 0.0000 | **0.0%** | 0.09 – 0.09 | 0.090 / 0.090 / 0.090 |
| CPU (`wasm`) | 5000 | **0.090 ms** | 0.090 | 0.0000 | **0.0%** | 0.09 – 0.09 | 0.090 / 0.090 / 0.090 |

> [!IMPORTANT]
> **The default of 100 runs is enough for the GPU and not enough for the CPU.**
> The GPU median moves 1.600 → 1.610 ms between 100 and 1000 runs — **+0.6%**,
> i.e. it is already on the plateau. The CPU median moves 0.140 → 0.090 ms —
> **−36%** — and only then stops, holding at exactly 0.090 through 5000 runs.
>
> At 100 runs the CPU figure is **56% too high**.

### Why, and why it is not quantisation

Quantisation was the obvious suspect and it is not the answer. It is real, and
visible: at 100 runs every CPU sample multiplied back to an exact whole
millisecond of elapsed time —

```
elapsed (mean × 100):  14 14 14 9 15 15 10 9 11 13 20 18 14 12 24 ms
deviation from integer: 0.0000 ms in all 15 samples
```

— which is `performance.now()`'s 1 ms clamp showing through, and it bounds the
resolution at ±0.005 ms, about ±3.5% of a 0.14 ms reading. That cannot produce a
36% shift, and quantisation is unbiased anyway: it would widen the spread, not
move the median in one direction.

What fits is **ramp**. 100 runs of a 0.09 ms inference is only ~9 ms of work —
too short a burst for the core to reach and hold its clock, so a large part of the
window is spent unramped. 1000 runs is ~90 ms, long enough that the ramped state
dominates, and 5000 runs changes nothing further. The GPU never showed the effect
because even 100 of its runs is ~160 ms of continuous work, already past the same
threshold.

This is the opposite of the failure mode the harness comment warns about. Longer
looping was expected to *inflate* figures by drifting into sustained-throughput
territory; on this CPU it deflates them, because the short burst never got the
clock up in the first place. Fixed-performance mode does not prevent this — it
holds a sustainable ceiling, it does not pin the instantaneous frequency.

### The stopping rule works

`CLAUDE.md` says to raise the count until the median stops moving. Applied here it
gives the right answer in three steps: 100 → 1000 moves the CPU by 36% (not
settled), 1000 → 5000 moves it by 0% (settled). The GPU passes at the first check.

## One-time costs

One observation per session, `timing_runs=5000` block for the CPU and the 1000
block for the GPU.

| Backend | Compile | Cold start |
| --- | --- | --- |
| CPU (`wasm`) | 2.80 / 4.60 / 2.10 ms | 0.70 / 0.80 / 0.50 ms |
| GPU (`webgpu`) | 8.70 / 27.30 / 22.30 ms | 17.40 / 15.60 / 15.40 ms |

The GPU compile figures here are far below the ~90 ms the same page produced in
the [companion session](2026-08-24-stock-chrome-litert-vs-webgpu.md) and in the
README's performance report. The difference is that this session switches
backends inside one page load rather than measuring a freshly started browser, so
these are warm-process compiles. They are not comparable with the cold ones and
are listed only for completeness.

## What this does not settle

- **Whether the default should change.** 1000 runs would make the CPU figure
  correct out of the box, but it also makes one Classify click take ~1.6 s on the
  GPU path, which is a poor interactive page. This measurement establishes the
  number to use for the CPU; it does not decide the default.
- **Whether other devices ramp the same way.** One device, one browser.
- **`webgpu.html` was not re-measured here**, so the 3.24 ms direct-WebGPU figure
  in the companion file is still a 100-run number. It is a GPU path, and the GPU
  was shown to be on its plateau at 100, so it is very likely sound — but it was
  not checked directly.
