# Coalescing the W1 read does not move the clock

**Date** 2026-08-24 · **Device** OnePlus 13 (`CPH2653`, Snapdragon 8 Elite,
Adreno 830, Android 16 / API 36) · **Browser** **stock Chrome 150.0.7871.188**
(`com.android.chrome`) · **Content** `http://localhost:8080/webgpu.html` vs
`http://localhost:8080/webgpu-old.html`, served locally over
`adb reverse tcp:8080 tcp:8080` (uncommitted A/B, `localhost` is a secure context
on the phone so WebGPU still initialises) · **Clocks**
`set-fixed-performance-mode-enabled true`, restored after · **Protocol** Chrome
force-stopped between sessions; **driven over the DevTools protocol**
(`adb forward tcp:9222`, `Runtime.evaluate`), an identical injected stroke and
synthetic input across both variants; 3 sessions × 5 samples per variant per
mode, `?timing_runs=1000`, median reported. Raw data:
[`2026-08-24-coalesce-w1-ab.json`](2026-08-24-coalesce-w1-ab.json).

Scores the prediction in
[the coalescing plan](../superpowers/plans/2026-08-24-coalesce-w1.md): **1.6 – 2.0
ms, from 3.27 ms.** The prediction is wrong.

## Headline

| Variant | fused | per-layer |
| --- | ---: | ---: |
| **old** — `w1[t*2352+i]`, strided (128 lanes 9.4 KB apart) | 3.240 ms | 3.320 ms |
| **new** — `w1[i*128+t]`, coalesced (128 lanes contiguous) | 3.260 ms | 3.330 ms |
| Δ | **+0.020 ms** | **+0.010 ms** |

**Coalescing the W1 read changes nothing measurable.** Both deltas are inside a
single session's spread and point the wrong way (very slightly slower). The
hand-written page stays at ~3.26 ms against LiteRT.js's 1.61 ms; the ~1.6 ms gap
is intact and now has one fewer candidate explanation.

## Full statistics

15 samples per cell (3 sessions × 5), each sample the mean of 1000 inferences.

| Variant | Mode | Median | Mean | SD | CV | Range |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| old | fused | 3.240 ms | 3.137 | 0.2148 | 6.8% | 2.70 – 3.34 |
| new | fused | 3.260 ms | 3.157 | 0.2251 | 7.1% | 2.63 – 3.32 |
| old | per-layer | 3.320 ms | 3.319 | 0.0371 | 1.1% | 3.24 – 3.38 |
| new | per-layer | 3.330 ms | 3.332 | 0.0299 | 0.9% | 3.27 – 3.38 |

The fused CV (~7%) is entirely the first sample of each session — 2.63 / 2.76 /
2.75 for new, 2.70 / 2.75 / 2.74 for old — the clock ramping on a freshly loaded
page before it settles to ~3.28. Both variants show the identical pattern, so it
cancels in the comparison. Per-layer, sampled after fused in the same session,
never sees a cold clock and runs at CV ~1%.

## Correctness — bit-identical, as predicted

The change touches only addressing, not the arithmetic or its order, so the
output must be **bit-identical**. It is. A fixed synthetic 2352-vector run through
`window.__engine.run` returned the **exact same float32** probability vector for
new and old, in **both** modes — and fused equals per-layer within a build:

```
new.fused == old.fused        : true   (full float32, all 10 logits)
new.layered == old.layered    : true
new.fused == new.layered      : true
```

End to end, an identical injected "1" stroke through the real `preprocess` →
Classify path predicts **1 at 100.0%** on all four configurations
(old/new × fused/per-layer), matching what `index.html` predicts for the same
shape. Correctness bar met: bit-identical, not merely "still predicts 1".

## What this rules out, and what is left

The commit reasoned that the strided read moved ~19 MB to read a 1.2 MB matrix
and that fixing it should approach LiteRT's floor. The traffic reduction is real
but **off the critical path**: at batch-1 the kernel does 0.6 MFLOP behind a
single `dispatchWorkgroups(1)`, and the wall clock is dominated by fixed
per-submit / readback overhead, not by the bandwidth of the weight read. Making
the read 8× cheaper against a cost that is a rounding error changes nothing.

The change is correct and harmless and stays in — the access pattern is the one a
larger model would want — but it earns no speedup and the README must not claim
one. The remaining suspect for the 3.26 vs 1.61 gap is the **`MAP_READ` staging
buffer round-trip** (allocated/mapped per inference), which the
[heap data](2026-08-24-stock-chrome-memory.json) already showed is not an
allocation-*volume* effect but could still be a per-call *latency* one. That is
the next thing to measure; occupancy (more workgroups) is moot until the readback
is ruled in or out, since it cannot explain a gap that survives here.

## Driver note

Per the README's DevTools-protocol method: the page was driven by injecting the
stroke and reading `#result`, not by `input tap` + OCR. `#result` was cleared to
`''` and waited on until it contained `Predicted`, never waited-for-change —
consecutive runs print byte-identical lines. `webgpu-old.html` was a temporary
`git show 36516e9^:webgpu.html` checkout, deleted after the run.
