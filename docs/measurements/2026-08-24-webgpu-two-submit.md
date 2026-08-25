# Two-submit readback: overlapping the round-trip in the hand-written page

**Date** 2026-08-24 · **Device** OnePlus 13 (`CPH2653`, Snapdragon 8 Elite,
Adreno 830, Android 16 / API 36) · **Browser** **stock Chrome 150.0.7871.188**
(`com.android.chrome`) · **Content** `webgpu.html` served locally over
`adb reverse tcp:8080` (uncommitted change) · **Clocks**
`set-fixed-performance-mode-enabled true`, restored after · **Protocol** driven
over the DevTools protocol; fused; `?timing_runs=1000`.

Follows the [round-trip trace](2026-08-24-litert-vs-handwritten-tracing.md), which
found that the hand-written `run()` pays a full CPU↔GPU round-trip per inference
because it fuses compute and readback into one submit and then blocks on
`mapAsync`. This applies the fix and measures it.

## The change

`run()` now issues **two** submits instead of one: the compute pass in its own
submit, then a separate encoder + submit for the `copyBufferToBuffer` into the
`MAP_READ` buffer, then the `mapAsync` await. The first submit lets the GPU begin
the dispatch while the CPU is still encoding the copy, so compute overlaps the
readback's setup instead of running strictly in series. This is the structural
overlap LiteRT gets by returning from its `run()` before syncing. Both fused and
per-layer use the same `run()`, so their comparison is unaffected. The arithmetic
is untouched.

## Correctness — bit-identical

Verified two independent ways:

- **Desktop** (headless Chrome 135, SwiftShader WebGPU): a fixed synthetic 2352-
  vector through `window.__engine.run` returns the **exact same float32** vector
  for the two-submit and the one-submit builds, in **both** modes, and fused
  equals per-layer within a build.
- **Phone**: an identical injected "1" stroke predicts **1 at 100%**.

The change only reorders submits; it cannot change the result, and does not.

## Timing — one session, not the full A/B

Fused, `timing_runs=1000`, **1 session × 5 samples** (the user asked for a single
measurement, not the 3-session paired A/B):

| | samples (ms) | median |
| --- | --- | ---: |
| **two-submit** `run()` | 2.50, 2.80, 2.85, 2.74, 2.75 | **2.750 ms** |
| one-submit baseline (prior sessions, for reference) | — | ~3.26 ms |

So on hardware the two-submit structure takes fused from **~3.26 → ~2.75 ms**, a
~16% drop, in the direction the [stage bench](2026-08-24-litert-vs-handwritten-tracing.md)
predicted (it isolated the two-submit structure at ~2.45 ms). It stays above
LiteRT's ~1.6 ms: two submits capture the compute/encode overlap but not the full
round-trip pipeline, which a self-contained `run()` cannot do without returning a
one-behind result.

> [!NOTE]
> **This is a single session against a baseline from earlier sessions**, not a
> back-to-back paired A/B, so it is weaker evidence than the repository's headline
> figures (3-session or 1000-run-plateau medians with CVs). The direction and
> rough magnitude are clear and repeatable, but the exact ms and any ratio should
> be re-measured with the full fixed-count protocol before they replace the
> published tables. Until then the README's results table keeps its pre-fix
> 3.270 ms figure, flagged as such.
