# Why LiteRT.js beats the hand-written WebGPU page

**Date** 2026-08-24 · **Device** OnePlus 13 (`CPH2653`, Snapdragon 8 Elite,
Adreno 830, Android 16 / API 36) · **Browser** **stock Chrome 150.0.7871.188**
(`com.android.chrome`) · **Content** local files over `adb reverse tcp:8080`,
`@litertjs/core` 2.5.3 installed locally · **Clocks**
`set-fixed-performance-mode-enabled true`, restored after · **Protocol** driven
over the DevTools protocol (`adb forward tcp:9222`, `Runtime.evaluate`); fused
mode; each figure the mean of 1000 inferences, median of 3 repeats. Raw data:
[`2026-08-24-litert-vs-handwritten-tracing.json`](2026-08-24-litert-vs-handwritten-tracing.json).

Answers the question the repo left open: LiteRT.js `webgpu` runs the same model at
**1.61 ms** and the hand-written page at **3.26 ms**. Why? Traced the LiteRT
source (its published bundle ships a source map with full TypeScript) and then
measured each stage of both paths on the device.

## Headline

**Neither path is compute-bound, and the gap is not in the shader.** For a
0.6 MFLOP model the arithmetic is microseconds; essentially the entire per-call
cost is the **CPU↔GPU round-trip** — the time to submit work and get a signal
back that the GPU finished. On Android Chrome that round-trip is **~3.3 ms**, and
it is the same whether you map the result or merely wait for it:

| Hand-written stage (fused, 1000 runs) | ms |
| --- | ---: |
| enqueue only (submit compute, don't wait) | **0.12** |
| compute + `onSubmittedWorkDone` (wait, no map) | 3.26 |
| compute + copy + `mapAsync`, reused buffer — **the current `run()`** | 3.25 |
| compute + copy + `mapAsync`, two submits + fresh buffer | 2.45 |
| **pipelined — one inference kept in flight** | **0.48** |

LiteRT, probed the same way:

| LiteRT stage (fused, 1000 runs) | ms |
| --- | ---: |
| `model.run()` only (enqueue) | **0.10** |
| `data()` only, isolated readback of one tensor | 3.52 |
| **full `run()` + `data()` — what `index.html` measures** | **1.70** |

The two enqueue costs match (0.10 vs 0.12 ms), the two isolated syncs match
(~3.3–3.5 ms). The whole contest is **how much of that round-trip each path hides
behind other work.**

## What LiteRT actually does

Reading `signature_runner.ts`, `tensor.ts` and `gpu_copy_functions.ts` from the
shipped source map:

1. **Input and output tensors are WebGPU buffers** (`WEB_GPU_BUFFER_PACKED`,
   confirmed on device), not host memory. So LiteRT must read back over
   `mapAsync` exactly like the hand-written page — its readback code
   (`gpuTensorToCpuTensor`) is the *same* pattern: create a `MAP_READ` buffer,
   `copyBufferToBuffer`, submit, `await mapAsync`, unmap.
2. **`model.run()` only enqueues** — it submits the compute and returns in
   0.10 ms without syncing. The GPU sync happens later, in `data()`.
3. That split means compute and readback are **two separate submits**, and the
   GPU starts executing the compute while the CPU is still encoding the readback.
   The `index.html` loop `await run(); await data()` gets partial overlap from
   this, landing at 1.70 ms — between the fully-serial 3.26 ms and the
   fully-pipelined floor.
4. It runs the **JSPI** wasm build (`jspi=true, threads=false` on device; no
   cross-origin isolation, so no threads).

The hand-written `run()`, by contrast, fuses compute+copy into **one** submit and
then **blocks** on `mapAsync` with nothing else in flight — submit, wait idle,
unmap, repeat. It pays the full round-trip every call: 3.25 ms.

## Things that are NOT the cause (ruled out by measurement)

- **Compute / occupancy.** Enqueue is 0.12 ms; per-layer (more workgroups) is not
  faster than fused. The math is free at this size.
- **The W1 memory layout.** Coalescing the strided read changed nothing —
  [separate A/B](2026-08-24-coalesce-w1-ab.md).
- **`MAP_READ` buffer reuse.** A fresh readback buffer per call (like LiteRT) was
  *slightly slower* (3.45 vs 3.28 ms), not faster. Reuse is not the problem; the
  plan's leading suspect is wrong.

## The real lever: hide the round-trip

Two changes to the hand-written path, measured:

- **Two submits instead of one** (compute submitted before the readback is
  encoded): 3.25 → **2.45 ms**. The GPU begins computing during CPU-side encode.
- **Pipelining** (submit iteration *i+1* before awaiting *i*'s `mapAsync`, one
  inference always in flight): 3.25 → **0.48 ms**, a 7× drop that also **beats
  LiteRT's 1.70 ms.** With the round-trip overlapped, throughput falls to the
  submit+copy+compute cost, and the latency vanishes from the average.

> [!IMPORTANT]
> **This is a throughput result, and the benchmark is a throughput test.** The
> harness times `TIMING_RUNS` back-to-back inferences of the same input and
> reports the mean, so overlap is fair game — and LiteRT is already exploiting it
> to beat a serial hand-written loop. But a user classifies **once** per drawing,
> and a single inference cannot hide its own round-trip: the true one-shot GPU
> latency is ~3.3 ms for *both* implementations, which is what the "cold start"
> line reports. The 1.61 vs 3.26 ms table is a measure of readback pipelining,
> not of the GPU doing the work faster.

## Bottom line

LiteRT is faster because it **decouples submit from readback and overlaps the
CPU↔GPU round-trip**; the hand-written page is slower only because its `run()`
serialises submit → block → unmap. Nothing about the gap is fundamental to
hand-written WGSL — a one-deep pipeline makes the hand-written page faster than
LiteRT. It does not touch the repo's thesis: the round-trip is exactly the
"per-call GPU overhead" that dwarfs the arithmetic, and the CPU (`wasm`) at
0.09 ms still beats every GPU path, pipelined or not.

## Method note

LiteRT source was read from `node_modules/@litertjs/core/dist/index.js.map`
(`sourcesContent`), installed locally with `npm install` (`node_modules` is
gitignored; `package.json` was reverted afterwards). Probes
(`litert-probe.html`, `webgpu-bench.html`, `webgpu-freshbuf.html`) were scratch
files, deleted after the run.
