# Three Paths to One Inference

The same 302,474-parameter MNIST classifier, run three ways. This document maps
the API each path uses, then does something the repository has not done
explicitly: **it writes down what the architecture predicts, before showing what
the device measured.**

The prediction section is a genuine a-priori reading of the diagrams below. It
gets memory exactly right and time exactly backwards, which is the most useful
thing in this document.

| Path | Page | API |
| --- | --- | --- |
| **Runtime** | `index.html` | LiteRT.js — `wasm` (XNNPack) or `webgpu` |
| **Hand-written** | `webgpu.html` | WebGPU directly, WGSL written by hand |
| **Browser-native** | `browser-model-api.html` | `navigator.digitclassifier`, C++ inside Blink |

The model is identical in all three: input `[1,28,28,3]` float32 — 2352 values —
through `2352 → 128` (ReLU) `→ 10` → softmax. Every path does 0.605 MFLOP of
arithmetic per inference.

---

## 1. What each path stacks

```mermaid
flowchart TB
    subgraph R["Runtime — index.html"]
        direction TB
        R1["page JavaScript"] --> R2["LiteRT.js ES module"]
        R2 --> R3["LiteRT wasm runtime<br/>2.6 MB"]
        R3 --> R4a["XNNPack kernels<br/>relaxed SIMD"]
        R3 --> R4b["WebGPU backend<br/>generated WGSL"]
        R4b --> R5["GPU driver"]
    end

    subgraph H["Hand-written — webgpu.html"]
        direction TB
        H1["page JavaScript"] --> H2["flatbuffer reader<br/>~90 lines"]
        H1 --> H3["WebGPU API"]
        H3 --> H4["hand-written WGSL"]
        H4 --> H5["GPU driver"]
    end

    subgraph B["Browser-native — browser-model-api.html"]
        direction TB
        B1["page JavaScript"] --> B2["navigator.digitclassifier"]
        B2 --> B3["Blink C++<br/>weights in the binary"]
        B3 --> B4["fused WGSL"]
        B4 --> B5["GPU driver"]
    end
```

Reading down the stacks, the hand-written path removes a runtime and the
browser-native path removes JavaScript. **Both are removing layers that sit
*above* the GPU driver** — which turns out to matter, because that is not where
the cost is.

---

## 2. Startup: what each path pays before it can classify

```mermaid
sequenceDiagram
    autonumber
    participant P as page
    participant L as LiteRT.js
    participant W as wasm runtime
    participant G as WebGPU

    Note over P,G: Runtime path — index.html
    P->>L: loadLiteRt(wasmDir)
    L->>W: fetch + instantiate 2.6 MB wasm
    W-->>L: ready
    P->>G: navigator.gpu.requestAdapter()
    P->>G: adapter.requestDevice()
    P->>L: setWebGpuDevice(device)
    P->>P: fetch digit_classifier.tflite (1.2 MB)
    P->>L: loadAndCompile(bytes, accelerator)
    L->>W: parse model, build kernels
    L->>G: create buffers, compile generated WGSL
    L-->>P: CompiledModel
    P->>L: getInputDetails() to read the shape
```

```mermaid
sequenceDiagram
    autonumber
    participant P as page
    participant F as flatbuffer reader
    participant G as WebGPU

    Note over P,G: Hand-written path — webgpu.html
    P->>P: fetch digit_classifier.tflite (1.2 MB)
    P->>F: parseWeights(bytes)
    F-->>P: W1 b1 W2 b2 as Float32Array
    P->>G: requestAdapter() then requestDevice()
    P->>G: createBuffer x9, weights mappedAtCreation
    P->>G: createShaderModule(FUSED_WGSL)
    P->>G: createComputePipeline + createBindGroup
    Note right of G: pipeline build measured at under 1 ms
```

```mermaid
sequenceDiagram
    autonumber
    participant P as page
    participant N as navigator.digitclassifier
    participant C as Blink C++

    Note over P,C: Browser-native path — browser-model-api.html
    P->>N: getModel()
    N->>C: RequestAdapter then RequestDevice
    C->>C: weights already in the binary, nothing fetched
    C-->>N: model handle
    N-->>P: DigitClassifierModel
    Note right of C: no network, no wasm, no shader source in the page
```

The asymmetry is stark. The runtime path fetches **3.65 MB** and compiles a model
before it can answer anything; the browser-native path fetches **nothing at all**
and its weights were linked into Chromium.

---

## 3. Steady state: the API calls that actually get timed

This is the part the benchmark measures, and where the three paths are far more
alike than their stacks suggest.

```mermaid
sequenceDiagram
    autonumber
    participant P as page
    participant L as LiteRT.js
    participant G as GPU

    Note over P,G: Runtime path, webgpu accelerator
    P->>L: new Tensor(inputBuffer, shape)
    P->>L: model.run(tensor)
    L->>G: writeBuffer, encode, submit
    G-->>L: compute
    L->>G: copyBufferToBuffer into a fresh MAP_READ buffer
    L->>G: await mapAsync
    G-->>L: 10 floats
    L-->>P: outputs
    P->>L: data() on the output tensor
    P->>L: tensor.delete() and outputs.delete()
    Note right of L: manual memory management, every call
```

```mermaid
sequenceDiagram
    autonumber
    participant P as page
    participant G as GPU

    Note over P,G: Hand-written path, fused mode
    P->>G: queue.writeBuffer(input, 2352 floats)
    P->>G: createCommandEncoder + beginComputePass
    P->>G: setPipeline, setBindGroup
    P->>G: dispatchWorkgroups(1) at workgroup_size 128
    Note right of G: hidden layer never leaves var workgroup
    P->>G: copyBufferToBuffer(result to read, 40 bytes)
    P->>G: queue.submit
    P->>G: await read.mapAsync(READ)
    G-->>P: 10 floats
    P->>G: read.unmap()
    Note right of P: one persistent MAP_READ buffer, reused
```

```mermaid
sequenceDiagram
    autonumber
    participant P as page
    participant C as Blink C++
    participant G as GPU

    Note over P,G: Browser-native path
    P->>C: await model.classify(Float32Array 2352)
    C->>G: upload, dispatch fused shader, read back
    G-->>C: logits
    C-->>P: one integer
    Note right of C: no JavaScript in the timed region at all
```

**Both GPU paths perform the same four operations**: upload the input, submit a
compute pass, copy 40 bytes back, await a map. Two differences look small and one
is not — LiteRT allocates a fresh `MAP_READ` buffer per readback where the
hand-written page reuses one (this turns out not to matter), and LiteRT issues its
readback copy in a separate encoder and submit (this is most of the 2× — it lets
the round-trip overlap; see [§6](#6-scorecard) and the
[trace](measurements/2026-08-24-litert-vs-handwritten-tracing.md)).

The CPU path has no diagram of its own because it has no round trip: `model.run`
goes into XNNPack kernels inside the wasm heap and returns. That is the whole
architectural argument for why it might win.

---

## 4. Predictions

Read only from the diagrams above, before any measurement.

### Time

| # | Prediction | Reasoning |
| --- | --- | --- |
| **T1** | Hand-written **beats** the runtime | It deletes an entire layer: no generic kernel dispatch, no runtime bookkeeping, one shader written for exactly this model. Section 1 shows LiteRT strictly above the same driver. |
| **T2** | GPU **beats** CPU | 301,056 multiply-accumulates in the first layer is embarrassingly parallel; a CPU walks it more or less serially. |
| **T3** | Browser-native is **fastest of all** | Section 2 shows it fetching nothing and section 3 shows no JavaScript in the timed region. Strictly less work than either. |

The common thread: **each prediction assumes cost lives in the layers, so
removing layers should remove cost.**

### Memory

| # | Prediction | Reasoning |
| --- | --- | --- |
| **M1** | Browser-native is smallest by a wide margin | It downloads nothing — no runtime, no model, no shader source. |
| **M2** | Hand-written sits in the middle | Page plus the 1.2 MB `.tflite`, nothing else. |
| **M3** | Runtime is largest | Everything the hand-written path needs, plus a wasm runtime and its loader. |
| **M4** | GPU-resident weights are ~1.2 MB in all three | Same 302,474 parameters at float32. Architecture cannot change this. |

---

## 5. Measured

**OnePlus 13 (CPH2653, Adreno 830), stock Chrome 150.0.7871.188**, served from
GitHub Pages over HTTPS, clocks fixed, run counts raised until each median
stopped moving. Full data in
[`2026-08-24-stock-chrome-cpu-baseline.md`](measurements/2026-08-24-stock-chrome-cpu-baseline.md)
and
[`2026-08-24-stock-chrome-litert-vs-webgpu.md`](measurements/2026-08-24-stock-chrome-litert-vs-webgpu.md).

### Time

| Path | Median | CV | vs CPU |
| --- | ---: | ---: | ---: |
| **Runtime — CPU (`wasm`)** | **0.090 ms** | 0.0% | 1.00× |
| Runtime — GPU (`webgpu`) | **1.610 ms** | 2.1% | 17.9× |
| Hand-written — fused | **3.270 ms** | 1.9% | 36.3× |

### Memory — bytes over the wire

Measured with `curl` against the live Pages site and the CDN it falls back to,
compressed as served.

| Path | Page | Model | Runtime | **Total** |
| --- | ---: | ---: | ---: | ---: |
| Runtime (`index.html`) | 9,896 | 1,126,982 | 2,689,378¹ | **3,826,256 B — 3.65 MiB** |
| Hand-written (`webgpu.html`) | 11,034 | 1,126,982 | — | **1,138,016 B — 1.09 MiB** |
| Browser-native | 8,589 | — | — | **8,589 B — 8.4 KiB** |

¹ `litert_wasm_internal.wasm` 2,612,382 + its loader 67,707 + `@litertjs/core`
8,687 + `@litertjs/wasm-utils` 602.

**445×** separates the largest from the smallest.

### Memory — decoded, and what the runtime actually costs

The wire figures understate the runtime path badly, because its bulk is
compressed wasm. Measured on the device from `PerformanceResourceTiming`
(favicon excluded):

| Path | On the wire | **Decoded** | Expansion |
| --- | ---: | ---: | ---: |
| Runtime | 3,826,256 B | **10,911,126 B — 10.41 MiB** | 2.9× |
| Hand-written | 1,138,016 B | **1,247,328 B — 1.19 MiB** | 1.1× |
| Browser-native | 8,589 B | **24,810 B — 0.02 MiB** | 2.9× |

**LiteRT's wasm alone decodes to 9,367,934 B — 8.93 MiB** from 2.6 MB on the
wire. That is 8.9 MiB of runtime code held in memory to execute a 1.2 MB model:
the runtime outweighs the thing it runs by **7.7×**.

### Memory — JS heap on the device

`Performance.getMetrics` over the DevTools socket, after a forced
`HeapProfiler.collectGarbage`. (`performance.memory` is useless here — it
returned a quantised 10,000,000 on all four configurations.)

| Config | After load | Steady, post-GC | Transient per inference |
| --- | ---: | ---: | ---: |
| Runtime — GPU | 1,695,340 B | **1,925,120 B — 1.84 MiB** | 2,106 B |
| Runtime — CPU | 1,692,912 B | **1,863,948 B — 1.78 MiB** | 1,125 B |
| Hand-written | 552,112 B | **581,924 B — 0.55 MiB** | 1,370 B |
| Browser-native | **495,484 B — 0.47 MiB** | —³ | —³ |

³ Cannot classify on stock Chrome, so only its load-time heap is observable.

Transient figures are garbage collected across ~505 inferences (5 activations ×
100 runs, plus warm-ups). **The runtime's GPU path allocates the most per
inference and is still the fastest** — 2,106 B against the hand-written page's
1,370 B, while running 2.03× quicker. Allocation volume is not what is costing
the hand-written page its time.

Raw probe output:
[`measurements/2026-08-24-stock-chrome-memory.json`](measurements/2026-08-24-stock-chrome-memory.json).

### Memory — GPU-resident

Computed exactly from `webgpu.html`, which is the only path that declares its
buffers in readable source:

| Buffer | Bytes |
| --- | ---: |
| `w1` (2352×128) | 1,204,224 |
| `w2` (128×10) | 5,120 |
| `input` (2352) | 9,408 |
| `b1`, `hidden` | 512 each |
| `b2`, `result`, `read`, `logits` | 40 each |
| **Total** | **1,219,936 B — 1.16 MiB** |

Of which **1,209,896 B is weights** — 302,474 parameters × 4. The other 10,040
bytes are working buffers. No path can do better on the weights without
quantising the model.

---

## 6. Scorecard

| # | Prediction | Outcome | |
| --- | --- | --- | --- |
| **T1** | Hand-written beats the runtime | **Wrong** — it is **2.03× slower** (3.270 vs 1.610 ms), on disjoint distributions | ❌ |
| **T2** | GPU beats CPU | **Wrong** — the CPU is **17.9× faster** | ❌ |
| **T3** | Browser-native is fastest | **Wrong**, on the evidence available² | ❌ |
| **M1** | Browser-native smallest | **Right** on all three metrics — 8.4 KiB wire, 0.02 MiB decoded, 0.47 MiB heap | ✅ |
| **M2** | Hand-written in the middle | **Right** — 1.09 MiB wire, 1.19 MiB decoded, 0.55 MiB heap | ✅ |
| **M3** | Runtime largest | **Right**, and by more than the wire suggests — 3.65 MiB wire but **10.41 MiB decoded** and 3.3× the heap | ✅ |
| **M4** | ~1.2 MB of weights everywhere | **Right** — 1,209,896 B exactly | ✅ |

² Not measurable on stock Chrome, which has no `navigator.digitclassifier`. On
the custom Chromium 153 build where it does exist it measured 2.70 ms against
2.50 ms for the hand-written page — not faster, and both far behind the CPU. A
different browser, so it is evidence rather than proof.

### Every memory prediction held. Every time prediction failed.

That split is the finding. Memory is **structural** — it follows directly from
what each path must download and allocate, so reading the architecture predicts
it exactly, and it predicts it on three independent metrics: bytes on the wire,
bytes decoded, and live JS heap all rank the three paths in the same order. Time is not, because all three predictions shared one assumption:
that cost lives in the layers, so removing layers removes cost.

Section 3 shows why that fails. Strip the runtime and you still issue
`writeBuffer` → `submit` → `mapAsync`. Strip JavaScript as well and you *still*
issue them, just from C++. The ~3 ms is that round trip, and it is charged
identically to a runtime, a hand-written page, and a browser built-in. **Layers
above the driver were never the cost.**

The CPU wins because it is the only path that never makes the round trip at all —
0.605 MFLOP finishes in 0.090 ms inside the wasm heap, while the fastest GPU path
spends 1.61 ms mostly waiting.

### What the diagrams could not have told you

The two GPU paths' per-inference sequences differ in exactly two visible ways —
LiteRT allocates a fresh `MAP_READ` buffer per call where the hand-written page
reuses one, and it splits its readback into a separate submit. The hand-written
choice looks more efficient and measures 2× slower.

The heap measurement confirms the allocation difference is real and rules out the
obvious objection: LiteRT's GPU path leaves **2,106 B** of garbage per inference
against the hand-written page's **1,370 B**, so it really is allocating more — and
winning anyway.

Of those two visible differences, the buffer is a red herring and the submit is
the point. [Tracing both paths stage by stage](measurements/2026-08-24-litert-vs-handwritten-tracing.md)
settles it: switching the hand-written `run()` to a fresh buffer per call was
*slightly slower*, not faster, so buffer reuse is not the cause. What is left is
one CPU↔GPU round-trip that costs **~3.3 ms** — the same whether you `mapAsync`
the result or merely `onSubmittedWorkDone`-wait for it, because it is the *waiting*
that costs, not the mapping. The hand-written `run()` fuses compute and readback
into one submit and then blocks idle on it; LiteRT's `run()` returns without
syncing, making compute and readback two submits that overlap. Keep one inference
in flight in the hand-written page and its cost falls to **0.48 ms**, beating
LiteRT — the round-trip was never the shader's or the buffer's, it was the
serial *wait*, and it hides under pipelining. (This is a throughput result; a
single classification still pays the round-trip once, which is the "cold start"
line, and even pipelined the GPU stays above the CPU's 0.09 ms.)

Reading an architecture predicts what it must *carry*. It does not predict what
it must *wait for* — nor that the wait is the entire bill.
