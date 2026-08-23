# Direct WebGPU inference — design

**Date:** 2026-08-23
**Status:** approved for planning

## Purpose

Measure what the same MNIST classification costs when written **directly against
the WebGPU API**, instead of going through LiteRT.js.

The existing experiment ([README](../../../README.md)) found the GPU losing to
the CPU by 7.7× on real mobile hardware, and attributed that to per-call fixed
overhead — readback synchronization, dispatch cost, batch-1 occupancy. That
attribution was reasoning, not measurement. This experiment tests it directly by
removing the runtime from the equation and controlling the dispatch count.

Two questions:

1. **Does a hand-written implementation beat LiteRT.js on this model?** If it
   does not, LiteRT's overhead is not the explanation and WebGPU itself is.
2. **What does a dispatch actually cost here?** Answered by running the same
   network as one fused dispatch and as three, and taking the difference.

## The model

Parsed from `digit_classifier.tflite` (flatbuffer, schema version 3,
description "MLIR Converted"). Exact, not estimated:

```
input  [1,28,28,3] f32
  RESHAPE                                        -> [1, 2352]
  FULLY_CONNECTED  W[128,2352]  b[128]  act=RELU -> [1, 128]
  FULLY_CONNECTED  W[10,128]    b[10]   act=NONE -> [1, 10]
  SOFTMAX                                        -> [1, 10]
```

302,474 trainable parameters, all `float32`, no quantization. (The file also
stores a 2-element `int32` constant holding the reshape target shape; counting it
as a parameter gives 302,476, which is wrong.) Weights are row-major
`[out, in]`, so `W[o*in + i]`. Total arithmetic is ~0.6 MFLOP per inference.

The `RESHAPE` is a no-op for us: preprocessing produces a flat
`Float32Array(2352)` directly.

**Preprocessing** is copied from `index.html` and must stay identical, because
the comparison assumes both pages feed the model the same numbers:

1. Fill the hidden 28×28 canvas **opaque black**, then `drawImage` the 280×280
   drawing onto it. Skipping the fill leaves anti-aliased stroke edges as RGB 255
   at low alpha, which read back as fully-lit pixels.
2. Grayscale as `0.299R + 0.587G + 0.114B`, normalized to `[0,1]`.
3. Write each pixel's value **three times**, once per channel, giving NHWC
   `[1,28,28,3]` = 2352 floats. The drawing is white-on-black so all three
   channels are equal, but the model declares three and the buffer must match.

## Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Weight source | Parse `digit_classifier.tflite` in the browser | Both pages provably use identical bytes; nothing extra committed; no build step |
| Dispatch strategy | Both fused and per-layer, switchable at runtime | The difference *is* the measurement |
| CPU baseline in-page | None | LiteRT's `wasm` backend next door already provides it, measured the same way |
| Code sharing | **None** — `webgpu.html` is self-contained | Matches how `index.html` is built; `index.html` is verified working and stays untouched |

**Accepted trade-off.** The drawing, preprocessing, timing and layout code is
duplicated from `index.html` rather than shared. The two copies can drift, and if
the *timing* code drifts the cross-page comparison silently stops being
apples-to-apples. Mitigation: the timing constants and loop structure are
specified exactly below, and both files carry a comment pointing at the other.

## Files

| File | Change |
| --- | --- |
| `webgpu.html` | **New.** Everything: markup, CSS, drawing, tflite parser, WebGPU engine, WGSL, timing, UI |
| `index.html` | No code change. One added link to the new page — the only edit |
| `README.md` | New section for this experiment; replace the estimated parameter count with the exact figure |

## Weight extraction

A minimal flatbuffer reader — no library. The needed schema fields, verified
against this file:

```
Model    : 1 = operator_codes, 2 = subgraphs, 4 = buffers
SubGraph : 0 = tensors, 1 = inputs, 2 = outputs, 3 = operators
Tensor   : 0 = shape, 1 = type, 2 = buffer, 3 = name
Operator : 0 = opcode_index, 1 = inputs, 2 = outputs
Buffer   : 0 = data
OperatorCode : 0 = deprecated_builtin_code (i8), 3 = builtin_code (i32)
```

Reader primitives: a table at `pos` has its vtable at `pos - i32(pos)`; field
`n` lives at `pos + u16(vtable + 4 + 2n)`, absent when that is `0` or beyond
`u16(vtable)`. Vectors and strings are `u32` length followed by data, reached
through a `u32` offset relative to its own position.

**Resolve tensors by walking operators, never by hard-coded index.** Find the
`FULLY_CONNECTED` operators (builtin code 9) in order; for each, `inputs[1]` is
the weight tensor and `inputs[2]` the bias. This survives a re-exported model.

**Alignment.** Buffer payloads are `ubyte` vectors with no float alignment
guarantee, so the byte range must be copied into a fresh buffer (or read through
a `DataView`) before being viewed as `Float32Array`. Reinterpreting in place will
throw on a misaligned offset.

**Validation.** After parsing, assert shapes are `[128,2352]`, `[128]`,
`[10,128]`, `[10]` and that every tensor is `float32` (type code 0). Fail loudly
with a specific message rather than producing wrong numbers.

Factor the parse as `parseWeights(arrayBuffer)` so it can be exercised in Node
against the real file, separately from any browser.

## WebGPU engine

### Resources

| Buffer | Size | Usage | Written |
| --- | --- | --- | --- |
| `input` | 2352 f32 | STORAGE + COPY_DST | per inference |
| `w1` | 128×2352 f32 (1.20 MB) | STORAGE (read-only) | once |
| `b1` | 128 f32 | STORAGE (read-only) | once |
| `w2` | 10×128 f32 | STORAGE (read-only) | once |
| `b2` | 10 f32 | STORAGE (read-only) | once |
| `hidden` | 128 f32 | STORAGE | per-layer mode only |
| `logits` | 10 f32 | STORAGE | per-layer mode only |
| `output` | 10 f32 | STORAGE + COPY_SRC | per inference |
| `readback` | 10 f32 | MAP_READ + COPY_DST | per inference |

All within default limits: the largest binding is 1.2 MB against a 128 MB
`maxStorageBufferBindingSize`, and workgroup memory peaks at 512 B against
16 KB.

### Fused pipeline — one dispatch

`@workgroup_size(128)`, `dispatchWorkgroups(1)`:

```
let t = local_invocation_id.x;
h[t] = max(dot(W1 row t, input) + b1[t], 0.0);   // 2352 MACs, h is var<workgroup>
workgroupBarrier();
if (t < 10) { logits[t] = dot(W2 row t, h) + b2[t]; }   // logits also var<workgroup>
workgroupBarrier();
if (t == 0) { softmax(logits) -> output }
```

One workgroup means one compute unit and poor occupancy — deliberate. Batch-1
cannot fill a GPU regardless, and the point is minimal overhead. The intermediate
never leaves workgroup memory.

### Per-layer pipeline — three dispatches

Three pipelines recorded into **one compute pass on one command encoder**, which
is how a general runtime would submit them. WebGPU orders dispatches within a
pass and makes prior writes visible to later ones, so no explicit barrier is
needed.

1. `layer1` — `@workgroup_size(64)`, 2 workgroups → `hidden`, fused ReLU
2. `layer2` — `@workgroup_size(64)`, 1 workgroup, guarded `o < 10` → `logits`
3. `softmax` — `@workgroup_size(1)`, 1 workgroup → `output`

The fused-vs-layered delta is the cost of two extra dispatches plus round-tripping
the intermediates through global memory.

### Running one inference

`queue.writeBuffer(input)` → encode → `copyBufferToBuffer(output, readback)` →
`submit` → `readback.mapAsync(GPUMapMode.READ)` → copy out → `unmap`. The
`mapAsync` await is the CPU↔GPU synchronization point that dominates, and it is
inside the timed region — as it is for LiteRT, so the two are comparable.

`softmax` runs on the GPU rather than on the 10 returned logits, so that the work
measured matches what LiteRT's graph does.

## Timing

Must match `index.html` exactly or the comparison is void. Same constants, same
structure:

- `TIMING_MIN_MS = 5`, `TIMING_MAX_RUNS = 50`
- One **untimed warm-up** run, discarded
- Then repeat until `elapsed >= TIMING_MIN_MS` or `runs === TIMING_MAX_RUNS - 1`
- Report `elapsed / runs` and the run count
- Sub-resolution durations render as `<1 ms`, never `0.00 ms`

Reported per dispatch mode, mirroring `index.html`'s backend line:

- **pipeline** — building pipelines and uploading weights (analogous to compile)
- **cold start** — first inference after pipeline creation, timed alone
- **steady state** — the mean above

## UI

The `index.html` layout, with the CPU/GPU switch replaced by
`[ fused (1 dispatch) | per-layer (3 dispatches) ]`. Switching rebuilds only the
pipelines, not the buffers, and re-times cold start. The canvas keeps its
drawing across a switch. Same responsive sizing rules, so it fits a phone screen.

## Error handling

Drawing must keep working no matter what fails — the same rule `index.html`
follows, and the reason its canvas broke originally.

| Failure | Behaviour |
| --- | --- |
| No `navigator.gpu` / no adapter | State that WebGPU is unavailable; drawing still works; Classify disabled |
| Weight parse or validation failure | Report the specific assertion; Classify disabled |
| Shader compile / pipeline error | Surface `device.createShaderModule` compilation messages to the console and a short message in the UI |
| `device.lost` | Report it; do not silently produce stale results |

## Verification

1. **Parser, in Node.** `parseWeights` against `digit_classifier.tflite`: shapes,
   dtypes, parameter count 302,474, and spot-check that the first weights match
   the bytes at the tensor's buffer offset.
2. **Numerical oracle.** A plain-JS matvec over the parsed weights, in the test
   harness rather than the page, compared elementwise with the GPU output.
   Tolerance `1e-4` absolute — floating-point summation order differs, so exact
   equality is the wrong test.
3. **Agreement with LiteRT.** Drive `index.html` and `webgpu.html` with an
   identical synthetic stroke; require the same argmax and probabilities within
   `1e-3`.
4. **Both modes agree.** Fused and per-layer must produce the same distribution
   within `1e-4` — the fused kernel's workgroup-memory path is where a barrier
   bug would show up.
5. **Layout.** The viewport sweep already used for `index.html`: fits without
   scrolling from 320×480 up, and panel width unaffected by result text.
6. **Measurement.** Numbers taken on the Android device under
   `set-fixed-performance-mode-enabled true`, 3 runs × 5 measurements, as the
   existing performance report was.

## Out of scope

Quantization; tiled or subgroup-optimized matmul; batching; WebNN; the
centre-of-mass input normalization (since implemented, and TODO.md removed);
sharing code with `index.html`.

## Risks

- **The fused kernel may lose.** One workgroup uses a fraction of the GPU. If
  per-layer wins, that is a finding about occupancy, not a bug.
- **Hand-written may not beat LiteRT.** Also a finding — it would mean the cost
  is WebGPU's dispatch-and-readback floor, not runtime overhead, which is the
  more interesting answer.
- **Timing code duplication** may drift from `index.html`; see the accepted
  trade-off above.
- **GPU variance.** The existing GPU numbers had CV 49% at 1–2 samples per
  measurement. A single inference here may also exceed the 5 ms budget. If it
  does, raise `TIMING_MIN_MS` for *both* pages and re-measure, rather than
  comparing across different budgets.
