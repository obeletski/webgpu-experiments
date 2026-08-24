# WGSL Explainer — the hand-written shaders in `webgpu.html`

The `webgpu.html` page runs the MNIST classifier by hand, directly against the
WebGPU API, with compute shaders **written by hand in WGSL** (WebGPU Shading
Language) — no ML framework, no shader generator. This document explains how
those shaders were derived from the model and how they work, line by line.

The weights come from the `.tflite` (parsed in JS by a ~90-line flatbuffer
reader); everything below — the *compute* — is authored by hand.

---

## 1. The model fixes the math

The `.tflite` is a tiny two-layer perceptron. The flatbuffer reader pulls out four
tensors and their shapes:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `W1` | `[128][2352]` | layer-1 weights |
| `b1` | `[128]` | layer-1 bias |
| `W2` | `[10][128]` | layer-2 weights |
| `b2` | `[10]` | layer-2 bias |

Those shapes *are* the specification. The whole inference is:

```
hidden[o] = ReLU( b1[o] + Σ_i  W1[o][i] · input[i] )   for o in 0..128,  i in 0..2352
logits[o] =       b2[o] + Σ_i  W2[o][i] · hidden[i]    for o in 0..10,   i in 0..128
result[o] = softmax(logits)[o]                          for o in 0..10
```

Each layer is a **matrix–vector multiply** (a stack of dot products) followed by
an activation. The constants `2352`, `128`, `10` that appear as loop bounds in the
shaders come straight from these shapes.

---

## 2. The one design idea: one GPU thread per output neuron

A GPU runs thousands of threads in parallel, so the question when writing the
shader is *what does one thread do?* Every output neuron's dot product is
independent of the others, so the natural mapping is:

> **one thread computes one output element.**

- Layer 1 has 128 outputs → 128 threads, each doing its own 2352-long dot product.
- Layer 2 has 10 outputs → 10 threads, each a 128-long dot product.
- Softmax is a reduction over 10 values → done by a single thread.

Everything else is a consequence of that choice.

---

## 3. The fused shader — the whole network in one dispatch

`FUSED_WGSL` runs all three stages in a **single** compute shader, dispatched as
one workgroup of 128 threads (`@workgroup_size(128)`). The thread index `t` is the
output neuron it owns.

```wgsl
@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(1) var<storage, read>       w1     : array<f32>;
@group(0) @binding(2) var<storage, read>       bias1  : array<f32>;
@group(0) @binding(3) var<storage, read>       w2     : array<f32>;
@group(0) @binding(4) var<storage, read>       bias2  : array<f32>;
@group(0) @binding(5) var<storage, read_write> result : array<f32>;

var<workgroup> hidden : array<f32, 128>;   // on-chip scratch, shared by the workgroup
var<workgroup> logits : array<f32, 10>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;

  // --- layer 1: thread t computes hidden[t] ---
  var acc = bias1[t];
  for (var i = 0u; i < 2352u; i = i + 1u) {
    acc = acc + w1[i * 128u + t] * input[i];
  }
  hidden[t] = max(acc, 0.0);              // ReLU

  workgroupBarrier();                     // all 128 hidden values now visible

  // --- layer 2: only threads 0..9 are needed ---
  if (t < 10u) {
    var acc2 = bias2[t];
    let base2 = t * 128u;
    for (var i = 0u; i < 128u; i = i + 1u) {
      acc2 = acc2 + w2[base2 + i] * hidden[i];
    }
    logits[t] = acc2;
  }

  workgroupBarrier();                     // logits ready

  // --- softmax: thread 0 reduces over the 10 logits ---
  if (t == 0u) {
    var m = logits[0];
    for (var i = 1u; i < 10u; i = i + 1u) { m = max(m, logits[i]); }
    var s = 0.0;
    for (var i = 0u; i < 10u; i = i + 1u) { s = s + exp(logits[i] - m); }
    for (var i = 0u; i < 10u; i = i + 1u) { result[i] = exp(logits[i] - m) / s; }
  }
}
```

Three things make this "fused":

1. **`hidden` and `logits` are `var<workgroup>`** — fast on-chip memory private to
   the workgroup. The intermediate results never travel out to global GPU memory
   and back between layers.
2. **`workgroupBarrier()`** synchronizes the 128 threads. After the first barrier
   every `hidden[i]` written by some thread is visible to every other thread, so
   layer 2 can read the whole hidden vector. The second barrier does the same for
   `logits` before softmax.
3. **One dispatch, one workgroup.** The entire model is a single
   `dispatchWorkgroups(1)`.

---

## 4. The per-layer shaders — three dispatches through global memory

`LAYER1_WGSL`, `LAYER2_WGSL`, `SOFTMAX_WGSL` express the *same math* as three
separate shaders, each its own dispatch, passing data through `storage` buffers in
global GPU memory. This is how a general runtime (LiteRT, ONNX Runtime, …) must
work: it composes arbitrary ops and cannot fuse them into one kernel.

```wgsl
// LAYER1_WGSL — global thread id picks the output; guard against the tail
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = gid.x;
  if (o >= 128u) { return; }
  var acc = bias1[o];
  for (var i = 0u; i < 2352u; i = i + 1u) {
    acc = acc + w1[i * 128u + o] * input[i];
  }
  hidden[o] = max(acc, 0.0);              // writes to a storage buffer, not workgroup memory
}
```

`LAYER2_WGSL` is the same shape (10 outputs over 128 inputs, no activation) and
`SOFTMAX_WGSL` is a single-thread reduction (`@workgroup_size(1)`) identical to the
softmax block in the fused shader. Each stage reads the previous stage's `storage`
buffer and writes the next — `input → hidden → logits → result`.

Because the outputs no longer fit in one workgroup, layer 1 is dispatched as
**2 workgroups × 64 threads = 128 lanes** (`@workgroup_size(64)`, with an
`if (o >= 128u) return;` guard so a partial final group does nothing).

---

## 5. Why both exist: the fused-vs-per-layer difference *is* the measurement

The two modes compute bit-identical outputs. What differs is **dispatch count**:
fused is one dispatch, per-layer is three. The page runs both so the cost of a
GPU dispatch can be isolated — see the second experiment in the
[README](README.md) and [`docs/architecture.md`](docs/architecture.md). To keep
that comparison honest, everything *except* the dispatch structure is held
identical between them — same arithmetic, same memory layout for `W1` (below).

---

## 6. Hand-tuned details written in deliberately

- **Bindings.** Each shader declares its buffers as
  `@group(0) @binding(N) var<storage, …>`, matched on the JS side by a bind group
  (`buf.input`, `buf.w1`, …). `read` buffers are inputs; the one `read_write`
  buffer is the output.
- **Transposed `W1` — `w1[i * 128u + t]`.** `W1` arrives from the `.tflite` as
  `[128][2352]` (output-major). Indexed naively, the 128 lanes of a step would
  read addresses 9,408 bytes apart — no coalescing. `createEngine` transposes it
  to `[2352][128]` once at upload so consecutive lanes read consecutive floats.
  (A/B'd on hardware: bit-identical, and it turned out to buy nothing at batch-1
  because the cost is the readback, not the weight read — see
  [`docs/measurements/2026-08-24-coalesce-w1-ab.md`](docs/measurements/2026-08-24-coalesce-w1-ab.md).
  Kept because it is the layout a larger model would want.)
- **Numerically stable softmax.** `exp(logits[i] - m)`, subtracting the max `m`
  before exponentiating, prevents `exp` from overflowing on large logits — a
  standard trick coded in by hand.

---

## 7. How the JS side drives it

`createEngine` in `webgpu.html` connects the shaders to data:

1. **Upload weights once.** `W1` (transposed), `b1`, `W2`, `b2` go into `storage`
   buffers at startup; they never move again.
2. **Compile pipelines.** `createShaderModule({ code: FUSED_WGSL })` →
   `createComputePipeline`; a bind group wires the buffers to the `@binding`
   slots.
3. **Per inference** (`run()`): `writeBuffer` the 2352 input floats, record the
   dispatch(es) into a compute pass, `submit`, then copy the 10 result floats into
   a `MAP_READ` buffer and `mapAsync` them back to JavaScript.

That last readback — not the shader — is where this page spends its time: the
arithmetic is ~0.6 MFLOP (microseconds), and the wait is the `mapAsync` fence.
That story is in [`docs/architecture.md`](docs/architecture.md) §7.

---

## In one sentence

The shaders were created by reading the two dense layers + softmax off the
`.tflite`, mapping **one GPU thread per output neuron** into WGSL by hand, and
writing it twice — fused (one dispatch, hidden layer kept in on-chip workgroup
memory) and per-layer (three dispatches through global buffers) — so the cost of a
GPU dispatch could be measured against a bit-identical baseline.
