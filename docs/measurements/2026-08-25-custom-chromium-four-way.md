# Four paths on the custom build: the C++ browser API is now the slowest GPU path

**Date** 2026-08-25 · **Device** OnePlus 13 (`CPH2653`, Snapdragon 8 Elite,
Adreno 830, Android 16 / API 36) · **Browser** **custom Chromium 153.0.8005.0
release** (`org.chromium.chrome`) — *not* the stock Chrome the other measurements
use; three milestones apart, **never pool these with stock figures** · **Content**
GitHub Pages over HTTPS · **Clocks** `set-fixed-performance-mode-enabled true`,
restored after · **Protocol** driven over the DevTools protocol, `timing_runs=1000`,
3 rounds × 5 samples, median. Raw data:
[`2026-08-25-custom-chromium-four-way.json`](2026-08-25-custom-chromium-four-way.json).

First run of all four paths on the custom build since `webgpu.html` gained the
[fence-poll](2026-08-24-webgpu-mapasync-poll.md), and the first to include
`navigator.digitclassifier` under the fixed-count harness. These are fixed-count
figures and **cannot be pooled** with the 2026-08-23 budget-harness run.

## How the DigitClassifier flag was finally enabled

`navigator.digitclassifier` needs `--enable-blink-features=DigitClassifier`. The
earlier attempt failed and I misblamed SELinux. The real gate (per
`CommandLineInitUtil.java`) is that Chrome reads `/data/local/tmp/chrome-command-line`
only when `Settings.Global.DEBUG_APP` names the package. The fix, verified
on-device this session:

```sh
adb shell am set-debug-app --persistent org.chromium.chrome
adb shell 'echo "chrome --enable-blink-features=DigitClassifier" > /data/local/tmp/chrome-command-line'
adb shell chmod 644 /data/local/tmp/chrome-command-line
adb shell am force-stop org.chromium.chrome
```

After this, `('digitclassifier' in navigator)` is `true` and the page reports
"Ready." CDP was used to drive it — the custom build opens the default
`chrome_devtools_remote` socket once past the first-run screen; the flag is set
before launch, CDP attaches after, and they do not interfere.

## Headline

| Path | Page | Median | CV | vs CPU |
| --- | --- | ---: | ---: | ---: |
| **LiteRT.js — CPU (`wasm`)** | `index.html` | **0.060 ms** | 29.5% | 1.0× |
| Direct WebGPU — fused (fence-poll) | `webgpu.html` | **0.520 ms** | 32.7% | 8.7× |
| LiteRT.js — GPU (`webgpu`) | `index.html` | **1.430 ms** | 3.0% | 24× |
| `navigator.digitclassifier` | `browser-model-api.html` | **3.730 ms** | 1.7% | 62× |

**The ordering of the GPU paths has completely inverted from what the third
experiment found.** The browser-native C++ path — no JavaScript, weights linked
into the binary, the leanest stack in the repo — is now the **slowest** GPU path
by far, and the hand-written JavaScript page is the fastest.

## Why: the C++ path does not poll the fence

This is the [lazy-fence finding](2026-08-24-webgpu-mapasync-poll.md) seen from a
new angle. `navigator.digitclassifier` runs its upload → dispatch → readback
inside Blink and blocks on the result the ordinary way — nothing pokes Chrome's
GPU-process completion fence, so it pays the full lazy-servicing wait. Its
**3.73 ms at CV 1.7%** is the signature: consistently pinned to the lazy-fence
floor, run after run. `webgpu.html` reaches **0.52 ms** *only* because its JS
`run()` polls `onSubmittedWorkDone` from every task to force the fence check —
something the C++ path, for all its efficiency, does not do.

So removing JavaScript did not remove the cost; it removed the *workaround*. The
~3.3–3.7 ms wait is **structural** — paid even by zero-JavaScript C++ inside the
browser — which is exactly the third experiment's thesis, now with the twist that
a JavaScript page can beat the C++ one by hurrying the GPU process along.

For contrast, the 2026-08-23 **budget-harness** run on this same build had Direct
WebGPU at 2.50 ms and `navigator.digitclassifier` at 2.70 ms — close together,
both paying the wait. The fence-poll pulled the JS path down to 0.52 ms and left
the C++ path where it was. (Different harness; not poolable, cited for the shape
only.)

## The CPU still wins

At **0.060 ms** the CPU is ~8.7× faster than the fastest GPU path and 62× faster
than the browser-native one. The repo's thesis is untouched: for a 0.6 MFLOP
model, the GPU's request-and-wait costs more than the arithmetic no matter who
issues it — and the leanest GPU implementation available (C++ in Blink) is the
worst of them here precisely because it waits politely.

## Caveats

- **CV on the fence-poll path is high (32.7%)**, as on stock — the poll's latency
  rides event-loop scheduling. The C++ and LiteRT paths are tight (1.7%, 3.0%).
- **Readout deviations**, noted for honesty: driven over CDP (enabled via
  `set-debug-app`, not the stock discovery); and isolated fresh tabs per page with
  others closed, rather than a force-stop between pages, because force-stopping
  this build re-triggers session restore. The timed region is in-page either way.
- Same drawn "1" across all four; release predictions consistent.
