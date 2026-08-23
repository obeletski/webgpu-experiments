# Direct WebGPU Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `webgpu.html`, a self-contained page that classifies a hand-drawn digit by driving the WebGPU API directly, with switchable fused (1 dispatch) and per-layer (3 dispatch) execution, so the cost of dispatch and readback can be measured against LiteRT.js.

**Architecture:** One HTML file holding everything — markup, CSS, drawing, a minimal TFLite flatbuffer reader, WGSL shaders, the WebGPU engine, timing and UI. It parses `digit_classifier.tflite` in the browser and uploads the weights to GPU buffers once. Nothing is shared with `index.html`; the two pages are independent by decision.

**Tech Stack:** Vanilla ES modules, WebGPU, WGSL. No build step, no dependencies. Node (built-ins only) for the parser unit test; headless Chrome plus an HTTP collector for browser tests.

**Spec:** `docs/superpowers/specs/2026-08-23-webgpu-direct-design.md`

## Global Constraints

- **Self-contained.** Everything lives in `webgpu.html`. Do not create shared modules and do not refactor `index.html`. Its only edit is one added link.
- **Drawing must never break.** No import, model load, GPU failure or shader error may prevent the canvas from drawing. Keep drawing code free of imports and top-level `await`, exactly as `index.html` does.
- **Timing must match `index.html` verbatim:** `TIMING_MIN_MS = 5`, `TIMING_MAX_RUNS = 50`, one untimed warm-up run, then `do { ... } while (elapsed < TIMING_MIN_MS && runs < TIMING_MAX_RUNS - 1)`, reporting `elapsed / runs` and the run count. Durations that round to zero render as `<1 ms`, never `0.00 ms`.
- **Model layout is fixed:** input 2352 f32; `W1[128,2352]`, `b1[128]`, ReLU; `W2[10,128]`, `b2[10]`; softmax to 10. Weights are row-major `[out, in]`, indexed `W[o * in + i]`.
- **Preprocessing:** fill the 28×28 canvas opaque black, downscale, grayscale `0.299R + 0.587G + 0.114B` normalized to `[0,1]`, then write each value three times (NHWC, 3 channels).
- **Layout:** must fit without scrolling from 320×480 upward, and panel width must not react to result text.
- **Scratch path:** `SP=/tmp/claude-1000/-mnt-c-Users-huawei-ai-html-litert/a880ee42-9482-4e02-9a8a-e6c833255ca5/scratchpad`. Test harnesses live there and are **not** committed, matching this repo's convention.
- **Chrome:** `/mnt/c/Program Files/Google/Chrome/Application/chrome.exe`. A software WebGPU adapter needs `--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader --enable-features=Vulkan,WebGPUService`.

### Reference values (from the real weight file — use these as expected values)

| Tensor | Shape | First 4 values | Sum |
| --- | --- | --- | --- |
| `W1` | `[128,2352]` | `0.03497049, -0.02951273, 0.02520331, 0.03269367` | `-1685.120657` |
| `b1` | `[128]` | `-0.02302678, -0.02484234, 0.03431824, -0.00699658` | `0.289120` |
| `W2` | `[10,128]` | `-0.30491373, 0.08268756, 0.03659625, 0.07241394` | `-36.477740` |
| `b2` | `[10]` | `-0.10003978, -0.08790810, 0.01560923, -0.06118724, 0.05900016, 0.00970219, -0.02434224, -0.02576081, 0.15351941, -0.01182457` | — |

Total parameters: **302,474**.

**Test vector A — all-zero input (2352 zeros):**
`[0.078401, 0.087700, 0.102487, 0.077888, 0.104479, 0.142801, 0.098027, 0.124000, 0.101656, 0.082562]`, argmax **5**.

**Test vector B — all-0.5 input:**
`[0.000000, 0.000000, 0.048560, 0.001220, 0.000000, 0.949500, 0.000397, 0.000126, 0.000197, 0.000000]`, argmax **5**.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `webgpu.html` | **Create.** The entire experiment: UI, drawing, parser, engine, shaders, timing |
| `index.html` | **Modify.** Add one link to `webgpu.html`. No other change |
| `README.md` | **Modify.** New experiment section; replace the estimated parameter count with 302,474 |
| `$SP/collect.js` | Scratch. Static server on :8099 that also accepts `POST /report` |
| `$SP/drive.html` | Scratch. Driver page, rewritten per task |
| `$SP/parser_test.mjs` | Scratch. Node unit test for the parser |

---

### Task 1: Page skeleton with working drawing

Everything else depends on the canvas working, and a dead canvas is exactly how this project's first bug manifested. Build and prove that first, before any GPU code exists.

**Files:**
- Create: `webgpu.html`
- Create: `$SP/collect.js`, `$SP/drive.html`

**Interfaces:**
- Consumes: nothing
- Produces: `webgpu.html` serving at `/webgpu.html` with `#paintCanvas`, `#clearBtn`, `#predictBtn`, `#result`, `#modeInfo`, `#fusedBtn`, `#layeredBtn`, and a `.app-container`. Page-local `const resultDiv = document.getElementById('result')` and
  `preprocess(channels) -> { input: Float32Array, ink: number }`.

- [ ] **Step 1: Write the collector**

```bash
mkdir -p "$SP"
cat > "$SP/collect.js" <<'EOF'
const http=require('http'),fs=require('fs'),path=require('path');
const ROOT='/mnt/c/Users/huawei/ai/html-litert';
const MIME={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8',
  '.json':'application/json','.wasm':'application/wasm','.tflite':'application/octet-stream'};
http.createServer((req,res)=>{
  if(req.method==='POST'&&req.url==='/report'){
    let b='';req.on('data',c=>b+=c);
    req.on('end',()=>{console.log('REPORT '+b);res.writeHead(204).end();});
    return;
  }
  const u=decodeURIComponent(req.url.split('?')[0]);
  const f=path.join(ROOT,u==='/'?'index.html':u);
  if(!f.startsWith(ROOT)){res.writeHead(403).end();return;}
  fs.readFile(f,(e,buf)=>{
    if(e){res.writeHead(404).end('404');return;}
    res.writeHead(200,{'Content-Type':MIME[path.extname(f)]||'application/octet-stream',
      'Cache-Control':'no-store'});
    res.end(buf);
  });
}).listen(8099,'0.0.0.0',()=>console.log('collector on 8099'));
EOF
sh -c "nohup node $SP/collect.js > $SP/collect.log 2>&1 &"
sleep 1; cat "$SP/collect.log"
```

- [ ] **Step 2: Write the failing driver test**

```bash
cat > "$SP/drive.html" <<'EOF'
<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<iframe id="f" src="/webgpu.html" width="420" height="820" style="border:0"></iframe>
<script>
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const say=o=>fetch('/report',{method:'POST',body:JSON.stringify(o)}).catch(()=>{});
f.addEventListener('load',async()=>{
 const w=f.contentWindow,d=f.contentDocument,errs=[];
 w.addEventListener('error',e=>errs.push('ERR:'+(e.message||e.type)));
 await wait(1500);
 const c=d.getElementById('paintCanvas');
 if(!c){await say({phase:'fail',reason:'no #paintCanvas'});return;}
 const r=c.getBoundingClientRect();
 const mk=(t,fx,fy)=>c.dispatchEvent(new w.PointerEvent(t,{bubbles:true,cancelable:true,
   clientX:r.left+r.width*fx,clientY:r.top+r.height*fy,buttons:1,button:0,
   pointerId:1,pointerType:'mouse',isPrimary:true,view:w}));
 mk('pointerdown',0.5,0.16);
 for(let i=0;i<=20;i++) mk('pointermove',0.5,0.16+i*0.68/20);
 mk('pointerup',0.5,0.84);
 const px=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
 let lit=0; for(let i=0;i<px.length;i+=4) if(px[i+3]>10&&px[i]>40) lit++;
 const panel=d.querySelector('.app-container').getBoundingClientRect().width;
 d.getElementById('result').textContent='X'.repeat(400);
 const panel2=d.querySelector('.app-container').getBoundingClientRect().width;
 await say({phase:'draw',lit,drawPass:lit>200,
   panel:Math.round(panel),widthStable:Math.round(panel)===Math.round(panel2),
   vscroll:d.documentElement.scrollHeight>w.innerHeight+1,errors:errs});
 await say({phase:'done'});
});
</script></body></html>
EOF
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --disable-gpu --virtual-time-budget=20000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t1' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log"
```

Expected: `404` for `/webgpu.html`, no `REPORT` lines — the page does not exist yet.

- [ ] **Step 3: Create `webgpu.html`**

Copy the `<style>` block, the `<h1>`/`<p>`/`.app-container` markup, the pointer-event drawing code and `preprocessCanvas` from `index.html` verbatim, with these changes:

- `<title>` → `Direct WebGPU MNIST Classifier`, `<h1>` → `MNIST — Direct WebGPU`
- The two backend buttons become `id="fusedBtn"` (`fused · 1 dispatch`) and `id="layeredBtn"` (`per-layer · 3 dispatches`)
- `id="backendInfo"` becomes `id="modeInfo"`, same `.backend-info` class
- Add near the top of the module, above the drawing code:

```js
// Sibling experiment: index.html runs the same model through LiteRT.js.
// The timing code below is duplicated from it deliberately and MUST stay
// identical, or the two pages' numbers stop being comparable.
```

- Rename `preprocessCanvas(channels)` to `preprocess(channels)` **and rename its returned
  field `inputBuffer` to `input`**, so it returns `{ input, ink }`. Later tasks destructure
  `{ input, ink }`; leaving it as `inputBuffer` yields `undefined` at the first Classify.
- Leave `#predictBtn` wired to a handler that sets `resultDiv.innerText = 'Engine not built yet.'` for now.
- Add a link back: `<p class="alt"><a href="./index.html">LiteRT.js version →</a></p>` after the container, styled `.alt a { color:#4aa3e0; font-size:0.85rem; }`.

- [ ] **Step 4: Run the driver test**

```bash
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --disable-gpu --virtual-time-budget=20000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t1b' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -2
```

Expected: `"drawPass":true`, `"widthStable":true`, `"vscroll":false`, `"errors":[]`.

- [ ] **Step 5: Verify the layout across viewports**

```bash
cat > "$SP/fit.html" <<'EOF'
<!DOCTYPE html><html><head><meta charset="utf-8"></head><body><div id="out">…</div>
<script>
const SIZES=[[320,480],[360,640],[360,780],[412,915],[768,1024],[1280,800],[1280,600]];
const rows=[];let done=0;
SIZES.forEach(([w,h])=>{
  const f=document.createElement('iframe');
  f.width=w;f.height=h;f.src='/webgpu.html';
  f.addEventListener('load',()=>setTimeout(()=>{
    const d=f.contentDocument,W=f.contentWindow;
    const kids=[...d.body.children];
    const top=Math.min(...kids.map(e=>e.getBoundingClientRect().top));
    const bot=Math.max(...kids.map(e=>e.getBoundingClientRect().bottom));
    const p0=Math.round(d.querySelector('.app-container').getBoundingClientRect().width);
    d.getElementById('result').textContent='X'.repeat(400);
    const p1=Math.round(d.querySelector('.app-container').getBoundingClientRect().width);
    rows.push(`${w}x${h} content ${Math.round(bot-top)} vscroll `+
      (d.documentElement.scrollHeight>W.innerHeight+1?'YES':'no')+
      ` hscroll `+(d.documentElement.scrollWidth>W.innerWidth+1?'YES':'no')+
      ` panel ${p0}`+(p0===p1?'':' WIDTH-CHANGED'));
    if(++done===SIZES.length) document.getElementById('out').textContent='RESULTS\n'+rows.join('\n');
  },1200));
  document.body.appendChild(f);
});
</script></body></html>
EOF
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox   --disable-gpu --virtual-time-budget=30000 --window-size=1600,1400   --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t1fit'   --dump-dom http://localhost:8099/fit.html 2>/dev/null | sed -n '/RESULTS/,/<\/div>/p' | sed 's/<[^>]*>//g'
```

Expected: every row `vscroll no`, `hscroll no`, and no `WIDTH-CHANGED`.

- [ ] **Step 6: Commit**

```bash
git add webgpu.html
git commit -m "Add webgpu.html skeleton with working drawing canvas"
```

---

### Task 2: TFLite weight parser

**Files:**
- Modify: `webgpu.html` (add the parser block)
- Create: `$SP/parser_test.mjs`

**Interfaces:**
- Consumes: nothing
- Produces: `parseWeights(bytes: ArrayBuffer) -> { W1: Float32Array, b1: Float32Array, W2: Float32Array, b2: Float32Array, inputSize: 2352, hidden: 128, classes: 10 }`. Throws `Error` with a specific message on shape, dtype or operator-count mismatch. The block is delimited by the exact sentinel comments `// --- tflite-parser:start ---` and `// --- tflite-parser:end ---` so the Node test can extract it; **do not remove or reword the sentinels.**

- [ ] **Step 1: Write the failing test**

```bash
cat > "$SP/parser_test.mjs" <<'EOF'
import fs from 'node:fs';
const ROOT='/mnt/c/Users/huawei/ai/html-litert';
const html=fs.readFileSync(`${ROOT}/webgpu.html`,'utf8');
const m=html.match(/\/\/ --- tflite-parser:start ---([\s\S]*?)\/\/ --- tflite-parser:end ---/);
if(!m) throw new Error('parser sentinels not found in webgpu.html');
const tmp='/tmp/parser_extracted.mjs';
fs.writeFileSync(tmp, m[1] + '\nexport { parseWeights };\n');
const { parseWeights } = await import(tmp + '?v=' + Date.now());

const buf = fs.readFileSync(`${ROOT}/digit_classifier.tflite`);
const w = parseWeights(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));

const near=(a,b,t=1e-6)=>Math.abs(a-b)<=t;
const sum=a=>a.reduce((x,y)=>x+y,0);
const checks=[
  ['W1 length', w.W1.length===128*2352],
  ['b1 length', w.b1.length===128],
  ['W2 length', w.W2.length===10*128],
  ['b2 length', w.b2.length===10],
  ['W1[0]', near(w.W1[0], 0.03497049, 1e-7)],
  ['W1[1]', near(w.W1[1], -0.02951273, 1e-7)],
  ['b1[0]', near(w.b1[0], -0.02302678, 1e-7)],
  ['W2[0]', near(w.W2[0], -0.30491373, 1e-7)],
  ['b2[8]', near(w.b2[8], 0.15351941, 1e-7)],
  ['W1 sum', near(sum(w.W1), -1685.120657, 0.05)],
  ['b1 sum', near(sum(w.b1), 0.289120, 1e-3)],
  ['W2 sum', near(sum(w.W2), -36.477740, 1e-3)],
  ['param count', w.W1.length+w.b1.length+w.W2.length+w.b2.length===302474],
  ['dims', w.inputSize===2352 && w.hidden===128 && w.classes===10],
];
let bad=0;
for(const [name,ok] of checks){ if(!ok){bad++; console.log('FAIL',name);} }
console.log(bad===0 ? 'PARSER OK ('+checks.length+' checks)' : 'PARSER FAILED: '+bad);
process.exit(bad===0?0:1);
EOF
node "$SP/parser_test.mjs"
```

Expected: FAIL — `parser sentinels not found in webgpu.html`.

- [ ] **Step 2: Add the parser to `webgpu.html`**

Insert inside the module, after the drawing code:

```js
// --- tflite-parser:start ---
// Minimal TFLite flatbuffer reader. A table at `pos` has its vtable at
// `pos - i32(pos)`; field n lives at `pos + u16(vtable + 4 + 2n)`, absent when
// that is 0 or beyond u16(vtable). Vectors and strings are a u32 length
// followed by data, reached through a u32 offset relative to its own position.
function parseWeights(bytes) {
  const dv = new DataView(bytes);
  const u8 = (p) => dv.getUint8(p);
  const i8 = (p) => dv.getInt8(p);
  const u16 = (p) => dv.getUint16(p, true);
  const i32 = (p) => dv.getInt32(p, true);
  const u32 = (p) => dv.getUint32(p, true);

  const table = (pos) => ({ pos, vt: pos - i32(pos) });
  const fieldOff = (t, f) => {
    const vtSize = u16(t.vt);
    const idx = 4 + f * 2;
    return idx < vtSize ? u16(t.vt + idx) : 0;
  };
  const scalar = (t, f, read, dflt) => {
    const o = fieldOff(t, f);
    return o ? read(t.pos + o) : dflt;
  };
  const vector = (t, f) => {
    const o = fieldOff(t, f);
    if (!o) return { start: 0, len: 0 };
    let p = t.pos + o;
    p = p + u32(p);
    return { start: p + 4, len: u32(p) };
  };
  const vecTables = (t, f) => {
    const { start, len } = vector(t, f);
    const out = [];
    for (let i = 0; i < len; i++) {
      const p = start + 4 * i;
      out.push(table(p + u32(p)));
    }
    return out;
  };
  const vecI32 = (t, f) => {
    const { start, len } = vector(t, f);
    const out = [];
    for (let i = 0; i < len; i++) out.push(i32(start + 4 * i));
    return out;
  };

  const model = table(u32(0));
  const opcodes = vecTables(model, 1).map((oc) => {
    const builtin = scalar(oc, 3, i32, 0);
    return builtin || scalar(oc, 0, i8, 0);
  });
  const buffers = vecTables(model, 4);
  const sub = vecTables(model, 2)[0];
  const tensors = vecTables(sub, 0);

  const FULLY_CONNECTED = 9;
  const fc = vecTables(sub, 3)
    .filter((op) => opcodes[scalar(op, 0, u32, 0)] === FULLY_CONNECTED);
  if (fc.length !== 2) {
    throw new Error(`expected 2 FULLY_CONNECTED operators, found ${fc.length}`);
  }

  const readTensor = (idx, label) => {
    const t = tensors[idx];
    const type = scalar(t, 1, u8, 0);
    if (type !== 0) throw new Error(`${label}: tensor type ${type}, expected float32 (0)`);
    const { start, len } = vector(buffers[scalar(t, 2, u32, 0)], 0);
    if (!len) throw new Error(`${label}: tensor has no buffer data`);
    // ubyte vector payloads carry no float alignment guarantee - copy first.
    return { shape: vecI32(t, 0), data: new Float32Array(bytes.slice(start, start + len)) };
  };

  const w1 = readTensor(vecI32(fc[0], 1)[1], 'W1');
  const bias1 = readTensor(vecI32(fc[0], 1)[2], 'b1');
  const w2 = readTensor(vecI32(fc[1], 1)[1], 'W2');
  const bias2 = readTensor(vecI32(fc[1], 1)[2], 'b2');

  const expect = (t, want, name) => {
    if (String(t.shape) !== String(want)) {
      throw new Error(`${name}: shape [${t.shape}], expected [${want}]`);
    }
  };
  expect(w1, [128, 2352], 'W1');
  expect(bias1, [128], 'b1');
  expect(w2, [10, 128], 'W2');
  expect(bias2, [10], 'b2');

  return {
    W1: w1.data, b1: bias1.data, W2: w2.data, b2: bias2.data,
    inputSize: 2352, hidden: 128, classes: 10,
  };
}
// --- tflite-parser:end ---
```

- [ ] **Step 3: Run the test**

```bash
node "$SP/parser_test.mjs"
```

Expected: `PARSER OK (14 checks)`, exit 0.

- [ ] **Step 4: Commit**

```bash
git add webgpu.html
git commit -m "Parse tflite weights in the browser for the WebGPU page"
```

---

### Task 3: WebGPU engine with the fused pipeline

**Files:**
- Modify: `webgpu.html`
- Modify: `$SP/drive.html`

**Interfaces:**
- Consumes: `parseWeights` from Task 2
- Produces: `createEngine(device, weights) -> engine` where `engine.setMode(mode: 'fused'|'layered') -> void` builds and caches that mode's pipelines, and `engine.run(input: Float32Array) -> Promise<Float32Array>` returns 10 probabilities. Also `getDevice() -> Promise<GPUDevice>`, throwing a specific `Error` when WebGPU or an adapter is unavailable.

The spec calls for a JS matvec oracle. Test vectors A and B replace it: they were
computed independently, in Python, straight from the same weight file, so they are a
stronger check than a JS reimplementation that could share a misreading of the layout.

- [ ] **Step 1: Write the failing test**

```bash
cat > "$SP/drive.html" <<'EOF'
<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<iframe id="f" src="/webgpu.html" width="420" height="820" style="border:0"></iframe>
<script>
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const say=o=>fetch('/report',{method:'POST',body:JSON.stringify(o)}).catch(()=>{});
const EXPECT_ZERO=[0.078401,0.087700,0.102487,0.077888,0.104479,
                   0.142801,0.098027,0.124000,0.101656,0.082562];
f.addEventListener('load',async()=>{
 const w=f.contentWindow,d=f.contentDocument;
 for(let i=0;i<60 && !w.__engine;i++) await wait(500);
 if(!w.__engine){await say({phase:'fail',reason:'window.__engine never appeared'});
   await say({phase:'done'});return;}
 const out=await w.__engine.run(new Float32Array(2352));
 const got=Array.from(out);
 let maxDiff=0;
 for(let i=0;i<10;i++) maxDiff=Math.max(maxDiff,Math.abs(got[i]-EXPECT_ZERO[i]));
 const s=got.reduce((a,b)=>a+b,0);
 await say({phase:'zero-input',got:got.map(v=>+v.toFixed(6)),maxDiff:+maxDiff.toFixed(8),
   pass:maxDiff<1e-4, sumsToOne:Math.abs(s-1)<1e-4,
   argmax:got.indexOf(Math.max(...got))});
 await say({phase:'done'});
});
</script></body></html>
EOF
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader \
  --enable-features=Vulkan,WebGPUService --virtual-time-budget=60000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t3' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -2
```

Expected: `"reason":"window.__engine never appeared"`.

- [ ] **Step 2: Add the fused shader and engine**

Add to `webgpu.html`'s module:

```js
const FUSED_WGSL = `
@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(1) var<storage, read>       w1     : array<f32>;
@group(0) @binding(2) var<storage, read>       bias1  : array<f32>;
@group(0) @binding(3) var<storage, read>       w2     : array<f32>;
@group(0) @binding(4) var<storage, read>       bias2  : array<f32>;
@group(0) @binding(5) var<storage, read_write> result : array<f32>;

var<workgroup> hidden : array<f32, 128>;
var<workgroup> logits : array<f32, 10>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;

  var acc = bias1[t];
  let base = t * 2352u;
  for (var i = 0u; i < 2352u; i = i + 1u) {
    acc = acc + w1[base + i] * input[i];
  }
  hidden[t] = max(acc, 0.0);

  workgroupBarrier();

  if (t < 10u) {
    var acc2 = bias2[t];
    let base2 = t * 128u;
    for (var i = 0u; i < 128u; i = i + 1u) {
      acc2 = acc2 + w2[base2 + i] * hidden[i];
    }
    logits[t] = acc2;
  }

  workgroupBarrier();

  if (t == 0u) {
    var m = logits[0];
    for (var i = 1u; i < 10u; i = i + 1u) { m = max(m, logits[i]); }
    var s = 0.0;
    for (var i = 0u; i < 10u; i = i + 1u) { s = s + exp(logits[i] - m); }
    for (var i = 0u; i < 10u; i = i + 1u) { result[i] = exp(logits[i] - m) / s; }
  }
}`;

async function getDevice() {
  if (!navigator.gpu) throw new Error('This browser does not support WebGPU.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter available in this browser.');
  return adapter.requestDevice();
}

function createEngine(device, weights) {
  const S = GPUBufferUsage.STORAGE;
  const mk = (data, usage) => {
    const b = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
    new Float32Array(b.getMappedRange()).set(data);
    b.unmap();
    return b;
  };
  const empty = (bytes, usage) => device.createBuffer({ size: bytes, usage });

  const buf = {
    input:  empty(2352 * 4, S | GPUBufferUsage.COPY_DST),
    w1:     mk(weights.W1, S),
    b1:     mk(weights.b1, S),
    w2:     mk(weights.W2, S),
    b2:     mk(weights.b2, S),
    result: empty(10 * 4, S | GPUBufferUsage.COPY_SRC),
    read:   empty(10 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
  };

  const built = {};
  let mode = 'fused';

  function buildFused() {
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: FUSED_WGSL }), entryPoint: 'main' },
    });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [buf.input, buf.w1, buf.b1, buf.w2, buf.b2, buf.result]
        .map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    return { passes: [{ pipeline, bind, groups: 1 }] };
  }

  function setMode(next) {
    mode = next;
    if (!built[next]) built[next] = next === 'fused' ? buildFused() : buildLayered();
  }

  async function run(input) {
    device.queue.writeBuffer(buf.input, 0, input);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const p of built[mode].passes) {
      pass.setPipeline(p.pipeline);
      pass.setBindGroup(0, p.bind);
      pass.dispatchWorkgroups(p.groups);
    }
    pass.end();
    enc.copyBufferToBuffer(buf.result, 0, buf.read, 0, 40);
    device.queue.submit([enc.finish()]);
    await buf.read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(buf.read.getMappedRange().slice(0));
    buf.read.unmap();
    return out;
  }

  setMode('fused');
  return { setMode, run, get mode() { return mode; } };
}
```

`buildLayered` arrives in Task 4; until then `setMode('layered')` will throw, which is correct — nothing calls it yet.

- [ ] **Step 3: Wire startup**

At the bottom of the module, replacing the placeholder Classify handler:

```js
let engine;

async function initEngine() {
  resultDiv.innerText = 'Loading weights...';
  const res = await fetch('./digit_classifier.tflite');
  if (!res.ok) throw new Error(`Could not fetch the model: ${res.status}`);
  const weights = parseWeights(await res.arrayBuffer());
  const device = await getDevice();
  device.lost.then((info) => {
    console.error('WebGPU device lost:', info);
    resultDiv.innerText = 'GPU device lost. Reload the page.';
  });
  engine = createEngine(device, weights);
  window.__engine = engine;              // test hook
  resultDiv.innerText = 'Ready (fused).';
}

initEngine().catch((err) => {
  console.error(err);
  resultDiv.innerText = `${err.message} Drawing still works.`;
});
```

- [ ] **Step 4: Run the test**

```bash
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader \
  --enable-features=Vulkan,WebGPUService --virtual-time-budget=60000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t3b' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -2
```

Expected: `"pass":true`, `"sumsToOne":true`, `"argmax":5`, `maxDiff < 1e-4`.

- [ ] **Step 5: Commit**

```bash
git add webgpu.html
git commit -m "Add WebGPU engine with fused single-dispatch pipeline"
```

---

### Task 4: Per-layer pipeline and the mode toggle

**Files:**
- Modify: `webgpu.html`
- Modify: `$SP/drive.html`

**Interfaces:**
- Consumes: `createEngine`, `engine.setMode`, `engine.run` from Task 3
- Produces: `buildLayered()` inside `createEngine`; `engine.setMode('layered')` becomes valid. Buttons `#fusedBtn` / `#layeredBtn` carry `aria-pressed`.

- [ ] **Step 1: Write the failing test**

```bash
cat > "$SP/drive.html" <<'EOF'
<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<iframe id="f" src="/webgpu.html" width="420" height="820" style="border:0"></iframe>
<script>
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const say=o=>fetch('/report',{method:'POST',body:JSON.stringify(o)}).catch(()=>{});
const EXPECT_HALF=[0.000000,0.000000,0.048560,0.001220,0.000000,
                   0.949500,0.000397,0.000126,0.000197,0.000000];
f.addEventListener('load',async()=>{
 const w=f.contentWindow,d=f.contentDocument;
 for(let i=0;i<60 && !w.__engine;i++) await wait(500);
 if(!w.__engine){await say({phase:'fail',reason:'no engine'});await say({phase:'done'});return;}
 const half=new Float32Array(2352).fill(0.5);
 w.__engine.setMode('fused');
 const a=Array.from(await w.__engine.run(half));
 let err=null, b=null;
 try { w.__engine.setMode('layered'); b=Array.from(await w.__engine.run(half)); }
 catch(e){ err=String(e && e.message || e); }
 let agree=0, vsRef=0;
 if(b){ for(let i=0;i<10;i++){ agree=Math.max(agree,Math.abs(a[i]-b[i]));
                               vsRef=Math.max(vsRef,Math.abs(b[i]-EXPECT_HALF[i])); } }
 await say({phase:'modes',err,
   fusedArgmax:a.indexOf(Math.max(...a)),
   fusedVsRef:+Math.max(...a.map((v,i)=>Math.abs(v-EXPECT_HALF[i]))).toFixed(8),
   modesAgree:b?+agree.toFixed(8):null, layeredVsRef:b?+vsRef.toFixed(8):null,
   pass: !!b && agree<1e-4 && vsRef<1e-4});
 const fb=d.getElementById('fusedBtn'), lb=d.getElementById('layeredBtn');
 lb.click(); await wait(1200);
 await say({phase:'toggle',fused:fb.getAttribute('aria-pressed'),
   layered:lb.getAttribute('aria-pressed')});
 await say({phase:'done'});
});
</script></body></html>
EOF
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader \
  --enable-features=Vulkan,WebGPUService --virtual-time-budget=60000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t4' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -3
```

Expected: `"err"` is non-null (`buildLayered is not defined`), `"pass":false`.

- [ ] **Step 2: Add the three layered shaders**

```js
const LAYER1_WGSL = `
@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(1) var<storage, read>       w1     : array<f32>;
@group(0) @binding(2) var<storage, read>       bias1  : array<f32>;
@group(0) @binding(3) var<storage, read_write> hidden : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = gid.x;
  if (o >= 128u) { return; }
  var acc = bias1[o];
  let base = o * 2352u;
  for (var i = 0u; i < 2352u; i = i + 1u) {
    acc = acc + w1[base + i] * input[i];
  }
  hidden[o] = max(acc, 0.0);
}`;

const LAYER2_WGSL = `
@group(0) @binding(0) var<storage, read>       hidden : array<f32>;
@group(0) @binding(1) var<storage, read>       w2     : array<f32>;
@group(0) @binding(2) var<storage, read>       bias2  : array<f32>;
@group(0) @binding(3) var<storage, read_write> logits : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = gid.x;
  if (o >= 10u) { return; }
  var acc = bias2[o];
  let base = o * 128u;
  for (var i = 0u; i < 128u; i = i + 1u) {
    acc = acc + w2[base + i] * hidden[i];
  }
  logits[o] = acc;
}`;

const SOFTMAX_WGSL = `
@group(0) @binding(0) var<storage, read>       logits : array<f32>;
@group(0) @binding(1) var<storage, read_write> result : array<f32>;

@compute @workgroup_size(1)
fn main() {
  var m = logits[0];
  for (var i = 1u; i < 10u; i = i + 1u) { m = max(m, logits[i]); }
  var s = 0.0;
  for (var i = 0u; i < 10u; i = i + 1u) { s = s + exp(logits[i] - m); }
  for (var i = 0u; i < 10u; i = i + 1u) { result[i] = exp(logits[i] - m) / s; }
}`;
```

- [ ] **Step 3: Add `buildLayered` inside `createEngine`**

Add these two buffers next to the others in `buf`:

```js
    hidden: empty(128 * 4, S),
    logits: empty(10 * 4, S),
```

Then, beside `buildFused`:

```js
  function buildLayered() {
    const stage = (code, buffers, groups) => {
      const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
      });
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
      });
      return { pipeline, bind, groups };
    };
    // Three dispatches recorded into one compute pass: WebGPU orders them and
    // makes each dispatch's writes visible to the next, so no explicit barrier.
    return {
      passes: [
        stage(LAYER1_WGSL,  [buf.input, buf.w1, buf.b1, buf.hidden], 2),
        stage(LAYER2_WGSL,  [buf.hidden, buf.w2, buf.b2, buf.logits], 1),
        stage(SOFTMAX_WGSL, [buf.logits, buf.result], 1),
      ],
    };
  }
```

- [ ] **Step 4: Wire the toggle buttons**

```js
const fusedBtn = document.getElementById('fusedBtn');
const layeredBtn = document.getElementById('layeredBtn');

function paintToggle() {
  fusedBtn.setAttribute('aria-pressed', String(engine && engine.mode === 'fused'));
  layeredBtn.setAttribute('aria-pressed', String(engine && engine.mode === 'layered'));
}

async function switchMode(next) {
  if (!engine || engine.mode === next) return;
  engine.setMode(next);
  paintToggle();
  resultDiv.innerText = `Ready (${next}).`;
}

fusedBtn.addEventListener('click', () => switchMode('fused'));
layeredBtn.addEventListener('click', () => switchMode('layered'));
```

Call `paintToggle()` at the end of `initEngine`.

- [ ] **Step 5: Run the test**

```bash
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader \
  --enable-features=Vulkan,WebGPUService --virtual-time-budget=60000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t4b' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -3
```

Expected: `"err":null`, `"pass":true`, `"fusedArgmax":5`, `modesAgree < 1e-4`, `layeredVsRef < 1e-4`, and toggle showing `"fused":"false"`, `"layered":"true"`.

- [ ] **Step 6: Commit**

```bash
git add webgpu.html
git commit -m "Add per-layer three-dispatch pipeline and mode toggle"
```

---

### Task 5: Timing and UI

**Files:**
- Modify: `webgpu.html`
- Modify: `$SP/drive.html`

**Interfaces:**
- Consumes: `engine.run`, `engine.setMode`, `preprocess`, `ms` from earlier tasks
- Produces: `timedInfer(input) -> Promise<{values, mean, runs}>`; `#modeInfo` shows two `.timing` lines (`pipeline`, `cold start`); `#result` shows prediction plus one steady-state figure.

- [ ] **Step 1: Write the failing test**

```bash
cat > "$SP/drive.html" <<'EOF'
<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<iframe id="f" src="/webgpu.html" width="420" height="820" style="border:0"></iframe>
<script>
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const say=o=>fetch('/report',{method:'POST',body:JSON.stringify(o)}).catch(()=>{});
const lines=el=>{const o=[];for(const n of el.childNodes){
  if(n.nodeType===3){const t=n.textContent.trim();if(t)o.push(t);}
  else if(n.classList&&n.classList.contains('confidence')){
    for(const c of n.childNodes){const t=c.textContent.trim();if(t)o.push(t);}}
  else {const t=n.textContent.trim();if(t)o.push(t);}} return o;};
f.addEventListener('load',async()=>{
 const w=f.contentWindow,d=f.contentDocument,errs=[];
 w.addEventListener('error',e=>errs.push('ERR:'+(e.message||e.type)));
 for(let i=0;i<60 && !w.__engine;i++) await wait(500);
 const c=d.getElementById('paintCanvas'),r=c.getBoundingClientRect();
 const mk=(t,fx,fy)=>c.dispatchEvent(new w.PointerEvent(t,{bubbles:true,cancelable:true,
   clientX:r.left+r.width*fx,clientY:r.top+r.height*fy,buttons:1,button:0,
   pointerId:1,pointerType:'mouse',isPrimary:true,view:w}));
 mk('pointerdown',0.5,0.16);
 for(let i=0;i<=20;i++) mk('pointermove',0.5,0.16+i*0.68/20);
 mk('pointerup',0.5,0.84);
 const snaps=[];
 for(let n=0;n<3;n++){
   d.getElementById('predictBtn').click(); await wait(4000);
   snaps.push({mode:lines(d.getElementById('modeInfo')),
               result:lines(d.getElementById('result'))});
 }
 const infoStable = JSON.stringify(snaps[0].mode)===JSON.stringify(snaps[2].mode);
 await say({phase:'timing',snaps,infoStable,
   hasPipeline:snaps[0].mode.some(s=>s.startsWith('pipeline')),
   hasCold:snaps[0].mode.some(s=>s.startsWith('cold start')),
   emptyGuard:null,errors:errs});
 d.getElementById('clearBtn').click();
 d.getElementById('predictBtn').click(); await wait(1500);
 await say({phase:'empty',result:d.getElementById('result').textContent});
 await say({phase:'done'});
});
</script></body></html>
EOF
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader \
  --enable-features=Vulkan,WebGPUService --virtual-time-budget=90000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t5' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -3
```

Expected: `"hasPipeline":false`, `"hasCold":false` — `#modeInfo` is still empty.

- [ ] **Step 2: Add the timing helpers**

Copied verbatim from `index.html` — do not adjust the constants:

```js
const TIMING_MIN_MS = 5;
const TIMING_MAX_RUNS = 50;

function ms(value) {
  return value > 0 ? `${value.toFixed(2)} ms` : '<1 ms';
}

async function timedInfer(input) {
  let values = await engine.run(input);

  const started = performance.now();
  let runs = 0;
  let elapsed = 0;

  do {
    values = await engine.run(input);
    runs++;
    elapsed = performance.now() - started;
  } while (elapsed < TIMING_MIN_MS && runs < TIMING_MAX_RUNS - 1);

  return { values, mean: elapsed / runs, runs };
}
```

- [ ] **Step 3: Track pipeline and cold-start cost**

Add `let pipelineMs, coldMs;` beside `let engine;`. Replace `setMode` calls with a helper that times them:

```js
async function useMode(next) {
  const pipelineStarted = performance.now();
  engine.setMode(next);
  pipelineMs = performance.now() - pipelineStarted;

  // First inference after building pipelines pays one-time cost - time it alone.
  const coldStarted = performance.now();
  await engine.run(new Float32Array(2352));
  coldMs = performance.now() - coldStarted;

  paintModeInfo();
  paintToggle();
}

function paintModeInfo() {
  document.getElementById('modeInfo').innerHTML =
    `<span class="timing">pipeline ${ms(pipelineMs)}</span>` +
    `<span class="timing">cold start ${ms(coldMs)}</span>`;
}
```

`switchMode` becomes:

```js
async function switchMode(next) {
  if (!engine || engine.mode === next) return;
  resultDiv.innerText = `Switching to ${next}...`;
  await useMode(next);
  resultDiv.innerText = `Ready on ${next}.`;
}
```

and `initEngine` ends with `await useMode('fused')` then `resultDiv.innerText = 'Ready on fused.'`.

- [ ] **Step 4: Wire Classify**

```js
document.getElementById('predictBtn').addEventListener('click', async () => {
  if (!engine) { resultDiv.innerText = 'Engine is not ready yet.'; return; }
  const { input, ink } = preprocess(3);
  if (ink < 1) { resultDiv.innerText = 'Draw a digit first.'; return; }

  const ranOn = engine.mode;
  resultDiv.innerText = 'Classifying...';
  try {
    const { values, mean, runs } = await timedInfer(input);
    const probs = Array.from(values);
    const digit = probs.indexOf(Math.max(...probs));
    resultDiv.innerHTML = `Predicted: <strong>${digit}</strong> ` +
      `<span class="confidence">` +
        `<span class="timing">(${(probs[digit] * 100).toFixed(1)}% match)</span>` +
        `<span class="timing">${ms(mean)} on ${ranOn}` +
        `${runs > 1 ? ` (mean of ${runs})` : ''}</span>` +
      `</span>`;
  } catch (err) {
    console.error(err);
    resultDiv.innerText = 'Inference failed. Check the console.';
  }
});
```

- [ ] **Step 5: Run the test**

```bash
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader \
  --enable-features=Vulkan,WebGPUService --virtual-time-budget=90000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t5b' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -3
```

Expected: `"hasPipeline":true`, `"hasCold":true`, `"infoStable":true` (the mode line must not change across three Classify clicks), each `result` ending in one `... on fused` line, `"errors":[]`, and the empty-canvas click reporting `Draw a digit first.`

- [ ] **Step 6: Commit**

```bash
git add webgpu.html
git commit -m "Add pipeline, cold-start and steady-state timing to the WebGPU page"
```

---

### Task 6: Failure paths

**Files:**
- Modify: `webgpu.html`
- Modify: `$SP/drive.html`

**Interfaces:**
- Consumes: `initEngine`, `getDevice` from Task 3
- Produces: no new API. Classify and both mode buttons are `disabled` whenever `engine` is undefined.

- [ ] **Step 1: Write the failing test — run with no WebGPU adapter**

```bash
cat > "$SP/drive.html" <<'EOF'
<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<iframe id="f" src="/webgpu.html" width="420" height="820" style="border:0"></iframe>
<script>
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const say=o=>fetch('/report',{method:'POST',body:JSON.stringify(o)}).catch(()=>{});
f.addEventListener('load',async()=>{
 const w=f.contentWindow,d=f.contentDocument,errs=[];
 w.addEventListener('error',e=>errs.push('ERR:'+(e.message||e.type)));
 w.addEventListener('unhandledrejection',e=>errs.push('REJ:'+(e.reason&&e.reason.message||e.reason)));
 await wait(12000);
 const c=d.getElementById('paintCanvas'),r=c.getBoundingClientRect();
 const mk=(t,fx,fy)=>c.dispatchEvent(new w.PointerEvent(t,{bubbles:true,cancelable:true,
   clientX:r.left+r.width*fx,clientY:r.top+r.height*fy,buttons:1,button:0,
   pointerId:1,pointerType:'mouse',isPrimary:true,view:w}));
 mk('pointerdown',0.5,0.16);
 for(let i=0;i<=20;i++) mk('pointermove',0.5,0.16+i*0.68/20);
 mk('pointerup',0.5,0.84);
 const px=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
 let lit=0; for(let i=0;i<px.length;i+=4) if(px[i+3]>10&&px[i]>40) lit++;
 await say({phase:'no-webgpu',status:d.getElementById('result').textContent,
   drawStillWorks:lit>200,
   predictDisabled:d.getElementById('predictBtn').disabled,
   fusedDisabled:d.getElementById('fusedBtn').disabled,
   errors:errs});
 await say({phase:'done'});
});
</script></body></html>
EOF
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --disable-gpu --virtual-time-budget=40000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t6' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -2
```

Expected: `"drawStillWorks":true` already, but `"predictDisabled":false` and `"fusedDisabled":false` — the controls are wrongly live with no engine.

- [ ] **Step 2: Disable controls when there is no engine**

```js
function setEnabled(on) {
  document.getElementById('predictBtn').disabled = !on;
  fusedBtn.disabled = !on;
  layeredBtn.disabled = !on;
}
```

Call `setEnabled(false)` immediately after `fusedBtn`/`layeredBtn` are declared (Task 4 defines them; `setEnabled` must appear after that point), `setEnabled(true)` at the end of a successful `initEngine`, and leave it false in the `initEngine().catch(...)` handler.

Also surface shader compilation diagnostics, which the spec's error table requires and which otherwise fail silently as a wrong-looking result. In `createEngine`, replace bare `device.createShaderModule({ code })` calls with:

```js
  async function checkShaders() {
    for (const [name, code] of Object.entries({
      FUSED_WGSL, LAYER1_WGSL, LAYER2_WGSL, SOFTMAX_WGSL,
    })) {
      const info = await device.createShaderModule({ code }).getCompilationInfo();
      for (const m of info.messages) {
        console[m.type === 'error' ? 'error' : 'warn'](
          `${name}:${m.lineNum}:${m.linePos} ${m.type}: ${m.message}`);
      }
      if (info.messages.some((m) => m.type === 'error')) {
        throw new Error(`${name} failed to compile - see console.`);
      }
    }
  }
```

and `await checkShaders()` once in `initEngine` before `useMode('fused')`. The catch already reports `` `${err.message} Drawing still works.` ``, which covers the no-adapter, no-`navigator.gpu`, fetch-failure and parse-failure messages produced in Tasks 2 and 3.

- [ ] **Step 3: Run the test**

```bash
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --disable-gpu --virtual-time-budget=40000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t6b' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -2
```

Expected: `"status"` contains `No WebGPU adapter available in this browser. Drawing still works.`, `"drawStillWorks":true`, `"predictDisabled":true`, `"fusedDisabled":true`, `"errors":[]`.

- [ ] **Step 4: Re-run the Task 5 happy path to confirm no regression**

```bash
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader \
  --enable-features=Vulkan,WebGPUService --virtual-time-budget=60000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t6c' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -2
```

Expected: with an adapter present, `"predictDisabled":false`.

- [ ] **Step 5: Commit**

```bash
git add webgpu.html
git commit -m "Disable controls when the WebGPU engine is unavailable"
```

---

### Task 7: Agreement with LiteRT, link and README

**Files:**
- Modify: `index.html` (one link only)
- Modify: `README.md`
- Modify: `$SP/drive.html`

**Interfaces:**
- Consumes: the finished `webgpu.html`
- Produces: nothing further

- [ ] **Step 1: Write the failing cross-page test**

```bash
cat > "$SP/drive.html" <<'EOF'
<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<iframe id="a" src="/index.html" width="420" height="820" style="border:0"></iframe>
<iframe id="b" src="/webgpu.html" width="420" height="820" style="border:0"></iframe>
<script>
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const say=o=>fetch('/report',{method:'POST',body:JSON.stringify(o)}).catch(()=>{});
const stroke=(w,d)=>{const c=d.getElementById('paintCanvas'),r=c.getBoundingClientRect();
  const mk=(t,fx,fy)=>c.dispatchEvent(new w.PointerEvent(t,{bubbles:true,cancelable:true,
    clientX:r.left+r.width*fx,clientY:r.top+r.height*fy,buttons:1,button:0,
    pointerId:1,pointerType:'mouse',isPrimary:true,view:w}));
  mk('pointerdown',0.5,0.16);
  for(let i=0;i<=20;i++) mk('pointermove',0.5,0.16+i*0.68/20);
  mk('pointerup',0.5,0.84); return c;};
const digit=t=>{const m=t.match(/Predicted:\s*(\d)/);return m?+m[1]:null;};
window.addEventListener('load',async()=>{
 const A=document.getElementById('a'),B=document.getElementById('b');
 const aw=A.contentWindow,ad=A.contentDocument,bw=B.contentWindow,bd=B.contentDocument;
 for(let i=0;i<80 && !/Ready on/.test(ad.getElementById('result').textContent);i++) await wait(1000);
 for(let i=0;i<80 && !bw.__engine;i++) await wait(500);
 stroke(aw,ad); stroke(bw,bd);
 ad.getElementById('predictBtn').click(); await wait(6000);
 bd.getElementById('predictBtn').click(); await wait(6000);
 const at=ad.getElementById('result').textContent, bt=bd.getElementById('result').textContent;
 await say({phase:'cross',litert:at,webgpu:bt,
   sameDigit:digit(at)!==null && digit(at)===digit(bt),
   link:!!ad.querySelector('a[href$="webgpu.html"]')});
 await say({phase:'done'});
});
</script></body></html>
EOF
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader \
  --enable-features=Vulkan,WebGPUService --virtual-time-budget=150000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t7' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -2
```

Expected: `"sameDigit":true` (both should already agree) but `"link":false`.

- [ ] **Step 2: Add the link to `index.html`**

After the closing `</div>` of `.app-container`, matching the style added to `webgpu.html` in Task 1:

```html
  <p class="alt"><a href="./webgpu.html">Direct WebGPU version →</a></p>
```

and in its `<style>`:

```css
    .alt a { color: #4aa3e0; font-size: 0.85rem; }
```

- [ ] **Step 3: Run the test**

```bash
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-sandbox \
  --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader \
  --enable-features=Vulkan,WebGPUService --virtual-time-budget=150000 \
  --user-data-dir='C:\Users\huawei\AppData\Local\Temp\cc-t7b' \
  --dump-dom http://localhost:8099/drive.html > /dev/null 2>&1
grep REPORT "$SP/collect.log" | tail -2
```

Expected: `"sameDigit":true`, `"link":true`.

- [ ] **Step 4: Confirm `index.html` is otherwise untouched**

```bash
git diff --stat index.html
```

Expected: a small diff — the link and one CSS rule only. If anything else changed, revert it.

- [ ] **Step 5: Update the README**

- Replace the hedged parameter estimate ("an estimate: … order of magnitude, not an exact count") with the exact figure: **302,474 parameters**, `2352 → 128 (ReLU) → 10 → softmax`, parsed from the flatbuffer.
- Add a `## Second experiment: direct WebGPU` section stating the question (does hand-written WebGPU beat LiteRT.js, and what does a dispatch cost), the two dispatch modes, and a placeholder-free note that numbers land in Task 8.
- Add `webgpu.html` and the spec/plan paths to the Files table.

- [ ] **Step 6: Commit**

```bash
git add index.html README.md
git commit -m "Link the two experiments and record the exact model architecture"
```

---

### Task 8: Measure on the Android device

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the finished `webgpu.html`
- Produces: measured numbers for the README

- [ ] **Step 1: Bridge the device and pin its clock**

```bash
ADB="/mnt/c/Users/huawei/AppData/Local/Android/Sdk/platform-tools/adb.exe"
"$ADB" devices -l
"$ADB" reverse tcp:8099 tcp:8099
"$ADB" shell svc power stayon usb
"$ADB" shell settings put global low_power 0
"$ADB" shell cmd power set-fixed-performance-mode-enabled true
```

- [ ] **Step 2: Run 3 × 5 measurements per mode, fused first**

Reuse the driver shape from Task 5, but loop `for (let n = 0; n < 5; n++)` collecting the `ms` figure parsed out of `#result` for each mode, after a 4-second warm-up burst of Classify clicks. Launch with:

```bash
"$ADB" shell am force-stop com.android.chrome
"$ADB" shell am start -a android.intent.action.VIEW \
  -d "http://localhost:8099/drive.html?run=1" com.android.chrome
```

Repeat for `run=2` and `run=3`.

- [ ] **Step 3: Restore the device**

```bash
"$ADB" shell cmd power set-fixed-performance-mode-enabled false
"$ADB" shell svc power stayon false
"$ADB" shell am force-stop com.android.chrome
"$ADB" reverse --remove-all
```

- [ ] **Step 4: Write the results into the README**

Report per mode: pipeline, cold start, and steady-state median with min/max across all 15 measurements. Then answer the two questions the spec set:

1. Fused steady state versus LiteRT's `webgpu` median of **4.30 ms** and its `wasm` median of **0.56 ms**.
2. Per-layer minus fused, as the cost of two extra dispatches.

State plainly if hand-written WebGPU does **not** beat LiteRT — that outcome means the cost is WebGPU's dispatch-and-readback floor, which is the more useful finding.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Measure the direct-WebGPU page on the Android device"
```
