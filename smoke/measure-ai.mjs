// Measure AI bootstrap: engine boot (local loader) + built-in model auto-load + session create + warmup.
// Drives headless Edge over CDP; records resource timing after each phase.
import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';

const HERE = dirname(fileURLToPath(import.meta.url));
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const APP_URL = process.env.APP_URL || 'http://localhost:5175/';
const CDP = 9346;
const MODEL = 'C:\\Users\\13114\\Desktop\\Python项目测试\\搞一个超简单高颜值的YOLO标注工具1\\ort.js推理测试1\\yolo26n.onnx';
const IMG_A = join(HERE, 'fixtures', 'cat-01.png');

class Cdp {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map(); this.listeners = []; ws.addEventListener('message', (ev) => { const m = JSON.parse(ev.data); if (m.id && this.pending.has(m.id)) { const { r, j } = this.pending.get(m.id); this.pending.delete(m.id); m.error ? j(new Error(JSON.stringify(m.error))) : r(m.result); return; } for (const f of this.listeners) f(m); }); }
  static async connect(url) { const ws = new WebSocket(url); await new Promise((res, rej) => { ws.addEventListener('open', res); ws.addEventListener('error', rej); }); return new Cdp(ws); }
  send(method, params = {}, sessionId) { this.id += 1; const id = this.id; const payload = { id, method, params }; if (sessionId) payload.sessionId = sessionId; this.ws.send(JSON.stringify(payload)); return new Promise((resolve, reject) => { this.pending.set(id, { r: resolve, j: reject }); setTimeout(() => { if (this.pending.has(id)) { this.pending.delete(id); reject(new Error('timeout: ' + method)); } }, 90000); }); }
  on(fn) { this.listeners.push(fn); }
}
async function waitFor(fn, timeout = 30000, label = 'cond') { const st = Date.now(); for (;;) { const v = await fn(); if (v) return v; if (Date.now() - st > timeout) throw new Error('timeout ' + label); await sleep(150); } }

const profileDir = join(tmpdir(), 'yolo-ai-' + Date.now());
rmSync(profileDir, { recursive: true, force: true });
const edge = spawn(EDGE, ['--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-extensions', '--hide-scrollbars', '--window-size=1440,900', `--remote-debugging-port=${CDP}`, `--user-data-dir=${profileDir}`, 'about:blank'], { stdio: 'ignore' });

let cdp;
const results = [];
const check = (n, ok, x = '') => { results.push({ n, ok, x }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${n}${x ? '  -- ' + x : ''}`); };

try {
  let target = null;
  await waitFor(async () => { try { const r = await fetch(`http://127.0.0.1:${CDP}/json/version`); if (!r.ok) return false; target = await r.json(); return !!target.webSocketDebuggerUrl; } catch { return false; } }, 30000, 'devtools');
  cdp = await Cdp.connect(target.webSocketDebuggerUrl);
  const page = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const attach = await cdp.send('Target.attachToTarget', { targetId: page.targetId, flatten: true });
  const sid = attach.sessionId;
  const send = (m, p = {}) => cdp.send(m, p, sid);
  const evalJS = async (expr) => { const r = await cdp.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true }, sid); if (r.exceptionDetails) throw new Error(r.exceptionDetails.text || 'eval error'); return r.result.value; };
  const perf = () => evalJS(`performance.getEntriesByType('resource').map(r => ({ name: r.name, dur: Math.round(r.duration), size: r.transferSize || 0 }))`);
  // Re-query the DOM fresh each time: node ids go stale after SPA re-renders.
  const setFileInput = async (selector, file) => {
    for (let i = 0; i < 30; i += 1) {
      try {
        const doc = await send('DOM.getDocument', { depth: -1, pierce: false });
        const n = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector });
        if (n && n.nodeId) {
          await send('DOM.setFileInputFiles', { files: [file], nodeId: n.nodeId });
          return;
        }
      } catch { /* stale node id; re-query and retry */ }
      await sleep(300);
    }
    throw new Error('setFileInput failed: ' + selector);
  };
  await send('Runtime.enable');
  await send('Page.enable');
  await send('DOM.enable');
  await send('Network.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url: APP_URL });
  await sleep(1800);
  const bodyText = () => evalJS(`document.body.innerText`);
  await waitFor(async () => !(await bodyText()).includes('正在恢复'), 15000, 'hydration');

  // import an image so the Editor (with AI button) renders
  await setFileInput('input[type=file]', IMG_A);
  await waitFor(async () => (await bodyText()).includes('1 / 1'), 15000, 'editor');

  // Let the background preload (Editor idle boot) finish, as a real user would
  // while browsing images before opening AI. Click→ready should then be ~0.
  await sleep(13000);

  // open AI dialog. No built-in model (by design): engine boots in background,
  // panel shows the "pick your own .onnx" drop zone. Measure from click.
  await evalJS(`[...document.querySelectorAll('button')].find(b => b.textContent.includes('AI 辅助'))?.click()`);
  const t0 = Date.now();
  await waitFor(async () => {
    const t = await bodyText();
    return t.includes('点击或拖入 .onnx 模型')
      || (t.includes('更换') && (t.includes('WebGPU 加速') || t.includes('CPU 运算')))
      || t.includes('出了一点问题');
  }, 120000, 'ai idle/ready/error');
  const engineMs = Date.now() - t0;
  const phase1 = await perf();
  let statusText = await evalJS(`(document.querySelector('[role="dialog"]')?.innerText || '').slice(0, 300)`);
  const isIdle = statusText.includes('点击或拖入 .onnx 模型');
  const hasSavedModel = statusText.includes('更换') && (statusText.includes('WebGPU 加速') || statusText.includes('CPU 运算'));
  check('no built-in model shipped - asks user to pick one', isIdle || hasSavedModel, JSON.stringify(statusText.split('\n').slice(0, 6)));
  check('engine ready (boot) happens in background', engineMs < 2000, `click→idle/ready ${engineMs} ms`);
  check('no CDN resources fetched for AI', !phase1.some((r) => r.name.includes('npmmirror') || r.name.includes('unpkg') || r.name.includes('jsdelivr')), `${phase1.length} resources`);

  console.log('\n--- resources fetched during AI boot (new since app load) ---');
  for (const r of phase1) console.log(`  ${r.name}  ${r.dur}ms  ${r.size}B`);

  // manual model pick (the whole point of no built-in: user's own model)
  if (isIdle) {
    const t1 = Date.now();
    await setFileInput('input[type=file][accept=".onnx"]', MODEL);
    await waitFor(async () => {
      const t = await bodyText();
      return t.includes('模型已就绪')
        || (t.includes('更换') && (t.includes('WebGPU 加速') || t.includes('CPU 运算')))
        || t.includes('出了一点问题');
    }, 120000, 'model ready');
    const pickMs = Date.now() - t1;
    statusText = await evalJS(`(document.querySelector('[role="dialog"]')?.innerText || '').slice(0, 300)`);
    check('manual model pick succeeds', statusText.includes('更换'), `${pickMs} ms`);
    const phase2 = await perf();
    console.log(`\n--- manual pick load: ${pickMs} ms ---`);
    for (const r of phase2) {
      if (!phase1.some((p) => p.name === r.name)) console.log(`  ${r.name}  ${r.dur}ms  ${r.size}B`);
    }
  }

  // sanity: run one real detection on the current image (cat fixture should not error)
  if (statusText.includes('更换')) {
    const t2 = Date.now();
    await evalJS(`[...document.querySelectorAll('[role="dialog"] button')].find(b => b.textContent.includes('标注当前这张图'))?.click()`);
    await waitFor(async () => !(await bodyText()).includes('标注当前这张图'), 10000, 'dialog close');
    await sleep(8000); // software-GPU inference on the fixture
    const detText = await bodyText();
    check('detection runs without error', !detText.includes('识别失败'), `${Date.now() - t2} ms | ${detText.includes('没找到目标') ? '0 boxes' : 'boxes produced'}`);
  }

  console.log('\n--- AI dialog text ---\n' + JSON.stringify(statusText));
} catch (err) {
  check('measure run completed', false, String(err && err.message ? err.message : err));
} finally {
  try { if (cdp) cdp.ws.close(); } catch {}
  try { edge.kill('SIGKILL'); } catch {}
  await sleep(300);
}

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} passed`);
process.exit(failed.length === 0 ? 0 : 1);
