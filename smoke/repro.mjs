// Focused repro: does the imported image actually render on the canvas?
// Drives headless Edge over CDP against the dev server (default http://127.0.0.1:5175).
import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { rmSync, openSync, closeSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const APP_URL = process.env.APP_URL || 'http://localhost:5175/';
const CDP = 9344;

const IMG_A = join(HERE, 'fixtures', 'cat-01.png');

const results = [];
function check(name, ok, extra = '') {
  results.push({ name, ok, extra });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? `  -- ${extra}` : ''}`);
}

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.listeners = [];
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
        return;
      }
      for (const fn of this.listeners) fn(msg);
    });
  }
  static async connect(url) {
    const ws = new WebSocket(url);
    await new Promise((res, rej) => {
      ws.addEventListener('open', res);
      ws.addEventListener('error', rej);
    });
    return new Cdp(ws);
  }
  send(method, params = {}, sessionId) {
    this.id += 1;
    const id = this.id;
    const payload = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;
    this.ws.send(JSON.stringify(payload));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`timeout: ${method}`));
        }
      }, 20000);
    });
  }
  on(fn) {
    this.listeners.push(fn);
  }
}

async function waitFor(fn, timeout = 15000, label = 'condition') {
  const started = Date.now();
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() - started > timeout) throw new Error(`timeout waiting for ${label}`);
    await sleep(120);
  }
}

const profileDir = join(tmpdir(), 'yolo-repro-' + Date.now());
rmSync(profileDir, { recursive: true, force: true });
const edge = spawn(
  EDGE,
  [
    '--headless=new',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--hide-scrollbars',
    '--window-size=1440,900',
    `--remote-debugging-port=${CDP}`,
    `--user-data-dir=${profileDir}`,
    'about:blank',
  ],
  { stdio: 'ignore' },
);

let cdp;
try {
  let target = null;
  await waitFor(
    async () => {
      try {
        const r = await fetch(`http://127.0.0.1:${CDP}/json/version`);
        if (!r.ok) return false;
        target = await r.json();
        return !!target.webSocketDebuggerUrl;
      } catch {
        return false;
      }
    },
    20000,
    'edge devtools',
  );

  cdp = await Cdp.connect(target.webSocketDebuggerUrl);
  const page = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const attach = await cdp.send('Target.attachToTarget', { targetId: page.targetId, flatten: true });
  const sid = attach.sessionId;
  const send = (method, params = {}) => cdp.send(method, params, sid);
  const evalJS = async (expr) => {
    const r = await cdp.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true }, sid);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text || 'eval error');
    return r.result.value;
  };

  await send('Runtime.enable');
  await send('Page.enable');
  await send('DOM.enable');
  await send('Log.enable');

  const consoleErrors = [];
  cdp.on((msg) => {
    if (msg.params?.entry?.level === 'error' && msg.method === 'Log.entryAdded') {
      consoleErrors.push(msg.params.entry.text);
    }
    if (msg.method === 'Runtime.exceptionThrown') {
      const d = msg.params.exceptionDetails;
      consoleErrors.push(d.exception?.description || d.text || 'exception');
    }
  });

  await send('Emulation.setDeviceMetricsOverride', {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await send('Page.navigate', { url: APP_URL });
  await sleep(1800);

  // wait for hydration splash to disappear
  const bodyText = () => evalJS(`document.body.innerText`);
  await waitFor(async () => !(await bodyText()).includes('\u6b63\u5728\u6062\u590d'), 15000, 'hydration');

  // import image via file input (retry until input exists in DOM)
  let found = null;
  try {
    found = await waitFor(async () => {
      const doc = await send('DOM.getDocument', { depth: -1, pierce: false });
      const f = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: 'input[type=file]' });
      return f && f.nodeId ? { root: doc.root.nodeId, nodeId: f.nodeId } : null;
    }, 15000, 'file input in DOM');
  } catch (e) {
    console.log('DIAG body:', JSON.stringify(await bodyText().catch(() => 'n/a')).slice(0, 300));
    console.log('DIAG consoleErrors:', JSON.stringify(consoleErrors).slice(0, 500));
    throw e;
  }
  await send('DOM.setFileInputFiles', { files: [IMG_A], nodeId: found.nodeId });

  await waitFor(async () => (await bodyText()).includes('1 / 1'), 15000, 'editor with 1 image');
  await sleep(1500);

  const sample = (px, py) => evalJS(`
    (() => {
      const c = document.querySelector('canvas');
      const r = c.getBoundingClientRect();
      const ctx = c.getContext('2d');
      const d = ctx.getImageData(Math.floor(${px}), Math.floor(${py}), 1, 1).data;
      return [d[0], d[1], d[2], d[3]];
    })()
  `);

  const headerInfo = await evalJS(`
    (() => {
      const glass = document.querySelector('.glass');
      return glass ? glass.innerText : '';
    })()
  `);

  // simulate pressing F (fit) to force a refit
  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'f', code: 'KeyF', windowsVirtualKeyCode: 70 });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'f', code: 'KeyF', windowsVirtualKeyCode: 70 });
  await sleep(500);

  const centerAfterFit = await sample(479, 393);
  const topLeftAfterFit = await sample(50, 50);

  // draw a box across the middle of the canvas
  const drag = async (fx, fy, tx, ty) => {
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: fx, y: fy, button: 'left', buttons: 1, clickCount: 1 });
    for (let i = 1; i <= 10; i += 1) {
      const t = i / 10;
      await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: fx + (tx - fx) * t, y: fy + (ty - fy) * t, button: 'left', buttons: 1 });
    }
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: tx, y: ty, button: 'left', buttons: 0, clickCount: 1 });
    await sleep(400);
  };
  await drag(300, 300, 700, 500);
  const boxCount = await evalJS(`document.body.innerText.includes('1 框')`);
  const insideBox = await sample(500, 400);

  const info = {
    headerInfo,
    centerAfterFit,
    topLeftAfterFit,
    boxCount,
    insideBox,
  };
  check('image dimensions shown in header', headerInfo.includes('400'), JSON.stringify(headerInfo));
  check('canvas center has image pixels (not black)', centerAfterFit[0] + centerAfterFit[1] + centerAfterFit[2] > 60, JSON.stringify({ centerAfterFit, topLeftAfterFit }));
  check('box drawn (1 框)', boxCount === true);
  check('box pixels visible on canvas', insideBox[0] + insideBox[1] + insideBox[2] > 40, JSON.stringify(insideBox));


  // screenshot for visual confirmation
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  const { writeFileSync } = await import('node:fs');
  writeFileSync(join(HERE, 'repro-shot.png'), Buffer.from(shot.data, 'base64'));
  check('screenshot saved', true, 'smoke/repro-shot.png');

  check('no console errors', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));
} catch (err) {
  check('repro run completed', false, String(err && err.message ? err.message : err));
} finally {
  try {
    if (cdp) cdp.ws.close();
  } catch {}
  try {
    edge.kill('SIGKILL');
  } catch {}
  await sleep(300);
}

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} passed`);
process.exit(failed.length === 0 ? 0 : 1);
