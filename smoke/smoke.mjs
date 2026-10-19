// End-to-end smoke test: drives the built app in headless Edge over CDP.
// ASCII only on purpose (PowerShell mangles UTF-8 when passing args).
import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { rmSync, openSync, closeSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';
import { HERE, ROOT, EDGE } from './env.mjs';

const PORT = 4173;
const CDP = 9333;
const URL_APP = `http://127.0.0.1:${PORT}/`;

const IMG_A = join(HERE, 'fixtures', 'cat-01.png');
const IMG_B = join(HERE, 'fixtures', 'dog-02.png');

// ---- helpers -------------------------------------------------------------
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
  async evalJS(expression) {
    const r = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text || 'eval error');
    return r.result.value;
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

// ---- boot ----------------------------------------------------------------
// tiny static server bound to 127.0.0.1: avoids vite-preview binding to ::1 only
const serverLog = openSync(join(HERE, 'server.log'), 'w');
const preview = spawn(
  process.execPath,
  [join(HERE, 'serve.mjs'), String(PORT), join(ROOT, 'dist')],
  { cwd: ROOT, stdio: ['ignore', serverLog, serverLog] },
);

// fresh profile each run -> deterministic IndexedDB state
const profileDir = join(HERE, '.edge-profile');
rmSync(profileDir, { recursive: true, force: true });
const edge = spawn(
  EDGE,
  [
    '--headless=new',
    '--disable-gpu',
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
  await waitFor(
    async () => {
      for (const base of [`http://127.0.0.1:${PORT}/`, `http://localhost:${PORT}/`]) {
        try {
          const ctl = new AbortController();
          const timer = setTimeout(() => ctl.abort(), 1200);
          const r = await fetch(base, { signal: ctl.signal });
          clearTimeout(timer);
          if (r.ok) return true;
        } catch {
          /* try next */
        }
      }
      return false;
    },
    30000,
    'preview server',
  );

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
  const sessionId = page.targetId;
  const attach = await cdp.send('Target.attachToTarget', { targetId: sessionId, flatten: true });
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
  await send('Page.navigate', { url: URL_APP });
  await sleep(1500);

  // ---- text helpers ------------------------------------------------------
  const bodyText = () => evalJS(`document.body.innerText`);
  const hasText = async (needle) => (await bodyText()).includes(needle);

  // Annotation rows of the CURRENT image (right panel), one per box.
  //
  // 以前是「title 里含 标签/类别 且文本非空」来猜，但右侧还有「套用当前类别」这类按钮
  // （title 含 类别、文本非空），于是 1 个框会被数成 2，断言 `=== 1` 永远不成立，
  // 表现为「timeout waiting for first box created」—— 实际应用完全正常。
  // 现在用 RightPanel 给出的 data-ann-row 精确计数。
  const chips = () =>
    evalJS(`
      (() => Array.from(document.querySelectorAll('[data-ann-row]'))
        .map((row) => {
          const chip = row.querySelector('button[title]');
          return chip ? (chip.textContent || '').trim() : '';
        }))()
    `);
  const countBoxes = async () => (await chips()).length;

  // ---- 1. landing page ---------------------------------------------------
  // hydration reads IndexedDB first; the splash has no nav text, so wait for it to go away
  await waitFor(
    async () => !(await bodyText()).includes('\u6b63\u5728\u6062\u590d'),
    15000,
    'hydration',
  );
  const landingOk = await hasText('YOLO');
  check('landing page renders', landingOk, landingOk ? '' : await bodyText().then((t) => JSON.stringify(t.slice(0, 200))));

  // ---- 2. import two images ---------------------------------------------
  const inputInfo = await evalJS(`
    (() => {
      const inputs = Array.from(document.querySelectorAll('input[type=file]'));
      return inputs.length;
    })()
  `);
  check('file input present on landing', inputInfo >= 1, `count=${inputInfo}`);

  const doc = await send('DOM.getDocument', { depth: -1, pierce: false });
  const found = await send('DOM.querySelector', {
    nodeId: doc.root.nodeId,
    selector: 'input[type=file]',
  });
  if (!found || !found.nodeId) throw new Error('file input not found in DOM');
  await send('DOM.setFileInputFiles', { files: [IMG_A, IMG_B], nodeId: found.nodeId });

  await waitFor(async () => (await bodyText()).includes('1 / 2'), 15000, 'editor with 2 images');
  check('two images imported', true);

  // Thumbnails now go through "read size from file header -> downsample while decoding"
  // (fixtures are 400x300, long edge > 160). Assert the image actually shows up and was
  // reduced to 160x120. This must be checked explicitly: loadThumbnail swallows errors
  // into an empty string, so a broken path only shows as a permanently grey tile with
  // zero console errors.
  const thumb = await waitFor(
    async () =>
      await evalJS(`
        (() => {
          const im = document.querySelector('[data-image-card] img');
          if (!im || !im.src || !im.src.startsWith('data:image')) return null;
          return { w: im.naturalWidth, h: im.naturalHeight };
        })()
      `),
    10000,
    'thumbnail rendered',
  );
  check(
    'thumbnail rendered at reduced size',
    thumb.w === 160 && thumb.h === 120,
    `${thumb.w}x${thumb.h}`,
  );

  const waitCanvas = async () => {
    const box = await evalJS(`
      (() => {
        const c = document.querySelector('canvas');
        if (!c) return null;
        const r = c.getBoundingClientRect();
        return { x: r.x, y: r.y, w: r.width, h: r.height };
      })()
    `);
    return box && box.w > 0 ? box : null;
  };
  const canvas = await waitFor(waitCanvas, 15000, 'canvas mounted');
  check('canvas mounted', !!canvas, `${Math.round(canvas.w)}x${Math.round(canvas.h)}`);

  // ---- drag helper (mouse -> pointer events) -----------------------------
  const drag = async (fromX, fromY, toX, toY, steps = 12) => {
    await send('Input.dispatchMouseEvent', {
      type: 'mousePressed',
      x: fromX,
      y: fromY,
      button: 'left',
      buttons: 1,
      clickCount: 1,
    });
    for (let i = 1; i <= steps; i += 1) {
      const t = i / steps;
      await send('Input.dispatchMouseEvent', {
        type: 'mouseMoved',
        x: fromX + (toX - fromX) * t,
        y: fromY + (toY - fromY) * t,
        button: 'left',
        buttons: 1,
      });
    }
    await send('Input.dispatchMouseEvent', {
      type: 'mouseReleased',
      x: toX,
      y: toY,
      button: 'left',
      buttons: 0,
      clickCount: 1,
    });
    await sleep(200);
  };

  // ---- 3. draw first box -> popover with inline input --------------------
  const cx = canvas.x + canvas.w / 2;
  const cy = canvas.y + canvas.h / 2;
  await drag(cx - 140, cy - 90, cx - 20, cy - 10);

  await waitFor(async () => (await countBoxes()) === 1, 8000, 'first box created');
  check('draw box creates annotation', true);

  const popoverShown = await evalJS(`
    (() => {
      const els = Array.from(document.querySelectorAll('input'));
      return els.some((e) => e.placeholder && e.placeholder.length > 0);
    })()
  `);
  check('label popover opens inline (0 labels)', popoverShown === true);

  // the inline input must already have focus: the whole point is "draw, then type"
  const focused = await evalJS(`
    (() => {
      const a = document.activeElement;
      return a ? a.tagName : 'none';
    })()
  `);
  check('popover input is auto-focused', focused === 'INPUT', focused);

  // ---- 4. name the first label via keyboard ------------------------------
  await send('Input.insertText', { text: 'cat' });
  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 });
  await sleep(400);

  // must check the chip text, not body text (image file names contain "cat")
  const chipsAfterLabel = await chips();
  check(
    'new label created and assigned to box',
    chipsAfterLabel.length === 1 && chipsAfterLabel[0] === 'cat',
    JSON.stringify(chipsAfterLabel),
  );

  // ---- 5. draw second box -> pick existing label from popover ------------
  await drag(cx + 30, cy - 60, cx + 150, cy + 40);
  await waitFor(async () => (await countBoxes()) === 2, 8000, 'second box created');
  check('second box created', true);

  // popover should be open; press "1" to pick the first label
  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: '1', code: 'Digit1', text: '1', windowsVirtualKeyCode: 49 });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: '1', code: 'Digit1', windowsVirtualKeyCode: 49 });
  await sleep(400);

  const chipsAfterPick = await chips();
  check(
    'number key assigns label from popover',
    chipsAfterPick.length === 2 && chipsAfterPick.every((c) => c === 'cat'),
    JSON.stringify(chipsAfterPick),
  );

  // the popover must be gone, otherwise it would swallow every later shortcut
  const blockers = await evalJS(`
    (() => {
      const roles = document.querySelectorAll('[role=dialog],[role=alertdialog],[role=menu]').length;
      const open = document.body.innerText.includes('\\u8fd9\\u662f\\u4ec0\\u4e48');
      return { roles, open };
    })()
  `);
  check('popover closes after picking a label', blockers.open === false, JSON.stringify(blockers));

  // ---- 6. undo / redo ----------------------------------------------------
  const before = await countBoxes();
  await send('Input.dispatchKeyEvent', {
    type: 'keyDown',
    key: 'z',
    code: 'KeyZ',
    ctrl: true,
    modifiers: 2,
    windowsVirtualKeyCode: 90,
  });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'z', code: 'KeyZ', windowsVirtualKeyCode: 90 });
  await sleep(400);
  const afterUndo = await countBoxes();
  check('ctrl+z undoes last change', afterUndo !== before, `${before} -> ${afterUndo}`);

  await send('Input.dispatchKeyEvent', {
    type: 'keyDown',
    key: 'z',
    code: 'KeyZ',
    ctrl: true,
    shift: true,
    modifiers: 10,
    windowsVirtualKeyCode: 90,
  });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'z', code: 'KeyZ', windowsVirtualKeyCode: 90 });
  await sleep(400);
  const afterRedo = await countBoxes();
  check('ctrl+shift+z redoes', afterRedo === before, `${afterUndo} -> ${afterRedo}`);

  // ---- 7. switch image, verify isolation ---------------------------------
  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39 });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39 });
  await sleep(500);
  const secondImageBoxes = await countBoxes();
  const navText = await evalJS(`document.body.innerText.includes('2 / 2')`);
  check('arrow key switches image', navText === true);
  check('annotations are per-image', secondImageBoxes === 0, `boxes=${secondImageBoxes}`);

  // ---- 8. back and drag outside image should NOT create a box ------------
  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'ArrowLeft', code: 'ArrowLeft', windowsVirtualKeyCode: 37 });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'ArrowLeft', code: 'ArrowLeft', windowsVirtualKeyCode: 37 });
  await sleep(600);
  const beforeOutside = await countBoxes();

  // drag in the far top-left corner of the canvas (well outside the fitted image)
  await drag(canvas.x + 6, canvas.y + 6, canvas.x + 40, canvas.y + 40);
  await sleep(300);
  const afterOutside = await countBoxes();
  check('drag outside image does not create a box', afterOutside === beforeOutside, `${beforeOutside} -> ${afterOutside}`);

  // ---- 9. export dialog --------------------------------------------------
  await send('Input.dispatchKeyEvent', {
    type: 'keyDown',
    key: 's',
    code: 'KeyS',
    ctrl: true,
    modifiers: 2,
    windowsVirtualKeyCode: 83,
  });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 's', code: 'KeyS', windowsVirtualKeyCode: 83 });
  await sleep(700);
  const exportOpen = await evalJS(`
    (() => {
      const t = document.body.innerText;
      return t.includes('YOLO') && t.includes('ZIP');
    })()
  `);
  check('ctrl+s opens export dialog', exportOpen === true);

  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  await sleep(300);

  // ---- 9b. 「照样子找」面板（YOLOE 视觉提示）-------------------------------
  // 与上面的「AI 辅助」并列的第二条 AI 链路，但**不需要用户准备任何 .onnx**
  // （模型随站点内置，见 public/models/yoloe26n-vp.onnx）。
  // 这里只验证 UI 挂到了对的地方 —— 推理本身已经在 .tmp/yoloe-web/verify_vp.mjs
  // 里与 Python 基准逐值对拍过（512 维提示向量最大差 1.2e-7）。
  // 刻意**不点「先试标一张」**：那会真的下载 20MB 模型并建会话，
  // 无头环境（--disable-gpu，只能走 WASM）要多花十几秒，而它不验证任何 UI 结构。
  const aiPanelOpen = await evalJS(`
    (() => {
      const b = Array.from(document.querySelectorAll('button'))
        .find((x) => (x.textContent || '').includes('AI \\u8f85\\u52a9'));
      if (!b) return false;
      b.click();
      return true;
    })()
  `);
  check('AI panel opens', aiPanelOpen === true);
  await sleep(900);

  const vpEntry = await evalJS(`!!document.querySelector('[data-vp-entry]')`);
  check('「照样子找」entry present in AI panel', vpEntry === true);

  const vpClicked = await evalJS(`
    (() => {
      const b = document.querySelector('[data-vp-entry]');
      if (!b) return false;
      b.click();
      return true;
    })()
  `);
  check('「照样子找」entry is clickable', vpClicked === true);
  await sleep(1200);

  const vp = await evalJS(`
    (() => {
      const trial = document.querySelector('[data-vp-trial]');
      const batch = document.querySelector('[data-vp-batch]');
      return {
        panel: !!document.querySelector('[data-vp-panel]'),
        aiClosed: !document.querySelector('[data-ai-panel]'),
        refs: document.querySelectorAll('[data-vp-ref]').length,
        refOn: document.querySelectorAll('[data-vp-ref-on="1"]').length,
        labels: document.querySelectorAll('[data-vp-label]').length,
        scopes: document.querySelectorAll('[data-vp-scope]').length,
        trial: !!trial,
        batch: !!batch,
        // 存在 ≠ 可用。模型是「点了按钮才下载」的，所以首次打开时
        // 这两个按钮必须是**可点**的 —— 拿 status==='ready' 当门槛会死锁
        // （按钮灰 → 点不了 → 模型不下 → ready 永远为 false）。
        // 这里只读 disabled，**不点**：点了会真去下 20MB 模型。
        trialEnabled: !!trial && trial.disabled === false,
        batchEnabled: !!batch && batch.disabled === false,
        // 批量不再边跑边改标注，而是先攒候选框 + 汇总。
        // 这句话必须出现在面板上：看不到它，用户不会知道自己还有一次反悔的机会。
        batchCopy: (document.querySelector('[data-vp-panel]')?.innerText || '').includes(
          '\\u4e0d\\u4f1a\\u76f4\\u63a5\\u6539\\u4f60\\u7684\\u6807\\u6ce8',
        ),
      };
    })()
  `);
  check('「照样子找」panel opens (AI panel closes)', vp.panel === true && vp.aiClosed === true, JSON.stringify(vp));
  check('reference picker lists labeled images', vp.refs >= 1, JSON.stringify(vp));
  check('current image auto-picked as reference', vp.refOn >= 1, JSON.stringify(vp));
  check('label picker rendered', vp.labels >= 1, JSON.stringify(vp));
  check('trial + batch actions rendered', vp.trial === true && vp.batch === true, JSON.stringify(vp));
  check(
    'trial + batch are enabled before model download (no ready-gate deadlock)',
    vp.trialEnabled === true && vp.batchEnabled === true,
    JSON.stringify(vp),
  );
  check('batch scope toggle rendered (仅未标注 / 全部重跑)', vp.scopes === 2, JSON.stringify(vp));
  check('batch copy says it will not write annotations directly', vp.batchCopy === true, JSON.stringify(vp));

  // ---- 9c. 引擎切换（YOLOE 20MB / DINOv2 85MB）-----------------------------
  // 两条链路的原理不同、擅长的场景不同，界面上并列摆出来让用户自己比。
  // 这里验证三件事：两个选项都在、默认是 YOLOE、**切换本身不会触发下载**
  // （模型是点了动作按钮才下的，切个引擎就偷偷下 85MB 是不可接受的）。
  //
  // 注意：DINOv2 的**分辨率守卫**不在这里测。夹具是 400x300 的色块图，
  // 守卫线是 42/gain ≈ 32px，而本用例画的框远大于它 —— 硬造一个小框就得
  // 依赖画布缩放比，会变成 flaky。守卫的算术由 test/parse.test.mjs 钉住
  // （10 条断言，含边界 110/111px），端到端行为在真实截图上另测。
  const eng = await evalJS(`
    (() => {
      const btns = Array.from(document.querySelectorAll('[data-vp-engine]'));
      return {
        count: btns.length,
        ids: btns.map((b) => b.getAttribute('data-vp-engine')),
        onIds: btns.filter((b) => b.getAttribute('data-vp-engine-on') === '1')
          .map((b) => b.getAttribute('data-vp-engine')),
        enabled: btns.every((b) => b.disabled === false),
      };
    })()
  `);
  check('both engines are offered', eng.count === 2 && eng.ids.join(',') === 'yoloe,dino', JSON.stringify(eng));
  check('YOLOE is the default engine', eng.onIds.length === 1 && eng.onIds[0] === 'yoloe', JSON.stringify(eng));

  const switched = await evalJS(`
    (() => {
      const d = document.querySelector('[data-vp-engine="dino"]');
      if (!d || d.disabled) return false;
      d.click();
      return true;
    })()
  `);
  await sleep(600);
  const afterSwitch = await evalJS(`
    (() => {
      const btns = Array.from(document.querySelectorAll('[data-vp-engine]'));
      const panel = document.querySelector('[data-vp-panel]');
      const text = panel ? panel.innerText : '';
      const trial = document.querySelector('[data-vp-trial]');
      return {
        onIds: btns.filter((b) => b.getAttribute('data-vp-engine-on') === '1')
          .map((b) => b.getAttribute('data-vp-engine')),
        hasStrictness: text.includes('匹配严格度'),
        hasConfidence: text.includes('置信度阈值'),
        trialEnabled: !!trial && trial.disabled === false,
        downloading: text.includes('正在准备内置模型'),
      };
    })()
  `);
  check('switching to DINOv2 flips the engine state', switched === true && afterSwitch.onIds[0] === 'dino', JSON.stringify(afterSwitch));
  check(
    'threshold card switches semantics (strictness, not confidence)',
    afterSwitch.hasStrictness === true && afterSwitch.hasConfidence === false,
    JSON.stringify(afterSwitch),
  );
  check(
    'switching engine does NOT start a download',
    afterSwitch.downloading === false && afterSwitch.trialEnabled === true,
    JSON.stringify(afterSwitch),
  );

  // 切回默认，免得影响后面的用例
  await evalJS(`
    (() => {
      const y = document.querySelector('[data-vp-engine="yoloe"]');
      if (y) y.click();
      return !!y;
    })()
  `);
  await sleep(400);

  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  await sleep(400);

  // ---- 10. delete a label: boxes fall back to unclassified, undo restores -
  // These icon buttons are display:none until their row is hovered, so hover the
  // parent row first, then re-measure the button before clicking it.
  const clickByTitle = async (title) => {
    const row = await evalJS(`
      (() => {
        const b = document.querySelector('button[title="${title}"]');
        if (!b) return null;
        const host = b.closest('.group') || b.parentElement;
        const r = host.getBoundingClientRect();
        return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
      })()
    `);
    if (!row) return false;
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: row.x, y: row.y });
    await sleep(220);
    const shown = await evalJS(`
      (() => {
        const b = document.querySelector('button[title="${title}"]');
        if (!b) return null;
        const r = b.getBoundingClientRect();
        return r.width > 0 ? { x: r.x + r.width / 2, y: r.y + r.height / 2 } : null;
      })()
    `);
    if (!shown) return false;
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: shown.x, y: shown.y, button: 'left', buttons: 1, clickCount: 1 });
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: shown.x, y: shown.y, button: 'left', buttons: 0, clickCount: 1 });
    await sleep(500);
    return true;
  };

  const removedLabel = await clickByTitle('\\u5220\\u9664\\u6807\\u7b7e');
  check('label delete button reachable', removedLabel === true);
  const chipsAfterLabelDelete = await chips();
  check(
    'deleting a label only un-classifies its boxes',
    chipsAfterLabelDelete.length === 2 && chipsAfterLabelDelete.every((c) => c !== 'cat'),
    JSON.stringify(chipsAfterLabelDelete),
  );

  await send('Input.dispatchKeyEvent', {
    type: 'keyDown',
    key: 'z',
    code: 'KeyZ',
    ctrl: true,
    modifiers: 2,
    windowsVirtualKeyCode: 90,
  });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'z', code: 'KeyZ', windowsVirtualKeyCode: 90 });
  await sleep(600);
  const chipsAfterLabelUndo = await chips();
  check(
    'ctrl+z restores the deleted label',
    chipsAfterLabelUndo.length === 2 && chipsAfterLabelUndo.every((c) => c === 'cat'),
    JSON.stringify(chipsAfterLabelUndo),
  );

  // ---- 11. re-importing the same files must not duplicate them -----------
  const sideDoc = await send('DOM.getDocument', { depth: -1, pierce: false });
  const sideInput = await send('DOM.querySelector', {
    nodeId: sideDoc.root.nodeId,
    selector: 'aside input[type=file]',
  });
  if (sideInput && sideInput.nodeId) {
    await send('DOM.setFileInputFiles', { files: [IMG_A, IMG_B], nodeId: sideInput.nodeId });
    await sleep(900);
  }
  const afterReimport = await evalJS(`document.body.innerText.includes('1 / 2')`);
  check('duplicate import is skipped', afterReimport === true);

  // ---- 11. remove image -> confirm -> undo restores it -------------------
  const cardBox = await evalJS(`
    (() => {
      const c = document.querySelector('[data-image-card][data-index="0"]');
      if (!c) return null;
      const r = c.getBoundingClientRect();
      return { x: r.x, y: r.y, w: r.width, h: r.height };
    })()
  `);
  await send('Input.dispatchMouseEvent', {
    type: 'mouseMoved',
    x: cardBox.x + cardBox.w / 2,
    y: cardBox.y + cardBox.h / 2,
  });
  await sleep(250);
  // the trash button only appears on hover
  const trashPoint = await evalJS(`
    (() => {
      const c = document.querySelector('[data-image-card][data-index="0"]');
      const b = c && c.querySelector('button[title="\\u79fb\\u9664\\u8fd9\\u5f20\\u56fe\\u7247"]');
      if (!b) return null;
      const r = b.getBoundingClientRect();
      return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    })()
  `);
  if (trashPoint) {
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: trashPoint.x, y: trashPoint.y, button: 'left', buttons: 1, clickCount: 1 });
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: trashPoint.x, y: trashPoint.y, button: 'left', buttons: 0, clickCount: 1 });
    await sleep(400);
  }
  const confirmPoint = await evalJS(`
    (() => {
      const b = Array.from(document.querySelectorAll('button')).find((x) => x.textContent.trim() === '\\u786e\\u8ba4\\u79fb\\u9664');
      if (!b) return null;
      const r = b.getBoundingClientRect();
      return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    })()
  `);
  check('remove image asks for confirmation', confirmPoint !== null);
  if (confirmPoint) {
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: confirmPoint.x, y: confirmPoint.y, button: 'left', buttons: 1, clickCount: 1 });
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: confirmPoint.x, y: confirmPoint.y, button: 'left', buttons: 0, clickCount: 1 });
    await sleep(600);
  }
  const afterRemove = await evalJS(`document.body.innerText.includes('1 / 1')`);
  check('image removed after confirm', afterRemove === true);

  await send('Input.dispatchKeyEvent', {
    type: 'keyDown',
    key: 'z',
    code: 'KeyZ',
    ctrl: true,
    modifiers: 2,
    windowsVirtualKeyCode: 90,
  });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'z', code: 'KeyZ', windowsVirtualKeyCode: 90 });
  await sleep(700);
  const afterUndoRemove = await evalJS(`document.body.innerText.includes('1 / 2')`);
  const boxesBack = await countBoxes();
  check('ctrl+z restores removed image', afterUndoRemove === true, `boxes=${boxesBack}`);
  check('annotations survive image restore', boxesBack === 2, `boxes=${boxesBack}`);

  // ---- 12. reload -> persistence ----------------------------------------
  await send('Page.navigate', { url: URL_APP });
  await sleep(2500);
  const restored = await evalJS(`
    (() => {
      const t = document.body.innerText;
      return { hasNav: t.includes('1 / 2'), hasCat: t.includes('cat') };
    })()
  `);
  check('project restored after reload', restored.hasNav === true, JSON.stringify(restored));
  check('labels restored after reload', restored.hasCat === true);

  // ---- console errors ----------------------------------------------------
  const real = consoleErrors.filter((e) => !/favicon|Failed to load resource.*404/i.test(e));
  check('no console errors', real.length === 0, real.slice(0, 3).join(' | '));
} catch (err) {
  check('smoke run completed', false, String(err && err.message ? err.message : err));
} finally {
  try {
    if (cdp) cdp.ws.close();
  } catch {}
  try {
    edge.kill('SIGKILL');
  } catch {}
  try {
    preview.kill('SIGKILL');
  } catch {}
  try {
    closeSync(serverLog);
  } catch {}
  await sleep(300);
}

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} passed`);
process.exit(failed.length === 0 ? 0 : 1);
