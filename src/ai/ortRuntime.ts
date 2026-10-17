/**
 * ONNX Runtime Web 的引导层（从 detector.ts 抽出来的共享部分）。
 *
 * 为什么单独成一个模块：现在有两个消费者 ——
 *   - `detector.ts`（YOLO 目标检测，一个会话）
 *   - `segmenter.ts`（MobileSAM 交互式分割，encoder/decoder 两个会话）
 * 两边都要「注入加载器 + 探测伴生 wasm 可达性 + 选 EP + 处理粘性 initWasm 失败」。
 * 复制两份必然会漂移，所以抽到这里。
 *
 * 这一层踩过的坑（都是真实的，别改回去）：
 *  1. wasmPaths **必须在第一次创建会话前设置**，且 initWasm 失败是「粘性」的：
 *     本地路径指向坏目录后，整个运行时都废了，只能重新注入 CDN 覆盖 window.ort。
 *  2. 多线程 wasm 依赖 SharedArrayBuffer，只在页面带 COOP/COEP 头时可用。
 *     普通静态托管没有 → 必须回退单线程，否则 `initWasm()` 直接失败。
 */

export interface OrtTensorOut {
  dims: Array<number | string>;
  /** int64 输出在 ORT Web 里是 BigInt64Array，其余是 Float32Array */
  data: Float32Array | BigInt64Array;
}

export interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorOut>>;
  inputNames: string[];
  outputNames: string[];
  inputMetadata: unknown;
}

export interface OrtGlobal {
  env: {
    wasm: {
      /**
       * ORT 支持两种形态（见 configure() 的说明，别改回字符串）：
       *   · string —— 被当成**目录前缀**去拼文件名
       *   · object —— 逐文件给完整 URL，不做任何拼接
       */
      wasmPaths: string | { mjs?: string; wasm?: string };
      proxy?: boolean;
      numThreads?: number;
    };
    webgpu?: Record<string, unknown>;
  };
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown;
  InferenceSession: {
    create(bytes: Uint8Array, opts: Record<string, unknown>): Promise<OrtSession>;
  };
  version?: string;
}

/* 加载器 + 伴生二进制必须同版本。国内优先 npmmirror；jsdelivr 在境内常不可达 */
export const ORT_VERSION = '1.30.0-dev.20260826-b1f76d586a';
export const ORT_BASES = [
  `https://registry.npmmirror.com/onnxruntime-web/${ORT_VERSION}/files/dist/`,
  `https://unpkg.com/onnxruntime-web@${ORT_VERSION}/dist/`,
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
];

/**
 * 判定 ONNX Runtime Web 的 wasm 可用线程数。
 * 线程版 wasm（ort-wasm-simd-threaded）依赖 SharedArrayBuffer，
 * 而 SharedArrayBuffer 只在 `crossOriginIsolated === true`（页面带 COOP/COEP 头）时可用。
 * 大多数静态托管（Netlify Drop / GitHub Pages / 对象存储…）没有这些头，
 * 此时必须回退单线程，否则 initWasm() 直接失败、AI 功能整体不可用。
 */
export function detectWasmThreads(): number {
  const isolated = (globalThis as unknown as { crossOriginIsolated?: boolean })
    .crossOriginIsolated;
  if (isolated !== true) return 1;
  return Math.min(4, navigator.hardwareConcurrency || 4);
}

export function injectScript(src: string): Promise<boolean> {
  return new Promise((resolve) => {
    const sc = document.createElement('script');
    sc.src = src;
    sc.onload = () => resolve(true);
    sc.onerror = () => resolve(false);
    document.head.appendChild(sc);
  });
}

export function getOrt(): OrtGlobal | null {
  return (window as unknown as { ort?: OrtGlobal }).ort ?? null;
}

/**
 * 随包分发的 ORT wasm 文件名（`public/ort/`）。
 * `ort.webgpu.min.js` 里**只引用这一个变体**（`jsep` 只是代码里的钩子函数名，不是文件名），
 * 所以本地只要这两个文件就是齐的 —— 别再去找 `*.jsep.wasm`。
 */
const WASM_GLUE = 'ort-wasm-simd-threaded.asyncify.mjs';
const WASM_BINARY = 'ort-wasm-simd-threaded.asyncify.wasm';

/**
 * 把 `base + 文件名` 解析成**绝对 URL**。这一步是必需的，且踩过一个很难看的坑：
 *
 * 本项目的 vite `base` 是 `'./'`（为了能丢进任意子目录），于是 `import.meta.env.BASE_URL`
 * 就是 `'./'`，`localBase` 是 `'./ort/'` —— 直接拼出来的是 `'./ort/ort-wasm-….mjs'`
 * 这种**相对 URL**。
 *
 * 而 ORT 是用 `import()` 加载这个 glue 的，**相对 URL 会相对于 ORT 加载器脚本自身**
 * （`/ort/ort.webgpu.min.js`）去解析，于是变成 `/ort/` + `./ort/ort-wasm-….mjs`
 * = `/ort/ort/ort-wasm-….mjs`（凭空多出一层 `ort/`）→ 404。
 *
 * 代价远不止一个文件没加载：glue 一失败，ORT 会**把 25MB 的 wasm 也一起改从 CDN 拉**
 * （压缩后也有 7.6MB），在国内网络下就是「加载慢得离谱」，而现场只留一条不起眼的 404。
 * 绝对 URL 没有任何解析歧义，从此与 `base` 取值无关。
 */
function absoluteWasmUrl(path: string): string {
  try {
    return new URL(path, document.baseURI).href;
  } catch {
    return path;
  }
}

/** 把已经注入的 ort 全局对象配置成可用状态 */
function configure(ort: OrtGlobal, wasmBase: string): void {
  /**
   * `wasmPaths` 用**对象形式**逐文件给 URL，并且每个 URL 都要绝对化
   * （绝对化的原因见 absoluteWasmUrl 的详细说明）。
   *
   * 为什么不用字符串形式：ORT 对字符串只做**纯字符串拼接**（其内部 `tn()` 就是
   * `${prefix}${filename}`），拼出来的仍是相对 URL —— 于是又回到「相对路径被 ORT
   * 自己再解析一次」那个坑里。对象形式则被 ORT 直接当作最终 URL 使用，配合绝对化就没有歧义了。
   */
  ort.env.wasm.wasmPaths = {
    mjs: absoluteWasmUrl(`${wasmBase}${WASM_GLUE}`),
    wasm: absoluteWasmUrl(`${wasmBase}${WASM_BINARY}`),
  };
  ort.env.wasm.proxy = false;
  ort.env.wasm.numThreads = detectWasmThreads();
  if (ort.env.webgpu) {
    ort.env.webgpu.powerPreference = 'high-performance';
    try {
      ort.env.webgpu.useWebgpuMemoryPool = true;
    } catch {
      /* 旧版本没有该选项 */
    }
  }
}

/**
 * 从指定 base 注入加载器并配置伴生二进制路径。
 *
 * ⚠️ 这里**刻意不再对 wasm 发 HEAD 探测可达性**（原来有一段，已删）。两个理由：
 *
 * 1. **它会在 vite dev server 上永久挂住。** 探测目标是 25MB 的
 *    `ort-wasm-simd-threaded.asyncify.wasm`，而 dev server 对它的 HEAD 请求不返回
 *    （实测超过 60 秒没有响应）。`boot()` 因此永远不 resolve，
 *    表现为「点开『点击标注』后一直没反应，连模型都不开始下载」——
 *    这个 bug 只在 `npm run dev`（5173）上出现，在构建产物（preview / serve.mjs）上正常，
 *    所以极易被误判成「功能坏了」。
 *
 * 2. **本来就不需要它。** 它唯一的作用是「本地 wasm 不可用时把路径改指 CDN」，
 *    而 `createSession()` 已经有一条更可靠的兜底：本地 wasm 建会话失败时会调用
 *    `recoverWithCdn()` 重新注入 CDN 运行时再试一次。用「失败后再兜底」代替
 *    「事先探测」，既少一次请求，也不会被探测本身的异常拖死。
 */
export async function tryLoadRuntime(base: string): Promise<OrtGlobal | null> {
  const loaded = await injectScript(`${base}ort.webgpu.min.js`);
  const ort = getOrt();
  if (!loaded || !ort) return null;
  configure(ort, base);
  return ort;
}

/** 引导运行时：优先随应用打包的本地运行时（public/ort/），缺失时回退 CDN 多镜像 */
export async function bootRuntime(): Promise<OrtGlobal> {
  const localBase = `${import.meta.env.BASE_URL}ort/`;
  const local = await tryLoadRuntime(localBase);
  if (local) return local;
  for (const base of ORT_BASES) {
    const ort = await tryLoadRuntime(base);
    if (ort) return ort;
  }
  throw new Error('推理引擎下载失败，请检查网络后重试');
}

/**
 * 本地 wasm 初始化失败后，从 CDN 重新注入运行时。
 * 会覆盖 window.ort —— 这是必须的，因为 initWasm 的失败状态是粘性的，
 * 不换一个全新的运行时对象就救不回来。
 */
async function recoverWithCdn(): Promise<OrtGlobal> {
  const base = ORT_BASES[0];
  const ok = await injectScript(`${base}ort.webgpu.min.js?fallback=${Date.now()}`);
  const ort = getOrt();
  if (!ok || !ort) throw new Error('CDN 推理引擎加载失败，请检查网络');
  configure(ort, base);
  return ort;
}

export interface CreatedSession {
  session: OrtSession;
  ep: 'WebGPU' | 'WASM';
  /**
   * 实际用上的运行时对象。
   * 本地 wasm 初始化失败、走 CDN 重新注入时，会换成**另一个** ort 对象，
   * 调用方必须把引用更新过去，否则下次建会话还会从坏掉的运行时重新兜底一遍。
   */
  ort: OrtGlobal;
}

/**
 * 建立推理会话：WebGPU 优先 → WASM 回退 → CDN 重注入兜底。
 * `onWarn` 用于把「本地 wasm 挂了、改走 CDN」这类降级如实透出，不静默。
 */
export async function createSession(
  ort: OrtGlobal,
  bytes: Uint8Array,
  onWarn?: (msg: string, err: unknown) => void,
): Promise<CreatedSession> {
  const opts = { graphOptimizationLevel: 'all' };
  try {
    const session = await ort.InferenceSession.create(bytes, {
      ...opts,
      executionProviders: ['webgpu'],
    });
    return { session, ep: 'WebGPU', ort };
  } catch {
    // WebGPU 不可用，回退 WASM。无跨源隔离时强制单线程，避免 SharedArrayBuffer 缺失
    ort.env.wasm.numThreads = detectWasmThreads();
    try {
      const session = await ort.InferenceSession.create(bytes, {
        ...opts,
        executionProviders: ['wasm'],
      });
      return { session, ep: 'WASM', ort };
    } catch (wasmErr) {
      // 本地 wasm 初始化失败（多为静态托管对 .wasm 的 MIME / 路径处理不当）。
      // initWasm 的失败是粘性的，本地模块已不可救 → 重新注入 CDN 运行时再试一次。
      onWarn?.('[ort] 本地 wasm 初始化失败，改用 CDN 回退：', wasmErr);
      const ort2 = await recoverWithCdn();
      ort2.env.wasm.numThreads = detectWasmThreads();
      const session = await ort2.InferenceSession.create(bytes, {
        ...opts,
        executionProviders: ['wasm'],
      });
      return { session, ep: 'WASM', ort: ort2 };
    }
  }
}
