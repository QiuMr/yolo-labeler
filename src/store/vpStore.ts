import { create } from 'zustand';
import { VisualPromptDetector, mergePrompts, type VpDetection } from '../ai/visualPrompt';
import type { Annotation, IRect } from '../types';
import { loadBitmap } from '../utils/imageCache';
import { fetchWithProgress } from '../utils/modelFetch';
import { loadConf, saveConf } from '../utils/modelStore';
import { buildAiAnnotations } from '../utils/aiBoxes';
import {
  summarizeBatch,
  batchHeadline,
  type VpBatchStat,
  type VpBatchSummary,
} from '../utils/vpBatch';
import { useAiStore, type AiCandidate } from './aiStore';
import { useStore } from './useStore';
import { toast } from './toastStore';

/**
 * 「照样子找」（YOLOE 视觉提示）的状态层。
 *
 * 与另外两条链路的区别，决定了它必须是第三个 store：
 *  - `aiStore`：用户**自备** .onnx，模型里写死了类别，跑完给候选框；
 *  - `segStore`：内置 MobileSAM，点一下出一个框，没有候选框这一步；
 *  - 这里：**内置** YOLOE，类别来自「用户自己标的那几个框」，
 *    跑完也给候选框，但**提示源**（用哪几张图、哪个类别）是这条链路独有的状态。
 *
 * 候选框本身**不在这里存**：它和 aiStore 产出的是同一种东西（待采纳的建议），
 * 画布只有一处渲染逻辑，分两份会互相盖住。产出后交给 `aiStore.setCandidates`。
 */

/**
 * 内置模型：YOLOE 视觉提示。
 *
 * 它是一个**检测器** —— 把「参考图 + 框」交给模型，模型直接给出框。
 * 与「AI 辅助标注」那条链路的区别在于：类别不是模型里写死的，
 * 而是**用户自己标的那几个框**。
 *
 * 曾经还有第二条引擎（DINOv2 特征匹配，85MB），2026-10-19 在真实截图上实测后砍掉：
 * 它是特征提取器、patch 固定 14px，目标在 518 输入上占不满 3 个 patch 时窗口退化成
 * 1×1、召回直接归零，还会吐几十个乱框 —— 那批 1366×768 截图里的目标只有 55px，
 * 属于**原理性天花板**，调阈值和换参考图都救不回来。留一条能用的，
 * 好过留两条「一条好用、另一条要用户自己猜什么时候能用」。
 */
export interface VpEngineInfo {
  name: string;
  /** 模型体积（MB），界面文案里如实告诉用户 */
  mb: number;
  /** 一句话说明这条链路是什么 */
  tagline: string;
}

export const VP_MODEL: VpEngineInfo = {
  name: 'YOLOE 视觉提示',
  mb: 20,
  tagline: '把例子交给内置检测模型，模型直接给出框',
};

/** 内置模型路径。用 BASE_URL 拼接，保证部署到子目录时也能取到 */
const MODEL_URL = `${import.meta.env.BASE_URL}models/yoloe26n-vp.onnx`;

/**
 * 阈值的存储键。
 *
 * 复用 modelStore 的「跟着模型记」机制 —— 下次打开还是你调好的值。
 * 键名保持 `yoloe-26n-visual-prompt` **不变**：老用户已经存过的阈值
 * 不能因为这次删引擎而丢掉。
 */
const CONF_KEY = 'yoloe-26n-visual-prompt';
const DEFAULT_CONF = 0.25;

export type VpStatus = 'idle' | 'loading' | 'ready' | 'error';
export type VpBusy = 'none' | 'prompt' | 'current' | 'batch';

interface VpState {
  status: VpStatus;
  error: string;
  /** 下载进度 0~100；-1 表示拿不到总长度 */
  progress: number;
  ep: '' | 'WebGPU' | 'WASM';
  /** 预热报错（非空基本等于「模型跑不起来」） */
  warmupError: string;

  /** 当参考图的图片 id（用户勾选的） */
  refImageIds: string[];
  /** 用哪个类别的框当提示。空串 = 还没选 */
  labelId: string;

  /** 提示向量是否已就绪 */
  promptReady: boolean;
  /** 提示实际用上了几张参考图 / 几个框（可能小于勾选数：没这个类别的图会被跳过） */
  promptRefs: number;
  promptBoxes: number;
  /**
   * 拦下这次操作的原因（编码失败 / 参考图打不开）。
   *
   * 用持久状态而不是只弹 toast：toast 一闪而过，用户回头看不到「为什么不让跑」，
   * 只会以为功能坏了。
   */
  promptWarn: string;

  /** 置信度阈值（0~1），跟着模型记住 */
  conf: number;
  busy: VpBusy;
  batchDone: number;
  batchTotal: number;
  /** 试标结果落在哪张图（界面据此提示「去那张图看看」） */
  trialImageId: string | null;
  /**
   * 上一次批量的汇总。批量跑完留着不擦，用户关掉面板再回来还能看到「哪几张要重点看」。
   * 跑新的批量时会被覆盖。
   */
  batchSummary: VpBatchSummary | null;
  /** 异常图的原因，按 imageId 索引（batchSummary.flags 的查找表） */
  batchFlags: Record<string, string>;

  /** 按需下载 + 加载内置模型。返回是否可用 */
  ensureReady: () => Promise<boolean>;
  /** 打开面板时调用：没选过参考图就默认选「当前这张」（有框的话） */
  initRefs: () => void;
  toggleRef: (imageId: string) => void;
  setLabelId: (id: string) => void;
  setConf: (v: number) => void;
  /** 编码提示：参考图 + 框 → 提示向量 */
  buildPrompt: () => Promise<void>;
  /** 先只标一张给用户过目 */
  runTrial: () => Promise<void>;
  /** 批量跑（会排除参考图本身）。结果先攒成待复核候选框，不直接落框 */
  runBatch: (scope?: 'empty' | 'all') => Promise<void>;
  cancelBatch: () => void;
  /** 采纳整批待复核候选框（合成一条撤销） */
  acceptBatch: () => void;
  /** 跳到某张图去复核它的候选框 */
  gotoFlagged: (imageId: string) => void;
}

const yoloeDetector = new VisualPromptDetector();
let batchCancelled = false;

/**
 * 提示放模块变量而不是 store：
 * 它是几百个 float 的裸数组，塞进 state 只会让每次 set 都多一次无意义的引用比较，
 * 而界面需要的只是「有没有就绪」这个布尔量（`promptReady`）。
 */
let prompt: Float32Array | null = null;

function messageOf(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** 取某张图里指定类别的框（原图坐标） */
function boxesOf(img: { annotations: readonly { labelId: string; rect: IRect }[] }, labelId: string): IRect[] {
  if (!labelId) return [];
  return img.annotations.filter((a) => a.labelId === labelId).map((a) => a.rect);
}

function toCandidates(dets: readonly VpDetection[], className: string): AiCandidate[] {
  return dets.map((d) => ({
    rect: d.rect,
    score: d.score,
    className,
    // 这条链路没有「类号」概念（提示就是类别本身），固定 0 即可
    classIndex: 0,
  }));
}

/**
 * 干跑一遍落框逻辑，返回**将来真会落下的那些框**。
 *
 * 为什么不直接存原始检测结果 —— 有三个坑，全都在用户那边才炸：
 *  · 越界的框：letterbox 反算出的坐标偶尔偏出图外，导出的 YOLO 标签会超出 0~1；
 *  · 退化的框（宽或高 < 2px）：手工画框时都不许入库，AI 的却进来了；
 *  · 与已有框重复：「调低阈值再跑一遍」是标注调优的标准动作，
 *    纯追加会让同一物体挂两个框，用户只能手工删（这条规则的理由见 utils/aiBoxes.ts）。
 *
 * 落框时本来就会走一遍这套规则（useStore.applyAiBoxes → buildAiAnnotations）；
 * 批量只是**提前**走，好处是用户在看汇总时看到的就是最终会落下的数量，
 * 不会出现「汇总说 412 个框，采纳完变成 391 个」这种对不上账的事。
 */
function dryRunCandidates(
  existing: readonly Annotation[],
  width: number,
  height: number,
  dets: readonly VpDetection[],
  className: string,
): AiCandidate[] {
  const st = useStore.getState();
  // 用一份**本地**标签表：这一步只是数框，不该真的往用户的类别表里插东西。
  // 类别还不存在时 existing 里也不可能有它，所以 createLabel 永远不会被调到。
  const labels = st.labels.slice();
  const nameToId = new Map(labels.map((l) => [l.name, l.id] as const));
  const { anns } = buildAiAnnotations({
    existing,
    entries: dets.map((d) => ({ rect: d.rect, className, score: d.score })),
    labels,
    nameToId,
    width,
    height,
    createLabel: () => ({ id: 'vp-dryrun', name: className, color: '#888888' }),
    createId: () => 'vp-dryrun',
  });
  return anns.map((a) => ({
    rect: a.rect,
    score: a.score ?? 0,
    className,
    classIndex: 0,
  }));
}

/** 清掉已编码的提示（参考图或类别一变，旧提示就失效了） */
function clearPrompt(set: (p: Partial<VpState>) => void): void {
  prompt = null;
  set({ promptReady: false, promptRefs: 0, promptBoxes: 0, promptWarn: '' });
}

export const useVpStore = create<VpState>((set, get) => ({
  status: 'idle',
  error: '',
  progress: 0,
  ep: '',
  warmupError: '',

  refImageIds: [],
  labelId: '',

  promptReady: false,
  promptRefs: 0,
  promptBoxes: 0,
  promptWarn: '',

  conf: loadConf(CONF_KEY) ?? DEFAULT_CONF,
  busy: 'none',
  batchDone: 0,
  batchTotal: 0,
  trialImageId: null,
  batchSummary: null,
  batchFlags: {},

  ensureReady: async () => {
    const st = get();
    if (st.status === 'ready') return true;
    // 正在下载时再点一次不应该重复发起 —— 静默返回，不弹错
    if (st.status === 'loading') return false;
    const det = yoloeDetector;
    if (det.ready) {
      set({ status: 'ready', progress: 100, ep: det.ep, warmupError: det.warmupError });
      return true;
    }
    set({ status: 'loading', error: '', progress: 0, warmupError: '' });
    try {
      await det.boot();
      const bytes = await fetchWithProgress(MODEL_URL, (pct) => set({ progress: pct }));
      await det.loadModel(new Uint8Array(bytes));
      set({
        status: 'ready',
        progress: 100,
        ep: det.ep,
        warmupError: det.warmupError,
      });
      return true;
    } catch (e) {
      set({ status: 'error', error: messageOf(e), progress: 0 });
      return false;
    }
  },

  initRefs: () => {
    const st = get();
    if (st.refImageIds.length > 0) return;
    const cur = useStore.getState();
    const withBoxes = cur.images.filter((i) => i.annotations.length > 0);
    if (withBoxes.length === 0) {
      set({ refImageIds: [], labelId: '' });
      return;
    }
    // 默认用**当前这张**：用户多半就是刚在这张图上标完，过来点「照样子找」的
    const active = cur.images[cur.activeIndex];
    const pick = active && active.annotations.length > 0 ? active : withBoxes[0];
    set({
      refImageIds: [pick.id],
      labelId: get().labelId || pick.annotations[0]?.labelId || '',
    });
  },

  toggleRef: (imageId) => {
    const cur = get().refImageIds;
    const next = cur.includes(imageId)
      ? cur.filter((id) => id !== imageId)
      : [...cur, imageId];
    set({ refImageIds: next });
    clearPrompt(set);
  },

  setLabelId: (id) => {
    if (get().labelId === id) return;
    set({ labelId: id });
    clearPrompt(set);
  },

  setConf: (v) => {
    set({ conf: v });
    saveConf(CONF_KEY, v);
  },

  buildPrompt: async () => {
    if (get().busy !== 'none') return;
    const { refImageIds, labelId } = get();
    const cur = useStore.getState();
    const refs = cur.images.filter((i) => refImageIds.includes(i.id));
    if (refs.length === 0) {
      toast.error('先选一张参考图');
      return;
    }
    if (!labelId) {
      toast.error('先选一个类别 —— 模型学的就是「这个类别的框长什么样」');
      return;
    }

    // 每张参考图都得有该类别的框，否则它提供不了任何信息，直接跳过并如实告知
    const usable: Array<{ img: (typeof refs)[number]; boxes: IRect[] }> = [];
    for (const img of refs) {
      const boxes = boxesOf(img, labelId);
      if (boxes.length > 0) usable.push({ img, boxes });
    }
    if (usable.length === 0) {
      toast.error('这些参考图上没有这个类别的框。换一张图，或者换个类别。');
      return;
    }

    if (!(await get().ensureReady())) {
      toast.error(`模型没准备好：${get().error || '请检查网络后重试'}`);
      return;
    }

    set({ busy: 'prompt' });
    try {
      let totalBoxes = 0;
      const vectors: Float32Array[] = [];
      for (const { img, boxes } of usable) {
        const bmp = await loadBitmap(img);
        if (!bmp) continue;
        vectors.push(await yoloeDetector.encodePrompt(bmp, boxes));
        totalBoxes += boxes.length;
      }
      if (vectors.length === 0) throw new Error('参考图都打不开，无法编码');
      prompt = mergePrompts(vectors);
      set({ promptReady: true, promptRefs: vectors.length, promptBoxes: totalBoxes, promptWarn: '' });
      const skipped = refs.length - usable.length;
      if (skipped > 0) {
        toast.info(`有 ${skipped} 张参考图上没有这个类别的框，已跳过`);
      }
    } catch (e) {
      clearPrompt(set);
      // 守卫的拒绝也要留在界面上 —— 只弹 toast 的话用户回头看不到原因
      set({ promptWarn: messageOf(e) });
      toast.error(`编码失败：${messageOf(e)}`);
    } finally {
      set({ busy: 'none' });
    }
  },

  runTrial: async () => {
    const st = get();
    if (st.busy !== 'none') return;
    if (!st.promptReady) {
      await get().buildPrompt();
      if (!get().promptReady) return;
    }
    const cur = useStore.getState();
    const refSet = new Set(get().refImageIds);
    const images = cur.images;

    // 挑一张「不是参考图、还没有框」的图来试。
    // 优先当前这张（用户正看着它），否则挑第一张符合条件的。
    const active = images[cur.activeIndex];
    let idx = -1;
    if (active && !refSet.has(active.id) && active.annotations.length === 0) {
      idx = cur.activeIndex;
    } else {
      idx = images.findIndex((i) => !refSet.has(i.id) && i.annotations.length === 0);
    }
    if (idx < 0) {
      toast.info('没有可试标的目标图了 —— 未标注的图都已经跑过或本身就是参考图');
      return;
    }

    const labelName = cur.labels.find((l) => l.id === get().labelId)?.name ?? '';
    set({ busy: 'current' });
    try {
      const img = images[idx];
      const bmp = await loadBitmap(img);
      if (!bmp) throw new Error('这张图打不开');
      const dets = await yoloeDetector.detect(bmp, prompt!, get().conf);
      // 交给 aiStore：画布与「全部采纳 / 放弃」状态条只认这一处
      useAiStore.getState().setCandidates(toCandidates(dets, labelName), img.id);
      cur.setActiveIndex(idx);
      set({ trialImageId: img.id });
      if (dets.length === 0) {
        toast.info(
          '这张图没找到像它的东西。可以先调低置信度；如果还不行，换一张参考图 —— 实测换图比多画框有效得多。',
        );
      }
    } catch (e) {
      toast.error(`识别失败：${messageOf(e)}`);
    } finally {
      set({ busy: 'none' });
    }
  },

  runBatch: async (scope = 'empty') => {
    const st = get();
    if (st.busy !== 'none') return;
    if (!st.promptReady) {
      await get().buildPrompt();
      if (!get().promptReady) return;
    }
    const cur = useStore.getState();
    const refSet = new Set(get().refImageIds);
    const labelName = cur.labels.find((l) => l.id === get().labelId)?.name ?? '';
    // 参考图永远排除：它已经有框了，再跑一遍只会产出重复框（去重会挡掉，但白费一次推理）
    const targets = cur.images.filter(
      (i) => !refSet.has(i.id) && (scope === 'all' || i.annotations.length === 0),
    );
    if (targets.length === 0) {
      toast.info(scope === 'all' ? '没有可处理的图片' : '剩下的图都已经有标注了');
      return;
    }

    batchCancelled = false;
    set({
      busy: 'batch',
      batchDone: 0,
      batchTotal: targets.length,
      // 试标留下的候选框不该和批量结果叠在同一张图上
      trialImageId: null,
      batchSummary: null,
      batchFlags: {},
    });
    // 上一批没采纳的候选框作废：用户点了新的批量，就等于重新要一批
    useAiStore.getState().clearPending();

    const stats: VpBatchStat[] = [];
    let i = 0;
    for (const img of targets) {
      if (batchCancelled) break;
      i += 1;
      let bmp: ImageBitmap | null = null;
      try {
        bmp = await loadBitmap(img);
      } catch {
        bmp = null;
      }
      if (!bmp) {
        // 解码失败要单独记账：用户分不清「模型没识别到」和「这张图打不开」
        stats.push({ imageId: img.id, count: 0, meanScore: 0, failed: true });
        set({ batchDone: i });
        continue;
      }
      try {
        const dets = await yoloeDetector.detect(bmp, prompt!, get().conf);
        const cands = dryRunCandidates(
          img.annotations,
          bmp.width,
          bmp.height,
          dets,
          labelName,
        );
        // 一张都没找到也要写：汇总里的「N 张没找到」靠这个计数，
        // 而它是用户最需要看到的一类（可能这张图真的没有，也可能例子选错了）。
        useAiStore.getState().putPending(img.id, cands, bmp.width, bmp.height);
        stats.push({
          imageId: img.id,
          count: cands.length,
          meanScore: cands.length > 0 ? cands.reduce((n, c) => n + c.score, 0) / cands.length : 0,
          failed: false,
        });
      } catch {
        // 单张失败不打断整批，但必须记账
        stats.push({ imageId: img.id, count: 0, meanScore: 0, failed: true });
      }
      set({ batchDone: i });
      // 每 3 张让出一次控制权，否则整个批次里浏览器一次都没 paint 过
      if (i % 3 === 0) await new Promise((r) => setTimeout(r, 0));
    }

    const summary = summarizeBatch(stats);
    const flags: Record<string, string> = {};
    for (const f of summary.flags) flags[f.imageId] = f.reason;
    set({ busy: 'none', batchSummary: summary, batchFlags: flags });
    // 把当前图的候选框装上：用户此时多半正停在某张图上，不装的话画布是空的
    useAiStore.getState().syncActive();

    if (batchCancelled) {
      const done = useAiStore.getState().pendingImages;
      toast.info(`已停止，${done} 张图有候选框待复核`);
    } else if (summary.withBoxes > 0) {
      const flagNote = summary.flags.length > 0 ? `，其中 ${summary.flags.length} 张建议重点看` : '';
      const failNote = summary.failed > 0 ? `，${summary.failed} 张失败` : '';
      toast.success(`照样子找完成：${batchHeadline(summary)}${flagNote}${failNote}`);
    } else if (summary.failed > 0) {
      toast.warning(`${summary.failed} 张图失败，没有找到任何目标（多为图片损坏或内存不足）`);
    } else {
      toast.info('这批图里没找到像它的东西。换一张参考图通常比调阈值更有效。');
    }
  },

  cancelBatch: () => {
    batchCancelled = true;
  },

  acceptBatch: () => {
    useAiStore.getState().acceptPendingAll();
    // 采纳完就没有「待复核」了，汇总面板该让位给「已采纳」的干净状态
    set({ batchSummary: null, batchFlags: {} });
  },

  gotoFlagged: (imageId) => {
    const idx = useStore.getState().images.findIndex((i) => i.id === imageId);
    if (idx >= 0) useStore.getState().setActiveIndex(idx);
  },
}));
