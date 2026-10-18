import { create } from 'zustand';
import { VisualPromptDetector, mergePrompts, type VpDetection } from '../ai/visualPrompt';
import type { IRect } from '../types';
import { loadBitmap } from '../utils/imageCache';
import { fetchWithProgress } from '../utils/modelFetch';
import { loadConf, saveConf } from '../utils/modelStore';
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

/** 内置模型路径。用 BASE_URL 拼接，保证部署到子目录时也能取到 */
const MODEL_URL = `${import.meta.env.BASE_URL}models/yoloe26n-vp.onnx`;
/** 模型体积，界面文案里如实告诉用户（首次打开面板才下载） */
export const VP_MODEL_MB = 20;
/**
 * 置信度阈值的存储键。
 * 复用 modelStore 的「跟着模型记」机制 —— 换回这个链路时阈值还在，
 * 否则「调好阈值 → 刷新 → 回默认」会让同一批图的标注结果前后不一致。
 */
const CONF_MODEL_KEY = 'yoloe-26n-visual-prompt';
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

  conf: number;
  busy: VpBusy;
  batchDone: number;
  batchTotal: number;
  /** 试标结果落在哪张图（界面据此提示「去那张图看看」） */
  trialImageId: string | null;

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
  /** 批量跑（会排除参考图本身） */
  runBatch: (scope?: 'empty' | 'all') => Promise<void>;
  cancelBatch: () => void;
}

const vpDetector = new VisualPromptDetector();
let batchCancelled = false;

/**
 * 提示向量放模块变量而不是 store：
 * 它是 512 个 float 的裸数组，塞进 state 只会让每次 set 都多一次无意义的引用比较，
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

/** 清掉已编码的提示（参考图或类别一变，旧提示就失效了） */
function clearPrompt(set: (p: Partial<VpState>) => void): void {
  prompt = null;
  set({ promptReady: false, promptRefs: 0, promptBoxes: 0 });
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

  conf: loadConf(CONF_MODEL_KEY) ?? DEFAULT_CONF,
  busy: 'none',
  batchDone: 0,
  batchTotal: 0,
  trialImageId: null,

  ensureReady: async () => {
    const st = get();
    if (st.status === 'ready') return true;
    // 正在下载时再点一次不应该重复发起 —— 静默返回，不弹错
    if (st.status === 'loading') return false;
    set({ status: 'loading', error: '', progress: 0, warmupError: '' });
    try {
      await vpDetector.boot();
      const bytes = await fetchWithProgress(MODEL_URL, (pct) => set({ progress: pct }));
      await vpDetector.loadModel(new Uint8Array(bytes));
      set({
        status: 'ready',
        progress: 100,
        ep: vpDetector.ep,
        warmupError: vpDetector.warmupError,
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
    saveConf(CONF_MODEL_KEY, v);
  },

  buildPrompt: async () => {
    if (get().busy !== 'none') return;
    if (!(await get().ensureReady())) {
      toast.error(`模型没准备好：${get().error || '请检查网络后重试'}`);
      return;
    }
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

    set({ busy: 'prompt' });
    try {
      const vectors: Float32Array[] = [];
      let totalBoxes = 0;
      for (const { img, boxes } of usable) {
        const bmp = await loadBitmap(img);
        if (!bmp) continue;
        vectors.push(await vpDetector.encodePrompt(bmp, boxes));
        totalBoxes += boxes.length;
      }
      if (vectors.length === 0) throw new Error('参考图都打不开，无法编码');
      prompt = mergePrompts(vectors);
      set({ promptReady: true, promptRefs: vectors.length, promptBoxes: totalBoxes });
      const skipped = refs.length - usable.length;
      if (skipped > 0) {
        toast.info(`有 ${skipped} 张参考图上没有这个类别的框，已跳过`);
      }
    } catch (e) {
      clearPrompt(set);
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
      const dets = await vpDetector.detect(bmp, prompt!, get().conf);
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
    });
    useAiStore.getState().setCandidates([], '');

    let boxCount = 0;
    let touched = 0;
    let failed = 0;
    let dupSkipped = 0;
    let i = 0;
    // 整批合成一条撤销记录：Ctrl+Z 一次退掉整批
    useStore.getState().beginAiBatch();
    for (const img of targets) {
      if (batchCancelled) break;
      i += 1;
      const bmp = await loadBitmap(img);
      if (!bmp) {
        failed += 1;
        set({ batchDone: i });
        continue;
      }
      try {
        const dets = await vpDetector.detect(bmp, prompt!, get().conf);
        if (dets.length > 0) {
          const res = useStore
            .getState()
            .applyAiBoxes(img.id, toCandidates(dets, labelName), bmp.width, bmp.height);
          boxCount += res.added;
          dupSkipped += res.skipped;
          if (res.added > 0) touched += 1;
        }
      } catch {
        // 单张失败不打断整批，但必须记账
        failed += 1;
      }
      set({ batchDone: i });
      // 每 3 张让出一次控制权，否则整个批次里浏览器一次都没 paint 过
      if (i % 3 === 0) await new Promise((r) => setTimeout(r, 0));
    }
    useStore.getState().endAiBatch();

    set({ busy: 'none' });
    const dupSuffix = dupSkipped > 0 ? `，跳过 ${dupSkipped} 个与已有框重复的` : '';
    if (batchCancelled) {
      toast.info(`已停止，本次共标注 ${touched} 张图${dupSuffix}`);
    } else if (touched > 0) {
      const suffix = failed > 0 ? `，另有 ${failed} 张推理失败（多为图片损坏或内存不足）` : '';
      toast.success(
        `照样子找完成：${touched} 张图、${boxCount} 个框${dupSuffix}（Ctrl+Z 一次撤销整批）${suffix}`,
      );
    } else if (dupSkipped > 0) {
      toast.info(`找到的目标都已经有框了，又跳过 ${dupSkipped} 个重复的，没有新增`);
    } else if (failed > 0) {
      toast.warning(`${failed} 张图推理失败，没有成功标注任何一张（多为图片损坏或内存不足）`);
    } else {
      toast.info('这批图里没找到像它的东西。换一张参考图通常比调阈值更有效。');
    }
  },

  cancelBatch: () => {
    batchCancelled = true;
  },
}));
