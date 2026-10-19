import { create } from 'zustand';
import { VisualPromptDetector, mergePrompts, type VpDetection } from '../ai/visualPrompt';
import {
  DinoPromptDetector,
  mergeProtos,
  dinoFeasibility,
  DINO_Z_DEFAULT,
  type DinoProto,
} from '../ai/dinoPrompt';
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

/**
 * 两种引擎。
 *
 *  · `yoloe`：内置 YOLOE 视觉提示 —— 是个**检测器**，把参考图 + 框交给模型，模型直接出框。
 *  · `dino` ：内置 DINOv2 特征匹配 —— 是个**特征提取器**，框内的 patch 特征当语义模板，
 *             在目标图的 patch 特征上滑动匹配。模型本身不输出框，框是外面算出来的。
 *
 * ⚠️ **DINOv2 有一个硬前提：目标在 518 输入上要占够 patch。**
 * patch 固定 14px、输入固定 518，所以目标在原图上至少要 3*14/gain 像素
 * （1366 宽的图约 111px）。低于这条线时它会输出几十个乱框且召回归零 ——
 * 2026-10-19 在真实截图上实测确认（详见 `ai/dinoPrompt.ts` 的 `dinoFeasibility`）。
 * 那个基于合成数据（目标占 8.3 个 patch）的「参考图少时更强」结论**不能外推到小目标**。
 * 界面上两条并列，让用户拿自己的图去比；但 DINOv2 这条会先过分辨率守卫。
 */
export type VpEngine = 'yoloe' | 'dino';

export interface VpEngineInfo {
  id: VpEngine;
  name: string;
  /** 模型体积（MB），界面文案里如实告诉用户 */
  mb: number;
  /** 一句话说明这条链路是什么 */
  tagline: string;
}

export const VP_ENGINES: readonly VpEngineInfo[] = [
  {
    id: 'yoloe',
    name: 'YOLOE 视觉提示',
    mb: 20,
    tagline: '把例子交给内置检测模型，模型直接给出框',
  },
  {
    id: 'dino',
    name: 'DINOv2 特征匹配',
    mb: 85,
    tagline: '用语义特征做模板匹配 —— 只在目标够大时有效',
  },
];

/** 内置模型路径。用 BASE_URL 拼接，保证部署到子目录时也能取到 */
const MODEL_URLS: Record<VpEngine, string> = {
  yoloe: `${import.meta.env.BASE_URL}models/yoloe26n-vp.onnx`,
  dino: `${import.meta.env.BASE_URL}models/dinov2-small-518.onnx`,
};
/**
 * 阈值的存储键，**按引擎分开**。
 *
 * 两个数字的含义根本不同：YOLOE 是模型输出的置信度（0~1），
 * DINOv2 是「高于本图均值几个标准差」（z，约 1.0~3.5）。混用一个键会让
 * 切引擎时阈值被对方的量纲覆盖，表现为「换条链路就找不到东西了」。
 *
 * 复用 modelStore 的「跟着模型记」机制 —— 换回来时阈值还在。
 */
const CONF_KEYS: Record<VpEngine, string> = {
  yoloe: 'yoloe-26n-visual-prompt',
  dino: 'dinov2-small-visual-prompt',
};
const DEFAULT_CONF: Record<VpEngine, number> = {
  yoloe: 0.25,
  dino: DINO_Z_DEFAULT,
};

export function modelMbOf(engine: VpEngine): number {
  return VP_ENGINES.find((e) => e.id === engine)?.mb ?? 0;
}

export type VpStatus = 'idle' | 'loading' | 'ready' | 'error';
export type VpBusy = 'none' | 'prompt' | 'current' | 'batch';

interface VpState {
  /** 当前用哪条引擎 */
  engine: VpEngine;
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
   * 拦下这次操作的原因（目前只有 DINOv2 的分辨率守卫）。
   *
   * 用持久状态而不是只弹 toast：toast 一闪而过，用户回头看不到「为什么不让跑」，
   * 只会以为功能坏了。文案由 `dinoFeasibility` 生成，带具体像素数字。
   */
  promptWarn: string;

  /** 当前引擎的阈值。YOLOE 是置信度（0~1）；DINOv2 是 z 值（1.0~3.5），两者不可比 */
  conf: number;
  busy: VpBusy;
  batchDone: number;
  batchTotal: number;
  /** 试标结果落在哪张图（界面据此提示「去那张图看看」） */
  trialImageId: string | null;

  /** 切换引擎：清掉已编码的提示，阈值按引擎各自恢复 */
  setEngine: (e: VpEngine) => void;
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

/** 两个引擎各一个实例：会话互不影响，切回来时不必重新下载 */
const yoloeDetector = new VisualPromptDetector();
const dinoDetector = new DinoPromptDetector();
let batchCancelled = false;

/**
 * 提示放模块变量而不是 store：
 * 它是几百个 float 的裸数组，塞进 state 只会让每次 set 都多一次无意义的引用比较，
 * 而界面需要的只是「有没有就绪」这个布尔量（`promptReady`）。
 *
 * 用带 tag 的联合类型而不是 `any`：检测时分派要能看出用的是哪条链路，
 * 写错引擎（拿 512 维向量去喂 DINOv2）必须在编译期就报错。
 */
type VpPrompt = { engine: 'yoloe'; vec: Float32Array } | { engine: 'dino'; proto: DinoProto };
let prompt: VpPrompt | null = null;

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

/** 按提示自带的 engine tag 分派到对应检测器（类型收窄由联合类型保证） */
function detectWith(p: VpPrompt, bmp: ImageBitmap, conf: number): Promise<VpDetection[]> {
  return p.engine === 'dino'
    ? dinoDetector.detect(bmp, p.proto, conf)
    : yoloeDetector.detect(bmp, p.vec, conf);
}

/** 清掉已编码的提示（参考图或类别一变，旧提示就失效了） */
function clearPrompt(set: (p: Partial<VpState>) => void): void {
  prompt = null;
  set({ promptReady: false, promptRefs: 0, promptBoxes: 0, promptWarn: '' });
}

export const useVpStore = create<VpState>((set, get) => ({
  engine: 'yoloe',
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

  conf: loadConf(CONF_KEYS.yoloe) ?? DEFAULT_CONF.yoloe,
  busy: 'none',
  batchDone: 0,
  batchTotal: 0,
  trialImageId: null,

  setEngine: (e) => {
    if (get().engine === e) return;
    // 换引擎等于换了一整套「提示」，旧的必须作废
    clearPrompt(set);
    const det = e === 'dino' ? dinoDetector : yoloeDetector;
    set({
      engine: e,
      conf: loadConf(CONF_KEYS[e]) ?? DEFAULT_CONF[e],
      error: '',
      progress: 0,
      // 该引擎的会话若还活着就直接标 ready —— 切回来不必重新下载那 85MB
      status: det.ready ? 'ready' : 'idle',
      ep: det.ready ? det.ep : '',
      warmupError: det.ready ? det.warmupError : '',
    });
  },

  ensureReady: async () => {
    const st = get();
    if (st.status === 'ready') return true;
    // 正在下载时再点一次不应该重复发起 —— 静默返回，不弹错
    if (st.status === 'loading') return false;
    const eng = st.engine;
    const det = eng === 'dino' ? dinoDetector : yoloeDetector;
    if (det.ready) {
      set({ status: 'ready', progress: 100, ep: det.ep, warmupError: det.warmupError });
      return true;
    }
    set({ status: 'loading', error: '', progress: 0, warmupError: '' });
    try {
      await det.boot();
      const bytes = await fetchWithProgress(MODEL_URLS[eng], (pct) => set({ progress: pct }));
      await det.loadModel(new Uint8Array(bytes));
      // 下载期间用户可能已经切走引擎了，这时不该把状态盖回当前引擎
      if (get().engine !== eng) return det.ready;
      set({
        status: 'ready',
        progress: 100,
        ep: det.ep,
        warmupError: det.warmupError,
      });
      return true;
    } catch (e) {
      if (get().engine === eng) set({ status: 'error', error: messageOf(e), progress: 0 });
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
    saveConf(CONF_KEYS[get().engine], v);
  },

  buildPrompt: async () => {
    if (get().busy !== 'none') return;
    const eng = get().engine;
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

    // DINOv2 的**分辨率守卫必须在下载 85MB 之前跑**。
    // 实测（冒险岛 1366x768、目标 55px）：目标只占 1 个 patch 时召回为 0，
    // 还会吐 30+ 个乱框。下完模型再拒绝等于让用户白等一场 —— 所以先拦。
    // 尺寸取真实解码出来的 bitmap，不用 store 里的 width/height（那是懒测量的，可能是 0）。
    const bitmaps = new Map<string, ImageBitmap>();
    if (eng === 'dino') {
      for (const { img, boxes } of usable) {
        const bmp = await loadBitmap(img);
        if (!bmp) continue;
        bitmaps.set(img.id, bmp);
        const feas = dinoFeasibility(bmp.width, bmp.height, boxes);
        if (!feas.ok) {
          clearPrompt(set);
          set({ promptWarn: feas.reason });
          toast.warning('DINOv2 用不了这批框 —— 目标太小了（详见面板里的说明）');
          return;
        }
      }
    }

    if (!(await get().ensureReady())) {
      toast.error(`模型没准备好：${get().error || '请检查网络后重试'}`);
      return;
    }

    set({ busy: 'prompt' });
    try {
      let totalBoxes = 0;
      let used = 0;
      if (eng === 'dino') {
        const protos: DinoProto[] = [];
        for (const { img, boxes } of usable) {
          const bmp = bitmaps.get(img.id) ?? (await loadBitmap(img));
          if (!bmp) continue;
          protos.push(await dinoDetector.encodePrompt(bmp, boxes));
          totalBoxes += boxes.length;
        }
        if (protos.length === 0) throw new Error('参考图都打不开，无法编码');
        prompt = { engine: 'dino', proto: mergeProtos(protos) };
        used = protos.length;
      } else {
        const vectors: Float32Array[] = [];
        for (const { img, boxes } of usable) {
          const bmp = await loadBitmap(img);
          if (!bmp) continue;
          vectors.push(await yoloeDetector.encodePrompt(bmp, boxes));
          totalBoxes += boxes.length;
        }
        if (vectors.length === 0) throw new Error('参考图都打不开，无法编码');
        prompt = { engine: 'yoloe', vec: mergePrompts(vectors) };
        used = vectors.length;
      }
      set({ promptReady: true, promptRefs: used, promptBoxes: totalBoxes, promptWarn: '' });
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
      const dets = await detectWith(prompt!, bmp, get().conf);
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
        const dets = await detectWith(prompt!, bmp, get().conf);
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
