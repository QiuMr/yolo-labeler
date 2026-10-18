import { create } from 'zustand';
import { Detector } from '../ai/detector';
import { classNameFor } from '../ai/cocoNames';
import type { IRect } from '../types';
import { loadBitmap } from '../utils/imageCache';
import { loadSavedModel, saveModel } from '../utils/modelStore';
import { useStore } from './useStore';
import { toast } from './toastStore';

/** AI 候选框（还没被采纳，画布上显示为虚线） */
export interface AiCandidate {
  rect: IRect;
  score: number;
  className: string;
}

export type AiStatus = 'idle' | 'booting' | 'loading' | 'ready' | 'error';
export type AiBusy = 'none' | 'current' | 'batch';

interface AiState {
  status: AiStatus;
  error: string;
  modelName: string;
  modelSize: number;
  ep: '' | 'WebGPU' | 'WASM';
  inputW: number;
  inputH: number;
  numClasses: number;
  /** 模型自带的类别名（ONNX metadata 的 `names`）。空数组=模型没写类名，走索引兜底 */
  modelNames: string[];
  /** 模型自带任务类型（metadata 的 `task`）。空串=读不到 */
  modelTask: string;
  /** 输入尺寸来源：onnx / meta / probe:768 / default（排查用） */
  inputSizeFrom: string;
  /** 最近一次推理的输出布局（排查用） */
  layoutInfo: string;
  /** 预热报错：非空基本等于「模型现在跑不起来」 */
  warmupError: string;
  /** 置信度阈值 */
  conf: number;
  busy: AiBusy;
  batchDone: number;
  batchTotal: number;
  /** 当前图的候选框（等待采纳 / 忽略） */
  candidates: AiCandidate[];
  candidatesImageId: string | null;

  /** 打开 AI 面板时调用：引导引擎 + 自动找模型（内置 → 上次选过的） */
  init: () => Promise<void>;
  loadModelFile: (file: File) => Promise<void>;
  setConf: (v: number) => void;
  runCurrent: () => Promise<void>;
  /** 批量跑：empty=只跑还没有任何框的图；all=所有图都跑（已有框的图会追加） */
  runBatch: (scope?: 'empty' | 'all') => Promise<void>;
  cancelBatch: () => void;
  acceptAll: () => void;
  discardCandidates: () => void;
}

const detector = new Detector();
let batchCancelled = false;

function messageOf(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function toCandidates(
  dets: Array<{ rect: IRect; score: number; classIndex: number }>,
): AiCandidate[] {
  return dets.map((d) => ({
    rect: d.rect,
    score: d.score,
    // 优先用模型自带的类名；读不到才回退到 COCO 表 / 类别 N
    className: classNameFor(d.classIndex, detector.numClasses, detector.modelNames),
  }));
}

export const useAiStore = create<AiState>((set, get) => {
  /** 模型字节 → 会话就绪。引擎必须已经 boot 过 */
  async function loadBytes(bytes: ArrayBuffer, name: string, size: number): Promise<void> {
    set({ status: 'loading', error: '', modelName: name, modelSize: size });
    await detector.loadModel(new Uint8Array(bytes));
    set({
      status: 'ready',
      ep: detector.ep,
      inputW: detector.inputW,
      inputH: detector.inputH,
      numClasses: detector.numClasses,
      modelNames: detector.modelNames ?? [],
      modelTask: detector.modelTask,
      inputSizeFrom: detector.inputSizeFrom,
      layoutInfo: detector.layoutInfo,
      warmupError: detector.warmupError,
    });
  }

  return {
    status: 'idle',
    error: '',
    modelName: '',
    modelSize: 0,
    ep: '',
    inputW: 0,
    inputH: 0,
    numClasses: 0,
    modelNames: [],
    modelTask: '',
    inputSizeFrom: '',
    layoutInfo: '',
    warmupError: '',
    conf: 0.25,
    busy: 'none',
    batchDone: 0,
    batchTotal: 0,
    candidates: [],
    candidatesImageId: null,

    init: async () => {
      const st = get();
      if (st.status === 'booting' || st.status === 'loading' || st.status === 'ready') return;
      set({ status: 'booting', error: '' });
      try {
        // 引擎引导与「上次选过的模型」查找并行进行。
        // 不内置任何默认模型——用户用什么模型只有他自己知道；
        // 选过之后浏览器记住选择，下次打开自动复用。
        const bootPromise = detector.boot();
        const modelPromise = loadSavedModel();
        await bootPromise;
        const model = await modelPromise;
        if (!model?.bytes) {
          // 引擎已就绪但还没挑过模型：面板引导用户手动选一个
          set({ status: 'idle' });
          return;
        }
        await loadBytes(model.bytes, model.name, model.size);
      } catch (e) {
        set({ status: 'error', error: messageOf(e) });
      }
    },

    loadModelFile: async (file) => {
      try {
        const bytes = await file.arrayBuffer();
        await loadBytes(bytes, file.name, file.size);
        // 存一份，下次打开直接用
        void saveModel(file.name, file.size, bytes);
        toast.success('模型已就绪');
      } catch (e) {
        set({ status: 'error', error: `模型加载失败：${messageOf(e)}` });
      }
    },

    setConf: (v) => set({ conf: v }),

    runCurrent: async () => {
      const st = get();
      if (st.status !== 'ready' || st.busy !== 'none') return;
      const img = useStore.getState().images[useStore.getState().activeIndex];
      if (!img) {
        toast.error('先导入一张图片再试');
        return;
      }
      set({ busy: 'current' });
      try {
        const bmp = await loadBitmap(img);
        if (!bmp) {
          toast.error('这张图打不开，无法识别');
          return;
        }
        const dets = await detector.detect(bmp, st.conf);
        set({
          numClasses: detector.numClasses,
          candidates: toCandidates(dets),
          candidatesImageId: img.id,
          layoutInfo: detector.layoutInfo,
          inputSizeFrom: detector.inputSizeFrom,
        });
        if (dets.length === 0) {
          toast.info('这张图没找到目标。可以先调低置信度，或看下方的模型信息确认输出布局是否识别正确。');
        }
      } catch (e) {
        toast.error(`识别失败：${messageOf(e)}`);
      } finally {
        set({ busy: 'none' });
      }
    },

    runBatch: async (scope = 'empty') => {
      const st = get();
      if (st.status !== 'ready' || st.busy !== 'none') return;
      const targets = useStore.getState().images.filter(
        // 'all' 会连已有标注的图一起跑；对已有框的图是「追加」，不会清掉旧框
        (i) => scope === 'all' || i.annotations.length === 0,
      );
      if (targets.length === 0) {
        toast.info(scope === 'all' ? '没有可处理的图片' : '所有图片都已有标注');
        return;
      }
      batchCancelled = false;
      set({
        busy: 'batch',
        batchDone: 0,
        batchTotal: targets.length,
        // 避免旧候选框和批量结果在同一张图上叠加
        candidates: [],
        candidatesImageId: null,
      });

      let boxCount = 0;
      let touched = 0;
      let failed = 0;
      let i = 0;
      for (const img of targets) {
        if (batchCancelled) break;
        i += 1;
        const bmp = await loadBitmap(img);
        if (!bmp) {
          // 解码失败要计数：否则用户分不清「模型没识别到」和「这张图打不开」
          failed += 1;
          set({ batchDone: i });
          continue;
        }
        try {
          const dets = await detector.detect(bmp, get().conf);
          if (dets.length > 0) {
            useStore
              .getState()
              .applyAiBoxes(img.id, toCandidates(dets), bmp.width, bmp.height);
            boxCount += dets.length;
            touched += 1;
          }
        } catch {
          // 单张失败不打断整批，但必须记账，最后如实告知
          failed += 1;
        }
        // 进度按「处理完」计，而不是提前 +1，否则显示会比实际快
        set({ batchDone: i });
        // 每 3 张让出一次控制权。
        // await 只让出微任务队列，不把控制权交回渲染任务，
        // 加上单线程 wasm 推理本就占着主线程，不显式让帧的话
        // 整个批次里浏览器一次都没 paint 过 —— 用户看到的是「点了之后界面冻住，结束后进度条直接跳到 100%」。
        if (i % 3 === 0) await new Promise((r) => setTimeout(r, 0));
      }

      set({
        busy: 'none',
        numClasses: detector.numClasses || get().numClasses,
        layoutInfo: detector.layoutInfo,
        inputSizeFrom: detector.inputSizeFrom,
      });
      if (batchCancelled) {
        toast.info(`已停止，共标注 ${touched} 张图`);
      } else if (touched > 0) {
        const suffix = failed > 0 ? `，另有 ${failed} 张推理失败（多为图片损坏或内存不足）` : '';
        toast.success(`一键标注完成：${touched} 张图、${boxCount} 个框（Ctrl+Z 可撤销）${suffix}`);
      } else if (failed > 0) {
        toast.warning(`${failed} 张图推理失败，没有成功标注任何一张（多为图片损坏或内存不足）`);
      } else {
        toast.info('未标注的图片里没找到目标');
      }
    },

    cancelBatch: () => {
      batchCancelled = true;
    },

    acceptAll: () => {
      const { candidates, candidatesImageId } = get();
      if (!candidatesImageId || candidates.length === 0) return;
      const img = useStore.getState().images.find((i) => i.id === candidatesImageId);
      if (!img) {
        set({ candidates: [], candidatesImageId: null });
        return;
      }
      // 图片尺寸正常都已解码登记；极端缺失时用候选框外包围盒兜底
      let w = img.width;
      let h = img.height;
      if (w <= 0 || h <= 0) {
        for (const c of candidates) {
          w = Math.max(w, c.rect.x + c.rect.width);
          h = Math.max(h, c.rect.y + c.rect.height);
        }
        w = Math.ceil(w);
        h = Math.ceil(h);
      }
      if (w <= 0 || h <= 0) return;
      useStore.getState().applyAiBoxes(candidatesImageId, candidates, w, h);
      set({ candidates: [], candidatesImageId: null });
      toast.success(`已添加 ${candidates.length} 个标注框`);
    },

    discardCandidates: () => {
      set({ candidates: [], candidatesImageId: null });
    },
  };
});
