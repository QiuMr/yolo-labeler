import { create } from 'zustand';
import { Detector } from '../ai/detector';
import { classNameFor } from '../ai/cocoNames';
import type { IRect } from '../types';
import { loadBitmap } from '../utils/imageCache';
import {
  loadClassNames,
  loadConf,
  loadSavedModel,
  saveClassNames,
  saveConf,
  saveModel,
} from '../utils/modelStore';
import { useStore } from './useStore';
import { toast } from './toastStore';

/** 默认置信度阈值。模型没记过自己的值时的起点 */
const DEFAULT_CONF = 0.25;

/** AI 候选框（还没被采纳，画布上显示为虚线） */
export interface AiCandidate {
  rect: IRect;
  score: number;
  className: string;
  /**
   * 模型输出的类别号。
   * 必须留着：用户在面板里改了某个类别的名字后，已经算好的候选框要按它重算名字
   * （只留 className 的话，改完名字旧候选框还挂着旧名，采纳下去就是错的标签）。
   */
  classIndex: number;
}

export type AiStatus = 'idle' | 'booting' | 'loading' | 'ready' | 'error';
export type AiBusy = 'none' | 'current' | 'batch';

/**
 * 一张图的待复核候选框，外加跑批量时的**真实解码尺寸**。
 *
 * 尺寸必须一起存：图片的 width/height 是懒测量的，批量时可能还是 0，
 * 而采纳时要靠它把候选框钳回图内。用候选框自己的外包围盒兜底会钳错
 * （目标没占满全图时会把框切小），所以宁可当时就记下来。
 */
export interface PendingBatch {
  cands: AiCandidate[];
  width: number;
  height: number;
}

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
  /**
   * 用户在面板的「类别名」里自己改过的名字（按类别号对齐，空串=没改）。
   * 跟着模型一起存在 localStorage（见 modelStore.saveClassNames）——
   * 不存的话，刷新后再跑一次批量标注会拿回「类别 1」并**新建一个同名的另一个标签**，
   * 同一类东西在左侧裂成两行，用户很难自己想明白。
   */
  classNamesOverride: string[];
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

  /**
   * 批量跑出来的候选框，**按图存**（照样子找的批量链路用）。
   *
   * 为什么不直接落成标注：批量最大的风险是静默出错 —— 100 张图跑完只看到一句「完成」，
   * 而实测同一批图 F1 能从 0.000 到 0.750，「一个都没找到」和「找到一堆误报」
   * 在「完成」里长得一模一样。先攒成候选框，用户看过汇总（含异常标记）再一键采纳，
   * 既保持了「一键」的速度，也保住了「发现不对能整车退掉」的机会。
   *
   * `candidates` 只是**当前图那一份的投影**：切到某张图时从这里取，
   * 改的时候两边同进同退（见 writeCandidates）—— 否则会出现
   * 「画布上删掉了，切回来又冒出来」。
   */
  pending: Record<string, PendingBatch>;
  /** 还有候选框待复核的张数（角标用） */
  pendingImages: number;
  /** 待复核候选框总数 */
  pendingBoxes: number;

  /** 打开 AI 面板时调用：引导引擎 + 自动找模型（内置 → 上次选过的） */
  init: () => Promise<void>;
  loadModelFile: (file: File) => Promise<void>;
  setConf: (v: number) => void;
  /** 改某个类别的名字（空串 = 恢复默认名）；会同步刷新已生成的候选框 */
  setClassName: (index: number, name: string) => void;
  /** 去掉一个候选框（在画布上点它）。不必为了一个错框把整批都放弃 */
  removeCandidate: (index: number) => void;
  /**
   * 由**别的链路**产出的候选框，交给画布走同一套「确认 / 剔除」流程。
   *
   * 为什么放在这里而不是各链路自己存一份：候选框在画布上只有一处渲染逻辑
   * （`CanvasStage` 读 `aiStore.candidates`，底部状态条给「全部采纳 / 放弃」）。
   * 「照样子找」（vpStore）产出的框语义完全一样 —— 都是「还没被采纳的建议」，
   * 分两份状态的话画布要同时读两个来源，两边还会互相盖住。
   */
  setCandidates: (cands: readonly AiCandidate[], imageId: string) => void;
  /** 批量结果写入 pending（一次一张图） */
  putPending: (imageId: string, cands: readonly AiCandidate[], width: number, height: number) => void;
  /** 清空整批待复核结果（重新跑批量之前） */
  clearPending: () => void;
  /** 切图时把那张图的候选框装载到「当前」上。批量跑完必须调一次，否则用户翻图看到的是空的 */
  syncActive: () => void;
  /** 一键采纳整批：合成一条撤销记录，Ctrl+Z 一次退掉 */
  acceptPendingAll: () => void;
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

/**
 * 写「当前候选框」，并把同一份改动同步进 pending。
 *
 * pending 是批量的存底，`candidates` 只是当前图那一份的投影。
 * 两边必须同进同退，否则会出现「画布上点掉一个框，切到别的图再切回来，它又冒出来了」——
 * 这种 bug 用户只会理解成「工具自己有主意」。
 */
function writeCandidates(
  set: (p: Partial<AiState>) => void,
  get: () => AiState,
  imageId: string | null,
  cands: readonly AiCandidate[],
): void {
  const pending: Record<string, PendingBatch> = { ...get().pending };
  if (imageId) {
    if (cands.length > 0) pending[imageId] = { ...pending[imageId], cands: cands.slice() };
    else delete pending[imageId];
  }
  let pendingImages = 0;
  let pendingBoxes = 0;
  for (const k in pending) {
    if (pending[k].cands.length > 0) {
      pendingImages += 1;
      pendingBoxes += pending[k].cands.length;
    }
  }
  set({
    pending,
    pendingImages,
    pendingBoxes,
    candidates: cands.slice(),
    // 空候选不要留下一个指向某张图的空状态条
    candidatesImageId: cands.length > 0 ? imageId : null,
  });
}

function toCandidates(
  dets: Array<{ rect: IRect; score: number; classIndex: number }>,
  overrides: readonly string[],
): AiCandidate[] {
  return dets.map((d) => ({
    rect: d.rect,
    score: d.score,
    classIndex: d.classIndex,
    // 优先用用户改过的名字，其次模型自带的类名，再回退到 COCO 表 / 类别 N
    className: classNameFor(d.classIndex, detector.numClasses, detector.modelNames, overrides),
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
      // 换模型了：上一份自定义类名不再对应，改成读**这个模型自己**那份（没有则为空）
      classNamesOverride: loadClassNames(name),
      // 置信度阈值同理：不同模型该用的阈值差很多，跟着模型记住才不用每次重调
      conf: loadConf(name) ?? DEFAULT_CONF,
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
    classNamesOverride: [],
    inputSizeFrom: '',
    layoutInfo: '',
    warmupError: '',
    conf: DEFAULT_CONF,
    busy: 'none',
    batchDone: 0,
    batchTotal: 0,
    candidates: [],
    candidatesImageId: null,
    pending: {},
    pendingImages: 0,
    pendingBoxes: 0,

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

    setConf: (v) => {
      set({ conf: v });
      // 跟着模型记住：调一次就够，下次打开还是这个值。
      // 不记的话「刷新后回 0.25」会让同一批图的标注结果前后不一致，用户很难自己反应过来。
      const name = get().modelName;
      if (name) saveConf(name, v);
    },

    setCandidates: (cands, imageId) => {
      writeCandidates(set, get, imageId, cands);
    },

    putPending: (imageId, cands, width, height) => {
      if (!imageId) return;
      // 空结果也写：批量汇总要能数出「这张图一个都没找到」
      const pending: Record<string, PendingBatch> = {
        ...get().pending,
        [imageId]: { cands: cands.slice(), width, height },
      };
      let pendingImages = 0;
      let pendingBoxes = 0;
      for (const k in pending) {
        if (pending[k].cands.length > 0) {
          pendingImages += 1;
          pendingBoxes += pending[k].cands.length;
        }
      }
      set({ pending, pendingImages, pendingBoxes });
    },

    clearPending: () => {
      set({ pending: {}, pendingImages: 0, pendingBoxes: 0 });
    },

    syncActive: () => {
      const st = get();
      const images = useStore.getState().images;
      const img = images[useStore.getState().activeIndex];
      const id = img?.id ?? null;

      // 顺手清掉已经不在项目里的图。不清的话：批量跑完用户删了几张图，
      // 角标会一直显示「还有 100 张待确认」，而项目里只剩 95 张 ——
      // 用户翻遍整个项目也找不到那 5 张，只会以为工具坏了。
      let pending = st.pending;
      const alive = new Set(images.map((i) => i.id));
      for (const k in pending) {
        if (!alive.has(k)) {
          if (pending === st.pending) pending = { ...pending };
          delete pending[k];
        }
      }
      if (pending !== st.pending) {
        let n = 0;
        let b = 0;
        for (const k in pending) {
          if (pending[k].cands.length > 0) {
            n += 1;
            b += pending[k].cands.length;
          }
        }
        set({ pending, pendingImages: n, pendingBoxes: b });
      }

      // 已经在显示这张图的候选框了：pending 的任何改动都是经 writeCandidates 同步过来的，
      // 再写一遍只会白跑一次 set。
      if (st.candidatesImageId === id) return;
      writeCandidates(set, get, id, id ? (pending[id]?.cands ?? []) : []);
    },

    acceptPendingAll: () => {
      const st = get();
      const entries = Object.entries(st.pending).filter(([, v]) => v.cands.length > 0);
      if (entries.length === 0) return;

      // 整批合成一条撤销记录：Ctrl+Z 一次退掉整批。
      // 这也是「先攒候选框再落」相对「边跑边落」唯一多出来的代价 —— 但很值：
      // 用户在看到汇总之前不会知道这批结果能不能要。
      useStore.getState().beginAiBatch();
      let added = 0;
      let skipped = 0;
      let touched = 0;
      let failed = 0;
      for (const [imageId, p] of entries) {
        const img = useStore.getState().images.find((i) => i.id === imageId);
        if (!img) {
          failed += 1;
          continue;
        }
        try {
          const res = useStore
            .getState()
            .applyAiBoxes(imageId, p.cands, p.width || img.width, p.height || img.height);
          added += res.added;
          skipped += res.skipped;
          if (res.added > 0) touched += 1;
        } catch {
          // 单张失败不打断整批，但必须记账
          failed += 1;
        }
      }
      useStore.getState().endAiBatch();

      set({ pending: {}, pendingImages: 0, pendingBoxes: 0, candidates: [], candidatesImageId: null });
      const dupSuffix = skipped > 0 ? `，跳过 ${skipped} 个与已有框重复的` : '';
      if (touched > 0) {
        const failSuffix = failed > 0 ? `，另有 ${failed} 张没能写入` : '';
        toast.success(`已采纳 ${touched} 张图、${added} 个框${dupSuffix}（Ctrl+Z 一次撤销整批）${failSuffix}`);
      } else if (skipped > 0) {
        toast.info(`这些框都和已有的重复，没有新增${dupSuffix}`);
      } else {
        toast.info('没有可采纳的候选框');
      }
    },

    removeCandidate: (index) => {
      const { candidates, candidatesImageId } = get();
      if (index < 0 || index >= candidates.length) return;
      // 经 writeCandidates 回写 pending：只改 candidates 的话，
      // 切图再切回来这个被点掉的框会复活
      writeCandidates(set, get, candidatesImageId, candidates.filter((_, i) => i !== index));
    },

    setClassName: (index, name) => {
      const num = get().numClasses;
      if (index < 0 || index >= num) return;
      const overrides = get().classNamesOverride.slice();
      while (overrides.length < num) overrides.push('');
      overrides[index] = name;
      const pending: Record<string, PendingBatch> = {};
      for (const k in get().pending) {
        pending[k] = {
          ...get().pending[k],
          // 已经算好的候选框要跟着改名，否则用户改完名字、采纳下去的还是旧名
          cands: get().pending[k].cands.map((c) => ({
            ...c,
            className: classNameFor(c.classIndex, num, detector.modelNames, overrides),
          })),
        };
      }
      set({
        classNamesOverride: overrides,
        pending,
        candidates: get().candidates.map((c) => ({
          ...c,
          className: classNameFor(c.classIndex, num, detector.modelNames, overrides),
        })),
      });
      // 跟着模型一起记住：下次打开、或换回来这个模型时名字还在
      saveClassNames(get().modelName, overrides);
    },

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
        writeCandidates(set, get, img.id, toCandidates(dets, get().classNamesOverride));
        set({
          numClasses: detector.numClasses,
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
        // 'all' 会连已有标注的图一起跑；对已有框的图是「补上缺的」，同类且高度重叠的会被跳过
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
        // 避免旧候选框和批量结果在同一张图上叠加。
        // pending 也一起清：这条链路是「边跑边落」，不留待复核的存底，
        // 否则上一批没采纳的候选会一直挂在计数上。
        candidates: [],
        candidatesImageId: null,
        pending: {},
        pendingImages: 0,
        pendingBoxes: 0,
      });

      let boxCount = 0;
      let touched = 0;
      let failed = 0;
      let dupSkipped = 0;
      let i = 0;
      // 整批合成一条撤销记录：Ctrl+Z 一次退掉整批，也不会因为张数超过历史栈上限而退不回来
      useStore.getState().beginAiBatch();
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
            const res = useStore
              .getState()
              .applyAiBoxes(img.id, toCandidates(dets, get().classNamesOverride), bmp.width, bmp.height);
            boxCount += res.added;
            dupSkipped += res.skipped;
            // 只按「真的落下去几个」计数：整张图全是重复框时它并没有被改动
            if (res.added > 0) touched += 1;
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
      // 中途停止也要收口：已经落下的框必须留下，且合成同一条历史
      useStore.getState().endAiBatch();

      set({
        busy: 'none',
        numClasses: detector.numClasses || get().numClasses,
        layoutInfo: detector.layoutInfo,
        inputSizeFrom: detector.inputSizeFrom,
      });
      const dupSuffix = dupSkipped > 0 ? `，跳过 ${dupSkipped} 个与已有框重复的` : '';
      if (batchCancelled) {
        toast.info(`已停止，本次共标注 ${touched} 张图${dupSuffix}`);
      } else if (touched > 0) {
        const suffix = failed > 0 ? `，另有 ${failed} 张推理失败（多为图片损坏或内存不足）` : '';
        toast.success(
          `一键标注完成：${touched} 张图、${boxCount} 个框${dupSuffix}（Ctrl+Z 一次撤销整批）${suffix}`,
        );
      } else if (dupSkipped > 0) {
        toast.info(`这些图上找到的目标都已经有框了，又跳过 ${dupSkipped} 个重复的，没有新增`);
      } else if (failed > 0) {
        toast.warning(`${failed} 张图推理失败，没有成功标注任何一张（多为图片损坏或内存不足）`);
      } else {
        toast.info(scope === 'all' ? '这批图里没找到目标' : '未标注的图片里没找到目标');
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
        // 图被删了：候选框跟着作废，pending 里那一份也要清，不然计数下不去
        writeCandidates(set, get, candidatesImageId, []);
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
      const res = useStore.getState().applyAiBoxes(candidatesImageId, candidates, w, h);
      // 采纳掉的要从 pending 里移除，否则「待复核」计数一直不减
      writeCandidates(set, get, candidatesImageId, []);
      const dup = res.skipped > 0 ? `，跳过 ${res.skipped} 个重复的` : '';
      if (res.added > 0) toast.success(`已添加 ${res.added} 个标注框${dup}`);
      else if (res.skipped > 0) toast.info(`这些框都和已有的重复，没有新增${dup}`);
    },

    discardCandidates: () => {
      const { candidatesImageId } = get();
      writeCandidates(set, get, candidatesImageId, []);
    },
  };
});

/**
 * 切图时自动装载那张图的待复核候选框。
 *
 * 没有这一步，批量跑完用户翻到第 50 张图，画布是空的 —— 他会以为没跑，
 * 于是把整批结果当垃圾丢掉。装在 store 模块里而不是某个组件的 effect 里：
 * 换图的入口有好几个（侧栏、键盘方向键、导出面板里的跳转、批量汇总里的跳转），
 * 逐个挂 effect 必然漏。
 */
useStore.subscribe((s, prev) => {
  const a = s.images[s.activeIndex]?.id ?? null;
  const b = prev.images[prev.activeIndex]?.id ?? null;
  if (a !== b) useAiStore.getState().syncActive();
});
