import { create } from 'zustand';
import { Segmenter, type SegmentPoint } from '../ai/segmenter';
import { loadBitmap } from '../utils/imageCache';
import { useStore } from './useStore';
import { toast } from './toastStore';

/**
 * 点击辅助标注（MobileSAM 交互式分割）的状态层。
 *
 * 刻意与 aiStore（YOLO 检测）分开：
 *  - YOLO 用的是**用户自己选的**模型文件（存 IndexedDB），跑完给「候选框」再采纳；
 *  - 分割用的是**随应用内置**的 encoder+decoder（public/models/），
 *    交互上是「点一下直接出一个框」，没有候选框这一步。
 * 两者在模型来源、交互模型、生命周期上都不一样，塞进一个 store 只会互相干扰。
 */

/** 内置模型路径。用 BASE_URL 拼接，保证部署到子目录时也能取到 */
const MODEL_BASE = `${import.meta.env.BASE_URL}models/`;
const ENCODER_URL = `${MODEL_BASE}mobilesam_encoder.onnx`;
const DECODER_URL = `${MODEL_BASE}mobilesam_decoder.onnx`;

/** 两个模型加起来约 42MB，首次访问要下这些 —— 文案里如实告诉用户 */
export const SEG_MODEL_MB = 42;

export type SegStatus = 'idle' | 'loading' | 'ready' | 'error';

interface SegState {
  status: SegStatus;
  error: string;
  /** 下载进度 0~100；-1 表示拿不到总长度（无法计算百分比） */
  progress: number;
  /** 是否处于「点击标注」模式（画布上单击即取点） */
  active: boolean;
  /** 正在编码还是正在解码（决定界面显示什么提示） */
  busy: '' | 'encode' | 'decode';
  /** 当前这张图是否已编码（true = 点击秒出） */
  encoded: boolean;
  ep: '' | 'WebGPU' | 'WASM';
  /**
   * 刚生成的框还没有类别时的 id —— 画布会消费它并弹「这是什么」菜单。
   * 用一次性标志而不是让 store 直接操作画布 UI：store 不该知道菜单怎么弹。
   */
  pendingLabelFor: string;

  /** 打开「点击标注」：按需下载模型 + 预编码当前图 */
  open: () => Promise<void>;
  close: () => void;
  /** 画布上点一下（原图坐标系） */
  click: (x: number, y: number, positive: boolean) => Promise<void>;
  /** 放弃当前累积的点 */
  resetPoints: () => void;
  /** 画布消费完「待定类别」后清空 */
  clearPendingLabel: () => void;
}

const segmenter = new Segmenter();

/**
 * 累积的点击点。多点用于修正：点偏了再加一个正点，
 * 想排除多出来的区域就加一个负点（右键）。
 */
let points: SegmentPoint[] = [];
/** 这些点属于哪张图。换图后旧点必须作废，否则会把上一张的点喂给这一张 */
let pointsImageId = '';

function messageOf(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * 两个 onnx 全部下完之后，进度恰好停在这个值（encoder 占 0~62、decoder 占 62~94），
 * 之后就是 `loadModels()` 建 ORT 会话的阶段。界面靠它区分「还在下载」和「在建会话」——
 * 因为建会话（尤其 WebGPU 编译着色器）在本地文件下也要好几秒，进度条会停在 94% 不动。
 * 改上面那两处权重时记得同步改这里。
 */
export const SEG_PROGRESS_DOWNLOAD_DONE = 94;

/**
 * 带进度的下载。42MB 在慢网络下要几十秒，
 * 不给进度条用户会以为卡死了——这正是本项目一贯反对的「静默」。
 *
 * `stallMs`：**两个分片之间**超过这么久没有任何字节就判定卡死并中止。
 * 为什么用「停顿」而不是「总时长上限」：42MB 在慢网络上要几分钟是正常的，
 * 设总时长上限会把正常用户一起误伤；而**长时间零字节**在任何网络下都只可能是真卡住了。
 * 没有这个兜底，一次卡死会让 `status` 永远停在 'loading'，而 `open()` 在 loading 时
 * 会**静默吞掉所有后续点击** —— 按钮看起来就是彻底坏了，只能刷新页面。
 */
async function fetchWithProgress(
  url: string,
  onPct: (pct: number) => void,
  stallMs = 20000,
): Promise<ArrayBuffer> {
  const ac = new AbortController();
  let stallTimer: number | undefined;
  const armStall = () => {
    if (stallTimer !== undefined) clearTimeout(stallTimer);
    stallTimer = window.setTimeout(() => ac.abort(), stallMs);
  };
  const disarmStall = () => {
    if (stallTimer !== undefined) {
      clearTimeout(stallTimer);
      stallTimer = undefined;
    }
  };

  try {
    armStall();
    const res = await fetch(url, { signal: ac.signal });
    if (!res.ok) {
      throw new Error(
        `模型文件下载失败（HTTP ${res.status}）。请确认 ${url} 存在 —— ` +
          '部署时 public/models/ 下的两个 onnx 必须一并上传。',
      );
    }
    const total = Number(res.headers.get('Content-Length') ?? 0);
    if (!res.body || !total) {
      onPct(-1);
      return await res.arrayBuffer();
    }
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    for (;;) {
      armStall();
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onPct(Math.min(100, Math.round((received / total) * 100)));
    }
    const merged = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) {
      merged.set(c, off);
      off += c.length;
    }
    return merged.buffer;
  } catch (e) {
    if (ac.signal.aborted) {
      throw new Error(
        `模型下载中断：超过 ${Math.round(stallMs / 1000)} 秒没有收到任何数据，` +
          '请检查网络后重试。',
      );
    }
    throw e;
  } finally {
    disarmStall();
  }
}

export const useSegStore = create<SegState>((set, get) => ({
  status: 'idle',
  error: '',
  progress: 0,
  active: false,
  busy: '',
  encoded: false,
  ep: '',
  pendingLabelFor: '',

  open: async () => {
    const st = get();
    if (st.active) return;

    if (st.status !== 'ready') {
      if (st.status === 'loading') return;
      set({ status: 'loading', error: '', progress: 0 });
      try {
        await segmenter.boot();
        // 先下 encoder（26MB，占大头），再下 decoder，进度能分开显示
        const encBytes = await fetchWithProgress(ENCODER_URL, (pct) =>
          set({ progress: Math.round(pct * 0.62) }),
        );
        const decBytes = await fetchWithProgress(DECODER_URL, (pct) =>
          set({ progress: 62 + Math.round(pct * 0.32) }),
        );
        await segmenter.loadModels(new Uint8Array(encBytes), new Uint8Array(decBytes));
        set({ status: 'ready', progress: 100, ep: segmenter.ep });
      } catch (e) {
        set({ status: 'error', error: messageOf(e), progress: 0 });
        toast.error(messageOf(e));
        return;
      }
    }

    set({ active: true });

    // 进入模式就先把当前图编码好，用户点第一下时不用等编码
    const img = useStore.getState().images[useStore.getState().activeIndex];
    if (img) {
      set({ busy: 'encode', encoded: false });
      try {
        const bmp = await loadBitmap(img);
        if (bmp) await segmenter.encode(img.id, bmp, img.width, img.height);
        set({ encoded: segmenter.hasEmbedding(img.id) });
      } catch {
        /* 编码失败不阻断：真正点击时会再报一次，这里静默即可避免开模式就弹错 */
      } finally {
        set({ busy: '' });
      }
    }
  },

  close: () => {
    points = [];
    pointsImageId = '';
    set({ active: false, busy: '', encoded: false });
  },

  resetPoints: () => {
    points = [];
    set({ encoded: segmenter.hasEmbedding(pointsImageId) });
  },

  clearPendingLabel: () => set({ pendingLabelFor: '' }),

  click: async (x, y, positive) => {
    const st = get();
    if (!st.active || st.busy) return;
    const img = useStore.getState().images[useStore.getState().activeIndex];
    if (!img) return;
    if (img.width <= 0 || img.height <= 0) {
      toast.error('这张图的尺寸还没就绪，稍等一下再点');
      return;
    }

    // 换图后旧点作废：否则会把上一张图的点混进这一张
    if (pointsImageId !== img.id) {
      points = [];
      pointsImageId = img.id;
      set({ encoded: segmenter.hasEmbedding(img.id) });
    }

    // 没编码过就先编码（首次点击会慢一次，之后都是秒出）
    if (!segmenter.hasEmbedding(img.id)) {
      set({ busy: 'encode' });
      try {
        const bmp = await loadBitmap(img);
        if (!bmp) throw new Error('这张图打不开，无法分割');
        await segmenter.encode(img.id, bmp, img.width, img.height);
        set({ encoded: true });
      } catch (e) {
        toast.error(`编码失败：${messageOf(e)}`);
        set({ busy: '' });
        return;
      }
    }

    const next = [...points, { x, y, positive }];
    set({ busy: 'decode' });
    try {
      const res = await segmenter.segment(img.id, img.width, img.height, next);
      // addBox 内部会 clamp 到图片范围、挡掉 <2px 的退化框，并压一条撤销历史
      const id = useStore.getState().addBox(res.rect);
      if (!id) {
        // 按 addBox 的既有契约，返回 null 说明框退化到不可用（多为点在边缘）
        toast.warning('生成的范围太小，换个位置点在物体内部');
        set({ busy: '' });
        return;
      }
      points = [];
      pointsImageId = img.id;

      // 没有类别时**不静默留一个「未分类」框** —— 与画框流程保持一致，当场问它是什么。
      // （用户反馈的「点一下没有选类别」就出在这里：labels 为空时 addBox 给的是空 labelId）
      const cur = useStore.getState();
      const created = cur.images[cur.activeIndex]?.annotations.find((a) => a.id === id);
      if (!created || !created.labelId) {
        set({ pendingLabelFor: id });
        return;
      }

      // 固定 id：连续点多个物体时只更新同一条提示，不会堆满屏幕
      const pct = Math.round((res.score || 0) * 100);
      toast.success(pct > 0 ? `已生成标注框（模型置信度 ${pct}%）` : '已生成标注框', 'seg-box');
    } catch (e) {
      // 失败时**保留已累积的点**：用户再点一下就能修正（负点可排除多余区域）
      points = next;
      toast.error(messageOf(e));
    } finally {
      set({ busy: '' });
    }
  },
}));
