import type { IRect } from '../types';
import {
  bootRuntime,
  createSession,
  type OrtGlobal,
  type OrtSession,
  type OrtTensorOut,
} from './ortRuntime';

/**
 * YOLOE 视觉提示检测 —— 界面上叫「照样子找」。
 *
 * 与 detector.ts（用户自备的 YOLO 检测模型）的区别：
 *  detector 是「训练好的固定类别」，这里给的是**参考图 + 示例框**，
 *  模型据此找出「长得像那个东西」的目标，**不需要任何训练过程**。
 *
 * 模型是**单文件双用途**（见 `.tmp/yoloe-onnx/export_merged.py`）：
 *   · 编码用途：喂 (参考图, 参考框掩码, 全零提示) → 读第 2 个输出 `pe` 拿提示向量
 *   · 检测用途：喂 (目标图, 全零掩码,    提示向量) → 读第 1 个输出拿框
 *  两个用途共用一个 backbone，所以只有一个文件、不需要开关分支。
 *
 * 为什么检测时掩码可以传全零：导出时把 `get_cls_pe` 打成了 `cls_pe = tpe`，
 * `vpe` 只参与产出 `pe`、不参与类别嵌入。已用 Python 对拍验证 ——
 * 全零掩码与参考掩码给出的框**逐位相同**（`.tmp/yoloe-onnx/eval_merged.py`）。
 * 这一点很值：JS 侧因此不必保存参考图的掩码，也不必纠结多张参考图的掩码怎么合并。
 *
 * 输出布局：`[1, 300, 38]`，每行 `x1,y1,x2,y2, conf, cls, …32 个 mask 系数`。
 * **不能交给 detector.ts 的 parseDetections 解析** —— 38 列会被它按「4+34 类」解读，
 * 于是 300 行全被当成候选、类号也是错的。这里必须自己解析，且 mask 系数直接丢弃
 * （本工具只做矩形框标注，不做分割）。
 */

/** 模型固定输入边长。导出时 imgsz=1280；换模型必须同步改这里 */
export const VP_SIZE = 1280;
/** 提示掩码网格边长：1280 / 8 = 160 */
export const VP_MASK = VP_SIZE / 8;
/** 提示向量维度 */
export const VP_DIM = 512;
/** 带 mask 分支的导出（本工具用的那个）每行列数：4 + 1 + 1 + 32 */
const SEG_COLS = 38;
/** 纯检测版导出（无 mask 分支）每行列数 */
const DET_COLS = 6;
/** 输出行数上限（max_det 量级）。远超它说明拿到的不是端到端输出 */
const MAX_ROWS = 2000;
/**
 * NMS 的 IoU 阈值。
 * 取 0.5：实测 F1 0.746 就是这组参数（不做 NMS 会掉到 0.485，同一物体出好几个框）。
 */
export const VP_NMS_IOU = 0.5;

/** letterbox 参数（与 ultralytics `LetterBox(auto=False)` 逐值对齐） */
export interface VpLetterbox {
  /** 缩放系数 */
  gain: number;
  /** 缩放后的宽高（像素，已取整） */
  w: number;
  h: number;
  /** 居中补边量（像素，已取整） */
  left: number;
  top: number;
}

/**
 * 计算 letterbox 参数。
 *
 * **必须按 ultralytics 的取整方式来**：它算完 `dw/dh` 后会 `round(dw - 0.1)`，
 * 不 round 的话坐标会系统性偏 1px —— 框本身看不出来，但导出成数据集后
 * 每个框都差一点，属于那种「事后才发现在静默污染标注」的错。
 */
export function letterboxParams(imgW: number, imgH: number): VpLetterbox {
  const gain = Math.min(VP_SIZE / imgW, VP_SIZE / imgH);
  const w = Math.round(imgW * gain);
  const h = Math.round(imgH * gain);
  return {
    gain,
    w,
    h,
    left: Math.round((VP_SIZE - w) / 2 - 0.1),
    top: Math.round((VP_SIZE - h) / 2 - 0.1),
  };
}

/** 一个原图坐标系的检测结果 */
export interface VpDetection {
  /** 原图像素坐标 */
  rect: IRect;
  score: number;
}

function toFloat32(data: Float32Array | BigInt64Array): Float32Array {
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i += 1) out[i] = Number(data[i]);
  return out;
}

/**
 * 把提示向量合并成一个。
 *
 * 导出的模型里 `head.forward_head` 的 `self.nc = x[-1].shape[1]` 在 trace 时被写死，
 * 多提示**不会**真的多出类别，反而会让底层 reshape 错位（实测：喂 3 个提示与喂 1 个
 * 的输出形状、最高 conf 逐位相同，说明多出来的提示被静默丢掉了）。
 * 所以多张参考图改为**在向量层取平均**再重新 L2 归一化 —— 语义上就是「这些例子的平均长相」。
 * 实测 3 张 / 5 张与单张 F1 持平（0.721 / 0.753 vs 0.746），是安全的合并方式。
 */
export function mergePrompts(prompts: readonly Float32Array[]): Float32Array {
  if (prompts.length === 0) throw new Error('至少需要一张参考图');
  if (prompts.length === 1) return prompts[0];
  const acc = new Float32Array(VP_DIM);
  for (const p of prompts) {
    if (p.length !== VP_DIM) throw new Error(`提示向量维度异常：${p.length}（应为 ${VP_DIM}）`);
    for (let i = 0; i < VP_DIM; i += 1) acc[i] += p[i];
  }
  let norm = 0;
  for (let i = 0; i < VP_DIM; i += 1) {
    acc[i] /= prompts.length;
    norm += acc[i] * acc[i];
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < VP_DIM; i += 1) acc[i] /= norm;
  return acc;
}

export class VisualPromptDetector {
  private ort: OrtGlobal | null = null;
  private session: OrtSession | null = null;
  private warmed = false;
  /** 复用的离屏画布。1280² 的画布每次新建会造成明显的 GC 抖动 */
  private ctx: CanvasRenderingContext2D | null = null;

  ep: 'WebGPU' | 'WASM' = 'WebGPU';
  /** 预热报错（空串 = 正常）。非空基本等于「这个模型现在跑不起来」 */
  warmupError = '';

  get ready(): boolean {
    return this.session !== null;
  }

  /** 引导推理引擎（与 detector / segmenter 共用同一套本地优先、CDN 兜底的运行时） */
  async boot(): Promise<void> {
    if (this.ort) return;
    this.ort = await bootRuntime();
  }

  async loadModel(bytes: Uint8Array): Promise<void> {
    const ort = this.ort;
    if (!ort) throw new Error('推理引擎未就绪');
    this.session = null;
    this.warmed = false;
    this.warmupError = '';
    const created = await createSession(ort, bytes, (msg, err) =>
      console.warn(`[visualPrompt] ${msg}`, err),
    );
    this.session = created.session;
    this.ep = created.ep;
    // CDN 兜底后运行时对象会换，必须把引用更新过去
    this.ort = created.ort;
    await this.warmup();
  }

  /**
   * 预热：跑一遍零输入，把 WebGPU 的一次性编译开销提前消化掉。
   * 不预热的话，用户点「先试标一张」要等好几秒才看到第一张结果，像是卡住了。
   */
  private async warmup(): Promise<void> {
    if (this.warmed || !this.session) return;
    this.warmed = true;
    try {
      await this.runRaw(
        new Float32Array(3 * VP_SIZE * VP_SIZE),
        new Float32Array(VP_MASK * VP_MASK),
        new Float32Array(VP_DIM),
      );
      this.warmupError = '';
    } catch (e) {
      // 记下来给界面展示。静默吞掉的话，用户只会看到「什么都没找到」
      this.warmupError = e instanceof Error ? e.message : String(e);
    }
  }

  private runRaw(
    image: Float32Array,
    mask: Float32Array,
    prompt: Float32Array,
  ): Promise<Record<string, OrtTensorOut>> {
    const ort = this.ort;
    const sess = this.session;
    if (!ort || !sess) throw new Error('模型未加载');
    return sess.run({
      images: new ort.Tensor('float32', image, [1, 3, VP_SIZE, VP_SIZE]),
      vpe: new ort.Tensor('float32', mask, [1, 1, VP_MASK, VP_MASK]),
      tpe: new ort.Tensor('float32', prompt, [1, 1, VP_DIM]),
    });
  }

  private getCtx(): CanvasRenderingContext2D {
    if (!this.ctx) {
      const c = document.createElement('canvas');
      // 尺寸固定，只在这里赋一次 —— 给 canvas.width 赋**同一个值**也会清空画布并
      // 重置上下文状态，每次推理都赋一遍等于白做一次重置（detector.ts 里踩过这个坑）
      c.width = VP_SIZE;
      c.height = VP_SIZE;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('当前浏览器不支持 Canvas 2D');
      this.ctx = ctx;
    }
    return this.ctx;
  }

  /** letterbox + 像素读取 + HWC→CHW 归一化 */
  private buildImage(bitmap: ImageBitmap, lb: VpLetterbox): Float32Array {
    const ctx = this.getCtx();
    ctx.fillStyle = '#727272'; // RGB(114)，Ultralytics 标准补边色
    ctx.fillRect(0, 0, VP_SIZE, VP_SIZE);
    ctx.drawImage(bitmap, lb.left, lb.top, lb.w, lb.h);

    const u32 = new Uint32Array(ctx.getImageData(0, 0, VP_SIZE, VP_SIZE).data.buffer);
    const plane = VP_SIZE * VP_SIZE;
    const data = new Float32Array(3 * plane);
    const inv = 1 / 255;
    for (let i = 0; i < plane; i += 1) {
      const p = u32[i]; // 0xAABBGGRR（小端）
      data[i] = (p & 255) * inv;
      data[plane + i] = ((p >>> 8) & 255) * inv;
      data[2 * plane + i] = ((p >>> 16) & 255) * inv;
    }
    return data;
  }

  /**
   * 把提示框画成 160×160 的掩码（与官方 `LoadVisualPrompt` 同一套网格规则：
   * 框在 letterbox 画布上的坐标除以 8、向上取整，落在哪个格子里就点亮哪个格子）。
   */
  private buildMask(boxes: readonly IRect[], lb: VpLetterbox): Float32Array {
    const m = new Float32Array(VP_MASK * VP_MASK);
    const clamp = (v: number) => Math.max(0, Math.min(VP_MASK, v));
    for (const b of boxes) {
      const ya = clamp(Math.ceil((b.y * lb.gain + lb.top) / 8));
      const yb = clamp(Math.ceil(((b.y + b.height) * lb.gain + lb.top) / 8));
      const xa = clamp(Math.ceil((b.x * lb.gain + lb.left) / 8));
      const xb = clamp(Math.ceil(((b.x + b.width) * lb.gain + lb.left) / 8));
      for (let y = ya; y < yb; y += 1) {
        const row = y * VP_MASK;
        for (let x = xa; x < xb; x += 1) m[row + x] = 1;
      }
    }
    return m;
  }

  /** 从输出里取提示向量（第 2 个输出） */
  private pickPrompt(out: Record<string, OrtTensorOut>): Float32Array {
    const named = out.pe;
    if (named && named.data instanceof Float32Array) return named.data;
    const names = this.session?.outputNames ?? [];
    for (const n of names.slice(1)) {
      const t = out[n];
      if (t && t.data instanceof Float32Array) return t.data;
    }
    throw new Error('模型没有返回提示向量（第 2 个输出缺失），请确认用的是带 visual prompt 的 YOLOE 导出。');
  }

  /** 从输出里取检测结果张量（第 1 个输出） */
  private pickBoxes(out: Record<string, OrtTensorOut>): OrtTensorOut {
    const named = out.output0;
    if (named) return named;
    const names = this.session?.outputNames ?? [];
    if (names[0] && out[names[0]]) return out[names[0]];
    const first = Object.values(out)[0];
    if (!first) throw new Error('模型没有输出张量');
    return first;
  }

  /**
   * 编码提示：参考图 + 该图上的框 → 一个 512 维、已 L2 归一化的提示向量。
   * 框为空时抛错 —— 没有提示的检测必然什么也找不出来，早点说清楚比让用户等一轮更好。
   */
  async encodePrompt(bitmap: ImageBitmap, boxes: readonly IRect[]): Promise<Float32Array> {
    if (!this.session) throw new Error('模型未加载');
    if (boxes.length === 0) throw new Error('这张参考图上还没有框，先画一个再试');
    const lb = letterboxParams(bitmap.width, bitmap.height);
    const out = await this.runRaw(
      this.buildImage(bitmap, lb),
      this.buildMask(boxes, lb),
      new Float32Array(VP_DIM),
    );
    const pe = this.pickPrompt(out);
    if (pe.length !== VP_DIM) {
      throw new Error(`提示向量维度异常：${pe.length}（应为 ${VP_DIM}）`);
    }
    return new Float32Array(pe);
  }

  /**
   * 用提示向量在一张图上找目标，返回**原图坐标**的框（按置信度降序，已做 NMS）。
   */
  async detect(
    bitmap: ImageBitmap,
    prompt: Float32Array,
    conf: number,
  ): Promise<VpDetection[]> {
    if (!this.session) throw new Error('模型未加载');
    if (prompt.length !== VP_DIM) throw new Error(`提示向量维度异常：${prompt.length}`);
    const lb = letterboxParams(bitmap.width, bitmap.height);
    const out = await this.runRaw(
      this.buildImage(bitmap, lb),
      // 全零掩码即可：vpe 不参与类别嵌入（见文件头说明）
      new Float32Array(VP_MASK * VP_MASK),
      prompt,
    );
    const raw = this.parseBoxes(this.pickBoxes(out), conf);

    const W = bitmap.width;
    const H = bitmap.height;
    const inv = 1 / lb.gain;
    const dets: VpDetection[] = [];
    for (const b of raw) {
      const x1 = Math.max(0, Math.min(W, (b.x1 - lb.left) * inv));
      const y1 = Math.max(0, Math.min(H, (b.y1 - lb.top) * inv));
      const x2 = Math.max(0, Math.min(W, (b.x2 - lb.left) * inv));
      const y2 = Math.max(0, Math.min(H, (b.y2 - lb.top) * inv));
      const width = x2 - x1;
      const height = y2 - y1;
      // 和手工画框同一条底线：退化的框不入库
      if (width < 2 || height < 2) continue;
      dets.push({ rect: { x: x1, y: y1, width, height }, score: b.score });
    }
    dets.sort((a, b) => b.score - a.score);
    return dets;
  }

  /**
   * 解析端到端输出：每行 `x1,y1,x2,y2, conf, cls, …`（坐标在 letterbox 画布空间）。
   * 布局认不出来时**抛错**，不静默返回空数组 —— 空数组会被用户理解成「模型没找到目标」。
   */
  private parseBoxes(t: OrtTensorOut, confThr: number): RawBox[] {
    const dims = t.dims.map((d) => Number(d));
    if (dims.length < 2) {
      throw new Error(`无法识别的输出形状 [${dims.join(', ')}]，期望至少 2 维。`);
    }
    const cols = dims[dims.length - 1];
    const rows = dims.slice(0, -1).reduce((a, b) => a * b, 1);
    if (cols !== SEG_COLS && cols !== DET_COLS) {
      throw new Error(
        `无法识别的输出布局 [${dims.join(', ')}]：每行应为 ${DET_COLS}（纯检测）或 ${SEG_COLS}` +
          `（带分割分支）列，实际 ${cols} 列。请确认这是 YOLOE 的端到端导出。`,
      );
    }
    if (rows > MAX_ROWS) {
      throw new Error(`输出行数异常（${rows}），这不是端到端布局的 YOLOE 导出。`);
    }

    const data = toFloat32(t.data);
    const cands: RawBox[] = [];
    for (let i = 0; i < rows; i += 1) {
      const o = i * cols;
      const score = data[o + 4];
      // NaN / 未填充的零分行会在这里被自然过滤掉
      if (!(score >= confThr)) continue;
      cands.push({
        x1: data[o],
        y1: data[o + 1],
        x2: data[o + 2],
        y2: data[o + 3],
        score,
      });
    }
    return nms(cands, VP_NMS_IOU);
  }
}

interface RawBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
}

function iou(a: RawBox, b: RawBox): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

/** 贪心 NMS：分数高者优先，丢弃与其重叠过大的框 */
function nms(cands: RawBox[], thr: number): RawBox[] {
  cands.sort((a, b) => b.score - a.score);
  const kept: RawBox[] = [];
  for (const c of cands) {
    if (kept.every((k) => iou(k, c) < thr)) kept.push(c);
  }
  return kept;
}
