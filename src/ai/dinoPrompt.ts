import type { IRect } from '../types';
import {
  bootRuntime,
  createSession,
  type OrtGlobal,
  type OrtSession,
  type OrtTensorOut,
} from './ortRuntime';
import type { VpDetection, VpLetterbox } from './visualPrompt';

/**
 * DINOv2 特征匹配 —— 「照样子找」的第二种引擎。
 *
 * 与 YOLOE 那条链路的**本质**差别（决定了它为什么值得单独接一条）：
 *  · YOLOE 是**检测器**：模型内部有「提示编码 → 检测头」的通路，直接输出框 + 置信度。
 *  · DINOv2 是**特征提取器**：它只会把图变成一串 patch 特征，本身不输出框。
 *    「找同样的东西」得我们在外面自己算 —— 也就是模板匹配的语义版：
 *    参考框内的 patch 特征当模板，在目标图的 patch 特征上滑动找相似的块。
 *
 * 为什么接它：像素模板匹配（NCC）在「同类不同个体」上只能靠堆参考图线性爬，
 * 而 DINOv2 是自监督语义特征，**一张参考图就能跨个体**。同一批图、同一评测口径下实测：
 *
 *   参考图数 k │ 1      │ 4      │ 16
 *   NCC        │ 0.083  │ 0.292  │ 0.708   ← 召回（精度恒 1.000）
 *   DINOv2     │ 0.750  │ 0.833  │ 1.000   ← 召回（精度 0.750 / 0.606 / 0.387）
 *
 *   F1 对比：k=1 时 0.750 vs 0.153（DINOv2 赢 5 倍）；
 *            k=16 时 0.558 vs 0.829（NCC 反超 —— 堆够参考图它也能到高精度）。
 *   两条路互补：**参考少用 DINOv2，参考多（16+）用 NCC 更准**。
 *   （完整实验在 `.tmp/dinov2/`，报告 `dino_report3.txt`）
 *
 * 三个**必须照做**的实现要点，都是实测踩出来的：
 *  1. **逐 patch 分类不够，要按窗口聚合**。单个 patch 语义很弱（ViT 感受野大），
 *     背景上零星几个 patch 撞高分就会成框 —— 实测正样本 p10 处背景误报率 17%。
 *     物体在特征图上是一**块**连续区域，所以在「与参考框同尺寸的窗口」上求均值，
 *     要求整块都像目标。这一条把 k=1 的 F1 从 0.441 拉到 0.750。
 *  2. **阈值必须用 z-score，不能用绝对余弦值**。余弦相似度基线很高（背景 patch
 *     彼此也 0.6+），绝对阈值跨图不可比 —— 直接搬 NCC 的 0.45 会得到 P=0.005。
 *     归一化成「高于本图均值几个标准差」之后，同一个阈值才跨图有效。
 *  3. **原型要用框内全部 patch，不要取均值**。均值把框内的结构多样性抹平了
 *     （实测均值原型 F1 0.394，全 patch 0.750）。
 */

/** 模型固定输入边长（14 × 37）。换模型必须同步改这三个常量 */
export const DINO_SIZE = 518;
export const DINO_PATCH = 14;
export const DINO_GRID = DINO_SIZE / DINO_PATCH;
/** patch 特征维度（ViT-S/14 的 hidden size） */
export const DINO_DIM = 384;

/**
 * 多尺度：同时缩放「原型取样区域」与「匹配窗口」。
 * 与 NCC / YOLOE 两侧取同一组值，方便横向对比。
 */
export const DINO_SCALES = [0.85, 0.92, 1.0, 1.08, 1.15] as const;

/** 单个参考框最多贡献多少个原型 patch（控制推理耗时） */
const MAX_PROTOS_PER_BOX = 48;
/** 合并多张参考图后，每个尺度的原型总数上限 */
const MAX_PROTOS_TOTAL = 96;

/** 框去重 IoU（与 YOLOE 侧的 NMS 0.5 同量级；这里是多尺度合并，用 0.6） */
export const DINO_DEDUP_IOU = 0.6;

/**
 * 每个尺度最多出多少个峰值（防止背景噪声刷屏）。
 * 取 20 与 Python 基准的 `nms_peaks(max_n=20)` 对齐 —— 不一致的话逐值对拍会
 * 差出几个低分框，看着像实现错了，其实只是上限不同。
 */
const MAX_PEAKS = 20;

/**
 * z 阈值 ↔ 界面百分比 的映射区间。
 *
 * 实测最佳工作点在 z=2.0（k=1 时 P=0.750 / R=0.750）；z<1.0 基本全是误检，
 * z>3.5 会把真目标也筛掉。界面上给用户的就是这个区间，默认 2.0。
 */
export const DINO_Z_MIN = 1.0;
export const DINO_Z_MAX = 3.5;
export const DINO_Z_DEFAULT = 2.0;

/**
 * z 值 → 0~1 的「匹配度」，供 aiStore 的 score 字段使用。
 *
 * 注意：**这个数与 YOLOE 的置信度不是一回事**，它只是把 z 线性搬到 0~1，
 * 好让画布上的分数、低置信复核（阈值 0.5）有个统一的量纲。
 *
 * 映射刻意取 `(1 + z) / 5`：实测真实截图上的 z 会到 3.8（不是理论上的 3），
 * 上限卡在 2.75 会把一大片框全截成同一个分数、丢掉排序信息（踩过）。
 * 这样 z=1.0→0.40（用户把阈值调到最低时确实该被当成可疑）、
 * z=2.0（默认工作点）→0.60（高于复核线）、z=4.0→1.00。
 */
function zToScore(z: number): number {
  return Math.max(0, Math.min(1, (1 + z) / 5));
}

export function zFromPercent(pct: number): number {
  return DINO_Z_MIN + (DINO_Z_MAX - DINO_Z_MIN) * pct;
}

export function percentFromZ(z: number): number {
  return (z - DINO_Z_MIN) / (DINO_Z_MAX - DINO_Z_MIN);
}

/** ImageNet 归一化（DINOv2 的官方预处理，与 YOLOE 的「只除 255」不同） */
const MEAN = [0.485, 0.456, 0.406] as const;
const STD = [0.229, 0.224, 0.225] as const;

/**
 * 一个参考框编码出来的「原型集合」。
 *
 * `groups[scaleIdx]` = 该尺度下的一组原型向量，每条已 L2 归一化。
 * 存一组而不是一条：框内不同位置的 patch 长相不同（主体/边缘/内部纹理），
 * 匹配时取「与其中任意一条的最大相似度」，保留这份多样性。
 */
export interface DinoProto {
  groups: Float32Array[][];
  /** 参考框在原图上的像素尺寸（多框取平均）—— 决定目标图上窗口多大 */
  refW: number;
  refH: number;
}

/**
 * 合并多张参考图的原型。
 *
 * 与 YOLOE 侧的 `mergePrompts` 策略不同：那边必须**取均值**（模型的类别嵌入只认一条
 * 提示向量），这边直接把各图的原型**拼起来** —— 匹配时本来就取 max，
 * 拼起来等于「这些例子的长相合集」，不会互相稀释。
 */
export function mergeProtos(list: readonly DinoProto[]): DinoProto {
  if (list.length === 0) throw new Error('至少需要一张参考图');
  if (list.length === 1) return list[0];
  const groups = DINO_SCALES.map((_, si) => {
    const all: Float32Array[] = [];
    for (const p of list) {
      const g = p.groups[si];
      if (g) all.push(...g);
    }
    return capProtos(all, MAX_PROTOS_TOTAL);
  });
  const refW = list.reduce((s, p) => s + p.refW, 0) / list.length;
  const refH = list.reduce((s, p) => s + p.refH, 0) / list.length;
  return { groups, refW, refH };
}

/** 等距采样到 max 条（**确定性**，不用随机 —— 同一批图重复跑必须给同样的结果） */
function capProtos(list: Float32Array[], max: number): Float32Array[] {
  if (list.length <= max) return list;
  const out: Float32Array[] = [];
  const step = list.length / max;
  for (let i = 0; i < max; i += 1) out.push(list[Math.floor(i * step)]);
  return out;
}

function toFloat32(data: Float32Array | BigInt64Array): Float32Array {
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i += 1) out[i] = Number(data[i]);
  return out;
}

/** 与 visualPrompt 的 letterbox 同规则，只是边长不同（用 floor 保持与 Python 对拍一致） */
export function dinoLetterbox(imgW: number, imgH: number): VpLetterbox {
  const gain = Math.min(DINO_SIZE / imgW, DINO_SIZE / imgH);
  const w = Math.max(1, Math.round(imgW * gain));
  const h = Math.max(1, Math.round(imgH * gain));
  return {
    gain,
    w,
    h,
    left: Math.floor((DINO_SIZE - w) / 2),
    top: Math.floor((DINO_SIZE - h) / 2),
  };
}

interface Peak {
  x: number;
  y: number;
  v: number;
}

/** 局部极大值：按分数降序贪心取，取过的点周围 rad 半径内不再取 */
function findPeaks(score: Float32Array, w: number, h: number, thr: number, rad: number): Peak[] {
  const order = new Int32Array(w * h);
  for (let i = 0; i < order.length; i += 1) order[i] = i;
  // Int32Array 没有自定义排序，转成普通数组排（1369 个元素，开销可忽略）
  const idx = Array.from(order).sort((a, b) => score[b] - score[a]);
  const occ = new Uint8Array(w * h);
  const taken: Peak[] = [];
  for (const i of idx) {
    const v = score[i];
    if (v < thr) break;
    if (occ[i]) continue;
    const y = (i / w) | 0;
    const x = i % w;
    taken.push({ x, y, v });
    for (let dy = -rad; dy <= rad; dy += 1) {
      const yy = y + dy;
      if (yy < 0 || yy >= h) continue;
      for (let dx = -rad; dx <= rad; dx += 1) {
        const xx = x + dx;
        if (xx < 0 || xx >= w) continue;
        occ[yy * w + xx] = 1;
      }
    }
    if (taken.length >= MAX_PEAKS) break;
  }
  return taken;
}

/** 窗口均值（用积分图，O(1) 一个窗口） */
function windowMean(src: Float32Array, w: number, h: number, pw: number, ph: number): Float32Array {
  const ii = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y += 1) {
    let rowSum = 0;
    for (let x = 0; x < w; x += 1) {
      rowSum += src[y * w + x];
      ii[(y + 1) * (w + 1) + (x + 1)] = ii[y * (w + 1) + (x + 1)] + rowSum;
    }
  }
  const ow = w - pw + 1;
  const oh = h - ph + 1;
  const out = new Float32Array(ow * oh);
  const n = pw * ph;
  for (let y = 0; y < oh; y += 1) {
    for (let x = 0; x < ow; x += 1) {
      const a = ii[y * (w + 1) + x];
      const b = ii[y * (w + 1) + (x + pw)];
      const c = ii[(y + ph) * (w + 1) + x];
      const d = ii[(y + ph) * (w + 1) + (x + pw)];
      out[y * ow + x] = (d - b - c + a) / n;
    }
  }
  return out;
}

export class DinoPromptDetector {
  private ort: OrtGlobal | null = null;
  private session: OrtSession | null = null;
  private warmed = false;
  private inputName = 'pixel_values';
  /** 复用的离屏画布（518²，与 YOLOE 的 1280² 各用各的） */
  private ctx: CanvasRenderingContext2D | null = null;

  ep: 'WebGPU' | 'WASM' = 'WebGPU';
  warmupError = '';

  get ready(): boolean {
    return this.session !== null;
  }

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
      console.warn(`[dinoPrompt] ${msg}`, err),
    );
    this.session = created.session;
    this.ep = created.ep;
    this.ort = created.ort;
    const names = created.session.inputNames;
    if (names.length > 0) this.inputName = names[0];
    await this.warmup();
  }

  /** 预热：跑一遍零输入，把 WebGPU 的一次性编译开销提前吃掉 */
  private async warmup(): Promise<void> {
    if (this.warmed || !this.session) return;
    this.warmed = true;
    try {
      await this.runRaw(new Float32Array(3 * DINO_SIZE * DINO_SIZE));
      this.warmupError = '';
    } catch (e) {
      this.warmupError = e instanceof Error ? e.message : String(e);
    }
  }

  private async runRaw(image: Float32Array): Promise<Float32Array> {
    const ort = this.ort;
    const sess = this.session;
    if (!ort || !sess) throw new Error('模型未加载');
    const out: Record<string, OrtTensorOut> = await sess.run({
      [this.inputName]: new ort.Tensor('float32', image, [1, 3, DINO_SIZE, DINO_SIZE]),
    });
    const named = out.patch_tokens;
    const t = named ?? Object.values(out)[0];
    if (!t) throw new Error('模型没有输出张量');
    const data = toFloat32(t.data);
    const expect = DINO_GRID * DINO_GRID * DINO_DIM;
    if (data.length !== expect) {
      throw new Error(
        `输出维度异常：${data.length}（期望 ${expect}）。请确认用的是 patch tokens 版 DINOv2 导出。`,
      );
    }
    return data;
  }

  private getCtx(): CanvasRenderingContext2D {
    if (!this.ctx) {
      const c = document.createElement('canvas');
      // 尺寸只赋一次：给 canvas.width 赋同一个值也会清空画布并重置上下文状态
      c.width = DINO_SIZE;
      c.height = DINO_SIZE;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('当前浏览器不支持 Canvas 2D');
      this.ctx = ctx;
    }
    return this.ctx;
  }

  /** letterbox + 像素读取 + CHW + ImageNet 归一化 */
  private buildImage(bitmap: ImageBitmap, lb: VpLetterbox): Float32Array {
    const ctx = this.getCtx();
    ctx.fillStyle = '#727272';
    ctx.fillRect(0, 0, DINO_SIZE, DINO_SIZE);
    ctx.drawImage(bitmap, lb.left, lb.top, lb.w, lb.h);

    const u32 = new Uint32Array(ctx.getImageData(0, 0, DINO_SIZE, DINO_SIZE).data.buffer);
    const plane = DINO_SIZE * DINO_SIZE;
    const data = new Float32Array(3 * plane);
    const inv = 1 / 255;
    for (let i = 0; i < plane; i += 1) {
      const p = u32[i]; // 0xAABBGGRR（小端）
      data[i] = ((p & 255) * inv - MEAN[0]) / STD[0];
      data[plane + i] = (((p >>> 8) & 255) * inv - MEAN[1]) / STD[1];
      data[2 * plane + i] = (((p >>> 16) & 255) * inv - MEAN[2]) / STD[2];
    }
    return data;
  }

  /**
   * 提取 L2 归一化后的 patch 特征图，扁平存放为 (GRID*GRID, DIM)。
   * 归一化在这里做一次，后面所有余弦相似度就退化成点积 —— 省掉每次除法。
   */
  private async features(bitmap: ImageBitmap): Promise<{ f: Float32Array; lb: VpLetterbox }> {
    const lb = dinoLetterbox(bitmap.width, bitmap.height);
    const raw = await this.runRaw(this.buildImage(bitmap, lb));
    const n = DINO_GRID * DINO_GRID;
    const f = new Float32Array(raw.length);
    for (let i = 0; i < n; i += 1) {
      const off = i * DINO_DIM;
      let sum = 0;
      for (let j = 0; j < DINO_DIM; j += 1) {
        const v = raw[off + j];
        sum += v * v;
      }
      const norm = Math.sqrt(sum) || 1;
      for (let j = 0; j < DINO_DIM; j += 1) f[off + j] = raw[off + j] / norm;
    }
    return { f, lb };
  }

  /** 原图坐标的框 → 特征图上的 patch 范围（含尺度缩放） */
  private gridRect(b: IRect, lb: VpLetterbox, scale: number): [number, number, number, number] {
    const cx = b.x + b.width / 2;
    const cy = b.y + b.height / 2;
    const w = b.width * scale;
    const h = b.height * scale;
    const gx0 = (cx - w / 2) * lb.gain + lb.left;
    const gy0 = (cy - h / 2) * lb.gain + lb.top;
    const gx1 = (cx + w / 2) * lb.gain + lb.left;
    const gy1 = (cy + h / 2) * lb.gain + lb.top;
    const x0 = Math.max(0, Math.min(DINO_GRID - 1, Math.floor(gx0 / DINO_PATCH)));
    const y0 = Math.max(0, Math.min(DINO_GRID - 1, Math.floor(gy0 / DINO_PATCH)));
    const x1 = Math.max(x0 + 1, Math.min(DINO_GRID, Math.ceil(gx1 / DINO_PATCH)));
    const y1 = Math.max(y0 + 1, Math.min(DINO_GRID, Math.ceil(gy1 / DINO_PATCH)));
    return [x0, y0, x1, y1];
  }

  /** 原图坐标的框 → 特征图上的 patch 尺寸（用于匹配窗口） */
  private gridSize(b: IRect, lb: VpLetterbox, scale: number): [number, number] {
    const pw = Math.max(1, Math.round((b.width * scale * lb.gain) / DINO_PATCH));
    const ph = Math.max(1, Math.round((b.height * scale * lb.gain) / DINO_PATCH));
    return [pw, ph];
  }

  /**
   * 编码参考：参考图 + 该图上的框 → 原型集合。
   * 框为空时抛错 —— 没有原型必然什么也找不出来，早点说清楚比让用户等一轮更好。
   */
  async encodePrompt(bitmap: ImageBitmap, boxes: readonly IRect[]): Promise<DinoProto> {
    if (!this.session) throw new Error('模型未加载');
    if (boxes.length === 0) throw new Error('这张参考图上还没有框，先画一个再试');
    const { f, lb } = await this.features(bitmap);

    const groups = DINO_SCALES.map((s) => {
      const all: Float32Array[] = [];
      for (const b of boxes) {
        const [x0, y0, x1, y1] = this.gridRect(b, lb, s);
        const local: Float32Array[] = [];
        for (let y = y0; y < y1; y += 1) {
          for (let x = x0; x < x1; x += 1) {
            const off = (y * DINO_GRID + x) * DINO_DIM;
            // 拷贝而不是 subarray：subarray 是视图，原数组要一直存活才能用
            local.push(f.slice(off, off + DINO_DIM));
          }
        }
        all.push(...capProtos(local, MAX_PROTOS_PER_BOX));
      }
      return capProtos(all, MAX_PROTOS_TOTAL);
    });

    const refW = boxes.reduce((s, b) => s + b.width, 0) / boxes.length;
    const refH = boxes.reduce((s, b) => s + b.height, 0) / boxes.length;
    return { groups, refW, refH };
  }

  /**
   * 在一张图上找目标，返回**原图坐标**的框（按分数降序，已做多尺度合并去重）。
   *
   * `zThr` 是「高于本图均值几个标准差」，不是余弦值本身（见文件头第 2 条）。
   */
  async detect(bitmap: ImageBitmap, proto: DinoProto, zThr: number): Promise<VpDetection[]> {
    if (!this.session) throw new Error('模型未加载');
    const { f, lb } = await this.features(bitmap);
    const n = DINO_GRID * DINO_GRID;

    const raw: Array<{ rect: IRect; z: number }> = [];

    for (let si = 0; si < DINO_SCALES.length; si += 1) {
      const group = proto.groups[si];
      if (!group || group.length === 0) continue;

      // 逐 patch 分数 = 与组内任意一条原型的最大余弦（f 与原型都已归一化 → 点积）
      const per = new Float32Array(n);
      for (let i = 0; i < n; i += 1) {
        const off = i * DINO_DIM;
        let best = -Infinity;
        for (const g of group) {
          let dot = 0;
          for (let j = 0; j < DINO_DIM; j += 1) dot += f[off + j] * g[j];
          if (dot > best) best = dot;
        }
        per[i] = best;
      }

      // z-score：让阈值跨图可比
      let mean = 0;
      for (let i = 0; i < n; i += 1) mean += per[i];
      mean /= n;
      let varSum = 0;
      for (let i = 0; i < n; i += 1) {
        const d = per[i] - mean;
        varSum += d * d;
      }
      const std = Math.sqrt(varSum / n) || 1e-6;
      for (let i = 0; i < n; i += 1) per[i] = (per[i] - mean) / std;

      // 窗口聚合：要求整块窗口都像目标，而不是一个 patch 像
      const [pw0, ph0] = this.gridSize({ x: 0, y: 0, width: proto.refW, height: proto.refH }, lb, DINO_SCALES[si]);
      const pw = Math.max(1, Math.min(pw0, DINO_GRID - 1));
      const ph = Math.max(1, Math.min(ph0, DINO_GRID - 1));
      const sm = windowMean(per, DINO_GRID, DINO_GRID, pw, ph);
      const sw = DINO_GRID - pw + 1;
      const sh = DINO_GRID - ph + 1;
      if (sw < 1 || sh < 1) continue;

      const rad = Math.max(2, Math.floor(Math.min(pw, ph) / 2));
      for (const pk of findPeaks(sm, sw, sh, zThr, rad)) {
        // 峰值是窗口左上角 → 加半个窗口得到中心 → 映射回原图
        const cx = (pk.x + pw / 2) * DINO_PATCH;
        const cy = (pk.y + ph / 2) * DINO_PATCH;
        const w = (pw * DINO_PATCH) / lb.gain;
        const h = (ph * DINO_PATCH) / lb.gain;
        const ox = (cx - lb.left) / lb.gain;
        const oy = (cy - lb.top) / lb.gain;
        raw.push({
          rect: { x: ox - w / 2, y: oy - h / 2, width: w, height: h },
          z: pk.v,
        });
      }
    }

    // 多尺度合并去重（分数高者优先）
    raw.sort((a, b) => b.z - a.z);
    const kept: Array<{ rect: IRect; z: number }> = [];
    for (const c of raw) {
      if (kept.every((k) => iou(k.rect, c.rect) < DINO_DEDUP_IOU)) kept.push(c);
    }

    const W = bitmap.width;
    const H = bitmap.height;
    const dets: VpDetection[] = [];
    for (const k of kept) {
      const x1 = Math.max(0, Math.min(W, k.rect.x));
      const y1 = Math.max(0, Math.min(H, k.rect.y));
      const x2 = Math.max(0, Math.min(W, k.rect.x + k.rect.width));
      const y2 = Math.max(0, Math.min(H, k.rect.y + k.rect.height));
      const width = x2 - x1;
      const height = y2 - y1;
      // 和手工画框同一条底线：退化的框不入库
      if (width < 2 || height < 2) continue;
      dets.push({ rect: { x: x1, y: y1, width, height }, score: zToScore(k.z) });
    }
    dets.sort((a, b) => b.score - a.score);
    return dets;
  }
}

function iou(a: IRect, b: IRect): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}
