import type { IRect } from '../types';
import {
  bootRuntime,
  createSession,
  type OrtGlobal,
  type OrtSession,
  type OrtTensorOut,
} from './ortRuntime';

/**
 * MobileSAM 交互式分割（点击辅助标注）。
 *
 * 与 detector.ts（YOLO 检测）的本质差异：这是**两段式**推理。
 *   encoder：整图 [1,3,1024,1024] → image_embeddings [1,256,64,64]，慢（1~3s），每张图只跑一次
 *   decoder：embeddings + 点击坐标 → masks，快（100~300ms），每次点击跑一次
 * 所以使用体验的关键是**把 encoder 的结果按图缓存住**，点击时才不用重新编码。
 *
 * 坐标语义（来自 `mobile_sam/utils/onnx.py`，必须严格对齐，否则框会整体错位）：
 *  1. `point_coords` 在「长边缩放到 1024」的坐标系里（内部会除以 img_size=1024 归一化），
 *     **不是**原图坐标，也不是任意 0~1 归一化坐标；
 *  2. `orig_im_size` 传**原图 (H, W)**，模型内部据此把 1024 尺度的 mask 裁掉 pad
 *     再缩放回原图尺寸——所以**输出的 masks 已经是原图尺寸**，前端不用再逆变换；
 *  3. 缩放系数模型内部用的是 `floor(scale * size + 0.5)`，前端必须用**同一个四舍五入**，
 *     否则 mask 裁剪边界差 1 像素，边框就会有系统性偏移；
 *  4. 预处理是**先归一化、再把 pad 区域补 0**（不是「补黑像素再归一化」——
 *     那会让 pad 区变成 -2.1 而不是 0，模型看到的是完全不同的输入）。
 *
 * 输出的 masks 是 **logits**，需要按阈值 0 二值化（`model.mask_threshold = 0.0`）。
 */

/** SAM 编码器的输入边长。TinyViT 用绝对位置编码，尺寸写死不能改 */
const SAM_SIZE = 1024;
/** encoder 输出的 embedding 空间尺寸（1024 / patch 16） */
const EMB_SIZE = 64;
/** embedding 元素数：1×256×64×64 */
const EMB_LEN = 256 * EMB_SIZE * EMB_SIZE;

/** SAM 的像素归一化参数（ImageNet 统计量，**0-255 尺度**，不是 0-1） */
const PIXEL_MEAN = [123.675, 116.28, 103.53];
const PIXEL_STD = [58.395, 57.12, 57.375];

/** logits > 此值判为前景 */
const MASK_THRESHOLD = 0;
/** subdivision 用的低分辨率 mask 边长 */
const LOW_MASK = 256;

/**
 * embedding 缓存上限（张）。
 * 单张 embedding = 256×64×64 × 4B ≈ **4MB**，缓存 24 张约 96MB —— 这是刻意压住的量级：
 * 不设上限的话，标 200 张图就是 800MB，标签页会被浏览器直接杀掉。
 */
const EMBED_CACHE_MAX = 24;

/** 一次点击（或多次点击累积）里的一个点 */
export interface SegmentPoint {
  /** 原图像素坐标 */
  x: number;
  y: number;
  /** true = 想要这个物体（左键）；false = 排除这个区域（右键） */
  positive: boolean;
}

export interface SegmentResult {
  /** 原图像素坐标的外接矩形（已按 1px 最小尺寸收口） */
  rect: IRect;
  /** 模型对自己这块 mask 的质量预测（0~1，越高越可信） */
  score: number;
  /** mask 面积占全图比例，用于识别「点空了」或「选中整张图」 */
  coverage: number;
}

/** 图片源，统一交给 drawImage 缩放 */
export type ImageSource = ImageBitmap | HTMLCanvasElement | HTMLImageElement;

export class Segmenter {
  private ort: OrtGlobal | null = null;
  private encoder: OrtSession | null = null;
  private decoder: OrtSession | null = null;

  /**
   * 复用的离屏画布：SAM 输入很大（1024²），每次新建画布会造成明显的 GC 抖动。
   * 只需要存 ctx —— ctx.canvas 就是那个 canvas，单独存字段会变成没人读的僵尸字段。
   */
  private ctx: CanvasRenderingContext2D | null = null;

  /**
   * imageId → embedding 的 LRU 缓存。
   * 用 Map 的插入顺序当 LRU 序：命中时先 delete 再 set 挪到末尾，超限时删最前面那个。
   */
  private cache = new Map<string, Float32Array>();

  ep: 'WebGPU' | 'WASM' = 'WebGPU';
  /** 非空表示加载/运行出的问题，透出到界面而不是静默 */
  error = '';

  get ready(): boolean {
    return this.encoder !== null && this.decoder !== null;
  }

  /** 缓存里已有哪张图的 embedding（界面可据此提示"已就绪，点击即可"） */
  hasEmbedding(imageId: string): boolean {
    return this.cache.has(imageId);
  }

  clearCache(): void {
    this.cache.clear();
  }

  /** 引导推理引擎（复用 detector 的运行时引导，本地优先、CDN 兜底） */
  async boot(): Promise<void> {
    if (this.ort) return;
    this.ort = await bootRuntime();
  }

  /**
   * 加载 encoder / decoder 两个会话。
   * 分成两次是为了**让界面能显示真实进度**：encoder 26MB、decoder 16MB，
   * 首次访问总共要下 42MB，不给进度提示会让人以为卡死了。
   */
  async loadModels(
    encoderBytes: Uint8Array,
    decoderBytes: Uint8Array,
    onProgress?: (stage: 'encoder' | 'decoder', done: boolean) => void,
  ): Promise<void> {
    await this.boot();
    const ort = this.ort;
    if (!ort) throw new Error('推理引擎未就绪');

    this.error = '';
    const warn = (msg: string, err: unknown) => console.warn(`[segmenter] ${msg}`, err);

    onProgress?.('encoder', false);
    const enc = await createSession(ort, encoderBytes, warn);
    this.encoder = enc.session;
    this.ort = enc.ort; // CDN 兜底后运行时对象会换，必须更新
    this.ep = enc.ep;
    onProgress?.('encoder', true);

    onProgress?.('decoder', false);
    const dec = await createSession(ort, decoderBytes, warn);
    this.decoder = dec.session;
    this.ort = dec.ort;
    onProgress?.('decoder', true);
  }

  /**
   * 取（或新建）离屏画布。
   * willReadFrequently 必须开：每个点都要 getImageData 一次整张 1024²，
   * 不开的话浏览器会把画布放在 GPU 上，每次读取都要回传，慢一个量级。
   */
  private getCtx(): CanvasRenderingContext2D {
    if (!this.ctx) {
      const c = document.createElement('canvas');
      c.width = SAM_SIZE;
      c.height = SAM_SIZE;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('无法创建 2D 上下文');
      this.ctx = ctx;
    }
    return this.ctx;
  }

  /**
   * 预处理：长边等比缩放到 1024 → 左上角对齐贴到 1024×1024 画布（右下留空）
   * → 归一化 → NCHW。
   *
   * 注意 pad 区**保持 0**：Float32Array 初始就是 0，所以只填有效区域即可 ——
   * 这正是 SAM `preprocess()` 的行为（先 normalize 再 pad 0）。
   */
  private preprocess(
    source: ImageSource,
    width: number,
    height: number,
  ): { tensor: Float32Array; scale: number; preW: number; preH: number } {
    const scale = SAM_SIZE / Math.max(width, height);
    // 必须和 onnx.py 的 resize_longest_image_size 用同一种取整，否则差 1px
    const preW = Math.min(SAM_SIZE, Math.round(width * scale));
    const preH = Math.min(SAM_SIZE, Math.round(height * scale));

    const ctx = this.getCtx();
    ctx.clearRect(0, 0, SAM_SIZE, SAM_SIZE);
    ctx.drawImage(source, 0, 0, preW, preH);

    const { data } = ctx.getImageData(0, 0, SAM_SIZE, SAM_SIZE);
    const tensor = new Float32Array(3 * SAM_SIZE * SAM_SIZE);
    const plane = SAM_SIZE * SAM_SIZE;

    // 只遍历有效区域：pad 区要的就是 0，写它纯属浪费
    for (let y = 0; y < preH; y += 1) {
      for (let x = 0; x < preW; x += 1) {
        const si = (y * SAM_SIZE + x) * 4;
        const di = y * SAM_SIZE + x;
        tensor[di] = (data[si] - PIXEL_MEAN[0]) / PIXEL_STD[0];
        tensor[plane + di] = (data[si + 1] - PIXEL_MEAN[1]) / PIXEL_STD[1];
        tensor[2 * plane + di] = (data[si + 2] - PIXEL_MEAN[2]) / PIXEL_STD[2];
      }
    }
    return { tensor, scale, preW, preH };
  }

  private putCache(imageId: string, emb: Float32Array): void {
    if (this.cache.has(imageId)) this.cache.delete(imageId);
    this.cache.set(imageId, emb);
    while (this.cache.size > EMBED_CACHE_MAX) {
      const oldest = this.cache.keys().next().value;
      if (oldest === undefined) break;
      this.cache.delete(oldest);
    }
  }

  /** 编码一张图并缓存。已经编过的直接复用，不重复跑 encoder。 */
  async encode(
    imageId: string,
    source: ImageSource,
    width: number,
    height: number,
  ): Promise<void> {
    if (this.cache.has(imageId)) {
      // 命中也要挪到 LRU 末尾，否则热点图会被后面的大批图挤掉
      const hit = this.cache.get(imageId)!;
      this.putCache(imageId, hit);
      return;
    }
    const ort = this.ort;
    const sess = this.encoder;
    if (!ort || !sess) throw new Error('分割模型未加载完成');

    const { tensor } = this.preprocess(source, width, height);
    const out = await sess.run({
      [sess.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, SAM_SIZE, SAM_SIZE]),
    });
    const emb = this.readFloat(out);
    if (!emb || emb.length !== EMB_LEN) {
      throw new Error(
        `编码输出形状异常（期望 ${EMB_LEN} 个元素，拿到 ${emb ? emb.length : 0} 个）。` +
          '请确认选的是 MobileSAM 的 encoder 模型，而不是完整 SAM 或 decoder。',
      );
    }
    this.putCache(imageId, emb);
  }

  /** 从会话输出里取第一个 float32 张量 */
  private readFloat(out: Record<string, OrtTensorOut>): Float32Array | null {
    for (const key of Object.keys(out)) {
      const t = out[key];
      if (t && t.data instanceof Float32Array) return t.data;
    }
    return null;
  }

  /**
   * 在已编码的图上做一次分割。
   * @param points 累积的点击点（原图坐标）。第一个是正点，后续可加负点排除。
   */
  async segment(
    imageId: string,
    width: number,
    height: number,
    points: SegmentPoint[],
  ): Promise<SegmentResult> {
    const ort = this.ort;
    const sess = this.decoder;
    if (!ort || !sess) throw new Error('分割模型未加载完成');
    const emb = this.cache.get(imageId);
    if (!emb) throw new Error('这张图还没编码，请先编码再点击');

    if (points.length === 0) throw new Error('至少需要一个点');

    const scale = SAM_SIZE / Math.max(width, height);

    // 坐标映射：原图 → 1024 尺度。pad 在右下、有效区左上角对齐原点，所以没有偏移量。
    // 夹到 [0,1024]：点在图片边缘外侧时（比如从画布拖出去）模型会算出离谱区域。
    const coords = new Float32Array(points.length * 2);
    const labels = new Float32Array(points.length);
    points.forEach((p, i) => {
      coords[i * 2] = Math.max(0, Math.min(SAM_SIZE, p.x * scale));
      coords[i * 2 + 1] = Math.max(0, Math.min(SAM_SIZE, p.y * scale));
      labels[i] = p.positive ? 1 : 0;
    });

    const out = await sess.run({
      image_embeddings: new ort.Tensor('float32', emb, [1, 256, EMB_SIZE, EMB_SIZE]),
      point_coords: new ort.Tensor('float32', coords, [1, points.length, 2]),
      point_labels: new ort.Tensor('float32', labels, [1, points.length]),
      // 不做多轮 refinement：mask_input 全零 + has_mask_input=0 走 no_mask_embed 分支。
      // 多点已经在 point 维度上表达了，第一版不必再叠一层低分辨率 mask。
      mask_input: new ort.Tensor('float32', new Float32Array(LOW_MASK * LOW_MASK), [
        1, 1, LOW_MASK, LOW_MASK,
      ]),
      has_mask_input: new ort.Tensor('float32', new Float32Array([0]), [1]),
      // 模型内部据此把 mask 裁 pad 再缩放回原图 —— 输出的 masks 直接就是原图尺寸
      orig_im_size: new ort.Tensor('float32', new Float32Array([height, width]), [2]),
    });

    const masks = out.masks;
    if (!masks || !(masks.data instanceof Float32Array)) {
      throw new Error('分割模型没有返回 masks，请确认 decoder 模型是否正确');
    }

    const score = this.readScore(out.iou_predictions);
    const rect = this.maskToRect(masks.data, width, height);

    const coverage = ((rect.width * rect.height) / (width * height)) || 0;
    return { rect, score, coverage };
  }

  private readScore(t: OrtTensorOut | undefined): number {
    if (!t || !(t.data instanceof Float32Array) || t.data.length === 0) return 0;
    return t.data[0];
  }

  /**
   * mask（logits，尺寸=原图）→ 二值化 → 外接矩形。
   *
   * 这里承担「不静默失败」的职责：
   *  - 一个前景像素都没有 → 抛错说清楚（用户点到了背景/空白处）
   *  - 前景占满整图 → 也抛错（多半点在了大面积背景上，出来的是全图框）
   * 这两种情况如果直接返回一个空框或全图框，用户会以为「工具坏了」。
   */
  private maskToRect(mask: Float32Array, width: number, height: number): IRect {
    // mask 可能是 [1,1,H,W]，也可能因为动态轴变成 [1,H,W]，这里按「最后 H*W 个元素」定位。
    const total = mask.length;
    const need = width * height;
    if (total < need) {
      throw new Error(
        `分割结果尺寸异常（mask ${total} 个值 < 图片 ${width}×${height}）。请确认 decoder 是 MobileSAM 版本。`,
      );
    }
    const base = total - need;

    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    let count = 0;
    for (let y = 0; y < height; y += 1) {
      const row = base + y * width;
      for (let x = 0; x < width; x += 1) {
        if (mask[row + x] > MASK_THRESHOLD) {
          count += 1;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (count === 0) {
      throw new Error('这一下没选中任何东西，试着点在物体内部（不是背景或边缘外）');
    }
    if (count > width * height * 0.92) {
      throw new Error('这一下把整张图都选中了，点在物体中心再试，或用右键点掉多余区域');
    }

    // 外接矩形按 1px 最小尺寸收口：抹平退化框，和 addBox 的守卫保持同一条底线
    const w = Math.max(1, maxX - minX + 1);
    const h = Math.max(1, maxY - minY + 1);
    return { x: minX, y: minY, width: w, height: h };
  }
}
