import type { IRect } from '../types';
import { parseClassNames } from './cocoNames';
import { readOnnxInputDims, readOnnxMetadata, parseImgszHint } from './onnxShape';
// ORT 引导层已抽到 ortRuntime.ts（segmenter 也要用），这里只做引用
import {
  bootRuntime,
  createSession,
  type OrtGlobal,
  type OrtSession,
  type OrtTensorOut,
} from './ortRuntime';

/**
 * ONNX Runtime Web 推理封装：
 *   加载运行时 → 创建会话（WebGPU，失败回退 WASM）→ letterbox 预处理 →
 *   推理（输入尺寸自愈 + 探测）→ 解析输出布局 → 逆变换回原图坐标。
 *
 * 与演示页的两点差异：
 *  1. 加载器与伴生二进制已随应用本地打包（public/ort/），缺失时才回退 CDN，
 *     加载器与伴生二进制始终取同一个 base，保证版本配套。
 *  2. 检测结果做【逆 letterbox 变换】：x_orig = (x_lb - padX) / scale，
 *     因为标注框要存进原图像素坐标系，而不是补边后的画布坐标系。
 *
 * 输出布局（依据 Ultralytics 官方导出文档）：
 *  - NMS-Free 端到端（YOLO26 `nms=False`、YOLOv10）：[B, max_det, 6] = x1,y1,x2,y2,conf,cls，**不能再做 NMS**
 *  - 原始一对多（YOLO26 `nms=None` 默认导出、v8/v11）：[B, 4+nc, N] 或 [B, N, 4+nc]，**必须自己做 NMS**
 * 两种布局都支持，且按「框数量轴」区分，避免把 2 类模型（4+2=6）误判成端到端。
 */

export interface AiDetection {
  /** 原图像素坐标 */
  rect: IRect;
  score: number;
  classIndex: number;
}

const DEFAULT_INPUT = 640;
const NMS_IOU = 0.45;
/**
 * 动态轴模型（dynamic=True 导出）的输入尺寸探测顺序。
 * 这一层只在「拿不到静态维度」时才用得上；静态维度的模型早已由
 * readOnnxInputDims 精确读出（实测 yolo26n.onnx = [1,3,640,640]，
 * 而官方文档里 YOLO26 又提到过 768 —— 正因为各版本/导出参数不一致，
 * 才必须读模型、不能写死）。动态轴下 768 排在前面：喂 640 也能跑，
 * 但与训练分辨率不一致时精度会下降。
 */
const PROBE_SIZES = [768, 640, 1024, 896, 512, 1280, 320];
/** 端到端输出的框数量上限（max_det 默认 300）；远小于此值才判定为端到端布局 */
const E2E_MAX_DET = 1000;

interface Letterbox {
  padX: number;
  padY: number;
  scale: number;
}

export class Detector {
  private ort: OrtGlobal | null = null;
  private session: OrtSession | null = null;
  private warmed = false;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  /** 模型实际输入尺寸（先读 ONNX 图，再读元数据，最后探测） */
  inputW = DEFAULT_INPUT;
  inputH = DEFAULT_INPUT;
  /** 类别数：第一次成功解析输出后才有值 */
  numClasses = 0;
  /**
   * 模型自带的类别名（ONNX metadata 的 `names`）。读不到时为 null。
   * 有这个之后自定义模型不再显示「类别 1 / 类别 2」，而是模型里写好的真名（含中文）。
   */
  modelNames: string[] | null = null;
  /** 模型自带的任务类型（metadata 的 `task`）：detect / segment / pose / classify */
  modelTask = '';
  ep: 'WebGPU' | 'WASM' = 'WebGPU';
  /** 输入尺寸是怎么定下来的：onnx / onnx-meta / meta / probe / default（展示给用户排查） */
  inputSizeFrom = 'default';
  /** 最近一次推理拿到的输出布局描述（排查用；空串=还没跑过） */
  layoutInfo = '';
  /** 预热阶段的报错（空串=正常）。非空基本等于「这个模型现在跑不起来」 */
  warmupError = '';
  /** 模型字节（保留引用以便需要时再解析） */
  private modelBytes: Uint8Array | null = null;
  /** metadata 里的 imgsz：动态轴模型确定输入尺寸的第二来源 */
  private metaImgsz: [number, number] | null = null;

  get ready(): boolean {
    return this.session !== null;
  }

  /** 加载推理引擎：优先用随应用打包的本地运行时（public/ort/），缺失时回退 CDN 多镜像 */
  async boot(): Promise<void> {
    if (this.ort) return;
    this.ort = await bootRuntime();
  }

  /** 加载模型字节并建立会话：WebGPU 优先，失败回退 CPU */
  async loadModel(bytes: Uint8Array): Promise<void> {
    const ort = this.ort;
    if (!ort) throw new Error('推理引擎未就绪');

    this.session = null;
    this.warmed = false;
    this.numClasses = 0;
    this.modelNames = null;
    this.modelTask = '';
    this.metaImgsz = null;
    this.layoutInfo = '';
    this.warmupError = '';
    this.modelBytes = bytes;
    // 元数据要在这里就读出来：类名必须在解析输出之前就位（numClasses 的推断依赖它），
    // imgsz 则是后面 applyGeometry 的第二层依据
    this.applyModelMetadata(bytes);

    const created = await createSession(ort, bytes, (msg, err) =>
      console.warn(`[detector] ${msg}`, err),
    );
    this.session = created.session;
    this.ep = created.ep;
    // CDN 兜底后运行时对象换了，必须把引用更新过去
    this.ort = created.ort;

    await this.applyGeometry();
    await this.warmup();
  }

  /**
   * 读模型自带的元数据（`names` / `task` / `imgsz`）。
   *
   * **任何一项读不到都只是降级，绝不能让模型加载失败** —— 被 `onnx.utils.extract_model`
   * 或 pnnx 转换过的模型整段元数据是空的（实测 yolov10n.onnx、yolo26n.pnnx.onnx），
   * 这时类名回退成「类别 N」、输入尺寸回退到后面的探测流程，一切照旧。
   */
  private applyModelMetadata(bytes: Uint8Array): void {
    const meta = readOnnxMetadata(bytes);
    this.modelNames = parseClassNames(meta?.names);
    this.modelTask = meta?.task ?? '';
    this.metaImgsz = parseImgszHint(meta?.imgsz);
  }

  /**
   * 确定输入尺寸。四层策略，从可靠到兜底：
   *  1. 直接解析 ONNX 图里的静态维度 —— 最权威，静态导出必走这一步；
   *  2. ONNX metadata 的 `imgsz` —— 训练分辨率，动态轴模型的最优解；
   *  3. ORT 的 inputMetadata（不同版本/EP 形态不一致，拿不到就跳过）；
   *  4. 运行时探测：拿候选尺寸逐个试跑，第一个不报错的即采用。
   * 全部失败才退回 640。
   */
  private async applyGeometry(): Promise<void> {
    const sess = this.session;
    if (!sess) return;

    // ① ONNX 图里的静态维度
    const dims = this.modelBytes ? readOnnxInputDims(this.modelBytes) : null;
    if (dims && dims.length >= 4) {
      const h = dims[dims.length - 2];
      const w = dims[dims.length - 1];
      if (h > 1 && w > 1) {
        this.inputH = h;
        this.inputW = w;
        this.inputSizeFrom = 'onnx';
        return;
      }
    }

    // ② metadata 的 imgsz。
    //    这层是补上一个真实存在的坑：动态轴模型在 ① 拿不到静态维（-1），
    //    以前直接进 ④「探测」，而探测列表里 768 排在 640 前面 ——
    //    于是 640 训出来的模型被喂 768：**能跑，不报错，但和训练分辨率不一致，精度会掉**，
    //    而且用户完全看不出哪里不对。Ultralytics 的导出里 imgsz 是必写的，读了它就不用猜。
    if (this.metaImgsz) {
      this.inputW = this.metaImgsz[0];
      this.inputH = this.metaImgsz[1];
      this.inputSizeFrom = 'onnx-meta';
      return;
    }

    // ③ ORT 元数据
    const meta = this.getInputShape(sess, sess.inputNames[0]).filter(
      (d): d is number => typeof d === 'number' && Number.isFinite(d) && d > 1,
    );
    if (meta.length >= 2) {
      this.inputH = meta[meta.length - 2];
      this.inputW = meta[meta.length - 1];
      this.inputSizeFrom = 'meta';
      return;
    }

    // ④ 探测：必须在 warmup 之前完成，否则预热会拿旧尺寸去跑
    this.inputSizeFrom = 'probe';
    await this.probeInputSize();
  }

  /**
   * 逐个试跑候选尺寸。动态轴模型怎么都能跑，所以第一个候选就会命中；
   * 固定尺寸模型则只有真正匹配的那个能跑通（其余都会 shape mismatch 被 catch 掉）。
   */
  private async probeInputSize(): Promise<void> {
    const sess = this.session;
    if (!sess) return;
    for (const size of PROBE_SIZES) {
      try {
        await sess.run({
          [sess.inputNames[0]]: this.makeTensor(
            new Float32Array(3 * size * size),
            size,
            size,
          ),
        });
        this.inputW = size;
        this.inputH = size;
        this.inputSizeFrom = `probe:${size}`;
        return;
      } catch {
        /* 换下一个尺寸继续试 */
      }
    }
    this.inputW = DEFAULT_INPUT;
    this.inputH = DEFAULT_INPUT;
    this.inputSizeFrom = 'default';
  }

  /** 兼容不同 ORT 构建的元数据形态（数组/对象、shape/dimensions 都出现过） */
  private getInputShape(sess: OrtSession, name: string): Array<number | string> {
    const all = sess.inputMetadata as
      | Array<{ name: string; shape?: Array<number | string>; dimensions?: Array<number | string> }>
      | Record<string, { shape?: Array<number | string>; dimensions?: Array<number | string> }>
      | null;
    if (!all) return [];
    const meta = Array.isArray(all) ? all.find((x) => x.name === name) : all[name];
    return meta?.shape ?? meta?.dimensions ?? [];
  }

  /** 预热：零张量先跑一遍，消化 WebGPU 一次性编译开销 */
  private async warmup(): Promise<void> {
    if (this.warmed || !this.session) return;
    this.warmed = true;
    try {
      await this.infer(() =>
        this.makeTensor(
          new Float32Array(3 * this.inputH * this.inputW),
          this.inputW,
          this.inputH,
        ),
      );
      this.warmupError = '';
    } catch (e) {
      // 记下来给 UI 展示：预热就失败通常意味着输入尺寸仍然不对，
      // 以前这里静默吞掉，用户只会看到「什么都没识别出来」。
      this.warmupError = e instanceof Error ? e.message : String(e);
    }
  }

  private makeTensor(data: Float32Array, w: number, h: number): unknown {
    const ort = this.ort;
    if (!ort) throw new Error('推理引擎未就绪');
    return new ort.Tensor('float32', data, [1, 3, h, w]);
  }

  /**
   * 带自愈的推理。
   *
   * 各版本 ORT 的 shape mismatch 报错文本差别很大（"Expected: N"、"expected [...] but got [...]"…），
   * 原来那条正则只认一种句式，稍微换个版本就彻底失效、直接把错误抛给用户。
   * 现在改成：先把报错里所有 "Expected: N" 收集起来（最后两个通常是 H、W），
   * 收集不到就重新探测一次尺寸。
   */
  private async infer(
    makeTensor: () => unknown,
  ): Promise<Record<string, OrtTensorOut>> {
    const sess = this.session;
    if (!sess) throw new Error('模型未加载');
    // 每次都重新建张量，所以上面改完 inputW/H 后重试会自动用上新尺寸
    const feed = () => ({ [sess.inputNames[0]]: makeTensor() });
    try {
      return await sess.run(feed());
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      const expected = Array.from(msg.matchAll(/Expected:?\s*\[?(\d+)/g))
        .map((m) => Number(m[1]))
        .filter((n) => Number.isFinite(n) && n > 1 && n <= 8192);
      if (expected.length >= 2) {
        this.inputH = expected[expected.length - 2];
        this.inputW = expected[expected.length - 1];
        this.inputSizeFrom = 'error-recover';
        return await sess.run(feed());
      }
      // 连期望尺寸都读不出来：多半是探测阶段就选错了，重新探一遍
      await this.probeInputSize();
      return await sess.run(feed());
    }
  }

  private letterbox(imgW: number, imgH: number): Letterbox {
    const scale = Math.min(this.inputW / imgW, this.inputH / imgH);
    return {
      padX: (this.inputW - imgW * scale) / 2,
      padY: (this.inputH - imgH * scale) / 2,
      scale,
    };
  }

  /** letterbox + 像素读取 + HWC→CHW 归一化，一次产出输入张量 */
  private buildInputTensor(bitmap: ImageBitmap): unknown {
    const W = this.inputW;
    const H = this.inputH;
    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    }
    const ctx = this.ctx;
    if (!this.canvas || !ctx) throw new Error('当前浏览器不支持 Canvas 2D');
    // 只在尺寸真的变了才改：给 canvas.width 赋**同一个值**同样会清空画布并重置上下文状态，
    // 等于每次推理都白做一遍。尺寸没变时下面本来就会整幅 fillRect 覆盖，不需要它来清屏。
    // 批量标注时尺寸恒为模型的输入尺寸，这一改动让每张图都少一次画布重置。
    if (this.canvas.width !== W || this.canvas.height !== H) {
      this.canvas.width = W;
      this.canvas.height = H;
    }

    const { padX, padY, scale } = this.letterbox(bitmap.width, bitmap.height);
    ctx.fillStyle = '#727272'; // RGB(114)，Ultralytics 标准补边色
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(bitmap, padX, padY, bitmap.width * scale, bitmap.height * scale);

    const u32 = new Uint32Array(ctx.getImageData(0, 0, W, H).data.buffer);
    const plane = W * H;
    const data = new Float32Array(3 * plane);
    const inv = 1 / 255;
    for (let i = 0; i < plane; i += 1) {
      const p = u32[i]; // 0xAABBGGRR（小端）
      data[i] = (p & 255) * inv;
      data[plane + i] = ((p >>> 8) & 255) * inv;
      data[2 * plane + i] = ((p >>> 16) & 255) * inv;
    }
    return this.makeTensor(data, W, H);
  }

  /** 对一张图做检测，返回原图坐标系的候选框（按置信度降序） */
  async detect(bitmap: ImageBitmap, conf: number): Promise<AiDetection[]> {
    if (!this.session) throw new Error('模型未加载');
    const results = await this.infer(() => this.buildInputTensor(bitmap));
    const out = this.pickMainOutput(results);
    const boxes = this.parseDetections(out, conf);

    // 逆 letterbox 变换：letterbox 画布坐标 → 原图像素坐标
    const { padX, padY, scale } = this.letterbox(bitmap.width, bitmap.height);
    const inv = 1 / scale;
    const detections: AiDetection[] = [];
    for (const b of boxes) {
      const x1 = Math.max(0, Math.min(bitmap.width, (b.x1 - padX) * inv));
      const y1 = Math.max(0, Math.min(bitmap.height, (b.y1 - padY) * inv));
      const x2 = Math.max(0, Math.min(bitmap.width, (b.x2 - padX) * inv));
      const y2 = Math.max(0, Math.min(bitmap.height, (b.y2 - padY) * inv));
      const width = x2 - x1;
      const height = y2 - y1;
      if (width < 2 || height < 2) continue;
      detections.push({
        rect: { x: x1, y: y1, width, height },
        score: b.score,
        classIndex: b.cls,
      });
    }
    detections.sort((a, b) => b.score - a.score);
    return detections;
  }

  /** 从会话输出里挑出主输出：优先三维 [B,...]，其次二维 [N,...] */
  private pickMainOutput(results: Record<string, OrtTensorOut>): OrtTensorOut {
    const names = this.session ? this.session.outputNames : Object.keys(results);
    for (const want of [3, 2]) {
      for (const name of names) {
        const t = results[name];
        if (t && t.dims.length === want) return t;
      }
    }
    const any = Object.values(results)[0];
    if (!any) throw new Error('模型没有输出张量');
    return any;
  }

  /**
   * 解析输出 → [{x1,y1,x2,y2,score,cls}]，坐标仍在 letterbox 画布空间。
   *
   * 关键：布局认不出来时**必须抛错**，绝不能静默返回空数组 ——
   * 之前正是这里 return [] 让 YOLO26 表现为「模型坏了，一个都识别不出来」，
   * 用户根本看不到任何线索。
   */
  private parseDetections(out: OrtTensorOut, confThr: number): RawBox[] {
    const dims = out.dims.map((d) => Number(d));
    const data = toFloat32(out.data);

    // [N, C]：二维输出同样要区分端到端与经典头，
    // 否则 [300,6] 的 NMS-Free 导出会被当成「2 类模型」解析出乱七八糟的框
    if (dims.length === 2) {
      const [cnt, M] = dims;
      if (M === 6 && cnt < E2E_MAX_DET) return this.end2end(data, cnt, false, confThr);
      return this.classic(data, cnt, M, false, confThr, `[${cnt}, ${M}]`);
    }
    if (dims.length !== 3) {
      throw new Error(
        `无法识别的输出形状 [${dims.join(', ')}]。请确认选择的是 YOLO「检测」模型（分类/分割/姿态模型的输出结构不同）。`,
      );
    }

    const [n, a, b] = dims;
    if (n > 1) {
      // 批推理才会出现多 batch；本工具一次只喂一张，这里明确拒绝而不是只取第 0 张
      throw new Error(`模型输出 batch=${n}，本工具一次只处理一张图片，请导出 batch=1 的模型。`);
    }

    // ① 端到端 NMS-Free：[1, max_det, 6] 或 [1, 6, max_det]
    //    判据不能只看「通道数=6」—— 2 类模型的经典头也是 4+2=6。
    //    端到端的框数量轴是 max_det（默认 300），远小于经典头的几千个锚点。
    if (b === 6 && a < E2E_MAX_DET) return this.end2end(data, a, false, confThr);
    if (a === 6 && b < E2E_MAX_DET) return this.end2end(data, b, true, confThr);

    // ② 经典一对多头：[1, 4+nc, N]（Ultralytics 默认，通道在前）或 [1, N, 4+nc]
    if (a >= 5 && a - 4 <= b) return this.classic(data, b, a, true, confThr, `[1, ${a}, ${b}]`);
    if (b >= 5 && b - 4 <= a) return this.classic(data, a, b, false, confThr, `[1, ${a}, ${b}]`);

    throw new Error(
      `无法识别的输出布局 [1, ${a}, ${b}]：既不像端到端（通道应为 6），也不像检测头（通道应为 4+类别数）。` +
        `请确认这是 YOLO 检测模型的 ONNX 导出。`,
    );
  }

  /** 端到端 NMS-Free 输出：每行 x1,y1,x2,y2,conf,cls，**模型已内置筛选，不要再 NMS** */
  private end2end(
    data: Float32Array,
    count: number,
    channelFirst: boolean,
    confThr: number,
  ): RawBox[] {
    const M = 6;
    const at = (i: number, c: number): number =>
      channelFirst ? data[c * count + i] : data[i * M + c];

    const boxes: RawBox[] = [];
    let maxCls = 0;
    for (let i = 0; i < count; i += 1) {
      const score = at(i, 4);
      // NaN / 未填充的零分行会在这里被自然过滤掉
      if (!(score >= confThr)) continue;
      const cls = at(i, 5) | 0;
      if (cls > maxCls) maxCls = cls;
      boxes.push({ x1: at(i, 0), y1: at(i, 1), x2: at(i, 2), y2: at(i, 3), score, cls });
    }
    // 有模型自带类名时以它为准：自定义端到端模型因此不再被误判成 COCO 80 类
    // （以前无条件 Math.max(80, …)，一个 7 类自训 e2e 模型会被贴上 'person' / 'bicycle'
    //  这些根本不属于它的名字 —— 静默产出错误标签）。
    // 没有类名时保持原启发式：官方 e2e 权重是 COCO 80 类，类别更多则以观测到的最大类号为准。
    // 用 length > 0 而不是 `??`：空数组同样应当回退，否则 numClasses 会变成 0。
    const named = this.modelNames && this.modelNames.length > 0 ? this.modelNames.length : 0;
    this.numClasses = named > 0 ? named : Math.max(80, maxCls + 1);
    this.layoutInfo =
      `端到端 NMS-Free [${count}, 6]，不再做 NMS` +
      (named > 0 ? '' : '（未读到模型类名，按 COCO 80 类推断）');
    return boxes;
  }

  /** 经典一对多头输出：自己做 NMS */
  private classic(
    data: Float32Array,
    count: number,
    M: number,
    channelFirst: boolean,
    confThr: number,
    shapeText: string,
  ): RawBox[] {
    const at = (i: number, c: number): number =>
      channelFirst ? data[c * count + i] : data[i * M + c];

    // v5 老格式 85 通道（第 5 列是 objness），v8/v11/26 经典头直接 4+nc
    const objness = M === 85;
    const clsStart = objness ? 5 : 4;
    this.numClasses = M - clsStart;
    this.layoutInfo = `一对多头 ${shapeText}，${this.numClasses} 类，含 NMS`;

    const cands: RawBox[] = [];
    for (let i = 0; i < count; i += 1) {
      let best = 0;
      let bestCls = -1;
      for (let c = clsStart; c < M; c += 1) {
        const s = at(i, c);
        if (s > best) {
          best = s;
          bestCls = c - clsStart;
        }
      }
      if (objness) best *= at(i, 4);
      if (!(best >= confThr) || bestCls < 0) continue;
      const cx = at(i, 0);
      const cy = at(i, 1);
      const w = at(i, 2);
      const h = at(i, 3);
      cands.push({
        x1: cx - w / 2,
        y1: cy - h / 2,
        x2: cx + w / 2,
        y2: cy + h / 2,
        score: best,
        cls: bestCls,
      });
    }
    return nms(cands, NMS_IOU);
  }
}

interface RawBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  cls: number;
}


/** 统一成 Float32Array：int64 输出（BigInt64Array）也能正常参与数值比较 */
function toFloat32(data: Float32Array | BigInt64Array): Float32Array {
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i += 1) out[i] = Number(data[i]);
  return out;
}

function iou(a: RawBox, b: RawBox): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union =
    (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
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
