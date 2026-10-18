/**
 * 极简 ONNX 解析：只做两件事 —— 读出「第一个输入张量的维度」，以及「模型自带的元数据」。
 *
 * 为什么需要读形状（这是 AI 识别全军覆没的主要原因）：
 *  - 同一个「YOLO26」，导出时 `imgsz` 不同，输入尺寸就不同：
 *    实测本项目的 `yolo26n.onnx` 是 `[1,3,640,640]`，而官方文档里 YOLO26 又出现过 768。
 *    **所以输入尺寸必须从模型本身读，绝不能写死、也不能猜**；
 *  - onnxruntime-web 不同版本/不同 EP（WebGPU / WASM）暴露的 `inputMetadata` 形态并不一致，
 *    拿不到时就会退化成「猜一个 640」；
 *  - 把 640×640 喂给固定 768×768 的模型，会直接 shape mismatch；即使模型是动态轴，
 *    与训练分辨率不一致也会掉精度。
 *
 * 为什么需要读元数据（`readOnnxMetadata`）：
 *  - Ultralytics 导出时会把训练信息写进模型，其中 `names` 就是**类别名**（自定义数据集
 *    的中文类名也在里面）、`task` 是任务类型、`imgsz` 是训练分辨率。
 *    也就是说「类别只能叫 类别 1 / 类别 2」这个限制根本不存在 —— 名字本来就在文件里，
 *    只是以前没人去读。
 *
 * ONNX 是 protobuf，这里只实现够用的那几条规则，不引入任何依赖。
 * 字段号（来自 onnx.proto）：
 *   ModelProto.graph = 7
 *   ModelProto.metadata_props = 14（repeated StringStringEntryProto）
 *   StringStringEntryProto.key = 1, .value = 2
 *   GraphProto.input = 11
 *   ValueInfoProto.type = 2
 *   TypeProto.tensor_type = 1
 *   TypeProto.Tensor.shape = 2
 *   TensorShapeProto.dim = 1（repeated Dimension）
 *   Dimension.dim_value = 1（varint，静态维）/ dim_param = 2（string，动态维）
 */

/** 读一个 base-128 varint，返回 [值, 结束位置]。用乘法而非位运算，避免 32 位截断 */
function readVarint(buf: Uint8Array, pos: number): [number, number] {
  let result = 0;
  let shift = 1;
  let p = pos;
  while (p < buf.length) {
    const b = buf[p];
    p += 1;
    result += (b & 0x7f) * shift;
    if ((b & 0x80) === 0) return [result, p];
    shift *= 128;
    if (shift > Number.MAX_SAFE_INTEGER) break;
  }
  throw new Error('varint 格式错误');
}

/** UTF-8 解码器（类名可能是中文，必须按 UTF-8 解而不是逐字节拼） */
let utf8: TextDecoder | null = null;
function decodeUtf8(buf: Uint8Array, start: number, end: number): string {
  if (!utf8) utf8 = new TextDecoder('utf-8');
  return utf8.decode(buf.subarray(start, end));
}

type FieldVisitor = (
  fieldNo: number,
  wireType: number,
  start: number,
  end: number,
  varintValue: number,
) => void;

/** 遍历一段 protobuf 区间内的所有字段 */
function walk(
  buf: Uint8Array,
  start: number,
  end: number,
  visit: FieldVisitor,
): void {
  let p = start;
  while (p < end) {
    let key = 0;
    try {
      const r = readVarint(buf, p);
      key = r[0];
      p = r[1];
    } catch {
      return;
    }
    const fieldNo = Math.floor(key / 8);
    const wireType = key % 8;

    if (wireType === 0) {
      let v = 0;
      try {
        const r = readVarint(buf, p);
        v = r[0];
        p = r[1];
      } catch {
        return;
      }
      visit(fieldNo, 0, p, p, v);
    } else if (wireType === 1) {
      if (p + 8 > end) return;
      visit(fieldNo, 1, p, p + 8, 0);
      p += 8;
    } else if (wireType === 2) {
      let len = 0;
      try {
        const r = readVarint(buf, p);
        len = r[0];
        p = r[1];
      } catch {
        return;
      }
      if (len < 0 || p + len > end) return;
      visit(fieldNo, 2, p, p + len, 0);
      p += len;
    } else if (wireType === 5) {
      if (p + 4 > end) return;
      visit(fieldNo, 5, p, p + 4, 0);
      p += 4;
    } else {
      // group(3/4)：ONNX 顶层不会用到，直接停止而不是猜
      return;
    }
  }
}

/** 找出第一个匹配的「长度定界」子消息，返回 [start, end] */
function findSubMessage(
  buf: Uint8Array,
  start: number,
  end: number,
  fieldNo: number,
): [number, number] | null {
  let found: [number, number] | null = null;
  walk(buf, start, end, (no, wire, s, e) => {
    if (found) return;
    if (no === fieldNo && wire === 2) found = [s, e];
  });
  return found;
}

/** 找出所有匹配的「长度定界」子消息 */
function findSubMessages(
  buf: Uint8Array,
  start: number,
  end: number,
  fieldNo: number,
): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  walk(buf, start, end, (no, wire, s, e) => {
    if (no === fieldNo && wire === 2) out.push([s, e]);
  });
  return out;
}

/**
 * 读第一个输入的形状。动态维（dim_param）记为 -1。
 * 解析失败返回 null —— 调用方必须准备好兜底方案，绝不能因此中断模型加载。
 */
export function readOnnxInputDims(bytes: Uint8Array): number[] | null {
  try {
    const graph = findSubMessage(bytes, 0, bytes.length, 7);
    if (!graph) return null;

    for (const [vs, ve] of findSubMessages(bytes, graph[0], graph[1], 11)) {
      const type = findSubMessage(bytes, vs, ve, 2); // ValueInfoProto.type
      if (!type) continue;
      const tensorType = findSubMessage(bytes, type[0], type[1], 1); // TypeProto.tensor_type
      if (!tensorType) continue;
      const shape = findSubMessage(bytes, tensorType[0], tensorType[1], 2); // Tensor.shape
      if (!shape) continue;

      const dims: number[] = [];
      for (const [ds, de] of findSubMessages(bytes, shape[0], shape[1], 1)) {
        let value = -1;
        walk(bytes, ds, de, (no, wire, _s, _e, v) => {
          if (no === 1 && wire === 0) value = v; // dim_value：静态维
          else if (no === 2 && wire === 2) value = -1; // dim_param：动态维
        });
        dims.push(value);
      }
      if (dims.length >= 3) return dims;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * 读 ONNX 自带的 metadata_props（`ModelProto.metadata_props` = 14）。
 *
 * Ultralytics 导出时会在模型里写一批训练信息，实测（yolo26n / yolo11n / yolov8n，
 * 以及自训模型的导出）稳定包含：
 *
 *   names  = {0: 'person', 1: 'bicycle', …}      ← **类别名**，自定义数据集的中文类名同样在里面
 *   task   = detect / segment / pose / classify  ← 能提前判断「这根本不是检测模型」
 *   imgsz  = [640, 640]                          ← 训练分辨率（动态轴模型的最佳输入尺寸）
 *   stride / batch / end2end / args / author / version / description …
 *
 * **读不到时必须返回 null，绝不能抛**：被 `onnx.utils.extract_model` 或 pnnx 转换过的
 * 模型会把整段元数据剥掉（实测 `yolov10n.onnx`、`yolo26n.pnnx.onnx` 就是空的），
 * 这时调用方回退到「类别 N」，属于正常降级而不是故障。
 */
export function readOnnxMetadata(bytes: Uint8Array): Record<string, string> | null {
  try {
    const out: Record<string, string> = {};
    // 顶层字段很少（graph 是一个被长度跳过的巨型子消息），所以整文件扫一遍也很便宜
    walk(bytes, 0, bytes.length, (fieldNo, wireType, start, end) => {
      if (fieldNo !== 14 || wireType !== 2) return;
      let key = '';
      let value = '';
      walk(bytes, start, end, (no, wire, s, e) => {
        if (no === 1 && wire === 2) key = decodeUtf8(bytes, s, e);
        else if (no === 2 && wire === 2) value = decodeUtf8(bytes, s, e);
      });
      if (key) out[key] = value;
    });
    return Object.keys(out).length > 0 ? out : null;
  } catch {
    return null;
  }
}

/**
 * metadata 里的 `imgsz`：正常写作 `[640, 640]`，也可能只有一个数。
 *
 * 取值区间（32~8192）做过滤，是为了「宁可返回 null 去走兜底，也不要拿一个离谱数字
 * 当输入尺寸」——元数据里的 `args` 键塞满各种数字，口径放宽就可能误命中。
 * 返回 `[宽, 高]`，与 `Detector.inputW / inputH` 的顺序一致。
 */
export function parseImgszHint(raw: string | undefined): [number, number] | null {
  if (!raw) return null;
  const nums = (raw.match(/\d+/g) ?? [])
    .map(Number)
    .filter((n) => Number.isFinite(n) && n >= 32 && n <= 8192);
  if (nums.length === 0) return null;
  const last = nums[nums.length - 1];
  const prev = nums.length >= 2 ? nums[nums.length - 2] : last;
  return [last, prev];
}
