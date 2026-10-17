/**
 * 极简 ONNX 形状解析：只做一件事 —— 从模型字节里读出「第一个输入张量的维度」。
 *
 * 为什么需要它（这是 AI 识别全军覆没的主要原因）：
 *  - 同一个「YOLO26」，导出时 `imgsz` 不同，输入尺寸就不同：
 *    实测本项目的 `yolo26n.onnx` 是 `[1,3,640,640]`，而官方文档里 YOLO26 又出现过 768。
 *    **所以输入尺寸必须从模型本身读，绝不能写死、也不能猜**；
 *  - onnxruntime-web 不同版本/不同 EP（WebGPU / WASM）暴露的 `inputMetadata` 形态并不一致，
 *    拿不到时就会退化成「猜一个 640」；
 *  - 把 640×640 喂给固定 768×768 的模型，会直接 shape mismatch；即使模型是动态轴，
 *    与训练分辨率不一致也会掉精度。
 *
 * ONNX 是 protobuf，这里只实现够用的那几条规则，不引入任何依赖。
 * 字段号（来自 onnx.proto）：
 *   ModelProto.graph = 7
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
