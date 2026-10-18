/** COCO-80 类别名（按模型输出 cls 索引）。绝大多数通用检测模型都是这套 */
export const COCO_NAMES: readonly string[] = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];

/** 类别名表的上限：比这更长基本等于解析错位，宁可当没读到 */
const MAX_CLASSES = 4096;

/**
 * 从 ONNX metadata 的 `names` 字段解析类别名。
 *
 * Ultralytics 写进去的是 **Python 字面量**（单引号、整数键）：
 *
 *   {0: 'person', 1: 'bicycle'}
 *   {0: '黄-半', 1: '黄-头', 2: '道具'}    ← 自定义数据集的中文类名同样在里面
 *
 * 但导出工具不止一种，JSON 形式（`{"0": "cat"}`）与数组形式（`['cat', 'dog']`）
 * 这里一并认掉。
 *
 * 返回值**按类别号对齐**：索引不连续时洞位留空串（空串在 classNameFor 里会回退成
 * 「类别 N」）。**绝不做「跳过空位把数组左移」** —— 那会让类别号和名字整体错位，
 * 静默产出错误标签（这个教训在 `data.yaml` 的 names 解析里已经吃过一次）。
 */
export function parseClassNames(raw: string | null | undefined): string[] | null {
  if (!raw) return null;
  const text = raw.trim();
  if (!text) return null;

  // ① 标准 JSON —— 一次就能覆盖 {"0":"cat"} 与 ["cat","dog"]
  try {
    const asJson = fromParsed(JSON.parse(text));
    if (asJson) return asJson;
  } catch {
    /* 不是 JSON，继续按 Python 字面量处理 */
  }

  // ② Python dict 字面量：{0: 'a', 1: 'b'}
  if (text.startsWith('{')) {
    const pairs = new Map<number, string>();
    const re = /(\d+)\s*:\s*(['"])((?:\\.|(?!\2)[\s\S])*?)\2/g;
    for (let m = re.exec(text); m; m = re.exec(text)) {
      pairs.set(Number(m[1]), unescapeLiteral(m[3]));
    }
    if (pairs.size === 0) return null;
    return fromPairs(pairs);
  }

  // ③ Python list 字面量：['a', 'b']
  if (text.startsWith('[')) {
    const items: string[] = [];
    const re = /(['"])((?:\\.|(?!\1)[\s\S])*?)\1/g;
    for (let m = re.exec(text); m; m = re.exec(text)) {
      items.push(unescapeLiteral(m[2]));
    }
    return items.length > 0 ? items : null;
  }

  return null;
}

/** 已经是对象/数组的 JSON 形态 */
function fromParsed(value: unknown): string[] | null {
  if (Array.isArray(value)) {
    const items = value.map((v) => (typeof v === 'string' ? v : String(v)));
    return items.length > 0 && items.length <= MAX_CLASSES ? items : null;
  }
  if (value && typeof value === 'object') {
    const pairs = new Map<number, string>();
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      const idx = Number(k);
      if (!Number.isInteger(idx) || idx < 0 || typeof v !== 'string') return null;
      pairs.set(idx, v);
    }
    return pairs.size > 0 ? fromPairs(pairs) : null;
  }
  return null;
}

/** 稀疏的「类别号 → 名字」→ 按索引对齐的数组（洞位留空串） */
function fromPairs(pairs: Map<number, string>): string[] | null {
  const max = Math.max(...pairs.keys());
  if (!Number.isInteger(max) || max < 0 || max >= MAX_CLASSES) return null;
  const out = Array.from({ length: max + 1 }, () => '');
  for (const [idx, name] of pairs) out[idx] = name;
  return out;
}

/** 反转义 Python/JS 字面量里的 `\uXXXX`、`\xNN`、`\n`、`\'` 等 */
function unescapeLiteral(s: string): string {
  return s.replace(/\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|[\s\S])/g, (_all, esc: string) => {
    if (esc[0] === 'u' || esc[0] === 'x') {
      return String.fromCharCode(parseInt(esc.slice(1), 16));
    }
    if (esc === 'n') return '\n';
    if (esc === 't') return '\t';
    if (esc === 'r') return '\r';
    return esc;
  });
}

/**
 * 类别显示名。优先级从权威到兜底：
 *
 *  1. **模型自带类名**（ONNX metadata 的 `names`）—— 最权威，自定义模型的中文类名走这条；
 *  2. 拿不到类名且类别数正好是 80 时，按 COCO 表显示英文名（绝大多数通用检测模型都是这套）；
 *  3. 都没有 → `类别 N`（索引名）。
 *
 * 这个名字同时也是「采纳后」自动创建的标签名，所以能读到真名时导出的数据集
 * 里就不再是「类别 1 / 类别 2」这种无意义的名字了。
 */
export function classNameFor(
  classIndex: number,
  numClasses: number,
  modelNames?: readonly string[] | null,
): string {
  const fromModel = modelNames?.[classIndex];
  if (fromModel) return fromModel;
  if (numClasses === COCO_NAMES.length) {
    return COCO_NAMES[classIndex] ?? `类别 ${classIndex + 1}`;
  }
  return `类别 ${classIndex + 1}`;
}
