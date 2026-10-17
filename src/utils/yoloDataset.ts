/**
 * YOLO 数据集标注解析。
 *
 * 解决的问题：以前拖入一个文件夹时，只有图片会被导入，
 * `labels/*.txt`、`classes.txt`、`data.yaml` 这些**已有的标注被静默丢弃**，
 * 用户看到的就是「图进来了，但框全没了」——已标注的数据集等于白导入。
 *
 * 支持的标注来源（按可靠性排序）：
 *  1. `classes.txt`（每行一个类别名，标准 YOLO 格式）
 *  2. `data.yaml` / `data.yml` 里的 `names`（行内数组、`- a` 列表、`0: a` 映射三种写法都认）
 *  3. 都找不到时，按标注文件里出现过的最大类别号补上 `类别0/类别1…` 占位
 *
 * 标注行格式（归一化中心点）：`cls cx cy w h`
 */

export interface NormalizedBox {
  classIndex: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
}

export interface DatasetTextFile {
  /** 相对路径，如 `labels/a.txt` */
  path: string;
  file: File;
}

/** 取不含扩展名的小写基名：图片 `images/A.JPG` 与标签 `labels/a.txt` 归到同一 key */
export function baseKey(name: string): string {
  const i = name.lastIndexOf('.');
  return (i > 0 ? name.slice(0, i) : name).toLowerCase();
}

function fileNameOf(path: string): string {
  const i = path.lastIndexOf('/');
  return i >= 0 ? path.slice(i + 1) : path;
}

function stripComment(line: string): string {
  const h = line.indexOf('#');
  return h >= 0 ? line.slice(0, h) : line;
}

/** 解析 classes.txt：每行一个类别名 */
export function parseClassesTxt(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((l) => stripComment(l).trim())
    .filter((l) => l.length > 0);
}

/** 解析 data.yaml 里的 names，兼容三种常见写法 */
export function parseDataYamlNames(text: string): string[] {
  const lines = text.split(/\r?\n/);
  const start = lines.findIndex((l) => /^\s*names\s*:/.test(l));
  if (start < 0) return [];

  const first = lines[start].replace(/^\s*names\s*:/, '').trim();

  // 写法一：names: [a, b, c]
  if (first.startsWith('[')) {
    const inner = first.replace(/^\[/, '').replace(/\]\s*$/, '');
    return inner
      .split(',')
      .map((s) => s.trim().replace(/^['"]|['"]$/g, ''))
      .filter((s) => s.length > 0);
  }
  // 写法二：names: a, b, c
  if (first.length > 0) {
    return first
      .split(',')
      .map((s) => s.trim().replace(/^['"]|['"]$/g, ''))
      .filter((s) => s.length > 0);
  }

  // 写法三：缩进列表 `- a` 或映射 `0: a`
  // 映射写法用 Map 收集再按 index 归位：YAML 允许键不连续（0, 2, 5），
  // 直接 `out[idx] = …` 会产生稀疏数组，后面 filter 会跳过空位洞并把数组左移，
  // 导致类别号整体错位（原本 1 号的槽位被 2 号的类别名占据）。
  const out: string[] = [];
  const mapped = new Map<number, string>();
  for (let i = start + 1; i < lines.length; i += 1) {
    const raw = lines[i];
    if (raw.trim().length === 0) continue;
    if (!/^\s/.test(raw)) break; // 缩进结束 → names 块结束
    const body = stripComment(raw).trim();
    const dash = /^-\s*(.+)$/.exec(body);
    if (dash) {
      out.push(dash[1].trim().replace(/^['"]|['"]$/g, ''));
      continue;
    }
    const kv = /^(\d+)\s*:\s*(.+)$/.exec(body);
    if (kv) mapped.set(Number(kv[1]), kv[2].trim().replace(/^['"]|['"]$/g, ''));
  }
  if (mapped.size > 0) {
    const max = Math.max(...mapped.keys());
    const list: string[] = [];
    for (let i = 0; i <= max; i += 1) list.push(mapped.get(i) ?? '');
    // 洞位保留空串（不能删）：删掉会让后面所有类别号前移。
    // 空串会在 importYoloDataset 里被跳过（labelIdByIndex[i] 留空 → 该框算未分类）。
    return list;
  }
  return out.filter((s) => s.length > 0);
}

/**
 * 解析单个标注文件。脏数据一律跳过而不是让整份文件失效。
 * 只取前 5 列（分割/姿态模型的多余列会被忽略，按外接框近似处理）。
 */
export function parseLabelTxt(text: string): NormalizedBox[] {
  const out: NormalizedBox[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = stripComment(raw).trim();
    if (!line) continue;
    const parts = line.split(/\s+/);
    if (parts.length < 5) continue;
    const nums = parts.slice(0, 5).map(Number);
    if (nums.some((n) => !Number.isFinite(n))) continue;
    const [cls, cx, cy, w, h] = nums;
    if (!Number.isInteger(cls) || cls < 0) continue;
    // 中心点允许轻微越界（部分工具会写出 1.0000001），尺寸必须为正
    if (!(w > 0) || !(h > 0)) continue;
    out.push({ classIndex: cls, cx, cy, w: Math.min(1, w), h: Math.min(1, h) });
  }
  return out;
}

export interface ParsedYoloDataset {
  /** 类别名，按类别号顺序；找不到类别表时按最大类别号补占位名 */
  classNames: string[];
  /** 图片基名（小写、不含扩展名）→ 归一化标注；歧义 key 已被剔除 */
  anns: Map<string, NormalizedBox[]>;
  /** 实际解析出标注的图片数 */
  labeledImages: number;
  /**
   * 因「同一个基名有多份内容不同的标注文件」而被剔除的 key。
   * 调用方应据此提示用户，避免用户以为标注都读进来了。
   */
  ambiguous: string[];
}

/** 两份标注是否逐条相同（用于判断「同一份文件被拖了两遍」而不是真同名冲突） */
function isSameBoxes(a: NormalizedBox[], b: NormalizedBox[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i];
    const y = b[i];
    if (
      x.classIndex !== y.classIndex ||
      x.cx !== y.cx ||
      x.cy !== y.cy ||
      x.w !== y.w ||
      x.h !== y.h
    ) {
      return false;
    }
  }
  return true;
}

/** 读文本失败（编码异常等）时返回空串，不让单个文件拖垮整次导入 */
async function readTextSafe(file: File): Promise<string> {
  try {
    return await file.text();
  } catch {
    return '';
  }
}

/**
 * 把拖进来的一堆文件解析成「类别表 + 每张图的标注」。
 * 类别文件与标注文件都按**文件名**判断，不依赖目录结构，
 * 因此 `images/`+`labels/`、或图片与 txt 混在同一层，都能对上。
 */
export async function parseYoloDataset(
  texts: DatasetTextFile[],
): Promise<ParsedYoloDataset> {
  // 一次性读完，避免同一个文件被读两遍
  const loaded: Array<{ path: string; name: string; text: string }> = [];
  for (const { path, file } of texts) {
    loaded.push({ path, name: fileNameOf(path).toLowerCase(), text: await readTextSafe(file) });
  }

  const anns = new Map<string, NormalizedBox[]>();
  /**
   * 同一个基名对应了**多份**标注文件的 key —— 说明数据集中存在同名图（例如
   * `train/labels/x.txt` 与 `val/labels/x.txt`，或 `a/x.jpg` 与 `b/x.jpg`）。
   * 这种情况下按基名匹配会把 A 的框套到 B 上，属于静默产出错误训练标签，
   * 比不导入更糟。所以这里把这些 key 标成歧义，交由调用方跳过并如实告知用户。
   * 要彻底支持这种数据集，需要把图片的相对目录也一起带进匹配（当前 ImageItem 只有文件名）。
   */
  const ambiguous = new Set<string>();
  let maxClass = -1;
  let labeledImages = 0;

  for (const { path, name, text } of loaded) {
    // classes.txt / *.names 是类别表，不是标注
    const isClassList = name === 'classes.txt' || name.endsWith('.names');
    if (!name.endsWith('.txt') || isClassList) continue;

    const key = baseKey(name);
    if (key === 'classes' || key === 'data' || key === 'readme' || key === 'notes') continue;

    const boxes = parseLabelTxt(text);
    if (boxes.length === 0) continue;

    // 同一份标注被拖了两遍（路径不同）不算歧义，内容一样就沿用先来的那份
    const prev = anns.get(key);
    if (prev) {
      if (!isSameBoxes(prev, boxes)) ambiguous.add(key);
      continue;
    }

    anns.set(key, boxes);
    labeledImages += 1;
    void path;
    for (const b of boxes) if (b.classIndex > maxClass) maxClass = b.classIndex;
  }
  // 歧义的 key 直接从匹配表里摘掉：宁可这张图没有框，也不能配上别人的框
  for (const key of ambiguous) anns.delete(key);

  // 类别表：优先 classes.txt，其次 data.yaml
  let classNames: string[] = [];
  for (const item of loaded) {
    if (item.name === 'classes.txt' || item.name.endsWith('.names')) {
      classNames = parseClassesTxt(item.text);
      if (classNames.length > 0) break;
    }
  }
  if (classNames.length === 0) {
    for (const item of loaded) {
      if (item.name.endsWith('.yaml') || item.name.endsWith('.yml')) {
        classNames = parseDataYamlNames(item.text);
        if (classNames.length > 0) break;
      }
    }
  }
  // 类别表比实际用到的类别号短时补占位，避免框变成「未分类」
  for (let i = classNames.length; i <= maxClass; i += 1) classNames.push(`类别${i}`);

  return { classNames, anns, labeledImages, ambiguous: [...ambiguous] };
}
