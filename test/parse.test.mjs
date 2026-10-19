// ONNX 形状/元数据解析 + YOLO 标注/类别解析 的回归测试
// 运行：npm run test:parse
// 原理：用 esbuild 把 TS 源码转成 .tmp/ 下的 ESM，再由 node 直接跑断言。
//       ONNX 是 protobuf，测试用手工编码的合法字节流（而不是依赖真实 .onnx 文件），
//       因此静态维、动态维、非方形、垃圾字节都能稳定覆盖。
import { readOnnxInputDims, readOnnxMetadata, parseImgszHint } from '../.tmp/ai/onnxShape.js';
import { parseClassNames, classNameFor } from '../.tmp/ai/cocoNames.js';
import { snapRectToEdges } from '../.tmp/utils/geometry.js';
import { buildAiAnnotations, iouOf, AI_DEDUP_IOU } from '../.tmp/utils/aiBoxes.js';
import { parseImageSize } from '../.tmp/utils/imageSize.js';
import { dinoLetterbox, gridSizeOf, dinoFeasibility } from '../.tmp/ai/dinoGeometry.js';
import {
  parseClassesTxt,
  parseDataYamlNames,
  parseLabelTxt,
  baseKey,
  parseYoloDataset,
} from '../.tmp/utils/yoloDataset.js';
import {
  medianOf,
  manyThreshold,
  lowConfThreshold,
  summarizeBatch,
  batchHeadline,
  MIN_STATS_SAMPLE,
} from '../.tmp/utils/vpBatch.js';

let pass = 0;
let fail = 0;
function check(name, got, want) {
  const g = JSON.stringify(got);
  const w = JSON.stringify(want);
  if (g === w) {
    pass += 1;
    console.log(`  PASS  ${name} -> ${g}`);
  } else {
    fail += 1;
    console.log(`  FAIL  ${name}\n        got  ${g}\n        want ${w}`);
  }
}

/* ---------- protobuf 编码工具 ---------- */
function varint(n) {
  const out = [];
  let v = n;
  do {
    let b = v % 128;
    v = Math.floor(v / 128);
    if (v > 0) b |= 0x80;
    out.push(b);
  } while (v > 0);
  return out;
}
function tag(field, wire) {
  return varint(field * 8 + wire);
}
function lenDelim(field, payload) {
  return [...tag(field, 2), ...varint(payload.length), ...payload];
}
function varintField(field, value) {
  return [...tag(field, 0), ...varint(value)];
}
function concat(...parts) {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

function buildOnnx(dims) {
  // Dimension 消息：dim_value=1(varint) 或 dim_param=2(string)
  // TensorShapeProto 的 dim 是 **repeated**，每个维度各自是一个 field=1 的子消息
  const dimBytes = dims.map((d) =>
    lenDelim(
      1,
      typeof d === 'string'
        ? lenDelim(2, [...new TextEncoder().encode(d)])
        : varintField(1, d),
    ),
  );
  const shape = concat(...dimBytes); // TensorShapeProto.dim = 1 (repeated)
  const tensorType = lenDelim(2, shape); // TypeProto.Tensor.shape = 2
  const type = lenDelim(1, tensorType); // TypeProto.tensor_type = 1
  const valueInfo = lenDelim(2, type); // ValueInfoProto.type = 2
  const graph = lenDelim(11, valueInfo); // GraphProto.input = 11
  return concat(lenDelim(7, graph)); // ModelProto.graph = 7
}

console.log('ONNX 形状解析：');
check('静态 640 (YOLOv8)', readOnnxInputDims(buildOnnx([1, 3, 640, 640])), [1, 3, 640, 640]);
check('静态 768 (YOLO26 默认)', readOnnxInputDims(buildOnnx([1, 3, 768, 768])), [1, 3, 768, 768]);
check('非方形 960x544', readOnnxInputDims(buildOnnx([1, 3, 544, 960])), [1, 3, 544, 960]);
check('动态轴', readOnnxInputDims(buildOnnx([1, 3, 'height', 'width'])), [1, 3, -1, -1]);
check('非 ONNX 垃圾字节', readOnnxInputDims(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])), null);
check('空文件', readOnnxInputDims(new Uint8Array(0)), null);

/* ---------- ONNX metadata_props（类名的来源） ----------
   ModelProto.metadata_props = 14（repeated StringStringEntryProto）
   StringStringEntryProto.key = 1, .value = 2
   Ultralytics 导出时把 names / task / imgsz 写在这里；被 extract_model 或 pnnx
   处理过的模型这段会整体消失，所以「读不到」必须是正常返回而不是抛错。 */
function utf8Field(field, s) {
  return lenDelim(field, [...new TextEncoder().encode(s)]);
}
/** 注意返回 Uint8Array（不是普通数组）：解析器要按字节区间 subarray，数组没有这个方法 */
function metaEntry(key, value) {
  return concat(lenDelim(14, concat(utf8Field(1, key), utf8Field(2, value))));
}

console.log('\nONNX 元数据解析：');
check(
  '读到 names / task / imgsz',
  readOnnxMetadata(
    concat(
      metaEntry('task', 'detect'),
      metaEntry('imgsz', '[640, 640]'),
      metaEntry('names', "{0: 'person', 1: 'bicycle'}"),
    ),
  ),
  { task: 'detect', imgsz: '[640, 640]', names: "{0: 'person', 1: 'bicycle'}" },
);
check(
  '中文类名按 UTF-8 正确解出',
  readOnnxMetadata(metaEntry('names', "{0: '黄-半', 1: '道具'}"))?.names,
  "{0: '黄-半', 1: '道具'}",
);
check('没有 metadata_props → null', readOnnxMetadata(buildOnnx([1, 3, 640, 640])), null);
check('空文件 → null', readOnnxMetadata(new Uint8Array(0)), null);
check('垃圾字节 → null', readOnnxMetadata(new Uint8Array([0xff, 0xff, 0xff, 0xff])), null);
check(
  '元数据与 graph 共存时形状仍能读出',
  readOnnxInputDims(concat(buildOnnx([1, 3, 768, 768]), metaEntry('task', 'detect'))),
  [1, 3, 768, 768],
);

console.log('\n类别名解析（parseClassNames）：');
check('Python dict', parseClassNames("{0: 'person', 1: 'bicycle'}"), ['person', 'bicycle']);
check('中文类名', parseClassNames("{0: '黄-半', 1: '黄-头', 2: '道具'}"), [
  '黄-半',
  '黄-头',
  '道具',
]);
// 索引不连续时必须保留洞位：以前 data.yaml 解析就踩过「跳过空位导致类别号整体错位」
check('索引有洞 → 洞位留空串，不左移', parseClassNames("{0: 'a', 2: 'c'}"), ['a', '', 'c']);
check('单引号里的转义', parseClassNames("{0: 'it\\'s'}"), ["it's"]);
check('\\u 转义', parseClassNames("{0: '\\u9ec4\\u534a'}"), ['黄半']);
check('JSON 对象', parseClassNames('{"0": "cat", "1": "dog"}'), ['cat', 'dog']);
check('JSON 数组', parseClassNames('["cat", "dog"]'), ['cat', 'dog']);
check('Python 列表', parseClassNames("['cat', 'dog']"), ['cat', 'dog']);
check('空串 → null', parseClassNames(''), null);
check('undefined → null', parseClassNames(undefined), null);
check('空 dict → null', parseClassNames('{}'), null);
check('非类别文本 → null', parseClassNames('hello'), null);

console.log('\nimgsz 解析（parseImgszHint）：');
check('[640, 640]', parseImgszHint('[640, 640]'), [640, 640]);
check('训练用小图 [256, 256]', parseImgszHint('[256, 256]'), [256, 256]);
check('非方形 [640, 480]', parseImgszHint('[640, 480]'), [480, 640]);
check('只写一个数', parseImgszHint('640'), [640, 640]);
check('空串 → null', parseImgszHint(''), null);
check('undefined → null', parseImgszHint(undefined), null);
check('无数字 → null', parseImgszHint('imgsz'), null);
// 区间过滤：离谱值宁可当没读到，也不要拿去当输入尺寸
check('过小被过滤 → null', parseImgszHint('[0, 8]'), null);
check('过大被过滤 → null', parseImgszHint('[99999]'), null);

console.log('\n类别显示名（classNameFor）：');
const MODEL_NAMES = ['黄-半', '黄-头', '道具'];
check('模型自带类名优先', classNameFor(1, 3, MODEL_NAMES), '黄-头');
check('模型类名有洞位时回退索引名', classNameFor(1, 3, ['a', '', 'c']), '类别 2');
check('无类名 + 80 类 → COCO 表', classNameFor(0, 80, null), 'person');
check('无类名 + 非 80 类 → 索引名', classNameFor(0, 7, null), '类别 1');
check('类名数组越界 → 索引名', classNameFor(9, 3, MODEL_NAMES), '类别 10');
check('无类名 + 80 类越界 → 索引名', classNameFor(99, 80, null), '类别 100');

// 用户在面板「类别名」里自己改的名字（第 4 个参数）优先级最高。
// 模型元数据被 extract_model / pnnx 剥掉时，这是唯一能拿到「人话类名」的路。
const OVERRIDES = ['', '黄-头（改）', ''];
check('用户改的名字优先于模型类名', classNameFor(1, 3, MODEL_NAMES, OVERRIDES), '黄-头（改）');
check('该项没改（空串）→ 回退模型类名', classNameFor(0, 3, MODEL_NAMES, OVERRIDES), '黄-半');
check('模型没类名时用户名字照样生效', classNameFor(2, 7, null, ['', '', '道具']), '道具');
check('用户清空输入框 → 回退索引名', classNameFor(2, 7, null, ['', '', '']), '类别 3');
check('overrides 比类别数短 → 缺的走回退', classNameFor(5, 7, null, ['仅第一个']), '类别 6');
check('overrides 传 undefined 时行为不变', classNameFor(0, 80, null, undefined), 'person');
check(
  '用户名字也压过 COCO 表',
  classNameFor(0, 80, null, ['我的车']),
  '我的车',
);

console.log('\nYOLO 标注解析：');
check(
  '标准两行',
  parseLabelTxt('0 0.5 0.5 0.2 0.4\n1 0.25 0.75 0.1 0.1'),
  [
    { classIndex: 0, cx: 0.5, cy: 0.5, w: 0.2, h: 0.4 },
    { classIndex: 1, cx: 0.25, cy: 0.75, w: 0.1, h: 0.1 },
  ],
);
check('空行与注释被跳过', parseLabelTxt('\n# comment\n0 0.5 0.5 0.2 0.4\n\n').length, 1);
check('字段不足被跳过', parseLabelTxt('0 0.5 0.5\n1 0.1 0.1 0.1 0.1').length, 1);
check('非数字被跳过', parseLabelTxt('abc 0.5 0.5 0.2 0.4\n0 0.5 0.5 0.2 0.4').length, 1);
check('负类别号被跳过', parseLabelTxt('-1 0.5 0.5 0.2 0.4\n0 0.5 0.5 0.2 0.4').length, 1);
check('零尺寸被跳过', parseLabelTxt('0 0.5 0.5 0 0.4\n0 0.5 0.5 0.2 0.4').length, 1);
check('超出 1 的宽被收敛到 1', parseLabelTxt('0 0.5 0.5 1.4 0.4')[0].w, 1);
check('多列（分割）只取前 5 列', parseLabelTxt('0 0.5 0.5 0.2 0.4 0.1 0.2 0.3').length, 1);
check('空文本', parseLabelTxt(''), []);

console.log('\n类别表解析：');
check('classes.txt', parseClassesTxt('cat\ndog\n\nperson\n'), ['cat', 'dog', 'person']);
check('classes.txt 去注释', parseClassesTxt('cat # 猫\ndog'), ['cat', 'dog']);
check(
  'data.yaml 行内数组',
  parseDataYamlNames('path: ../x\nnc: 2\nnames: [\'cat\', \'dog\']\n'),
  ['cat', 'dog'],
);
check(
  'data.yaml 短横线列表',
  parseDataYamlNames('names:\n  - cat\n  - dog\nval: images/val\n'),
  ['cat', 'dog'],
);
check(
  'data.yaml 映射写法',
  parseDataYamlNames('names:\n  0: cat\n  1: dog\nnc: 2\n'),
  ['cat', 'dog'],
);
check('data.yaml 无 names', parseDataYamlNames('path: x\ntrain: images\n'), []);
// 回归：YAML 允许映射键不连续。以前 `out[idx]=…` 产生稀疏数组，
// 后面 filter 跳过空位洞并把数组左移 → 类别号整体错位（1 号槽位被 2 号的名字占了）。
// 现在洞位保留空串，空串在 importYoloDataset 里会被跳过（该类框算未分类），绝不移位。
check(
  'data.yaml 映射索引不连续（洞位保留，不左移）',
  parseDataYamlNames('names:\n  0: cat\n  2: dog\n  3: bird\n'),
  ['cat', '', 'dog', 'bird'],
);
check(
  'data.yaml 映射索引不连续（首项非 0）',
  parseDataYamlNames('names:\n  1: cat\n  2: dog\n'),
  ['', 'cat', 'dog'],
);

console.log('\n基名匹配：');
check('图片大写扩展名', baseKey('A.JPG'), 'a');
check('标签同基名', baseKey('a.txt'), 'a');
check('多点文件名', baseKey('my.data.v2.png'), 'my.data.v2');
check('无扩展名', baseKey('noext'), 'noext');

/* ---------- 同名标注文件 → 歧义判定（B5 回归） ----------
   数据集里存在 a/x.jpg 与 b/x.jpg（基名相同）时，按基名匹配会把其中一份的框
   套到另一份上 —— 静默产出错误训练标签，比不导入更糟。
   现在：同一基名对应**内容不同**的多份标注文件 → 标为歧义、从匹配表剔除、如实上报。
   同一份文件被拖两遍（内容完全相同）→ 不算歧义，沿用先来那份。            */
console.log('\n同名标注文件的歧义判定：');

/** 造一个最小 File 替身：只需要 .name 与 .text() */
function fakeFile(name, text) {
  return { name, text: async () => text };
}

const dupDiff = await parseYoloDataset([
  { path: 'train/labels/x.txt', file: fakeFile('x.txt', '0 0.5 0.5 0.2 0.2\n') },
  { path: 'val/labels/x.txt', file: fakeFile('x.txt', '1 0.3 0.3 0.4 0.4\n') },
]);
check('同名但内容不同 → 剔除该基名', dupDiff.anns.has('x'), false);
check('同名但内容不同 → 记为歧义', dupDiff.ambiguous, ['x']);

const dupSame = await parseYoloDataset([
  { path: 'labels/x.txt', file: fakeFile('x.txt', '0 0.5 0.5 0.2 0.2\n') },
  { path: 'backup/x.txt', file: fakeFile('x.txt', '0 0.5 0.5 0.2 0.2\n') },
]);
check('同名且内容相同 → 不算歧义', dupSame.ambiguous, []);
check('同名且内容相同 → 保留标注', dupSame.anns.get('x')?.length ?? null, 1);

const distinct = await parseYoloDataset([
  { path: 'labels/a.txt', file: fakeFile('a.txt', '0 0.5 0.5 0.2 0.2\n') },
  { path: 'labels/b.txt', file: fakeFile('b.txt', '1 0.3 0.3 0.4 0.4\n') },
  { path: 'classes.txt', file: fakeFile('classes.txt', 'cat\ndog\n') },
]);
check('不同基名 → 正常匹配两个', distinct.anns.size, 2);
check('classes.txt 被当类别表而非标注', distinct.classNames, ['cat', 'dog']);
check('无歧义时 ambiguous 为空', distinct.ambiguous, []);

/* ---------- 图片尺寸：从文件头直接读，不解码 ----------
   缩略图降采样、以及「导出前补齐尺寸」都依赖它。读错尺寸是**静默错误**：
   要么降采样目标算错，要么导出的归一化坐标用错宽高。
   EXIF 方向那一条尤其关键 —— 它保证这里读到的尺寸和浏览器实际显示的尺寸一致。 */
console.log('\n图片尺寸（文件头解析）：');

const bytes = (...parts) => Uint8Array.from(parts.flat(Infinity));
const str = (s) => [...s].map((c) => c.charCodeAt(0));
const be16w = (n) => [(n >> 8) & 255, n & 255];
const le16w = (n) => [n & 255, (n >> 8) & 255];
const le24w = (n) => [n & 255, (n >> 8) & 255, (n >> 16) & 255];
const le32w = (n) => [n & 255, (n >> 8) & 255, (n >> 16) & 255, (n >> 24) & 255];

check(
  'PNG 宽高（大端）',
  parseImageSize(
    bytes([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], le32w(13), str('IHDR'), [
      0, 0, 0x02, 0x80, 0, 0, 0x01, 0x90,
    ]),
  ),
  { width: 640, height: 400 },
);

check(
  'PNG 宽为 0 → null',
  parseImageSize(
    bytes([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], le32w(13), str('IHDR'), [
      0, 0, 0, 0, 0, 0, 0x01, 0x90,
    ]),
  ),
  null,
);

check(
  'GIF 宽高（小端）',
  parseImageSize(bytes(str('GIF89a'), le16w(320), le16w(300))),
  { width: 320, height: 300 },
);

// BMP：偏移 14 是 DIB 头大小；宽高是有符号 int32（高为负 = top-down）
const bmpHead = [
  ...str('BM'),
  ...Array.from({ length: 12 }, () => 0),
  ...le32w(40), // BITMAPINFOHEADER
  ...le32w(1920),
  ...le32w(1080),
];
check('BMP 宽高（有符号小端）', parseImageSize(bytes(bmpHead)), { width: 1920, height: 1080 });
check(
  'BMP 负高度取绝对值（top-down）',
  parseImageSize(bytes([...bmpHead.slice(0, 22), ...le32w(-1080)])),
  { width: 1920, height: 1080 },
);

// WebP 三种变体：扩展(VP8X) / 无损(VP8L) / 有损(VP8 )
const webpPrefix = [...str('RIFF'), ...le32w(30), ...str('WEBP')];
check(
  'WebP(VP8X) 宽高（存的是值-1）',
  parseImageSize(
    bytes(webpPrefix, str('VP8X'), le32w(10), 0, 0, 0, 0, le24w(639), le24w(479)),
  ),
  { width: 640, height: 480 },
);
check(
  'WebP(VP8L) 宽高（14 位打包）',
  parseImageSize(bytes(webpPrefix, str('VP8L'), le32w(10), 0x2f, 0x7f, 0xc2, 0x77, 0x00, 0, 0, 0, 0, 0)),
  { width: 640, height: 480 },
);
check(
  'WebP(VP8 ) 宽高（起始码后 14 位）',
  parseImageSize(
    bytes(
      webpPrefix,
      str('VP8 '),
      le32w(10),
      le24w(0),
      0x9d, 0x01, 0x2a,
      le16w(640),
      le16w(480),
      le16w(0),
    ),
  ),
  { width: 640, height: 480 },
);

// JPEG：SOF0 里的高/宽
const sof0 = [0xff, 0xc0, ...be16w(17), 0x08, ...be16w(400), ...be16w(640), 0x03, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0];
check('JPEG SOF 宽高', parseImageSize(bytes([0xff, 0xd8], sof0)), { width: 640, height: 400 });

// JPEG + EXIF Orientation（6 表示旋转 90°，宽高必须互换才能对上浏览器显示尺寸）
const exifApp1With = (orientation) => [
  0xff, 0xe1, ...be16w(34),
  ...str('Exif'), 0x00, 0x00,
  0x49, 0x49, 0x2a, 0x00, ...le32w(8), // TIFF 头（小端），IFD0 在偏移 8
  ...le16w(1), // 1 个条目
  0x12, 0x01, 0x03, 0x00, ...le32w(1), ...le32w(orientation), // 0x0112 Orientation
  ...le32w(0), // 无下一个 IFD
];
check(
  'JPEG + EXIF 方向 6 → 宽高互换',
  parseImageSize(bytes([0xff, 0xd8], exifApp1With(6), sof0)),
  { width: 400, height: 640 },
);
check(
  'JPEG + EXIF 方向 1 → 不互换',
  parseImageSize(bytes([0xff, 0xd8], exifApp1With(1), sof0)),
  { width: 640, height: 400 },
);

check('垃圾字节 → null', parseImageSize(bytes(Array.from({ length: 32 }, () => 0xab))), null);
check('过短 → null', parseImageSize(bytes([0x89, 0x50])), null);

/* ---------- 边缘吸附不得把框压成 0 宽/0 高 ----------
   吸附阈值大于最小框尺寸时，左右边会被吸附到同一位置 → 宽度 0 →
   画布不绘制、YOLO 导出 0.000000。现在吸附后做收口，至少留 1px。 */
console.log('\n边缘吸附收口：');
const TOL = 14; // 相当于 7 屏幕px / scale 0.5
check(
  '窄框贴右边界不再塌缩为 0 宽',
  snapRectToEdges({ x: 1912, y: 100, width: 8, height: 40 }, 1920, 1080, TOL).width,
  1,
);
check(
  '窄框贴右边界时 x 仍在图内',
  (() => {
    const r = snapRectToEdges({ x: 1912, y: 100, width: 8, height: 40 }, 1920, 1080, TOL);
    return r.x >= 0 && r.x + r.width <= 1920 ? 'in' : 'out';
  })(),
  'in',
);
check(
  '窄框贴下边界不再塌缩为 0 高',
  snapRectToEdges({ x: 100, y: 1072, width: 40, height: 8 }, 1920, 1080, TOL).height,
  1,
);
check(
  '正常框吸附到边界后仍是原尺寸',
  (() => {
    const r = snapRectToEdges({ x: 1800, y: 1000, width: 100, height: 60 }, 1920, 1080, TOL);
    return `${r.width}x${r.height}`;
  })(),
  '100x60',
);
check(
  '远离边界时不做任何改动',
  (() => {
    const r = snapRectToEdges({ x: 500, y: 400, width: 100, height: 60 }, 1920, 1080, TOL);
    return `${r.x},${r.y},${r.width},${r.height}`;
  })(),
  '500,400,100,60',
);
check(
  '贴左边界的小框保留 1px',
  snapRectToEdges({ x: 3, y: 100, width: 8, height: 40 }, 1920, 1080, TOL).width,
  1,
);

/* ---------- AI 落框：重复框治理 ----------
   这条规则直接影响导出数据集的正确性：
   「调低阈值再跑一遍」必然在同一张图上产出与已有框几乎重合的新框，
   不拦的话每个物体挂两个框（是两份真值，而不是"多一个框要删"这么轻）。 */
console.log('\nAI 落框与去重：');
const rect = (x, y, w, h) => ({ x, y, width: w, height: h });
const CAT = { id: 'lb_cat', name: 'cat', color: '#000000' };
/** 跑一次落框。labels 可预置，用来构造「已有同类框」的场景 */
const runAi = (existing, entries, preset = [], w = 100, h = 100) => {
  const labels = preset.slice();
  const nameToId = new Map(labels.map((l) => [l.name, l.id]));
  const res = buildAiAnnotations({
    existing,
    entries,
    labels,
    nameToId,
    width: w,
    height: h,
    createLabel: (name, index) => ({ id: `lb_new${index}`, name, color: '#111111' }),
    createId: (prefix) => `${prefix}fixed`,
  });
  return { ...res, labels, nameToId };
};

check('IoU：完全重合 = 1', iouOf(rect(0, 0, 10, 10), rect(0, 0, 10, 10)), 1);
check('IoU：不相交 = 0', iouOf(rect(0, 0, 10, 10), rect(20, 20, 10, 10)), 0);
check('IoU：仅边界相接 = 0', iouOf(rect(0, 0, 10, 10), rect(10, 0, 10, 10)), 0);
check('IoU：三分之一重叠', Math.round(iouOf(rect(0, 0, 10, 10), rect(5, 0, 10, 10)) * 1e6) / 1e6, 0.333333);

const r1 = runAi(
  [],
  [
    { rect: rect(0, 0, 10, 10), className: 'cat', score: 0.9 },
    { rect: rect(40, 40, 10, 10), className: 'cat', score: 0.8 },
  ],
  [CAT],
);
check('两个同类框都落下', r1.anns.length, 2);
check('同类名不重复建标签', r1.labels.length, 1);
check('标注带来源 ai', r1.anns[0].source, 'ai');
check('标注带置信度', r1.anns[0].score, 0.9);
check(
  '置信度缺失时不编造',
  runAi([], [{ rect: rect(0, 0, 10, 10), className: 'cat' }], [CAT]).anns[0].score ?? 'none',
  'none',
);

// 与已有框同类别且高度重叠 → 跳过（Iou 恰好等于阈值也拦）
const dup60 = runAi(
  [{ id: 'a1', labelId: 'lb_cat', rect: rect(0, 0, 10, 10) }],
  [{ rect: rect(2.5, 0, 10, 10), className: 'cat' }],
  [CAT],
);
check(`IoU 恰为 ${AI_DEDUP_IOU} → 跳过`, [dup60.anns.length, dup60.skipped], [0, 1]);
const nearDup = runAi(
  [{ id: 'a1', labelId: 'lb_cat', rect: rect(0, 0, 10, 10) }],
  [{ rect: rect(7, 0, 10, 10), className: 'cat' }],
  [CAT],
);
check('IoU 低于阈值 → 保留', [nearDup.anns.length, nearDup.skipped], [1, 0]);
// 不同类别高度重叠是正常情况（人骑在车上），绝不能被当成重复
const crossClass = runAi(
  [{ id: 'a1', labelId: 'lb_cat', rect: rect(0, 0, 10, 10) }],
  [{ rect: rect(0, 0, 10, 10), className: 'dog' }],
  [CAT],
);
check('不同类别重合 → 不跳过', [crossClass.anns.length, crossClass.skipped], [1, 0]);

// 同一批里重复（模型给了两个几乎一样的框）也要拦
const inBatch = runAi(
  [],
  [
    { rect: rect(0, 0, 10, 10), className: 'cat' },
    { rect: rect(0, 0, 10, 10), className: 'cat' },
  ],
  [CAT],
);
check('同批内重复 → 只留一个', [inBatch.anns.length, inBatch.skipped], [1, 1]);

check(
  '退化框（<2px）直接丢弃',
  runAi([], [{ rect: rect(0, 0, 1, 40), className: 'cat' }], [CAT]).anns.length,
  0,
);
check(
  '超出图片的框被夹回图内',
  (() => {
    const r = runAi([], [{ rect: rect(95, 95, 20, 20), className: 'cat' }], [CAT], 100, 100)
      .anns[0].rect;
    return r.x + r.width <= 100 && r.y + r.height <= 100 ? 'in' : 'out';
  })(),
  'in',
);

/* ---------- DINOv2 分辨率守卫（dinoGeometry） ----------
   2026-10-19 的事故：合成基准（320x320 图、72px 精灵 → 目标占 8.3 个 patch）上
   k=1 的 F1 是 0.750，于是把 DINOv2 接进了「照样子找」。但真实截图
   （1366x768、目标 55px → 只占 1.05 个 patch）上召回是 0，还吐了 31 个乱框。
   根因全在这几个数字上，必须有回归测试钉住 —— 别再让「合成数据上的好成绩」
   悄悄外推到真实图片。 */
const LB = dinoLetterbox(1366, 768);
check(
  '1366x768 → letterbox 参数（gain/w/h/left/top）',
  [Number(LB.gain.toFixed(6)), LB.w, LB.h, LB.left, LB.top],
  [0.379209, 518, 291, 0, 113],
);
check('参考框 55px 落到网格上只有 1x1 个 patch', gridSizeOf(55, 55, LB.gain, 1), [1, 1]);

const feasSmall = dinoFeasibility(1366, 768, [rect(34, 520, 55, 55)]);
check(
  '小目标被守卫拦下（真实截图里的史莱姆）',
  [feasSmall.ok, feasSmall.pw, feasSmall.ph],
  [false, 1, 1],
);
check('守卫算出原图上的最小可用边长', Math.ceil(feasSmall.needPx), 111);
check('拒绝文案指向 YOLOE', feasSmall.reason.includes('YOLOE'), true);

check('刚好够大就放行（111px）', dinoFeasibility(1366, 768, [rect(0, 0, 111, 111)]).ok, true);
check('差一点点就拦下（110px）', dinoFeasibility(1366, 768, [rect(0, 0, 110, 110)]).ok, false);
check(
  '合成基准那种尺寸当然放行（320 图 / 72px 精灵）',
  dinoFeasibility(320, 320, [rect(40, 40, 72, 72)]).ok,
  true,
);
check(
  '多框取平均尺寸（200 与 100 → 150px，放行）',
  dinoFeasibility(1366, 768, [rect(0, 0, 200, 200), rect(0, 0, 100, 100)]).ok,
  true,
);
check(
  '尺寸未知时不报错也不放行（懒测量还没填）',
  [dinoFeasibility(0, 0, [rect(0, 0, 9, 9)]).ok, dinoFeasibility(0, 0, [rect(0, 0, 9, 9)]).reason],
  [false, ''],
);
check('没有框时不报错也不放行', dinoFeasibility(100, 100, []).ok, false);

// ==================== 批量汇总 / 异常标记 ====================
// 这一段守的是「跑完之后用户知道该看哪几张」。阈值写错不会报错，
// 只会静默地难用（要么一张都不标，要么标满误报），所以必须逐条钉住。

check('中位数：奇数个取中间', medianOf([3, 1, 2]), 2);
check('中位数：偶数个取中间两个的均值', medianOf([1, 2, 3, 4]), 2.5);
check('中位数：空数组返回 0 而不是抛异常', medianOf([]), 0);

// 「框特别多」的判定线：3 倍中位数，但有 4 的下限 ——
// 稀疏目标（中位数 1~2）的图里多找出一个常常是真的，不能当成异常
check('多框线：中位数 5 → 15', manyThreshold(5), 15);
check('多框线：中位数 2 → 6（3 倍，仍高于 4 的下限）', manyThreshold(2), 6);
check('多框线：中位数 1 → 兜底 4（3 个框不算异常）', manyThreshold(1), 4);
check('多框线：中位数 0 → 兜底 4', manyThreshold(0), 4);

check('低分线：中位分的 70%', lowConfThreshold(0.6), 0.42);

const stat = (imageId, count, meanScore = 0.6, failed = false) => ({
  imageId,
  count,
  meanScore,
  failed,
});

// 全批只有 2 张有框 → 够不上 MIN_STATS_SAMPLE，一张都不标。
// 这是故意的：2 个样本的中位数没有意义，硬标只会制造狼来了。
{
  const s = summarizeBatch([stat('a', 3), stat('b', 3)]);
  check('样本不足 3 张时不标异常', [s.withBoxes, s.flags.length], [2, 0]);
  check('样本不足时中位数仍然给出（供文案用）', s.median, 3);
}

// 门槛是硬边界：少一张不标，够数就标
check('门槛值本身是 3', MIN_STATS_SAMPLE, 3);
check(
  '恰好 3 张有框时开始标记',
  summarizeBatch([stat('a', 3), stat('b', 3), stat('c', 30)]).flags.map((f) => f.imageId),
  ['c'],
);

// 典型场景：8 张图里 6 张正常（3 个框左右）、1 张空、1 张框数爆炸、1 张解码失败。
// 注意「框数爆炸」那张也计入 withBoxes 和中位数 —— 它确实产出了框，
// 汇总必须如实反映，不能为了数字好看把它剔出去。
{
  const s = summarizeBatch([
    stat('n1', 3),
    stat('n2', 4),
    stat('n3', 3),
    stat('n4', 5),
    stat('n5', 4),
    stat('n6', 3),
    stat('empty', 0),
    stat('spam', 18),
    stat('bad', 0, 0, true),
  ]);
  check('汇总：有框张数', s.withBoxes, 7);
  check('汇总：框总数', s.total, 40);
  check('汇总：没找到的张数（不含失败）', s.empty, 1);
  check('汇总：失败张数单独计', s.failed, 1);
  check('汇总：框数中位数（含没找到的图）', s.median, 3.5);
  check('汇总：只标出异常的那一张', s.flags.map((f) => f.imageId), ['spam']);
  check('异常原因带具体倍数', s.flags[0].reason, '找到 18 个，是其他图的 5 倍 —— 多半是误报，建议重点看');
}

// 中位数为 0 时（多数图都没找到）倍数会除零 → 必须换成另一套文案，不能出现 Infinity/NaN。
// 这一条同时守一个真实缺陷：整批基本全空时，「唯独某张图找出一堆」是最强的异常信号，
// 早期版本被「样本不足 3 张」的规则挡掉，等于把最危险的情况静默放过了。
{
  const s = summarizeBatch([stat('a', 0), stat('b', 0), stat('c', 0), stat('d', 7), stat('e', 0)]);
  check('中位数为 0 时仍能标记（不被样本量规则挡住）', s.flags.map((f) => f.imageId), ['d']);
  check('中位数为 0 时倍数文案不出现 NaN/Infinity', /NaN|Infinity/.test(s.flags[0].reason), false);
}

// 低分异常：框数正常但平均分只有中位分的四成
{
  const s = summarizeBatch([stat('a', 3, 0.8), stat('b', 3, 0.75), stat('c', 4, 0.7), stat('d', 3, 0.3)]);
  check('低分异常被标出', s.flags.map((f) => [f.imageId, f.kind]), [['d', 'lowconf']]);
  check('低分原因带百分比', s.flags[0].reason, '找到 3 个，但平均分只有其他图的 41% —— 多半是勉强凑数的，建议重点看');
}

// 「没找到」不参与异常判定，但必须能在 empty 里数出来
{
  const s = summarizeBatch([stat('a', 3), stat('b', 0), stat('c', 0), stat('d', 0), stat('e', 2)]);
  check('全空不算异常标记', s.flags.length, 0);
  check('全空计入 empty', s.empty, 3);
}

// 一张图可以既多又分低，但 UI 只显示最值得看的那条 → many 优先
{
  const s = summarizeBatch([
    stat('a', 3, 0.8),
    stat('b', 3, 0.8),
    stat('c', 3, 0.8),
    stat('d', 20, 0.2),
  ]);
  check('同时命中两条时 many 优先', [s.flags.length, s.flags[0].kind], [1, 'many']);
}

check('标题：正常一批', batchHeadline(summarizeBatch([stat('a', 3), stat('b', 4), stat('c', 5)])), '3 张图、12 个框');
check('标题：全没找到', batchHeadline(summarizeBatch([stat('a', 0), stat('b', 0), stat('c', 0)])), '这批图里没找到像它的东西');
check(
  '标题：混合三种情况都要说',
  batchHeadline(summarizeBatch([stat('a', 3), stat('b', 0), stat('c', 0, 0, true), stat('d', 2)])),
  '2 张图、5 个框 · 1 张没找到 · 1 张失败',
);

console.log(`\n结果：${pass} 通过，${fail} 失败`);
if (fail > 0) process.exit(1);
