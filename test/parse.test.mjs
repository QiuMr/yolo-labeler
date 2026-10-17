// ONNX 形状解析 + YOLO 标注/类别解析 的回归测试
// 运行：npm run test:parse
// 原理：用 esbuild 把 TS 源码转成 .tmp/ 下的 ESM，再由 node 直接跑断言。
//       ONNX 是 protobuf，测试用手工编码的合法字节流（而不是依赖真实 .onnx 文件），
//       因此静态维、动态维、非方形、垃圾字节都能稳定覆盖。
import { readOnnxInputDims } from '../.tmp/ai/onnxShape.js';
import { snapRectToEdges } from '../.tmp/utils/geometry.js';
import {
  parseClassesTxt,
  parseDataYamlNames,
  parseLabelTxt,
  baseKey,
  parseYoloDataset,
} from '../.tmp/utils/yoloDataset.js';

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

console.log(`\n结果：${pass} 通过，${fail} 失败`);
if (fail > 0) process.exit(1);
