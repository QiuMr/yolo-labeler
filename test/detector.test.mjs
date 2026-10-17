// 输出布局判别的回归测试：端到端(NMS-Free) vs 经典头(含 NMS)，以及 2 类模型的歧义场景
// 运行：npm run test:detector
// 这是「YOLO26 一个框都识别不出来」的核心防线：布局判错 → 结果全错或全空。
import { Detector } from '../.tmp/detector.js';

const det = new Detector();
// TS 的 private 在编译后就是普通方法，这里直接调用内部解析函数
const parse = (dims, data, conf = 0.25) => det.parseDetections({ dims, data }, conf);

let pass = 0, fail = 0;
function check(name, fn, want) {
  let got;
  try { got = fn(); } catch (e) { got = `THROW: ${e.message}`; }
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g === w) { pass += 1; console.log(`  PASS  ${name}`); }
  else { fail += 1; console.log(`  FAIL  ${name}\n        got  ${g}\n        want ${w}`); }
}
function checkThrows(name, fn, keyword) {
  let msg = '';
  try { fn(); } catch (e) { msg = e.message; }
  if (msg.includes(keyword)) { pass += 1; console.log(`  PASS  ${name} (抛出预期错误)`); }
  else { fail += 1; console.log(`  FAIL  ${name}\n        期望报错含「${keyword}」，实际：${msg || '没有抛错'}`); }
}

/* ---- 构造端到端输出 [1, 300, 6]：x1,y1,x2,y2,conf,cls ---- */
function e2eRowMajor(rows) {
  const d = new Float32Array(300 * 6);
  rows.forEach((r, i) => d.set(r, i * 6));
  return { dims: [1, 300, 6], data: d };
}
/* ---- 构造端到端输出 [1, 6, 300]：通道在前 ---- */
function e2eChannelFirst(rows) {
  const d = new Float32Array(6 * 300);
  rows.forEach((r, i) => r.forEach((v, c) => { d[c * 300 + i] = v; }));
  return { dims: [1, 6, 300], data: d };
}
/* ---- 构造经典头 [1, 4+nc, N]：通道在前（只有 boxes 指定的位置非零，其余是背景） ---- */
function classicChannelFirst(nc, n, boxes) {
  const M = 4 + nc;
  const d = new Float32Array(M * n);
  boxes.forEach((b, i) => {
    d[i] = b.cx;
    d[1 * n + i] = b.cy;
    d[2 * n + i] = b.w;
    d[3 * n + i] = b.h;
    for (let c = 0; c < nc; c += 1) d[(4 + c) * n + i] = c === b.cls ? b.score : 0;
  });
  return { dims: [1, M, n], data: d };
}
/* ---- 构造 v5 老格式 [1, 85, N]：通道 4 是 objness，5 起才是类别分 ---- */
function classicV5(nc, n, boxes) {
  const M = 5 + nc;
  const d = new Float32Array(M * n);
  boxes.forEach((b, i) => {
    d[i] = b.cx;
    d[1 * n + i] = b.cy;
    d[2 * n + i] = b.w;
    d[3 * n + i] = b.h;
    d[4 * n + i] = b.objness; // objness 单独一列
    for (let c = 0; c < nc; c += 1) d[(5 + c) * n + i] = c === b.cls ? b.score : 0;
  });
  return { dims: [1, M, n], data: d };
}
/* ---- 构造经典头 [1, N, 4+nc]：框在前 ---- */
function classicRowMajor(nc, n, boxes) {
  const M = 4 + nc;
  const d = new Float32Array(n * M);
  boxes.forEach((b, i) => {
    d[i * M + 0] = b.cx; d[i * M + 1] = b.cy; d[i * M + 2] = b.w; d[i * M + 3] = b.h;
    for (let c = 0; c < nc; c += 1) d[i * M + 4 + c] = c === b.cls ? b.score : 0;
  });
  return { dims: [1, n, M], data: d };
}

const ROWS = [
  [10, 20, 110, 220, 0.9, 3],
  [30, 40, 130, 140, 0.6, 1],
];

console.log('端到端 NMS-Free（YOLO26 nms=False / YOLOv10）：');
check('[1,300,6] 框在前', () => parse([1, 300, 6], e2eRowMajor(ROWS).data).length, 2);
check('[1,6,300] 通道在前', () => parse([1, 6, 300], e2eChannelFirst(ROWS).data).length, 2);
check('低置信度被过滤', () => parse([1, 300, 6], e2eRowMajor(ROWS).data, 0.95).length, 0);
check('坐标原样保留(xyxy)', () => {
  const b = parse([1, 300, 6], e2eRowMajor([[10, 20, 110, 220, 0.9, 0]]).data)[0];
  return { x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2, cls: b.cls, score: Math.round(b.score * 1000) / 1000 };
}, { x1: 10, y1: 20, x2: 110, y2: 220, cls: 0, score: 0.9 });
check('零填充行不产生框', () => parse([1, 300, 6], e2eRowMajor(ROWS).data).length, 2);
check('numClasses 取 80 起', () => { parse([1, 300, 6], e2eRowMajor(ROWS).data); return det.numClasses; }, 80);

console.log('\n经典一对多头（YOLO26 默认 nms=None / v8 / v11）：');
check('[1,84,8400] 通道在前', () => parse([1, 84, 8400], classicChannelFirst(80, 8400, [
  { cx: 50, cy: 60, w: 20, h: 30, cls: 5, score: 0.9 },
  { cx: 200, cy: 210, w: 40, h: 50, cls: 7, score: 0.7 },
]).data).length, 2);
check('[1,8400,84] 框在前', () => parse([1, 8400, 84], classicRowMajor(80, 8400, [
  { cx: 50, cy: 60, w: 20, h: 30, cls: 5, score: 0.9 },
]).data).length, 1);
check('xywh 转 xyxy', () => {
  const b = parse([1, 84, 8400], classicChannelFirst(80, 8400, [
    { cx: 50, cy: 60, w: 20, h: 30, cls: 1, score: 0.8 },
  ]).data)[0];
  return { x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2, cls: b.cls, score: Math.round(b.score * 1000) / 1000 };
}, { x1: 40, y1: 45, x2: 60, y2: 75, cls: 1, score: 0.8 });
check('numClasses = M-4', () => { parse([1, 84, 8400], classicChannelFirst(80, 8400, []).data); return det.numClasses; }, 80);
check('v5 的 85 通道带 objness', () => {
  const r = parse([1, 85, 8400], classicV5(80, 8400, [
    { cx: 50, cy: 60, w: 20, h: 30, cls: 2, score: 0.9, objness: 0.95 },
  ]).data);
  return { n: r.length, score: r.length ? Math.round(r[0].score * 1000) / 1000 : null };
}, { n: 1, score: 0.855 });
check('v5 objness 低时被压掉', () => parse([1, 85, 8400], classicV5(80, 8400, [
  { cx: 50, cy: 60, w: 20, h: 30, cls: 2, score: 0.9, objness: 0.1 },
]).data).length, 0);

console.log('\n歧义场景：2 类模型（4+2=6）不能被当成端到端：');
check('[1,6,8400] 走经典头', () => { parse([1, 6, 8400], classicChannelFirst(2, 8400, [
  { cx: 50, cy: 60, w: 20, h: 30, cls: 1, score: 0.9 },
]).data); return det.numClasses; }, 2);
check('[1,8400,6] 走经典头', () => { parse([1, 8400, 6], classicRowMajor(2, 8400, [
  { cx: 50, cy: 60, w: 20, h: 30, cls: 0, score: 0.9 },
]).data); return det.numClasses; }, 2);

console.log('\nNMS 生效（经典头重叠框应被合并）：');
check('重叠框只留一个', () => parse([1, 84, 8400], classicChannelFirst(80, 8400, [
  { cx: 100, cy: 100, w: 100, h: 100, cls: 0, score: 0.9 },
  { cx: 102, cy: 102, w: 100, h: 100, cls: 0, score: 0.8 },
]).data).length, 1);
check('端到端不再 NMS（保持 2 个）', () => parse([1, 300, 6], e2eRowMajor([
  [10, 20, 110, 220, 0.9, 0],
  [12, 22, 112, 222, 0.85, 0],
]).data).length, 2);

console.log('\n二维输出与异常输入：');
check('[300,6] 二维端到端', () => parse([300, 6], e2eRowMajor(ROWS).data).length, 2);
check('[300,6] 不被当成 2 类模型', () => { parse([300, 6], e2eRowMajor(ROWS).data); return det.numClasses; }, 80);
checkThrows('batch>1 明确报错', () => parse([4, 300, 6], e2eRowMajor(ROWS).data), 'batch');
checkThrows('未知布局明确报错', () => parse([1, 300, 3], new Float32Array(900)), '无法识别');
checkThrows('四维输出明确报错', () => parse([1, 1, 300, 6], new Float32Array(1800)), '无法识别');

console.log(`\n结果：${pass} 通过，${fail} 失败`);
if (fail > 0) process.exit(1);
