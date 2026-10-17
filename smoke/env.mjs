// smoke 测试脚本的共享配置。
//
// 单独抽出来的原因：浏览器可执行文件路径原先在 smoke.mjs / repro.mjs / measure-ai.mjs
// 里各写了一遍，而 measure-ai.mjs 里还写死了一条**个人机器上的模型绝对路径**
// （C:\Users\...\ort.js推理测试1\yolo26n.onnx）—— 那是本机的目录结构，既不该出现在
// 公开仓库里，换台机器也跑不起来。现在统一走「环境变量 > 常见安装位置探测」。
//
// 必须用 fileURLToPath 而不是 new URL().pathname：后者会把非 ASCII 做百分号编码，
// 而本项目的路径里带中文，Windows 上会解不开。
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const HERE = dirname(fileURLToPath(import.meta.url));
export const ROOT = join(HERE, '..');

/** 浏览器候选位置：环境变量最优先，其后是各平台的常见安装路径 */
const BROWSER_CANDIDATES = [
  process.env.SMOKE_BROWSER,
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  '/usr/bin/microsoft-edge',
  '/usr/bin/microsoft-edge-stable',
  '/usr/bin/google-chrome',
].filter(Boolean);

/**
 * 用哪个浏览器跑无头测试。
 *
 * 探测不到已知路径时**不抛错**，而是退回首选路径让 spawn 去失败 ——
 * spawn 的报错信息里带着完整路径，比在这里抛一句「找不到浏览器」更好定位。
 */
export const EDGE =
  BROWSER_CANDIDATES.find((p) => existsSync(p)) ??
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

/**
 * 取要测的 YOLO onnx 模型路径。
 *
 * 不再写死个人目录：优先 SMOKE_MODEL 环境变量，其次项目根目录下的 yolo26n.onnx。
 * 都没有就抛错并说清怎么给 —— 总比拿一个不存在的路径去喂给页面、然后报一句
 * 看不懂的 CDP 错误要好。
 */
export function resolveModelPath() {
  const p = process.env.SMOKE_MODEL || join(ROOT, 'yolo26n.onnx');
  if (!existsSync(p)) {
    throw new Error(
      '找不到要测的模型文件。把 yolo26n.onnx 放到项目根目录，或设置环境变量 SMOKE_MODEL 指向它：\n' +
        '  set SMOKE_MODEL=D:\\path\\to\\yolo26n.onnx     (Windows)\n' +
        '  export SMOKE_MODEL=/path/to/yolo26n.onnx      (macOS / Linux)',
    );
  }
  return p;
}
