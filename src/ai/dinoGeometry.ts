import type { IRect } from '../types';

/**
 * DINOv2 的**几何算术** —— 与模型推理完全无关，因此单独成模块。
 *
 * 为什么要拆出来：`dinoPrompt.ts` 顶部 import 了 onnxruntime 运行时，
 * 任何测试一旦碰到它就等于把整个 wasm 运行时拖进来。而这里全是纯函数，
 * 可以被 `test:parse` 直接单测 —— 2026-10-19 那次「小目标召回为 0」的事故
 * 根因就在这几个数字上，必须有回归测试钉住。
 */

/** 模型固定输入边长（14 × 37）。换模型必须同步改这三个常量 */
export const DINO_SIZE = 518;
export const DINO_PATCH = 14;
export const DINO_GRID = DINO_SIZE / DINO_PATCH;
/** patch 特征维度（ViT-S/14 的 hidden size） */
export const DINO_DIM = 384;

/** letterbox 结果。与 `visualPrompt.ts` 的 `VpLetterbox` 结构一致（便于互相赋值） */
export interface DinoLetterbox {
  gain: number;
  w: number;
  h: number;
  left: number;
  top: number;
}

/** 与 visualPrompt 的 letterbox 同规则，只是边长不同（用 floor 保持与 Python 对拍一致） */
export function dinoLetterbox(imgW: number, imgH: number): DinoLetterbox {
  const gain = Math.min(DINO_SIZE / imgW, DINO_SIZE / imgH);
  const w = Math.max(1, Math.round(imgW * gain));
  const h = Math.max(1, Math.round(imgH * gain));
  return {
    gain,
    w,
    h,
    left: Math.floor((DINO_SIZE - w) / 2),
    top: Math.floor((DINO_SIZE - h) / 2),
  };
}

/** 原图坐标的框 → 特征网格上的尺寸（patch 数）。守卫与检测共用同一套算术 */
export function gridSizeOf(refW: number, refH: number, gain: number, scale: number): [number, number] {
  return [
    Math.max(1, Math.round((refW * scale * gain) / DINO_PATCH)),
    Math.max(1, Math.round((refH * scale * gain) / DINO_PATCH)),
  ];
}

/**
 * 目标至少要占多少个 patch（**每边**）才算「DINOv2 看得见」。
 *
 * 依据是**分辨率算术**，不是统计拟合：
 *   patch 固定 14px、输入固定 518 → 网格 37x37，所以「目标在原图上多大」
 *   要除以 gain 才能落到网格上。窗口聚合（dinoPrompt 的核心先验）要求整块窗口
 *   都像目标，目标只占 1 个 patch 时窗口退化成 1x1 —— 那个 patch 是
 *   「目标 + 周围背景」的混合，于是原型匹配到的是背景，不是目标。
 *
 * 取 3：窗口 3x3 = 9 个 patch，均值才有意义。
 * 换算：目标边长 ≥ 3 * 14 / gain = 42 / gain。
 *   · 1366x768 的图 gain=0.379 → 目标要 ≥ 111px
 *   · 320x320 的图  gain=1.619 → 目标只要 ≥ 26px（合成基准的 72px 精灵轻松通过）
 */
export const DINO_MIN_OBJ_PATCH = 3;

export interface DinoFeasibility {
  ok: boolean;
  /** 参考框在特征网格上的尺寸（patch 数，按 scale=1.0 算） */
  pw: number;
  ph: number;
  /** 参考框的较短边长（原图像素） */
  objPx: number;
  /** 原图上目标至少要多大才可用（像素） */
  needPx: number;
  /** 目标在 518 输入上占多少像素 */
  objPxInInput: number;
  /** 给用户看的一句话；ok 时为空串 */
  reason: string;
}

/**
 * 判断「这批参考框在这张图上能不能用 DINOv2」。
 *
 * 纯函数、不碰模型 —— 界面可以在用户点按钮**之前**就给出提示，
 * 而不是等他跑完拿到一堆乱框。`encodePrompt` 里还会再拦一次（防止绕过界面调用）。
 */
export function dinoFeasibility(
  imgW: number,
  imgH: number,
  boxes: readonly IRect[],
): DinoFeasibility {
  if (boxes.length === 0 || imgW <= 0 || imgH <= 0) {
    return { ok: false, pw: 0, ph: 0, objPx: 0, needPx: 0, objPxInInput: 0, reason: '' };
  }
  const lb = dinoLetterbox(imgW, imgH);
  const refW = boxes.reduce((s, b) => s + b.width, 0) / boxes.length;
  const refH = boxes.reduce((s, b) => s + b.height, 0) / boxes.length;
  const [pw, ph] = gridSizeOf(refW, refH, lb.gain, 1);
  const needPx = (DINO_MIN_OBJ_PATCH * DINO_PATCH) / lb.gain;
  const objPx = Math.min(refW, refH);
  const objPxInInput = objPx * lb.gain;
  /**
   * 判据用**未取整**的 patch 数，不是 `pw`。
   *
   * `pw` 是 round 过的（检测时窗口必须落在整格上），但拿它当门槛会把
   * 「2.5 个 patch」也放进来说明可用 —— 那时 round 成 3 的窗口（42px）
   * 已经比目标（35px）还大，窗口均值被背景稀释，正是要避免的情形。
   */
  const patches = objPxInInput / DINO_PATCH;
  if (patches >= DINO_MIN_OBJ_PATCH) {
    return { ok: true, pw, ph, objPx, needPx, objPxInInput, reason: '' };
  }
  return {
    ok: false,
    pw,
    ph,
    objPx,
    needPx,
    objPxInInput,
    reason:
      `目标太小，DINOv2 认不出来：缩到模型的 518 像素输入之后，它只占 ${objPxInInput.toFixed(1)} 个像素` +
      `（= ${patches.toFixed(1)} 个特征块），至少要 ${DINO_MIN_OBJ_PATCH * DINO_PATCH} 个像素` +
      `（${DINO_MIN_OBJ_PATCH} 个块）。` +
      `换算到这张 ${imgW}x${imgH} 的图，目标边长得有 ${Math.ceil(needPx)} 个像素以上。` +
      `硬跑的话它会吐几十个乱框（实测这种尺寸下召回为 0）。` +
      `请改用「YOLOE 视觉提示」—— 它对这个尺寸的目标是有效的。`,
  };
}
