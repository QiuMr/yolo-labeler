import type { IPoint, IRect } from '../types';

export function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}

/** 两点确定矩形 */
export function rectFromPoints(a: IPoint, b: IPoint): IRect {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(a.x - b.x),
    height: Math.abs(a.y - b.y),
  };
}

/** 把矩形限制在图片范围内 */
export function clampRectToImage(r: IRect, imgW: number, imgH: number): IRect {
  const width = clamp(r.width, 0, imgW);
  const height = clamp(r.height, 0, imgH);
  const x = clamp(r.x, 0, Math.max(0, imgW - width));
  const y = clamp(r.y, 0, Math.max(0, imgH - height));
  return { x, y, width, height };
}

export const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'] as const;
export type HandleId = (typeof HANDLES)[number];

export interface HandlePoint {
  id: HandleId;
  point: IPoint;
}

/** 八个缩放手柄的位置（图像坐标） */
export function handlePoints(r: IRect): HandlePoint[] {
  const { x, y, width: w, height: h } = r;
  return [
    { id: 'nw', point: { x, y } },
    { id: 'n', point: { x: x + w / 2, y } },
    { id: 'ne', point: { x: x + w, y } },
    { id: 'e', point: { x: x + w, y: y + h / 2 } },
    { id: 'se', point: { x: x + w, y: y + h } },
    { id: 's', point: { x: x + w / 2, y: y + h } },
    { id: 'sw', point: { x, y: y + h } },
    { id: 'w', point: { x, y: y + h / 2 } },
  ];
}

/** 拖拽手柄后的新矩形（自动限制在图片内、保证最小尺寸） */
export function applyResize(
  origin: IRect,
  handle: HandleId,
  dx: number,
  dy: number,
  imgW: number,
  imgH: number,
  minSize: number,
): IRect {
  let x0 = origin.x;
  let y0 = origin.y;
  let x1 = origin.x + origin.width;
  let y1 = origin.y + origin.height;

  if (handle.includes('w')) x0 = clamp(x0 + dx, 0, x1 - minSize);
  if (handle.includes('e')) x1 = clamp(x1 + dx, x0 + minSize, imgW);
  if (handle.includes('n')) y0 = clamp(y0 + dy, 0, y1 - minSize);
  if (handle.includes('s')) y1 = clamp(y1 + dy, y0 + minSize, imgH);

  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}

/** 靠近图片边缘时吸附（阈值按屏幕像素传入，换算为图像像素） */
export function snapEdges(value: number, max: number, tolerance: number): number {
  if (Math.abs(value) <= tolerance) return 0;
  if (Math.abs(value - max) <= tolerance) return max;
  return value;
}

/**
 * 靠近图片边缘时吸附（阈值按屏幕像素传入，换算为图像像素）。
 *
 * 关键：左右两条边是**各自独立**吸附的，而吸附阈值（`7/scale`）大于最小框尺寸
 * （`4/scale`）—— 于是把一个窄框拖到右边界时，两条边会先后被吸到同一个位置，
 * 宽度直接变成 0：画布不再绘制（`Renderer` 对 `w<1` 直接 return），
 * 用户表现为「框拽不动」，而导出时 YOLO 会输出 `0.000000` 的宽度。
 *
 * 所以这里在吸附后做一次收口：一旦某条边塌缩，就让**离边界更远的那条**退让，
 * 至少保留 1px。这样吸附依然成立（视觉上贴边），框也不会消失。
 */
export function snapRectToEdges(r: IRect, imgW: number, imgH: number, tolerance: number): IRect {
  let x = r.x;
  let y = r.y;
  let right = r.x + r.width;
  let bottom = r.y + r.height;

  const x0 = snapEdges(x, imgW, tolerance);
  const x1 = snapEdges(right, imgW, tolerance);
  const y0 = snapEdges(y, imgH, tolerance);
  const y1 = snapEdges(bottom, imgH, tolerance);

  if (x0 !== x) x = x0;
  if (x1 !== right) right = x1;
  if (y0 !== y) y = y0;
  if (y1 !== bottom) bottom = y1;

  // 塌缩收口：宽度/高度归零时，把远离边界的那条边拉回来 1px
  if (right - x < 1) {
    if (imgW - right < right - x) {
      // 右边界侧更「近」→ 让左边界退让
      right = Math.min(imgW, x + 1);
      x = Math.max(0, right - 1);
    } else {
      x = Math.max(0, Math.min(x, imgW - 1));
      right = x + 1;
    }
  }
  if (bottom - y < 1) {
    if (imgH - bottom < bottom - y) {
      bottom = Math.min(imgH, y + 1);
      y = Math.max(0, bottom - 1);
    } else {
      y = Math.max(0, Math.min(y, imgH - 1));
      bottom = y + 1;
    }
  }

  return { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y) };
}

