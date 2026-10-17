/** 标签配色：高饱和、彼此易区分，在深色画布上都清晰 */
export const PALETTE = [
  '#5b8cff',
  '#ff5b8c',
  '#2fd4a7',
  '#ffb020',
  '#a97bff',
  '#38d6ff',
  '#ff6b3d',
  '#7ed957',
  '#ff4d6d',
  '#00c2a8',
  '#f4a4ff',
  '#ffd166',
] as const;

export function colorForIndex(index: number): string {
  const i = ((index % PALETTE.length) + PALETTE.length) % PALETTE.length;
  return PALETTE[i];
}

/** #rgb / #rrggbb -> [r, g, b]；非法输入回落到品牌色 */
function parseHexUncached(hex: string): [number, number, number] {
  let h = hex.trim().replace('#', '');
  if (h.length === 3) {
    h = h
      .split('')
      .map((c) => c + c)
      .join('');
  }
  if (h.length !== 6 || /[^0-9a-fA-F]/.test(h)) return [37, 99, 235];
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * 带缓存的解析。
 *
 * 这是渲染循环里最热的一段：每个框每帧要解析约 3 次颜色
 * （两次 hexToRgba + 一次 textOn），密集图 500 框 × 60fps ≈ 9 万次/秒，
 * 而每一次都要 trim / replace / 正则 / parseInt 并新建一个三元组 —— 纯白烧的 CPU 和 GC。
 *
 * 实际用到的颜色只有调色板那十来个 + 未分类色 + 主题色，缓存后基本全是命中。
 *
 * ⚠️ 返回的是**共享对象**，调用方不得修改它（现有三处调用都只做解构）。
 */
const HEX_CACHE = new Map<string, [number, number, number]>();

export function parseHex(hex: string): [number, number, number] {
  const hit = HEX_CACHE.get(hex);
  if (hit) return hit;
  const rgb = parseHexUncached(hex);
  // 颜色可以来自用户标签，理论上无上界；给个软上限避免长期驻留
  if (HEX_CACHE.size >= 256) HEX_CACHE.clear();
  HEX_CACHE.set(hex, rgb);
  return rgb;
}

/**
 * 返回在该底色上可读的文字颜色。
 * 调色板里有 #ffd166 / #7ed957 这类很浅的颜色，一律用白字会看不清。
 */
export function textOn(hex: string): string {
  const [r, g, b] = parseHex(hex);
  // WCAG 相对亮度
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return lum > 0.62 ? '#12151d' : '#ffffff';
}

/**
 * 把标签色压暗到可放在浅色底上当前景文字使用的程度（感知亮度约 0.45）。
 * #ffd166 / #2fd4a7 这类高亮度色直接当文字用，在 12% 透明度的淡底上几乎看不清。
 */
export function readableInk(hex: string): string {
  const [r, g, b] = parseHex(hex);
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  if (lum <= 0.45) return hex;
  const k = 0.45 / lum;
  return `rgb(${Math.round(r * k)}, ${Math.round(g * k)}, ${Math.round(b * k)})`;
}

/**
 * #rgb / #rrggbb -> rgba(r,g,b,a)
 *
 * 同 parseHex：每个框每帧调两次，字符串构造本身也是分配。
 * 按 (颜色 → alpha → 字符串) 二级缓存，同一个 (颜色, alpha) 组合直接复用同一个字符串对象
 * —— 这对 canvas 也有好处：`ctx.fillStyle = 同一个字符串` 时浏览器不必反复判定内容。
 */
const RGBA_CACHE = new Map<string, Map<number, string>>();

export function hexToRgba(hex: string, alpha: number): string {
  let byAlpha = RGBA_CACHE.get(hex);
  if (!byAlpha) {
    if (RGBA_CACHE.size >= 256) RGBA_CACHE.clear();
    byAlpha = new Map<number, string>();
    RGBA_CACHE.set(hex, byAlpha);
  }
  const hit = byAlpha.get(alpha);
  if (hit) return hit;
  const [r, g, b] = parseHex(hex);
  const s = `rgba(${r}, ${g}, ${b}, ${alpha})`;
  byAlpha.set(alpha, s);
  return s;
}
