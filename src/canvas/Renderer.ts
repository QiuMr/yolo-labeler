import type { ViewTransform } from '../store/viewStore';
import type { Annotation, IPoint, IRect, Label } from '../types';
import { hexToRgba, textOn } from '../utils/color';
import { handlePoints } from '../utils/geometry';

export interface RenderState {
  bitmap: ImageBitmap | null;
  imgW: number;
  imgH: number;
  annotations: Annotation[];
  labels: Map<string, Label>;
  selectedId: string | null;
  hoverId: string | null;
  /** 正在拖动/缩放中的标注（用草稿矩形覆盖原始矩形） */
  draftId: string | null;
  draftRect: IRect | null;
  /** 正在画的新框 */
  pendingRect: IRect | null;
  /** AI 候选框（尚未采纳）：虚线样式，和已有标注明显区分 */
  aiCandidates: Array<{ rect: IRect; score: number; className: string }> | null;
  /**
   * 「点击标注」累积的点击点（原图坐标）。
   * 正点 = 要这个物体，负点 = 排除这块。不画出来的话，用户没法知道
   * 「已经点了几个、点在哪、哪些是排除的」，而修正全靠再点一下。
   */
  segPoints: readonly { x: number; y: number; positive: boolean }[] | null;
  /** 被用户隐藏的类别：这些框不画（导出与数据不受影响） */
  hiddenLabelIds: ReadonlySet<string>;
  cursor: IPoint | null;
  showCrosshair: boolean;
  /** 是否绘制缩放手柄 */
  showHandles: boolean;
  view: ViewTransform;
  width: number;
  height: number;
}

const HANDLE_RADIUS = 4.5;
/** 还没有类别的框：用中性灰 + 虚线，和任何标签色都不撞 */
const UNCLASSIFIED_COLOR = '#9aa2b1';
/** AI 候选框：青色虚线，与用户已采纳的标签框一眼可分 */
const AI_COLOR = '#38bdf8';
/** 点击标注的正点（要这块）/ 负点（排除这块）。绿=保留、红=排除是分割工具的通行约定 */
const SEG_POS_COLOR = '#34d399';
const SEG_NEG_COLOR = '#fb7185';
const SEG_POINT_RADIUS = 6;
const CHIP_HEIGHT = 19;
const CHIP_RADIUS = 6;
const CHIP_MIN_BOX_W = 26;
const FONT =
  '700 11px "Inter Variable", Inter, "MiSans VF", ui-sans-serif, system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif';
const PENDING_FONT =
  '600 11px "Inter Variable", Inter, "MiSans VF", ui-sans-serif, system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif';
/** 芯片可能画在框上方，比框高出一截；剔除余量要盖住这部分，否则放大后芯片会闪没 */
const CULL_PAD = 48;

/**
 * measureText 涉及文本布局，同一 (font, text) 每帧重测纯属浪费。
 * 类别名就那么多，缓存收益稳定；设上限防止异常输入把缓存撑爆。
 */
const textWidthCache = new Map<string, number>();
const MAX_TEXT_WIDTH_CACHE = 400;
function measureTextWidth(ctx: CanvasRenderingContext2D, font: string, text: string): number {
  const key = `${font}\u0000${text}`;
  const cached = textWidthCache.get(key);
  if (cached !== undefined) {
    // 命中缓存也必须把字体设回去：调用方紧接着就要 fillText，
    // 而缓存里只存了宽度，不设字体的话文字会沿用上一个测量过的字体
    ctx.font = font;
    return cached;
  }
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width);
  if (textWidthCache.size >= MAX_TEXT_WIDTH_CACHE) textWidthCache.clear();
  textWidthCache.set(key, w);
  return w;
}

/**
 * 作废文字宽度缓存。
 *
 * webfont（MiSans VF 有 56 个 unicode-range 分片）是**异步**加载的：
 * 首屏那次测量很可能落在系统字体上，宽度和换成 MiSans 之后并不一样。
 * 字体加载完成后必须清一次，否则芯片底片会整场会话都偏窄或偏宽 ——
 * 而画布不会因为有字体加载就自动重排（DOM 会，canvas 不会）。
 */
export function clearTextWidthCache(): void {
  textWidthCache.clear();
}

function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const radius = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.arcTo(x + w, y, x + w, y + radius, radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.arcTo(x + w, y + h, x + w - radius, y + h, radius);
  ctx.lineTo(x + radius, y + h);
  ctx.arcTo(x, y + h, x, y + h - radius, radius);
  ctx.lineTo(x, y + radius);
  ctx.arcTo(x, y, x + radius, y, radius);
  ctx.closePath();
}

/** 画四角的直角括号，暗示"这个框可以抓" */
function strokeCorners(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  len: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x, y + len);
  ctx.lineTo(x, y);
  ctx.lineTo(x + len, y);
  ctx.moveTo(x + w - len, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + len);
  ctx.moveTo(x + w, y + h - len);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w - len, y + h);
  ctx.moveTo(x + len, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + h - len);
  ctx.stroke();
}

export class Renderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  /**
   * 静态层缓存：图片 + 所有框 + 芯片 + AI 候选框 + 正在画的新框。
   *
   * 唯一被排除的是跟随鼠标的**十字准线** —— 鼠标一动它每帧都变，其余内容却一帧都没变。
   * 没有这一层时，鼠标在画面上每移动一像素都要把全部可见框和芯片重画一遍
   * （每框 3 次 strokeRect + 圆角路径 + measureText/fillText），框一多光标就跟不上手。
   * 而「点击标注」恰恰是最需要不停移动鼠标找物体的模式 —— 体感最差的就是它。
   *
   * 顺带省掉的还有每帧一次的 O(n) 视口剔除扫描：命中缓存时连标注数组都不遍历。
   */
  private layer: HTMLCanvasElement;
  private layerCtx: CanvasRenderingContext2D;
  /** 静态层的输入签名；null = 缓存作废。全部按引用比较，见 isLayerFresh */
  private layerKey: unknown[] | null = null;
  private dpr = 1;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('当前浏览器不支持 Canvas 2D');
    this.ctx = ctx;
    this.layer = document.createElement('canvas');
    const layerCtx = this.layer.getContext('2d');
    if (!layerCtx) throw new Error('当前浏览器不支持 Canvas 2D');
    this.layerCtx = layerCtx;
  }

  resize(cssWidth: number, cssHeight: number): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const w = Math.max(1, Math.round(cssWidth * dpr));
    const h = Math.max(1, Math.round(cssHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.layer.width = w;
      this.layer.height = h;
    }
    this.canvas.style.width = `${cssWidth}px`;
    this.canvas.style.height = `${cssHeight}px`;
    this.dpr = dpr;
    // 尺寸变了，缓存里的像素已经是错的
    this.layerKey = null;
  }

  render(state: RenderState): void {
    const { ctx } = this;
    const W = state.width;
    const H = state.height;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (W <= 0 || H <= 0) {
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.layerKey = null;
      return;
    }

    if (!this.isLayerFresh(state)) this.paintLayer(state);

    // 'copy' 是**覆盖**而不是混合：缓存里没有内容的地方是透明的，
    // 用默认的 source-over 擦不掉上一帧留下的准线，还得再补一次全幅 clearRect。
    // 换成 copy 之后这一步同时兼任「清屏」和「贴图」，全幅操作仍然只有一次。
    ctx.globalCompositeOperation = 'copy';
    ctx.drawImage(this.layer, 0, 0);
    ctx.globalCompositeOperation = 'source-over';

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.drawCrosshair(state);
  }

  /**
   * 静态层是否仍然可用。
   *
   * 这里全部用**引用比较**（`===`），不做深比较 —— 参与绘制的这些值在「没变」时
   * 引用本身就不变（annotations / labels 数组、view 对象、draftRect 对象都是被整体替换的），
   * 所以一次命中判定是 17 次指针比较，而它省掉的是「重画 N 个框」这件线性开销的事。
   *
   * 反向要求同样重要：任何会影响静态画面的输入都必须列进来。
   * 漏一个就会出现「画面不更新」——那比慢更难排查。
   */
  private isLayerFresh(s: RenderState): boolean {
    const k = this.layerKey;
    return (
      k !== null &&
      k[0] === s.bitmap &&
      k[1] === s.imgW &&
      k[2] === s.imgH &&
      k[3] === s.annotations &&
      k[4] === s.labels &&
      k[5] === s.selectedId &&
      k[6] === s.hoverId &&
      k[7] === s.draftId &&
      k[8] === s.draftRect &&
      k[9] === s.pendingRect &&
      k[10] === s.aiCandidates &&
      k[11] === s.hiddenLabelIds &&
      k[12] === s.view &&
      k[13] === s.width &&
      k[14] === s.height &&
      k[15] === s.showHandles &&
      k[16] === s.segPoints
    );
  }

  /**
   * 把静态内容画进离屏层，并记下签名。
   * 绘制方法（drawBox / drawChip / drawAiCandidate / drawPending）一行没改，
   * 只是把 this.ctx 临时指向离屏上下文 —— 这样加缓存不需要动任何一个绘制分支。
   */
  private paintLayer(state: RenderState): void {
    const main = this.ctx;
    const lctx = this.layerCtx;
    lctx.setTransform(1, 0, 0, 1, 0, 0);
    lctx.clearRect(0, 0, this.layer.width, this.layer.height);
    lctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.ctx = lctx;
    try {
      this.drawScene(state);
    } finally {
      this.ctx = main;
    }
    this.layerKey = [
      state.bitmap,
      state.imgW,
      state.imgH,
      state.annotations,
      state.labels,
      state.selectedId,
      state.hoverId,
      state.draftId,
      state.draftRect,
      state.pendingRect,
      state.aiCandidates,
      state.hiddenLabelIds,
      state.view,
      state.width,
      state.height,
      state.showHandles,
      state.segPoints,
    ];
  }

  /** 静态内容（图片 / 标注 / 候选框 / 新框）—— 绘制顺序与加缓存前完全一致 */
  private drawScene(state: RenderState): void {
    const { ctx } = this;
    const v = state.view;
    const ix = v.ox;
    const iy = v.oy;
    const iw = state.imgW * v.scale;
    const ih = state.imgH * v.scale;

    // 1. 图片（先垫一层投影，让它从深色背景里"浮"起来）
    if (state.bitmap && iw > 0 && ih > 0) {
      ctx.save();
      ctx.shadowColor = 'rgba(0,0,0,0.6)';
      ctx.shadowBlur = 18;
      ctx.shadowOffsetY = 6;
      ctx.fillStyle = 'rgba(0,0,0,0.92)';
      ctx.fillRect(ix, iy, iw, ih);
      ctx.restore();

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = v.scale >= 1 ? 'low' : 'high';
      ctx.drawImage(state.bitmap, ix, iy, iw, ih);

      const rx = Math.round(ix) - 0.5;
      const ry = Math.round(iy) - 0.5;
      const rw = Math.round(iw) + 1;
      const rh = Math.round(ih) + 1;
      // 外圈暗描边
      ctx.strokeStyle = 'rgba(0,0,0,0.5)';
      ctx.lineWidth = 1;
      ctx.strokeRect(rx - 1.5, ry - 1.5, rw + 3, rh + 3);
      // 内圈亮描边
      ctx.strokeStyle = 'rgba(255,255,255,0.14)';
      ctx.strokeRect(rx, ry, rw, rh);
    }

    // 2. 已有标注：有选中项时把其它框压暗，视线自动聚焦
    // 先做视口剔除：放大后大部分框都在视口外，全部绘制是白花钱
    const hasSelection = state.selectedId !== null;
    const { ox, oy, scale } = state.view;
    for (let i = 0; i < state.annotations.length; i += 1) {
      const ann = state.annotations[i];
      // 被隐藏的类别：整框跳过（不画，也不占序号）
      if (ann.labelId && state.hiddenLabelIds.has(ann.labelId)) continue;
      const rect = state.draftId === ann.id && state.draftRect ? state.draftRect : ann.rect;
      const sx = ox + rect.x * scale;
      const sy = oy + rect.y * scale;
      const sw = rect.width * scale;
      const sh = rect.height * scale;
      if (
        sx > state.width + CULL_PAD ||
        sy > state.height + CULL_PAD ||
        sx + sw < -CULL_PAD ||
        sy + sh < -CULL_PAD
      ) {
        continue;
      }
      this.drawBox(state, ann, rect, i, hasSelection);
    }

    // 3. AI 候选框（未采纳）：叠在标注之上、新框之下
    if (state.aiCandidates) {
      for (const cand of state.aiCandidates) {
        this.drawAiCandidate(state, cand.rect, cand.score, cand.className);
      }
    }

    // 4. 正在画的新框
    if (state.pendingRect) {
      this.drawPending(state, state.pendingRect);
    }

    // 5. 点击标注累积的点（画在最上层，保证任何情况下都看得见）
    if (state.segPoints && state.segPoints.length > 0) {
      this.drawSegPoints(state, state.segPoints);
    }
  }

  /**
   * 点击标注的点位标记。
   *
   * 正点：实心绿圆 + 白圈；负点：深色圆 + 红圈 + 红叉。
   * 颜色之外再加**形状**区分（叉号），是因为红绿是色觉障碍最常见的混淆对 ——
   * 只靠颜色区分等于对这部分用户没区分。
   * 每个点都先垫一圈半透明深色，保证在白墙、雪地这类浅背景上也看得清。
   */
  private drawSegPoints(
    state: RenderState,
    points: readonly { x: number; y: number; positive: boolean }[],
  ): void {
    const { ctx } = this;
    const v = state.view;
    const r = SEG_POINT_RADIUS;
    for (const p of points) {
      const cx = v.ox + p.x * v.scale;
      const cy = v.oy + p.y * v.scale;
      if (cx < -20 || cy < -20 || cx > state.width + 20 || cy > state.height + 20) continue;

      // 深色垫圈：浅色背景上的唯一保障
      ctx.beginPath();
      ctx.arc(cx, cy, r + 1.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = p.positive ? SEG_POS_COLOR : 'rgba(12,15,22,0.92)';
      ctx.fill();
      ctx.strokeStyle = p.positive ? '#ffffff' : SEG_NEG_COLOR;
      ctx.lineWidth = 2;
      ctx.stroke();

      if (!p.positive) {
        const s = r * 0.52;
        ctx.beginPath();
        ctx.moveTo(cx - s, cy - s);
        ctx.lineTo(cx + s, cy + s);
        ctx.moveTo(cx + s, cy - s);
        ctx.lineTo(cx - s, cy + s);
        ctx.strokeStyle = SEG_NEG_COLOR;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }

  /** 十字准线：唯一每帧都要重画的浮层，所以它不进静态缓存 */
  private drawCrosshair(state: RenderState): void {
    if (!state.showCrosshair || !state.cursor || state.pendingRect) return;
    const v = state.view;
    const ix = v.ox;
    const iy = v.oy;
    const iw = state.imgW * v.scale;
    const ih = state.imgH * v.scale;
    const inside =
      state.cursor.x >= ix &&
      state.cursor.x <= ix + iw &&
      state.cursor.y >= iy &&
      state.cursor.y <= iy + ih;
    if (!inside) return;

    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.26)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 5]);
    const cx = Math.round(state.cursor.x) + 0.5;
    const cy = Math.round(state.cursor.y) + 0.5;
    ctx.beginPath();
    ctx.moveTo(ix, cy);
    ctx.lineTo(ix + iw, cy);
    ctx.moveTo(cx, iy);
    ctx.lineTo(cx, iy + ih);
    ctx.stroke();
    ctx.restore();
  }

  private drawBox(
    state: RenderState,
    ann: Annotation,
    rect: IRect,
    index: number,
    hasSelection: boolean,
  ): void {
    const { ctx } = this;
    const v = state.view;
    const label = ann.labelId ? state.labels.get(ann.labelId) : undefined;
    const unclassified = !label;
    const color = label?.color ?? UNCLASSIFIED_COLOR;
    const selected = ann.id === state.selectedId;
    const hovered = ann.id === state.hoverId;

    const x = Math.round(v.ox + rect.x * v.scale);
    const y = Math.round(v.oy + rect.y * v.scale);
    const w = Math.round(rect.width * v.scale);
    const h = Math.round(rect.height * v.scale);
    if (w < 1 || h < 1) return;

    // 填充
    const alpha = selected ? 0.2 : hovered ? 0.14 : hasSelection ? 0.05 : 0.09;
    ctx.fillStyle = hexToRgba(color, alpha);
    ctx.fillRect(x, y, w, h);

    // 暗色底描边：任何颜色的框在任意背景上都清晰
    ctx.strokeStyle = selected ? 'rgba(0,0,0,0.55)' : 'rgba(0,0,0,0.42)';
    ctx.lineWidth = selected ? 3.5 : 3;
    ctx.strokeRect(x, y, w, h);

    // 主描边（选中时带一圈本色光晕；未分类用虚线标出来）
    ctx.save();
    if (selected) {
      ctx.shadowColor = hexToRgba(color, 0.6);
      ctx.shadowBlur = 12;
    }
    if (unclassified) ctx.setLineDash([7, 5]);
    ctx.strokeStyle = selected || hovered ? color : hexToRgba(color, hasSelection ? 0.72 : 0.92);
    ctx.lineWidth = selected ? 2 : hovered ? 1.8 : 1.5;
    ctx.strokeRect(x, y, w, h);
    ctx.restore();

    // 四角括号：hover 或 selected 时给出"可操作"的暗示
    if ((hovered || selected) && w > 18 && h > 18) {
      const corner = Math.min(12, w / 3, h / 3);
      ctx.strokeStyle = color;
      ctx.lineWidth = selected ? 3 : 2.5;
      strokeCorners(ctx, x, y, w, h, corner);
    }

    // 标签芯片：未分类的框无论多小都要标出来，否则用户不知道它缺什么
    if (unclassified || w >= CHIP_MIN_BOX_W || selected || hovered) {
      this.drawChip(index, label?.name ?? '未分类', color, x, y, state.width, selected || hovered);
    }

    // 缩放手柄：先垫一个略大的深色圆当投影，再画白圆，避免每帧开 8 次 shadowBlur
    if (selected && state.showHandles) {
      for (const hp of handlePoints(rect)) {
        const hx = v.ox + hp.point.x * v.scale;
        const hy = v.oy + hp.point.y * v.scale;
        ctx.beginPath();
        ctx.arc(hx, hy + 0.8, HANDLE_RADIUS + 1, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.42)';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(hx, hy, HANDLE_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }

  /** 框左上角的类别芯片：序号 + 名称，文字颜色随底色亮度自动切换 */
  private drawChip(
    index: number,
    name: string,
    color: string,
    x: number,
    y: number,
    canvasW: number,
    emphasized: boolean,
  ): void {
    const { ctx } = this;
    const fg = textOn(color);

    const text = String(Math.min(index + 1, 99));
    const nameW = measureTextWidth(ctx, FONT, name);
    const badgeW = 15;
    const chipW = badgeW + nameW + 14;

    let chipX = x;
    let chipY = y - CHIP_HEIGHT - 3;
    if (chipY < 2) chipY = y + 2;
    chipX = Math.max(2, Math.min(chipX, canvasW - chipW - 2));

    // Phase 11：原来用 shadowBlur 给芯片造投影 —— 模糊是每帧每框都要重算的昂贵合成操作
    // （缩放手柄那边早就用「先垫一个深色圆」绕开了，芯片漏了）。
    // 改成在下方 1px 处铺一层半透明深色圆角矩形：深色画布上肉眼几乎无差，成本降一个量级，
    // 顺带省掉一对 save/restore。
    ctx.fillStyle = emphasized ? 'rgba(0,0,0,0.45)' : 'rgba(0,0,0,0.32)';
    roundRectPath(ctx, chipX, chipY + 1, chipW, CHIP_HEIGHT, CHIP_RADIUS);
    ctx.fill();

    ctx.fillStyle = color;
    roundRectPath(ctx, chipX, chipY, chipW, CHIP_HEIGHT, CHIP_RADIUS);
    ctx.fill();

    // 序号小方块
    ctx.fillStyle = fg === '#ffffff' ? 'rgba(0,0,0,0.26)' : 'rgba(0,0,0,0.1)';
    roundRectPath(ctx, chipX + 2.5, chipY + 2.5, badgeW, CHIP_HEIGHT - 5, 4);
    ctx.fill();

    ctx.fillStyle = fg;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, chipX + 2.5 + badgeW / 2, chipY + CHIP_HEIGHT / 2 + 0.5);
    ctx.textAlign = 'left';
    ctx.fillText(name, chipX + badgeW + 8, chipY + CHIP_HEIGHT / 2 + 0.5);
    ctx.textBaseline = 'alphabetic';
  }

  /** AI 候选框：青色虚线 + 「类名 + 置信度」深色小标，等用户决定要不要采纳 */
  private drawAiCandidate(
    state: RenderState,
    rect: IRect,
    score: number,
    className: string,
  ): void {
    const { ctx } = this;
    const v = state.view;
    const x = Math.round(v.ox + rect.x * v.scale);
    const y = Math.round(v.oy + rect.y * v.scale);
    const w = Math.round(rect.width * v.scale);
    const h = Math.round(rect.height * v.scale);
    if (w < 1 || h < 1) return;

    ctx.fillStyle = hexToRgba(AI_COLOR, 0.08);
    ctx.fillRect(x, y, w, h);

    // 暗底描边保证任意背景可辨
    ctx.strokeStyle = 'rgba(0,0,0,0.48)';
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    ctx.save();
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = AI_COLOR;
    ctx.lineWidth = 1.6;
    ctx.strokeRect(x, y, w, h);
    ctx.restore();

    if (w > 16 && h > 16) {
      const corner = Math.min(11, w / 3, h / 3);
      ctx.strokeStyle = AI_COLOR;
      ctx.lineWidth = 2.2;
      strokeCorners(ctx, x, y, w, h, corner);
    }

    // 置信度小标：深底青字，和彩色标签芯片区分开
    const text = `${className} ${Math.round(score * 100)}%`;
    const tw = measureTextWidth(ctx, PENDING_FONT, text);
    const chipW = tw + 14;
    let chipX = x;
    let chipY = y - CHIP_HEIGHT - 3;
    if (chipY < 2) chipY = y + 2;
    chipX = Math.max(2, Math.min(chipX, state.width - chipW - 2));

    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.4)';
    ctx.shadowBlur = 6;
    ctx.shadowOffsetY = 1;
    ctx.fillStyle = 'rgba(10,14,22,0.9)';
    roundRectPath(ctx, chipX, chipY, chipW, CHIP_HEIGHT, CHIP_RADIUS);
    ctx.fill();
    ctx.restore();
    ctx.strokeStyle = hexToRgba(AI_COLOR, 0.55);
    ctx.lineWidth = 1;
    roundRectPath(ctx, chipX, chipY, chipW, CHIP_HEIGHT, CHIP_RADIUS);
    ctx.stroke();

    ctx.fillStyle = '#9bdcfd';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, chipX + 7, chipY + CHIP_HEIGHT / 2 + 0.5);
    ctx.textBaseline = 'alphabetic';
  }

  private drawPending(state: RenderState, rect: IRect): void {
    const { ctx } = this;
    const v = state.view;
    const x = Math.round(v.ox + rect.x * v.scale);
    const y = Math.round(v.oy + rect.y * v.scale);
    const w = Math.round(rect.width * v.scale);
    const h = Math.round(rect.height * v.scale);

    ctx.save();

    // 半透明白填充：在任何图上都看得清，又不会盖住内容
    ctx.fillStyle = 'rgba(255,255,255,0.13)';
    ctx.fillRect(x, y, w, h);

    // 深色描边 + 白色虚线，双线保证在任何背景上都可辨
    ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);
    ctx.strokeStyle = 'rgba(255,255,255,0.94)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    // 四角实心小方块
    const s = 3;
    ctx.fillStyle = 'rgba(255,255,255,0.95)';
    for (const [px, py] of [
      [x, y],
      [x + w, y],
      [x, y + h],
      [x + w, y + h],
    ]) {
      ctx.fillRect(Math.round(px) - s, Math.round(py) - s, s * 2, s * 2);
    }

    // 尺寸提示
    const text = `${Math.round(rect.width)} × ${Math.round(rect.height)}`;
    const tw = measureTextWidth(ctx, PENDING_FONT, text);
    const cw = Math.ceil(tw) + 18;
    let cx = x + w + 10;
    let cy = y + h + 10;
    if (cx + cw > state.width - 2) cx = Math.max(2, x + w - cw - 10);
    if (cy + CHIP_HEIGHT > state.height - 2) cy = Math.max(2, y - CHIP_HEIGHT - 10);

    ctx.fillStyle = 'rgba(11,13,19,0.95)';
    roundRectPath(ctx, cx, cy, cw, CHIP_HEIGHT, CHIP_RADIUS);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.14)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.88)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, cx + cw / 2, cy + CHIP_HEIGHT / 2 + 0.5);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.restore();
  }
}
