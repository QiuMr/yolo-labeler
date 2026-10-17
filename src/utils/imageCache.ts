import type { ImageItem } from '../types';
import { uid } from './id';

/**
 * 图片缓存：
 * - 解码后的 ImageBitmap 用 LRU 限制数量，避免大图集把内存吃满
 * - 缩略图只生成一次，转成体积很小的 jpeg dataURL，永不释放（单张约 2~4KB）
 * - 通过 protect() 保证「当前正在显示」的那张不会被回收
 */

const MAX_BITMAPS = 48;
const THUMB_SIZE = 160;

const bitmaps = new Map<string, ImageBitmap>();
const inflight = new Map<string, Promise<ImageBitmap | null>>();
const thumbnails = new Map<string, string>();
const thumbInflight = new Map<string, Promise<string>>();
const sources = new Map<string, Blob>();

let protectedId: string | null = null;

export function protect(id: string | null): void {
  protectedId = id;
}

export function registerSource(id: string, blob: Blob): void {
  sources.set(id, blob);
}

function evict(): void {
  if (bitmaps.size <= MAX_BITMAPS) return;
  for (const key of bitmaps.keys()) {
    if (bitmaps.size <= MAX_BITMAPS) break;
    if (key === protectedId) continue;
    const bmp = bitmaps.get(key);
    bitmaps.delete(key);
    if (bmp) bmp.close();
  }
  // 极端情况下（位图数仍超标）兜底释放最旧的一张。
  // 跳过 protectedId —— 它正在被画布持有，close() 会让画布画出空白。
  // 宁可暂时超出上限（多留一张的内存远优于黑屏）。
  if (bitmaps.size > MAX_BITMAPS) {
    for (const key of bitmaps.keys()) {
      if (bitmaps.size <= MAX_BITMAPS) break;
      if (key === protectedId) continue;
      const bmp = bitmaps.get(key);
      bitmaps.delete(key);
      if (bmp) bmp.close();
    }
  }
}

function sourceOf(item: ImageItem): Blob | null {
  return item.file ?? sources.get(item.id) ?? null;
}

/**
 * 统一入口：显式指定 imageOrientation。
 *
 * 规范的默认值确实是 'from-image'（会按 EXIF 摆正），但不同浏览器/版本处理过差异，
 * 这里写死成显式参数，避免「画布显示方向」随环境漂移 —— 导出端依赖这个行为保持一致。
 * 老浏览器不认识第二个参数时会抛错，回退到单参数形式。
 */
export async function decodeBitmap(src: Blob): Promise<ImageBitmap> {
  try {
    return await createImageBitmap(src, { imageOrientation: 'from-image' } as ImageBitmapOptions);
  } catch {
    return await createImageBitmap(src);
  }
}

/** 取已解码的位图（不触发解码） */
export function peekBitmap(id: string): ImageBitmap | null {
  return bitmaps.get(id) ?? null;
}

/** 解码位图，带并发去重与 LRU 缓存 */
export function loadBitmap(item: ImageItem): Promise<ImageBitmap | null> {
  const cached = bitmaps.get(item.id);
  if (cached) {
    // 命中也要刷新「最近使用」顺序，否则淘汰的永远是最久*插入*的那张。
    // 典型症状：图片数超过 MAX_BITMAPS 后，「在两张图之间来回比对」持续失效 ——
    // 按 ← 回到上一张（它是 48 张里最早插入的）已被淘汰，于是重新全尺寸解码一次，
    // 表现为「按 ← 回去看上一张图会明显卡一下再出图」。
    // Map 保持插入序，所以 delete + set 就是在把它挪到末尾（即最新）。
    bitmaps.delete(item.id);
    bitmaps.set(item.id, cached);
    return Promise.resolve(cached);
  }

  const pending = inflight.get(item.id);
  if (pending) return pending;

  const src = sourceOf(item);
  if (!src) return Promise.resolve(null);

  const task = decodeBitmap(src)
    .then((bmp) => {
      inflight.delete(item.id);
      if (bitmaps.has(item.id)) {
        bmp.close();
        return bitmaps.get(item.id) ?? null;
      }
      bitmaps.set(item.id, bmp);
      evict();
      return bmp;
    })
    .catch(() => {
      inflight.delete(item.id);
      return null;
    });

  inflight.set(item.id, task);
  return task;
}

export function getThumbnailSync(id: string): string | null {
  return thumbnails.get(id) ?? null;
}

/**
 * 缩略图生成的并发闸。
 *
 * 快速滚动左栏会一次性触发十几张缩略图。不限流的话，峰值内存和 CPU 会一起冲高
 * —— 每一张都要把原图解一次全尺寸（6000×4000 的位图约 96MB）。
 * 对比：store 里补图片尺寸的 ensureImageSizes 也是限 5 并发，这里取 4。
 */
const THUMB_CONCURRENCY = 4;
let thumbActive = 0;
const thumbQueue: Array<() => void> = [];

function acquireThumbSlot(): Promise<void> {
  if (thumbActive < THUMB_CONCURRENCY) {
    thumbActive += 1;
    return Promise.resolve();
  }
  return new Promise<void>((resolve) => {
    thumbQueue.push(() => {
      thumbActive += 1;
      resolve();
    });
  });
}

function releaseThumbSlot(): void {
  thumbActive -= 1;
  const next = thumbQueue.shift();
  if (next) next();
}

/**
 * 把一张图渲染成缩略图。
 *
 * ⚠️ 关键：**不把全尺寸位图留进 LRU**。
 * 旧实现走 `loadBitmap()` —— 为了 160px 的缩略图先把整图解成全尺寸，再把这张大位图
 * 塞进缓存。两个后果：
 *   1. 滚动左栏时会持续把「用户正在看的那张」挤出 LRU，切图时又要重新全尺寸解码
 *      （表现为按 ← 回去看上一张会卡一下）；
 *   2. `MAX_BITMAPS = 48` 限制的是**张数**不是字节数，48 张 96MB 的位图就是几个 GB。
 * 现在改成：已经在缓存里的直接复用（当前显示的那张），否则解一张**临时**位图，
 * 画完立刻 close()。
 */
async function renderThumb(item: ImageItem, src: Blob): Promise<string> {
  const reused = peekBitmap(item.id);
  const bmp = reused ?? (await decodeBitmap(src));
  try {
    const scale = Math.min(THUMB_SIZE / bmp.width, THUMB_SIZE / bmp.height, 1);
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bmp, 0, 0, w, h);

    try {
      return canvas.toDataURL('image/jpeg', 0.72);
    } catch {
      return '';
    }
  } finally {
    // 只关「我们自己解出来的」那张，复用的那张归 LRU 管
    if (!reused) bmp.close();
  }
}

/** 生成缩略图（懒加载，结果缓存） */
export function loadThumbnail(item: ImageItem): Promise<string> {
  const cached = thumbnails.get(item.id);
  if (cached) return Promise.resolve(cached);

  const pending = thumbInflight.get(item.id);
  if (pending) return pending;

  const src = sourceOf(item);
  if (!src) return Promise.resolve('');

  const task = acquireThumbSlot()
    .then(() => renderThumb(item, src))
    .then((url) => {
      if (url) thumbnails.set(item.id, url);
      thumbInflight.delete(item.id);
      return url;
    })
    .catch(() => {
      thumbInflight.delete(item.id);
      return '';
    })
    .finally(() => releaseThumbSlot());

  thumbInflight.set(item.id, task);
  return task;
}

/** 移除图片时释放资源 */
export function dropImage(id: string): void {
  const bmp = bitmaps.get(id);
  if (bmp) {
    bitmaps.delete(id);
    bmp.close();
  }
  inflight.delete(id);
  thumbInflight.delete(id);
  thumbnails.delete(id);
  sources.delete(id);
  if (protectedId === id) protectedId = null;
}

/** 从目录拖拽等场景拿到的 File 可能重名，这里补一个稳定标识 */
export function makeImageId(): string {
  return uid('im_');
}
