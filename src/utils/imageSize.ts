/**
 * 从图片文件的**头部字节**直接读出像素尺寸，不解码整张图。
 *
 * 为什么需要它：
 *  - 左栏缩略图只需要 160px，但 `createImageBitmap` 默认会把原图**解成全尺寸位图**
 *    （一张 4000×3000 的照片 = 48MB 位图）。先知道原始尺寸，就能在解码时就传
 *    `resizeWidth` / `resizeHeight` 直接降采样 —— 内存和解码耗时都降一个数量级。
 *  - 导出前补齐尺寸（`ensureImageSizes`）以前也是逐张全尺寸解码，只为拿两个整数。
 *  - 而拿尺寸这件事，文件头里本来就写着，读前 256KB 即可，连解码器都不用启动。
 *
 * 支持 PNG / JPEG / GIF / BMP / WebP —— 即 `isImageName` 放行的全部格式。
 * 读到就返回；读不到（未知格式、头部损坏、SOF 超出读取窗口）**返回 null**，
 * 调用方回退到「老老实实解码」。所以这个模块只会变快，不会改变正确性。
 *
 * JPEG 的 EXIF 方向会被解析：orientation 5~8 会交换宽高，让这里的尺寸与
 * 浏览器 `imageOrientation: 'from-image'` 的显示尺寸保持一致。
 * （不一致的后果很隐蔽：导出的 YOLO 归一化坐标会按错误的宽高算，静默产出错标注。）
 */

export interface ImageSize {
  width: number;
  height: number;
}

/**
 * 只读前 256KB。
 * JPEG 的 SOF 段位于各种元数据之后，带大 EXIF + 内嵌缩略图的照片可能把它推到几十 KB 处；
 * 256KB 足够覆盖，又不会把一张 20MB 的照片整个读进内存。
 */
const HEAD_BYTES = 256 * 1024;

/** 读文件头并解析尺寸。任何异常都返回 null，交给调用方回退解码 */
export async function readImageSize(blob: Blob): Promise<ImageSize | null> {
  try {
    const head = new Uint8Array(await blob.slice(0, HEAD_BYTES).arrayBuffer());
    return parseImageSize(head);
  } catch {
    return null;
  }
}

/* ---------- 字节读取小工具（大端 / 小端 / 有符号 / 无符号） ---------- */

function be16(b: Uint8Array, o: number): number {
  return (b[o] << 8) | b[o + 1];
}
function be32(b: Uint8Array, o: number): number {
  return ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;
}
function le16(b: Uint8Array, o: number): number {
  return b[o] | (b[o + 1] << 8);
}
function le24(b: Uint8Array, o: number): number {
  return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16);
}
function le32u(b: Uint8Array, o: number): number {
  return (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0;
}
/** BMP 的宽高是**有符号** int32：负高度表示自上而下存储 */
function le32s(b: Uint8Array, o: number): number {
  return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24);
}

function ok(s: ImageSize | null): ImageSize | null {
  return s && s.width > 0 && s.height > 0 ? s : null;
}

/** 按 EXIF orientation 校正宽高（5~8 表示旋转了 90°，宽高互换） */
function applyOrientation(size: ImageSize, orientation: number): ImageSize {
  return orientation >= 5 && orientation <= 8
    ? { width: size.height, height: size.width }
    : size;
}

/* ---------- 各格式解析 ---------- */

export function parseImageSize(b: Uint8Array): ImageSize | null {
  // 最短的合法头是 GIF 的 10 字节（签名 6 + 逻辑屏幕描述符的宽高 4）；
  // 其余格式各自的分支里还会再检查自己需要的最小长度。
  if (b.length < 10) return null;

  // PNG：8 字节签名后紧跟 IHDR（长度 13，内容宽 4 + 高 4，均为大端）
  if (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) {
    if (b.length < 24) return null;
    return ok({ width: be32(b, 16), height: be32(b, 20) });
  }

  // GIF87a / GIF89a：逻辑屏幕描述符在固定位置，小端
  if (b[0] === 0x47 && b[1] === 0x49 && b[2] === 0x46) {
    return ok({ width: le16(b, 6), height: le16(b, 8) });
  }

  // BMP：'BM' + 偏移 14 处的 DIB 头大小决定字段宽度
  if (b[0] === 0x42 && b[1] === 0x4d) {
    if (b.length < 26) return null;
    const dib = le32u(b, 14);
    if (dib === 12) return ok({ width: le16(b, 18), height: le16(b, 20) });
    if (dib >= 40) {
      return ok({ width: Math.abs(le32s(b, 18)), height: Math.abs(le32s(b, 22)) });
    }
    return null;
  }

  // WebP：RIFF....WEBP，再按变体块取尺寸（三种编码各有各的布局）
  if (
    b.length >= 30 &&
    b[0] === 0x52 &&
    b[1] === 0x49 &&
    b[2] === 0x46 &&
    b[3] === 0x46 &&
    b[8] === 0x57 &&
    b[9] === 0x45 &&
    b[10] === 0x42 &&
    b[11] === 0x50
  ) {
    const c12 = b[12];
    const c13 = b[13];
    const c14 = b[14];
    const c15 = b[15];
    const isVP8X = c12 === 0x56 && c13 === 0x50 && c14 === 0x38 && c15 === 0x58; // 'VP8X'
    const isVP8L = c12 === 0x56 && c13 === 0x50 && c14 === 0x38 && c15 === 0x4c; // 'VP8L'
    const isVP8 = c12 === 0x56 && c13 === 0x50 && c14 === 0x38 && c15 === 0x20; // 'VP8 '

    // 扩展格式：画布尺寸是 24 位小端、存的是「值 - 1」
    if (isVP8X) return ok({ width: le24(b, 24) + 1, height: le24(b, 27) + 1 });
    // 无损：签名 0x2F 之后，14 位宽 + 14 位高，同样存「值 - 1」
    if (isVP8L && b[20] === 0x2f) {
      const v = le32u(b, 21);
      return ok({ width: (v & 0x3fff) + 1, height: ((v >>> 14) & 0x3fff) + 1 });
    }
    // 有损：帧起始码 9D 01 2A 之后是 14 位宽 / 高（低 2 位是缩放比例，要掩掉）
    if (isVP8) {
      if (b[23] !== 0x9d || b[24] !== 0x01 || b[25] !== 0x2a) return null;
      return ok({ width: le16(b, 26) & 0x3fff, height: le16(b, 28) & 0x3fff });
    }
    return null;
  }

  // JPEG：SOI(FFD8) 之后逐个 marker 扫描，直到 SOF 段
  if (b[0] === 0xff && b[1] === 0xd8) return parseJpeg(b);

  return null;
}

/** 扫描 JPEG 段，从 SOF 里取宽高；途中若遇到 APP1 则先记下 EXIF 方向 */
function parseJpeg(b: Uint8Array): ImageSize | null {
  let i = 2; // 跳过 SOI
  let orientation = 1;

  while (i + 3 < b.length) {
    // marker 必须以 0xFF 开头；段间可能有多余填充字节
    if (b[i] !== 0xff) {
      i += 1;
      continue;
    }
    const marker = b[i + 1];
    if (marker === 0xff) {
      i += 1;
      continue;
    }
    i += 2;

    // 无长度字段的 marker：TEM(01)、RSTn(D0-D7)、SOI(D8)、EOI(D9)
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd9)) continue;
    if (i + 1 >= b.length) break;

    const len = be16(b, i);
    if (len < 2 || i + len > b.length) break;

    // SOFn：C4(DHT) / C8(JPG) / CC(DAC) 不是帧头，要排除
    if (
      marker >= 0xc0 &&
      marker <= 0xcf &&
      marker !== 0xc4 &&
      marker !== 0xc8 &&
      marker !== 0xcc
    ) {
      // 段内布局：precision(1) + height(2) + width(2) + ...
      if (i + 7 >= b.length) break;
      const size = ok({ width: be16(b, i + 5), height: be16(b, i + 3) });
      return size ? applyOrientation(size, orientation) : null;
    }

    // APP1：EXIF 的方向信息就在这里（XMP 也会用 APP1，靠 Exif 魔数区分）
    if (marker === 0xe1) {
      const o = readExifOrientation(b, i + 2, len - 2);
      if (o !== 0) orientation = o;
    }

    i += len;
  }
  return null;
}

/**
 * 从 APP1 段里读 EXIF Orientation。
 * 布局：`Exif\0\0` + TIFF 头（字节序 II/MM、0x002A、IFD0 偏移）+ IFD0 条目表。
 * 返回 1~8；读不到返回 0。
 */
function readExifOrientation(b: Uint8Array, start: number, len: number): number {
  if (len < 14 || start + 14 > b.length) return 0;
  // "Exif\0\0"
  if (
    b[start] !== 0x45 ||
    b[start + 1] !== 0x78 ||
    b[start + 2] !== 0x69 ||
    b[start + 3] !== 0x66
  ) {
    return 0;
  }

  const tiff = start + 6;
  const le = b[tiff] === 0x49 && b[tiff + 1] === 0x49; // 'II'
  const be = b[tiff] === 0x4d && b[tiff + 1] === 0x4d; // 'MM'
  if (!le && !be) return 0;

  const u16 = (o: number): number => (le ? b[o] | (b[o + 1] << 8) : be16(b, o));
  const u32 = (o: number): number =>
    le
      ? (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0
      : be32(b, o);

  if (u16(tiff + 2) !== 0x2a) return 0;
  const ifd = tiff + u32(tiff + 4);
  if (ifd + 2 > b.length) return 0;

  const count = u16(ifd);
  let p = ifd + 2;
  for (let k = 0; k < count; k += 1) {
    if (p + 12 > b.length) break;
    if (u16(p) === 0x0112) {
      // type=3(SHORT) 时值内联在条目末 2 字节，否则取 4 字节偏移处的值
      const type = u16(p + 2);
      const value = type === 3 ? u16(p + 8) : u32(p + 8);
      return value >= 1 && value <= 8 ? value : 0;
    }
    p += 12;
  }
  return 0;
}
