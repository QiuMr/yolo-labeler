const IMAGE_EXT = /\.(jpe?g|png|bmp|webp|gif|avif|tiff?)$/i;

export function isImageName(name: string): boolean {
  return IMAGE_EXT.test(name);
}

export function isImageFile(file: File): boolean {
  return file.type ? file.type.startsWith('image/') : isImageName(file.name);
}

/**
 * 读取 JPEG 的 EXIF Orientation（1~8，1 表示无需旋转）。
 *
 * 为什么需要它：`createImageBitmap()` 默认按 EXIF 方向解码（imageOrientation 默认
 * 'from-image'），所以画布上看到的是「已摆正」的图，尺寸也是摆正后的尺寸。
 * 但导出打包的是原始文件（未旋转）。两者不一致会让标注框在训练端整体错位，
 * 因此导出前必须知道哪些图带方向、并把它们烘焙成摆正后的版本。
 *
 * 防御：非 JPEG（PNG/WEBP 等没有标准 orientation）、没有 EXIF、解析异常，
 * 一律返回 1（按「无需旋转」处理），绝不让解析失败影响正常导出。
 */
export async function readExifOrientation(file: Blob): Promise<number> {
  try {
    // Orientation 在 APP1 段里，通常紧贴文件开头。多读一段以覆盖较大的 APP1。
    const buf = await file.slice(0, 128 * 1024).arrayBuffer();
    const view = new DataView(buf);
    if (view.byteLength < 4) return 1;
    // JPEG 必须以 SOI 开头
    if (view.getUint16(0) !== 0xffd8) return 1;

    let offset = 2;
    while (offset + 4 <= view.byteLength) {
      const marker = view.getUint16(offset);
      // 不是合法 marker → 数据异常，放弃解析
      if ((marker & 0xff00) !== 0xff00) return 1;
      // 填充字节（连续 FF），逐字节跳过
      if (marker === 0xffff) {
        offset += 1;
        continue;
      }
      // 进入压缩数据（SOS）或帧头（SOF）后不会再有 APP 段
      if (marker === 0xffda) return 1;
      if (marker >= 0xffc0 && marker <= 0xffcf && marker !== 0xffc4 && marker !== 0xffc8) {
        return 1;
      }

      const size = view.getUint16(offset + 2);
      if (size < 2) return 1;

      if (marker === 0xffe1) {
        const start = offset + 4;
        if (start + 6 > view.byteLength) return 1;
        // "Exif\0\0"
        if (view.getUint32(start) === 0x45786966 && view.getUint16(start + 4) === 0x0000) {
          return parseTiffOrientation(view, start + 6);
        }
      }

      offset += 2 + size;
    }
    return 1;
  } catch {
    return 1;
  }
}

/** 解析 TIFF 头 + IFD0，取出 Orientation 标签（0x0112） */
function parseTiffOrientation(view: DataView, tiffStart: number): number {
  try {
    if (tiffStart + 8 > view.byteLength) return 1;
    const bom = view.getUint16(tiffStart);
    const little = bom === 0x4949; // 'II'
    if (!little && bom !== 0x4d4d) return 1; // 不是 'MM'
    if (view.getUint16(tiffStart + 2, little) !== 0x002a) return 1; // TIFF 魔数

    const ifd = tiffStart + view.getUint32(tiffStart + 4, little);
    if (ifd + 2 > view.byteLength) return 1;

    const count = view.getUint16(ifd, little);
    for (let i = 0; i < count; i += 1) {
      const entry = ifd + 2 + i * 12;
      if (entry + 12 > view.byteLength) return 1;
      if (view.getUint16(entry, little) === 0x0112) {
        const value = view.getUint16(entry + 8, little);
        return value >= 1 && value <= 8 ? value : 1;
      }
    }
    return 1;
  } catch {
    return 1;
  }
}

/** 按扩展名兜底推断 MIME（file.type 偶尔为空） */
export function mimeForImage(file: File): string {
  if (file.type && file.type.startsWith('image/')) return file.type;
  const name = file.name.toLowerCase();
  if (name.endsWith('.png')) return 'image/png';
  if (name.endsWith('.webp')) return 'image/webp';
  if (name.endsWith('.bmp')) return 'image/bmp';
  return 'image/jpeg';
}
