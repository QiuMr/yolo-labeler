import type { ExportOptions, ImageItem, IRect, Label } from '../types';
import { clamp, clampRectToImage } from './geometry';
import { textBlob, triggerDownload } from './download';
import { decodeBitmap } from './imageCache';
import { mimeForImage, readExifOrientation } from './media';

export interface OutputFile {
  path: string;
  blob: Blob;
}

/** 不同文件夹里的同名图片会互相覆盖，这里统一做去重 */
class PathDeduper {
  private used = new Set<string>();

  take(path: string): string {
    if (!this.used.has(path)) {
      this.used.add(path);
      return path;
    }
    const dot = path.lastIndexOf('.');
    const stem = dot > 0 ? path.slice(0, dot) : path;
    const ext = dot > 0 ? path.slice(dot) : '';
    let n = 2;
    let candidate = `${stem}_${n}${ext}`;
    while (this.used.has(candidate)) {
      n += 1;
      candidate = `${stem}_${n}${ext}`;
    }
    this.used.add(candidate);
    return candidate;
  }
}

function baseName(name: string): string {
  return name.replace(/\.[^.]+$/, '');
}

function extOf(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot) : '';
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function round(v: number): number {
  return Math.round(v * 100) / 100;
}

/** 稳定哈希：保证同一张图每次导出都落在同一个集合 */
function hash32(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % 100000;
}

function escapeXml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

/**
 * 划分依据只用文件名：img.id 是运行时生成的随机 uid，重新导入同一批图后 id 会全部变化，
 * 划分结果也就跟着变了。用文件名才能保证「同一批图无论何时导出都落在同一个集合」。
 */
function splitGroup(img: ImageItem, ratio: number): 'train' | 'val' {
  return hash32(img.name) / 100000 < ratio ? 'val' : 'train';
}

function yoloLine(rect: IRect, index: number, imgW: number, imgH: number): string {
  const cx = clamp01((rect.x + rect.width / 2) / imgW);
  const cy = clamp01((rect.y + rect.height / 2) / imgH);
  const w = clamp01(rect.width / imgW);
  const h = clamp01(rect.height / imgH);
  return `${index} ${cx.toFixed(6)} ${cy.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`;
}

/** YAML 里纯数字、冒号开头这类名字必须加引号，否则 ultralytics 读出来的类别会不对 */
function yamlName(name: string): string {
  if (/^[\w\u4e00-\u9fa5][\w\u4e00-\u9fa5 ./-]*$/.test(name) && !/^\d+$/.test(name)) return name;
  return `"${name.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
}

function buildYaml(labels: Label[], split: boolean): string {
  const lines = [
    '# 由 YOLO 标注工具自动生成',
    '# 若图片未打包进 zip，请把 path / train / val 改成你本机的图片目录',
    'path: .',
    split ? 'train: train/images' : 'train: images',
    split ? 'val: val/images' : 'val: images',
    '',
    `nc: ${labels.length}`,
    'names:',
  ];
  labels.forEach((l, i) => lines.push(`  ${i}: ${yamlName(l.name)}`));
  return `${lines.join('\n')}\n`;
}

interface CocoImage {
  id: number;
  file_name: string;
  width: number;
  height: number;
}

interface CocoAnnotation {
  id: number;
  image_id: number;
  category_id: number;
  bbox: [number, number, number, number];
  area: number;
  iscrowd: number;
  segmentation: number[][];
}

interface CocoCategory {
  id: number;
  name: string;
  supercategory: string;
}

function buildVocXml(img: ImageItem, labels: Label[], indexOf: Map<string, number>): string {
  const objects: string[] = [];
  for (const ann of img.annotations) {
    const idx = indexOf.get(ann.labelId);
    if (idx === undefined) continue;
    const name = escapeXml(labels[idx].name);
    // VOC 是 1-based 闭区间，且必须落在图片范围内，否则训练端解析会报错
    const bx = clamp(Math.round(ann.rect.x), 0, Math.max(0, img.width - 1));
    const by = clamp(Math.round(ann.rect.y), 0, Math.max(0, img.height - 1));
    const bw = clamp(Math.round(ann.rect.width), 1, Math.max(1, img.width - bx));
    const bh = clamp(Math.round(ann.rect.height), 1, Math.max(1, img.height - by));
    const xmin = bx + 1;
    const ymin = by + 1;
    const xmax = bx + bw;
    const ymax = by + bh;
    objects.push(
      [
        '  <object>',
        `    <name>${name}</name>`,
        '    <pose>Unspecified</pose>',
        '    <truncated>0</truncated>',
        '    <difficult>0</difficult>',
        '    <bndbox>',
        `      <xmin>${xmin}</xmin>`,
        `      <ymin>${ymin}</ymin>`,
        `      <xmax>${xmax}</xmax>`,
        `      <ymax>${ymax}</ymax>`,
        '    </bndbox>',
        '  </object>',
      ].join('\n'),
    );
  }

  return [
    '<annotation>',
    '  <folder>images</folder>',
    `  <filename>${escapeXml(img.name)}</filename>`,
    '  <source><database>YOLO Labeler</database></source>',
    '  <size>',
    `    <width>${img.width}</width>`,
    `    <height>${img.height}</height>`,
    '    <depth>3</depth>',
    '  </size>',
    ...objects,
    '</annotation>',
    '',
  ].join('\n');
}

export function buildExportFiles(
  images: ImageItem[],
  labels: Label[],
  options: ExportOptions,
  /** 已按 EXIF 方向烘焙过的图片数据（key = 图片 id）；缺省回退到原始文件 */
  imageBlobs?: Map<string, Blob>,
): OutputFile[] {
  const files: OutputFile[] = [];
  const indexOf = new Map<string, number>();
  labels.forEach((l, i) => indexOf.set(l.id, i));
  const dedupe = new PathDeduper();
  const imageDataOf = (img: ImageItem): Blob | null => imageBlobs?.get(img.id) ?? img.file;

  const usable = images.filter((img) => img.width > 0 && img.height > 0);
  const targets = options.exportEmpty ? usable : usable.filter((img) => img.annotations.length > 0);
  const split = options.splitRatio > 0.001 && options.splitRatio < 0.999;
  const groupOf = (img: ImageItem) => (split ? splitGroup(img, options.splitRatio) : null);

  if (options.format === 'yolo') {
    for (const img of targets) {
      const lines: string[] = [];
      for (const ann of img.annotations) {
        const idx = indexOf.get(ann.labelId);
        if (idx === undefined) continue;
        lines.push(yoloLine(ann.rect, idx, img.width, img.height));
      }
      const group = groupOf(img);
      const labelDir = group ? `${group}/labels` : 'labels';
      const stem = dedupe.take(`${labelDir}/${baseName(img.name)}`);
      files.push({ path: `${stem}.txt`, blob: textBlob(lines.join('\n') + '\n') });
      const yoloImage = imageDataOf(img);
      if (options.includeImages && yoloImage) {
        const imageDir = group ? `${group}/images` : 'images';
        const imageStem = dedupe.take(`${imageDir}/${baseName(img.name)}`);
        files.push({ path: `${imageStem}${extOf(img.name)}`, blob: yoloImage });
      }
    }
    files.push({ path: 'classes.txt', blob: textBlob(`${labels.map((l) => l.name).join('\n')}\n`) });
    if (options.withYaml) {
      files.push({ path: 'data.yaml', blob: textBlob(buildYaml(labels, split), 'text/yaml;charset=utf-8') });
    }
    return files;
  }

  if (options.format === 'voc') {
    for (const img of targets) {
      const group = groupOf(img);
      const prefix = group ? `${group}/` : '';
      const stem = dedupe.take(`${prefix}${baseName(img.name)}`);
      files.push({ path: `${stem}.xml`, blob: textBlob(buildVocXml(img, labels, indexOf), 'application/xml') });
      const vocImage = imageDataOf(img);
      if (options.includeImages && vocImage) {
        files.push({ path: `${stem}${extOf(img.name)}`, blob: vocImage });
      }
    }
    files.push({ path: 'classes.txt', blob: textBlob(`${labels.map((l) => l.name).join('\n')}\n`) });
    return files;
  }

  // COCO
  const cocoImages: CocoImage[] = [];
  const cocoAnns: CocoAnnotation[] = [];
  const categories: CocoCategory[] = labels.map((l, i) => ({
    id: i + 1,
    name: l.name,
    supercategory: '',
  }));

  let imageId = 1;
  let annId = 1;
  for (const img of targets) {
    const id = imageId;
    imageId += 1;
    const group = groupOf(img);
    // 和 YOLO / VOC 保持一致：图片统一放在 images/ 下，file_name 才对得上真实路径
    const prefix = group ? `${group}/images/` : 'images/';
    const stem = dedupe.take(`${prefix}${baseName(img.name)}`);
    cocoImages.push({
      id,
      file_name: `${stem}${extOf(img.name)}`,
      width: img.width,
      height: img.height,
    });
    for (const ann of img.annotations) {
      const idx = indexOf.get(ann.labelId);
      if (idx === undefined) continue;
      // 与 YOLO / VOC 保持一致：先把矩形收敛到图片范围内再输出。
      // 否则越界的框会产出超出图片边界、甚至负面积的 bbox，训练端解析会出问题。
      const r = clampRectToImage(ann.rect, img.width, img.height);
      if (r.width < 1 || r.height < 1) continue;
      cocoAnns.push({
        id: annId,
        image_id: id,
        category_id: idx + 1,
        bbox: [round(r.x), round(r.y), round(r.width), round(r.height)],
        area: round(r.width * r.height),
        iscrowd: 0,
        segmentation: [],
      });
      annId += 1;
    }
    const cocoImage = imageDataOf(img);
    if (options.includeImages && cocoImage) {
      files.push({ path: `${stem}${extOf(img.name)}`, blob: cocoImage });
    }
  }

  const json = JSON.stringify(
    {
      info: { description: 'Generated by YOLO Labeler', version: '1.0', year: new Date().getFullYear() },
      licenses: [],
      images: cocoImages,
      annotations: cocoAnns,
      categories,
    },
    null,
    2,
  );
  files.push({ path: '_annotations.coco.json', blob: textBlob(json, 'application/json') });
  files.push({ path: 'classes.txt', blob: textBlob(`${labels.map((l) => l.name).join('\n')}\n`) });
  return files;
}

/** 导出进度回调：ratio 为 0~1 */
export interface ExportProgress {
  ratio: number;
  label: string;
}

export interface ExportResult {
  /** 因 EXIF 方向被自动摆正后重新编码的图片数 */
  rotated: number;
  /** 需要摆正但重新编码失败、只能导出原图的图片数 */
  failedBake: number;
}

/** 能安全重新编码的格式；bmp/gif 等不参与烘焙（它们也不带 EXIF 方向） */
const BAKABLE = /^image\/(jpeg|png|webp)$/;

/**
 * 把带 EXIF 方向的图「烘焙」成摆正后的位图再编码。
 * 画布始终按 EXIF 摆正显示，若直接打包原图，训练端读到的像素排布与标注坐标系不一致 → 框错位。
 * 失败一律返回 null，由调用方回退到原文件并提示，绝不中断导出。
 */
async function bakeOrientation(file: File): Promise<Blob | null> {
  let bmp: ImageBitmap | null = null;
  try {
    const type = mimeForImage(file);
    if (!BAKABLE.test(type)) return null;

    bmp = await decodeBitmap(file);
    if (bmp.width <= 0 || bmp.height <= 0) return null;

    const canvas = document.createElement('canvas');
    canvas.width = bmp.width;
    canvas.height = bmp.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bmp, 0, 0);

    const blob = await new Promise<Blob | null>((resolve) => {
      try {
        canvas.toBlob((b) => resolve(b), type, type === 'image/jpeg' ? 0.94 : undefined);
      } catch {
        resolve(null);
      }
    });
    return blob && blob.size > 0 ? blob : null;
  } catch {
    return null;
  } finally {
    // 必须释放：批量导出时不关会持续占用显存/内存
    if (bmp) bmp.close();
  }
}

/**
 * 准备要打包的图片数据：检测 EXIF 方向，必要时烘焙成摆正版本。
 * 串行处理（烘焙很吃内存），配合进度回调让界面有反馈。
 */
export async function prepareExportImages(
  images: ImageItem[],
  onProgress?: (done: number, total: number) => void,
): Promise<{ blobs: Map<string, Blob>; rotated: number; failed: number }> {
  const targets = images.filter((i): i is ImageItem & { file: File } => !!i.file);
  const blobs = new Map<string, Blob>();
  let rotated = 0;
  let failed = 0;

  for (let i = 0; i < targets.length; i += 1) {
    const img = targets[i];
    let out: Blob = img.file;
    try {
      if ((await readExifOrientation(img.file)) !== 1) {
        const baked = await bakeOrientation(img.file);
        if (baked) {
          out = baked;
          rotated += 1;
        } else {
          // 摆正失败：退回原图（宁可标注有偏差风险，也不能丢图），交给上层提示用户
          failed += 1;
        }
      }
    } catch {
      /* 保守：任何异常都按「无需处理」走 */
    }
    blobs.set(img.id, out);
    onProgress?.(i + 1, targets.length);
  }

  return { blobs, rotated, failed };
}

/** 打包并下载（jszip 按需加载，不拖慢首屏） */
export async function exportZip(
  images: ImageItem[],
  labels: Label[],
  options: ExportOptions,
  filename: string,
  onProgress?: (p: ExportProgress) => void,
): Promise<ExportResult> {
  let blobs: Map<string, Blob> | undefined;
  let rotated = 0;
  let failedBake = 0;

  if (options.includeImages) {
    onProgress?.({ ratio: 0, label: '正在校正图片方向…' });
    // 只处理真正会进包的图：尺寸未知的、以及「无标注且不导出空图」的都直接跳过。
    // 否则一次导出要把几百张根本用不上的图全部解码 + 重编码方向，纯属白干。
    const exportable = images.filter(
      (img) =>
        img.width > 0 &&
        img.height > 0 &&
        (options.exportEmpty || img.annotations.length > 0),
    );
    const prepared = await prepareExportImages(exportable, (done, total) => {
      const t = Math.max(1, total);
      onProgress?.({ ratio: (done / t) * 0.6, label: `正在处理图片 ${done}/${total}…` });
    });
    blobs = prepared.blobs;
    rotated = prepared.rotated;
    failedBake = prepared.failed;
  }

  onProgress?.({ ratio: 0.6, label: '正在生成标注文件…' });
  const files = buildExportFiles(images, labels, options, blobs);
  if (files.length === 0) throw new Error('没有可导出的内容');

  const { default: JSZip } = await import('jszip');
  const zip = new JSZip();
  // createFolders: false —— 不写入空的目录条目，zip 更干净
  for (const f of files) zip.file(f.path, f.blob, { createFolders: false });

  const blob = await zip.generateAsync(
    {
      type: 'blob',
      compression: 'DEFLATE',
      compressionOptions: { level: 6 },
    },
    (meta) => {
      // 打包阶段占后 40%
      onProgress?.({
        ratio: 0.6 + (Math.min(100, Math.max(0, meta.percent)) / 100) * 0.4,
        label: '正在打包…',
      });
    },
  );
  triggerDownload(blob, filename);

  return { rotated, failedBake };
}

