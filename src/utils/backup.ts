import type { Annotation, ImageItem, Label } from '../types';
import { textBlob, triggerDownload } from './download';
import { registerSource } from './imageCache';

/**
 * 项目备份 / 还原。
 *
 * 为什么需要它：导出的 YOLO / VOC / COCO 只是「数据集产物」，只能拿去训练，
 * 导不回这个工具。也就是说误点「关闭项目」之后，就算之前导出过也没法恢复工程 ——
 * 「导出是唯一备份手段」这句话其实是假的。这里补上真正的备份闭环。
 *
 * 备份包结构：
 *   project.json          项目元信息（版本 / 类别 / 图片清单 + 标注）
 *   images/<id><ext>      图片原始二进制
 *
 * 图片名用 id 而不是文件名：文件名可能重名、含非法字符，id 唯一且安全。
 */

const BACKUP_VERSION = 1;

interface BackupMeta {
  version: number;
  savedAt: string;
  labels: Label[];
  activeLabelId: string;
  images: Array<{
    id: string;
    name: string;
    width: number;
    height: number;
    /** 原始文件修改时间；恢复时还原，保证判重指纹仍然有效 */
    lastModified: number;
    annotations: Annotation[];
  }>;
}

export interface RestoredProject {
  images: ImageItem[];
  labels: Label[];
  activeLabelId: string;
  /** 备份里声明了、但包内找不到图片文件的数量（标注仍在，重新导入原图即可） */
  missingImages: number;
}

function extOf(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot) : '';
}

function isLabel(v: unknown): v is Label {
  const l = v as Label | null;
  return !!l && typeof l.id === 'string' && typeof l.name === 'string' && typeof l.color === 'string';
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** 校验并清洗标注：坐标非法的直接丢弃，避免脏数据污染画布与导出 */
function sanitizeAnnotations(raw: unknown): Annotation[] {
  if (!Array.isArray(raw)) return [];
  const out: Annotation[] = [];
  for (const a of raw) {
    const ann = a as Annotation | null;
    if (!ann || typeof ann.id !== 'string') continue;
    const r = ann.rect;
    if (!r || !isFiniteNumber(r.x) || !isFiniteNumber(r.y)) continue;
    if (!isFiniteNumber(r.width) || !isFiniteNumber(r.height)) continue;
    if (r.width <= 0 || r.height <= 0) continue;
    out.push({
      id: ann.id,
      labelId: typeof ann.labelId === 'string' ? ann.labelId : '',
      rect: { x: r.x, y: r.y, width: r.width, height: r.height },
    });
  }
  return out;
}

/** 导出项目备份（zip） */
export async function exportProjectBackup(
  images: ImageItem[],
  labels: Label[],
  activeLabelId: string,
  filename: string,
  onProgress?: (ratio: number, label: string) => void,
): Promise<void> {
  const meta: BackupMeta = {
    version: BACKUP_VERSION,
    savedAt: new Date().toISOString(),
    labels,
    activeLabelId,
    images: images.map((i) => ({
      id: i.id,
      name: i.name,
      width: i.width,
      height: i.height,
      lastModified: i.file?.lastModified ?? 0,
      annotations: i.annotations,
    })),
  };

  onProgress?.(0, '正在整理项目…');

  const { default: JSZip } = await import('jszip');
  const zip = new JSZip();
  zip.file('project.json', textBlob(JSON.stringify(meta, null, 2), 'application/json'));

  const withFile = images.filter((i) => i.file);
  let done = 0;
  for (const img of withFile) {
    zip.file(`images/${img.id}${extOf(img.name)}`, img.file as Blob);
    done += 1;
    onProgress?.((done / Math.max(1, withFile.length)) * 0.5, `正在打包图片 ${done}/${withFile.length}…`);
  }

  const blob = await zip.generateAsync(
    {
      type: 'blob',
      compression: 'DEFLATE',
      compressionOptions: { level: 6 },
    },
    (m) => {
      onProgress?.(
        0.5 + (Math.min(100, Math.max(0, m.percent)) / 100) * 0.5,
        '正在生成备份…',
      );
    },
  );
  triggerDownload(blob, filename);
}

/** 从备份包还原项目。任何不合法/损坏的输入都抛出可直接展示给用户的中文错误 */
export async function importProjectBackup(file: File): Promise<RestoredProject> {
  const { default: JSZip } = await import('jszip');

  let zip: import('jszip');
  try {
    zip = await JSZip.loadAsync(file);
  } catch {
    throw new Error('这个文件读不出来，可能不是有效的备份包（需要本工具导出的 zip）');
  }

  const metaFile = zip.file('project.json');
  if (!metaFile) {
    throw new Error('不是项目备份文件：里面缺少 project.json');
  }

  let meta: BackupMeta;
  try {
    meta = JSON.parse(await metaFile.async('string')) as BackupMeta;
  } catch {
    throw new Error('备份文件已损坏：project.json 无法解析');
  }

  if (!meta || typeof meta !== 'object') {
    throw new Error('备份文件已损坏：内容为空');
  }
  if (meta.version !== BACKUP_VERSION) {
    throw new Error(`备份版本不兼容（文件是 v${String(meta.version)}，当前支持 v${BACKUP_VERSION}）`);
  }
  if (!Array.isArray(meta.images)) {
    throw new Error('备份文件已损坏：缺少图片清单');
  }

  const labels = Array.isArray(meta.labels) ? meta.labels.filter(isLabel) : [];
  const labelIds = new Set(labels.map((l) => l.id));

  // 图片文件按 id 索引，避免对每个条目都遍历一遍 zip
  const entryById = new Map<string, string>();
  for (const path of Object.keys(zip.files)) {
    if (!path.startsWith('images/')) continue;
    const name = path.slice('images/'.length);
    const dot = name.lastIndexOf('.');
    const id = dot > 0 ? name.slice(0, dot) : name;
    if (id && !entryById.has(id)) entryById.set(id, path);
  }

  const images: ImageItem[] = [];
  let missingImages = 0;

  for (const m of meta.images) {
    if (!m || typeof m.id !== 'string' || typeof m.name !== 'string') continue;

    let restored: File | null = null;
    const path = entryById.get(m.id);
    if (path) {
      const entry = zip.file(path);
      if (entry) {
        try {
          const blob = await entry.async('blob');
          if (blob && blob.size > 0) {
            // 还原原始修改时间，判重指纹才不会在恢复后失效（同 persist.ts 的处理）
            restored = new File([blob], m.name, {
              type: blob.type || 'image/*',
              lastModified: m.lastModified || 0,
            });
          }
        } catch {
          restored = null;
        }
      }
    }
    if (!restored) missingImages += 1;
    else registerSource(m.id, restored);

    const annotations = sanitizeAnnotations(m.annotations).map((a) =>
      // 标注指向了不存在的类别 → 退回「未分类」，而不是留一个悬空引用
      a.labelId && !labelIds.has(a.labelId) ? { ...a, labelId: '' } : a,
    );

    images.push({
      id: m.id,
      name: m.name,
      file: restored,
      width: isFiniteNumber(m.width) ? m.width : 0,
      height: isFiniteNumber(m.height) ? m.height : 0,
      annotations,
    });
  }

  if (images.length === 0) {
    throw new Error('备份里没有可恢复的图片');
  }

  const activeLabelId =
    typeof meta.activeLabelId === 'string' && labelIds.has(meta.activeLabelId)
      ? meta.activeLabelId
      : (labels[0]?.id ?? '');

  return { images, labels, activeLabelId, missingImages };
}
