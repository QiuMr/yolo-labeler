import { isImageName } from './media';

interface EntryLike {
  isFile: boolean;
  isDirectory: boolean;
  name: string;
  file: (success: (file: File) => void, error?: (err: unknown) => void) => void;
  createReader: () => {
    readEntries: (
      success: (entries: EntryLike[]) => void,
      error?: (err: unknown) => void,
    ) => void;
  };
}

type ItemLike = {
  webkitGetAsEntry?: () => EntryLike | null;
};

function walk(entry: EntryLike, out: File[]): Promise<void> {
  if (entry.isFile) {
    if (!isImageName(entry.name)) return Promise.resolve();
    return new Promise((resolve) => {
      try {
        entry.file(
          (file) => {
            out.push(file);
            resolve();
          },
          () => resolve(),
        );
      } catch {
        resolve();
      }
    });
  }

  if (!entry.isDirectory) return Promise.resolve();

  return new Promise((resolve) => {
    let reader: ReturnType<EntryLike['createReader']>;
    try {
      reader = entry.createReader();
    } catch {
      resolve();
      return;
    }
    const collected: EntryLike[] = [];
    const readBatch = () => {
      reader.readEntries(
        (batch) => {
          if (!batch || batch.length === 0) {
            void Promise.all(collected.map((e) => walk(e, out))).then(() => resolve());
            return;
          }
          collected.push(...batch);
          readBatch();
        },
        () => resolve(),
      );
    };
    readBatch();
  });
}

/** 从拖放事件中提取图片，支持直接拖文件夹 */
export async function extractFiles(dataTransfer: DataTransfer): Promise<File[]> {
  const items = Array.from(dataTransfer.items ?? []) as unknown as ItemLike[];
  const canUseEntries = items.some((i) => typeof i.webkitGetAsEntry === 'function');

  if (!canUseEntries) {
    return Array.from(dataTransfer.files ?? []).filter((f) => isImageName(f.name));
  }

  const out: File[] = [];
  const tasks: Promise<void>[] = [];
  for (const item of items) {
    let entry: EntryLike | null = null;
    try {
      entry = item.webkitGetAsEntry?.() ?? null;
    } catch {
      entry = null;
    }
    if (entry) tasks.push(walk(entry, out));
  }
  await Promise.all(tasks);

  // 部分场景（合成的 DataTransfer、特殊浏览器）拿不到 entry，退回直接用 files
  if (out.length === 0 && dataTransfer.files && dataTransfer.files.length > 0) {
    return Array.from(dataTransfer.files).filter((f) => isImageName(f.name));
  }
  return out;
}

/** 自然排序：img2 排在 img10 前面 */
export function naturalCompare(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

/* ============================================================================
   数据集导入（图片 + 已有标注）
   ========================================================================== */

/** 标注/配置类文件：YOLO 的 labels/*.txt、classes.txt、data.yaml 等 */
const TEXT_EXT = /\.(txt|yaml|yml|names)$/i;

export interface DatasetTextFile {
  /** 相对路径，如 `labels/a.txt`；拿不到 entry 时退化为文件名 */
  path: string;
  file: File;
}

export interface ExtractedDataset {
  images: File[];
  texts: DatasetTextFile[];
}

/**
 * 遍历目录树，收集图片与文本标注，并保留相对路径。
 * 与 walk() 的区别：以前只收图片，标注文件被静默丢掉，
 * 导致「导入一个已标注好的 YOLO 数据集 = 图进来了、框全没了」。
 */
function walkDataset(entry: EntryLike, prefix: string, out: DatasetTextFile[]): Promise<void> {
  if (entry.isFile) {
    if (!isImageName(entry.name) && !TEXT_EXT.test(entry.name)) return Promise.resolve();
    return new Promise((resolve) => {
      try {
        entry.file(
          (file) => {
            out.push({ path: prefix + entry.name, file });
            resolve();
          },
          () => resolve(),
        );
      } catch {
        resolve();
      }
    });
  }

  if (!entry.isDirectory) return Promise.resolve();

  return new Promise((resolve) => {
    let reader: ReturnType<EntryLike['createReader']>;
    try {
      reader = entry.createReader();
    } catch {
      resolve();
      return;
    }
    const collected: EntryLike[] = [];
    const readBatch = () => {
      reader.readEntries(
        (batch) => {
          if (!batch || batch.length === 0) {
            void Promise.all(
              collected.map((e) => walkDataset(e, `${prefix}${entry.name}/`, out)),
            ).then(() => resolve());
            return;
          }
          collected.push(...batch);
          readBatch();
        },
        () => resolve(),
      );
    };
    readBatch();
  });
}

/** 拖入文件夹/多文件：同时取出图片与标注文件 */
export async function extractDataset(dataTransfer: DataTransfer): Promise<ExtractedDataset> {
  const items = Array.from(dataTransfer.items ?? []) as unknown as ItemLike[];
  const canUseEntries = items.some((i) => typeof i.webkitGetAsEntry === 'function');

  const collected: DatasetTextFile[] = [];
  if (canUseEntries) {
    const tasks: Promise<void>[] = [];
    for (const item of items) {
      let entry: EntryLike | null = null;
      try {
        entry = item.webkitGetAsEntry?.() ?? null;
      } catch {
        entry = null;
      }
      if (entry) tasks.push(walkDataset(entry, '', collected));
    }
    await Promise.all(tasks);
  }

  // 拿不到 entry（合成 DataTransfer / 某些浏览器）时退回 files，此时没有路径信息
  if (collected.length === 0 && dataTransfer.files && dataTransfer.files.length > 0) {
    for (const file of Array.from(dataTransfer.files)) {
      if (isImageName(file.name) || TEXT_EXT.test(file.name)) {
        collected.push({ path: file.name, file });
      }
    }
  }

  const images: File[] = [];
  const texts: DatasetTextFile[] = [];
  for (const item of collected) {
    if (isImageName(item.file.name)) images.push(item.file);
    else texts.push(item);
  }
  return { images, texts };
}
