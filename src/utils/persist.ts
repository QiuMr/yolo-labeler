import type { Annotation, ImageItem, Label, PersistAnns, PersistBlob, PersistMeta } from '../types';

/**
 * 自动保存：IndexedDB 增量写入
 * - meta      项目级信息（标签、图片顺序、尺寸）
 * - blobs     图片二进制，只在新增图片时写一次
 * - anns      标注数据，只在内容变化时写
 * 刷新页面后可以无缝接着标，不需要反复导入图片。
 */

const DB_NAME = 'yolo-labeler';
const DB_VERSION = 1;
const STORE_META = 'meta';
const STORE_BLOBS = 'blobs';
const STORE_ANNS = 'anns';
const META_KEY = 'project';

let dbPromise: Promise<IDBDatabase | null> | null = null;

// 已落盘状态，用于增量对比
const savedBlobIds = new Set<string>();
const savedAnns = new Map<string, Annotation[]>();
let savedOrder: string[] = [];
let enabled = true;

function openDB(): Promise<IDBDatabase | null> {
  if (!enabled) return Promise.resolve(null);
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve) => {
    if (typeof indexedDB === 'undefined') {
      enabled = false;
      resolve(null);
      return;
    }
    let req: IDBOpenDBRequest;
    try {
      req = indexedDB.open(DB_NAME, DB_VERSION);
    } catch {
      enabled = false;
      resolve(null);
      return;
    }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_META)) db.createObjectStore(STORE_META);
      if (!db.objectStoreNames.contains(STORE_BLOBS)) db.createObjectStore(STORE_BLOBS, { keyPath: 'id' });
      if (!db.objectStoreNames.contains(STORE_ANNS)) db.createObjectStore(STORE_ANNS, { keyPath: 'id' });
    };
    req.onsuccess = () => {
      const db = req.result;
      db.onversionchange = () => db.close();
      resolve(db);
    };
    req.onerror = () => {
      enabled = false;
      resolve(null);
    };
    req.onblocked = () => {
      enabled = false;
      resolve(null);
    };
  });

  return dbPromise;
}

function toPromise<T>(req: IDBRequest<T>): Promise<T | null> {
  return new Promise((resolve) => {
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => resolve(null);
  });
}

function txDone(tx: IDBTransaction): Promise<{ ok: boolean; error?: DOMException }> {
  return new Promise((resolve) => {
    tx.oncomplete = () => resolve({ ok: true });
    // 失败时必须把 error 带出去：配额写满时如果当作成功，
    // 用户会以为数据已经存好，刷新后才发现全没了。
    const fail = () => resolve({ ok: false, error: tx.error ?? undefined });
    tx.onerror = fail;
    tx.onabort = fail;
  });
}

/** 浏览器存储写满（各浏览器报错名不完全一致，都要认） */
function isQuotaError(e: unknown): boolean {
  const err = e as { name?: string; code?: number } | null | undefined;
  if (!err) return false;
  return (
    err.name === 'QuotaExceededError' ||
    err.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
    err.code === 22
  );
}

export interface LoadedProject {
  images: ImageItem[];
  labels: Label[];
  activeLabelId: string;
  /** 原始数据丢失、只剩标注的图片数量（blob 被浏览器清掉了） */
  missingImages: number;
}

/** 保存失败的原因：quota=存储写满，unavailable=IndexedDB 不可用，error=其它写入失败 */
export type PersistFailure = 'quota' | 'unavailable' | 'error';

export interface PersistOutcome {
  ok: boolean;
  reason?: PersistFailure;
}

export async function loadProject(): Promise<LoadedProject | null> {
  const db = await openDB();
  if (!db) return null;

  let tx: IDBTransaction;
  try {
    tx = db.transaction([STORE_META, STORE_BLOBS, STORE_ANNS], 'readonly');
  } catch {
    return null;
  }

  const meta = await toPromise<PersistMeta>(tx.objectStore(STORE_META).get(META_KEY));
  if (!meta || !Array.isArray(meta.images) || meta.images.length === 0) return null;

  const blobs = (await toPromise<PersistBlob[]>(tx.objectStore(STORE_BLOBS).getAll())) ?? [];
  const anns = (await toPromise<PersistAnns[]>(tx.objectStore(STORE_ANNS).getAll())) ?? [];
  await txDone(tx);

  const blobMap = new Map<string, Blob>();
  for (const b of blobs) if (b && b.id) blobMap.set(b.id, b.blob);
  const annMap = new Map<string, Annotation[]>();
  for (const a of anns) if (a && a.id) annMap.set(a.id, Array.isArray(a.annotations) ? a.annotations : []);

  const images: ImageItem[] = [];
  let missingImages = 0;

  for (const m of meta.images) {
    if (!m || !m.id) continue;

    const blob = blobMap.get(m.id);
    let file: File | null = null;
    if (blob) {
      try {
        // 关键：把 meta 里记下的原始 lastModified 还原回去。
        // 不传的话规范默认 Date.now()，每次刷新得到的都是「刚恢复」的假时间，
        // 判重指纹 name::size::假时间 与真实文件 name::size::真时间 对不上，
        // 于是同一批图第二次导入时全部重复入库（导出时变成 cat_2.txt 那种副本）。
        file = new File([blob], m.name, {
          type: blob.type || 'image/*',
          lastModified: m.lastModified || 0,
        });
      } catch {
        file = null;
      }
    }

    const annotations = annMap.get(m.id) ?? [];
    if (!file) {
      // 原始数据没了（例如浏览器清了站点数据）。
      // 以前这里直接 continue，图片连标注一起消失且毫无提示 —— 标注是用户的心血，必须留住。
      // 保留成 file=null 的条目：画布上会明确提示「这张图打不开了」，
      // 重新导入同名图片即可自动接回去（见 useStore.addImages 的续接逻辑）。
      missingImages += 1;
      images.push({
        id: m.id,
        name: m.name,
        file: null,
        width: m.width || 0,
        height: m.height || 0,
        annotations,
      });
      continue;
    }

    savedBlobIds.add(m.id);
    savedAnns.set(m.id, annotations);
    savedOrder.push(m.id);
    images.push({
      id: m.id,
      name: m.name,
      file,
      width: m.width || 0,
      height: m.height || 0,
      annotations,
    });
  }

  if (images.length === 0) return null;

  const labels = Array.isArray(meta.labels) ? meta.labels.filter((l) => l && l.id) : [];
  const activeLabelId =
    meta.activeLabelId && labels.some((l) => l.id === meta.activeLabelId)
      ? meta.activeLabelId
      : (labels[0]?.id ?? '');

  return { images, labels, activeLabelId, missingImages };
}

export async function persistProject(
  images: ImageItem[],
  labels: Label[],
  activeLabelId: string,
): Promise<PersistOutcome> {
  const db = await openDB();
  // IndexedDB 不可用（隐私模式 / 被禁用）：明确告知，别让用户以为已经存好了
  if (!db) return { ok: false, reason: 'unavailable' };

  let tx: IDBTransaction;
  try {
    tx = db.transaction([STORE_META, STORE_BLOBS, STORE_ANNS], 'readwrite');
  } catch {
    return { ok: false, reason: 'error' };
  }

  const metaStore = tx.objectStore(STORE_META);
  const blobStore = tx.objectStore(STORE_BLOBS);
  const annStore = tx.objectStore(STORE_ANNS);

  if (images.length === 0) {
    metaStore.delete(META_KEY);
    blobStore.clear();
    annStore.clear();
    const cleared = await txDone(tx);
    if (!cleared.ok) {
      return { ok: false, reason: isQuotaError(cleared.error) ? 'quota' : 'error' };
    }
    // 清空确实成功了才重置内存标记；失败就保持原样，让下一次保存重新尝试
    savedBlobIds.clear();
    savedAnns.clear();
    savedOrder = [];
    return { ok: true };
  }

  const meta: PersistMeta = {
    version: 1,
    labels,
    activeLabelId,
    images: images.map((i) => ({
      id: i.id,
      name: i.name,
      width: i.width,
      height: i.height,
      // 记下原始修改时间：刷新恢复后判重指纹仍能和真实文件对上（见 types/index.ts 注释）
      lastModified: i.file?.lastModified ?? 0,
    })),
  };
  metaStore.put(meta, META_KEY);

  const currentIds = new Set<string>();
  // 写请求发出时先只记「该标记哪些」，等事务真的提交成功才更新内存里的落盘标记。
  // 原来的写法是一 put 就置位：配额写满导致事务 abort 时标记已经生效，
  // 用户清出空间再存会整批跳过，数据悄无声息地丢掉。
  const blobToMark: string[] = [];
  const annsToMark: Array<{ id: string; annotations: Annotation[] }> = [];

  for (const img of images) {
    currentIds.add(img.id);
    if (!savedBlobIds.has(img.id) && img.file) {
      blobStore.put({ id: img.id, blob: img.file } satisfies PersistBlob);
      blobToMark.push(img.id);
    }
    const prev = savedAnns.get(img.id);
    if (prev !== img.annotations) {
      annStore.put({ id: img.id, annotations: img.annotations } satisfies PersistAnns);
      annsToMark.push({ id: img.id, annotations: img.annotations });
    }
  }

  const removedIds: string[] = [];
  for (const id of savedOrder) {
    if (currentIds.has(id)) continue;
    blobStore.delete(id);
    annStore.delete(id);
    removedIds.push(id);
  }

  const res = await txDone(tx);
  if (!res.ok) {
    return { ok: false, reason: isQuotaError(res.error) ? 'quota' : 'error' };
  }

  // 事务已提交，落盘标记现在才可信
  for (const id of blobToMark) savedBlobIds.add(id);
  for (const a of annsToMark) savedAnns.set(a.id, a.annotations);
  for (const id of removedIds) {
    savedBlobIds.delete(id);
    savedAnns.delete(id);
  }
  savedOrder = images.map((i) => i.id);
  return { ok: true };
}

export async function clearProject(): Promise<void> {
  const db = await openDB();
  if (!db) return;
  try {
    const tx = db.transaction([STORE_META, STORE_BLOBS, STORE_ANNS], 'readwrite');
    tx.objectStore(STORE_META).delete(META_KEY);
    tx.objectStore(STORE_BLOBS).clear();
    tx.objectStore(STORE_ANNS).clear();
    await txDone(tx);
  } catch {
    /* 忽略 */
  }
  savedBlobIds.clear();
  savedAnns.clear();
  savedOrder = [];
}
