/**
 * AI 模型文件的本地持久化（独立的 IndexedDB 库，不碰项目数据）。
 * 用户选过一次 .onnx 后存进来，下次打开页面自动复用，不必反复挑文件。
 */

const DB_NAME = 'yolo-labeler-ai';
const DB_VERSION = 1;
const STORE = 'models';
const KEY = 'current';

export interface SavedModel {
  id: string;
  name: string;
  size: number;
  bytes: ArrayBuffer;
}

let dbPromise: Promise<IDBDatabase | null> | null = null;

function openDB(): Promise<IDBDatabase | null> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve) => {
    if (typeof indexedDB === 'undefined') {
      resolve(null);
      return;
    }
    let req: IDBOpenDBRequest;
    try {
      req = indexedDB.open(DB_NAME, DB_VERSION);
    } catch {
      resolve(null);
      return;
    }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
    req.onblocked = () => resolve(null);
  });
  return dbPromise;
}

export async function loadSavedModel(): Promise<SavedModel | null> {
  const db = await openDB();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(KEY);
      req.onsuccess = () => {
        const m = req.result as SavedModel | undefined;
        resolve(m && m.bytes ? m : null);
      };
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

export async function saveModel(name: string, size: number, bytes: ArrayBuffer): Promise<void> {
  const db = await openDB();
  if (!db) return;
  await new Promise<void>((resolve) => {
    try {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put({ id: KEY, name, size, bytes } satisfies SavedModel);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
      tx.onabort = () => resolve();
    } catch {
      resolve();
    }
  });
}

/**
 * 用户手改的类别名，跟着模型一起记住。
 *
 * 刻意**不塞进上面那条 IndexedDB 记录**里：那条记录的 value 里带着几十 MB 的 onnx 字节，
 * 把类名放进去的话，用户每敲一个字母都要把整个大对象重写一遍。
 * 类名只是一小段文本，放 localStorage 就够；并且用**模型名做校验** ——
 * 换了模型旧名字直接失效，绝不会把 A 模型的类名安到 B 模型头上。
 *
 * 为什么必须持久化：不存的话，刷新页面后再跑一次批量标注，
 * 新框会用回「类别 1」并**新建一个同名不同 id 的标签**，
 * 于是同一类东西在左侧裂成两行 —— 用户很难自己想明白这是怎么回事。
 */
const NAMES_KEY = 'yolo-labeler:class-names';

interface SavedClassNames {
  model: string;
  names: string[];
}

export function loadClassNames(modelName: string): string[] {
  if (!modelName) return [];
  try {
    const raw = localStorage.getItem(NAMES_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Partial<SavedClassNames>;
    if (parsed?.model !== modelName || !Array.isArray(parsed.names)) return [];
    return parsed.names.filter((n): n is string => typeof n === 'string');
  } catch {
    return [];
  }
}

export function saveClassNames(modelName: string, names: string[]): void {
  if (!modelName) return;
  try {
    // 一个名字都没改（全是空串）等于没自定义，别留下无用记录，
    // 否则下次同名模型会被一段空名字覆盖
    if (!names.some((n) => n)) {
      localStorage.removeItem(NAMES_KEY);
      return;
    }
    localStorage.setItem(NAMES_KEY, JSON.stringify({ model: modelName, names }));
  } catch {
    /* 隐私模式 / 配额满：存不下不影响本次改名与标注 */
  }
}

/**
 * 置信度阈值，同样跟着模型记住。
 *
 * 为什么必须记住：不同模型该用的阈值差得很远（自训的小数据集模型常常要压到 0.1~0.15
 * 才找得全），而「调完阈值 → 刷新 / 关掉再打开 → 回 0.25」意味着**同一批图的标注结果
 * 前后不一致**，用户看到的只是「AI 这次怎么又漏了」，无从归因。
 */
const CONF_KEY = 'yolo-labeler:conf-threshold';

interface SavedConf {
  model: string;
  conf: number;
}

export function loadConf(modelName: string): number | null {
  if (!modelName) return null;
  try {
    const raw = localStorage.getItem(CONF_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<SavedConf>;
    if (parsed?.model !== modelName || typeof parsed.conf !== 'number') return null;
    // 越界值一律当没存过：阈值被写坏会让「一键标注」直接什么都不出框
    if (!Number.isFinite(parsed.conf) || parsed.conf <= 0 || parsed.conf >= 1) return null;
    return parsed.conf;
  } catch {
    return null;
  }
}

export function saveConf(modelName: string, conf: number): void {
  if (!modelName || !Number.isFinite(conf)) return;
  try {
    localStorage.setItem(CONF_KEY, JSON.stringify({ model: modelName, conf } satisfies SavedConf));
  } catch {
    /* 存不下不影响本次使用 */
  }
}
