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
