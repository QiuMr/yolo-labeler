import { create } from 'zustand';
import type { AddImagesResult, Annotation, ImageItem, IRect, Label } from '../types';
import { colorForIndex } from '../utils/color';
import { clampRectToImage } from '../utils/geometry';
import { uid } from '../utils/id';
import { naturalCompare } from '../utils/files';
import { baseKey } from '../utils/yoloDataset';
import { decodeBitmap, dropImage, makeImageId, registerSource } from '../utils/imageCache';
import { isImageFile } from '../utils/media';
import { toast } from './toastStore';

const HISTORY_LIMIT = 100;

/**
 * 撤销记录：只记录被改动的那张图，避免整棵状态树快照。
 * labels 非空时表示这一步同时也改动了标签（例如删除标签）。
 * restoreImage / removedImage 用于「移除图片」这种整图增删的操作。
 * 注意：restoreImage.item.file 保留的是原 File 引用（不复制二进制），
 * 撤销时重新登记进图片缓存即可，代价可忽略。
 */
/** 导入已标注数据集的结果 */
export interface YoloImportResult {
  /** 新增图片数 */
  added: number;
  skipped: number;
  /** 新建的类别数 */
  newLabels: number;
  /** 实际写入的框数 */
  boxes: number;
  /** 找到标注但没能写进去的图片数（通常是尺寸测量失败） */
  unresolved: number;
  /**
   * 因标注文件同名冲突而**故意跳过**的图片数。
   * 数据集里存在 `a/x.jpg` 与 `b/x.jpg` 这类同名图时，按基名匹配会把一份的框套到另一份上，
   * 属于静默产出错误标签，所以宁可跳过并让用户知道。
   */
  ambiguous: number;
}

interface HistoryEntry {
  imageId: string | null;
  annotations: Annotation[];
  labels: Label[] | null;
  /** 撤销时把这张被移除的图插回原来的位置 */
  restoreImage?: { index: number; item: ImageItem };
  /** 撤销时移除 imageId 对应的图片（重做「移除图片」时使用） */
  removedImage?: boolean;
  /**
   * 图片顺序快照（imageId 顺序）。排序 / 拖拽重排时记录，撤销即恢复原顺序。
   * 有了它，排序才可被撤销；同时保证「移除图片 → 排序 → 撤销」时
   * restoreImage.index 对应的顺序已经被先恢复，图片能插回正确位置。
   */
  order?: string[];
}



function makeLabel(name: string, index: number): Label {
  return { id: uid('lb_'), name, color: colorForIndex(index) };
}

/** 图片指纹：判重用，名字 + 体积 + 修改时间基本不会撞 */
function fileKey(f: { name: string; size?: number; lastModified?: number }): string {
  return `${f.name}::${f.size ?? 0}::${f.lastModified ?? 0}`;
}

/** 移除图片时释放了位图和缩略图，撤销回来要重新登记原始文件才能再次解码 */
function reviveImage(item: ImageItem): void {
  if (item.file) registerSource(item.id, item.file);
}

interface AppState {
  /** 是否已完成本地工程恢复 */
  hydrated: boolean;
  images: ImageItem[];
  activeIndex: number;
  labels: Label[];
  activeLabelId: string;
  selectedId: string | null;
  /** 右侧列表悬停高亮 */
  hoverId: string | null;
  /** 右侧列表多选（批量改类别用），单选时与 selectedId 保持一致 */
  selectedBoxIds: string[];
  /** 隐藏的类别 id：画布上不绘制、不参与命中，但导出不受影响 */
  hiddenLabelIds: string[];
  /**
   * 「标签集合或归属」版本号。只在下面这些操作里 +1：
   * 建/删/改名类别、框的类别变更、框增删、AI 落框、导入数据集。
   * **纯几何编辑（拖动、缩放框）不动它** —— 类别计数没变。
   * useLabelCounts 用它做缓存 key，避免「拖一下框边」就重算 6 万次迭代。
   */
  labelRevision: number;
  past: HistoryEntry[];
  future: HistoryEntry[];

  markHydrated: () => void;
  hydrateProject: (data: { images: ImageItem[]; labels: Label[]; activeLabelId: string }) => void;

  addImages: (files: File[]) => AddImagesResult;
  /**
   * 导入「已标注」的 YOLO 数据集：先按 addImages 的规则收图，再把 classes.txt / data.yaml
   * 的类别与 labels/*.txt 的框合并进来。
   * 刻意不进撤销栈——一次可能导入上千张图，塞进历史栈既占内存又没法逐步回退。
   */
  importYoloDataset: (payload: {
    files: File[];
    classNames: string[];
    /** 图片基名（小写无扩展名）→ 归一化标注；歧义 key 已被剔除 */
    anns: Map<string, Array<{ classIndex: number; cx: number; cy: number; w: number; h: number }>>;
    /**
     * 因「同一基名有多份内容不同的标注文件」而被剔除的基名。
     * 这些图**不会**被套上任何标注（宁可少导入也不静默错标），导入结果里会如实上报。
     */
    ambiguous: string[];
  }) => Promise<YoloImportResult>;
  removeImage: (id: string) => void;
  setActiveIndex: (index: number) => void;
  stepImage: (delta: number) => void;
  /** 跳到「还没有任何框」的下一张图（环形查找）。全标完了返回 false */
  gotoNextUnannotated: () => boolean;
  setImageSize: (id: string, width: number, height: number) => void;

  addLabel: (name?: string) => string;
  /** 返回是否真的改了（重名或空名字会被拒绝） */
  renameLabel: (id: string, name: string) => boolean;
  setLabelColor: (id: string, color: string) => void;
  removeLabel: (id: string) => void;
  setActiveLabel: (id: string) => void;

  /** 返回新框 id，便于画完后就地弹出标签选择器 */
  addBox: (rect: IRect) => string | null;
  updateBox: (annId: string, rect: IRect) => void;
  setBoxLabel: (annId: string, labelId: string) => void;
  deleteBox: (annId: string) => void;
  duplicateBox: (annId: string) => void;
  /** AI 自动标注：按类别名批量落框，类别不存在时自动创建，整体只占一步撤销 */
  applyAiBoxes: (
    imageId: string,
    entries: Array<{ rect: IRect; className: string }>,
    imgW: number,
    imgH: number,
  ) => void;

  select: (id: string | null) => void;
  setHover: (id: string | null) => void;
  toggleLabelHidden: (id: string) => void;
  clearImageBoxes: () => void;
  clearAllBoxes: () => void;
  resetProject: () => void;
  /** 用另一份完整项目替换当前项目（「从备份恢复」用） */
  replaceProject: (data: { images: ImageItem[]; labels: Label[]; activeLabelId: string }) => void;

  /**
   * 排序与手动拖拽重排：只改顺序，但同样进撤销栈，
   * 且始终保持「当前正在看的那张图」不变（否则一点排序画布就跳到别的图了）
   */
  sortImages: (by: 'name' | 'name-desc') => void;
  reorderImages: (from: number, to: number) => void;

  /** 批量提效：复制上一张图的标注到当前图 */
  copyPrevAnnotations: () => boolean;
  /** 把当前激活类别套用到本图所有未分类框 */
  applyActiveToUnclassified: () => boolean;

  /** 右侧列表多选（批量改类别） */
  toggleSelectBox: (id: string, exclusive: boolean) => void;
  clearSelectBoxes: () => void;
  /** 返回真正被改动的框数；0 表示无变化（不压历史、不弹成功提示） */
  applyLabelToBoxes: (ids: string[], labelId: string) => number;

  /** 导出前补齐未打开图片的尺寸（尺寸缺失时会被导出跳过） */
  ensureImageSizes: (onProgress?: (done: number, total: number) => void) => Promise<void>;

  undo: () => void;
  redo: () => void;
}

export const useStore = create<AppState>((set, get) => {
  function pushEntry(entry: HistoryEntry): void {
    const past = get().past.slice(-(HISTORY_LIMIT - 1));
    past.push(entry);
    set({ past, future: [] });
  }

  /** 记录修改前的状态，供撤销使用（labels 传 null 表示这一步没动标签） */
  function pushHistory(imageId: string | null, labels: Label[] | null = null): void {
    const state = get();
    const img = imageId ? state.images.find((i) => i.id === imageId) : null;
    if (imageId && !img) return;
    pushEntry({ imageId, annotations: img ? img.annotations : [], labels });
  }

  function pushMany(entries: HistoryEntry[]): void {
    if (entries.length === 0) return;
    const past = get().past.concat(entries).slice(-HISTORY_LIMIT);
    set({ past, future: [] });
  }

  /** 反向记录：把「当前值」存进另一条栈，供重做使用 */
  function inverseOf(entry: HistoryEntry, state: AppState): HistoryEntry {
    const current = entry.imageId ? state.images.find((i) => i.id === entry.imageId) : undefined;
    const base: HistoryEntry = {
      imageId: entry.imageId,
      annotations: current ? current.annotations : [],
      labels: entry.labels ? state.labels : null,
    };
    // 顺序的逆操作，就是把「现在的顺序」存下来
    if (entry.order) return { ...base, order: state.images.map((i) => i.id) };
    // 撤销方向是「插回图片」→ 重做方向就是「删掉它」
    if (entry.restoreImage) return { ...base, removedImage: true };
    // 撤销方向是「删掉图片」→ 重做方向就是「插回原位」
    if (entry.removedImage && current) {
      return { ...base, restoreImage: { index: state.images.indexOf(current), item: current } };
    }
    return base;
  }

  function indexOfImage(state: AppState, imageId: string | null): number {
    if (!imageId) return state.activeIndex;
    const index = state.images.findIndex((i) => i.id === imageId);
    return index < 0 ? state.activeIndex : index;
  }

  /**
   * 递增「标签分布版本号」。凡是**类别计数会变**的操作都要调：
   * 建/删类别、框的类别变更、框增删、AI 落框、导入数据集。
   * 纯几何编辑（updateBox / 拖动 / 缩放）**不要**调 —— 那样每次拖框边
   * 都会让 useLabelCounts 缓存失效并重算全图集（6 万次迭代量级）。
   */
  function bumpLabelRevision(partial: Partial<AppState>): Partial<AppState> {
    return { ...partial, labelRevision: get().labelRevision + 1 };
  }

  /**
   * 按 id 顺序重排。order 里不存在的（期间新增的图）追加到末尾，
   * 已不存在于当前列表的 id 直接忽略 —— 保证不会因为历史记录过期而丢图。
   */
  function reorderById(images: ImageItem[], order: string[]): ImageItem[] {
    const byId = new Map(images.map((i) => [i.id, i] as const));
    const out: ImageItem[] = [];
    for (const id of order) {
      const it = byId.get(id);
      if (it) {
        out.push(it);
        byId.delete(id);
      }
    }
    for (const it of byId.values()) out.push(it);
    return out;
  }

  /**
   * 顺序变化后重新定位「当前正在看的那张图」。
   * 不做这一步的话，点一下排序画布就会跳到另一个位置的别的图。
   */
  function activeIndexFor(state: AppState, images: ImageItem[]): number {
    const currentId = state.images[state.activeIndex]?.id ?? null;
    if (!currentId) return 0;
    const at = images.findIndex((i) => i.id === currentId);
    if (at >= 0) return at;
    // 当前图已经不在列表里了，夹到有效范围
    return Math.min(state.activeIndex, Math.max(0, images.length - 1));
  }

  /** 撤销/重做共用的落库逻辑 */
  function applyEntry(entry: HistoryEntry, state: AppState) {
    let images = state.images;
    let activeIndex = state.activeIndex;

    if (entry.order) {
      images = reorderById(state.images, entry.order);
      activeIndex = activeIndexFor(state, images);
    } else if (entry.removedImage && entry.imageId) {
      images = images.filter((i) => i.id !== entry.imageId);
      activeIndex = Math.min(state.activeIndex, Math.max(0, images.length - 1));
    } else if (entry.restoreImage) {
      images = images.slice();
      const at = Math.max(0, Math.min(entry.restoreImage.index, images.length));
      images.splice(at, 0, entry.restoreImage.item);
      activeIndex = Math.min(at, Math.max(0, images.length - 1));
    } else if (entry.imageId) {
      images = images.map((i) => (i.id === entry.imageId ? { ...i, annotations: entry.annotations } : i));
      activeIndex = indexOfImage(state, entry.imageId);
    }

    const labels = entry.labels ?? state.labels;
    const selectedId =
      entry.imageId && entry.annotations.some((a) => a.id === state.selectedId) ? state.selectedId : null;
    return {
      images,
      labels,
      activeLabelId: labels.some((l) => l.id === state.activeLabelId)
        ? state.activeLabelId
        : (labels[0]?.id ?? ''),
      activeIndex,
      selectedId: entry.imageId && !entry.restoreImage ? selectedId : state.selectedId,
    };
  }

  /** 归一当前选中标签；没有任何标签时返回空串，代表「未分类」 */
  function normalizeActive(state: AppState): string {
    if (state.labels.length === 0) return '';
    return state.labels.some((l) => l.id === state.activeLabelId)
      ? state.activeLabelId
      : state.labels[0].id;
  }

  function replaceAnnotations(imageId: string, next: Annotation[]): void {
    set((state) => ({
      images: state.images.map((i) => (i.id === imageId ? { ...i, annotations: next } : i)),
    }));
  }

  return {
    hydrated: false,
    images: [],
    activeIndex: 0,
    labels: [],
    activeLabelId: '',
    selectedId: null,
    hoverId: null,
    selectedBoxIds: [],
    hiddenLabelIds: [],
    labelRevision: 0,
    past: [],
    future: [],

    markHydrated: () => set({ hydrated: true }),

    hydrateProject: (data) =>
      set(() => {
        const labels = data.labels;
        const activeLabelId = labels.some((l) => l.id === data.activeLabelId)
          ? data.activeLabelId
          : (labels[0]?.id ?? '');
        return {
          images: data.images,
          labels,
          activeLabelId,
          activeIndex: 0,
          selectedId: null,
          hoverId: null,
          selectedBoxIds: [],
          hiddenLabelIds: [],
          // 递增而不是归零：useLabelCounts 的模块级缓存按版本号存，
          // 归零有可能与上一个项目的版本号撞上（上一个恰好是 0 时）→ 命中脏缓存。
          labelRevision: get().labelRevision + 1,
          past: [],
          future: [],
          hydrated: true,
        };
      }),

    addImages: (files) => {
      const list = files.filter(isImageFile);
      if (list.length === 0) return { added: 0, skipped: 0, revived: 0 };

      const state = get();
      // 同一张图拖进来两次会变成两个一模一样的条目，这里按「名字 + 大小 + 修改时间」判重
      const seen = new Set(
        state.images.map((i) => fileKey({ name: i.name, size: i.file?.size, lastModified: i.file?.lastModified })),
      );
      // 恢复出来的条目（从 IndexedDB / 备份重建）拿不到真实修改时间时（lastModified 为 0），
      // 严格指纹必然对不上真实文件。这里额外记一份「名字 + 大小」的宽松指纹，
      // 只用于这些时间未知的条目，避免刷新后同一批图被反复导入成重复条目。
      const seenLoose = new Set(
        state.images
          .filter((i) => (i.file?.lastModified ?? 0) === 0)
          .map((i) => `${i.name}::${i.file?.size ?? 0}`),
      );

      // 「图片数据丢了但标注还在」的条目（例如浏览器清了站点数据）：
      // 按名字索引，用户重新导入同名图片时直接把图接回去，而不是新建一个空条目。
      // 丢失条目的指纹是 name::0::0，不会和真实文件冲突。
      const orphans = new Map<string, string>();
      for (const img of state.images) {
        if (!img.file && !orphans.has(img.name)) orphans.set(img.name, img.id);
      }

      const fresh: File[] = [];
      const revived = new Map<string, File>();
      let skipped = 0;
      for (const file of list) {
        const key = fileKey(file);
        const loose = `${file.name}::${file.size}`;
        if (seen.has(key) || seenLoose.has(loose)) {
          skipped += 1;
          continue;
        }
        seen.add(key);
        seenLoose.add(loose);
        const orphanId = orphans.get(file.name);
        if (orphanId) {
          revived.set(orphanId, file);
          orphans.delete(file.name);
        } else {
          fresh.push(file);
        }
      }
      if (fresh.length === 0 && revived.size === 0) {
        return { added: 0, skipped, revived: 0 };
      }

      const items: ImageItem[] = fresh
        .map((file) => {
          const id = makeImageId();
          registerSource(id, file);
          return { id, name: file.name, file, width: 0, height: 0, annotations: [] };
        })
        .sort((a, b) => naturalCompare(a.name, b.name));

      set((s) => {
        // 先把丢失的图原地补回去（保留原有标注，不新增条目）
        const images = s.images.map((img) => {
          const file = revived.get(img.id);
          if (!file) return img;
          registerSource(img.id, file);
          return { ...img, file };
        });
        return {
          images: [...images, ...items],
          activeLabelId: normalizeActive(s),
          activeIndex: s.images.length === 0 ? 0 : s.activeIndex,
          selectedId: null,
        };
      });
      return { added: items.length, skipped, revived: revived.size };
    },

    // ---------- 导入已标注的 YOLO 数据集 ----------
    importYoloDataset: async ({ files, classNames, anns, ambiguous }) => {
      const addResult = get().addImages(files);
      if (files.length === 0) {
        return {
          added: 0,
          skipped: addResult.skipped,
          newLabels: 0,
          boxes: 0,
          unresolved: 0,
          ambiguous: 0,
        };
      }

      // ① 合并类别：同名复用已有类别（保住颜色与已有框的类别），缺失的按调色板新建
      const state0 = get();
      const existing = new Map(state0.labels.map((l) => [l.name, l.id] as const));
      const labelIdByIndex: string[] = [];
      const fresh: Label[] = [];
      classNames.forEach((raw, i) => {
        const name = raw.trim();
        if (!name) return;
        const hit = existing.get(name);
        if (hit) {
          labelIdByIndex[i] = hit;
          return;
        }
        // 类别表里有重名项：只认第一个，避免建出两个同名类别
        if (fresh.some((l) => l.name === name)) return;
        const id = uid('lb_');
        fresh.push({ id, name, color: colorForIndex(fresh.length) });
        labelIdByIndex[i] = id;
      });
      if (fresh.length > 0) {
        set((s) => ({
          labels: [...s.labels, ...fresh],
          activeLabelId: s.activeLabelId || fresh[0].id,
        }));
      }

      // ② 归一化坐标要乘图片像素，必须先拿到尺寸
      await get().ensureImageSizes();

      // ③ 写框。只碰「还没有任何框」且有对应标注的图片，绝不覆盖用户已有标注
      let boxes = 0;
      let unresolved = 0;
      // 标注文件同名冲突的 key 已在 parseYoloDataset 里从 anns 剔除（宁可少导入也不静默错标）。
      // 这里无法区分「这张图本来就没标注」和「因重名被剔除」，因此按可能受影响的张数上报，
      // 文案会写明是「可能」，让用户自己核对，不当成确定结论。
      const ambiguousKeys = new Set(ambiguous);
      let ambiguousSkipped = 0;
      set((s) =>
        bumpLabelRevision({
        images: s.images.map((img) => {
          if (img.annotations.length > 0) return img;
          const key = baseKey(img.name);
          const list = anns.get(key);
          if (!list || list.length === 0) {
            if (ambiguousKeys.has(key)) ambiguousSkipped += 1;
            return img;
          }
          const W = img.width;
          const H = img.height;
          if (W <= 0 || H <= 0) {
            unresolved += 1;
            return img;
          }
          const created: Annotation[] = [];
          for (const b of list) {
            const w = b.w * W;
            const h = b.h * H;
            if (!(w >= 1) || !(h >= 1)) continue;
            const x = Math.max(0, Math.min(W - w, (b.cx - b.w / 2) * W));
            const y = Math.max(0, Math.min(H - h, (b.cy - b.h / 2) * H));
            created.push({
              id: uid('an_'),
              labelId: labelIdByIndex[b.classIndex] ?? '',
              rect: { x, y, width: w, height: h },
            });
          }
          if (created.length === 0) return img;
          boxes += created.length;
          return { ...img, annotations: created };
        }),
      }),
      );

      return {
        added: addResult.added,
        skipped: addResult.skipped,
        newLabels: fresh.length,
        boxes,
        unresolved,
        ambiguous: ambiguousSkipped,
      };
    },

    removeImage: (id) => {
      const state = get();
      const index = state.images.findIndex((i) => i.id === id);
      if (index < 0) return;
      const item = state.images[index];
      dropImage(id);
      set((s) => {
        const images = s.images.filter((i) => i.id !== id);
        // 清掉这张图残留的历史，再压一条「可插回」的记录 —— 保证 Ctrl+Z 一定能救回来
        const past = s.past
          .filter((e) => e.imageId !== id)
          .slice(-(HISTORY_LIMIT - 1));
        past.push({
          imageId: id,
          annotations: item.annotations,
          labels: null,
          restoreImage: { index, item },
        });
        // 移除图片会带走它身上的框 → 类别计数变了，必须 bump
        // （早先漏了这里，useLabelCounts 换成按版本号缓存后就会显示过期计数）
        return bumpLabelRevision({
          images,
          activeIndex: Math.min(s.activeIndex, Math.max(0, images.length - 1)),
          selectedId: null,
          selectedBoxIds: [],
          past,
          future: s.future.filter((e) => e.imageId !== id),
        });
      });
    },

    setActiveIndex: (index) =>
      set((s) => {
        if (s.images.length === 0) return { activeIndex: 0, selectedId: null, selectedBoxIds: [] };
        const next = Math.max(0, Math.min(s.images.length - 1, index));
        // 返回原 state 引用：zustand 会跳过通知，避免空点也惊动所有订阅者
        if (next === s.activeIndex) return s;
        // 批量多选是「当前图内」的概念，换图必须一起清掉。
        // 否则 ids 仍指向上张图，套用类别时会「一个框都没改」却提示成功。
        return { activeIndex: next, selectedId: null, selectedBoxIds: [] };
      }),

    stepImage: (delta) => {
      const state = get();
      if (state.images.length === 0) return;
      const next = state.activeIndex + delta;
      if (next < 0 || next >= state.images.length) return;
      set({ activeIndex: next, selectedId: null, selectedBoxIds: [] });
    },

    gotoNextUnannotated: () => {
      const state = get();
      const n = state.images.length;
      if (n === 0) return false;
      // 从当前图的下一张开始环形找「一个框都没有」的图
      for (let step = 1; step <= n; step += 1) {
        const idx = (state.activeIndex + step) % n;
        const img = state.images[idx];
        if (img.annotations.length === 0) {
          set({ activeIndex: idx, selectedId: null });
          return true;
        }
      }
      return false;
    },

    setImageSize: (id, width, height) =>
      set((s) => {
        const img = s.images.find((i) => i.id === id);
        if (!img || (img.width === width && img.height === height)) return s;
        return {
          images: s.images.map((i) => (i.id === id ? { ...i, width, height } : i)),
        };
      }),

    addLabel: (name) => {
      const state = get();
      const trimmed = name?.trim() ?? '';
      // 同名类别会让导出的数据集里出现两个一样的类别，这里直接复用已有的
      if (trimmed) {
        const existing = state.labels.find((l) => l.name === trimmed);
        if (existing) {
          set({ activeLabelId: existing.id });
          return existing.id;
        }
      }
      const idx = state.labels.length;
      const label = makeLabel(trimmed || `目标 ${idx + 1}`, idx);
      set((s) => bumpLabelRevision({ labels: [...s.labels, label], activeLabelId: label.id }));
      return label.id;
    },

    renameLabel: (id, name) => {
      const state = get();
      const trimmed = name.trim();
      if (!trimmed) return false;
      // 不允许改成别的类别已经占用的名字，否则导出的类别表会重复
      if (state.labels.some((l) => l.id !== id && l.name === trimmed)) return false;
      set({ labels: state.labels.map((l) => (l.id === id ? { ...l, name: trimmed } : l)) });
      return true;
    },

    setLabelColor: (id, color) =>
      set((s) => ({ labels: s.labels.map((l) => (l.id === id ? { ...l, color } : l)) })),

    removeLabel: (id) => {
      const state = get();
      if (!state.labels.some((l) => l.id === id)) return;

      // 只从类别表里移除，**不去改任何一张图的 annotations**。
      //
      // 以前是把命中框的 labelId 置空，于是「恢复类别归属」依赖每张图各自的历史快照；
      // 而历史栈上限 100 条（HISTORY_LIMIT），一个类别被 100+ 张图用到时，
      // 前面那些图的快照直接被 slice 掉 → 按到底 Ctrl+Z 也恢复不回来，框永久变未分类。
      //
      // 现在 annotations 原样不动，悬空的 labelId 在各处都有兜底：
      //   - Renderer.ts:255-256 走 unclassified 分支，chip 显示「未分类」
      //   - RightPanel.tsx620 label 取不到就是未分类
      //   - 三种导出都用 indexOf.get() 判 undefined 后跳过（exporters.ts:134/202/264）
      // 于是撤销只需要 labels 一条快照，**无论多少张图都能完整恢复**。
      pushHistory(null, state.labels);

      set((s) => {
        const labels = s.labels.filter((l) => l.id !== id);
        return bumpLabelRevision({
          labels,
          activeLabelId: s.activeLabelId === id ? (labels[0]?.id ?? '') : s.activeLabelId,
        });
      });
    },

    setActiveLabel: (id) => set({ activeLabelId: id }),

    addBox: (rect) => {
      const state = get();
      const index = state.activeIndex;
      const img = state.images[index];
      if (!img || img.width <= 0 || img.height <= 0) return null;

      const labelId = normalizeActive(state);
      const r = clampRectToImage(rect, img.width, img.height);
      if (r.width < 2 || r.height < 2) return null;

      pushHistory(img.id);
      const ann: Annotation = { id: uid('an_'), labelId, rect: r };

      set((s) => {
        const images = s.images.slice();
        const target = images[index];
        if (!target) return {};
        images[index] = { ...target, annotations: [...target.annotations, ann] };
        return bumpLabelRevision({ images, selectedId: ann.id });
      });
      return ann.id;
    },

    updateBox: (annId, rect) => {
      const state = get();
      const img = state.images[state.activeIndex];
      if (!img || img.width <= 0 || img.height <= 0) return;
      const ann = img.annotations.find((a) => a.id === annId);
      if (!ann) return;

      const r = clampRectToImage(rect, img.width, img.height);
      // 边缘吸附可能把两条边同时吸到同一侧，宽度被压成 0。
      // 这种框画布不会绘制（Renderer 对 w<1/h<1 直接 return），
      // 右侧列表显示成 0×N，导出 YOLO 时还会输出 0.000000 的宽度污染训练集。
      // 这里与 addBox 保持同一条底线：小于 2px 的框一律不落库。
      if (r.width < 2 || r.height < 2) return;

      const prev = ann.rect;
      if (prev.x === r.x && prev.y === r.y && prev.width === r.width && prev.height === r.height) {
        return;
      }

      pushHistory(img.id);
      replaceAnnotations(
        img.id,
        img.annotations.map((a) => (a.id === annId ? { ...a, rect: r } : a)),
      );
    },

    setBoxLabel: (annId, labelId) => {
      const state = get();
      const img = state.images[state.activeIndex];
      if (!img) return;
      const ann = img.annotations.find((a) => a.id === annId);
      if (!ann || ann.labelId === labelId) return;
      pushHistory(img.id);
      replaceAnnotations(
        img.id,
        img.annotations.map((a) => (a.id === annId ? { ...a, labelId } : a)),
      );
      // 空串代表「退回未分类」，此时底栏的默认类别不该被清掉
      set((s) => bumpLabelRevision({ activeLabelId: labelId ? labelId : s.activeLabelId }));
    },

    deleteBox: (annId) => {
      const state = get();
      const img = state.images[state.activeIndex];
      if (!img) return;
      if (!img.annotations.some((a) => a.id === annId)) return;
      pushHistory(img.id);
      replaceAnnotations(
        img.id,
        img.annotations.filter((a) => a.id !== annId),
      );
      set((s) =>
        bumpLabelRevision({ selectedId: s.selectedId === annId ? null : s.selectedId }),
      );
    },

    duplicateBox: (annId) => {
      const state = get();
      const img = state.images[state.activeIndex];
      if (!img || img.width <= 0 || img.height <= 0) return;
      const ann = img.annotations.find((a) => a.id === annId);
      if (!ann) return;

      const offset = Math.max(8, Math.round(Math.min(img.width, img.height) * 0.02));
      let rect = clampRectToImage(
        { ...ann.rect, x: ann.rect.x + offset, y: ann.rect.y + offset },
        img.width,
        img.height,
      );
      // 框已经贴着右下角时，往右下偏移会被夹回原位 —— 复制品会和原框完全重合
      if (rect.x === ann.rect.x && rect.y === ann.rect.y) {
        rect = clampRectToImage(
          { ...ann.rect, x: ann.rect.x - offset, y: ann.rect.y - offset },
          img.width,
          img.height,
        );
      }
      const copy: Annotation = { id: uid('an_'), labelId: ann.labelId, rect };

      pushHistory(img.id);
      replaceAnnotations(img.id, [...img.annotations, copy]);
      set(() => bumpLabelRevision({ selectedId: copy.id }));
    },

    applyAiBoxes: (imageId, entries, imgW, imgH) => {
      const state = get();
      const img = state.images.find((i) => i.id === imageId);
      if (!img || entries.length === 0) return;
      const W = imgW > 0 ? imgW : img.width;
      const H = imgH > 0 ? imgH : img.height;
      if (W <= 0 || H <= 0) return;

      const labels = state.labels.slice();
      const nameToId = new Map(labels.map((l) => [l.name, l.id] as const));
      let colorIndex = labels.length;
      const anns: Annotation[] = [];
      for (const en of entries) {
        let labelId = nameToId.get(en.className);
        if (!labelId) {
          const label = makeLabel(en.className, colorIndex);
          colorIndex += 1;
          labels.push(label);
          nameToId.set(en.className, label.id);
          labelId = label.id;
        }
        const rect = clampRectToImage(en.rect, W, H);
        if (rect.width >= 2 && rect.height >= 2) {
          anns.push({ id: uid('an_'), labelId, rect });
        }
      }
      if (anns.length === 0) return;

      const labelsChanged = labels.length !== state.labels.length;
      // 新建过标签时把旧标签表一并存档，撤销时类别和框一起退回
      pushEntry({
        imageId,
        annotations: img.annotations,
        labels: labelsChanged ? state.labels : null,
      });
      set((s) =>
        bumpLabelRevision({
          images: s.images.map((i) =>
            i.id === imageId
              ? { ...i, width: W, height: H, annotations: [...i.annotations, ...anns] }
              : i,
          ),
          labels: labelsChanged ? labels : s.labels,
        }),
      );
    },

    select: (id) => set({ selectedId: id }),
    setHover: (id) => set((s) => (s.hoverId === id ? s : { hoverId: id })),
    toggleLabelHidden: (id) =>
      set((s) => {
        const wasHidden = s.hiddenLabelIds.includes(id);
        const hiddenLabelIds = wasHidden
          ? s.hiddenLabelIds.filter((x) => x !== id)
          : [...s.hiddenLabelIds, id];
        // 防御：隐藏的正是当前默认类别时，画新框会"画完就隐身"，很迷惑。
        // 自动把默认换成一个仍然可见的类别。
        let activeLabelId = s.activeLabelId;
        if (!wasHidden && hiddenLabelIds.includes(activeLabelId)) {
          const fallback = s.labels.find(
            (l) => l.id !== id && !hiddenLabelIds.includes(l.id),
          );
          activeLabelId = fallback ? fallback.id : '';
        }
        return { hiddenLabelIds, activeLabelId };
      }),

    clearImageBoxes: () => {
      const state = get();
      const img = state.images[state.activeIndex];
      if (!img || img.annotations.length === 0) return;
      pushHistory(img.id);
      replaceAnnotations(img.id, []);
      set(() => bumpLabelRevision({ selectedId: null, selectedBoxIds: [] }));
    },

    clearAllBoxes: () => {
      const state = get();
      const entries: HistoryEntry[] = state.images
        .filter((i) => i.annotations.length > 0)
        .map((i) => ({ imageId: i.id, annotations: i.annotations, labels: null }));
      if (entries.length === 0) return;
      pushMany(entries);
      set((s) =>
        bumpLabelRevision({
          images: s.images.map((i) => (i.annotations.length ? { ...i, annotations: [] } : i)),
          selectedId: null,
          selectedBoxIds: [],
        }),
      );
    },

    resetProject: () => {
      for (const img of get().images) dropImage(img.id);
      set({
        images: [],
        activeIndex: 0,
        labels: [],
        activeLabelId: '',
        selectedId: null,
        hoverId: null,
        selectedBoxIds: [],
        hiddenLabelIds: [],
        labelRevision: get().labelRevision + 1,
        past: [],
        future: [],
      });
    },

    replaceProject: (data) => {
      // 释放旧图的位图/缩略图，避免内存里堆着两批图片
      for (const img of get().images) dropImage(img.id);
      const labels = data.labels;
      const activeLabelId = labels.some((l) => l.id === data.activeLabelId)
        ? data.activeLabelId
        : (labels[0]?.id ?? '');
      set({
        images: data.images,
        labels,
        activeLabelId,
        activeIndex: 0,
        selectedId: null,
        hoverId: null,
        selectedBoxIds: [],
        hiddenLabelIds: [],
        labelRevision: get().labelRevision + 1,
        past: [],
        future: [],
      });
    },

    // ---------- 排序与重排（可撤销，且始终跟着「当前正在看的图」） ----------
    sortImages: (by) => {
      const state = get();
      if (state.images.length < 2) return;
      const sorted = [...state.images].sort((a, b) =>
        by === 'name' ? naturalCompare(a.name, b.name) : naturalCompare(b.name, a.name),
      );
      // 顺序本来就没变 → 不动，也不往撤销栈里塞废记录
      if (sorted.every((img, i) => img.id === state.images[i].id)) return;

      pushEntry({
        imageId: null,
        annotations: [],
        labels: null,
        order: state.images.map((i) => i.id),
      });
      set((s) => ({
        images: sorted,
        activeIndex: activeIndexFor(s, sorted),
        selectedId: null,
        selectedBoxIds: [],
      }));
    },

    reorderImages: (from, to) => {
      const state = get();
      const n = state.images.length;
      // 防御：越界 / 原地不动 / 非整数索引一律忽略
      if (!Number.isInteger(from) || !Number.isInteger(to)) return;
      if (from < 0 || from >= n || to < 0 || to >= n || from === to) return;

      const images = state.images.slice();
      const [moved] = images.splice(from, 1);
      images.splice(to, 0, moved);
      if (images.every((img, i) => img.id === state.images[i].id)) return;

      pushEntry({
        imageId: null,
        annotations: [],
        labels: null,
        order: state.images.map((i) => i.id),
      });
      // 不清 selectedId：拖的是当前图时，正在编辑的选中状态应该保留
      set((s) => ({ images, activeIndex: activeIndexFor(s, images) }));
    },

    // ---------- 批量提效 ----------
    copyPrevAnnotations: () => {
      const state = get();
      if (state.images.length < 2) return false;
      const idx = state.activeIndex;
      if (idx <= 0) return false;
      const prev = state.images[idx - 1];
      const cur = state.images[idx];
      if (!cur || prev.annotations.length === 0) return false;
      // 防御：当前图还没量过尺寸（一次都没打开过）就不能复制 ——
      // 没有目标尺寸无从收敛，复制过去的框会整个跑出图外。
      if (cur.width <= 0 || cur.height <= 0) return false;

      const labelIds = new Set(state.labels.map((l) => l.id));
      const newAnns: Annotation[] = [];
      for (const a of prev.annotations) {
        // 上一张图里指向已删除类别的框，直接跳过（否则会复制出「未分类」的孤儿框）
        if (a.labelId && !labelIds.has(a.labelId)) continue;
        // 相邻图尺寸可能不同，必须按当前图收敛
        const rect = clampRectToImage({ ...a.rect }, cur.width, cur.height);
        // 收敛后小到没有意义就丢弃，避免塞进去一堆 1px 的空框
        if (rect.width < 2 || rect.height < 2) continue;
        newAnns.push({ id: uid('an_'), labelId: a.labelId, rect });
      }
      if (newAnns.length === 0) return false;
      pushHistory(cur.id);
      replaceAnnotations(cur.id, [...cur.annotations, ...newAnns]);
      set(() =>
        bumpLabelRevision({ selectedId: newAnns[0].id, selectedBoxIds: [newAnns[0].id] }),
      );
      return true;
    },

    applyActiveToUnclassified: () => {
      const state = get();
      const img = state.images[state.activeIndex];
      if (!img || !state.activeLabelId) return false;
      const targets = img.annotations.filter((a) => !a.labelId);
      if (targets.length === 0) return false;
      pushHistory(img.id);
      replaceAnnotations(
        img.id,
        img.annotations.map((a) => (a.labelId ? a : { ...a, labelId: state.activeLabelId })),
      );
      set(() => bumpLabelRevision({}));
      return true;
    },

    // ---------- 右侧列表多选 ----------
    toggleSelectBox: (id, exclusive) => {
      set((s) => {
        if (exclusive) return { selectedBoxIds: [id], selectedId: id };
        const has = s.selectedBoxIds.includes(id);
        return {
          selectedBoxIds: has
            ? s.selectedBoxIds.filter((x) => x !== id)
            : [...s.selectedBoxIds, id],
          selectedId: id,
        };
      });
    },

    clearSelectBoxes: () => set({ selectedBoxIds: [] }),

    /** 返回真正被改动的框数（0 表示没有任何变化，不压历史也不提示成功） */
    applyLabelToBoxes: (ids, labelId) => {
      const state = get();
      const img = state.images[state.activeIndex];
      if (!img || ids.length === 0) return 0;
      const idSet = new Set(ids);
      // 只统计「属于当前图且类别确实不同」的框。
      // 以前无条件 map + pushHistory：ids 指向别的图时一个框都没改，
      // 却压进一条内容相同的伪历史（下次 Ctrl+Z 像坏了），并弹出「已为 N 个框套用类别」。
      const targets = img.annotations.filter((a) => idSet.has(a.id) && a.labelId !== labelId);
      if (targets.length === 0) return 0;

      const targetIds = new Set(targets.map((a) => a.id));
      pushHistory(img.id);
      replaceAnnotations(
        img.id,
        img.annotations.map((a) => (targetIds.has(a.id) ? { ...a, labelId } : a)),
      );
      set((s) => bumpLabelRevision({ activeLabelId: labelId || s.activeLabelId, selectedBoxIds: [] }));
      return targets.length;
    },

    // ---------- 导出前补齐尺寸 ----------
    ensureImageSizes: async (onProgress) => {
      const missing = get().images.filter((i) => i.width <= 0 && i.height <= 0 && i.file);
      if (missing.length === 0) return;

      // 限并发：串行时几百张图要等很久，全并发又会把内存/显存吃满。
      const CONCURRENCY = 5;
      const sizes = new Map<string, { width: number; height: number }>();
      let cursor = 0;
      let done = 0;
      const total = missing.length;

      const worker = async (): Promise<void> => {
        for (;;) {
          const i = cursor;
          cursor += 1;
          if (i >= total) return;
          const img = missing[i];
          let bmp: ImageBitmap | null = null;
          try {
            bmp = await decodeBitmap(img.file as Blob);
            if (bmp.width > 0 && bmp.height > 0) {
              sizes.set(img.id, { width: bmp.width, height: bmp.height });
            }
          } catch {
            /* 尺寸读不出就跳过这张，导出时仍会被滤掉 */
          } finally {
            // 必须释放，否则批量解码会持续占用内存
            if (bmp) bmp.close();
          }
          done += 1;
          onProgress?.(done, total);
        }
      };

      await Promise.all(
        Array.from({ length: Math.min(CONCURRENCY, total) }, () => worker()),
      );

      // 一次性写回：逐张 set 会触发 N 次 store 更新和 N 次自动保存
      if (sizes.size > 0) {
        set((s) => ({
          images: s.images.map((i) => {
            const sz = sizes.get(i.id);
            return sz ? { ...i, width: sz.width, height: sz.height } : i;
          }),
        }));
      }
    },

    undo: () => {
      const state = get();
      if (state.past.length === 0) return;
      const entry = state.past[state.past.length - 1];
      // 目标图片已经不在列表里，且这条记录也不是「把图加回来」的 → 失效，直接丢弃
      const alive = !entry.imageId || state.images.some((i) => i.id === entry.imageId);
      if (!alive && !entry.restoreImage) {
        // 以前静默丢弃：用户按了 Ctrl+Z 却「什么都没发生」，
        // 还会以为撤销坏了。必须说清楚为什么没反应。
        toast.info('这一步撤不回来了：它涉及的那张图片已经被移除');
        set({ past: state.past.slice(0, -1) });
        return;
      }
      if (entry.restoreImage) reviveImage(entry.restoreImage.item);
      set({
        past: state.past.slice(0, -1),
        future: [...state.future, inverseOf(entry, state)].slice(-HISTORY_LIMIT),
        ...applyEntry(entry, state),
        // 撤销会恢复标注 → 类别计数可能变了。放在展开之后，确保不被覆盖。
        labelRevision: state.labelRevision + 1,
      });
    },

    redo: () => {
      const state = get();
      if (state.future.length === 0) return;
      const entry = state.future[state.future.length - 1];
      const alive = !entry.imageId || state.images.some((i) => i.id === entry.imageId);
      if (!alive && !entry.restoreImage) {
        set({ future: state.future.slice(0, -1) });
        return;
      }
      if (entry.restoreImage) reviveImage(entry.restoreImage.item);
      set({
        future: state.future.slice(0, -1),
        past: [...state.past, inverseOf(entry, state)].slice(-HISTORY_LIMIT),
        ...applyEntry(entry, state),
        // 同 undo：重做也会改标注
        labelRevision: state.labelRevision + 1,
      });
    },
  };
});
