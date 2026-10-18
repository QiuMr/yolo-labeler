import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from 'react';
import { Renderer, clearTextWidthCache } from '../canvas/Renderer';
import { canvasApi } from '../canvas/canvasApi';
import { useStore } from '../store/useStore';
import { viewStore } from '../store/viewStore';
import type { Annotation, IPoint, IRect } from '../types';
import { useAiStore, type AiCandidate } from '../store/aiStore';
import { SEG_PROGRESS_DOWNLOAD_DONE, useSegStore } from '../store/segStore';
import { LabelPopover, type PopoverMode } from './LabelPopover';
import {
  ImageOff,
  Loader2,
  Maximize,
  MousePointerClick,
  RotateCcw,
  Sparkles,
  TriangleAlert,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { loadBitmap, peekBitmap, protect } from '../utils/imageCache';
import {
  applyResize,
  clamp,
  clampRectToImage,
  snapEdges,
  snapRectToEdges,
  type HandleId,
} from '../utils/geometry';

const EMPTY_ANNOTATIONS: Annotation[] = [];
const EMPTY_HIDDEN: ReadonlySet<string> = new Set();
const MIN_SCALE = 0.02;
const MAX_SCALE = 60;
const FIT_PADDING = 56;
const HANDLE_TOLERANCE = 8;
const SNAP_TOLERANCE = 7;
const MIN_DRAW_PX = 5;
/** 起笔允许略微越出图片边界，画贴边的框时不用像素级精准 */
const DRAW_EDGE_GRACE = 3;
/** 超过这个位移（屏幕像素）才算「拖动」，否则视作原地点击 */
const CLICK_SLOP_PX = 2;

const HANDLE_CURSOR: Record<HandleId, string> = {
  nw: 'nwse-resize',
  se: 'nwse-resize',
  ne: 'nesw-resize',
  sw: 'nesw-resize',
  n: 'ns-resize',
  s: 'ns-resize',
  e: 'ew-resize',
  w: 'ew-resize',
};

type Interaction =
  | { kind: 'draw'; start: IPoint; imageId: string }
  | {
      kind: 'move';
      id: string;
      start: IPoint;
      origin: IRect;
      /** 光标下所有框的 id，从上层到下层，用于原地点击时逐个轮换 */
      under: string[];
      moved: boolean;
    }
  | { kind: 'resize'; id: string; handle: HandleId; start: IPoint; origin: IRect; moved: boolean }
  | {
      kind: 'pan';
      start: IPoint;
      origin: { ox: number; oy: number };
      moved: boolean;
      /** 右键按下时的屏幕坐标：没拖动就抬起 → 在框上弹菜单 */
      contextAt: IPoint | null;
    };

type HitResult =
  | { kind: 'handle'; id: string; handle: HandleId }
  | { kind: 'move'; id: string; under: string[] }
  | null;

/** 一次按下/抬起在 AI 候选框上的暂存：抬起时若没移动就单独剔除这个候选框 */
type CandidateTap = { index: number; sx: number; sy: number; moved: boolean };

/** 命中哪个 AI 候选框（后画的在上面，所以从后往前找）。没命中返回 -1 */
function hitCandidateIndex(
  p: IPoint,
  cands: readonly { rect: IRect }[],
): number {
  for (let i = cands.length - 1; i >= 0; i -= 1) {
    const r = cands[i].rect;
    if (p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= r.y + r.height) return i;
  }
  return -1;
}

/**
 * 「点击标注」按钮里显示什么。
 *
 * 单独拆成一个小组件，是为了**订阅粒度**：模型加载期间 `segStore.progress`
 * 会随下载分片更新上百次，若让 CanvasStage 订阅它，整棵重 JSX
 * （细网格、两层渐变、暗角、信息条、图例、缩放条）会跟着重渲染上百次，
 * 而真正需要变的只有这十来个字。
 *
 * 为什么必须有这段 UI —— 这是「点了完全不动、点了也没有加载动效」的根因：
 * 模型是两个 onnx 共约 42MB，加上 ORT 建会话（WebGPU 还要编译着色器），
 * 首次点开要等好几秒到几十秒。而这段时间里**按钮原本一个像素都不变**，
 * 用户看到的就是「点了没反应」，于是反复点，然后判定功能坏了。
 * 更糟的是 `open()` 在 loading 期间是静默 return 的，连点多少次都不会有反馈。
 */
function SegButtonLabel({ active }: { active: boolean }) {
  const status = useSegStore((s) => s.status);
  const progress = useSegStore((s) => s.progress);

  if (status === 'loading') {
    // 进度到 94 说明两个 onnx 都下完了，此时卡住的是 ORT 建会话（本地文件也要几秒）
    const downloading = progress < SEG_PROGRESS_DOWNLOAD_DONE;
    return (
      <span className="flex items-center gap-1.5 whitespace-nowrap">
        <Loader2 className="size-[14px] flex-shrink-0 animate-spin" />
        {downloading
          ? progress > 0
            ? `正在下载模型 ${progress}%`
            : '正在下载模型…'
          : '正在初始化推理引擎…'}
      </span>
    );
  }

  // 失败必须留在按钮上，不能只靠一闪而过的 toast ——
  // 否则按钮此后和「没点过」长得一模一样，用户只会得出「坏了」的结论。
  // 文案里的「点此」是实指：status 既非 ready 也非 loading 时，再点会重新走一遍 open()。
  if (status === 'error') {
    return (
      <span className="flex items-center gap-1.5 whitespace-nowrap text-amber-300">
        <TriangleAlert className="size-[14px] flex-shrink-0" />
        加载失败 · 点此重试
      </span>
    );
  }

  return (
    <span className="flex items-center gap-1.5 whitespace-nowrap">
      <MousePointerClick className="size-[14px] flex-shrink-0" />
      {active ? '点击标注中' : '点击标注'}
    </span>
  );
}

export function CanvasStage() {
  // Phase 11：只订阅「当前这张图」，不再订阅整个 images 数组。
  // 数组引用在**任何一张图**的标注变化时都会变（批量 AI 落框时尤其明显），
  // 而这个组件的 JSX 很重（细网格、两层渐变、暗角、信息条、空状态、图例、缩放条），
  // 别的图被改不该惊动它。选择器里读 s.activeIndex 也能拿到同一个快照，更安全。
  const image = useStore((s) => s.images[s.activeIndex] ?? null);
  const imageId = image?.id ?? null;

  const labels = useStore((s) => s.labels);
  const selectedId = useStore((s) => s.selectedId);
  const hoverId = useStore((s) => s.hoverId);
  const hiddenLabelIds = useStore((s) => s.hiddenLabelIds);

  const addBox = useStore((s) => s.addBox);
  const updateBox = useStore((s) => s.updateBox);
  const select = useStore((s) => s.select);
  const setHover = useStore((s) => s.setHover);
  const setImageSize = useStore((s) => s.setImageSize);
  /** 图片文件在、但浏览器解码失败（损坏 / 不支持的格式）。区别于「数据丢失」 */
  const [decodeFailed, setDecodeFailed] = useState(false);

  // AI 候选框只属于某一张图：切图后自动不再显示
  const aiCandidates = useAiStore((s) =>
    s.candidatesImageId === imageId ? s.candidates : null,
  );
  const aiBusy = useAiStore((s) => s.busy);
  const aiAcceptAll = useAiStore((s) => s.acceptAll);
  const aiDiscard = useAiStore((s) => s.discardCandidates);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<Renderer | null>(null);

  const imageRef = useRef(image);
  const selectedRef = useRef(selectedId);
  /** 选中框对象的缓存，由下方 layout effect 每次渲染同步。hitTest 靠它避开每次 mousemove 的 O(n) 查找 */
  const selectedAnnRef = useRef<Annotation | null>(null);
  const hoverRef = useRef(hoverId);
  /** 隐藏类别（画布侧不订阅 store 每帧渲染，只随变化同步到 ref） */
  const hiddenRef = useRef<ReadonlySet<string>>(EMPTY_HIDDEN);
  const bitmapRef = useRef<ImageBitmap | null>(null);
  const viewRef = useRef(viewStore.get());
  const sizeRef = useRef({ w: 0, h: 0 });
  const fitModeRef = useRef(true);
  const interactionRef = useRef<Interaction | null>(null);
  const draftRef = useRef<{ id: string; rect: IRect } | null>(null);
  const pendingRef = useRef<IRect | null>(null);
  const aiCandidatesRef = useRef<AiCandidate[] | null>(null);
  const cursorRef = useRef<IPoint | null>(null);
  const spaceRef = useRef(false);

  /**
   * 点击标注（MobileSAM）模式。
   * 订阅值同步进 ref：指针回调是 long-lived 的 useCallback，
   * 直接读订阅值会闭包住旧值，导致「刚打开模式点一下没反应 / 关掉后还在取点」。
   */
  const segActiveRef = useRef(false);
  /**
   * 一次按下/抬起的暂存。用来区分「单击取点」和「按住拖动平移」——
   * 不能一按下就取点，否则想移动画布时会误生成一堆框。
   */
  const segTapRef = useRef<{
    sx: number;
    sy: number;
    ix: number;
    iy: number;
    button: number;
    moved: boolean;
  } | null>(null);

  const segActive = useSegStore((s) => s.active);
  const segBusy = useSegStore((s) => s.busy);
  const segEncoded = useSegStore((s) => s.encoded);
  const segPoints = useSegStore((s) => s.points);
  const segResetPoints = useSegStore((s) => s.resetPoints);
  /**
   * 点击标注的点位与 AI 候选框的「单击剔除」暂存。
   *
   * 点位必须走 ref 交给渲染循环：它是画在 canvas 上的（静态层缓存），
   * 而平移时视图变化刻意不触发 React 渲染 —— 用绝对定位的 DOM 覆盖层画点的话，
   * 一平移标记就会和图片脱开。
   */
  const segPointsRef = useRef<readonly { x: number; y: number; positive: boolean }[]>([]);
  const candTapRef = useRef<CandidateTap | null>(null);
  const aiRemoveCandidate = useAiStore((s) => s.removeCandidate);
  useEffect(() => {
    segActiveRef.current = segActive;
    // 退出模式时清掉半途的暂存，避免残留状态影响下一次进入
    if (!segActive) segTapRef.current = null;
  }, [segActive]);
  /** 框菜单是否开着 —— 全局快捷键要据此让位 */
  const menuOpenRef = useRef(false);
  /** 触屏多点：正在按下的触点集合（捏合缩放用） */
  const pointersRef = useRef(new Map<number, IPoint>());
  /** 捏合起始快照：两指距离、起始视图、两指中点 */
  const pinchRef = useRef<{
    dist: number;
    scale: number;
    ox: number;
    oy: number;
    mid: IPoint;
  } | null>(null);

  // 框旁边的标签菜单：画完框就地问「这是什么」，双击 / 右键已有框则改标签。
  // 记下 imageId，切换图片时靠它自动失效，不必在 effect 里清理。
  const [menu, setMenu] = useState<{
    imageId: string;
    annId: string;
    mode: PopoverMode;
    anchor: { left: number; top: number; right: number; bottom: number };
  } | null>(null);

  const labelMap = useMemo(() => {
    const map = new Map<string, (typeof labels)[number]>();
    for (const l of labels) map.set(l.id, l);
    return map;
  }, [labels]);
  const labelMapRef = useRef(labelMap);

  // ---------- 渲染调度：每帧最多一次重绘 ----------
  const draw = useCallback(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    const img = imageRef.current;
    renderer.render({
      bitmap: bitmapRef.current,
      imgW: img?.width ?? 0,
      imgH: img?.height ?? 0,
      annotations: img?.annotations ?? EMPTY_ANNOTATIONS,
      labels: labelMapRef.current,
      selectedId: selectedRef.current,
      hoverId: hoverRef.current,
      draftId: draftRef.current?.id ?? null,
      draftRect: draftRef.current?.rect ?? null,
      pendingRect: pendingRef.current,
      aiCandidates: aiCandidatesRef.current,
      segPoints: segPointsRef.current,
      hiddenLabelIds: hiddenRef.current,
      cursor: cursorRef.current,
      showCrosshair: interactionRef.current === null,
      showHandles: true,
      view: viewRef.current,
      width: sizeRef.current.w,
      height: sizeRef.current.h,
    });
  }, []);

  const frameRef = useRef<() => void>(() => {});
  const rafRef = useRef(0);

  // 渲染循环直接读 ref，避免每帧走 React。
  // 用 layout effect 保证在任何 requestAnimationFrame 回调之前完成同步。
  useLayoutEffect(() => {
    frameRef.current = draw;
    imageRef.current = image;
    selectedRef.current = selectedId;
    hoverRef.current = hoverId;
    labelMapRef.current = labelMap;
    aiCandidatesRef.current = aiCandidates;
    segPointsRef.current = segPoints;
    // 选中框对象的缓存：hitTest 每次 mousemove 都要靠它判定缩放手柄命中，
    // 缓存下来就不必每次再 annotations.find()（那又是一遍 O(n) 全图扫描）。
    selectedAnnRef.current =
      selectedId && image ? (image.annotations.find((a) => a.id === selectedId) ?? null) : null;
  });

  // Phase 11：隐藏类别要建 Set 供命中测试做 O(1) 查询，但**只在它真的变了**时才需要重建。
  // 原来这一行混在上面那个无依赖的 layout effect 里 —— 只要存在隐藏类别，
  // 组件每次渲染（落框、hover、缩放、任何 store 变化）都要白白新建一个 Set。
  useLayoutEffect(() => {
    hiddenRef.current = hiddenLabelIds.length === 0 ? EMPTY_HIDDEN : new Set(hiddenLabelIds);
  }, [hiddenLabelIds]);

  const markDirty = useCallback(() => {
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = 0;
      frameRef.current();
    });
  }, []);

  /**
   * webfont 是**异步**加载的（MiSans VF 切成 56 个 unicode-range 分片，按需下载），
   * 而 canvas 与 DOM 不同：DOM 在字体到位后自己会重排，**画过的像素不会自己变**。
   * 不做这一步，首屏在字体还没到位时画出的芯片会一直停在系统字体的字形上，
   * 且 measureText 缓存里存的也是系统字体的宽度 —— 芯片底片会整场会话都偏窄或偏宽。
   */
  useEffect(() => {
    const fonts = document.fonts;
    if (!fonts) return;
    const onFontsLoaded = () => {
      clearTextWidthCache();
      markDirty();
    };
    fonts.addEventListener('loadingdone', onFontsLoaded);
    return () => fonts.removeEventListener('loadingdone', onFontsLoaded);
  }, [markDirty]);

  useEffect(
    () => () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        // 必须清掉 id，否则 StrictMode 的双执行会把首个 rAF 取消后留下一个
        // 非零的脏引用，导致后续所有 markDirty 都误判「已在排队」而直接返回，
        // 画布从此再也不重绘（表现为导入图片后一片黑）。
        rafRef.current = 0;
      }
    },
    [],
  );

  // AI 候选框增减时刷一次画布
  useEffect(() => {
    markDirty();
  }, [aiCandidates, markDirty]);

  // 点击标注的点位变了也要刷一次（点在静态层里，不刷的话新点要等下一次重绘才出现）
  useEffect(() => {
    markDirty();
  }, [segPoints, markDirty]);

  // 只有缩放比例变化才通知 React，纯平移不产生任何重渲染
  const publishView = useCallback(() => {
    if (viewStore.get().scale !== viewRef.current.scale) {
      viewStore.set({ ...viewRef.current });
    }
  }, []);

  // ---------- 视图控制 ----------
  const fitTo = useCallback(
    (imgW: number, imgH: number) => {
      const { w, h } = sizeRef.current;
      if (w <= 0 || h <= 0 || imgW <= 0 || imgH <= 0) return;
      const raw = Math.min((w - FIT_PADDING) / imgW, (h - FIT_PADDING) / imgH);
      const scale = clamp(raw, MIN_SCALE, MAX_SCALE);
      viewRef.current = { scale, ox: (w - imgW * scale) / 2, oy: (h - imgH * scale) / 2 };
      fitModeRef.current = true;
      setMenu(null);
      publishView();
      markDirty();
    },
    [markDirty, publishView],
  );

  const fit = useCallback(() => {
    const img = imageRef.current;
    fitTo(img?.width ?? 0, img?.height ?? 0);
  }, [fitTo]);

  const zoomAt = useCallback(
    (factor: number, cx: number, cy: number) => {
      const v = viewRef.current;
      const next = clamp(v.scale * factor, MIN_SCALE, MAX_SCALE);
      if (Math.abs(next - v.scale) < 1e-9) return;
      const k = next / v.scale;
      viewRef.current = { scale: next, ox: cx - (cx - v.ox) * k, oy: cy - (cy - v.oy) * k };
      fitModeRef.current = false;
      setMenu(null);
      publishView();
      markDirty();
    },
    [markDirty, publishView],
  );

  const zoomBy = useCallback(
    (factor: number) => {
      const { w, h } = sizeRef.current;
      zoomAt(factor, w / 2, h / 2);
    },
    [zoomAt],
  );

  // ---------- 初始化 ----------
  useEffect(() => {
    if (!canvasRef.current) return;
    rendererRef.current = new Renderer(canvasRef.current);
    return () => {
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const apply = (w: number, h: number) => {
      sizeRef.current = { w, h };
      rendererRef.current?.resize(w, h);
      if (fitModeRef.current) {
        const img = imageRef.current;
        fitTo(img?.width ?? 0, img?.height ?? 0);
      } else {
        markDirty();
      }
    };

    const rect = el.getBoundingClientRect();
    apply(rect.width, rect.height);

    const ro = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (!box) return;
      apply(box.width, box.height);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [fitTo, markDirty]);

  // 切换图片时重置临时状态
  useEffect(() => {
    fitModeRef.current = true;
    draftRef.current = null;
    pendingRef.current = null;
    interactionRef.current = null;
    setMenu(null);
  }, [imageId]);

  // 防御：当前选中的框所属类别被隐藏后，它在画布上已看不见 —— 取消选中，
  // 避免出现"看不见却能点手柄/按 Delete"的隐形操作
  useEffect(() => {
    if (!selectedId || !image) return;
    const ann = image.annotations.find((a) => a.id === selectedId);
    if (ann && ann.labelId && hiddenLabelIds.includes(ann.labelId)) {
      select(null);
    }
  }, [selectedId, image, hiddenLabelIds, select]);

  // 加载 / 复用解码后的位图
  useEffect(() => {
    const img = imageRef.current;
    if (!imageId || !img) {
      bitmapRef.current = null;
      markDirty();
      return;
    }

    let cancelled = false;
    protect(img.id);
    const cached = peekBitmap(img.id);
    bitmapRef.current = cached;

    const finish = (bmp: ImageBitmap | null) => {
      if (cancelled) return;
      bitmapRef.current = bmp;
      // 解码失败（文件损坏 / 浏览器不支持该格式）以前只 markDirty()，
      // 结果是一块纯黑画布 + 空白缩略图，用户完全不知道发生了什么。
      // 这里显式区分「解码失败」与「数据丢失」两种状态。
      setDecodeFailed(!bmp && !!img.file);
      if (bmp && (bmp.width !== img.width || bmp.height !== img.height)) {
        setImageSize(img.id, bmp.width, bmp.height);
      }
      if (fitModeRef.current) fitTo(bmp?.width ?? img.width, bmp?.height ?? img.height);
      markDirty();
    };

    if (cached) {
      finish(cached);
    } else if (img.file) {
      void loadBitmap(img).then(finish);
    } else {
      setDecodeFailed(false);
      markDirty();
    }

    return () => {
      cancelled = true;
    };
  }, [imageId, fitTo, markDirty, setImageSize]);

  // 对外暴露视图 API
  useEffect(() => {
    canvasApi.current = {
      fit,
      zoomBy,
      setSpace: (down: boolean) => {
        if (spaceRef.current === down) return;
        spaceRef.current = down;
        const canvas = canvasRef.current;
        if (canvas) canvas.style.cursor = down ? 'grab' : 'crosshair';
      },
      menuOpen: () => menuOpenRef.current,
    };
    return () => {
      canvasApi.current = null;
    };
  }, [fit, zoomBy]);

  // 滚轮 / 触控板缩放（必须用原生监听，React 的 wheel 是被动的）
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const intensity = e.ctrlKey ? 0.01 : 0.0022;
      zoomAt(Math.exp(-e.deltaY * intensity), cx, cy);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [zoomAt]);

  // ---------- 坐标 / 命中 ----------
  const pointerPos = useCallback((e: { clientX: number; clientY: number }): IPoint => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }, []);

  const toImage = useCallback((p: IPoint): IPoint => {
    const v = viewRef.current;
    return { x: (p.x - v.ox) / v.scale, y: (p.y - v.oy) / v.scale };
  }, []);

  const hitTest = useCallback((p: IPoint): HitResult => {
    const img = imageRef.current;
    if (!img) return null;
    // 缩得很小时 8px 会换算成巨大的图像坐标容差（scale=0.02 时高达 400px），
    // 命中范围会失控、误选一堆框；加个上限兜住
    const tol = Math.min(HANDLE_TOLERANCE / viewRef.current.scale, 40);
    const hidden = hiddenRef.current;
    const anns = img.annotations;

    // ---- 缩放手柄 ----
    // Phase 11：这段每次 pointermove 都要跑，所以刻意做到「零多余分配」：
    //   · 选中框走缓存 —— 原来是每帧 annotations.find()，等于又白扫一遍全图
    //   · 八个手柄直接按坐标比 —— handlePoints() 每次要新建 8 个对象
    //   · 不再建 isHidden 闭包 —— 原来每次调用都新建一个函数对象
    const selId = selectedRef.current;
    if (selId) {
      let sel = selectedAnnRef.current;
      // 缓存与当前选中 id 对不上时（极端时序）回退到实时查找，正确性优先
      if (!sel || sel.id !== selId) sel = anns.find((a) => a.id === selId) ?? null;
      // 选中的框属于被隐藏的类别时，不响应手柄，也不把它当命中目标
      if (sel && !(sel.labelId && hidden.has(sel.labelId))) {
        const r = sel.rect;
        const cx = r.x + r.width / 2;
        const cy = r.y + r.height / 2;
        const rx = r.x + r.width;
        const by = r.y + r.height;
        const near = (px: number, py: number) =>
          Math.abs(px - p.x) <= tol && Math.abs(py - p.y) <= tol;
        // 顺序与 geometry.HANDLES 一致：nw n ne e se s sw w
        if (near(r.x, r.y)) return { kind: 'handle', id: sel.id, handle: 'nw' };
        if (near(cx, r.y)) return { kind: 'handle', id: sel.id, handle: 'n' };
        if (near(rx, r.y)) return { kind: 'handle', id: sel.id, handle: 'ne' };
        if (near(rx, cy)) return { kind: 'handle', id: sel.id, handle: 'e' };
        if (near(rx, by)) return { kind: 'handle', id: sel.id, handle: 'se' };
        if (near(cx, by)) return { kind: 'handle', id: sel.id, handle: 's' };
        if (near(r.x, by)) return { kind: 'handle', id: sel.id, handle: 'sw' };
        if (near(r.x, cy)) return { kind: 'handle', id: sel.id, handle: 'w' };
      }
    }

    // ---- 框体 ----
    // 先按严格命中找；一个都没有时再放宽到边框附近，方便抓细长的框。
    // `under` 延后到「真的命中」才分配 —— 鼠标在空白处移动是常态，
    // 原来每次都会先建一个空数组（外加 `[0, tol]` 那个临时数组）。
    for (let pass = 0; pass < 2; pass += 1) {
      const margin = pass === 0 ? 0 : tol;
      let under: string[] | null = null;
      for (let i = anns.length - 1; i >= 0; i -= 1) {
        const ann = anns[i];
        if (ann.labelId && hidden.has(ann.labelId)) continue;
        const r = ann.rect;
        if (
          p.x >= r.x - margin &&
          p.x <= r.x + r.width + margin &&
          p.y >= r.y - margin &&
          p.y <= r.y + r.height + margin
        ) {
          if (!under) under = [];
          under.push(ann.id);
        }
      }
      if (under) return { kind: 'move', id: under[0], under };
    }
    return null;
  }, []);

  /** 把标签菜单钉在画布内的某个屏幕坐标上（右键 / 双击已有框） */
  const openMenuAtPoint = useCallback((annId: string, p: IPoint) => {
    const img = imageRef.current;
    if (!img) return;
    setMenu({
      imageId: img.id,
      annId,
      mode: 'edit',
      anchor: { left: p.x, top: p.y, right: p.x, bottom: p.y },
    });
  }, []);

  /**
   * 点击标注生成的新框若还没有类别，segStore 会把它的 id 放进 pendingLabelFor，
   * 这里负责当场弹「这是什么」——与画框流程保持一致，
   * 不再静默留一个「未分类」框（用户反馈的「点一下没有选类别」就是它）。
   *
   * 走 store 标志而不是给 canvasApi 加方法：openMenuAtPoint 定义在本组件内，
   * 而 canvasApi 的挂载 effect 位置更靠前，直接引用会踩 used-before-declaration。
   *
   * 锚点取框中心并换算到**画布坐标**（与 pointerPos 同一套），菜单才会贴着框出现。
   */
  const segPendingLabel = useSegStore((s) => s.pendingLabelFor);
  useEffect(() => {
    if (!segPendingLabel) return;
    useSegStore.getState().clearPendingLabel();
    const img = imageRef.current;
    if (!img) return;
    const ann = img.annotations.find((a) => a.id === segPendingLabel);
    if (!ann) return;
    const v = viewRef.current;
    select(segPendingLabel);
    openMenuAtPoint(segPendingLabel, {
      x: ann.rect.x * v.scale + v.ox + (ann.rect.width * v.scale) / 2,
      y: ann.rect.y * v.scale + v.oy + (ann.rect.height * v.scale) / 2,
    });
  }, [segPendingLabel, openMenuAtPoint, select]);

  // ---------- 指针交互 ----------
  const onPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      const img = imageRef.current;
      if (!img) return;
      if (e.button === 1) e.preventDefault();

      try {
        canvasRef.current?.setPointerCapture(e.pointerId);
      } catch {
        /* 忽略 */
      }

      const p = pointerPos(e);
      const ip = toImage(p);
      const v = viewRef.current;

      // 触屏：维护活动触点；第二根手指落下即进入捏合缩放
      if (e.pointerType === 'touch') {
        pointersRef.current.set(e.pointerId, p);
        if (pointersRef.current.size === 2) {
          // 放弃进行中的单指操作（画框/拖动/平移），改走捏合，避免两指动作打架
          interactionRef.current = null;
          draftRef.current = null;
          pendingRef.current = null;
          setMenu(null);
          const ids = [...pointersRef.current.keys()];
          const a = pointersRef.current.get(ids[0])!;
          const b = pointersRef.current.get(ids[1])!;
          pinchRef.current = {
            dist: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)),
            scale: viewRef.current.scale,
            ox: viewRef.current.ox,
            oy: viewRef.current.oy,
            mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
          };
          markDirty();
          return;
        }
      }

      setMenu(null);

      // ---- AI 候选框：点一下就能单独去掉它 ----
      // 以前只有「全部采纳 / 忽略」两条路，为一个错框得整批放弃、采纳完再手工删。
      // 只记下按下点，抬起且没移动才真剔除 —— 从候选框上按下再拖出去看别处，
      // 不该被当成「我要删掉它」。
      const cands = aiCandidatesRef.current;
      if (cands && cands.length > 0 && !segActiveRef.current && e.button === 0 && !spaceRef.current && !e.altKey) {
        const idx = hitCandidateIndex(ip, cands);
        if (idx >= 0) {
          candTapRef.current = { index: idx, sx: p.x, sy: p.y, moved: false };
          return;
        }
      }

      // ---- 点击标注模式：左键 =「要这个物体」，右键 =「排除这块」----
      // 放在平移判断之前，但把「空格 / 中键 / Alt」让出去 ——
      // 取点过程中还得能移动画布去看图片别的区域，否则大图上没法用。
      // 这里只记下起点，真正的取点发生在 pointerUp：
      // 一按下就取点的话，用户想拖着平移画布时会被误生成一堆框。
      //
      // **但必须先让出「落在已有框上」的点击**：早先这里是无条件拦截所有左键，
      // 导致框一旦生成就再也选不中、拖不动、缩不了（用户反馈的「无法调整大小」）。
      // 现在命中原有框时交还给下面的 resize / move 分支。
      // 想在被框住的物体上追加新框 → 按住 **Shift** 点（Shift 跳过命中检测）。
      if (segActiveRef.current && !spaceRef.current && !e.altKey && e.button !== 1) {
        const segHit = e.shiftKey ? null : hitTest(ip);
        if (!segHit) {
          segTapRef.current = {
            sx: p.x,
            sy: p.y,
            ix: ip.x,
            iy: ip.y,
            button: e.button,
            moved: false,
          };
          return;
        }
        // 右键落在框上 → 弹标签菜单（与常规模式一致），不再被当成负点。
        // 负点仍可在**没有框的区域**右键使用。
        if (e.button === 2) {
          select(segHit.id);
          openMenuAtPoint(segHit.id, p);
          return;
        }
        // 左键落在框上 → 落到下面的 resize / move 流程
      }

      if (spaceRef.current || e.button === 1 || e.button === 2 || e.altKey) {
        interactionRef.current = {
          kind: 'pan',
          start: p,
          origin: { ox: v.ox, oy: v.oy },
          moved: false,
          contextAt: e.button === 2 && !spaceRef.current && !e.altKey ? p : null,
        };
        markDirty();
        return;
      }
      if (e.button !== 0) return;

      const hit = hitTest(ip);
      if (hit?.kind === 'handle') {
        const ann = img.annotations.find((a) => a.id === hit.id);
        if (!ann) return;
        select(hit.id);
        interactionRef.current = {
          kind: 'resize',
          id: hit.id,
          handle: hit.handle,
          start: ip,
          origin: ann.rect,
          moved: false,
        };
        draftRef.current = { id: hit.id, rect: ann.rect };
        markDirty();
        return;
      }
      if (hit?.kind === 'move') {
        const ann = img.annotations.find((a) => a.id === hit.id);
        if (!ann) return;
        select(hit.id);
        interactionRef.current = {
          kind: 'move',
          id: hit.id,
          start: ip,
          origin: ann.rect,
          under: hit.under,
          moved: false,
        };
        draftRef.current = { id: hit.id, rect: ann.rect };
        markDirty();
        return;
      }

      select(null);

      const grace = DRAW_EDGE_GRACE / v.scale;
      const inside =
        img.width > 0 &&
        img.height > 0 &&
        ip.x >= -grace &&
        ip.y >= -grace &&
        ip.x <= img.width + grace &&
        ip.y <= img.height + grace;

      // 在图外的空白处拖动 = 平移画布，而不是硬贴着一个边界框出来
      if (!inside) {
        interactionRef.current = {
          kind: 'pan',
          start: p,
          origin: { ox: v.ox, oy: v.oy },
          moved: false,
          contextAt: null,
        };
        markDirty();
        return;
      }

      const start = { x: clamp(ip.x, 0, img.width), y: clamp(ip.y, 0, img.height) };
      interactionRef.current = { kind: 'draw', start, imageId: img.id };
      pendingRef.current = { x: start.x, y: start.y, width: 0, height: 0 };
      markDirty();
    },
    [hitTest, markDirty, pointerPos, select, toImage],
  );

  const onPointerMove = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      const img = imageRef.current;
      if (!img) return;
      const p = pointerPos(e);
      cursorRef.current = p;

      // 触屏捏合：双指同时缩放 + 拖动画面，一切以两指中点为锚
      if (e.pointerType === 'touch') {
        if (pointersRef.current.has(e.pointerId)) {
          pointersRef.current.set(e.pointerId, p);
        }
        const pinch = pinchRef.current;
        if (pinch && pointersRef.current.size >= 2) {
          const ids = [...pointersRef.current.keys()];
          const a = pointersRef.current.get(ids[0])!;
          const b = pointersRef.current.get(ids[1])!;
          const dist = Math.max(1, Math.hypot(a.x - b.x, a.y - b.y));
          const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
          const nextScale = clamp(pinch.scale * (dist / pinch.dist), MIN_SCALE, MAX_SCALE);
          const k = nextScale / pinch.scale;
          viewRef.current = {
            scale: nextScale,
            ox: mid.x - (pinch.mid.x - pinch.ox) * k,
            oy: mid.y - (pinch.mid.y - pinch.oy) * k,
          };
          fitModeRef.current = false;
          publishView();
          markDirty();
          return;
        }
      }

      const it = interactionRef.current;
      const v = viewRef.current;

      if (!it) {
        // 点击标注：只判「按下后有没有真的移动」，不参与 hover 高亮 ——
        // 这一模式下用户关注的是「往哪点」，把框高亮起来只会干扰判断。
        const tap = segTapRef.current;
        if (tap) {
          if (!tap.moved && Math.hypot(p.x - tap.sx, p.y - tap.sy) > CLICK_SLOP_PX) {
            tap.moved = true;
          }
          const canvas = canvasRef.current;
          const cursor = spaceRef.current ? 'grab' : 'crosshair';
          if (canvas && canvas.style.cursor !== cursor) canvas.style.cursor = cursor;
          return;
        }

        const candTap = candTapRef.current;
        if (candTap) {
          if (!candTap.moved && Math.hypot(p.x - candTap.sx, p.y - candTap.sy) > CLICK_SLOP_PX) {
            candTap.moved = true;
          }
          const canvas = canvasRef.current;
          if (canvas && canvas.style.cursor !== 'pointer') canvas.style.cursor = 'pointer';
          return;
        }

        const canvas = canvasRef.current;
        if (canvas) {
          const hit = hitTest(toImage(p));
          const cands = aiCandidatesRef.current;
          let cursor = 'crosshair';
          if (spaceRef.current) cursor = 'grab';
          // 候选框压在已有框上时优先给「可点掉」的手型：此刻用户能操作的确实是那个候选框
          else if (cands && cands.length > 0 && hitCandidateIndex(toImage(p), cands) >= 0)
            cursor = 'pointer';
          else if (hit?.kind === 'handle') cursor = HANDLE_CURSOR[hit.handle];
          else if (hit?.kind === 'move') cursor = 'move';
          if (canvas.style.cursor !== cursor) canvas.style.cursor = cursor;
          const id = hit ? hit.id : null;
          if (id !== hoverRef.current) {
            hoverRef.current = id;
            setHover(id);
          }
        }
        markDirty();
        return;
      }

      const ip = toImage(p);
      const snapTol = SNAP_TOLERANCE / v.scale;

      // 判断这次是不是真的在拖动（用于区分「拖框」与「原地点击」）
      if (it.kind === 'move' || it.kind === 'resize') {
        const movedPx = Math.hypot(ip.x - it.start.x, ip.y - it.start.y) * v.scale;
        if (movedPx > CLICK_SLOP_PX) it.moved = true;
      }

      if (it.kind === 'pan') {
        if (!it.moved && Math.hypot(p.x - it.start.x, p.y - it.start.y) > CLICK_SLOP_PX) {
          it.moved = true;
        }
        viewRef.current = {
          scale: v.scale,
          ox: it.origin.ox + (p.x - it.start.x),
          oy: it.origin.oy + (p.y - it.start.y),
        };
        fitModeRef.current = false;
        publishView();
        markDirty();
        return;
      }

      if (it.kind === 'move') {
        // 只吸附位置，不改动尺寸，避免标注框被"拉变形"
        const w = it.origin.width;
        const h = it.origin.height;
        const x = snapEdges(it.origin.x + (ip.x - it.start.x), Math.max(0, img.width - w), snapTol);
        const y = snapEdges(it.origin.y + (ip.y - it.start.y), Math.max(0, img.height - h), snapTol);
        const rect = clampRectToImage({ x, y, width: w, height: h }, img.width, img.height);
        draftRef.current = { id: it.id, rect };
        markDirty();
        return;
      }

      if (it.kind === 'resize') {
        const dx = ip.x - it.start.x;
        const dy = ip.y - it.start.y;
        const minSize = Math.max(2, 4 / v.scale);
        let rect = applyResize(it.origin, it.handle, dx, dy, img.width, img.height, minSize);
        rect = snapRectToEdges(rect, img.width, img.height, snapTol);
        draftRef.current = { id: it.id, rect };
        markDirty();
        return;
      }

      // draw
      const x0 = clamp(Math.min(it.start.x, ip.x), 0, img.width);
      const y0 = clamp(Math.min(it.start.y, ip.y), 0, img.height);
      const x1 = clamp(Math.max(it.start.x, ip.x), 0, img.width);
      const y1 = clamp(Math.max(it.start.y, ip.y), 0, img.height);
      let rect: IRect = { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
      if (e.shiftKey) {
        const side = Math.max(rect.width, rect.height);
        rect = clampRectToImage({ ...rect, width: side, height: side }, img.width, img.height);
      }
      rect = snapRectToEdges(rect, img.width, img.height, snapTol);
      pendingRef.current = rect;
      markDirty();
    },
    [hitTest, markDirty, pointerPos, publishView, setHover, toImage],
  );

  const onPointerUp = useCallback(
    (e?: ReactPointerEvent<HTMLCanvasElement>) => {
      if (e) {
        // 触屏：移除触点；少于两指就退出捏合（剩一根手指不触发任何画布操作）
        if (e.pointerType === 'touch') {
          pointersRef.current.delete(e.pointerId);
          if (pointersRef.current.size < 2) pinchRef.current = null;
        }
        try {
          canvasRef.current?.releasePointerCapture(e.pointerId);
        } catch {
          /* 忽略 */
        }
      }
      // ---- 点击标注：抬起时才判定是「单击取点」还是「拖动平移」----
      const tap = segTapRef.current;
      if (tap) {
        segTapRef.current = null;
        if (!tap.moved) {
          // button 0 = 正点（要这个物体），其余 = 负点（排除这块）
          void useSegStore.getState().click(tap.ix, tap.iy, tap.button === 0);
        }
        return;
      }

      // ---- AI 候选框：原地点击 = 剔掉这一个 ----
      const candTap = candTapRef.current;
      if (candTap) {
        candTapRef.current = null;
        if (!candTap.moved) aiRemoveCandidate(candTap.index);
        return;
      }

      const it = interactionRef.current;
      interactionRef.current = null;
      if (!it) return;

      // 原地点击（没有拖动）叠在一起的区域 → 依次选中下面的框
      if (it.kind === 'move' && !it.moved && it.under.length > 1) {
        const at = it.under.indexOf(it.id);
        if (at >= 0) select(it.under[(at + 1) % it.under.length]);
      }

      // 原地右键 → 在光标下的框上弹标签菜单；右键拖动仍然是平移画布
      if (it.kind === 'pan') {
        if (it.contextAt && !it.moved) {
          const hit = hitTest(toImage(it.contextAt));
          if (hit) {
            select(hit.id);
            openMenuAtPoint(hit.id, it.contextAt);
          }
        }
        markDirty();
        return;
      }

      if (it.kind === 'draw') {
        const rect = pendingRef.current;
        pendingRef.current = null;
        const v = viewRef.current;
        if (rect && rect.width * v.scale >= MIN_DRAW_PX && rect.height * v.scale >= MIN_DRAW_PX) {
          const id = addBox(rect);
          const img = imageRef.current;
          // 每画一个框都弹一次「这是什么」：即使已有默认类别，
          // 也让人当场确认或换类别，不产生"标签怎么自己变了"的错觉
          if (id && img) {
            setMenu({
              imageId: img.id,
              annId: id,
              mode: 'assign',
              anchor: {
                left: v.ox + rect.x * v.scale,
                top: v.oy + rect.y * v.scale,
                right: v.ox + (rect.x + rect.width) * v.scale,
                bottom: v.oy + (rect.y + rect.height) * v.scale,
              },
            });
          }
        }
        markDirty();
        return;
      }

      if (it.kind === 'move' || it.kind === 'resize') {
        const draft = draftRef.current;
        draftRef.current = null;
        if (draft) updateBox(draft.id, draft.rect);
        markDirty();
      }
    },
    [addBox, aiRemoveCandidate, hitTest, markDirty, openMenuAtPoint, select, toImage, updateBox],
  );

  const onPointerLeave = useCallback(() => {
    cursorRef.current = null;
    if (interactionRef.current) return;
    if (hoverRef.current !== null) {
      hoverRef.current = null;
      setHover(null);
    }
    markDirty();
  }, [markDirty, setHover]);

  // 双击框 = 就地改标签（以前会悄悄循环到下一个类别，很容易让人以为标签自己乱了）
  const onDoubleClick = useCallback(
    (e: ReactMouseEvent<HTMLCanvasElement>) => {
      const p = pointerPos(e);
      const hit = hitTest(toImage(p));
      if (!hit) return;
      select(hit.id);
      openMenuAtPoint(hit.id, p);
    },
    [hitTest, openMenuAtPoint, pointerPos, select, toImage],
  );

  /**
   * 只订阅「整数百分比」这一个数字，而不是整个 view 对象。
   *
   * 整个组件里 view 只有一个消费者（右下角的缩放百分比），但订阅整个对象意味着
   * 每次滚轮缩放（触控板 60~100Hz）都会重渲染 CanvasStage 的整棵 JSX：
   * 细网格、两层 radial-gradient、暗角、左上角信息条、空状态引导、左下角图例、右下角缩放条。
   * 快照返回数字时 useSyncExternalStore 用 Object.is 比较，
   * 于是只有百分比跨越整数位时才通知 —— 一次缩放手势从 60~100 次重渲染降到 1~4 次。
   * 平移（ox/oy 变化）完全不触发 React 渲染，视图更新走 rAF 重绘。
   */
  const zoomPercent = useSyncExternalStore(
    viewStore.subscribe,
    () => Math.round(viewStore.get().scale * 100),
  );
  const annotationCount = image?.annotations.length ?? 0;

  /** 顶部中央是否已被 AI 状态条占用（占用了就把点击提示下移，避免叠在一起） */
  const aiBarVisible = aiBusy === 'current' || (aiCandidates?.length ?? 0) > 0;
  const toggleSeg = useCallback(() => {
    const s = useSegStore.getState();
    if (s.active) s.close();
    else void s.open();
  }, []);
  // 切换图片、或这个框已经被删掉（比如按了 Delete）时，菜单自动消失
  const activeMenu = useMemo(() => {
    if (!menu || menu.imageId !== imageId) return null;
    if (!image?.annotations.some((a) => a.id === menu.annId)) return null;
    return menu;
  }, [menu, imageId, image]);
  const menuAnn = activeMenu
    ? image?.annotations.find((a) => a.id === activeMenu.annId)
    : undefined;

  useLayoutEffect(() => {
    menuOpenRef.current = activeMenu !== null;
  }, [activeMenu]);

  // 图片文件丢失（例如浏览器清了缓存）时给出明确提示，而不是只留一块黑
  const imageMissing = !!image && !image.file;
  // 文件在、但浏览器解不开（损坏 / 不支持的格式）—— 与「数据丢失」是两回事，文案也不同
  const imageBroken = !!image && !imageMissing && decodeFailed;

  return (
    <div
      ref={containerRef}
      className="relative min-w-0 flex-1 overflow-hidden rounded-xl border border-hairline [box-shadow:var(--shadow-panel)]"
      style={{ background: 'var(--canvas-bg)' }}
    >
      {/* 细网格 */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: 'radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px)',
          backgroundSize: '24px 24px',
        }}
      />
      {/* 顶部微光，增加纵深 */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'radial-gradient(130% 72% at 50% -18%, rgba(37,99,235,0.13), transparent 62%)',
        }}
      />

      <canvas
        ref={canvasRef}
        className="absolute inset-0 block touch-none select-none"
        style={{ cursor: 'crosshair' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={onPointerLeave}
        onDoubleClick={onDoubleClick}
        onContextMenu={(e) => e.preventDefault()}
        onDragStart={(e) => e.preventDefault()}
      />

      {/* 框旁边的标签菜单：画完框问一句，双击 / 右键随时改 */}
      {activeMenu ? (
        <LabelPopover
          key={activeMenu.annId}
          anchor={activeMenu.anchor}
          containerRef={containerRef}
          annotationId={activeMenu.annId}
          currentLabelId={menuAnn?.labelId ?? ''}
          mode={activeMenu.mode}
          onDone={() => setMenu(null)}
        />
      ) : null}

      {/* 极淡暗角：只增加纵深，不能影响判断图片边缘内容 */}
      <div
        className="pointer-events-none absolute inset-0 z-[5]"
        style={{
          background:
            'radial-gradient(125% 95% at 50% 50%, transparent 72%, rgba(0,0,0,0.16) 100%)',
        }}
      />

      {/* 顶部中央：AI 识别状态 / 候选框确认条 */}
      {aiBusy === 'current' ? (
        <div className="glass anim-up absolute left-1/2 top-3.5 z-20 flex -translate-x-1/2 items-center gap-2 rounded-full py-[7px] pl-3.5 pr-4">
          <Loader2 className="size-[14px] animate-spin text-sky-300" />
          <span className="text-caption font-semibold text-white/85">AI 正在看这张图…</span>
        </div>
      ) : aiCandidates && aiCandidates.length > 0 ? (
        <div className="glass anim-up absolute left-1/2 top-3.5 z-20 flex -translate-x-1/2 items-center gap-2.5 rounded-full py-[5px] pl-3.5 pr-[5px]">
          <Sparkles className="size-[15px] flex-shrink-0 text-sky-300" />
          <span className="whitespace-nowrap text-caption font-bold text-white/92">
            AI 找到 <span className="num">{aiCandidates.length}</span> 个目标
          </span>
          {/* 把「可以只去掉一个」说出来：不说的话用户只会看到全部采纳/忽略两个选项，
              于是为了一个错框而整批放弃。窄屏藏起来，别把这条挤出画布 */}
          <span className="hidden whitespace-nowrap text-micro text-white/45 md:inline">
            点框可去掉不要的
          </span>
          <button
            type="button"
            onClick={aiAcceptAll}
            className="surface-brand whitespace-nowrap rounded-full px-3.5 py-[5px] text-caption font-bold text-white transition-[filter,transform] duration-150 hover:brightness-110 active:scale-95"
            style={{ boxShadow: 'var(--shadow-brand)' }}
          >
            全部采纳
          </button>
          <button
            type="button"
            onClick={aiDiscard}
            className="glass-btn whitespace-nowrap rounded-full px-3 py-[5px] text-caption font-semibold"
          >
            忽略
          </button>
        </div>
      ) : null}

      {/* 顶部中央：点击标注的操作引导。
          这是「小白模式」的关键一环 —— 进入模式后必须让人当场知道「点哪、点完会怎样、
          怎么退出」，否则点一下没反应就会被判定为坏了。
          AI 状态条占用同一位置时自动下移，两条不叠。 */}
      {segActive ? (
        <div
          className="glass anim-up absolute left-1/2 z-20 flex -translate-x-1/2 items-center gap-2.5 rounded-full py-[7px] pl-3.5 pr-4"
          style={{ top: aiBarVisible ? 54 : 14 }}
        >
          {segBusy ? (
            <>
              <Loader2 className="size-[14px] animate-spin text-sky-300" />
              <span className="whitespace-nowrap text-caption font-semibold text-white/85">
                {segBusy === 'encode' ? '正在编码这张图（只需一次）…' : '正在生成标注框…'}
              </span>
            </>
          ) : (
            <>
              <MousePointerClick className="size-[14px] flex-shrink-0 text-sky-300" />
              <span className="whitespace-nowrap text-caption font-semibold text-white/85">
                {!segEncoded
                  ? '正在准备…'
                  : segPoints.length > 0
                    ? `已点 ${segPoints.length} 个点`
                    : '点一下物体，自动出框'}
              </span>
              <span className="whitespace-nowrap text-micro text-white/45">
                {segPoints.length > 0
                  ? '再点一下可修正 · 右键点掉多余区域'
                  : '点在框上可直接调整它 · 按住 Shift 点可叠加新框 · 右键排除 · Esc 退出'}
              </span>
              {/* 点歪了要有退路：以前只能 Esc 退出再进一次（点本身还看不见）。
                  现在点上看得见，也一键扔掉重来。 */}
              {segPoints.length > 0 ? (
                <button
                  type="button"
                  onClick={segResetPoints}
                  title="放弃这些点，重新开始"
                  className="flex flex-shrink-0 items-center gap-1 rounded-full bg-white/12 px-2 py-[3px] text-micro font-semibold text-white/85 transition-colors hover:bg-white/20"
                >
                  <RotateCcw className="size-3" />
                  清空重来
                </button>
              ) : null}
            </>
          )}
        </div>
      ) : null}

      {/* 左上角：图片信息 + 点击标注开关合成**一条**卡片。
          旧版是上下两个独立浮层（信息条 pointer-events-none + 下方按钮），
          圆角还不一致（16px vs 14px），左上角因此显得零碎。
          合并的另一个好处：高度从两行 76px 降到一行 34px，画布左上更干净。

          容器本身必须保持 pointer-events-none —— 它盖在画布上，
          整块可点会挡住画框 / 拖拽平移；只有按钮单独开 pointer-events-auto。 */}
      <div className="glass pointer-events-none absolute left-3.5 top-3.5 z-20 flex items-center gap-2.5 rounded-[16px] py-[5px] pl-3 pr-[5px]">
        <span className="max-w-[200px] truncate text-caption font-semibold text-white/90">
          {image?.name ?? '未选择图片'}
        </span>
        {image && image.width > 0 ? (
          <>
            <span className="h-3 w-px bg-white/15" />
            <span className="num text-micro text-white/45">
              {image.width}×{image.height}
            </span>
          </>
        ) : null}
        <span className="h-3 w-px bg-white/15" />
        <span className="num rounded-[7px] bg-white/12 px-1.5 py-[2px] text-micro font-bold text-white/85">
          {annotationCount} 框
        </span>
        {image && image.width > 0 && !imageMissing ? (
          <button
            type="button"
            onClick={toggleSeg}
            title={segActive ? '退出点击标注（Esc）' : '点一下物体，自动生成标注框'}
            className={`pointer-events-auto flex items-center gap-1.5 rounded-[11px] py-[5px] pl-2 pr-2.5 text-caption font-semibold transition-all duration-fast ease-smooth active:scale-[0.97] ${
              segActive
                ? 'surface-brand text-white [box-shadow:var(--shadow-brand)]'
                : 'bg-white/10 text-white/75 hover:bg-white/16 hover:text-white'
            }`}
          >
            <SegButtonLabel active={segActive} />
          </button>
        ) : null}
      </div>

      {/* 图片丢失：说清楚原因，别让人对着黑屏以为卡了 */}
      {imageMissing ? (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
          <div className="glass flex max-w-[320px] flex-col items-center gap-2 rounded-[20px] px-6 py-5 text-center">
            <ImageOff className="size-[22px] text-amber-400" />
            <div className="text-body font-bold text-white/90">这张图打不开了</div>
            <div className="text-caption leading-relaxed text-white/55">
              浏览器里保存的图片数据没了（可能是清理了站点数据）。
              <br />
              标注还在，重新导入同名图片即可继续。
            </div>
          </div>
        </div>
      ) : null}

      {/* 文件在但解不开：同样给明确原因，别让人对着黑屏以为程序坏了 */}
      {imageBroken ? (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
          <div className="glass flex max-w-[320px] flex-col items-center gap-2 rounded-[20px] px-6 py-5 text-center">
            <ImageOff className="size-[22px] text-amber-400" />
            <div className="text-body font-bold text-white/90">这张图打不开</div>
            <div className="text-caption leading-relaxed text-white/55">
              浏览器无法解码「{image.name}」。
              <br />
              多半是文件本身损坏，或格式不被当前浏览器支持。
              <br />
              标注不会丢，可以在左侧把它移除后重新导入别的文件。
            </div>
          </div>
        </div>
      ) : null}

      {/* 空状态引导 */}
      {image && !imageMissing && !imageBroken && annotationCount === 0 ? (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
          <div className="anim-up flex flex-col items-center gap-5">
            <div className="relative size-[88px]">
              <svg width="88" height="88" viewBox="0 0 88 88" fill="none" className="absolute inset-0">
                <rect
                  x="9"
                  y="9"
                  width="70"
                  height="70"
                  rx="11"
                  stroke="rgba(255,255,255,0.32)"
                  strokeWidth="1.5"
                  strokeDasharray="6 5"
                  className="anim-dash"
                />
              </svg>
              <div
                className="absolute left-1/2 top-1/2 size-9 -translate-x-1/2 -translate-y-1/2 rounded-[12px]"
                style={{ background: 'rgba(37,99,235,0.18)', border: '1px solid rgba(37,99,235,0.35)' }}
              />
              <div
                className="anim-breathe absolute -right-0.5 -top-0.5 size-[11px] rounded-full"
                style={{
                  background: 'var(--brand)',
                  boxShadow: '0 0 0 4px rgba(37,99,235,0.24)',
                }}
              />
            </div>
            <div className="text-center">
              <div className="text-title font-bold tracking-tight text-white/92">
                按住左键拖出一个框
              </div>
              <div className="mt-1.5 text-body text-white/45">
                {labels.length === 0
                  ? '松手后给它起个名字，第一个类别就建好了'
                  : '松手后会在框旁边弹出标签，点一下就完成'}
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {/* 左下角：操作图例（有选中项时让位） */}
      {!activeMenu && !selectedId ? (
        <div className="glass pointer-events-none absolute bottom-3.5 left-3.5 z-10 flex items-center gap-2 rounded-[16px] px-2.5 py-[7px] text-micro text-white/45">
          {(
            [
              ['拖拽', '画框'],
              ['拖角点', '缩放'],
              ['滚轮', '缩放'],
              ['图外拖动', '平移'],
              ['右键框', '改标签'],
            ] as const
          ).map(([key, act]) => (
            <span key={key} className="flex items-center gap-1">
              <kbd className="rounded-[6px] border border-white/15 bg-white/10 px-[5px] py-[1px] text-micro font-bold text-white/75">
                {key}
              </kbd>
              {act}
            </span>
          ))}
        </div>
      ) : null}

      {/* 右下角：缩放控制 */}
      <div className="glass absolute bottom-3.5 right-3.5 z-10 flex items-center gap-0.5 rounded-[16px] p-1">
        <button type="button" onClick={() => zoomBy(1 / 1.25)} className="glass-btn size-[27px]" title="缩小 (-)">
          <ZoomOut className="size-[15px]" />
        </button>
        <span className="num w-12 text-center text-caption font-semibold text-white/75">
          {zoomPercent}%
        </span>
        <button type="button" onClick={() => zoomBy(1.25)} className="glass-btn size-[27px]" title="放大 (+)">
          <ZoomIn className="size-[15px]" />
        </button>
        <div className="mx-1 h-4 w-px bg-white/12" />
        <button
          type="button"
          onClick={fit}
          className="glass-btn h-[27px] gap-1 px-2 text-micro font-semibold"
          title="适应窗口 (F)"
        >
          <Maximize className="size-3" />
          适应
        </button>
      </div>
    </div>
  );
}
