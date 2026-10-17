import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDownAZ, ArrowUpAZ, GripVertical, ImageOff, Images, Plus, Search, Trash2, X } from 'lucide-react';
import { useStore } from '../store/useStore';
import type { ImageItem } from '../types';
import { getThumbnailSync, loadThumbnail } from '../utils/imageCache';
import { reportAddImages } from '../store/toastStore';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

/* ===================================================================
   左栏虚拟列表的几何常量
   -------------------------------------------------------------------
   为什么必须虚拟化：703 张图 = 703 个卡片 DOM（每张还挂 Tooltip 子树）。
   原先是靠 `contentVisibility: auto` 兜着，但它只跳过视口外的**布局与绘制**，
   React 元素、DOM 节点、IntersectionObserver 一个都没少。

   为什么用「固定行高」而不是测量行高：测量方案要监听栏宽、按 aspectRatio
   反推行高，任何 1px 误差都会在几百行后累积成滚动错位（滚到底时高度对不上，
   出现空白或被截断）。这里把**图区锁成固定高**，配 `object-cover` 保证不变形，
   行高就成了纯常量 —— 总高恒等于 `ROW_H*(rows-1)+CARD_H`，与滚动位置无关，永不漂移。
   =================================================================== */
const COLS = 2; // 网格列数
const GAP = 8; // 对应 gap-2
const TILE_H = 84; // 缩略图区域固定高（不锁 1:1，靠 object-cover 抗变形）
const CARD_H = 108; // 卡片总高 = 图区 84 + 名字行
const ROW_H = CARD_H + GAP; // 一行占位高度
const OVERSCAN = 3; // 视口外多渲染几行，快速滚动时不露白

/** 缩略图：结果永久缓存。虚拟列表下只有可见卡片会挂载，无需再观察视口 */
const Thumb = memo(function Thumb({ item }: { item: ImageItem }) {
  const [src, setSrc] = useState<string | null>(() => getThumbnailSync(item.id));
  /** 有文件但解不出缩略图 → 区分于「文件本来就不在」 */
  const [failed, setFailed] = useState(false);
  // item 每次标注变化都是新对象，放 ref 里给 effect 读，
  // 这样 observer 的依赖可以只认 id（见下）
  const itemRef = useRef(item);
  itemRef.current = item;

  useEffect(() => {
    if (src || !item.file) return;
    let cancelled = false;
    // 虚拟列表下只有可见卡片会被挂载，IntersectionObserver 已是多余；
    // loadThumbnail 内部有**永久缓存 + inflight 去重**，来回滚动不会重复解码。
    // 依赖只认 id：item 在每次标注变化时都是新对象，依赖它会白白重建副作用。
    void loadThumbnail(itemRef.current).then((url) => {
      if (cancelled) return;
      // url 为 null 就什么都不做会留一块灰底，用户分不清「还没加载」和「这张图坏了」
      if (url) setSrc(url);
      else setFailed(true);
    });
    return () => {
      cancelled = true;
    };
  }, [item.id, item.file, src]);

  return (
    <div className="h-full w-full overflow-hidden bg-muted">
      {src ? (
        <img
          src={src}
          alt={item.name}
          draggable={false}
          className="h-full w-full scale-[1.001] object-cover transition-transform duration-300 group-hover:scale-[1.06]"
        />
      ) : failed ? (
        <div className="flex h-full w-full flex-col items-center justify-center gap-1 text-ink-4">
          <ImageOff className="size-[15px]" />
          <span className="text-micro leading-none">无法解码</span>
        </div>
      ) : null}
    </div>
  );
});

interface CardProps {
  item: ImageItem;
  index: number;
  active: boolean;
  onSelect: (index: number) => void;
  onRemove: (item: ImageItem) => void;
  onDragStart: (id: string) => void;
  onDragEnter: (id: string) => void;
  onDragEnd: () => void;
  dragging: boolean;
  dragOver: boolean;
}

/**
 * 单独 memo：标注变化 / 切图时只有受影响的卡片会重渲染。
 *
 * 关键：所有回调必须来自父组件的稳定引用（useCallback 或 store action），
 * 且这里传的是**图片 id**而不是下标 —— 早先父组件在 map 里内联箭头函数，
 * 每次渲染都是新引用，memo 的浅比较必然失败，等于白包一层：
 * 2000 张图时每按一次 → 全部卡片重渲染（每卡还带两个 Radix Tooltip 子树）。
 */
const ImageCard = memo(function ImageCard({
  item,
  index,
  active,
  onSelect,
  onRemove,
  onDragStart,
  onDragEnter,
  onDragEnd,
  dragging,
  dragOver,
}: CardProps) {
  const count = item.annotations.length;
  const missing = !item.file;

  return (
    <div
      className="group relative overflow-hidden"
      // data-image-card 是给测试用的稳定钩子：单靠 [data-index] 会被 sonner 的
      // toast 元素误命中（它内部也用 data-index），测试会莫名其妙数错卡片
      data-image-card=""
      data-index={index}
      // 固定高度：虚拟列表的滚动占位全靠它精确，任何浮动都会让滚动条漂移。
      // （原先的 contentVisibility 已移除 —— 虚拟化后本来就只有可见项存在）
      style={{ height: CARD_H }}
      draggable
      onDragStart={(e) => {
        e.dataTransfer.effectAllowed = 'move';
        onDragStart(item.id);
      }}
      onDragEnter={() => onDragEnter(item.id)}
      onDragEnd={onDragEnd}
      onClick={() => onSelect(index)}
      title={item.name}
    >
      {/* 拖拽手柄：悬停时浮现，提示可重新排序 */}
      <span className="pointer-events-none absolute left-[3px] top-1/2 z-10 hidden -translate-y-1/2 group-hover:flex">
        <GripVertical className="size-3.5 text-white/75 [filter:drop-shadow(0_1px_1px_rgba(0,0,0,0.5))]" />
      </span>

      <div
        className={`cv-item relative overflow-hidden rounded-[16px] border bg-card transition-all duration-200 ${
          active
            ? 'border-primary ring-[2px] ring-primary/30 [box-shadow:0_4px_14px_-2px_rgba(37,99,235,0.34)]'
            : 'border-border [box-shadow:var(--shadow-xs)] group-hover:border-input group-hover:[box-shadow:var(--shadow-md)]'
        } ${missing ? 'opacity-55' : ''} ${dragging ? 'opacity-40' : ''} ${
          dragOver ? 'ring-[2px] ring-primary/55' : ''
        }`}
        style={{ height: TILE_H }}
      >
        <Thumb item={item} />

          {/* 底部渐隐，让浮在图上的读数始终可读 */}
          <div
            className="pointer-events-none absolute inset-x-0 bottom-0 h-[42%]"
            style={{ background: 'linear-gradient(to top, rgba(8,10,16,0.5), transparent)' }}
          />

          <span className="num absolute left-1 top-1 rounded-[8px] bg-black/55 px-[5px] py-[1px] text-micro font-bold text-white/90 backdrop-blur-[2px]">
            {index + 1}
          </span>

          {count > 0 ? (
            // 只留一个「已标注」信号。以前右下计数徽标 + 左下绿点说的是同一件事，
            // 叠上左上序号、右上删除，**四个角全被占满** —— 少一个角，画面立刻安静。
            <span className="surface-brand num absolute bottom-1 right-1 rounded-[8px] px-[5px] py-[1px] text-micro font-bold text-white [box-shadow:0_1px_5px_rgba(0,0,0,0.32)]">
              {count}
            </span>
          ) : null}

          {active ? (
            <span className="pointer-events-none absolute inset-x-0 top-0 h-[3px] surface-brand" />
          ) : null}

          {missing ? (
            <span className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-card/70 text-micro font-semibold text-destructive">
              <ImageOff className="size-[18px]" />
              图片丢失
            </span>
          ) : null}
        </div>

        {/* Phase 9：文件名按「有没有标注」分色。扫一眼左栏就能看出进度，
            不必逐个去看右下角有没有蓝徽标。选中的仍用最强的一档。 */}
        <div
          className={`mt-1.5 truncate px-0.5 text-micro leading-[17px] ${
            active ? 'font-semibold text-foreground' : count > 0 ? 'text-ink-2' : 'text-ink-4'
          }`}
        >
          {item.name}
        </div>

      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onRemove(item);
            }}
            title="移除这张图片"
            className="absolute right-1.5 top-1.5 hidden size-[22px] items-center justify-center rounded-[9px] bg-black/55 text-white/85 backdrop-blur-[3px] transition-all hover:bg-destructive hover:text-white group-hover:flex"
          >
            <Trash2 className="size-[11px]" />
          </button>
        </TooltipTrigger>
        <TooltipContent>移除这张图片</TooltipContent>
      </Tooltip>
    </div>
  );
});

/** 网格末尾的虚线追加块，点击与拖放都走原有的添加流程 */
function AddTile({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title="添加图片"
      className="cv-item group/add flex flex-col items-center justify-center gap-1 rounded-[16px] border-2 border-dashed border-line-strong bg-[var(--surface-inset)] text-ink-4 transition-all duration-200 hover:border-primary/60 hover:bg-primary/[0.04] hover:text-primary"
      style={{ height: TILE_H }}
    >
      <Plus className="size-[18px] transition-transform duration-200 group-hover/add:rotate-90" />
      <span className="text-micro font-semibold">添加</span>
    </button>
  );
}

export function ImageSidebar() {
  const images = useStore((s) => s.images);
  // 订阅版本号（数字）用于给上面的统计做缓存；它只在「框增删 / 归属变化」时才变
  const labelRevision = useStore((s) => s.labelRevision);
  const activeIndex = useStore((s) => s.activeIndex);
  const setActiveIndex = useStore((s) => s.setActiveIndex);
  const addImages = useStore((s) => s.addImages);
  const removeImage = useStore((s) => s.removeImage);
  const sortImages = useStore((s) => s.sortImages);
  const reorderImages = useStore((s) => s.reorderImages);

  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [adding, setAdding] = useState(false);
  const [pendingRemove, setPendingRemove] = useState<ImageItem | null>(null);
  const [dragId, setDragId] = useState<string | null>(null);
  const [overId, setOverId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  // 虚拟列表：滚动位置 + 视口高度
  const [scrollTop, setScrollTop] = useState(0);
  // 初值给个合理估计：ResizeObserver 首帧才回填真实高度，若从 0 起步，
  // 首帧只会渲染 OVERSCAN 那几行、下一帧再补齐，会看到一次跳动
  const [viewH, setViewH] = useState(600);

  // 拖拽相关回调必须是稳定引用，否则下面每张卡片的 memo 都会失效
  const dragIdRef = useRef<string | null>(null);
  dragIdRef.current = dragId;
  const handleDragStart = useCallback((id: string) => setDragId(id), []);
  const handleDragEnter = useCallback((id: string) => {
    if (dragIdRef.current) setOverId(id);
  }, []);
  const handleDragEnd = useCallback(() => {
    setDragId(null);
    setOverId(null);
  }, []);

  const handleAdd = useCallback(
    (files: File[]) => {
      if (files.length === 0) return;
      setAdding(true);
      // 让浏览器先把「正在导入」这一帧画出来，再去做同步的建条目工作
      window.setTimeout(() => {
        const result = addImages(files);
        setAdding(false);
        reportAddImages(result);
      }, 20);
    },
    [addImages],
  );

  // 标注统计：只依赖「图片数量」和「标签分布版本号」。
  // 直接依赖 images 的话，**拖动框边也会换掉数组引用**（而计数一个都没变），
  // 于是每帧都白跑一遍全量 reduce —— 这是上万张图时实打实的浪费。
  const annotatedCount = useMemo(
    () => images.reduce((sum, i) => sum + (i.annotations.length > 0 ? 1 : 0), 0),
    // labelRevision 涵盖了「框增删 / 归属变化 / 删除图片 / 撤销重做 / 切换项目」
    [images.length, labelRevision],
  );

  // 按名称筛选后保留原数组下标：排序/选中/拖拽都继续用真实 index
  const normQuery = query.trim().toLowerCase();
  // 无筛选时的下标数组只跟「图片数量」有关，单独缓存 ——
  // 否则每次落框都要为一万张图的列表重新分配一个等长数组。
  const allIndices = useMemo(() => images.map((_, i) => i), [images.length]);
  const filtered = useMemo(() => {
    if (!normQuery) return allIndices;
    const out: number[] = [];
    for (let i = 0; i < images.length; i += 1) {
      if (images[i].name.toLowerCase().includes(normQuery)) out.push(i);
    }
    return out;
  }, [allIndices, normQuery, images]);

  /* ---------------- 虚拟列表 ---------------- */

  // 视口高度：栏宽 / 窗口变化都要重算
  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    const sync = () => setViewH(el.clientHeight);
    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // 末尾还有一个「添加」格子，它同样占一格
  const showAddTile = images.length > 0 && filtered.length > 0;
  const total = filtered.length + (showAddTile ? 1 : 0);
  const rows = Math.ceil(total / COLS);

  const firstRow = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
  const lastRow =
    rows === 0 ? -1 : Math.min(rows - 1, Math.floor((scrollTop + viewH) / ROW_H) + OVERSCAN);
  const startIdx = firstRow * COLS;
  const endIdx = Math.min(total, (lastRow + 1) * COLS);

  // 上下两块占位：它们与可见区加起来恒等于内容总高，所以滚动条不会随滚动漂移
  const padTop = firstRow * ROW_H;
  const padBottom = Math.max(0, (rows - 1 - lastRow) * ROW_H);

  // 可见项（过滤后的位置 → 真实下标）
  const visible: number[] = [];
  for (let i = startIdx; i < Math.min(endIdx, filtered.length); i += 1) visible.push(filtered[i]);

  /**
   * 切图时把当前图滚进视口。
   * **不能用 scrollIntoView** —— 虚拟列表下目标卡片可能压根没挂载，
   * 它会静默失败（既不滚动也不报错）。这里直接按行号算 scrollTop。
   */
  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    const pos = filtered.indexOf(activeIndex);
    if (pos < 0) return;
    const rowTop = Math.floor(pos / COLS) * ROW_H;
    const rowBottom = rowTop + CARD_H;
    if (rowTop < el.scrollTop) el.scrollTop = rowTop;
    else if (rowBottom > el.scrollTop + el.clientHeight) {
      el.scrollTop = rowBottom - el.clientHeight;
    }
  }, [activeIndex, filtered]);

  return (
    // 左栏宽度：176px 下限下缩略图只有约 78px，扫视时基本认不出内容。
    // 提到 196px 起。左右栏合计上限 560px，1280 屏上画布仍有 720px，不会被挤没。
    <aside className="flex w-[clamp(196px,17vw,240px)] flex-shrink-0 flex-col overflow-hidden rounded-xl border border-hairline bg-surface-panel [box-shadow:var(--shadow-panel)]">
      <div className="panel-head">
        <div className="flex items-center gap-1.5">
          <span className="section-label">图片</span>
          <span className="count-pill">{images.length}</span>
        </div>
        <div className="flex items-center gap-0.5">
          <DropdownMenu>
            <Tooltip>
              <TooltipTrigger asChild>
                <DropdownMenuTrigger asChild>
                  <button
                    type="button"
                    title="排序"
                    className="raised flex size-[24px] items-center justify-center rounded-[10px] border border-border bg-card text-ink-3 transition-all hover:border-primary/45 hover:bg-primary/[0.06] hover:text-primary active:scale-95"
                  >
                    <ArrowDownAZ className="size-[14px]" />
                  </button>
                </DropdownMenuTrigger>
              </TooltipTrigger>
              <TooltipContent>按名称排序</TooltipContent>
            </Tooltip>
            <DropdownMenuContent align="end" className="w-40">
              <DropdownMenuItem onClick={() => sortImages('name')}>
                <ArrowDownAZ />
                名称升序
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => sortImages('name-desc')}>
                <ArrowUpAZ />
                名称降序
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                title="添加图片"
                className="raised flex size-[24px] items-center justify-center rounded-[10px] border border-border bg-card text-ink-3 transition-all hover:border-primary/45 hover:bg-primary/[0.06] hover:text-primary active:scale-95"
              >
                <Plus className="size-[14px]" />
              </button>
            </TooltipTrigger>
            <TooltipContent>添加图片（可多选 / 拖文件夹）</TooltipContent>
          </Tooltip>
        </div>
      </div>

      {/* 按名称筛选：图片一多，光靠排序翻不过来 */}
      <div className="shrink-0 px-2.5 pb-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-[9px] top-1/2 size-3 -translate-y-1/2 text-ink-4" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') {
                setQuery('');
                e.currentTarget.blur();
              }
            }}
            placeholder="按名称筛选…"
            className="h-[27px] w-full rounded-[12px] border border-border bg-card pl-[22px] pr-[46px] text-caption text-ink-2 outline-none transition-colors placeholder:text-ink-4 focus:border-primary/50 focus:ring-[2px] focus:ring-primary/20"
          />
          {/* 筛选计数：以前筛选后完全看不出「漏了多少张」 */}
          {query ? (
            <span className="num pointer-events-none absolute right-[22px] top-1/2 -translate-y-1/2 text-micro text-ink-4">
              {filtered.length}/{images.length}
            </span>
          ) : null}
          {query ? (
            <button
              type="button"
              onClick={() => setQuery('')}
              title="清除筛选"
              className="absolute right-1 top-1/2 flex size-[17px] -translate-y-1/2 items-center justify-center rounded-[6px] text-ink-4 transition-colors hover:bg-hover hover:text-ink-2"
            >
              <X className="size-[11px]" />
            </button>
          ) : null}
        </div>
      </div>

      {/* 原生滚动容器（替换 radix ScrollArea）：
          虚拟化必须能拿到 onScroll，而 ScrollArea 的滚动元素藏在内部 viewport 里，
          得跨组件钻一层 ref 才监听得到。全站 ::-webkit-scrollbar 样式在 index.css，
          所以换掉后滚动条外观**完全不变**。
          px-3 挂在滚动容器上：滚动条因此贴着面板边缘，不会压在卡片上。 */}
      <div
        ref={listRef}
        data-image-scroller=""
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
        className="scroll-thin min-h-0 flex-1 overflow-x-hidden overflow-y-auto px-3 pb-3"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          // 外部拖入的文件（图片 / 已标注数据集）不在这里处理：
          // 必须让它继续冒泡给 Editor，由那边统一处理（还能识别 labels/*.txt）。
          // 之前在这里 stopPropagation，Editor 永远收不到，标注也读不进来。
          if (e.dataTransfer.files && e.dataTransfer.files.length > 0) return;
          // 没有文件 → 视为排序拖拽的落点
          e.preventDefault();
          // 拖拽态存的是图片 id，落点下标从落点元素上读回来
          const from = dragId ? images.findIndex((i) => i.id === dragId) : -1;
          const to = overId ? images.findIndex((i) => i.id === overId) : -1;
          if (from >= 0 && to >= 0 && from !== to) {
            reorderImages(from, to);
          }
          handleDragEnd();
        }}
      >
        {images.length === 0 ? (
          <div className="mt-4 flex flex-col items-center gap-2 rounded-[18px] border border-dashed border-line-strong bg-card/50 px-4 py-7 text-center">
            {/* Phase 9「蓝色收敛」：空状态图标原本是淡蓝底 + 蓝图标 —— 蓝色会让人
                以为它是个**可点的按钮**。空状态只是提示，退成中性才对。 */}
            <span className="flex size-[34px] items-center justify-center rounded-[14px] bg-[var(--surface-inset)] text-ink-4">
              <Images className="size-[17px]" />
            </span>
            <p className="text-caption leading-relaxed text-muted-foreground">
              还没有图片
              <br />
              点上面的 + 或直接把图拖进来
            </p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="mt-1 flex flex-col items-center gap-2 rounded-[16px] border border-dashed border-line-strong bg-card/40 px-3 py-5 text-center">
            <ImageOff className="size-[15px] text-ink-4" />
            <p className="text-micro leading-relaxed text-ink-4">
              没有名字匹配
              <br />
              「{query.trim()}」的图片
            </p>
            {/* 以前这里只把 query 重复一遍，用户没有下一步动作 */}
            <button
              type="button"
              onClick={() => setQuery('')}
              className="rounded-[9px] border border-border bg-card px-2 py-[3px] text-micro text-ink-3 transition-colors hover:border-primary/45 hover:text-primary"
            >
              清除筛选
            </button>
          </div>
        ) : (
          <>
            {/* 上下两块占位撑起总高度：内容总高恒为 ROW_H*(rows-1)+CARD_H，与 scrollTop 无关 */}
            <div style={{ height: padTop }} />
            {/* gridAutoRows 强制每行都是 CARD_H：
                「添加」格子只有 84px 高（和缩略图区对齐），若最后一行恰好只有它，
                行高就会塌成 84，与占位公式里的 108 差 24px → 滚动条会漂。 */}
            <div className="grid grid-cols-2 gap-2" style={{ gridAutoRows: CARD_H }}>
              {visible.map((index) => {
                const item = images[index];
                return (
                  <ImageCard
                    key={item.id}
                    item={item}
                    index={index}
                    active={index === activeIndex}
                    onSelect={setActiveIndex}
                    onRemove={setPendingRemove}
                    onDragStart={handleDragStart}
                    onDragEnter={handleDragEnter}
                    onDragEnd={handleDragEnd}
                    dragging={dragId === item.id}
                    dragOver={overId === item.id && dragId !== null && dragId !== item.id}
                  />
                );
              })}
              {showAddTile && endIdx > filtered.length ? (
                <AddTile onClick={() => inputRef.current?.click()} />
              ) : null}
            </div>
            <div style={{ height: padBottom }} />
          </>
        )}
      </div>

      {/* 底部统计 */}
      <div className="flex flex-shrink-0 items-center justify-between gap-2 border-t border-hairline px-3 py-2">
        {adding ? (
          <span className="text-micro font-medium text-primary">正在导入…</span>
        ) : (
          <span className="num text-micro text-ink-4">
            已标 {annotatedCount} / {images.length}
          </span>
        )}
        <span className="text-micro text-ink-4">改动自动保存</span>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = '';
          handleAdd(files);
        }}
      />

      <AlertDialog
        open={pendingRemove !== null}
        onOpenChange={(v) => {
          if (!v) setPendingRemove(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>移除这张图片？</AlertDialogTitle>
            <AlertDialogDescription>
              {pendingRemove && pendingRemove.annotations.length > 0 ? (
                <>
                  「{pendingRemove.name}」上还有{' '}
                  <b className="num">{pendingRemove.annotations.length}</b> 个标注框，移除后会一起消失。
                  {images.length > 1 ? ' 移错了按 Ctrl+Z 就能撤回来。' : ''}
                </>
              ) : (
                <>
                  「{pendingRemove?.name}」会被移出这个项目。
                  {images.length > 1 ? ' 移错了按 Ctrl+Z 就能撤回来。' : ''}
                </>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-white hover:bg-destructive/90"
              onClick={() => {
                if (pendingRemove) removeImage(pendingRemove.id);
                setPendingRemove(null);
              }}
            >
              确认移除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </aside>
  );
}
