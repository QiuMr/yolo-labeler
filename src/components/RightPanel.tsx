import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { CheckSquare, CirclePlus, Copy, Eye, EyeOff, Layers, ListChecks, MousePointerClick, Plus, Tag, Trash2 } from 'lucide-react';
import { useStore } from '../store/useStore';
import { toast } from '../store/toastStore';
import { PALETTE, readableInk } from '../utils/color';
import type { Annotation, IRect, Label } from '../types';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { useLabelCounts } from '../hooks/useLabelCounts';

/** 两个矩形是否相交（用于「重叠框」识别） */
function rectsOverlap(a: IRect, b: IRect): boolean {
  return (
    a.x < b.x + b.width &&
    a.x + a.width > b.x &&
    a.y < b.y + b.height &&
    a.y + a.height > b.y
  );
}

function SectionHeader({
  title,
  count,
  hint,
  action,
  tone = 'default',
}: {
  title: string;
  count?: number;
  hint?: string;
  /** 右侧操作元素（如「多选」开关）。有它，两个分区才能共用同一套 padding，
   *  否则手写标题行的内边距和 panel-head 对不上，两条标题基线会明显错开 */
  action?: ReactNode;
  tone?: 'default' | 'brand';
}) {
  return (
    <div className="panel-head">
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="section-label">{title}</span>
        {count !== undefined ? <span className="count-pill">{count}</span> : null}
      </div>
      {action ??
        (hint ? (
          <span
            className={`flex-shrink-0 whitespace-nowrap text-micro font-semibold tabular-nums ${
              tone === 'brand' ? 'text-primary' : 'text-ink-4'
            }`}
          >
            {hint}
          </span>
        ) : null)}
    </div>
  );
}

/** 非选中行传同一个空数组引用，避免每次渲染产生新数组导致 memo 失效 */
const NO_OVERLAPS: string[] = [];

/**
 * 「本图标注」里的一行。单独 memo：
 * 鼠标在框之间移动时 hoverId 会变化，若整列都不 memo，几十上百个框的行会被反复 reconcile。
 * memo 后，只有 ann / 选中态 / 悬停态真正变化的行才会重渲染。
 */
const AnnotationRow = memo(function AnnotationRow({
  ann,
  index,
  label,
  selected,
  batchMode,
  overlaps,
  onRequestAddLabel,
}: {
  ann: Annotation;
  index: number;
  label: Label | undefined;
  selected: boolean;
  batchMode: boolean;
  /** 当前框的重叠组（含自己，按图层顺序；仅选中行有值，其余为空数组） */
  overlaps: string[];
  onRequestAddLabel: () => void;
}) {
  // 订阅的都是稳定的 action 引用，不会因 store 变化触发多余渲染
  const select = useStore((s) => s.select);
  const toggleSelectBox = useStore((s) => s.toggleSelectBox);
  const setActiveLabel = useStore((s) => s.setActiveLabel);
  const setBoxLabel = useStore((s) => s.setBoxLabel);
  const deleteBox = useStore((s) => s.deleteBox);
  const setHover = useStore((s) => s.setHover);
  // Phase 11：悬停态下沉到行内部自订阅。
  // 原来由父组件订阅 hoverId 再当 prop 传下来 ——
  // 鼠标每跨过一个框，父组件就要重渲染，并把整列 `annotations.map()` 重跑一遍
  // （几百个元素重建 + 几百次 memo props 比对），而真正变化的只有两行。
  // 现在父组件完全不感知 hover，只有「刚离开的行」和「刚进入的行」会重渲染。
  const hovered = useStore((s) => s.hoverId === ann.id);

  const color = label?.color ?? '#9aa2b1';
  const hasOverlap = overlaps.length > 1;

  return (
    <div
      onMouseEnter={() => setHover(ann.id)}
      onMouseLeave={() => setHover(null)}
      className="group relative mb-[3px] flex items-center gap-2 rounded-[12px] border border-transparent py-[6px] pl-2 pr-2 transition-colors duration-150 hover:bg-hover"
      style={
        selected
          ? { background: `${color}16`, borderColor: `${color}55` }
          : hovered
            ? { background: 'var(--hover)' }
            : undefined
      }
    >
      <span
        className="absolute left-0 top-1/2 h-[58%] w-[3px] -translate-y-1/2 rounded-r-full transition-opacity duration-150"
        style={{ background: color, opacity: selected ? 1 : 0 }}
      />

      {batchMode ? (
        <span
          className={`flex size-[14px] flex-shrink-0 items-center justify-center rounded-[4px] border ${
            selected ? 'border-transparent bg-primary text-white' : 'border-line-strong bg-card'
          }`}
        >
          {selected ? <CheckSquare className="size-[11px]" /> : null}
        </span>
      ) : (
        <span className="num w-3.5 flex-shrink-0 text-micro font-semibold text-ink-4">
          {index + 1}
        </span>
      )}

      <div
        role="button"
        tabIndex={0}
        // 测试钩子：端到端测试要精确数「本图有几个框」。
        // 以前靠 title 里含「标签/类别」来猜，会把「套用当前类别」这类按钮也数进去，
        // 导致有 1 个框时算出 2，断言 `=== 1` 永远不成立。
        data-ann-row={ann.id}
        className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 outline-none"
        onClick={() => {
          if (batchMode) {
            toggleSelectBox(ann.id, false);
            return;
          }
          select(ann.id);
          if (ann.labelId) setActiveLabel(ann.labelId);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            if (batchMode) toggleSelectBox(ann.id, false);
            else select(ann.id);
          }
        }}
      >
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                select(ann.id);
                const labels = useStore.getState().labels;
                if (labels.length === 0) {
                  onRequestAddLabel();
                  return;
                }
                const i = labels.findIndex((l) => l.id === ann.labelId);
                const next = labels[(i + 1) % labels.length];
                if (next) setBoxLabel(ann.id, next.id);
              }}
              title={
                useStore.getState().labels.length === 0
                  ? '还没有类别，点这里新建一个'
                  : label
                    ? '点击切换到下一个标签'
                    : '点击给它选一个类别'
              }
              className={`max-w-[106px] flex-shrink-0 truncate rounded-[9px] px-1.5 py-[3px] text-caption font-bold transition-transform duration-150 hover:scale-[1.04] ${
                label ? '' : 'border border-dashed border-ink-4'
              }`}
              style={{
                background: `${color}1f`,
                color: label ? readableInk(color) : '#4e5666',
              }}
            >
              {label?.name ?? '未分类'}
            </button>
          </TooltipTrigger>
          <TooltipContent>
            <span className="flex items-center gap-1">
              <MousePointerClick className="size-3" />
              点击切换标签
            </span>
          </TooltipContent>
        </Tooltip>

        <span className="num ml-auto flex-shrink-0 text-micro text-ink-4">
          {Math.round(ann.rect.width)}×{Math.round(ann.rect.height)}
        </span>
      </div>

      {/* 重叠框：选中某层时给出「切到下一层」入口，被完全盖住的框也能选 */}
      {hasOverlap ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                // overlaps 含自己且已按图层顺序排列：直接取「下一个」就能一圈圈轮换
                const at = overlaps.indexOf(ann.id);
                const next = overlaps[(at + 1) % overlaps.length];
                select(next);
              }}
              title="还有重叠的框，点这里切换到下一层"
              aria-label="切换重叠框"
              className="hidden size-[19px] flex-shrink-0 items-center justify-center rounded-[8px] text-ink-3 transition-colors hover:bg-primary/12 hover:text-primary group-hover:flex"
            >
              <Layers className="size-[12px]" />
            </button>
          </TooltipTrigger>
          <TooltipContent>切换重叠框（共 {overlaps.length} 层）</TooltipContent>
        </Tooltip>
      ) : null}

      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              deleteBox(ann.id);
            }}
            title="删除这个框"
            aria-label="删除这个框"
            className="hidden size-[19px] flex-shrink-0 items-center justify-center rounded-[8px] text-ink-4 transition-colors hover:bg-destructive/12 hover:text-destructive group-hover:flex"
          >
            <Trash2 className="size-[11px]" />
          </button>
        </TooltipTrigger>
        <TooltipContent>删除 (Delete)</TooltipContent>
      </Tooltip>
    </div>
  );
});

export function RightPanel() {
  const images = useStore((s) => s.images);
  const activeIndex = useStore((s) => s.activeIndex);
  const labels = useStore((s) => s.labels);
  const activeLabelId = useStore((s) => s.activeLabelId);
  const selectedId = useStore((s) => s.selectedId);
  const selectedBoxIds = useStore((s) => s.selectedBoxIds);
  const hiddenLabelIds = useStore((s) => s.hiddenLabelIds);
  const toggleLabelHidden = useStore((s) => s.toggleLabelHidden);
  const applyActiveToUnclassified = useStore((s) => s.applyActiveToUnclassified);

  const addLabel = useStore((s) => s.addLabel);
  const renameLabel = useStore((s) => s.renameLabel);
  const removeLabel = useStore((s) => s.removeLabel);
  const setLabelColor = useStore((s) => s.setLabelColor);
  const setActiveLabel = useStore((s) => s.setActiveLabel);
  const clearSelectBoxes = useStore((s) => s.clearSelectBoxes);
  const applyLabelToBoxes = useStore((s) => s.applyLabelToBoxes);
  const copyPrevAnnotations = useStore((s) => s.copyPrevAnnotations);

  const image = images[activeIndex] ?? null;

  const [batchMode, setBatchMode] = useState(false);

  /**
   * 当前**选中框**的重叠组（含自己，按图层顺序排列），用于「切换重叠框」。
   *
   * 以前是给每个框都算一遍重叠组（O(n²) + 每组一次排序 + n 次 Map/数组分配），
   * 但结果只在「该行正好被选中」时才被读到，其余 n-1 组算完就丢。
   * 依赖是 [image]，也就是说拖完框、改类别、AI 落框……每次编辑都会重算一遍。
   * 密集检测图单图 200~500 框很常见，200² = 4 万次比较 × 每次编辑。
   * 现在只算选中的那一个：O(n)，一次分配。
   */
  const selectedOverlaps = useMemo(() => {
    const anns = image?.annotations ?? [];
    if (!selectedId) return NO_OVERLAPS;
    const sel = anns.find((a) => a.id === selectedId);
    if (!sel) return NO_OVERLAPS;
    const order = new Map(anns.map((a, i) => [a.id, i] as const));
    const group = [sel.id];
    for (const a of anns) {
      if (a.id !== sel.id && rectsOverlap(sel.rect, a.rect)) group.push(a.id);
    }
    // 按图层顺序排好，轮换才会沿着「下一层」稳定前进，而不是原地打转
    group.sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));
    return group;
  }, [image, selectedId]);

  const onCopyPrev = () => {
    if (copyPrevAnnotations()) toast.success('已复制上一张图的标注');
    else toast.error(activeIndex <= 0 ? '已经是第一张了' : '上一张还没有标注');
  };

  const onApplyActive = () => {
    if (applyActiveToUnclassified()) toast.success('已为未分类的框套用当前类别');
    else toast.error(!activeLabelId ? '先在右侧或顶栏选一个类别' : '本图没有未分类的框');
  };

  const onApplyToSelected = (labelId: string) => {
    if (selectedBoxIds.length === 0) return;
    // 用 store 返回的「真实改动数」而不是传入的 ids.length：
    // ids 全部指向别的图时不该弹出「已为 N 个框套用类别」这种假成功。
    const changed = applyLabelToBoxes(selectedBoxIds, labelId);
    if (changed === 0) {
      toast.info('这些框的类别已经是一样的了，没有改动');
      clearSelectBoxes();
      return;
    }
    toast.success(`已为 ${changed} 个框套用类别`);
  };

  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState('');
  const [newLabel, setNewLabel] = useState('');
  const [adding, setAdding] = useState(false);

  // 稳定引用：行组件 memo 需要每次渲染都能用同一个回调
  const requestAddLabel = useCallback(() => setAdding(true), []);
  const newInputRef = useRef<HTMLInputElement>(null);
  const editInputRef = useRef<HTMLInputElement>(null);
  // Esc 取消时会立刻卸载输入框，blur 里读到的还是旧草稿 —— 用标记阻断那一次提交
  const cancelNewRef = useRef(false);
  const cancelEditRef = useRef(false);

  useEffect(() => {
    if (adding) newInputRef.current?.focus();
  }, [adding]);
  useEffect(() => {
    if (editingId) editInputRef.current?.select();
  }, [editingId]);

  const labelCounts = useLabelCounts();

  const labelMap = useMemo(() => new Map(labels.map((l) => [l.id, l])), [labels]);

  const commitNewLabel = () => {
    if (cancelNewRef.current) {
      cancelNewRef.current = false;
      return;
    }
    const name = newLabel.trim();
    if (name) addLabel(name);
    setNewLabel('');
    setAdding(false);
  };

  const cancelNewLabel = () => {
    cancelNewRef.current = true;
    setNewLabel('');
    setAdding(false);
  };

  const commitRename = () => {
    if (cancelEditRef.current) {
      cancelEditRef.current = false;
      return;
    }
    if (editingId) {
      const name = draftName.trim();
      // 改名被拒时一定要说出来，否则用户只会看到输入框自己跳回去，不知道为什么
      if (name && !renameLabel(editingId, name)) {
        toast.error(`已经有叫「${name}」的类别了，换个名字吧`);
      }
    }
    setEditingId(null);
    setDraftName('');
  };

  const cancelRename = () => {
    cancelEditRef.current = true;
    setEditingId(null);
    setDraftName('');
  };

  return (
    // 右栏宽度：212px 下限太窄，「类别名 + 尺寸」会被挤成两行（见 Phase 5 诊断）。
    // 提到 248px 起、上限 320px：标注工具里右栏是高频读数区，
    // 宽度换来的可读性比多让给画布的 36px 更值。
    <aside className="flex w-[clamp(248px,21vw,320px)] flex-shrink-0 flex-col overflow-hidden rounded-xl border border-hairline bg-surface-panel [box-shadow:var(--shadow-panel)]">
      {/* ===== 标签 ===== */}
      <div className="flex shrink-0 flex-col border-b border-hairline">
        {/* hint 只在真的有标签时才给。0 个标签时「单击选用 · 双击改名」
            在描述两个此刻根本不存在的交互，属于空项目噪音。 */}
        <SectionHeader
          title="标签"
          count={labels.length}
          hint={labels.length > 0 ? '单击选用 · 双击改名' : undefined}
        />
        <div className="scroll-thin max-h-[32vh] shrink-0 overflow-y-auto">
          <div className="px-2 pb-2">
            {labels.map((label, labelIndex) => {
              const active = label.id === activeLabelId;
              const hidden = hiddenLabelIds.includes(label.id);
              const count = labelCounts.get(label.id) ?? 0;
              return (
                <div
                  key={label.id}
                  className="group relative mb-[3px] flex items-center gap-2 rounded-[12px] border border-transparent py-[6px] pl-[9px] pr-2 transition-colors duration-150 hover:bg-hover"
                  style={{
                    ...(active
                      ? { background: `${label.color}14`, borderColor: `${label.color}40` }
                      : null),
                    // 隐藏的类别整行淡化，一眼能看出"画布上不显示它"
                    opacity: hidden ? 0.55 : 1,
                  }}
                >
                  {/* 选中时左侧一道色条 */}
                  <span
                    className="absolute left-0 top-1/2 h-[62%] w-[3px] -translate-y-1/2 rounded-r-full transition-opacity duration-150"
                    style={{ background: label.color, opacity: active ? 1 : 0 }}
                  />

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        onClick={() => {
                          const idx = PALETTE.indexOf(label.color as (typeof PALETTE)[number]);
                          setLabelColor(label.id, PALETTE[(idx + 1) % PALETTE.length]);
                        }}
                        title="点击换颜色"
                        className="size-[13px] flex-shrink-0 rounded-[4px] transition-transform duration-150 hover:scale-[1.25]"
                        style={{
                          background: label.color,
                          boxShadow: `0 0 0 3px ${label.color}24`,
                        }}
                      />
                    </TooltipTrigger>
                    <TooltipContent>点击换颜色</TooltipContent>
                  </Tooltip>

                  {editingId === label.id ? (
                    <Input
                      ref={editInputRef}
                      value={draftName}
                      onChange={(e) => setDraftName(e.target.value)}
                      onBlur={commitRename}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') commitRename();
                        if (e.key === 'Escape') {
                          e.preventDefault();
                          cancelRename();
                        }
                      }}
                      className="h-[24px] min-w-0 flex-1 rounded-[9px] px-1.5 text-body"
                      autoFocus
                    />
                  ) : (
                    <button
                      type="button"
                      onClick={() => setActiveLabel(label.id)}
                      onDoubleClick={() => {
                        setEditingId(label.id);
                        setDraftName(label.name);
                      }}
                      title="单击选中，双击改名"
                      className={`min-w-0 flex-1 truncate text-left text-body ${
                        active ? 'font-bold' : 'font-medium text-ink-2'
                      }`}
                      style={active ? { color: readableInk(label.color) } : undefined}
                    >
                      {label.name}
                    </button>
                  )}

                  {labelIndex < 9 ? (
                    <kbd
                      className={`num flex h-[16px] min-w-[16px] flex-shrink-0 items-center justify-center rounded-[6px] border px-[3px] text-micro font-bold ${
                        active
                          ? 'surface-brand border-transparent text-white'
                          : 'border-line bg-muted text-ink-4'
                      }`}
                    >
                      {labelIndex + 1}
                    </kbd>
                  ) : null}

                  {count > 0 ? (
                    <span className="num flex-shrink-0 rounded-full bg-ink/8 px-[6px] py-[1px] text-micro font-bold text-ink-3">
                      {count}
                    </span>
                  ) : (
                    <span className="w-[14px] flex-shrink-0" />
                  )}

                  {/* 显示 / 隐藏这个类别（画布上不画、点不到；导出不受影响） */}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        onClick={() => toggleLabelHidden(label.id)}
                        title={hidden ? '显示这个类别' : '隐藏这个类别'}
                        aria-label={hidden ? '显示这个类别' : '隐藏这个类别'}
                        className={`size-[19px] flex-shrink-0 items-center justify-center rounded-[8px] transition-colors ${
                          hidden
                            ? 'flex bg-primary/12 text-primary'
                            : 'hidden text-ink-4 hover:bg-primary/10 hover:text-primary group-hover:flex'
                        }`}
                      >
                        {hidden ? (
                          <EyeOff className="size-[11px]" />
                        ) : (
                          <Eye className="size-[11px]" />
                        )}
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      {hidden ? '重新显示这个类别' : '隐藏这个类别（画布不显示，导出不受影响）'}
                    </TooltipContent>
                  </Tooltip>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        onClick={() => removeLabel(label.id)}
                        title="删除标签"
                        aria-label="删除标签"
                        className="hidden size-[19px] flex-shrink-0 items-center justify-center rounded-[8px] text-ink-4 transition-colors hover:bg-destructive/12 hover:text-destructive group-hover:flex"
                      >
                        <Trash2 className="size-[11px]" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>删除标签（它的框会退回未分类，不会被删）</TooltipContent>
                  </Tooltip>
                </div>
              );
            })}

            {/* 这里原来还有一段居中两行灰字（「还没有类别 / 直接在图上画个框，就能顺手建一个」）。
                删掉的理由：① 紧跟其下的「新建标签」按钮已经把动作说完了；
                ② 同一条信息在底栏等四处重复（见 LabelBar 那段注释）；
                ③ 0 个标签时它只贡献一段空话，把侧栏往下压。 */}

            {/* Phase 10：这里是「新建标签」入口。原来做成满宽虚线大按钮，
                视觉重量和标签行持平 —— 一条「创建」入口不该和内容一样响。
                改成安静的文字行（无边框、左对齐），只在被找的时候被看见。
                （底部条上那个虚线按钮保留：那是「画图时顺手建一个」的主路径。） */}
            {adding ? (
              <div className="mt-1 px-1 pb-1">
                <Input
                  ref={newInputRef}
                  value={newLabel}
                  onChange={(e) => setNewLabel(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commitNewLabel();
                    if (e.key === 'Escape') {
                      e.preventDefault();
                      cancelNewLabel();
                    }
                  }}
                  onBlur={commitNewLabel}
                  placeholder="输入标签名，回车确认…"
                  className="h-[30px] rounded-[12px] text-body"
                  autoFocus
                />
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setAdding(true)}
                className="mt-1 flex h-[26px] w-full items-center gap-1.5 rounded-[9px] px-2 text-micro font-semibold text-ink-4 transition-colors duration-150 hover:bg-hover hover:text-ink-2"
              >
                <Plus className="size-[12px]" />
                新建标签
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ===== 本图标注 ===== */}
      <div className="flex min-h-0 flex-1 flex-col">
        {/* 本图标注 头：计数 + 多选开关。
            复用 SectionHeader —— 之前这里是手写的一行，内边距和 panel-head 不一致，
            「标签」与「本图标注」两条标题的基线是错开的 */}
        <SectionHeader
          title="本图标注"
          count={image?.annotations.length ?? 0}
          action={
            <button
              type="button"
              onClick={() => {
                setBatchMode((v) => !v);
                if (batchMode) clearSelectBoxes();
              }}
              title="多选模式：勾选多个框后批量改类别"
              className={`flex items-center gap-1 rounded-[9px] border px-2 py-[3px] text-micro font-semibold transition-colors duration-150 ${
                batchMode
                  ? 'border-primary bg-primary/[0.1] text-primary'
                  : 'border-hairline bg-surface-panel text-ink-3 hover:border-primary/40 hover:bg-surface-hover hover:text-primary'
              }`}
            >
              <ListChecks className="size-3.5" />
              多选
            </button>
          }
        />

        {/* 批量提效快捷条 */}
        {image && image.annotations.length > 0 ? (
          <div className="flex items-center gap-1.5 px-3 pb-2">
            <button
              type="button"
              onClick={onCopyPrev}
              title="把上一张图的标注复制过来（适合连续帧）"
              className="flex h-[26px] flex-1 items-center justify-center gap-1 rounded-[9px] text-micro font-semibold text-ink-3 transition-colors duration-150 hover:bg-hover hover:text-ink"
            >
              <Copy className="size-3" />
              复制上一张
            </button>
            <button
              type="button"
              onClick={onApplyActive}
              title="把当前激活类别套用到本图所有未分类框"
              className="flex h-[26px] flex-1 items-center justify-center gap-1 rounded-[9px] text-micro font-semibold text-ink-3 transition-colors duration-150 hover:bg-hover hover:text-ink"
            >
              <CheckSquare className="size-3" />
              套用当前类别
            </button>
          </div>
        ) : null}

        {/* 用原生滚动容器 + scroll-thin，与上方标签区的滚动条保持同一粗细
            （radix ScrollArea 的滚动条约 10px，和标签区的 7px 并排看很明显） */}
        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
          <div className="px-2 pb-3">
            {!image || image.annotations.length === 0 ? (
              <div className="mx-1 mt-5 flex flex-col items-center gap-2.5 rounded-[14px] border border-dashed border-line-strong bg-surface-inset/70 px-4 py-7 text-center">
                {/* 与左栏空状态统一：蓝色会让空状态读成「可点的按钮」，退成中性 */}
                <span className="flex size-[34px] items-center justify-center rounded-[14px] bg-[var(--surface-inset)] text-ink-4">
                  <Tag className="size-[17px]" />
                </span>
                <p className="text-caption leading-relaxed text-muted-foreground">
                  这张图还没有标注
                  <br />
                  在图上按住左键拖一下就有框了
                </p>
                <p className="flex items-center gap-1 text-micro text-ink-4">
                  <MousePointerClick className="size-3" />
                  松手后点一下标签即完成
                </p>
              </div>
            ) : (
              image.annotations.map((ann, index) => {
                const label = ann.labelId ? labelMap.get(ann.labelId) : undefined;
                const isSelected = batchMode
                  ? selectedBoxIds.includes(ann.id)
                  : ann.id === selectedId;
                const overlaps = ann.id === selectedId ? selectedOverlaps : NO_OVERLAPS;
                return (
                  <AnnotationRow
                    key={ann.id}
                    ann={ann}
                    index={index}
                    label={label}
                    selected={isSelected}
                    batchMode={batchMode}
                    overlaps={overlaps}
                    onRequestAddLabel={requestAddLabel}
                  />
                );
              })
            )}
          </div>
        </div>

        {/* 多选模式底部：批量改类别 */}
        {batchMode ? (
          <div className="border-t border-hairline px-3 py-2.5">
            {selectedBoxIds.length > 0 ? (
              <>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-caption text-ink-3">
                    已选 <b className="num text-foreground">{selectedBoxIds.length}</b> 个框
                  </span>
                  <button
                    type="button"
                    onClick={clearSelectBoxes}
                    className="text-micro font-semibold text-ink-4 transition-colors hover:text-destructive"
                  >
                    取消
                  </button>
                </div>
                <div className="flex gap-2">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button size="sm" className="flex-1">
                        选类别套用
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="max-h-[260px] w-44 overflow-y-auto">
                      {labels.length === 0 ? (
                        <div className="px-2 py-1.5 text-micro text-ink-4">还没有类别</div>
                      ) : (
                        labels.map((l) => (
                          <DropdownMenuItem key={l.id} onClick={() => onApplyToSelected(l.id)}>
                            <span
                              className="size-[11px] flex-shrink-0 rounded-[3px]"
                              style={{ background: l.color }}
                            />
                            <span className="truncate">{l.name}</span>
                          </DropdownMenuItem>
                        ))
                      )}
                    </DropdownMenuContent>
                  </DropdownMenu>
                  {activeLabelId ? (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => onApplyToSelected(activeLabelId)}
                    >
                      套用当前
                    </Button>
                  ) : null}
                </div>
              </>
            ) : (
              <p className="text-center text-micro text-ink-4">
                点选多个框，即可批量改类别
              </p>
            )}
          </div>
        ) : (
          /* 引导去画框 */
          image && image.annotations.length > 0 ? (
            <div className="flex items-center gap-1.5 border-t border-hairline px-3 py-2 text-micro text-ink-4">
              <CirclePlus className="size-3 flex-shrink-0" />
              在图上继续拖拽即可追加框
            </div>
          ) : null
        )}
      </div>
    </aside>
  );
}
