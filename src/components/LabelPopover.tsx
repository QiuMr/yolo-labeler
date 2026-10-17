import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Check, Plus, Trash2 } from 'lucide-react';
import { useStore } from '../store/useStore';
import { readableInk } from '../utils/color';
import type { Label } from '../types';
import { useLabelCounts } from '../hooks/useLabelCounts';

const WIDTH = 216;
const GAP = 10;
const MARGIN = 8;

/** assign = 刚画完框问「这是什么」；edit = 对已有框改标签 */
export type PopoverMode = 'assign' | 'edit';

interface LabelPopoverProps {
  /** 锚点在画布容器内的屏幕坐标（CSS 像素）；就地菜单可传一个点（四个值相同） */
  anchor: { left: number; top: number; right: number; bottom: number };
  /** 画布容器，用于测量可用空间 */
  containerRef: React.RefObject<HTMLDivElement | null>;
  annotationId: string;
  /** 这个框当前挂的类别，决定列表里哪一项打勾；未分类时是空串 */
  currentLabelId: string;
  mode: PopoverMode;
  onDone: () => void;
}

/**
 * 框旁边的标签菜单。两种入口共用一个组件：
 * 画完框就地问「这是什么」，以及双击 / 右键已有框来改标签。
 */
export function LabelPopover({
  anchor,
  containerRef,
  annotationId,
  currentLabelId,
  mode,
  onDone,
}: LabelPopoverProps) {
  const labels = useStore((s) => s.labels);
  const setBoxLabel = useStore((s) => s.setBoxLabel);
  const deleteBox = useStore((s) => s.deleteBox);
  const duplicateBox = useStore((s) => s.duplicateBox);
  const addLabel = useStore((s) => s.addLabel);

  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  // 没有任何标签时，直接展开输入框：第一个类别就地命名，不用多点一次
  const [adding, setAdding] = useState(() => labels.length === 0);
  const [draft, setDraft] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  // 输入框卸载时可能触发 blur，用标记区分「真提交」和「点了取消/关闭」
  const cancelRef = useRef(false);

  const counts = useLabelCounts();

  // 先渲染到屏幕外测出真实高度，再决定放锚点下面还是上面
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const box = containerRef.current?.getBoundingClientRect();
    const boundsW = box?.width ?? 0;
    const boundsH = box?.height ?? 0;
    const height = el.offsetHeight;
    const width = el.offsetWidth;

    let left = anchor.left;
    let top = anchor.bottom + GAP;

    if (top + height > boundsH - MARGIN) {
      // 下方放不下，翻到锚点上方
      const above = anchor.top - GAP - height;
      top = above >= MARGIN ? above : Math.max(MARGIN, boundsH - MARGIN - height);
    }
    left = Math.min(Math.max(MARGIN, left), Math.max(MARGIN, boundsW - width - MARGIN));
    top = Math.max(MARGIN, top);

    setPos({ left, top });
  }, [anchor, containerRef, labels.length, adding]);

  // 首帧是 visibility:hidden（为了先离屏测量尺寸），而隐藏元素拿不到焦点。
  // 所以必须等定位完成、元素真正可见之后再聚焦，否则「松手直接打字」会失效。
  useEffect(() => {
    if (adding && pos) inputRef.current?.focus();
  }, [adding, pos]);

  // 键盘：数字选标签，Enter 确认，Delete 删框，Esc 关闭
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        onDone();
        return;
      }
      if (adding) return;
      if (/^[1-9]$/.test(e.key)) {
        const label = labels[Number(e.key) - 1];
        if (label) {
          e.preventDefault();
          e.stopPropagation();
          setBoxLabel(annotationId, label.id);
          onDone();
        }
        return;
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        e.stopPropagation();
        deleteBox(annotationId);
        onDone();
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        onDone();
      }
    };
    // 捕获阶段，抢在全局快捷键之前
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [adding, annotationId, deleteBox, labels, onDone, setBoxLabel]);

  // 点到侧栏 / 顶栏等别的地方也要收起来，否则弹窗会一直浮在画布上
  useEffect(() => {
    const onDown = (e: PointerEvent) => {
      const el = ref.current;
      if (el && !el.contains(e.target as Node)) onDone();
    };
    document.addEventListener('pointerdown', onDown, true);
    return () => document.removeEventListener('pointerdown', onDown, true);
  }, [onDone]);

  const commitNew = () => {
    if (cancelRef.current) {
      cancelRef.current = false;
      return;
    }
    const name = draft.trim();
    if (name) {
      const id = addLabel(name);
      setBoxLabel(annotationId, id);
      setDraft('');
      setAdding(false);
      onDone();
      return;
    }
    setAdding(false);
  };

  const pick = (label: Label) => {
    setBoxLabel(annotationId, label.id);
    onDone();
  };

  const action = (text: string, title: string, onClick: () => void, danger = false) => {
    const hover = danger
      ? 'hover:border-destructive/45 hover:bg-destructive/10 hover:text-destructive'
      : 'hover:border-primary/45 hover:bg-primary/[0.06] hover:text-primary';
    return (
      <button
        type="button"
        title={title}
        onClick={() => {
          onClick();
          onDone();
        }}
        className={`flex h-[26px] min-w-0 flex-1 items-center justify-center rounded-[10px] border border-line bg-card text-micro font-semibold text-ink-3 transition-colors duration-150 ${hover}`}
      >
        {text}
      </button>
    );
  };

  return (
    <div
      ref={ref}
      className="anim-pop absolute z-30 overflow-hidden rounded-[18px] border border-border bg-popover text-popover-foreground [box-shadow:var(--shadow-xl)]"
      style={{
        left: pos ? pos.left : -9999,
        top: pos ? pos.top : -9999,
        width: WIDTH,
        visibility: pos ? 'visible' : 'hidden',
      }}
      onPointerDown={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
      onContextMenu={(e) => e.preventDefault()}
    >
      {/* 头 */}
      <div className="flex items-center justify-between gap-2 border-b border-border bg-muted/45 px-2.5 py-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="section-label">{mode === 'assign' ? '这是什么' : '换个类别'}</span>
          <span className="num rounded-[6px] border border-line bg-card px-1 text-micro font-bold text-ink-4">
            1~9
          </span>
        </div>
        {mode === 'assign' ? (
          <button
            type="button"
            onClick={() => {
              deleteBox(annotationId);
              onDone();
            }}
            title="删掉这个框"
            className="flex size-[20px] flex-shrink-0 items-center justify-center rounded-[8px] text-ink-4 transition-colors hover:bg-destructive/12 hover:text-destructive"
          >
            <Trash2 className="size-[11px]" />
          </button>
        ) : null}
      </div>

      {/* 标签列表 */}
      <div className="scroll-thin max-h-[248px] overflow-y-auto p-1.5">
        {labels.map((label, index) => {
          const active = label.id === currentLabelId;
          return (
            <button
              key={label.id}
              type="button"
              onClick={() => pick(label)}
              className="mb-[2px] flex w-full items-center gap-2 rounded-[12px] border border-transparent px-2 py-[7px] text-left transition-colors duration-150 hover:border-line hover:bg-muted/70"
              style={{
                background: active ? `${label.color}16` : undefined,
                borderColor: active ? `${label.color}45` : undefined,
              }}
            >
              <span
                className="size-[11px] flex-shrink-0 rounded-[3px]"
                style={{
                  background: label.color,
                  boxShadow: `0 0 0 3px ${label.color}24`,
                }}
              />
              <span
                className={`min-w-0 flex-1 truncate text-body ${
                  active ? 'font-bold' : 'font-medium text-ink-2'
                }`}
                style={active ? { color: readableInk(label.color) } : undefined}
              >
                {label.name}
              </span>
              {counts.get(label.id) ? (
                <span className="num flex-shrink-0 text-micro font-semibold text-ink-4">
                  {counts.get(label.id)}
                </span>
              ) : null}
              {active ? (
                <Check
                  className="size-[13px] flex-shrink-0"
                  strokeWidth={2.75}
                  style={{ color: label.color }}
                />
              ) : index < 9 ? (
                <kbd className="num flex h-[16px] min-w-[16px] flex-shrink-0 items-center justify-center rounded-[6px] border border-line bg-muted px-1 text-micro font-bold text-ink-4">
                  {index + 1}
                </kbd>
              ) : null}
            </button>
          );
        })}

        {labels.length > 0 && !currentLabelId && !adding ? (
          <p className="px-2 py-1 text-micro font-semibold text-ink-4">这个框还没分类</p>
        ) : null}

        {adding ? (
          <div className="px-1 py-1">
            <input
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commitNew}
              onKeyDown={(e) => {
                e.stopPropagation();
                if (e.key === 'Enter') commitNew();
                if (e.key === 'Escape') {
                  // 只收起输入框，整个弹窗的关闭交给外面的 Esc 处理
                  cancelRef.current = true;
                  setDraft('');
                  setAdding(false);
                }
              }}
              placeholder="新标签名，回车确认"
              className="w-full rounded-[12px] border-2 border-primary bg-card px-2 py-1.5 text-body text-foreground outline-none [box-shadow:0_0_0_3px_var(--brand-soft)] placeholder:text-muted-foreground"
            />
          </div>
        ) : (
          <button
            type="button"
            onClick={() => setAdding(true)}
            className="mt-1 flex w-full items-center gap-2 rounded-[12px] border-2 border-dashed border-line-strong px-2 py-[6px] text-left text-caption font-semibold text-ink-3 transition-all duration-150 hover:border-primary/55 hover:bg-primary/[0.05] hover:text-primary"
          >
            <Plus className="size-[13px] flex-shrink-0" />
            新建标签
          </button>
        )}
      </div>

      {/* 脚：已有框给三个动作，新框只说明怎么退出 */}
      {mode === 'edit' ? (
        <div className="flex items-center gap-1.5 border-t border-border bg-muted/45 px-1.5 py-1.5">
          {action('清除', '取消这个框的类别（框保留，导出时跳过）', () =>
            setBoxLabel(annotationId, ''),
          )}
          {action('复制', '复制一份 (Ctrl + D)', () => duplicateBox(annotationId))}
          {action('删除', '删除这个框 (Delete)', () => deleteBox(annotationId), true)}
        </div>
      ) : (
        <div className="flex items-center justify-between gap-2 border-t border-border bg-muted/45 px-2.5 py-1.5 text-micro text-ink-4">
          <span>Enter 沿用当前标签</span>
          <span>Esc 关闭</span>
        </div>
      )}
    </div>
  );
}
