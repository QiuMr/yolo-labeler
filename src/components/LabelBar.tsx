import { useEffect, useRef, useState } from 'react';
import { Plus } from 'lucide-react';
import { useStore } from '../store/useStore';
import { textOn } from '../utils/color';
import { Input } from '@/components/ui/input';
import { useLabelCounts } from '../hooks/useLabelCounts';

export function LabelBar() {
  const labels = useStore((s) => s.labels);
  const activeLabelId = useStore((s) => s.activeLabelId);
  const setActiveLabel = useStore((s) => s.setActiveLabel);
  const addLabel = useStore((s) => s.addLabel);
  const setBoxLabel = useStore((s) => s.setBoxLabel);
  const selectedId = useStore((s) => s.selectedId);

  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  // Esc 取消会立刻卸载输入框，blur 里读到的还是旧草稿 —— 用标记阻断那一次提交
  const cancelRef = useRef(false);

  useEffect(() => {
    if (adding) inputRef.current?.focus();
  }, [adding]);

  const counts = useLabelCounts();

  const commit = () => {
    if (cancelRef.current) {
      cancelRef.current = false;
      return;
    }
    const name = draft.trim();
    if (name) addLabel(name);
    setDraft('');
    setAdding(false);
  };

  const cancel = () => {
    cancelRef.current = true;
    setDraft('');
    setAdding(false);
  };

  const pick = (labelId: string) => {
    setActiveLabel(labelId);
    if (selectedId) setBoxLabel(selectedId, labelId);
  };

  return (
    <div className="relative flex h-[60px] flex-shrink-0 items-center gap-3 overflow-hidden rounded-xl border border-hairline bg-surface-panel px-3.5 [box-shadow:var(--shadow-panel)]">
      {/* 顶边高光 */}
      <div className="brand-rule pointer-events-none absolute inset-x-0 top-0 h-px opacity-60" />

      <div className="flex flex-shrink-0 items-center gap-1.5">
        <span className="section-label">标签</span>
        <span className="count-pill">{labels.length}</span>
      </div>

      <div className="scroll-thin flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto py-2">
        {labels.map((label, index) => {
          const active = label.id === activeLabelId;
          const count = counts.get(label.id) ?? 0;
          const fg = active ? textOn(label.color) : '';
          const onLight = fg === '#ffffff';
          return (
            <button
              key={label.id}
              type="button"
              onClick={() => pick(label.id)}
              className={`group/chip flex flex-shrink-0 items-center gap-[7px] rounded-[12px] border py-[7px] pl-2 pr-2.5 text-body font-semibold transition-[border-color,color,box-shadow,transform] duration-150 ease-smooth active:scale-[0.96] ${
                active
                  ? 'border-transparent [box-shadow:0_3px_12px_rgba(0,0,0,0.16)]'
                  : 'border-line bg-card text-ink-2 hover:-translate-y-[1px] hover:[box-shadow:var(--shadow-md)]'
              }`}
              style={
                active
                  ? { background: label.color, color: fg }
                  : { borderColor: `${label.color}30`, background: `${label.color}0a` }
              }
            >
              {index < 9 ? (
                <kbd
                  className={`num flex h-[18px] min-w-[18px] items-center justify-center rounded-[6px] border px-1 text-micro font-bold ${
                    active ? '' : 'border-line bg-muted text-ink-4'
                  }`}
                  style={
                    active
                      ? {
                          borderColor: onLight ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.14)',
                          background: onLight ? 'rgba(255,255,255,0.22)' : 'rgba(0,0,0,0.07)',
                          color: fg,
                        }
                      : undefined
                  }
                >
                  {index + 1}
                </kbd>
              ) : (
                <span
                  className="size-[9px] flex-shrink-0 rounded-[3px]"
                  style={{
                    background: active ? fg : label.color,
                    boxShadow: active ? 'none' : `0 0 0 3px ${label.color}22`,
                  }}
                />
              )}
              <span className="max-w-[132px] truncate">{label.name}</span>
              {count > 0 ? (
                <span
                  className="num flex h-[17px] min-w-[17px] flex-shrink-0 items-center justify-center rounded-full px-[5px] text-micro font-bold"
                  style={
                    active
                      ? {
                          background: onLight ? 'rgba(255,255,255,0.28)' : 'rgba(0,0,0,0.13)',
                          color: fg,
                        }
                      : {
                          // 走主题变量而不是写死浅色：写死的 rgba(14,18,32,.07) + #6b7385
                          // 在深色模式下等于「深底 + 深字」，计数几乎看不见
                          background: 'color-mix(in srgb, var(--ink) 8%, transparent)',
                          color: 'var(--ink-3)',
                        }
                  }
                >
                  {count}
                </span>
              ) : null}
            </button>
          );
        })}

        {/* 空项目的引导**只在这里出现一次**。
            改之前同一条信息在四处重复：这里、本行右侧、右栏「标签」区、
            右栏「本图标注」空状态 —— 0/168 的窗口里四条同时在屏上。
            而且同一个东西一会儿叫「类别」一会儿叫「标签」，
            而按钮写的是「新标签」、分区标题写的是「标签」，所以统一取「标签」。
            保留这一句的理由：它是唯一贴着「新标签」按钮出现的一句，
            也就是用户真正要动手的那一刻在看的唯一一句。 */}
        {labels.length === 0 && !adding ? (
          <span className="flex-shrink-0 pl-0.5 text-caption text-ink-4">
            还没有标签 —— 在图上拖一个框，松手后就能顺手起个名字
          </span>
        ) : null}

        {adding ? (
          <Input
            ref={inputRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') commit();
              if (e.key === 'Escape') {
                e.preventDefault();
                cancel();
              }
            }}
            placeholder="标签名，回车确认…"
            className="h-[34px] w-[150px] flex-shrink-0 rounded-[12px]"
            autoFocus
          />
        ) : (
          <button
            type="button"
            onClick={() => setAdding(true)}
            className="flex h-[32px] flex-shrink-0 items-center gap-1 rounded-[12px] border-2 border-dashed border-line-strong px-2.5 text-caption font-semibold text-ink-3 transition-all duration-150 ease-smooth hover:border-primary/55 hover:bg-primary/[0.05] hover:text-primary active:scale-[0.97]"
          >
            <Plus className="size-[13px]" />
            新标签
          </button>
        )}
      </div>

      <div className="hidden flex-shrink-0 items-center gap-2 text-caption xl:flex">
        {selectedId ? (
          <>
            <span className="surface-brand-soft rounded-[9px] px-2 py-[3px] text-micro font-bold text-primary">
              已选中 1 个框
            </span>
            <span className="text-muted-foreground">点标签改类别 · 右键框弹菜单</span>
          </>
        ) : labels.length > 0 ? (
          <span className="text-muted-foreground">
            画完框会在旁边弹出标签，也可先按
            <span className="num mx-1 rounded-[6px] border border-line bg-muted px-1 py-[1px] text-micro font-bold text-ink-3">
              1
            </span>
            ~
            <span className="num mx-1 rounded-[6px] border border-line bg-muted px-1 py-[1px] text-micro font-bold text-ink-3">
              9
            </span>
            选好再画
          </span>
        ) : null}
      </div>
    </div>
  );
}
