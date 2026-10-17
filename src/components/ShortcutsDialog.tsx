import { MousePointerClick, FolderCog, Tags, View } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

const GROUPS: Array<{
  title: string;
  icon: typeof MousePointerClick;
  items: Array<[string, string]>;
}> = [
  {
    title: '画框与分类',
    icon: MousePointerClick,
    items: [
      ['图内拖拽', '画一个新框'],
      ['松手后', '框旁弹出标签，点一下完成'],
      ['1 ~ 9', '在弹窗里直接选标签'],
      ['Enter', '确认当前标签'],
      ['Esc', '关掉弹窗，保留当前标签'],
    ],
  },
  {
    title: '调整框',
    icon: Tags,
    items: [
      ['拖框内', '移动'],
      ['拖八个圆点', '缩放'],
      ['右键框', '弹出菜单：改类别 / 复制 / 删除'],
      ['双击框', '和右键一样弹菜单'],
      ['原地点击', '在重叠的框之间切换'],
      ['Delete', '删除选中的框'],
      ['Ctrl + D', '复制选中的框'],
      ['Tab / Shift+Tab', '在框之间循环选中'],
      ['Shift + 拖拽', '画正方形'],
    ],
  },
  {
    title: '视图',
    icon: View,
    items: [
      ['滚轮', '以光标为中心缩放'],
      ['图外拖动', '平移画布'],
      ['空格 / 右键拖动', '平移画布'],
      ['+ / −', '放大 / 缩小'],
      ['F', '适应窗口'],
    ],
  },
  {
    title: '项目',
    icon: FolderCog,
    items: [
      ['← / →', '上一张 / 下一张'],
      ['N', '跳到下一张未标注的图'],
      ['Ctrl + Z', '撤销'],
      ['Ctrl + Shift + Z', '重做'],
      ['Ctrl + S', '导出'],
      ['?', '打开这个面板'],
    ],
  },
];

function Keys({ text }: { text: string }) {
  return (
    <span className="flex flex-shrink-0 items-center gap-1">
      {text.split(' + ').map((part, i) => (
        <span key={i} className="flex items-center gap-1">
          {i > 0 ? <span className="text-micro font-bold text-ink-4">+</span> : null}
          <kbd className="num inline-flex h-[20px] min-w-[20px] items-center justify-center rounded-[8px] border border-line bg-card px-1.5 text-micro font-bold text-ink-2 shadow-[0_1.5px_0_var(--line-strong)]">
            {part}
          </kbd>
        </span>
      ))}
    </span>
  );
}

export function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      {/* max-h + 内部滚动：4 个分组共 29 行，窄屏单列时内容高约 1150px，
          1024×768 笔记本再叠加浏览器 125% 缩放就会上下被裁掉且滚不动 */}
      <DialogContent className="flex max-h-[88vh] w-full max-w-[calc(100vw-2rem)] flex-col rounded-[24px] border-border p-5 [box-shadow:var(--shadow-xl)] sm:max-w-[620px]">
        <DialogHeader className="gap-0.5">
          <DialogTitle className="text-title font-bold tracking-tight">快捷键</DialogTitle>
          <DialogDescription className="text-caption">
            其实只要会拖拽就够了，这些是给想更快的你的
          </DialogDescription>
        </DialogHeader>

        <div className="grid min-h-0 grid-cols-1 gap-2.5 overflow-y-auto pr-1 sm:grid-cols-2">
          {GROUPS.map((group) => (
            <div
              key={group.title}
              className="rounded-[13px] border border-border bg-card p-3 [box-shadow:var(--shadow-sm)]"
            >
              <div className="mb-2 flex items-center gap-1.5">
                <span className="surface-brand-soft flex size-[19px] items-center justify-center rounded-[8px] text-primary">
                  <group.icon className="size-[11px]" strokeWidth={2.4} />
                </span>
                <span className="section-label">{group.title}</span>
              </div>
              <div className="flex flex-col gap-[7px]">
                {group.items.map(([key, desc]) => (
                  <div key={key} className="flex items-baseline justify-between gap-2.5">
                    <Keys text={key} />
                    <span className="text-right text-caption leading-[20px] text-muted-foreground">
                      {desc}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
