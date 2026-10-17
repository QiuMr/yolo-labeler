import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArchiveRestore,
  ChevronLeft,
  ChevronRight,
  CircleCheck,
  Download,
  Eraser,
  FolderX,
  HardDriveDownload,
  Keyboard,
  Moon,
  MoreHorizontal,
  Redo2,
  SkipForward,
  Sparkles,
  Sun,
  Undo2,
} from 'lucide-react';
import { useTheme } from 'next-themes';
import { useStore } from '../store/useStore';
import { toast } from '../store/toastStore';
import { exportProjectBackup, importProjectBackup, type RestoredProject } from '../utils/backup';
import { clearProject } from '../utils/persist';
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
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Progress } from '@/components/ui/progress';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

interface TopBarProps {
  onExport: () => void;
  onShortcuts: () => void;
  onAi: () => void;
}

function Tip({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent side="bottom">{label}</TooltipContent>
    </Tooltip>
  );
}

/**
 * 图标按钮组。
 * Phase 3 改动：旧版每组外面套一圈 `border + bg-card + shadow-xs`，
 * 顶栏里导航组、进度、AI 按钮、撤销组、更多组一共 5 个"带边框的盒子"横排，
 * 视觉噪声极大（截图里那排"不明物体"就是它们）。
 * 现在改成**无容器**：只留 0.5px 的间距，靠按钮自身的 hover 圆角背景表达分组，
 * 边框只保留在真正需要强调「这是一个整体控件」的地方（比如图片导航）。
 */
function ToolGroup({ children }: { children: React.ReactNode }) {
  return <div className="flex items-center gap-0.5">{children}</div>;
}

/* Phase 10：BrandMark 已删除（顶栏品牌块与落地页重复，见下方渲染处的说明）。 */

export function TopBar({ onExport, onShortcuts, onAi }: TopBarProps) {
  const images = useStore((s) => s.images);
  const labels = useStore((s) => s.labels);
  const activeIndex = useStore((s) => s.activeIndex);
  // Phase 11：只订阅「能不能撤销」这个**布尔值**，而不是整个历史数组。
  // 订阅数组的话，每改一个框 / 每次 Ctrl+Z 历史都会换引用 → 整条顶栏
  // （连同里面一堆 Radix Tooltip、Dropdown 子树）全部重渲染，而实际变化只是按钮的禁用态。
  const canUndo = useStore((s) => s.past.length > 0);
  const canRedo = useStore((s) => s.future.length > 0);
  const stepImage = useStore((s) => s.stepImage);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);
  const clearImageBoxes = useStore((s) => s.clearImageBoxes);
  const clearAllBoxes = useStore((s) => s.clearAllBoxes);
  const resetProject = useStore((s) => s.resetProject);
  const replaceProject = useStore((s) => s.replaceProject);

  const onNextUnannotated = () => {
    if (!useStore.getState().gotoNextUnannotated()) {
      toast.info('所有图片都已经标注了');
    }
  };
  const [confirmClose, setConfirmClose] = useState(false);
  const [confirmClearAll, setConfirmClearAll] = useState(false);

  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // ---------- 项目备份 / 还原 ----------
  const [backingUp, setBackingUp] = useState(false);
  const [pendingRestore, setPendingRestore] = useState<RestoredProject | null>(null);
  const restoreInputRef = useRef<HTMLInputElement>(null);

  const onBackup = async () => {
    if (backingUp || images.length === 0) return;
    setBackingUp(true);
    try {
      const d = new Date();
      const p = (n: number) => String(n).padStart(2, '0');
      const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
      await exportProjectBackup(
        images,
        labels,
        useStore.getState().activeLabelId,
        `yolo-labeler-backup-${stamp}.zip`,
      );
      toast.success('项目备份已开始下载');
    } catch (e) {
      toast.error(e instanceof Error ? `备份失败：${e.message}` : '备份失败');
    } finally {
      setBackingUp(false);
    }
  };

  const onPickRestoreFile = (file: File) => {
    toast.info('正在读取备份…');
    void importProjectBackup(file)
      .then((p) => setPendingRestore(p))
      .catch((e) => toast.error(e instanceof Error ? e.message : '恢复失败'));
  };

  const total = images.length;
  /**
   * 进度统计。必须 memo：这段是「图片数 × 框数」的双层遍历，
   * 而 TopBar 同时订阅了 past / future —— 每按一次 Ctrl+Z、每改一个框，
   * 历史数组都会换引用，TopBar 重渲染，这段就会重跑一遍。
   * 8000 张 × 50 框 = 40 万次迭代/次，密集标注时是纯浪费。
   */
  const { annotated, boxes, unlabeled } = useMemo(() => {
    const known = new Set(labels.map((l) => l.id));
    let a = 0;
    let b = 0;
    let u = 0;
    for (const img of images) {
      let hasClassified = false;
      for (const ann of img.annotations) {
        b += 1;
        if (known.has(ann.labelId)) hasClassified = true;
        else u += 1;
      }
      if (hasClassified) a += 1;
    }
    return { annotated: a, boxes: b, unlabeled: u };
  }, [images, labels]);
  const progress = total === 0 ? 0 : Math.round((annotated / total) * 100);
  const done = total > 0 && annotated === total;

  return (
    <header className="relative z-30 flex h-[54px] flex-shrink-0 items-center gap-2.5 overflow-hidden rounded-xl border border-hairline bg-surface-panel px-3.5 [box-shadow:var(--shadow-panel)]">
      {/* 顶边高光 */}
      <div className="brand-rule pointer-events-none absolute inset-x-0 top-0 h-px" />

      {/* Phase 10：去掉左上角品牌块（logo + 「YOLO 标注 / 纯本地标注工作台」）。
          它是营销页该有的东西，不是工作台该有的：占掉顶栏最贵的一段宽度，
          却每时每刻都在重复用户早已知道的信息（他打开了什么工具）。
          Logo 仍在落地页与标签页标题上，不缺这一个。 */}

      {/* 图片导航。这是顶栏里唯一保留「容器边框」的控件 ——
          它是最高频操作（每标一张图都要点），需要和其他按钮拉开层级。 */}
      <div className="flex flex-shrink-0 items-center gap-0.5 rounded-[12px] border border-hairline bg-surface-inset p-[3px]">
        <Tip label="上一张 (←)">
          <Button
            variant="ghost"
            size="icon-xs"
            className="size-[26px] rounded-[9px] hover:bg-surface-panel"
            onClick={() => stepImage(-1)}
            disabled={activeIndex <= 0}
            aria-label="上一张"
          >
            <ChevronLeft />
          </Button>
        </Tip>
        <span className="num min-w-[62px] text-center text-caption font-semibold text-ink-2">
          {total === 0 ? '0 / 0' : `${activeIndex + 1} / ${total}`}
        </span>
        <Tip label="下一张 (→)">
          <Button
            variant="ghost"
            size="icon-xs"
            className="size-[26px] rounded-[9px] hover:bg-surface-panel"
            onClick={() => stepImage(1)}
            disabled={activeIndex >= total - 1}
            aria-label="下一张"
          >
            <ChevronRight />
          </Button>
        </Tip>
        <span className="mx-0.5 h-3.5 w-px bg-hairline" />
        <Tip label="跳到下一张未标注的图 (N)">
          <Button
            variant="ghost"
            size="icon-xs"
            className="size-[26px] rounded-[9px] hover:bg-surface-panel hover:text-primary"
            onClick={onNextUnannotated}
            disabled={total === 0 || done}
            aria-label="跳到下一张未标注的图"
          >
            <SkipForward />
          </Button>
        </Tip>
      </div>

      <div className="h-7 w-px flex-shrink-0 bg-hairline" />

      {/* 进度。用 xl 而非 md：1024 宽的笔记本上顶栏最小需求约 880px，
          进度块再占 230px 就会把「导出」挤出视口（header 有 z-30，会压在画布上）。

          文案上做了一个语义拆分：进度条讲的是「标注完成度」，
          旁边的数字讲的是「框总数」—— 两者含义不同，混在一句里容易被读错。 */}
      <div className="hidden w-[230px] flex-shrink-0 items-center gap-2.5 xl:flex">
        <Progress value={progress} className="h-[6px] flex-1" />
        {done ? (
          <span className="inline-flex flex-shrink-0 items-center gap-1 rounded-full bg-emerald-500/12 px-2 py-[3px] text-micro font-semibold text-emerald-600">
            <CircleCheck className="size-3.5" />
            <span className="num">{boxes}</span> 框 · 全部标完
          </span>
        ) : (
          <span className="num flex-shrink-0 whitespace-nowrap text-caption text-ink-3">
            已标 <span className="font-semibold text-ink-2">{annotated}</span>
            <span className="text-ink-4">/{total}</span> · {boxes} 框
          </span>
        )}
        {unlabeled > 0 ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="num flex-shrink-0 rounded-full bg-amber-400/16 px-2 py-[3px] text-micro font-semibold text-amber-600">
                {unlabeled} 未分类
              </span>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              未分类的框不会被导出，选中框点一下标签即可
            </TooltipContent>
          </Tooltip>
        ) : null}
      </div>

      <div className="min-w-0 flex-1" />

      {/* AI 辅助：本地推理自动找框。
          Phase 12：它原来同时挂 `border` + `bg-card` + `.raised` —— 三重描边感，
          等于自己又变成一个「带边框的盒子」，而这正是本文件顶部注释里
          已判定为「视觉噪声极大」并明确移除的做法。更重要的是它有 36px 的导出按钮
          和它抢主次：两者都用「实体块」语言，顶栏就有两个并列的强元素。
          压平成幽灵按钮后，层级变成 导出(实心) > AI 辅助(有名有姓的次要动作) > 图标按钮。
          字号与内边距保留，所以它依旧是个「看得见的动作」，只是不再装成一块面板。 */}
      <Tip label="AI 自动找框（本地推理，图片不上传）">
        <button
          type="button"
          onClick={onAi}
          className="flex h-[32px] flex-shrink-0 items-center gap-1.5 rounded-[12px] px-2.5 text-caption font-semibold text-ink-2 transition-colors duration-fast ease-smooth hover:bg-surface-hover hover:text-foreground active:scale-[0.97]"
        >
          <Sparkles className="size-[14px]" />
          AI 辅助
        </button>
      </Tip>

      <Tip label={mounted && resolvedTheme === 'dark' ? '切换到浅色' : '切换到深色'}>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => setTheme(mounted && resolvedTheme === 'dark' ? 'light' : 'dark')}
        >
          {mounted && resolvedTheme === 'dark' ? <Sun /> : <Moon />}
        </Button>
      </Tip>

      <ToolGroup>
        <Tip label="撤销 (Ctrl+Z)">
          <Button variant="ghost" size="icon-sm" onClick={undo} disabled={!canUndo}>
            <Undo2 />
          </Button>
        </Tip>
        <Tip label="重做 (Ctrl+Shift+Z)">
          <Button variant="ghost" size="icon-sm" onClick={redo} disabled={!canRedo}>
            <Redo2 />
          </Button>
        </Tip>
      </ToolGroup>

      <ToolGroup>
        <Tip label="快捷键 (?)">
          <Button variant="ghost" size="icon-sm" onClick={onShortcuts}>
            <Keyboard />
          </Button>
        </Tip>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon-sm" aria-label="更多操作">
              <MoreHorizontal />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuItem onClick={clearImageBoxes}>
              <Eraser />
              清空本图标注
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => setConfirmClearAll(true)}>
              <Eraser />
              清空全部标注
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={onBackup} disabled={backingUp || images.length === 0}>
              <HardDriveDownload />
              {backingUp ? '正在备份…' : '备份项目'}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => restoreInputRef.current?.click()}>
              <ArchiveRestore />
              从备份恢复…
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem variant="destructive" onSelect={() => setConfirmClose(true)}>
              <FolderX />
              关闭项目…
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </ToolGroup>

      <AlertDialog open={confirmClearAll} onOpenChange={setConfirmClearAll}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>清空全部标注？</AlertDialogTitle>
            <AlertDialogDescription>
              会删掉所有图片上的标注框。撤销时会按 Ctrl+Z 逐张恢复（有标注的图片有几张就按几次）。
              如果只是想去掉某一页，用这里的「清空本图标注」更安全，一步就能撤回来。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-white hover:bg-destructive/90"
              onClick={() => {
                clearAllBoxes();
                setConfirmClearAll(false);
              }}
            >
              确认清空
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* 从备份恢复：确认会替换当前项目 */}
      <AlertDialog
        open={pendingRestore !== null}
        onOpenChange={(v) => {
          if (!v) setPendingRestore(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>用备份替换当前项目？</AlertDialogTitle>
            <AlertDialogDescription>
              备份里有 <b className="num">{pendingRestore?.images.length ?? 0}</b> 张图片、
              <b className="num">{pendingRestore?.labels.length ?? 0}</b> 个类别。
              {pendingRestore && pendingRestore.missingImages > 0 ? (
                <>
                  {' '}
                  其中 <b className="num">{pendingRestore.missingImages}</b> 张的图片数据缺失（标注仍在，
                  恢复后重新导入原图即可接回）。
                </>
              ) : null}
              <br />
              当前正在编辑的项目会被覆盖，此操作无法撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (pendingRestore) {
                  replaceProject(pendingRestore);
                  toast.success('备份已恢复');
                }
                setPendingRestore(null);
              }}
            >
              确认恢复
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* 恢复备份的隐藏文件选择 */}
      <input
        ref={restoreInputRef}
        type="file"
        accept=".zip,application/zip"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          // 清掉 value，保证同一次能再次选择同一个文件
          e.target.value = '';
          if (f) onPickRestoreFile(f);
        }}
      />

      <AlertDialog open={confirmClose} onOpenChange={setConfirmClose}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>关闭项目？</AlertDialogTitle>
            <AlertDialogDescription>
              会清除浏览器里保存的所有图片和标注数据。此操作无法撤销，建议先导出备份。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-white hover:bg-destructive/90"
              onClick={() => {
                void clearProject();
                resetProject();
              }}
            >
              确认关闭
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* 「导出」是顶栏唯一的主动作，给足视觉重量。
          Phase 12 重做：原来直接套用通用 Button 的 default 变体
          （.surface-brand 斜向渐变 + --shadow-brand 小而硬的彩色投影），
          结果和顶栏其余控件「白底 + 发丝线 + 大扩散柔影」不是一套语言，
          看着像贴上去的一块蓝。现在改用专用样式 .btn-export：
          垂直渐变 + 1px 深蓝描边 + 柔影，靠**材质与阴影语言一致**来显得协调，
          只把色相留给强调 —— 强调靠唯一性，不靠浮夸。

          动效一律走 CSS（不引 framer-motion）：
            · hover  抬升 2px + 四周**散开**的柔光（不是硬边环 —— 硬环是焦点态的语法）
                     + 表面受光变亮 + 一道斜切扫光（延迟 200ms：鼠标只是路过不会触发）
            · 按下   从落点扩散一道**浅色**涟漪（蓝涟漪压在蓝底上等于没做），
                     按钮落回原位并缩小 3%、受光收掉、内阴影翻成「上暗影」——
                     位移读「落」、缩放读「实」，只做位移会像在往下滑
            · 图标   只在按下时向下沉 1px，呼应「导出 = 往下走」；
                     hover 不给图标位移（按钮整体已在动，两层动读起来像在抖）
          用普通 <button> 而非 Button 组件，是为了不被 buttonVariants 的
          base 类串干扰 —— 那套 base 里的 text-body 曾经把白字类挤掉过（见 lib/utils.ts）。 */}
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={onExport}
            onPointerDown={(e) => {
              // 记录落点，让涟漪从手指/鼠标按下的位置扩散
              const r = e.currentTarget.getBoundingClientRect();
              e.currentTarget.style.setProperty('--rx', `${e.clientX - r.left}px`);
              e.currentTarget.style.setProperty('--ry', `${e.clientY - r.top}px`);
              // 重启动画：先摘掉类并强制回流，否则连点只响第一次
              const ripple = e.currentTarget.querySelector('.btn-ripple');
              if (ripple) {
                ripple.classList.remove('run');
                void (ripple as HTMLElement).offsetWidth;
                ripple.classList.add('run');
              }
            }}
            className="btn-export relative ml-1 inline-flex h-[36px] flex-shrink-0 items-center gap-[7px] px-[18px]"
            style={{ fontSize: 13, fontWeight: 600 }}
            aria-label="导出数据集 (Ctrl+S)"
          >
            <span className="btn-ripple" aria-hidden />
            <span className="btn-sheen" aria-hidden />
            <Download className="relative size-[15px]" />
            <span className="relative">导出</span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          导出数据集 <span className="opacity-70">(Ctrl+S)</span>
        </TooltipContent>
      </Tooltip>
    </header>
  );
}
