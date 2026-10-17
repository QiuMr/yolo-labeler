import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { Loader2, TriangleAlert } from 'lucide-react';
import { canvasApi } from '../canvas/canvasApi';
import { useStore } from '../store/useStore';
import { useAiStore } from '../store/aiStore';
import { useSegStore } from '../store/segStore';
import { describeImport, reportAddImages, toast } from '../store/toastStore';
import { extractDataset } from '../utils/files';
import { parseYoloDataset } from '../utils/yoloDataset';
import { persistProject } from '../utils/persist';
import { CanvasStage } from './CanvasStage';
import { ImageSidebar } from './ImageSidebar';
import { LabelBar } from './LabelBar';
import { RightPanel } from './RightPanel';
import { TopBar } from './TopBar';

// 三个弹窗按需加载：不打开就不下载对应代码块（jszip / AI 面板等），
// 打开时才挂载，装完的模块会留在内存里，之后秒开。
const AiDialog = lazy(() => import('./AiDialog').then((m) => ({ default: m.AiDialog })));
const ExportDialog = lazy(() => import('./ExportDialog').then((m) => ({ default: m.ExportDialog })));
const ShortcutsDialog = lazy(() =>
  import('./ShortcutsDialog').then((m) => ({ default: m.ShortcutsDialog })),
);

function DialogMount({
  open,
  children,
}: {
  open: boolean;
  children: React.ReactNode;
}) {
  if (!open) return null;
  return <Suspense fallback={null}>{children}</Suspense>;
}

export function Editor() {
  const addImages = useStore((s) => s.addImages);
  const importYoloDataset = useStore((s) => s.importYoloDataset);
  const [exportOpen, setExportOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [aiOpen, setAiOpen] = useState(false);
  const [dropActive, setDropActive] = useState(false);
  /** 正在解析「已标注数据集」（要读 txt、量尺寸，比普通导入慢） */
  const [importing, setImporting] = useState(false);
  const dropDepth = useRef(0);
  const modalRef = useRef(false);

  // 数据只存在本地浏览器，明确提醒用户：导出才是唯一备份手段
  const [backupTip, setBackupTip] = useState<boolean>(() => {
    try {
      return localStorage.getItem('yl-backup-tip') !== '1';
    } catch {
      return true;
    }
  });
  const dismissBackup = () => {
    try {
      localStorage.setItem('yl-backup-tip', '1');
    } catch {
      /* 忽略 */
    }
    setBackupTip(false);
  };

  useEffect(() => {
    modalRef.current = exportOpen || shortcutsOpen || aiOpen;
  }, [exportOpen, shortcutsOpen, aiOpen]);

  // ---------- 快捷键 ----------
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (modalRef.current) return;
      // 确认框 / 下拉菜单打开时，方向键、空格这些要留给它们，不能顺手切了图片
      if (document.querySelector('[role="dialog"],[role="alertdialog"],[role="menu"]')) return;
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
      ) {
        return;
      }
      // 点击标注模式：Esc 直接退出。
      // 必须先于下面的 menuOpen 判断 —— 进入这个模式就是要能「一句话全身而退」，
      // 否则用户点了几下想退出，却不知道该再点哪里。
      if (e.key === 'Escape' && useSegStore.getState().active) {
        e.preventDefault();
        useSegStore.getState().close();
        return;
      }

      // 画布上正开着框的标签菜单时，数字键 / Delete 这些要交给它，Esc 由它自己处理
      if (e.key !== 'Escape' && canvasApi.current?.menuOpen()) return;

      const s = useStore.getState();
      const mod = e.ctrlKey || e.metaKey;

      if (mod) {
        const key = e.key.toLowerCase();
        if (key === 'z') {
          e.preventDefault();
          if (e.shiftKey) s.redo();
          else s.undo();
          return;
        }
        if (key === 'y') {
          e.preventDefault();
          s.redo();
          return;
        }
        if (key === 's') {
          e.preventDefault();
          setExportOpen(true);
          return;
        }
        if (key === 'd') {
          e.preventDefault();
          if (s.selectedId) s.duplicateBox(s.selectedId);
          return;
        }
        return;
      }

      if (e.altKey) return;

      // Tab / Shift+Tab：在框之间循环选中（不依赖鼠标，也方便触屏板外的键盘流）
      //
      // 只在「焦点不在可交互控件上」时才接管。
      // 以前无条件 preventDefault，导致界面里所有按钮（顶栏、右侧列表行、侧栏卡片）
      // 都无法用键盘到达 —— 纯键盘用户实际不可用。
      // 规则：焦点在 body / 画布 canvas 上才接管；落在 button / a / [tabindex] 上时交回浏览器。
      const active = document.activeElement;
      const onWidget =
        !!active &&
        active !== document.body &&
        active.tagName !== 'CANVAS' &&
        (active.tagName === 'BUTTON' || active.tagName === 'A' || active.hasAttribute('tabindex'));
      if (e.key === 'Tab') {
        if (onWidget) return;
        e.preventDefault();
        const st = useStore.getState();
        const img = st.images[st.activeIndex];
        if (!img || img.annotations.length === 0) return;
        if (st.selectedId) {
          const idx = img.annotations.findIndex((a) => a.id === st.selectedId);
          const nextIdx = e.shiftKey ? idx - 1 : idx + 1;
          const wrapped =
            ((nextIdx % img.annotations.length) + img.annotations.length) %
            img.annotations.length;
          st.select(img.annotations[wrapped].id);
        } else {
          st.select(
            e.shiftKey
              ? img.annotations[img.annotations.length - 1].id
              : img.annotations[0].id,
          );
        }
        return;
      }

      switch (e.key) {
        case 'Delete':
        case 'Backspace':
          if (s.selectedId) {
            e.preventDefault();
            s.deleteBox(s.selectedId);
          }
          return;
        case 'Escape':
          s.select(null);
          return;
        case 'ArrowLeft':
          e.preventDefault();
          s.stepImage(-1);
          return;
        case 'ArrowRight':
          e.preventDefault();
          s.stepImage(1);
          return;
        case ' ':
          e.preventDefault();
          if (!e.repeat) canvasApi.current?.setSpace(true);
          return;
        case '+':
        case '=':
          canvasApi.current?.zoomBy(1.25);
          return;
        case '-':
        case '_':
          canvasApi.current?.zoomBy(1 / 1.25);
          return;
        default:
          break;
      }

      const key = e.key.toLowerCase();
      if (key === 'f') {
        canvasApi.current?.fit();
        return;
      }
      if (key === '?' || (key === '/' && e.shiftKey)) {
        setShortcutsOpen(true);
        return;
      }
      if (key === 'a') {
        s.stepImage(-1);
        return;
      }
      if (key === 'd') {
        s.stepImage(1);
        return;
      }
      if (key === 'n') {
        e.preventDefault();
        if (!s.gotoNextUnannotated()) toast.info('所有图片都已经标注了');
        return;
      }
      if (/^[1-9]$/.test(e.key)) {
        const label = s.labels[Number(e.key) - 1];
        if (label) {
          s.setActiveLabel(label.id);
          if (s.selectedId) s.setBoxLabel(s.selectedId, label.id);
        }
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === ' ') canvasApi.current?.setSpace(false);
    };
    // 切走窗口时 keyup 可能收不到，兜底解除平移状态
    const onBlur = () => canvasApi.current?.setSpace(false);

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    };
  }, []);

  // ---------- 自动保存 ----------
  useEffect(() => {
    let timer = 0;
    // 保存失败的提示不能跟着每次保存刷屏：同类错误 30 秒内只提醒一次
    let lastWarnAt = 0;
    let lastWarnKey = '';

    const save = async () => {
      const s = useStore.getState();
      const res = await persistProject(s.images, s.labels, s.activeLabelId);
      if (res.ok) return;
      const key = res.reason ?? 'error';
      const now = Date.now();
      if (key === lastWarnKey && now - lastWarnAt < 30000) return;
      lastWarnAt = now;
      lastWarnKey = key;
      if (key === 'quota') {
        toast.error('浏览器存储空间已满，最近的改动可能没保存成功。请立刻「导出」或「备份项目」，并考虑移除一些图片再继续。');
      } else if (key === 'unavailable') {
        toast.error('当前浏览器不支持本地存储，改动在你关掉页面后不会保留。请尽快导出备份。');
      } else {
        toast.error('自动保存失败，最近的改动可能丢失。');
      }
    };

    const schedule = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => void save(), 800);
    };
    // 只关心真正需要落盘的数据，悬停/选中这类 UI 状态不触发写入
    const unsub = useStore.subscribe((state, prev) => {
      if (state.images === prev.images && state.labels === prev.labels && state.activeLabelId === prev.activeLabelId) {
        return;
      }
      schedule();
    });
    const onHide = () => {
      if (document.visibilityState === 'hidden') {
        window.clearTimeout(timer);
        void save();
      }
    };
    document.addEventListener('visibilitychange', onHide);
    return () => {
      unsub();
      window.clearTimeout(timer);
      document.removeEventListener('visibilitychange', onHide);
      void save();
    };
  }, []);

  // ---------- 后台预热 AI ----------
  // 若等用户点开「AI 辅助」才初始化，要干等引擎加载 + 模型下载 + 会话编译。
  // 进编辑器后空闲时就把这套流程跑起来，等真正打开面板时通常已经就绪（秒开）。
  useEffect(() => {
    const schedule = (cb: () => void): (() => void) => {
      if (typeof window.requestIdleCallback === 'function') {
        const id = window.requestIdleCallback(cb, { timeout: 3000 });
        return () => window.cancelIdleCallback(id);
      }
      const t = window.setTimeout(cb, 1500);
      return () => window.clearTimeout(t);
    };
    return schedule(() => void useAiStore.getState().init());
  }, []);

  // ---------- 拖入追加图片 / 已标注数据集 ----------
  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    dropDepth.current = 0;
    setDropActive(false);

    // 同时收图片与 labels/*.txt、classes.txt、data.yaml —— 以前只收图片，
    // 拖入一个已标注好的数据集时框会被静默丢掉
    const { images, texts } = await extractDataset(e.dataTransfer);
    if (images.length === 0) {
      // 以前直接 return：拖错文件（zip/pdf/单个 txt）时界面零反应，用户以为程序卡了。
      // window 上有全局 preventDefault(drop)，不会退回浏览器默认的「打开文件」，所以必须自己说话。
      if (texts.length > 0) {
        toast.info('只找到标注文件、没找到图片。请把包含 images/ 的整个数据集文件夹拖进来。');
      } else {
        toast.error('没有找到可用图片，支持 JPG / PNG / WEBP / BMP');
      }
      return;
    }

    // 只有真的像标注文件时才走「数据集导入」通道。
    // 以前「含任意 txt/yaml」就走，拖一张图 + 一个 notes.txt 会得到
    // 「只发现图片、没有可用标注」这种让人困惑的提示。
    const hasDataset = texts.some((t) => {
      const n = t.file.name.toLowerCase();
      if (n === 'classes.txt' || n.endsWith('.names') || n.endsWith('.yaml') || n.endsWith('.yml')) {
        return true;
      }
      if (!n.endsWith('.txt')) return false;
      const key = t.file.name.replace(/\.[^.]+$/, '').toLowerCase();
      // readme / notes / data 这类不是逐图标注
      return !['readme', 'notes', 'data', 'classes'].includes(key);
    });

    if (!hasDataset) {
      reportAddImages(addImages(images));
      return;
    }

    setImporting(true);
    try {
      const parsed = await parseYoloDataset(texts);
      const res = await importYoloDataset({
        files: images,
        classNames: parsed.classNames,
        anns: parsed.anns,
        ambiguous: parsed.ambiguous,
      });
      const parts = describeImport(res);
      if (res.boxes > 0) toast.success(`已识别 YOLO 数据集：${parts}`);
      else if (res.added > 0) toast.info(`只发现图片、没有可用标注：${parts}`);
      else toast.error(parts || '没有可导入的内容');
    } catch (err) {
      toast.error(err instanceof Error ? `导入数据集失败：${err.message}` : '导入数据集失败');
    } finally {
      setImporting(false);
    }
  };

  return (
    // 浮动面板布局（Phase 7）：外层留 8px 边距 + 面板之间 8px 间隙，
    // 每个面板各自是一个圆角卡片浮在 app 底色上。
    // 旧版是四个面板全部贴边、只靠 1px 单边线区分 —— 那是「工程工具」的排法；
    // Figma / Linear / Arc 这类现代工具的标志恰恰是**面板与画布之间有底色透出来**。
    <div
      className="relative flex h-screen w-screen flex-col gap-2 overflow-hidden bg-surface-app p-2"
      onDragEnter={(e) => {
        e.preventDefault();
        dropDepth.current += 1;
        setDropActive(true);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={(e) => {
        e.preventDefault();
        dropDepth.current -= 1;
        if (dropDepth.current <= 0) {
          dropDepth.current = 0;
          setDropActive(false);
        }
      }}
      onDrop={handleDrop}
    >
      <TopBar
        onExport={() => setExportOpen(true)}
        onShortcuts={() => setShortcutsOpen(true)}
        onAi={() => setAiOpen(true)}
      />

      {backupTip ? (
        <div className="flex items-center gap-2 rounded-xl border border-[var(--hairline)] bg-[var(--brand-soft)] px-4 py-[7px] text-caption text-primary">
          <TriangleAlert className="size-3.5 flex-shrink-0" />
          <span className="flex-1">
            数据只保存在你这台电脑的浏览器里，「关闭项目」会清空一切。标完后记得点右上角「导出」下载备份。
          </span>
          <button
            type="button"
            onClick={dismissBackup}
            className="flex-shrink-0 rounded-[9px] px-2 py-[3px] text-micro font-bold text-primary transition-colors hover:bg-primary/10"
          >
            知道了
          </button>
        </div>
      ) : null}

      <div className="flex min-h-0 flex-1 gap-2">
        <ImageSidebar />
        <CanvasStage />
        <RightPanel />
      </div>

      <LabelBar />

      {importing ? (
        <div
          className="pointer-events-none absolute inset-0 z-[95] flex items-center justify-center"
          style={{ background: 'rgba(14,18,32,0.34)', backdropFilter: 'blur(2px)' }}
        >
          <div
            className="surface-brand flex items-center gap-3 rounded-[24px] px-7 py-5 text-white [box-shadow:inset_0_1px_0_rgba(255,255,255,0.2),var(--shadow-brand-lg)]"
          >
            <Loader2 className="size-5 animate-spin" />
            <div className="text-body font-bold">
              正在读取数据集里的已有标注…
              <div className="mt-0.5 text-micro font-medium opacity-85">
                含 classes.txt / data.yaml / labels 的 YOLO 数据集会连框一起导入
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {dropActive ? (
        <div
          className="pointer-events-none absolute inset-0 z-[90] flex items-center justify-center"
          style={{ background: 'rgba(14,18,32,0.34)', backdropFilter: 'blur(2px)' }}
        >
          <div className="absolute inset-4 rounded-[22px] border-2 border-dashed border-white/45" />
          <div
            className="anim-pop surface-brand flex items-center gap-3 rounded-[24px] px-7 py-5 text-white [box-shadow:inset_0_1px_0_rgba(255,255,255,0.2),var(--shadow-brand-lg)]"
          >
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
              <rect x="2.5" y="2.5" width="8" height="8" rx="2" stroke="currentColor" strokeWidth="2" />
              <rect x="13.5" y="2.5" width="8" height="8" rx="2" stroke="currentColor" strokeWidth="2" opacity="0.55" />
              <rect x="2.5" y="13.5" width="8" height="8" rx="2" stroke="currentColor" strokeWidth="2" opacity="0.55" />
              <rect x="13.5" y="13.5" width="8" height="8" rx="2" fill="currentColor" />
            </svg>
            <div className="flex flex-col leading-tight">
              <span className="text-title font-bold tracking-tight">松手即可添加图片</span>
              <span className="text-caption font-medium opacity-75">
                支持单张、多选，或整个文件夹
              </span>
            </div>
          </div>
        </div>
      ) : null}

      <DialogMount open={exportOpen}>
        <ExportDialog open onClose={() => setExportOpen(false)} />
      </DialogMount>
      <DialogMount open={shortcutsOpen}>
        <ShortcutsDialog open onClose={() => setShortcutsOpen(false)} />
      </DialogMount>
      <DialogMount open={aiOpen}>
        <AiDialog open onClose={() => setAiOpen(false)} />
      </DialogMount>
    </div>
  );
}
