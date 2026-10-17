import { useEffect } from 'react';
import { Editor } from '@/components/Editor';
import { LandingPage } from '@/components/LandingPage';
import { Toaster } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import { toast } from './store/toastStore';
import { useStore } from './store/useStore';
import { loadProject } from './utils/persist';

function Splash() {
  return (
    <div
      className="relative flex min-h-screen flex-col items-center justify-center gap-4 overflow-hidden bg-background"
    >
      <div className="dot-grid pointer-events-none absolute inset-0" />
      <div
        className="pointer-events-none absolute left-1/2 top-1/2 size-[520px] -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{ background: 'radial-gradient(circle, rgba(37,99,235,0.16), transparent 62%)' }}
      />

      <div className="relative flex flex-col items-center gap-4">
        <div className="surface-brand flex size-[52px] animate-pulse items-center justify-center rounded-[20px] text-white [box-shadow:var(--shadow-brand-lg)]">
          <svg width="25" height="25" viewBox="0 0 24 24" fill="none">
            <rect x="2.5" y="2.5" width="8" height="8" rx="2" stroke="currentColor" strokeWidth="2" />
            <rect x="13.5" y="2.5" width="8" height="8" rx="2" stroke="currentColor" strokeWidth="2" opacity="0.55" />
            <rect x="2.5" y="13.5" width="8" height="8" rx="2" stroke="currentColor" strokeWidth="2" opacity="0.55" />
            <rect x="13.5" y="13.5" width="8" height="8" rx="2" fill="currentColor" />
          </svg>
        </div>
        <div className="flex flex-col items-center gap-2.5">
          <span className="text-caption font-medium text-ink-3">正在恢复上次的项目…</span>
          <div className="relative h-[3px] w-[132px] overflow-hidden rounded-full bg-primary/12">
            <div className="surface-brand anim-sweep absolute inset-y-0 w-2/5 rounded-full" />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const hydrated = useStore((s) => s.hydrated);
  const hasImages = useStore((s) => s.images.length > 0);
  const hydrateProject = useStore((s) => s.hydrateProject);
  const markHydrated = useStore((s) => s.markHydrated);

  useEffect(() => {
    let cancelled = false;
    void loadProject().then((project) => {
      if (cancelled) return;
      if (project) {
        hydrateProject(project);
        if (project.missingImages > 0) {
          // 等一帧再弹，确保 Toaster 已挂载
          window.setTimeout(() => {
            toast.warning(
              `${project.missingImages} 张图的原始数据在浏览器里丢了，标注已保留。重新导入这些图片即可自动接回（框不会丢）。`,
            );
          }, 0);
        }
      } else {
        markHydrated();
      }
    });
    return () => {
      cancelled = true;
    };
  }, [hydrateProject, markHydrated]);

  // 阻止浏览器直接打开拖进来的图片
  useEffect(() => {
    const prevent = (e: DragEvent) => e.preventDefault();
    window.addEventListener('dragover', prevent);
    window.addEventListener('drop', prevent);
    return () => {
      window.removeEventListener('dragover', prevent);
      window.removeEventListener('drop', prevent);
    };
  }, []);

  if (!hydrated) return <Splash />;

  return (
    <TooltipProvider delayDuration={350}>
      {hasImages ? <Editor /> : <LandingPage />}
      <Toaster position="bottom-center" />
    </TooltipProvider>
  );
}
