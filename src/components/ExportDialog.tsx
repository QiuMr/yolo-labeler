import { useMemo, useState } from 'react';
import { Braces, Check, FileCode, FileText, FolderOpen, PackageOpen, TriangleAlert } from 'lucide-react';
import { useStore } from '../store/useStore';
import { toast } from '../store/toastStore';
import type { ExportFormat, ExportOptions } from '../types';
import { exportZip } from '../utils/exporters';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';

const FORMATS: Array<{
  key: ExportFormat;
  label: string;
  ext: string;
  desc: string;
  icon: typeof FileText;
}> = [
  { key: 'yolo', label: 'YOLO', ext: '.txt', desc: '每张图一个 txt，最常用', icon: FileText },
  { key: 'coco', label: 'COCO', ext: '.json', desc: '单个 json 汇总', icon: Braces },
  { key: 'voc', label: 'VOC', ext: '.xml', desc: '每张图一个 xml', icon: FileCode },
];

const SPLITS: Array<{ label: string; value: number }> = [
  { label: '不划分', value: 0 },
  { label: '9 : 1', value: 0.1 },
  { label: '8 : 2', value: 0.2 },
  { label: '7 : 3', value: 0.3 },
];

function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

interface ExportDialogProps {
  open: boolean;
  onClose: () => void;
}

function OptionRow({
  checked,
  onCheckedChange,
  title,
  desc,
  disabled,
}: {
  checked: boolean;
  onCheckedChange: (v: boolean) => void;
  title: string;
  desc: string;
  disabled?: boolean;
}) {
  return (
    <div
      className={`flex items-center justify-between gap-3 px-3.5 py-2.5 transition-colors ${
        disabled ? 'opacity-40' : 'hover:bg-hover'
      }`}
    >
      <Label className="cursor-pointer flex-col items-start! gap-0.5">
        <span className="text-body font-semibold text-foreground">{title}</span>
        <span className="text-micro font-normal leading-snug text-muted-foreground">{desc}</span>
      </Label>
      <Switch checked={checked} disabled={disabled} onCheckedChange={onCheckedChange} />
    </div>
  );
}

function Metric({ value, label, warn }: { value: number | string; label: string; warn?: boolean }) {
  // Phase 10：原来每条都是「描边 + 底色 + 阴影」的小卡片，5 个等重的盒子横排，
  // 头部于是变成又一条「边框带」，和下面的格式卡抢注意力。
  // 统计数字不需要容器 —— 数字够大、标签够小、上下留白一压，层级自明。
  // （warn 仍用琥珀色文字保留警示语义，只是不再铺一块琥珀底。）
  return (
    <div className="flex flex-1 flex-col items-center px-1 py-0.5">
      <span
        className={`num text-body font-bold leading-none ${
          warn ? 'text-amber-600 dark:text-amber-300' : 'text-foreground'
        }`}
      >
        {value}
      </span>
      <span
        className={`mt-1 text-micro leading-none ${warn ? 'text-amber-500 dark:text-amber-400/80' : 'text-ink-4'}`}
      >
        {label}
      </span>
    </div>
  );
}

export function ExportDialog({ open, onClose }: ExportDialogProps) {
  const images = useStore((s) => s.images);
  const labels = useStore((s) => s.labels);

  const [format, setFormat] = useState<ExportFormat>('yolo');
  const [splitRatio, setSplitRatio] = useState(0);
  const [withYaml, setWithYaml] = useState(true);
  const [includeImages, setIncludeImages] = useState(false);
  const [exportEmpty, setExportEmpty] = useState(false);
  const [busy, setBusy] = useState(false);
  /** 导出进行中的阶段进度（measure/烘焙/打包共用一根条） */
  const [progress, setProgress] = useState<{ ratio: number; label: string } | null>(null);

  const stats = useMemo(() => {
    const known = new Set(labels.map((l) => l.id));
    let boxes = 0;
    let annotated = 0;
    let unlabeled = 0;
    let emptyLabelImages = 0;
    // 尺寸还没读出来的图（没加载到过）导出时会被跳过，计数要跟着一致
    let usable = 0;
    let usableWithFile = 0;
    let annotatedUsable = 0;
    let annotatedUsableWithFile = 0;
    for (const img of images) {
      const measurable = img.width > 0 && img.height > 0;
      const hasFile = measurable && !!img.file;
      if (measurable) usable += 1;
      if (hasFile) usableWithFile += 1;
      if (img.annotations.length > 0) {
        annotated += 1;
        if (measurable) annotatedUsable += 1;
        if (hasFile) annotatedUsableWithFile += 1;
      }
      boxes += img.annotations.length;
      let exportable = 0;
      for (const a of img.annotations) {
        if (known.has(a.labelId)) exportable += 1;
        else unlabeled += 1;
      }
      // 有框但一个都没分类 → 会导出成空 txt，被训练端当成负样本
      if (img.annotations.length > 0 && exportable === 0) emptyLabelImages += 1;
    }
    return {
      boxes,
      annotated,
      unlabeled,
      emptyLabelImages,
      total: images.length,
      usable,
      usableWithFile,
      annotatedUsable,
      annotatedUsableWithFile,
    };
  }, [images, labels]);

  const options: ExportOptions = {
    format,
    splitRatio,
    withYaml: format === 'yolo' && withYaml,
    includeImages,
    exportEmpty,
  };

  const targets = exportEmpty ? stats.usable : stats.annotatedUsable;
  const imageTargets = exportEmpty ? stats.usableWithFile : stats.annotatedUsableWithFile;
  const extraImages = includeImages ? imageTargets : 0;
  const fileCount =
    format === 'coco'
      ? 2 + extraImages
      : targets + extraImages + 1 + (options.withYaml ? 1 : 0);

  const structure = useMemo(() => {
    const lines: string[] = [];
    if (format === 'yolo') {
      lines.push(splitRatio > 0 ? 'train/labels/*.txt' : 'labels/*.txt');
      if (splitRatio > 0) lines.push('val/labels/*.txt');
      if (options.withYaml) lines.push('data.yaml');
      lines.push('classes.txt');
    } else if (format === 'voc') {
      lines.push(splitRatio > 0 ? 'train/*.xml' : '*.xml', 'classes.txt');
    } else {
      lines.push('_annotations.coco.json', 'classes.txt');
    }
    if (includeImages) {
      lines.push(splitRatio > 0 ? 'train/images/*, val/images/*' : 'images/*');
    }
    return lines;
  }, [format, splitRatio, includeImages, options.withYaml]);

  const handleExport = async () => {
    if (stats.boxes === 0) {
      toast.error('还没有画任何标注框');
      return;
    }
    if (stats.unlabeled === stats.boxes) {
      toast.error('所有框都还没有类别，导出来会是空数据集 —— 先给框选个类别');
      return;
    }
    setBusy(true);
    setProgress({ ratio: 0, label: '正在测量图片尺寸…' });
    try {
      // 导出前补齐未打开图片的尺寸：之前没翻看过的图尺寸是 0，会被当成空图跳过。
      // 这里实测一遍（限并发），保证「一次都没打开过」的图也能正常进数据集。
      await useStore.getState().ensureImageSizes((done, total) => {
        const t = Math.max(1, total);
        setProgress({
          ratio: (done / t) * 0.2,
          label: `正在测量图片尺寸 ${done}/${total}…`,
        });
      });

      const fresh = useStore.getState().images;
      let usableTargets = 0;
      for (const img of fresh) {
        const measurable = img.width > 0 && img.height > 0;
        if (!measurable) continue;
        if (exportEmpty || img.annotations.length > 0) usableTargets += 1;
      }
      if (usableTargets === 0) {
        toast.error('没有可导出的图片，先在编辑器里打开一下这些图');
        return;
      }

      const summary = await exportZip(
        fresh,
        labels,
        options,
        `yolo-dataset-${stamp()}.zip`,
        (p) => setProgress({ ratio: 0.2 + p.ratio * 0.8, label: p.label }),
      );

      if (summary.rotated > 0) {
        toast.success(
          `已开始下载 zip（已自动摆正 ${summary.rotated} 张带方向的照片）`,
        );
      } else {
        toast.success('已开始下载 zip');
      }
      if (summary.failedBake > 0) {
        toast.warning(`${summary.failedBake} 张带方向的照片没能自动摆正（已按原图导出），标注坐标可能需要留意`);
      }
      onClose();
    } catch (err) {
      toast.error(err instanceof Error ? `导出失败：${err.message}` : '导出失败');
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        if (!v && !busy) onClose();
      }}
    >
      <DialogContent
        className="flex max-h-[88vh] w-full max-w-[calc(100vw-2rem)] flex-col gap-0 overflow-hidden rounded-[22px] border-border p-0 [box-shadow:var(--shadow-xl)] sm:max-w-[548px]"
        showCloseButton={!busy}
      >
        <DialogHeader className="gap-0 px-6 pt-5 pb-4">
          <div className="flex items-center gap-2.5">
            <span className="surface-brand flex size-[30px] items-center justify-center rounded-[12px] text-white [box-shadow:var(--shadow-brand)]">
              <PackageOpen className="size-4" />
            </span>
            <div className="flex min-w-0 flex-col gap-0.5">
              <DialogTitle className="text-title font-bold tracking-tight">导出数据集</DialogTitle>
              <DialogDescription className="text-caption">
                打包成 zip，下载后直接拿去训练
              </DialogDescription>
            </div>
          </div>

          <div className="mt-3.5 flex items-stretch gap-2">
            <Metric value={stats.total} label="张图片" />
            <Metric value={stats.annotated} label="已标注" />
            <Metric value={stats.boxes} label="个框" />
            <Metric value={labels.length} label="个类别" />
            <Metric value={stats.unlabeled} label="未分类" warn={stats.unlabeled > 0} />
          </div>
        </DialogHeader>

        {/* Phase 10：去掉 body 的 bg-muted/25。纯白主题下 25% 的浅灰既看不出来、
            又让弹窗出现一道说不清的色差；分隔交给上面那条 border-t 就够了。 */}
        <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto border-t border-border px-6 py-4">
          {stats.unlabeled > 0 ? (
            <div className="flex items-start gap-2.5 rounded-[16px] border border-amber-300 bg-amber-50 px-3.5 py-2.5 text-caption leading-snug text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">
              <TriangleAlert className="mt-[1px] size-[14px] flex-shrink-0 text-amber-500 dark:text-amber-400" />
              <div>
                有 <b className="num">{stats.unlabeled}</b> 个框还没有类别，
                <span className="font-semibold">它们不会出现在导出的数据集里</span>。
                {stats.emptyLabelImages > 0 ? (
                  <>
                    {' '}
                    其中 <b className="num">{stats.emptyLabelImages}</b> 张图画了框但全没分类，
                    会导出成<span className="font-semibold">内容为空的标注文件</span>
                    ——训练时会被当成你有意标注的负样本，不是报错。
                  </>
                ) : null}
                <div className="mt-1 opacity-80">想保留就在图上选中框、点一下标签给它定类别。</div>
              </div>
            </div>
          ) : null}

          {/* 格式 */}
          <div className="flex flex-col gap-2">
            <span className="section-label">导出格式</span>
            <div className="grid grid-cols-3 gap-2">
              {FORMATS.map((f) => {
                const active = format === f.key;
                return (
                  <button
                    key={f.key}
                    type="button"
                    onClick={() => setFormat(f.key)}
                    className={`relative flex flex-col gap-1 rounded-[16px] border-2 p-3 text-left transition-all duration-150 ${
                      active
                        ? 'border-primary bg-card [box-shadow:var(--shadow-brand)]'
                        : 'border-border bg-card [box-shadow:var(--shadow-xs)] hover:border-input hover:[box-shadow:var(--shadow-md)]'
                    }`}
                  >
                    {active ? (
                      <span className="surface-brand absolute right-2 top-2 flex size-[16px] items-center justify-center rounded-full text-white">
                        <Check className="size-[10px]" strokeWidth={3.5} />
                      </span>
                    ) : null}
                    <f.icon
                      className={`size-[18px] ${active ? 'text-primary' : 'text-ink-4'}`}
                      strokeWidth={2}
                    />
                    <div className="flex items-baseline gap-1.5">
                      <span
                        className={`text-body font-bold ${active ? 'text-primary' : 'text-foreground'}`}
                      >
                        {f.label}
                      </span>
                      <span className="num text-micro font-semibold text-ink-4">{f.ext}</span>
                    </div>
                    <div className="text-micro leading-snug text-muted-foreground">{f.desc}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* 数据集划分 */}
          <div className="flex flex-col gap-2">
            <div className="flex items-baseline justify-between">
              <span className="section-label">训练集 / 验证集</span>
              <span className="text-micro text-ink-4">按文件名划分，重新导入同一批图结果也不变</span>
            </div>
            <div className="flex gap-2">
              {SPLITS.map((s) => {
                const active = Math.abs(splitRatio - s.value) < 1e-6;
                return (
                  <button
                    key={s.label}
                    type="button"
                    onClick={() => setSplitRatio(s.value)}
                    className={`num flex-1 rounded-[12px] border-2 py-2 text-body font-bold transition-all duration-150 ${
                      active
                        ? 'border-primary bg-primary/[0.07] text-primary'
                        : 'border-border bg-card text-ink-3 [box-shadow:var(--shadow-xs)] hover:border-input hover:bg-hover'
                    }`}
                  >
                    {s.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* 选项 */}
          <div className="flex flex-col gap-2">
            <span className="section-label">打包选项</span>
            <div className="flex flex-col divide-y divide-border overflow-hidden rounded-[16px] border border-border bg-card [box-shadow:var(--shadow-sm)]">
              <OptionRow
                checked={withYaml}
                onCheckedChange={setWithYaml}
                disabled={format !== 'yolo'}
                title="生成 data.yaml"
                desc="ultralytics / YOLOv5 训练直接可用"
              />
              <OptionRow
                checked={includeImages}
                onCheckedChange={setIncludeImages}
                title="把图片一起打包"
                desc="zip 会比较大，适合直接拿到服务器上训练"
              />
              <OptionRow
                checked={exportEmpty}
                onCheckedChange={setExportEmpty}
                title="为没有标注的图片也生成文件"
                desc="YOLO 训练通常需要负样本对应的空 txt"
              />
            </div>
          </div>

          {/* 结构预览 */}
          <div className="flex flex-col gap-2">
            <span className="section-label">ZIP 内部结构</span>
            <div className="overflow-hidden rounded-[16px] border border-border bg-card [box-shadow:var(--shadow-sm)]">
              <div className="flex items-center gap-1.5 border-b border-border bg-muted/50 px-3 py-1.5 text-micro font-semibold text-ink-3">
                <FolderOpen className="size-3" />
                yolo-dataset-….zip
              </div>
              <div className="font-mono text-micro leading-[1.75] text-ink-2">
                {structure.map((line) => (
                  <div key={line} className="flex items-center gap-2 px-3 py-[1px]">
                    <span className="text-ink-4">└</span>
                    <span className="truncate">{line}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <DialogFooter className="flex flex-col gap-2.5 border-t border-border bg-card px-6 py-3.5">
          {busy && progress ? (
            <div className="flex items-center gap-2.5">
              <Progress value={Math.round(progress.ratio * 100)} className="h-[6px] flex-1" />
              <span className="num w-[180px] flex-shrink-0 truncate text-right text-micro text-ink-3">
                {progress.label}
              </span>
            </div>
          ) : null}
          <div className="flex items-center justify-end gap-2.5">
            <Button variant="outline" onClick={onClose} disabled={busy}>
              取消
            </Button>
            <Button onClick={handleExport} disabled={busy} className="gap-1.5 font-semibold">
              <PackageOpen />
              {busy ? '导出中…' : `导出 ZIP（${fileCount} 个文件）`}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
