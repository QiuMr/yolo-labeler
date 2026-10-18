import { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  ChevronRight,
  Cpu,
  FileUp,
  Loader2,
  MousePointerClick,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  X,
  Zap,
} from 'lucide-react';
import { classNameFor } from '../ai/cocoNames';
import { useAiStore } from '../store/aiStore';
import { useSegStore } from '../store/segStore';
import { useStore } from '../store/useStore';
import { toast } from '../store/toastStore';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';

interface AiDialogProps {
  open: boolean;
  onClose: () => void;
}

function formatSize(bytes: number): string {
  if (bytes <= 0) return '';
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

/**
 * AI 辅助标注面板：
 * 模型状态卡 + 置信度 + 「标注当前图 / 一键标注全部」两个动作。
 * 首次打开才引导推理引擎，平时不占任何加载时间。
 */
export function AiDialog({ open, onClose }: AiDialogProps) {
  const status = useAiStore((s) => s.status);
  const error = useAiStore((s) => s.error);
  const modelName = useAiStore((s) => s.modelName);
  const modelSize = useAiStore((s) => s.modelSize);
  const ep = useAiStore((s) => s.ep);
  const inputW = useAiStore((s) => s.inputW);
  const inputH = useAiStore((s) => s.inputH);
  const numClasses = useAiStore((s) => s.numClasses);
  const modelNames = useAiStore((s) => s.modelNames);
  const modelTask = useAiStore((s) => s.modelTask);
  const classNamesOverride = useAiStore((s) => s.classNamesOverride);
  const inputSizeFrom = useAiStore((s) => s.inputSizeFrom);
  const layoutInfo = useAiStore((s) => s.layoutInfo);
  const warmupError = useAiStore((s) => s.warmupError);
  const conf = useAiStore((s) => s.conf);
  const busy = useAiStore((s) => s.busy);
  const batchDone = useAiStore((s) => s.batchDone);
  const batchTotal = useAiStore((s) => s.batchTotal);

  const init = useAiStore((s) => s.init);
  const loadModelFile = useAiStore((s) => s.loadModelFile);
  const setConf = useAiStore((s) => s.setConf);
  const setClassName = useAiStore((s) => s.setClassName);
  const runCurrent = useAiStore((s) => s.runCurrent);
  const runBatch = useAiStore((s) => s.runBatch);
  const cancelBatch = useAiStore((s) => s.cancelBatch);

  const hasImage = useStore((s) => s.images.length > 0);
  const totalImages = useStore((s) => s.images.length);
  const labelRevision = useStore((s) => s.labelRevision);
  /**
   * 未标注图片数。
   *
   * Phase 11：原来写成 `useStore((s) => { 遍历 s.images … })` —— 但**选择器会在每一次
   * store 变更时执行**（zustand 必须靠它算出切片才能判断要不要重渲染），
   * 于是选个框、切个 hover 都要把整个图集数一遍。
   *
   * 改成 useMemo，只依赖「图片数量」和「标签分布版本号」：
   * 这两者涵盖了所有会改变该计数的操作（框增删、改归属、删图、撤销重做、切换项目）。
   * 真要重算时再用 getState() 现读一次，避免订阅整个数组。
   */
  const unannotated = useMemo(() => {
    let n = 0;
    for (const i of useStore.getState().images) if (i.annotations.length === 0) n += 1;
    return n;
  }, [totalImages, labelRevision]);

  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  // 批量范围：empty=只跑没有框的图（默认）；all=全部重跑（对已有框的图追加）
  const [batchScope, setBatchScope] = useState<'empty' | 'all'>('empty');

  /**
   * 「类别名」区块是否展开。默认由**模型有没有自带类名**决定：
   * 有 → 一般不用改，收起；没有（多半是被 extract_model / pnnx 剥掉了元数据）→ 展开，
   * 因为这时候满屏的「类别 1 / 类别 2」正是用户要改的东西，藏起来等于没做。
   * 用户手动折叠后不会被重置（依赖里没有 namesOpen 本身）。
   */
  const [namesOpen, setNamesOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    setNamesOpen(modelNames.length === 0);
  }, [open, modelName, modelNames.length]);

  // 首次打开才引导引擎 + 自动复用上次选过的模型；没选过则引导手动挑一个
  useEffect(() => {
    if (open) void init();
  }, [open, init]);

  const pickFile = () => fileRef.current?.click();

  const handleFile = (file: File | undefined) => {
    if (file) void loadModelFile(file);
  };

  const ready = status === 'ready';
  const working = busy !== 'none';

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        if (v) return;
        // 批量标注进行中：进度条和「停止」按钮都住在这个弹窗里，
        // 直接关掉的话任务还在后台跑，用户既看不到进度也**没法中途停止**。
        if (busy === 'batch') {
          toast.info('批量标注还在后台跑，关掉弹窗就看不到进度、也无法中途停止了。');
          return;
        }
        onClose();
      }}
    >
      {/* 加高上限 + 内部滚动：出错态下内容会变多，矮屏（笔记本 / 平板）不能顶出视口 */}
      <DialogContent className="max-h-[88vh] overflow-y-auto sm:max-w-[460px]">
        <input
          ref={fileRef}
          type="file"
          accept=".onnx"
          className="hidden"
          onChange={(e) => {
            handleFile(e.target.files?.[0]);
            e.target.value = '';
          }}
        />

        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="surface-brand flex size-[38px] flex-shrink-0 items-center justify-center rounded-[16px] text-white [box-shadow:var(--shadow-brand)]">
              <Sparkles className="size-[19px]" />
            </div>
            <div className="min-w-0">
              <DialogTitle className="text-title">AI 辅助标注</DialogTitle>
              <DialogDescription className="mt-0.5 text-caption">
                一次给整张图找出目标，你只负责确认
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        {/* 隐私承诺：本地推理是这个功能的卖点，放在最显眼的位置 */}
        <div className="flex items-center gap-2 rounded-[12px] bg-emerald-500/10 px-3 py-2 text-caption font-semibold text-emerald-700">
          <ShieldCheck className="size-[14px] flex-shrink-0" />
          全程在你的浏览器里本地推理，图片不会上传到任何服务器
        </div>

        {/* ===== 模型状态卡 ===== */}
        {status === 'booting' || status === 'loading' ? (
          <div className="flex items-center gap-3 rounded-[16px] border border-line bg-muted/60 px-4 py-4">
            <Loader2 className="size-[17px] animate-spin text-primary" />
            <div className="min-w-0 text-body">
              <div className="font-bold text-ink-2">
                {status === 'booting' ? '正在准备推理引擎…' : '正在加载模型…'}
              </div>
              <div className="mt-0.5 truncate text-micro text-ink-4">
                {status === 'booting'
                  ? '首次使用需要下载约 2 MB 的运行时'
                  : modelName || ' '}
              </div>
            </div>
          </div>
        ) : status === 'error' ? (
          <div className="rounded-[16px] border border-destructive/25 bg-destructive/5 px-4 py-3.5">
            <div className="flex items-start gap-2.5">
              <AlertTriangle className="mt-[1px] size-[15px] flex-shrink-0 text-destructive" />
              <div className="min-w-0 flex-1">
                <div className="text-body font-bold text-destructive">出了一点问题</div>
                <div className="mt-0.5 break-all text-caption leading-relaxed text-ink-3">
                  {error}
                </div>
              </div>
            </div>
            <div className="mt-3 flex gap-2">
              <Button size="sm" variant="outline" className="flex-1" onClick={() => void init()}>
                <RefreshCw />
                重试
              </Button>
              <Button size="sm" variant="outline" className="flex-1" onClick={pickFile}>
                <FileUp />
                选择模型文件
              </Button>
            </div>
          </div>
        ) : ready ? (
          <div className="rounded-[16px] border border-line bg-card px-4 py-3.5 [box-shadow:var(--shadow-xs)]">
            <div className="flex items-center gap-2">
              <span className="size-[7px] flex-shrink-0 rounded-full bg-emerald-500" />
              <span className="min-w-0 flex-1 truncate text-body font-bold text-foreground">
                {modelName}
              </span>
              <button
                type="button"
                onClick={pickFile}
                className="flex-shrink-0 rounded-[9px] px-2 py-[3px] text-micro font-semibold text-primary transition-colors hover:bg-primary/[0.08]"
              >
                更换
              </button>
            </div>
            <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
              {modelSize > 0 ? (
                <span className="rounded-full bg-muted px-2 py-[3px] text-micro font-bold text-ink-3">
                  {formatSize(modelSize)}
                </span>
              ) : null}
              <span
                className={`flex items-center gap-1 rounded-full px-2 py-[3px] text-micro font-bold ${
                  ep === 'WebGPU'
                    ? 'bg-primary/10 text-primary'
                    : 'bg-amber-400/15 text-amber-600'
                }`}
              >
                {ep === 'WebGPU' ? <Zap className="size-3" /> : <Cpu className="size-3" />}
                {ep === 'WebGPU' ? 'WebGPU 加速' : 'CPU 运算'}
              </span>
              {inputW > 0 ? (
                <span className="num rounded-full bg-muted px-2 py-[3px] text-micro font-bold text-ink-3">
                  输入 {inputW}×{inputH}
                </span>
              ) : null}
              {numClasses > 0 ? (
                <span className="rounded-full bg-muted px-2 py-[3px] text-micro font-bold text-ink-3">
                  {numClasses} 类
                  {modelNames.length > 0
                    ? ' · 模型自带类名'
                    : numClasses === 80
                      ? ' · 按常见 80 类套用'
                      : ''}
                </span>
              ) : null}
            </div>

            {/* 任务类型不对时直接说清楚：这三类模型的输出结构和检测完全不同，
                以前只会得到一句「无法识别的输出布局」，用户并不知道是选错了模型 */}
            {modelTask && modelTask !== 'detect' ? (
              <p className="mt-2.5 rounded-[10px] border border-amber-400/40 bg-amber-400/15 px-2.5 py-2 text-micro leading-relaxed text-amber-600">
                模型元数据里写的任务是「{modelTask}」，不是「detect」。本工具只解析检测输出，
                请改用检测版（task=detect）导出的模型。
              </p>
            ) : null}

            {/* 诊断信息。以前是常驻的一行「输出布局：e2e [1,300,6] × nc=80」——
                对不写代码的人来说是天书，而且平时根本不需要看，
                真正需要它的只有「模型跑了但什么都没有」这一种情况。
                所以收进默认收起的折叠里，只在下面这句话里告诉用户「什么时候来看」。
                真错误（预热失败）仍然常驻，那个是必须马上知道的。 */}
            {warmupError ? (
              <p className="mt-2.5 rounded-[10px] border border-destructive/30 bg-destructive/[0.07] px-2.5 py-2 text-micro leading-relaxed text-destructive">
                模型预热失败：{warmupError}
              </p>
            ) : layoutInfo ? (
              <details className="group mt-2.5">
                <summary className="flex cursor-pointer list-none items-center gap-1 text-micro text-ink-4 transition-colors hover:text-ink-2 [&::-webkit-details-marker]:hidden">
                  <ChevronRight className="size-3 flex-shrink-0 transition-transform group-open:rotate-90" />
                  识别不出来？点开看诊断信息
                </summary>
                <p className="mt-1.5 break-all text-micro leading-relaxed text-ink-4">
                  输出布局：{layoutInfo}
                  {inputSizeFrom ? ` · 输入尺寸来自 ${inputSizeFrom}` : ''}
                </p>
              </details>
            ) : null}
          </div>
        ) : (
          /* idle：引擎就绪但没有模型，给一个大大的拖放区 */
          <button
            type="button"
            onClick={pickFile}
            onDragEnter={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragOver={(e) => e.preventDefault()}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFile(e.dataTransfer.files[0]);
            }}
            className={`flex flex-col items-center gap-1.5 rounded-[16px] border-2 border-dashed px-4 py-6 text-center transition-colors duration-150 ${
              dragOver
                ? 'border-primary bg-primary/[0.06]'
                : 'border-line-strong hover:border-primary/50 hover:bg-primary/[0.04]'
            }`}
          >
            <FileUp className="size-[19px] text-primary" />
            <span className="text-body font-bold text-ink-2">点击或拖入 .onnx 模型</span>
            <span className="text-micro leading-relaxed text-ink-4">
              支持 YOLO26 / YOLOv8 / YOLOv11 / YOLOv5 检测模型
              <br />
              模型里写了类名就会自动用上（常见的 80 类模型能直接认出英文名）
            </span>
          </button>
        )}

        {/* 手上没有 .onnx，是这里最常见的死路：面板只写「拖入模型」，
            可绝大多数人根本没有 onnx 文件，看到这句只会以为
            「AI 功能得自己先准备模型」，然后直接放弃这个功能。
            而画布左上角的「点击标注」用的是**随工具内置**的模型（42MB 随站点发），
            不需要用户提供任何文件 —— 这个事实必须在这里说出来，
            否则一条本来开箱即用的链路会被彻底埋掉。
            按钮直接帮他进入那个模式，省掉「回去找那个小按钮」的一步。 */}
        {status === 'idle' ? (
          <div className="rounded-[16px] border border-line bg-muted/50 px-3.5 py-3">
            <div className="flex items-center gap-1.5 text-caption font-bold text-ink-2">
              <MousePointerClick className="size-[14px] flex-shrink-0 text-primary" />
              手上没有模型文件？
            </div>
            <p className="mt-1 text-micro leading-relaxed text-ink-3">
              画布左上角的「<span className="font-semibold text-ink-2">点击标注</span>
              」用的是随工具内置的模型，不用准备任何文件 —— 点一下物体就能出框。
            </p>
            <Button
              size="sm"
              variant="outline"
              className="mt-2.5 w-full"
              disabled={!hasImage}
              onClick={() => {
                void useSegStore.getState().open();
                onClose();
              }}
            >
              用内置模型标注（不需要文件）
            </Button>
          </div>
        ) : null}

        {/* ===== 类别名 =====
            模型没写类名（被 extract_model / pnnx 剥掉元数据）时，
            「类别 1 / 类别 2」会一路带进标签、带进导出的数据集。
            这里给一个就地改名的入口 —— 这比让用户去准备一个 classes.txt 现实得多：
            他要的是「这个框叫『黄-半』」，不是「先搞清楚 classes.txt 是什么格式」。 */}
        {ready && numClasses > 0 ? (
          <div className="rounded-[16px] border border-line bg-card px-4 py-3.5 [box-shadow:var(--shadow-xs)]">
            <button
              type="button"
              aria-expanded={namesOpen}
              onClick={() => setNamesOpen((v) => !v)}
              className="flex w-full items-center gap-1.5 text-left"
            >
              <ChevronRight
                className={`size-3 flex-shrink-0 text-ink-4 transition-transform ${
                  namesOpen ? 'rotate-90' : ''
                }`}
              />
              <span className="flex-shrink-0 text-caption font-bold text-ink-2">类别名</span>
              <span className="min-w-0 flex-1 truncate text-micro text-ink-4">
                {modelNames.length > 0 ? '模型自带，一般不用改' : '模型没写类名，可以在这里改'}
              </span>
            </button>

            {namesOpen ? (
              <>
                <div className="mt-2.5 max-h-[188px] space-y-1 overflow-y-auto pr-0.5">
                  {Array.from({ length: numClasses }, (_, i) => (
                    <label key={i} className="flex items-center gap-2">
                      <span className="num w-[34px] flex-shrink-0 text-right text-micro text-ink-4">
                        #{i + 1}
                      </span>
                      <Input
                        value={classNamesOverride[i] ?? ''}
                        placeholder={classNameFor(i, numClasses, modelNames)}
                        onChange={(e) => setClassName(i, e.target.value)}
                        className="h-[28px] flex-1 text-caption"
                        spellCheck={false}
                        autoComplete="off"
                        aria-label={`类别 ${i + 1} 的名字`}
                      />
                    </label>
                  ))}
                </div>
                <p className="mt-2 text-micro leading-relaxed text-ink-4">
                  框里灰色的是现在的默认名，直接输入就能改。改完的名字会用在自动生成的标签上；
                  已经标好的框要改名，双击左侧标签改一次，所有框会一起变。
                </p>
              </>
            ) : null}
          </div>
        ) : null}

        {/* ===== 置信度 ===== */}
        <div className="rounded-[16px] border border-line bg-card px-4 py-3.5 [box-shadow:var(--shadow-xs)]">
          <div className="flex items-center justify-between">
            <span className="text-caption font-bold text-ink-2">置信度阈值</span>
            <span className="num rounded-[8px] bg-muted px-1.5 py-[2px] text-micro font-bold text-primary">
              {Math.round(conf * 100)}%
            </span>
          </div>
          <Slider
            className="mt-3"
            value={[Math.round(conf * 100)]}
            min={5}
            max={90}
            step={5}
            onValueChange={(v) => setConf((v[0] ?? 25) / 100)}
            disabled={working}
          />
          <p className="mt-2 text-micro text-ink-4">
            调低能找出更多目标，调高则更精准 · 这个值会跟着当前模型记住，下次不用重调
          </p>
        </div>

        {/* ===== 动作 ===== */}
        {busy === 'batch' ? (
          <div className="rounded-[16px] border border-primary/25 bg-primary/[0.05] px-4 py-3.5">
            <div className="flex items-center justify-between text-caption font-bold">
              <span className="flex items-center gap-1.5 text-primary">
                <Loader2 className="size-[13px] animate-spin" />
                正在批量标注…
              </span>
              <span className="num text-ink-3">
                {batchDone} / {batchTotal} 张
              </span>
            </div>
            <Progress
              value={batchTotal > 0 ? (batchDone / batchTotal) * 100 : 0}
              className="mt-2.5 h-[6px]"
            />
            <Button
              size="sm"
              variant="outline"
              className="mt-3 w-full"
              onClick={cancelBatch}
            >
              <X />
              停止
            </Button>
          </div>
        ) : (
          <div className="grid gap-2">
            <Button
              disabled={!ready || !hasImage}
              onClick={() => {
                void runCurrent();
                onClose();
              }}
            >
              <Sparkles />
              标注当前这张图
            </Button>

            {/* 批量范围：只在两种语义里选，不给多余的开关 */}
            <div className="flex rounded-[14px] border border-border bg-card p-[3px] [box-shadow:var(--shadow-xs)]">
              <button
                type="button"
                onClick={() => setBatchScope('empty')}
                disabled={!ready}
                className={`flex-1 rounded-[10px] px-2 py-[5px] text-caption font-semibold transition-colors disabled:opacity-50 ${
                  batchScope === 'empty'
                    ? 'bg-primary text-white'
                    : 'text-ink-3 hover:text-ink-2'
                }`}
              >
                仅未标注 · {unannotated}
              </button>
              <button
                type="button"
                onClick={() => setBatchScope('all')}
                disabled={!ready}
                className={`flex-1 rounded-[10px] px-2 py-[5px] text-caption font-semibold transition-colors disabled:opacity-50 ${
                  batchScope === 'all'
                    ? 'bg-primary text-white'
                    : 'text-ink-3 hover:text-ink-2'
                }`}
              >
                全部重跑
              </button>
            </div>

            <Button
              variant="outline"
              disabled={
                !ready || (batchScope === 'empty' ? unannotated === 0 : totalImages === 0)
              }
              onClick={() => void runBatch(batchScope)}
            >
              一键标注{batchScope === 'empty' ? '未标注的图片' : '全部图片'}
            </Button>
            <p className="text-center text-micro text-ink-4">
              {batchScope === 'all'
                ? '已有标注的图会补上新框；与已有框重复的会自动跳过 · Ctrl+Z 一次撤销整批'
                : '标注当前图会先给你过目再采纳；批量结果直接落框 · Ctrl+Z 一次撤销整批'}
            </p>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
