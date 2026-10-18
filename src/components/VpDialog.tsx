import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Check,
  ChevronRight,
  Cpu,
  Loader2,
  RefreshCw,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  X,
  Zap,
} from 'lucide-react';
import { useStore } from '../store/useStore';
import { useVpStore, VP_ENGINES, modelMbOf } from '../store/vpStore';
import { percentFromZ, zFromPercent } from '../ai/dinoPrompt';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';

interface VpDialogProps {
  open: boolean;
  onClose: () => void;
}

/**
 * 「照样子找」面板 —— YOLOE 视觉提示。
 *
 * 与「AI 辅助标注」并列的第二条 AI 链路，产品上最大的差别是：
 * **不需要用户准备任何文件**（模型随工具内置），而且**不需要训练**。
 * 用户只要在几张图上画过框，就能让模型按「样子」把剩下的图找出来。
 *
 * 界面刻意把「选参考图」放在最前面而不是塞进设置里：实测决定效果的第一因素
 * 就是**选哪张参考图**（同一批图里 F1 从 0.000 到 0.750 都有），
 * 而「多选几张」几乎没用（1 张 0.704 → 5 张 0.712）。这两件事必须让用户看得见。
 */
export function VpDialog({ open, onClose }: VpDialogProps) {
  const engine = useVpStore((s) => s.engine);
  const setEngine = useVpStore((s) => s.setEngine);
  const status = useVpStore((s) => s.status);
  const error = useVpStore((s) => s.error);
  const progress = useVpStore((s) => s.progress);
  const ep = useVpStore((s) => s.ep);
  const warmupError = useVpStore((s) => s.warmupError);
  const refImageIds = useVpStore((s) => s.refImageIds);
  const labelId = useVpStore((s) => s.labelId);
  const promptReady = useVpStore((s) => s.promptReady);
  const promptRefs = useVpStore((s) => s.promptRefs);
  const promptBoxes = useVpStore((s) => s.promptBoxes);
  const conf = useVpStore((s) => s.conf);
  const busy = useVpStore((s) => s.busy);
  const batchDone = useVpStore((s) => s.batchDone);
  const batchTotal = useVpStore((s) => s.batchTotal);

  const ensureReady = useVpStore((s) => s.ensureReady);
  const initRefs = useVpStore((s) => s.initRefs);
  const toggleRef = useVpStore((s) => s.toggleRef);
  const setLabelId = useVpStore((s) => s.setLabelId);
  const setConf = useVpStore((s) => s.setConf);
  const runTrial = useVpStore((s) => s.runTrial);
  const runBatch = useVpStore((s) => s.runBatch);
  const cancelBatch = useVpStore((s) => s.cancelBatch);

  const labels = useStore((s) => s.labels);
  const totalImages = useStore((s) => s.images.length);
  const labelRevision = useStore((s) => s.labelRevision);
  /** 已有框的图 = 可以当参考的图。用 useMemo 避免每次 store 变更都把整个图集数一遍 */
  const labeled = useMemo(
    () => useStore.getState().images.filter((i) => i.annotations.length > 0),
    [totalImages, labelRevision],
  );

  const [refsOpen, setRefsOpen] = useState(true);
  const [batchScope, setBatchScope] = useState<'empty' | 'all'>('empty');

  // 打开面板**不**下载模型：用户可能只是想看看这是什么。
  // 真正的意图出现在他点「先试标一张 / 照样子标剩下的」那一刻，那时才下（见 vpStore）。
  useEffect(() => {
    if (!open) return;
    initRefs();
  }, [open, initRefs]);

  /** 选中的参考图上出现过的类别 —— 只有这些才有框可用作提示 */
  const labelOptions = useMemo(() => {
    const ids = new Set<string>();
    for (const img of labeled) {
      if (!refImageIds.includes(img.id)) continue;
      for (const a of img.annotations) ids.add(a.labelId);
    }
    return labels.filter((l) => ids.has(l.id));
  }, [labeled, refImageIds, labels]);

  const isDino = engine === 'dino';
  const modelMb = modelMbOf(engine);
  const ready = status === 'ready';
  const loading = status === 'loading';
  const working = busy !== 'none';
  /**
   * 动作按钮该被禁用的**唯一**理由：正在下载模型 / 正在推理。
   *
   * 这里**不能**用 `!ready`：模型是「点了按钮之后」才下载的，
   * 拿 `ready` 当门槛会形成死锁 —— 按钮灰 → 点不了 → 模型永远不下载 → `ready` 永远为 false。
   * 首次打开面板时 status 是 'idle'，按钮必须是**可点**的。
   */
  const blocked = working || loading;
  const refCount = refImageIds.length;
  const refBoxes = labeled
    .filter((i) => refImageIds.includes(i.id))
    .reduce((n, i) => n + i.annotations.length, 0);

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        if (v) return;
        // 批量进行中：进度条和「停止」都在这个弹窗里，关掉就既看不到进度也没法停
        if (busy === 'batch') return;
        onClose();
      }}
    >
      <DialogContent data-vp-panel className="max-h-[88vh] overflow-y-auto sm:max-w-[460px]">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="surface-brand flex size-[38px] flex-shrink-0 items-center justify-center rounded-[16px] text-white [box-shadow:var(--shadow-brand)]">
              <ScanSearch className="size-[19px]" />
            </div>
            <div className="min-w-0">
              <DialogTitle className="text-title">照样子找</DialogTitle>
              <DialogDescription className="mt-0.5 text-caption">
                用你已经标好的框当例子，找出同类的目标
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="flex items-center gap-2 rounded-[12px] bg-emerald-500/10 px-3 py-2 text-caption font-semibold text-emerald-700">
          <ShieldCheck className="size-[14px] flex-shrink-0" />
          不用准备任何文件，也不用训练 · 全程本地推理
        </div>

        {/* ===== 引擎选择 =====
            两条链路的**原理完全不同**（一个是被训练过的检测器、一个是拿特征现算的匹配），
            擅长的场景也不同，所以并列摆出来让用户拿自己的图去比 ——
            而不是替用户选一条、把另一条藏起来。 */}
        <div>
          <div className="flex rounded-[14px] border border-border bg-card p-[3px] [box-shadow:var(--shadow-xs)]">
            {VP_ENGINES.map((e) => {
              const on = engine === e.id;
              return (
                <button
                  key={e.id}
                  type="button"
                  data-vp-engine={e.id}
                  data-vp-engine-on={on ? '1' : '0'}
                  onClick={() => setEngine(e.id)}
                  disabled={working || loading}
                  className={`flex-1 rounded-[10px] px-2 py-[6px] text-caption font-semibold transition-colors disabled:opacity-50 ${
                    on ? 'bg-primary text-white' : 'text-ink-3 hover:text-ink-2'
                  }`}
                >
                  {e.name}
                  <span className={`ml-1 text-micro font-normal ${on ? 'text-white/70' : 'text-ink-4'}`}>
                    {e.mb}MB
                  </span>
                </button>
              );
            })}
          </div>
          <p className="mt-1.5 text-micro leading-relaxed text-ink-4">
            {VP_ENGINES.find((e) => e.id === engine)?.tagline}
            {isDino
              ? ' · 参考图少（1~4 张）时更灵，多标几张反而可能变差'
              : ' · 参考图越多越稳，适合真实照片与复杂场景'}
          </p>
        </div>

        {/* ===== 模型状态 ===== */}
        {status === 'loading' ? (
          <div className="rounded-[16px] border border-line bg-muted/60 px-4 py-4">
            <div className="flex items-center gap-2 text-body font-bold text-ink-2">
              <Loader2 className="size-[15px] animate-spin text-primary" />
              正在准备内置模型…
            </div>
            <Progress
              value={progress < 0 ? 0 : progress}
              className="mt-2.5 h-[6px]"
            />
            <p className="mt-2 text-micro text-ink-4">
              {progress < 0
                ? `正在下载（约 ${modelMb} MB，只有第一次需要）`
                : progress >= 100
                  ? '下载完成，正在建立推理会话…'
                  : `首次使用需要下载约 ${modelMb} MB（${progress}%）`}
            </p>
          </div>
        ) : status === 'error' ? (
          <div className="rounded-[16px] border border-destructive/25 bg-destructive/5 px-4 py-3.5">
            <div className="flex items-start gap-2.5">
              <AlertTriangle className="mt-[1px] size-[15px] flex-shrink-0 text-destructive" />
              <div className="min-w-0 flex-1">
                <div className="text-body font-bold text-destructive">模型没能准备好</div>
                <div className="mt-0.5 break-all text-caption leading-relaxed text-ink-3">
                  {error}
                </div>
              </div>
            </div>
            <Button
              size="sm"
              variant="outline"
              className="mt-3 w-full"
              onClick={() => void ensureReady()}
            >
              <RefreshCw />
              重试
            </Button>
          </div>
        ) : ready ? (
          <div className="flex flex-wrap items-center gap-1.5 rounded-[16px] border border-line bg-card px-4 py-3 [box-shadow:var(--shadow-xs)]">
            <span className="size-[7px] flex-shrink-0 rounded-full bg-emerald-500" />
            <span className="flex-1 text-body font-bold text-foreground">内置模型已就绪</span>
            <span
              className={`flex items-center gap-1 rounded-full px-2 py-[3px] text-micro font-bold ${
                ep === 'WebGPU' ? 'bg-primary/10 text-primary' : 'bg-amber-400/15 text-amber-600'
              }`}
            >
              {ep === 'WebGPU' ? <Zap className="size-3" /> : <Cpu className="size-3" />}
              {ep === 'WebGPU' ? 'WebGPU 加速' : 'CPU 运算'}
            </span>
          </div>
        ) : (
          <div className="rounded-[16px] border border-line bg-muted/50 px-3.5 py-3">
            <p className="text-micro leading-relaxed text-ink-3">
              内置模型会在开始前自动下载（约 {modelMb} MB，只有第一次需要），
              下载一次之后反复可用。两条链路各自缓存，切换回来不用重下。
            </p>
          </div>
        )}

        {warmupError ? (
          <p className="rounded-[10px] border border-destructive/30 bg-destructive/[0.07] px-2.5 py-2 text-micro leading-relaxed text-destructive">
            模型预热失败：{warmupError}
          </p>
        ) : null}

        {/* ===== 参考图 ===== */}
        {labeled.length === 0 ? (
          <div className="rounded-[16px] border border-line bg-muted/50 px-3.5 py-3">
            <div className="text-caption font-bold text-ink-2">还没有可当例子的图</div>
            <p className="mt-1 text-micro leading-relaxed text-ink-3">
              先在几张图上画框（画布左上角的「点击标注」最快），再回来这里 ——
              模型就是照着这些框的样子去找的。
            </p>
          </div>
        ) : (
          <div className="rounded-[16px] border border-line bg-card px-4 py-3.5 [box-shadow:var(--shadow-xs)]">
            <button
              type="button"
              aria-expanded={refsOpen}
              onClick={() => setRefsOpen((v) => !v)}
              className="flex w-full items-center gap-1.5 text-left"
            >
              <ChevronRight
                className={`size-3 flex-shrink-0 text-ink-4 transition-transform ${
                  refsOpen ? 'rotate-90' : ''
                }`}
              />
              <span className="flex-shrink-0 text-caption font-bold text-ink-2">用哪几张图当例子</span>
              <span className="min-w-0 flex-1 truncate text-micro text-ink-4">
                {refCount > 0 ? `已选 ${refCount} 张 · ${refBoxes} 个框` : '还没选'}
              </span>
            </button>

            {refsOpen ? (
              <>
                <div className="mt-2.5 max-h-[168px] space-y-1 overflow-y-auto pr-0.5">
                  {labeled.map((img) => {
                    const on = refImageIds.includes(img.id);
                    return (
                      <button
                        key={img.id}
                        type="button"
                        data-vp-ref={img.id}
                        data-vp-ref-on={on ? '1' : '0'}
                        onClick={() => toggleRef(img.id)}
                        className={`flex w-full items-center gap-2 rounded-[10px] px-2 py-[6px] text-left transition-colors ${
                          on ? 'bg-primary/[0.08]' : 'hover:bg-muted'
                        }`}
                      >
                        <span
                          className={`flex size-[15px] flex-shrink-0 items-center justify-center rounded-[5px] border ${
                            on ? 'border-primary bg-primary text-white' : 'border-line-strong'
                          }`}
                        >
                          {on ? <Check className="size-[11px]" /> : null}
                        </span>
                        <span className="min-w-0 flex-1 truncate text-caption text-ink-2">
                          {img.name}
                        </span>
                        <span className="num flex-shrink-0 text-micro text-ink-4">
                          {img.annotations.length} 框
                        </span>
                      </button>
                    );
                  })}
                </div>
                <p className="mt-2 text-micro leading-relaxed text-ink-4">
                  <span className="font-semibold text-ink-3">1 张就够</span>
                  {isDino
                    ? '，而且标得越多反而可能越差（实测 1 张 F1 0.75、16 张掉到 0.56 —— 例子多了原型就杂）。关键是这张图要和你想标的图'
                    : '，多选几乎没用（实测 1 张和 5 张效果持平）。关键是这张图要和你想标的图'}
                  <span className="font-semibold text-ink-3">长得像</span>
                  —— 换地图、换怪物款式就换一张。
                </p>
              </>
            ) : null}
          </div>
        )}

        {/* ===== 类别 ===== */}
        {labeled.length > 0 ? (
          <div className="rounded-[16px] border border-line bg-card px-4 py-3.5 [box-shadow:var(--shadow-xs)]">
            <div className="flex items-center gap-1.5">
              <span className="text-caption font-bold text-ink-2">找哪个类别</span>
              <span className="text-micro text-ink-4">（用它的框当例子）</span>
            </div>
            {labelOptions.length === 0 ? (
              <p className="mt-2 text-micro leading-relaxed text-ink-4">
                {refCount === 0
                  ? '上面还没勾选参考图 —— 勾一张有框的图当例子。'
                  : '勾选的参考图上没有这个类别的框 —— 换一张，或先在上面画一个。'}
              </p>
            ) : (
              <div className="mt-2.5 flex flex-wrap gap-1.5">
                {labelOptions.map((l) => (
                  <button
                    key={l.id}
                    type="button"
                    data-vp-label={l.id}
                    onClick={() => setLabelId(l.id)}
                    className={`rounded-full px-3 py-[5px] text-caption font-semibold transition-colors ${
                      labelId === l.id
                        ? 'bg-primary text-white'
                        : 'bg-muted text-ink-3 hover:text-ink-2'
                    }`}
                  >
                    {l.name}
                  </button>
                ))}
              </div>
            )}
            <p className="mt-2 text-micro leading-relaxed text-ink-4">
              一次只找一个类别。类别名只是给你自己看的标签 —— 模型学的是「这个框里的东西长什么样」。
            </p>
          </div>
        ) : null}

        {/* ===== 阈值 =====
            两条链路的分数**不是一回事**：YOLOE 是模型输出的置信度（0~1），
            DINOv2 是「高于本图均值几个标准差」。所以控件的范围、显示方式、
            说明文字都跟着引擎换 —— 用一个数字糊弄过去只会让用户调不明白。 */}
        <div className="rounded-[16px] border border-line bg-card px-4 py-3.5 [box-shadow:var(--shadow-xs)]">
          <div className="flex items-center justify-between">
            <span className="text-caption font-bold text-ink-2">
              {isDino ? '匹配严格度' : '置信度阈值'}
            </span>
            <span className="num rounded-[8px] bg-muted px-1.5 py-[2px] text-micro font-bold text-primary">
              {isDino ? conf.toFixed(2) : `${Math.round(conf * 100)}%`}
            </span>
          </div>
          <Slider
            className="mt-3"
            value={[isDino ? Math.round(percentFromZ(conf) * 100) : Math.round(conf * 100)]}
            min={isDino ? 0 : 5}
            max={isDino ? 100 : 90}
            step={5}
            onValueChange={(v) => {
              const raw = v[0] ?? 0;
              setConf(isDino ? zFromPercent(raw / 100) : raw / 100);
            }}
            disabled={working}
          />
          <p className="mt-2 text-micro leading-relaxed text-ink-4">
            {isDino
              ? '数值 = 高于本图平均水平几个标准差 · 实测 2.00 附近最平衡 · 这个值会记住'
              : '调低能找出更多目标，调高则更精准 · 这个值会记住，下次不用重调'}
          </p>
        </div>

        {/* ===== 动作 ===== */}
        {busy === 'batch' ? (
          <div className="rounded-[16px] border border-primary/25 bg-primary/[0.05] px-4 py-3.5">
            <div className="flex items-center justify-between text-caption font-bold">
              <span className="flex items-center gap-1.5 text-primary">
                <Loader2 className="size-[13px] animate-spin" />
                正在照样子找…
              </span>
              <span className="num text-ink-3">
                {batchDone} / {batchTotal} 张
              </span>
            </div>
            <Progress value={batchTotal > 0 ? (batchDone / batchTotal) * 100 : 0} className="mt-2.5 h-[6px]" />
            <Button size="sm" variant="outline" className="mt-3 w-full" onClick={cancelBatch}>
              <X />
              停止
            </Button>
          </div>
        ) : (
          <div className="grid gap-2">
            <Button
              data-vp-trial
              disabled={blocked || labelOptions.length === 0}
              onClick={() => {
                // 等推理真正结束（含首次的模型下载）再关面板：
                // 立刻关掉的话，用户在「下 20MB + 建会话」那 7~8 秒里看不到任何反馈，会以为没反应。
                void runTrial().finally(onClose);
              }}
            >
              {busy === 'current' || busy === 'prompt' || loading ? (
                <Loader2 className="animate-spin" />
              ) : (
                <Sparkles />
              )}
              先试标一张
            </Button>

            <div className="flex rounded-[14px] border border-border bg-card p-[3px] [box-shadow:var(--shadow-xs)]">
              <button
                type="button"
                onClick={() => setBatchScope('empty')}
                disabled={working}
                className={`flex-1 rounded-[10px] px-2 py-[5px] text-caption font-semibold transition-colors disabled:opacity-50 ${
                  batchScope === 'empty' ? 'bg-primary text-white' : 'text-ink-3 hover:text-ink-2'
                }`}
              >
                仅未标注
              </button>
              <button
                type="button"
                onClick={() => setBatchScope('all')}
                disabled={working}
                className={`flex-1 rounded-[10px] px-2 py-[5px] text-caption font-semibold transition-colors disabled:opacity-50 ${
                  batchScope === 'all' ? 'bg-primary text-white' : 'text-ink-3 hover:text-ink-2'
                }`}
              >
                全部重跑
              </button>
            </div>

            <Button
              data-vp-batch
              variant="outline"
              disabled={blocked || labelOptions.length === 0}
              onClick={() => void runBatch(batchScope)}
            >
              照样子标{batchScope === 'empty' ? '剩下的' : '全部图片'}
            </Button>
            <p className="text-center text-micro text-ink-4">
              {promptReady
                ? `例子已就绪（${promptRefs} 张图 · ${promptBoxes} 个框）`
                : '选好参考图和类别即可开始 · 参考图本身不会被重标'}
            </p>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
