import { useCallback, useEffect, useRef, useState } from 'react';
import { FolderOpen, Images, Lock, MousePointerClick, PackageOpen, ShieldCheck, Zap } from 'lucide-react';
import { useStore } from '../store/useStore';
import { describeImport, reportAddImages, toast } from '../store/toastStore';
import { extractDataset } from '../utils/files';
import { parseYoloDataset } from '../utils/yoloDataset';
import { Button } from '@/components/ui/button';

/**
 * 特性只保留三条，且**不配卡片、不配边框**。
 * 旧版是 3 个「带边框 + 阴影 + hover 上浮」的彩色图标卡片，
 * 这种「每个卖点都要一个盒子」的排法是 2020-2022 年国产 SaaS 落地页的固定套路，
 * 也是这页显得不上时代的首要原因。现代做法是大留白 + 弱化容器 + 靠排版分层。
 *
 * Phase 11：这里仍**不加卡片、不改成 1→2→3 时间线、不换成 01/02/03 大号数字**。
 * 时间线的问题是把副标题里已经说过的「三步就好」再说一遍；
 * 大号数字的问题更实在 —— 这三条是具体能力（免安装、拖一下成框、三种导出格式），
 * 换成装饰性序号反而把信息丢了。原来的观感问题只在于图标 19px 且 /70 太淡、
 * 整块沉在底部没收口；现在用 brand-soft 圆角底托把图标立起来，
 * 上方加一条极淡分隔线给三栏收口，就够了。
 */
const FEATURES = [
  {
    icon: Zap,
    title: '拖进来就能用',
    desc: '不用安装、不用配置、不用注册账号',
  },
  {
    icon: MousePointerClick,
    title: '拖一下就是一个框',
    desc: '松手后选类别，手速就是上限',
  },
  {
    icon: PackageOpen,
    title: '一键导出训练集',
    desc: 'YOLO / COCO / VOC 三种格式，打包 zip 直接训练',
  },
];

function BrandMark({ size = 40 }: { size?: number }) {
  return (
    <div
      className="surface-brand flex items-center justify-center rounded-[14px] text-white"
      style={{ width: size, height: size, boxShadow: 'var(--shadow-brand)' }}
    >
      <svg width={size * 0.48} height={size * 0.48} viewBox="0 0 24 24" fill="none">
        <rect x="2.5" y="2.5" width="8" height="8" rx="2" stroke="currentColor" strokeWidth="2" />
        <rect
          x="13.5"
          y="2.5"
          width="8"
          height="8"
          rx="2"
          stroke="currentColor"
          strokeWidth="2"
          opacity="0.55"
        />
        <rect
          x="2.5"
          y="13.5"
          width="8"
          height="8"
          rx="2"
          stroke="currentColor"
          strokeWidth="2"
          opacity="0.55"
        />
        <rect x="13.5" y="13.5" width="8" height="8" rx="2" fill="currentColor" />
      </svg>
    </div>
  );
}

export function LandingPage() {
  const addImages = useStore((s) => s.addImages);
  const importYoloDataset = useStore((s) => s.importYoloDataset);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const dragDepth = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  // 文件夹选择需要非标准属性，直接设置更可靠
  useEffect(() => {
    const el = folderInputRef.current;
    if (!el) return;
    el.setAttribute('webkitdirectory', '');
    el.setAttribute('directory', '');
  }, []);

  const handleFiles = useCallback(
    (files: File[]) => {
      if (files.length === 0) return;
      setBusy(true);
      window.setTimeout(() => {
        const result = addImages(files);
        setBusy(false);
        if (result.added === 0 && result.skipped === 0) {
          toast.error('没有找到可用图片，支持 JPG / PNG / WEBP / BMP');
          return;
        }
        reportAddImages(result);
      }, 30);
    },
    [addImages],
  );

  /**
   * 落地页也要能导入「已标注好的 YOLO 数据集」。
   * 以前这里走 extractFiles（只收图片），把 labels/*.txt、classes.txt 全部丢掉，
   * 于是新用户第一次拖入数据集 = 图片进来、已有标注全丢，而且界面上看不出来。
   */
  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);

      const { images, texts } = await extractDataset(e.dataTransfer);
      if (images.length === 0) {
        if (texts.length > 0) {
          toast.info('只找到标注文件、没找到图片。请把包含 images/ 的整个数据集文件夹拖进来。');
        } else {
          toast.error('没有找到可用图片，支持 JPG / PNG / WEBP / BMP');
        }
        return;
      }

      const hasDataset = texts.length > 0;
      if (!hasDataset) {
        handleFiles(images);
        return;
      }

      setBusy(true);
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
        setBusy(false);
      }
    },
    [handleFiles, importYoloDataset],
  );

  return (
    <div
      className="relative flex min-h-screen w-full items-center justify-center overflow-hidden px-6 py-16"
      style={{ background: 'var(--hero-glow), var(--background)' }}
      onDragEnter={(e) => {
        e.preventDefault();
        dragDepth.current += 1;
        setDragging(true);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={(e) => {
        e.preventDefault();
        dragDepth.current -= 1;
        if (dragDepth.current <= 0) {
          dragDepth.current = 0;
          setDragging(false);
        }
      }}
      onDrop={handleDrop}
    >
      {/* 刻意只在画布中央留一层极淡的方格：给「这里可以拖东西」一个物理直觉，
          删掉了旧版三团 17s/21s 循环漂移的光斑 —— 那种持续动的东西会让人分心，
          也是「廉价感」的主要来源。 */}
      <div className="dot-grid pointer-events-none absolute inset-0 opacity-60" />

      {/* Phase 11：噪点层。极低透明度的分形噪声压在背景上，
          目的是消掉纯色大面积产生的「数字感」——
          屏幕上的绝对平坦在人眼看来是「空」，加一层几乎看不见的颗粒就变成「有材质」。
          用内联 SVG 的 feTurbulence 生成而不是引图片（零额外请求、零体积），
          颗粒度与混合模式见 .noise-layer / .dark .noise-layer。
          只放在落地页：编辑器的画布区是深色 + 网格，本身就够厚，不需要再压噪点。 */}
      <div className="noise-layer pointer-events-none absolute inset-0" aria-hidden />

      <div className="relative flex w-full max-w-[760px] flex-col items-center">
        {/* 标题区：字要够大、留白要够多。现代感很大程度来自「敢空着」。
            Phase 11：第二行由**纯色**改为**两段渐变（蓝→青）**。
            这里推翻的是上一轮「纯色更克制」的判断 —— 两行标题里一行黑一行纯蓝，
            色差只体现在色相上、明度几乎齐平，读起来是「两行同样重的字」，
            而渐变的**明度是流动的**，视线会自然从起笔滑到收尾，这是纯色给不了的。
            仍坚持两点：只用两段（三段在大字上会花）、仍然只作用在第二行。 */}
        <div className="anim-rise flex flex-col items-center text-center">
          <BrandMark />

          <h1 className="mt-7 text-[clamp(35px,6vw,57px)] font-bold leading-[1.1] tracking-[-0.035em] text-foreground">
            拖进图片
            <br />
            <span className="title-gradient">立刻开始标注</span>
          </h1>

          {/* 文案配合角标一起改：原文「图片和标注都存在你自己的浏览器里，不上传任何服务器」
              和卡片角标「100% 本地运行」是同一个意思说了两遍。
              把「不上传」这个最强信任钩子留在副标题（它需要这句完整的话），
              角标只负责在卡片上做一个随时可见的徽章。 */}
          <p className="mt-6 max-w-[430px] text-title leading-[1.65] text-muted-foreground">
            不用装、不用注册、不上传。
            <br />
            图片就在你自己的浏览器里，三步标完导出。
          </p>
        </div>

        {/* 拖放区：整页唯一的「容器」。
            旧版是 2px 虚线框 + 内层细框 + 四角括号 + 渐变图标，
            四层装饰叠在一起显得吵。现在就是**一张白卡片**。
            Phase 11：1px 描边 + 顶边高光 + 底边内阴影 + 外投影合成 --shadow-card，
            边框改由阴影的 0 0 0 1px 那层承担（不再单独写 border 类），
            这样描边和投影永远同步、hover 时一起加深，不会出现「线变了影没变」。 */}
        <div
          onClick={() => fileInputRef.current?.click()}
          className={`anim-rise relative mt-12 w-full cursor-pointer rounded-[24px] bg-surface-panel px-8 py-14 transition-all duration-200 ease-smooth ${
            dragging
              ? 'scale-[1.008] [box-shadow:var(--shadow-card-drag)]'
              : '[box-shadow:var(--shadow-card)] hover:-translate-y-0.5 hover:[box-shadow:var(--shadow-card-hover)]'
          }`}
          style={{ animationDelay: '0.08s' }}
        >
          {/* Phase 11 新增：卡片角标。
              「100% 本地运行」是这个产品最硬的卖点（别人的标注工具要上传服务器），
              原来只作为页面底部一行 12px 灰字存在，等于埋掉了。
              角标是**常驻可见**的 —— 不需要滚动、不需要读完整句就能看懂。
              用青（--accent-cyan-*）而不是品牌蓝：蓝在这里会和卡片内的主按钮撞色，
              角标是「补充信息」，青是这轮新引入的辅助色，正好只占这一小块面积。
              颜色走三个独立的 token 而不是复用 --surface-inset / --ink-3：
              它在纯白卡片上必须自己站得住，灰底 + 灰字的组合实测几乎看不见。 */}
          <div
            className="absolute right-4 top-4 flex items-center gap-1.5 rounded-full px-2.5 py-1 text-micro font-medium text-accent-cyan-ink"
            style={{
              background: 'var(--accent-cyan-soft)',
              boxShadow: 'inset 0 0 0 1px var(--accent-cyan-line)',
            }}
          >
            <Lock className="size-3" />
            100% 本地运行
          </div>

          <div className="flex flex-col items-center">
            {/* 图标 + 脉冲光环。
                光环放在图标后面（同尺寸、absolute 居中），
                常态下持续向外扩散 —— 这是「这里可以放东西」的动效暗示，
                比静态虚线框更能把视线拉过来，而且不依赖拖拽态。
                拖拽时叠加 scale-110 + 变实心品牌渐变，反馈再上一层。 */}
            <div className="relative mb-5">
              <span
                className="anim-halo pointer-events-none absolute inset-0 rounded-[18px] bg-halo"
                aria-hidden
              />
              <div
                className={`relative flex size-[56px] items-center justify-center rounded-[18px] transition-all duration-200 ${
                  dragging ? 'scale-110 text-white' : 'surface-brand-soft text-primary'
                }`}
                style={dragging ? { backgroundImage: 'var(--gradient-brand)' } : undefined}
              >
                <Images className="size-[26px]" />
              </div>
            </div>

            <p className="text-title font-semibold tracking-tight text-foreground">
              {busy ? '正在导入…' : dragging ? '松手就导进来' : '把图片或文件夹拖到这里'}
            </p>
            <p className="mt-2 text-caption text-muted-foreground">
              支持 JPG / PNG / WEBP / BMP
            </p>

            {busy ? (
              <div className="relative mt-7 h-[3px] w-[180px] overflow-hidden rounded-full bg-primary/12">
                <div className="surface-brand anim-sweep absolute inset-y-0 w-2/5 rounded-full" />
              </div>
            ) : (
              <div
                className="mt-8 flex flex-wrap items-center justify-center gap-2.5"
                onClick={(e) => e.stopPropagation()}
              >
                <Button
                  size="lg"
                  className="h-11 gap-2 rounded-[14px] px-6 font-semibold"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Images />
                  选择图片
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  className="h-11 gap-2 rounded-[14px] px-6"
                  onClick={() => folderInputRef.current?.click()}
                >
                  <FolderOpen />
                  选择文件夹
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* 信任说明：只留一条。小白最需要知道的是「我的图会不会被传走」。 */}
        <div
          className="anim-rise mt-8 flex items-center gap-1.5 text-caption text-muted-foreground"
          style={{ animationDelay: '0.16s' }}
        >
          <ShieldCheck className="size-[14px] text-emerald-600" />
          图片和标注都存在你自己的浏览器里，不上传任何服务器
        </div>

        {/* 特性：三列纯文字，无卡片无边框，靠大留白和字重分层。
            旧版这里还并列了一排「1 拖入图片 / 2 画框选标签 / 3 导出 ZIP」的步骤卡，
            等于把同一件事说两遍，而且小白根本不需要被教怎么拖 —— 删掉。

            字号特意比正文大一号（title/body 而不是 body/caption）：
            整页留白很大，12px 的小字会被留白吞掉，看起来像「没做完」。

            Phase 11：补一条极淡分隔线给这段「收口」，并给图标一个 brand-soft 圆角底托。
            这是三栏观感问题的真正病灶 —— 卡片里那个图标块已经有「圆角底托」的语言，
            三栏却只有一枚 19px、透明度 70% 的裸图标，立不住也压不住阵；
            加上分隔线后，整块不再是「飘在页面最底部」的一行小字，
            而是一个有明确起止的区块。仍然没有边框、没有卡片、没有 hover 上浮。 */}
        <div className="anim-rise mt-16 w-full" style={{ animationDelay: '0.24s' }}>
          <div className="brand-rule mb-10 h-px w-full" />
          <div className="grid w-full grid-cols-1 gap-10 sm:grid-cols-3">
            {FEATURES.map((f) => (
              <div key={f.title} className="flex flex-col gap-2.5">
                <div className="surface-brand-soft flex size-9 items-center justify-center rounded-[11px] text-primary">
                  <f.icon className="size-[17px]" />
                </div>
                <div className="text-title font-semibold tracking-tight text-foreground">
                  {f.title}
                </div>
                <div className="text-body leading-[1.65] text-muted-foreground">{f.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = '';
          handleFiles(files);
        }}
      />
      <input
        ref={folderInputRef}
        type="file"
        multiple
        hidden
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = '';
          handleFiles(files);
        }}
      />
    </div>
  );
}
