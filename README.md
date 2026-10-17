# YOLO 标注 · 拖入即标

一个纯前端的 YOLO 目标检测标注工具。没有后端、没有账号、不用配置，**拖进图片就能开始标**。
所有图片和标注都保存在你自己的浏览器里，不上传任何服务器。

## 三步上手

1. **拖入图片** —— 可以拖单张、多选，也可以直接把整个文件夹拖进来
2. **画框** —— 在图上按住左键拖一下就有一个框，松手自动带上当前标签
3. **导出** —— 点右上角「导出」，选好格式，一键下载 zip

就这些。没有「新建项目」，没有「保存」按钮——所有改动都是自动保存的，刷新页面也能接着标。

## 功能

**标注**
- 拖拽画框，松手即完成，自动带上当前标签
- 单击选中，拖框内移动，拖八个角点缩放
- 双击框 / 双击标签名 / 点标签条：快速改类别
- 滚轮以光标为中心缩放，空格或右键拖动平移
- 靠近图片边缘自动吸附
- 按住 Shift 画正方形
- 100 步撤销 / 重做

**标签**
- 不再有自动创建的默认标签：一个类别都没有时，画完框会直接让你输入名字，第一个类别就地建好
- 允许一个标签都没有；未分类的框在画布上画成灰色虚线框
- 按 `1`~`9` 直接切换标签
- 点色块换颜色，双击改名
- 删除标签只把它的框退回「未分类」，不会销毁已经画好的标注
- 未分类的框不会被导出（三种格式都需要类别）；导出面板会显示数量并给出提示

**数据**
- 自动保存到浏览器 IndexedDB，刷新 / 关掉再打开都还在
- 缩略图侧栏，带序号和标注数量，几千张图也不卡
- 顶部实时显示标注进度

**导出**
- YOLO（`.txt` + `classes.txt` + `data.yaml`）
- COCO（`_annotations.coco.json`）
- VOC（`.xml`）
- 可选 train / val 划分，按文件名稳定分配，每次结果一致
- 可选把图片一起打包
- 可选为没有标注的图片生成空文件（负样本）

## 快捷键

| 操作 | 快捷键 |
| --- | --- |
| 画框 | 在图上拖拽 |
| 移动 / 缩放框 | 拖框内 / 拖角点 |
| 切换标签 | `1` ~ `9` |
| 改选中框的类别 | 双击框，或点底部标签 |
| 删除选中框 | `Delete` / `Backspace` |
| 复制选中框 | `Ctrl + D` |
| 撤销 / 重做 | `Ctrl + Z` / `Ctrl + Shift + Z` |
| 上一张 / 下一张 | `←` `→` 或 `A` `D` |
| 缩放 | 滚轮 / `+` `-` |
| 适应窗口 | `F` |
| 平移画布 | 空格 / 右键 / 中键拖动 |
| 导出 | `Ctrl + S` |
| 快捷键面板 | `?` |

## 开发

```bash
npm install
npm run dev      # 开发
npm run build    # 类型检查 + 打包
npm run preview  # 预览打包结果
```

## 技术要点

- **React 19 + TypeScript + Vite + Tailwind v4 + Zustand**
- **UI 组件库：shadcn/ui**（基于 Radix UI + Tailwind，`src/components/ui/`）+ **lucide-react** 图标
- **主题**：shadcn 设计令牌（CSS 变量）+ 品牌色定制，见 `src/index.css` 的 `:root`
- **⚠️ `@layer` 约束**：自定义的 base 样式必须写在 `@layer base { ... }` 里。未分层的
  `* { margin:0; padding:0 }` 优先级高于 `@layer utilities`，会把全站每一个 `p-*` / `px-*` /
  `m-*` 静默清零——功能都在，但界面挤成一团。自定义外观件写在 `@layer components`，工具类才能覆盖它
- 画布用原生 Canvas 2D 绘制，`requestAnimationFrame` + 脏标记调度，一帧最多重绘一次
- 平移 / 缩放 / 拖拽期间只改 ref，不触发 React 重渲染，松手才提交到 store
- 图片解码成 `ImageBitmap` 并做 LRU 缓存；缩略图用 `IntersectionObserver` 懒生成，只解码一次
- 撤销栈只记录被修改那张图的标注，不做整棵状态树快照
- 自动保存是增量的：图片二进制只写一次，只回写变动过的标注
- 图片列表卡片做了 `memo`，标注某张图时其它卡片不会重渲染
- `jszip` 按需动态加载，不拖慢首屏
- **字体**：`Inter`（拉丁与数字）+ `MiSans VF`（中文），**两者都必须是可变字体**，理由是
  中文基准字重用了 `460` —— MiSans 的名义字重偏轻（400 的内部 usWeightClass 只有 330，
  比微软雅黑细一档，小字号纯白底上发虚）。而旧的回落字体微软雅黑只有 300/400/700，
  写 500 会被 CSS 字重匹配算法解析成 400（等于没写）、写 600 直接跳 700：
  **中文的「稍微粗一点」在那个字体下是无法表达的**。
  顺序也不能把 MiSans 提到 Inter 前面——MiSans 自带拉丁字母，会连数字一起抢走

## 字体授权

| 字体 | 用途 | 授权 |
| --- | --- | --- |
| **MiSans**（小米） | 界面中文，可变字重 `MiSans VF` | [MiSans 字体知识产权许可协议](https://hyperos.mi.com/font/zh/)：**免费商用**，但要求**注明使用了 MiSans 字体**、不得对字体本身改编或二次开发、不得**单独**再分发字体文件 |
| **Inter**（Rasmus Andersson） | 界面拉丁字母与数字 | SIL Open Font License 1.1 |

> MiSans 经由 npm 包 [`misans`](https://github.com/dsrkafuu/misans) 引入，该包只做
> **子集化与格式转换**（按 `unicode-range` 切成 56 片，浏览器只下载实际渲染到的分片），
> **不改变小米的授权条件**。字体随构建产物打包而非挂 CDN —— 这个工具的核心承诺是纯本地可用，
> 引一个运行时依赖网络的字体等于自毁承诺。

## 添加新 UI 组件

```bash
npx shadcn@latest add <component>   # 例如：npx shadcn@latest add command
```

组件源码落在 `src/components/ui/`，可直接修改定制。已有：
button / dialog / alert-dialog / dropdown-menu / tooltip / progress /
separator / switch / scroll-area / input / label / kbd / badge / sonner。

## 为未来功能预留的扩展点

- **tf.js 自动标注**：`src/store/useStore.ts` 是唯一的标注数据源（`addBox` /
  `updateBox` / `setBoxLabel` 已是纯函数式接口），模型推理只需生成矩形数组后调
  `addBox` 批量写入即可，画布 / 持久化 / 撤销全部自动生效
- **更多导出格式**：`src/utils/exporters.ts` 的 `buildExportFiles` 是纯函数，
  加一个分支即可
- **自定义视图控件**：`src/canvas/canvasApi.ts` 已暴露 `fit / zoomBy / setSpace`，
  供快捷键或未来的工具栏调用

## 目录结构

```
src/
├─ canvas/        渲染引擎与视图状态
├─ components/    业务组件
│  └─ ui/         shadcn/ui 组件集（Radix + Tailwind）
├─ store/         Zustand 状态（标注数据 / 视图 / 提示）
├─ utils/         几何、图片缓存、导出、持久化
├─ lib/           cn() 工具（clsx + tailwind-merge）
└─ types/         类型定义
```
