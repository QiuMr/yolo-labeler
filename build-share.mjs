// 构建「发布包」：把 dist/ 复制成一个干净的文件夹，只保留真正需要上传的文件。
//
// 与 dist/ 的区别：发布包里**没有** .map 之类的调试产物，只留浏览器真正要取的东西，
// 这样上传时不容易漏、也不容易把无关文件带上。
//
// 用法： node build-share.mjs [输出目录名]     默认 yolo-labeler-pages
//
// ⚠️ 部署方式已变更：GitHub Pages 现在由 .github/workflows/deploy.yml 自动发布，
//    推到 main 即可，**不需要**再跑这个脚本、也不需要把产物提交进仓库。
//    本脚本保留给两种场景：① 传到不支持 CI 的静态托管（对象存储 / 内网）；
//    ② 打一个离线分发包发给别人。
import { readdir, stat, mkdir, copyFile, rm, readFile } from 'node:fs/promises'
import { join, relative, sep, dirname, extname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = fileURLToPath(new URL('.', import.meta.url))
const DIST = join(ROOT, 'dist')
const OUT_NAME = process.argv[2] || 'yolo-labeler-pages'
const OUT = join(ROOT, OUT_NAME)

/** 必须原样发布的内容；其余一律不带 */
const KEEP_EXT = new Set([
  '.html', '.js', '.mjs', '.css', '.svg', '.json', '.png', '.jpg', '.jpeg',
  '.webp', '.ico', '.woff2', '.wasm', '.onnx', '.txt',
])

async function walk(dir) {
  const out = []
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name)
    if (e.isDirectory()) out.push(...(await walk(p)))
    else out.push(p)
  }
  return out
}

// 前置检查：dist 必须存在且是新的
let distFiles
try {
  distFiles = await walk(DIST)
} catch {
  console.error('找不到 dist/ —— 请先运行： npm run build')
  process.exit(1)
}
const indexHtml = join(DIST, 'index.html')
try {
  await stat(indexHtml)
} catch {
  console.error('dist/index.html 不存在 —— 构建似乎没完成，请重新运行： npm run build')
  process.exit(1)
}

// 关键校验：crossorigin 必须已经被去掉（否则在鉴权回源的托管上会整站 403 → 白屏）
const html = await readFile(indexHtml, 'utf8')
if (/\scrossorigin(?=[\s/>])/.test(html)) {
  console.error('⚠️  dist/index.html 里还残留 crossorigin —— 部署到鉴权回源的托管上会导致白屏。')
  console.error('    请确认 vite.config.ts 里的 stripCrossorigin 插件还在，然后重新 npm run build。')
  process.exit(1)
}

// 清空输出目录 —— 但**必须保留 .git**。
// 历史上部署就是把产物直接推上 GitHub 的，所以这个目录本身可能就是一个 git 仓库。
// 早先这里写的是 rm -rf 整个 OUT，等于把仓库连根删掉（.git 一没，本地历史就没了，
// 而产物目录又是能被脚本随时重建的，很容易在毫无察觉的情况下丢东西）。
await mkdir(OUT, { recursive: true })
for (const entry of await readdir(OUT, { withFileTypes: true })) {
  if (entry.name === '.git') continue
  await rm(join(OUT, entry.name), { recursive: true, force: true })
}

let copied = 0
let bytes = 0
const skipped = []
for (const f of distFiles) {
  const rel = relative(DIST, f)
  const name = rel.split(sep).pop()
  // 点开头的小文件（.nojekyll）要保留：GitHub Pages 靠它跳过 Jekyll
  const keep = KEEP_EXT.has(extname(f).toLowerCase()) || name.startsWith('.')
  if (!keep) {
    skipped.push(rel)
    continue
  }
  const dest = join(OUT, rel)
  await mkdir(dirname(dest), { recursive: true })
  await copyFile(f, dest)
  copied += 1
  bytes += (await stat(f)).size
}

// .nojekyll 是 GitHub Pages 的必需项，单独确认一次
let hasNojekyll = false
try {
  await stat(join(OUT, '.nojekyll'))
  hasNojekyll = true
} catch { /* 缺失 */ }

const groups = new Map()
for (const f of distFiles.filter((f) => KEEP_EXT.has(extname(f).toLowerCase()) || f.split(sep).pop().startsWith('.'))) {
  const rel = relative(DIST, f)
  const d = rel.includes(sep) ? rel.split(sep)[0] : '(根目录)'
  if (!groups.has(d)) groups.set(d, { n: 0, b: 0 })
  const g = groups.get(d)
  g.n += 1
  g.b += (await stat(f)).size
}

console.log(`\n发布包已生成： ${OUT_NAME}/`)
console.log(`  文件数 ${copied}    合计 ${(bytes / 1048576).toFixed(2)} MB\n`)
console.log('  目录           文件数     体积')
for (const [d, g] of [...groups.entries()].sort((a, b) => b[1].b - a[1].b)) {
  console.log(`  ${d.padEnd(14)} ${String(g.n).padStart(4)}   ${(g.b / 1048576).toFixed(2).padStart(7)} MB`)
}
if (skipped.length) console.log(`\n  已跳过 ${skipped.length} 个非必要文件： ${skipped.slice(0, 5).join(', ')}${skipped.length > 5 ? ' …' : ''}`)

console.log(`\n  .nojekyll: ${hasNojekyll ? '已包含 ✓（GitHub Pages 不会拿 Jekyll 处理，下划线开头的文件不会被跳过）' : '缺失 ✗ —— 请在 public/ 里放一个空文件后重新构建'}`)
console.log(`  crossorigin: 已清除 ✓（不会因缺 cookie 被鉴权回源的 CDN 拒绝）`)

console.log('\n下一步（按需二选一）：')
console.log('  ① 部署到 GitHub Pages —— 什么都不用做，推到 main 后 Actions 会自动构建发布')
console.log('  ② 传到别的静态托管 —— 把 ' + OUT_NAME + '/ 里的**内容**（不是文件夹本身）传上去\n')