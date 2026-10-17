// 零依赖本地静态服务器：用于在本机打开已构建的 dist/，或发给装了 Node.js 的人本地运行。
// 用法： node serve.mjs   （可选 PORT=8080 node serve.mjs）
import { createServer } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import { join, extname, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import { exec } from 'node:child_process'

const ROOT = fileURLToPath(new URL('dist/', import.meta.url))
const PORT = Number(process.env.PORT) || 4173

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.wasm': 'application/wasm',
  '.map': 'application/json; charset=utf-8',
}

const server = createServer(async (req, res) => {
  try {
    const urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname)
    const rel = normalize(urlPath).replace(/^(\.\.[/\\])+/, '')
    let filePath = join(ROOT, rel)
    let info = null
    try { info = await stat(filePath) } catch { /* not found */ }
    if (!info || info.isDirectory()) {
      // 非资源路径（无扩展名）回退到 index.html，支持 SPA；资源缺失直接 404
      if (!extname(rel)) filePath = join(ROOT, 'index.html')
      else { res.writeHead(404); res.end('Not found'); return }
    }
    // ETag + 304。dist 里躺着 25MB 的 ORT wasm 和 43MB 的模型，而 `no-cache`
    // 在没有校验器的情况下等于「每次都重新下载」—— 刷新一次就白拉将近 70MB。
    // 带上 ETag 之后浏览器会发 If-None-Match，命中就只回一个空的 304。
    // 必须在 readFile **之前**判断：否则为了回一个 304 还得先把 25MB 读进内存。
    const etag = info
      ? `W/"${info.size.toString(16)}-${Math.round(info.mtimeMs).toString(16)}"`
      : ''
    if (etag && req.headers['if-none-match'] === etag) {
      res.writeHead(304, { ETag: etag, 'Cache-Control': 'no-cache' })
      res.end()
      return
    }
    const data = await readFile(filePath)
    const type = MIME[extname(filePath).toLowerCase()] || 'application/octet-stream'
    const headers = { 'Content-Type': type, 'Cache-Control': 'no-cache' }
    if (etag) headers.ETag = etag
    res.writeHead(200, headers)
    res.end(data)
  } catch {
    res.writeHead(500); res.end('Server error')
  }
})

server.listen(PORT, '127.0.0.1', () => {
  const url = `http://127.0.0.1:${PORT}/`
  console.log(`YOLO 标注工具已启动： ${url}`)
  console.log('按 Ctrl+C 停止。')
  const cmd = process.platform === 'win32' ? `start "" "${url}"`
    : process.platform === 'darwin' ? `open "${url}"`
    : `xdg-open "${url}"`
  exec(cmd, () => {})
})
