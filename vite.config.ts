import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'
import { defineConfig, type Plugin } from 'vite'

// 说明：曾在这里挂过一个 `misansMetrics` 插件，构建期给 MiSans 的 56 个 @font-face
// 注入 ascent-override / descent-override 来修正「中文在行盒里偏上」的问题。
// 机制本身是成立的（100px 下基线确实从 104 移到 109），**但这个 app 的字号是 11~15px**：
// `109% × 13px = 14.17` 与 `104% × 13px = 13.52` 在布局时**都被取整成 14**，
// 覆盖在小字号上等于没改 —— 在 100px 上验证机制、却在 13px 上使用，是我的判断失误。
// 该方案已撤销，改用元素级补偿（见 src/index.css 的 @layer base）。

/**
 * 去掉产物 HTML 里的 `crossorigin` 属性。
 *
 * Vite 默认会给 `<script type="module">`、`<link rel="modulepreload">`、
 * `<link rel="stylesheet">` 都加上 `crossorigin`（`base` 可能指向 CDN 时是必要的）。
 *
 * 但 `crossorigin` 不带值 = `anonymous`，会让浏览器**主动不发送 cookie**。
 * 部署到**要求登录态 / 走鉴权回源**的托管方时（实测：整个站点匿名请求一律 403），
 * 这几个请求就因为缺 cookie 被拒 → 模块链断 → **界面全白**，
 * 而控制台只留一条不起眼的 403，极难定位。
 *
 * 本项目的产物全部用相对路径（`base: './'`）与页面同源，**根本不需要 CORS**，
 * 去掉它不会有任何副作用：同源请求本来就不走 CORS，而字体请求按规范始终是 CORS 模式，
 * 与这个属性无关。
 */
function stripCrossorigin(): Plugin {
  return {
    name: 'strip-crossorigin',
    enforce: 'post',
    transformIndexHtml(html) {
      return html.replace(/\s+crossorigin(?=[\s/>])/g, '')
    },
  }
}

export default defineConfig({
  // 相对 base：构建产物用 ./assets/... 形式引用资源，
  // 可放到任何子目录 / 任意静态托管（GitHub Pages 项目页、Netlify、Vercel、对象存储…）直接运行
  base: './',
  plugins: [stripCrossorigin(), react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
    // 保证全应用只有一份 React，避免多实例导致的 "Invalid hook call"
    dedupe: ['react', 'react-dom'],
  },
  server: {
    headers: {
      'Cache-Control': 'no-store, must-revalidate',
      'Pragma': 'no-cache',
    },
    watch: {
      usePolling: true,
    },
  },
})
