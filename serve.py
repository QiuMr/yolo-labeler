# 零依赖本地静态服务器（Python 标准库）：用于在本机打开已构建的 dist/，
# 或发给装了 Python 的人本地运行。
# 用法： python serve.py   （可选 PORT=8080 python serve.py）
import http.server
import socketserver
import webbrowser
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dist')
PORT = int(os.environ.get('PORT', 4173))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('127.0.0.1', PORT), Handler) as httpd:
        url = f'http://127.0.0.1:{PORT}/'
        print(f'YOLO 标注工具已启动： {url}')
        print('按 Ctrl+C 停止。')
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
