@echo off
cd /d "%~dp0"
where node >nul 2>nul
if %errorlevel% neq 0 (
  echo 未检测到 Node.js。
  echo 方式一：安装 Node.js 后重双击本文件（ https://nodejs.org ）
  echo 方式二：直接用 Python 启动 —— 双击 start.py（需已安装 Python）
  pause
  exit /b 1
)
node serve.mjs
pause
