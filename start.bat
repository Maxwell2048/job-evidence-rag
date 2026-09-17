@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title 求职 RAG

echo ==========================================
echo   求职 RAG  -  本机启动
echo ==========================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 首次运行：创建虚拟环境并安装依赖（需要网络，几分钟）...
    py -3 -m venv .venv || goto :fail
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || goto :fail
) else (
    echo [1/3] 虚拟环境已存在。
)

if not exist "config.local.json" (
    echo [!] 缺少 config.local.json。请复制 config.local.example.json 为 config.local.json 并填写本机模型服务地址。
    goto :fail
)

if not exist "resume\base_resume.docx" (
    echo [!] 缺少 resume\base_resume.docx（简历底稿）。把底稿放到 resume\ 目录后再启动。
    goto :fail
)

echo [2/3] 检查本机模型服务...
".venv\Scripts\python.exe" -X utf8 -c "import sys,webapp; ok,info=webapp.Pipeline().model_online(); print(('   在线：' if ok else '   离线：')+str(info)); sys.exit(0 if ok else 2)"
if errorlevel 2 (
    echo [!] 模型服务未在线。页面仍会打开，但"生成"按钮不可用；请先启动 llama-server 再刷新页面。
)

echo [3/3] 启动网页服务（会打开一个标题为"求职 RAG 服务"的窗口，关闭它即停止）
start "求职 RAG 服务" ".venv\Scripts\python.exe" -X utf8 webapp.py --port 5000
timeout /t 4 /nobreak >nul
start "" "http://127.0.0.1:5000"
echo 已在浏览器打开 http://127.0.0.1:5000 。本窗口可以关闭。
timeout /t 5 >nul
goto :eof

:fail
echo.
echo 启动失败，见上方提示。
pause
