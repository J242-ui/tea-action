@echo off
title 茶园智能执行系统 - 一键启动
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [错误] 未找到虚拟环境 Python：%PY%
    echo 请先在项目目录执行以下命令创建环境：
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    echo   pip install -r requirements.txt
    echo   pip install ultralytics opencv-python
    pause
    exit /b 1
)

echo ============================================
echo   茶园智能执行系统 · 一键启动（venv 版）
echo ============================================
echo.

netstat -ano | findstr ":8510" | findstr "LISTENING" >nul 2>&1
if not %errorlevel%==0 goto start_ws
echo [1/3] WebSocket 服务已在 8510 端口运行
goto check_sl
:start_ws
echo [1/3] 启动 WebSocket 服务 8510...
start "WS服务-8510" "%PY%" -m src.ws_server

:check_sl
netstat -ano | findstr ":8501" | findstr "LISTENING" >nul 2>&1
if not %errorlevel%==0 goto start_sl
echo [2/3] Streamlit 主站已在 8501 端口运行
goto open_page
:start_sl
echo [2/3] 启动 Streamlit 主站 8501...
start "Streamlit主站-8501" "%PY%" -m streamlit run app.py --server.headless true

:open_page
timeout /t 5 /nobreak >nul
echo [3/3] 打开动态展示网页...
start http://127.0.0.1:8510/

echo.
echo ============================================
echo   启动完成！
echo   动态展示网页 : http://127.0.0.1:8510/
echo   Streamlit主站 : http://localhost:8501
echo   请在 Streamlit 的智能执行页面手动启动 Agent。
echo   关闭本窗口不影响两个服务运行。
echo ============================================
pause
