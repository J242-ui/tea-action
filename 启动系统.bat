@echo off
title 茶园智能执行系统 - 一键启动
cd /d "%~dp0"

echo ============================================
echo   茶园智能执行系统 · 一键启动
echo ============================================
echo.

netstat -ano | findstr ":8510" | findstr "LISTENING" >nul 2>&1
if not %errorlevel%==0 goto start_ws
echo [1/3] WebSocket 服务已在运行 8510，跳过启动
goto check_sl
:start_ws
echo [1/3] 启动 WebSocket 服务 8510...
start "WS服务-8510" "C:\Users\Lenovo\anaconda3\python.exe" -m src.ws_server

:check_sl
netstat -ano | findstr ":8501" | findstr "LISTENING" >nul 2>&1
if not %errorlevel%==0 goto start_sl
echo [2/3] Streamlit 主站已在运行 8501，跳过启动
goto open_page
:start_sl
echo [2/3] 启动 Streamlit 主站 8501...
start "Streamlit主站-8501" "C:\Users\Lenovo\anaconda3\python.exe" -m streamlit run app.py

:open_page
timeout /t 5 /nobreak >nul
echo [3/3] 打开动态展示网页...
start http://127.0.0.1:8510/

echo.
echo ============================================
echo   启动完成！
echo   动态展示网页 : http://127.0.0.1:8510/
echo   Streamlit主站 : http://localhost:8501
echo   （在 Streamlit 侧边栏进入智能执行页可手动控制 Agent）
echo   关闭本窗口不会影响服务运行。
echo ============================================
pause
