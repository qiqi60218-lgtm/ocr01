@echo off
setlocal

rem 统一启动后端与前端服务
set BASE=%~dp0

echo 安装后端依赖...
cd /d "%BASE%backend"
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 依赖安装失败，请检查网络或pip配置。
    pause
    exit /b 1
)

rem 初始化 .env（如未存在）
if not exist ".env" (
    if exist ".env.example" (
        copy /Y ".env.example" ".env" >nul
        echo 已创建默认配置 backend\.env，请按需填写远程OCR/LLM参数。
    )
)

echo 启动后端服务 http://localhost:5000/ ...
start "" python app.py

echo 启动前端静态服务器 http://localhost:8000/ ...
cd /d "%BASE%frontend"
start "" python -m http.server 8000

echo 已启动。后端 http://localhost:5000/ 前端 http://localhost:8000/
pause