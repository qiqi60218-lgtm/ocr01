@echo off

rem 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python。请先安装Python并添加到环境变量。
    pause
    exit /b 1
)

echo 安装依赖包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 错误: 安装依赖失败。
    pause
    exit /b 1
)

rem 如果不存在 .env，则复制 .env.example
if not exist ".env" (
    if exist ".env.example" (
        copy /Y ".env.example" ".env" >nul
        echo 已创建默认配置 .env，请按需填写远程OCR/LLM参数。
    )
)

echo 启动DocVision AI后端服务...
echo 服务将运行在 http://0.0.0.0:5000
python app.py