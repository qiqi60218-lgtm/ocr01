# DocVision AI - 智能文档识别系统

DocVision AI 是一个基于前后端分离的文档识别应用，支持本地 Tesseract OCR 与云端（豆包/兼容）OCR，提供文本纠错与结构化抽取能力。后端使用 Flask 提供 API，前端为纯 HTML/CSS/JavaScript 页面。

## 项目结构
```
ocr01/
├── backend/
│   ├── app.py               # Flask 后端主程序（同时提供前端静态页面）
│   ├── requirements.txt     # 后端依赖
│   ├── .env.example         # 配置示例（LLM 与远程OCR）
│   ├── llm_client.py        # LLM 客户端（文本纠错、结构化抽取）
│   ├── remote_ocr_client.py # 远程OCR客户端（豆包/网关）
│   └── start_server.bat     # 一键启动后端脚本
├── frontend/
│   └── index.html           # 前端页面（含“开始识别/豆包识别”按钮）
├── run_server.bat           # 一键启动后端与前端
└── README.md
```

## 快速开始
- 安装依赖与启动后端（推荐）
  - 进入 `backend` 目录，执行：
    - `pip install -r requirements.txt`
    - `python app.py`
  - 浏览器访问 `http://localhost:5000/`
- 可选前端本地服务器
  - 进入 `frontend` 目录，执行：`python -m http.server 8000`
  - 访问 `http://localhost:8000/`（确保后端已启动）
- Windows 用户安装 Tesseract OCR
  - 到 `https://github.com/UB-Mannheim/tesseract/wiki` 下载并安装
  - 安装后建议加入系统环境变量；后端会自动探测常见安装路径

## 配置说明
- 复制配置示例
  - 将 `backend/.env.example` 复制为 `backend/.env`
- 云端 OCR（豆包/兼容）
  - 填写：`REMOTE_OCR_API_BASE`、`REMOTE_OCR_ENDPOINT`（如非默认）、`REMOTE_OCR_API_KEY`、`REMOTE_OCR_MODEL`、`REMOTE_OCR_TIMEOUT`
  - 启动后端后，在前端点击“豆包识别”按钮即可走云端OCR
- LLM（可选）
  - 如需文本纠错/抽取，填入：`LLM_API_BASE`、`LLM_API_KEY`、`LLM_MODEL`（以及必要的 `LLM_ENDPOINT`、`LLM_TIMEOUT` 等）

## 前端使用
- “开始识别”：使用本地 Tesseract OCR
- “豆包识别”：使用云端 OCR（需配置 `.env`）
- 支持框选局部区域识别；未框选时默认识别整图

## API 接口
- `GET /health`
  - 返回服务状态：`{ status, service, version }`
- `POST /api/rotate`
  - 请求：`{ image: <base64>, angle: <number>, enhance?: <bool> }`
  - 返回：`{ success, image: 'data:image/...;base64,...', message }`
- `POST /api/ocr`
  - 请求：
    - `image`: Base64 图像字符串（支持 dataURL 或纯 base64）
    - `crop_area?`: `{ x, y, width, height }`
    - `engine?`: `'doubao' | 'remote'`（云端识别）或留空（本地识别）
    - `auto_fix?`: 布尔，是否使用本地规则做简单纠错
  - 返回：`{ success, text, message, char_count }`
- `POST /api/recognize-table`
  - 请求：`{ image: <base64>, crop_area?: {…} }`
  - 返回：`{ success, table_text, table_html?, has_table, message }`
- `POST /api/fix-text`
  - 请求：`{ text: <string>, use_llm?: <bool>, instructions?: <string> }`
  - 返回：`{ success, text, message, original_length, fixed_length }`
- `POST /api/llm/extract`
  - 请求：`{ text: <string>, schema?: <any>, instruction?: <string> }`
  - 返回：`{ success, data }`（结构化 JSON）
- `POST /api/save-word`
  - 请求：`{ text: <string>, has_table?: <bool> }`
  - 返回：Word 文档（下载）

## 常见问题
- 远程 OCR 未配置
  - 点击“豆包识别”时报错 `远程OCR未配置`，请在 `backend/.env` 填写远程OCR参数并重启后端
- 摄像头/权限问题
  - 首次使用需授权浏览器摄像头；如在生产环境建议使用 HTTPS
- 端口被占用
  - 确认 `5000`（后端）或 `8000`（前端）未被其他程序占用

## 许可证
MIT License