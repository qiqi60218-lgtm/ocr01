# DocVision AI - 智能文档识别系统

## 项目介绍
DocVision AI 是一个基于前后端分离架构的智能文档识别系统，支持相机实时扫描、图像上传、OCR文本识别和智能文本纠错等功能。

## 架构说明

### 前后端分离架构
- **前端**：纯HTML/CSS/JavaScript实现，使用Tailwind CSS构建响应式界面
- **后端**：基于Flask框架的RESTful API，处理OCR识别、图像处理等核心功能

### 项目结构
```
ocr01/
├── backend/                  # 后端服务
│   ├── app.py                # Flask应用主文件
│   ├── requirements.txt      # Python依赖
│   └── start_server.bat      # 启动脚本
└── frontend/                 # 前端页面
    └── index.html            # 主页面
```

## 核心功能

1. **相机实时扫描**
   - 支持前后摄像头切换
   - 实时预览和拍摄文档
   - 适配移动设备和桌面设备

2. **图像处理**
   - 图像旋转（支持90度、180度等任意角度）
   - 图像上传（支持各种常见图片格式）
   - 智能图像预处理

3. **OCR识别**
   - 支持中英文混合识别
   - 高精度文本提取
   - 优化的识别算法

4. **文本处理**
   - 智能文本纠错
   - 文本复制和下载
   - 文本编辑功能

## 技术栈

### 前端
- HTML5 + CSS3 + JavaScript
- Tailwind CSS v3（UI框架）
- Font Awesome（图标库）
- HTML5 Canvas API（图像处理）
- MediaDevices API（相机访问）

### 后端
- Python 3.8+
- Flask 3.0（Web框架）
- OpenCV（图像处理）
- Tesseract OCR（文本识别）
- NumPy（数值计算）
- PIL/Pillow（图像处理）

## 环境要求

### 后端要求
- Python 3.8 或更高版本
- Tesseract OCR 4.1 或更高版本
  - Windows用户：从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装
  - 安装后需确保添加到系统环境变量，或在app.py中修改路径

## 使用方法

### 1. 启动后端服务

#### Windows系统：
1. 进入 `backend` 目录
2. 双击运行 `start_server.bat` 脚本
3. 脚本会自动安装依赖并启动服务
4. 服务将在 http://0.0.0.0:5000 启动

#### 手动启动（可选）：
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 2. 访问前端页面

启动后端服务后，可以通过以下方式访问前端页面：

1. **通过后端服务访问**（推荐）：
   - 打开浏览器，访问 http://localhost:5000
   - 后端会自动提供前端静态文件

2. **直接访问本地文件**：
   - 打开浏览器，直接打开 `frontend/index.html` 文件
   - 确保后端服务仍在运行

## 摄像头权限说明

首次使用时，浏览器会请求摄像头访问权限，请确保：

1. 允许浏览器访问摄像头
2. 使用HTTPS协议（如果在生产环境）
3. 在移动设备上启用相机权限

## 常见问题解答

1. **摄像头无法启动**
   - 检查浏览器是否已授予相机权限
   - 确保没有其他应用正在使用摄像头
   - 尝试切换前后摄像头

2. **OCR识别不准确**
   - 确保图像光线充足
   - 尽量使文档与摄像头平行
   - 使用文本纠错功能优化结果

3. **后端服务启动失败**
   - 检查Python是否正确安装
   - 确认Tesseract OCR路径配置正确
   - 检查端口5000是否被占用

## API接口说明

### 1. 旋转图像
- **URL**: `/api/rotate`
- **方法**: `POST`
- **参数**: 
  - `image`: base64编码的图像数据
  - `angle`: 旋转角度
- **返回**: 旋转后的图像数据

### 2. OCR识别
- **URL**: `/api/ocr`
- **方法**: `POST`
- **参数**: 
  - `image`: base64编码的图像数据
  - `region`: 可选，要识别的区域坐标
- **返回**: 识别出的文本内容

### 3. 文本纠错
- **URL**: `/api/fix-text`
- **方法**: `POST`
- **参数**: 
  - `text`: 待纠错的文本
- **返回**: 纠错后的文本

## 注意事项

1. 本系统需要Tesseract OCR引擎支持，请确保正确安装
2. 相机功能需要浏览器支持并授予权限
3. 前后摄像头切换功能在移动设备上效果最佳
4. 大图像可能需要较长的处理时间

## 许可证

MIT License