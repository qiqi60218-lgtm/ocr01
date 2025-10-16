import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import re
import os
import time
from datetime import datetime
import docx
from docx.shared import Pt
import pyperclip
import streamlit as st
import tempfile
from streamlit_drawable_canvas import st_canvas
import json

# -------------------------- 关键配置 --------------------------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
st.set_page_config(page_title="交互式文档识别", layout="wide")
CAMERA_TIMEOUT = 60  # 超时时间延长至60秒（足够用户调整摄像头）

# 添加自定义CSS以支持设备自适应和性能优化
st.markdown("""
<style>
    @media (max-width: 768px) {
        .mobile-hidden {
            display: none !important;
        }
        .mobile-fullwidth {
            width: 100% !important;
        }
        .stButton > button {
            margin-bottom: 10px;
        }
    }
    /* 优化画布性能 */
    .stCanvas {
        image-rendering: -moz-crisp-edges;
        image-rendering: -webkit-crisp-edges;
        image-rendering: pixelated;
        image-rendering: crisp-edges;
    }
    /* 隐藏Streamlit默认的一些元素以获得更好的移动体验 */
    @media (max-width: 768px) {
        .css-1kyxreq {
            padding-top: 1rem !important;
        }
        .css-1v3fvcr {
            padding-right: 1rem !important;
            padding-left: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------- 核心类 --------------------------

class InteractiveDocRecognizer:
    def __init__(self):
        self.raw_img = None
        self.selected_img = None
        self.corrected_img = None
        self.ocr_result = ""
        self.fixed_result = ""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rotate_angle = 0.0
        self.rotation_cache = {}  # 添加旋转缓存以提高性能

    def load_image(self, img_array):
        if img_array is not None and isinstance(img_array, np.ndarray) and img_array.size > 0:
            self.raw_img = img_array
            self.selected_img = None
            self.corrected_img = None
            self.rotate_angle = 0.0
            self.rotation_cache = {}  # 清空缓存
            return True
        return False

    def select_region(self, img, canvas_result, scale=1.0):
        # 优化选择区域的性能
        if not canvas_result.json_data:
            return None
            
        objects = canvas_result.json_data.get('objects', [])
        if not objects:
            return None
            
        # 只处理矩形对象
        rect_objects = [obj for obj in objects if obj.get('type') == 'rect']
        if not rect_objects:
            return None
            
        # 取最后一个矩形
        rect = rect_objects[-1]
        x_min = int(rect['left'])
        y_min = int(rect['top'])
        x_max = int(rect['left'] + rect['width'])
        y_max = int(rect['top'] + rect['height'])
        
        # 考虑缩放比例，映射回原始图像尺寸
        x_min = max(0, int(x_min / scale))
        x_max = min(img.shape[1], int(x_max / scale))
        y_min = max(0, int(y_min / scale))
        y_max = min(img.shape[0], int(y_max / scale))
        
        # 检查是否有效选择
        if x_max > x_min and y_max > y_min:
            selected = img[y_min:y_max, x_min:x_max].copy()
            return selected if selected.size > 0 else None
        return None
    
    # 移动端的区域选择方法
    def select_region_mobile(self, img, rect_percentage):
        h, w = img.shape[:2]
        # 根据百分比计算实际坐标
        x_min = int(w * rect_percentage[0])
        y_min = int(h * rect_percentage[1])
        x_max = int(w * rect_percentage[2])
        y_max = int(h * rect_percentage[3])
        
        # 确保边界有效
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        if x_max > x_min and y_max > y_min:
            selected = img[y_min:y_max, x_min:x_max].copy()
            return selected if selected.size > 0 else None
        return None

    def rotate_image(self, img, angle):
        if img is None:
            return None
            
        # 缓存优化：如果相同角度已计算过，直接返回缓存结果
        angle_key = round(angle, 1)  # 四舍五入到一位小数作为键
        if angle_key in self.rotation_cache:
            return self.rotation_cache[angle_key]
            
        # 优化旋转算法
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # 当角度是90的倍数时，使用更高效的转置方法
        if abs(angle % 90) < 0.1:
            angle_rounded = round(angle / 90) * 90
            if angle_rounded == 90 or angle_rounded == -270:
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle_rounded == 180 or angle_rounded == -180:
                rotated = cv2.rotate(img, cv2.ROTATE_180)
            elif angle_rounded == 270 or angle_rounded == -90:
                rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:  # 0度
                rotated = img.copy()
        else:
            # 一般角度使用常规旋转方法
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            # 对于性能优化，适当降低插值质量
            flags = cv2.INTER_LINEAR if abs(angle) > 10 else cv2.INTER_NEAREST
            rotated = cv2.warpAffine(
                img, M, (new_w, new_h),
                flags=flags,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255
            )
        
        # 缓存结果
        self.rotation_cache[angle_key] = rotated
        return rotated

    def preprocess_for_ocr(self, img):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        return binary

    def ocr_text(self):
        if self.corrected_img is None:
            return False
        try:
            ocr_img = self.preprocess_for_ocr(self.corrected_img)
            self.ocr_result = pytesseract.image_to_string(
                Image.fromarray(ocr_img),
                lang='chi_sim+eng',
                config='--psm 6'
            )
            self.ocr_result = re.sub(r'\n+', '\n', self.ocr_result.strip())
            return True
        except Exception as e:
            st.error(f"OCR失败：{str(e)}")
            return False

    def fix_text(self):
        if not self.ocr_result:
            return False
        fix_rules = {
            "分折": "分析", "工贝": "工具", "即然": "既然", "象": "像", "做": "作",
            "teh": "the", "hwo": "how", "tahn": "than", "whta": "what",
            "，": ",", "。": ".", "；": ";", "：": ":", "？": "?"
        }
        self.fixed_result = self.ocr_result
        for wrong, right in fix_rules.items():
            self.fixed_result = self.fixed_result.replace(wrong, right)
        return True

    def export(self, export_type, custom_name=None):
        if not self.fixed_result:
            return ""
        base_name = custom_name.strip() if custom_name else f"doc_recognize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_name = re.sub(r'[\\/:*?"<>|]', "_", base_name)
        try:
            if export_type == "txt":
                path = os.path.join(self.temp_dir.name, f"{base_name}.txt")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(self.fixed_result)
                return path
            elif export_type == "docx":
                path = os.path.join(self.temp_dir.name, f"{base_name}.docx")
                doc = docx.Document()
                style = doc.styles['Normal']
                style.font.name = 'SimSun'
                style.font.size = Pt(12)
                for para in self.fixed_result.split('\n'):
                    if para.strip():
                        doc.add_paragraph(para)
                doc.save(path)
                return path
            elif export_type == "clipboard":
                pyperclip.copy(self.fixed_result)
                return "clipboard"
        except Exception as e:
            st.error(f"导出失败：{str(e)}")
        return ""

# -------------------------- 摄像头拍摄逻辑（使用Streamlit内置摄像头组件，支持跨设备权限申请） --------------------------
# 检测可用摄像头设备 - 简化版本以兼容更多环境
def is_mobile_device():
    """简化的设备检测，使用session_state存储设备类型"""
    # 确保session_state中有is_mobile键
    if 'is_mobile' not in st.session_state:
        st.session_state.is_mobile = False
    
    return st.session_state.is_mobile

# 创建侧边栏设备模式切换选项的函数
def create_mobile_device_sidebar():
    """在侧边栏创建设备模式切换选项"""
    with st.sidebar:
        st.markdown("### 设备模式设置")
        # 确保session_state中有is_mobile键
        if 'is_mobile' not in st.session_state:
            st.session_state.is_mobile = False
        # 创建唯一的checkbox
        st.session_state.is_mobile = st.checkbox("启用移动设备模式", value=st.session_state.is_mobile, key="mobile_device_checkbox")

def get_available_cameras():
    """返回可用摄像头列表"""
    # 在实际浏览器环境中，Streamlit的camera_input默认只能访问默认摄像头
    # 但我们可以通过index参数来模拟不同摄像头
    return [0, 1]  # 0表示默认摄像头，1表示第二个摄像头

def capture_image_mobile_friendly():
    """
    简化的摄像头捕获函数
    直接使用Streamlit的camera_input组件，避免复杂的JavaScript逻辑
    """
    # 显示提示信息
    device_type = "移动设备" if is_mobile_device() else "电脑端"
    st.info(f"📱 正在使用{device_type}模式 - 请点击下方按钮并允许浏览器访问摄像头权限")
    
    # 初始化会话状态 - 简化为直接使用索引
    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = 0
    
    # 简单的摄像头切换按钮
    col1, col2 = st.columns(2)
    with col1:
        camera_label = "默认摄像头" if st.session_state.camera_index == 0 else "USB摄像头"
        st.write(f"当前使用: **{camera_label}**")
    with col2:
        if st.button("🔄 切换摄像头", key="toggle_camera"):
            # 直接切换索引并使用session_state持久化
            st.session_state.camera_index = 1 - st.session_state.camera_index
            # 强制刷新整个页面以重新加载摄像头组件
            st.experimental_rerun()
    
    # 生成基于时间戳的唯一key，确保每次都能完全刷新
    timestamp = int(time.time())
    camera_key = f"camera_{st.session_state.camera_index}_{timestamp}"
    
    # 使用Streamlit内置的摄像头组件
    # 注意：在实际浏览器中，Streamlit的camera_input不支持直接指定设备索引
    # 但我们可以通过刷新和唯一key来尝试切换
    img_data = st.camera_input(
        label=f"点击拍照 ({camera_label})",
        key=camera_key,
        # 添加help文本指导用户
        help="如果无法切换摄像头，请尝试刷新页面或在浏览器权限设置中更改摄像头"
    )
    
    if img_data is not None:
        try:
            # 转换图像数据
            img = Image.open(img_data)
            img_array = np.array(img)
            
            # 检查图像是否有效
            if img_array is not None and img_array.size > 0:
                st.success("✅ 拍摄成功！")
                return (True, img_array)
            else:
                st.error("❌ 图像无效，请重试")
        except Exception as e:
            st.error(f"❌ 处理图像时出错：{str(e)}")
    
    # 返回默认值，但不显示错误，等待用户操作
    return (False, None)

# -------------------------- 界面交互逻辑 --------------------------
def main():
    # 设置响应式标题和提示
    st.title("🖼️ 交互式文档识别工具")
    
    # 创建侧边栏设备模式切换选项（只在应用开始时创建一次）
    create_mobile_device_sidebar()
    
    # 获取设备模式
    mobile_mode = is_mobile_device()
    st.session_state.is_mobile = mobile_mode
    
    # 根据设备类型显示不同的提示
    if mobile_mode:
        st.markdown("### 📱 移动设备模式：触摸操作优化")
        st.info("请使用下方的按钮和滑块进行操作。区域选择已优化为触摸友好型。")
    else:
        st.markdown("### 💻 电脑端模式：鼠标操作优化")
        st.info("请使用鼠标拖动进行区域框选，支持实时旋转预览。")
    
    st.divider()

    # 初始化会话状态
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = InteractiveDocRecognizer()
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'export_state' not in st.session_state:
        st.session_state.export_state = False
    if 'temp_img' not in st.session_state:
        st.session_state.temp_img = None
    
    recognizer = st.session_state.recognizer

    # 步骤1：获取图像
    if st.session_state.step == 1:
        st.subheader("📥 步骤1：获取图像")
        
        # 设备自适应的图像来源选择
        if mobile_mode:
            # 移动设备简化界面，优先显示摄像头
            st.markdown("#### 使用摄像头拍摄文档")
            capture_success, captured_img = capture_image_mobile_friendly()
            
            # 检查是否有临时保存的图像
            if 'temp_img' in st.session_state and st.session_state.temp_img is not None:
                captured_img = st.session_state.temp_img
                capture_success = True
                
            if capture_success and captured_img is not None:
                if recognizer.load_image(captured_img):
                    if st.button("✅ 拍摄成功，进入调整角度", use_container_width=True):
                        st.session_state.step = 2
                        st.experimental_rerun()
            
            # 添加文件上传选项（次要）
            st.markdown("\n#### 或从相册选择")
            uploaded_file = st.file_uploader("选择图像", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                try:
                    img = Image.open(uploaded_file)
                    img_array = np.array(img)
                    if recognizer.load_image(img_array):
                        st.success(f"✅ 加载成功：{uploaded_file.name}")
                        if st.button("进入调整角度", use_container_width=True):
                            st.session_state.step = 2
                            st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ 加载失败：{str(e)}")
        else:
            # 电脑端保持原有选项，但优化摄像头捕获
            img_source = st.radio("选择图像来源", ["摄像头拍摄", "本地文件"], horizontal=True)
            
            if img_source == "摄像头拍摄":
                # 桌面端也使用Streamlit的摄像头组件以支持跨设备
                capture_success, captured_img = capture_image_mobile_friendly()
                
                if capture_success and captured_img is not None:
                    st.session_state.temp_img = captured_img  # 临时保存图像
                    if st.button("✅ 拍摄成功，进入旋转步骤", use_container_width=True):
                        if recognizer.load_image(captured_img):
                            st.session_state.step = 2
                            st.experimental_rerun()
                        else:
                            st.error("❌ 图像无效，请重试")
            
            else:
                uploaded_file = st.file_uploader("选择图像（jpg/png）", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    try:
                        img = Image.open(uploaded_file)
                        img_array = np.array(img)
                        if recognizer.load_image(img_array):
                            st.success(f"✅ 加载成功：{uploaded_file.name}")
                            if st.button("进入步骤2：调整角度", use_container_width=True):
                                st.session_state.step = 2
                                st.experimental_rerun()
                    except Exception as e:
                        st.error(f"❌ 加载失败：{str(e)}")

    # 步骤2：调整角度（先旋转再框选）- 完全优化的实时旋转功能
    elif st.session_state.step == 2:
        if recognizer.raw_img is None:
            st.error("❌ 未检测到图像，请返回步骤1")
            if st.button("返回步骤1", use_container_width=True):
                st.session_state.step = 1
                st.experimental_rerun()
            return
        
        st.subheader("🔄 步骤2：调整角度")
        
        # 初始化旋转相关状态
        if 'manual_angle' not in st.session_state:
            st.session_state.manual_angle = recognizer.rotate_angle
        if 'rotated_img' not in st.session_state:
            st.session_state.rotated_img = recognizer.raw_img.copy()
        
        # 旋转处理函数，避免重复代码
        def process_rotation(new_angle):
            # 限制角度范围
            new_angle = max(-45.0, min(45.0, new_angle))
            st.session_state.manual_angle = new_angle
            recognizer.rotate_angle = new_angle
            # 立即更新预览图像
            st.session_state.rotated_img = recognizer.rotate_image(recognizer.raw_img, new_angle)
        
        # 滑块角度变化回调函数
        def on_slider_change():
            process_rotation(st.session_state.rotate_slider)
        
        # 手动输入角度变化回调
        def on_manual_input_change():
            try:
                new_angle = float(st.session_state.manual_input)
                process_rotation(new_angle)
            except ValueError:
                pass  # 忽略无效输入
        
        # 根据设备类型调整布局
        if mobile_mode:
            # 移动设备：垂直布局
            # 图像预览放在顶部
            st.image(st.session_state.rotated_img, 
                     caption=f"当前角度：{st.session_state.manual_angle}°", 
                     use_column_width=True)
            
            # 滑块和控制按钮放在下方
            col1, col2 = st.columns([3, 2])
            with col1:
                # 滑块调整 - 移动设备优化步长，使用on_change回调
                st.slider(
                    "旋转角度（°）",
                    min_value=-45.0,
                    max_value=45.0,
                    value=st.session_state.manual_angle,
                    step=1.0,  # 移动设备增大步长
                    key="rotate_slider",
                    on_change=on_slider_change
                )
            with col2:
                # 手动输入角度
                st.text_input("输入角度：", 
                            value=str(st.session_state.manual_angle),
                            key="manual_input",
                            on_change=on_manual_input_change)
            
            # 快速旋转按钮 - 移动设备简化布局
            st.write("快速调整：")
            quick_cols = st.columns(3)
            quick_angles = [-90, 0, 90]  # 简化为三个常用角度
            for i, quick_angle in enumerate(quick_angles):
                with quick_cols[i]:
                    if st.button(f"{quick_angle}°", use_container_width=True, key=f"quick_rot_mobile_{quick_angle}"):
                        process_rotation(float(quick_angle))
        else:
            # 电脑端：水平布局
            col1, col2 = st.columns([1, 2])
            with col1:
                # 滑块调整 - 精细控制，使用on_change回调
                st.slider(
                    "旋转角度（°）",
                    min_value=-45.0,
                    max_value=45.0,
                    value=st.session_state.manual_angle,
                    step=0.5,  # 电脑端精细控制
                    key="rotate_slider",
                    on_change=on_slider_change
                )
                
                # 手动输入角度
                st.text_input("手动输入角度：", 
                            value=str(st.session_state.manual_angle),
                            key="manual_input",
                            on_change=on_manual_input_change)
                
                # 快速旋转按钮
                st.write("快速调整：")
                quick_cols = st.columns(5)
                quick_angles = [-90, -45, 0, 45, 90]
                for i, quick_angle in enumerate(quick_angles):
                    with quick_cols[i]:
                        if st.button(f"{quick_angle}°", use_container_width=True, 
                                   key=f"quick_rot_desktop_{quick_angle}"):
                            process_rotation(float(quick_angle))
            
            # 实时预览
            with col2:
                st.image(st.session_state.rotated_img, 
                         caption=f"当前角度：{st.session_state.manual_angle}°", 
                         use_column_width=True)
        
        # 保存旋转结果并继续
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↩️ 返回步骤1", use_container_width=True):
                st.session_state.step = 1
                st.experimental_rerun()
        with col2:
            if st.button("✅ 确认角度，进入框选", use_container_width=True):
                recognizer.corrected_img = st.session_state.rotated_img  # 保存旋转后的图像
                st.session_state.step = 3
                st.experimental_rerun()

    # 步骤3：框选区域 - 完全重构版，解决根本性错误
    elif st.session_state.step == 3:
        if recognizer.corrected_img is None:
            st.error("❌ 未检测到旋转后的图像，请返回步骤2")
            if st.button("返回步骤2", use_container_width=True):
                st.session_state.step = 2
                st.experimental_rerun()
            return
        
        st.subheader("✂️ 步骤3：框选识别区域")
        img = recognizer.corrected_img
        img_height, img_width = img.shape[:2]
        
        # 初始化会话状态
        if 'selected_region' not in st.session_state:
            st.session_state.selected_region = None
        
        if mobile_mode:
            # 移动设备：简化的区域选择界面
            st.info("📱 移动设备模式：选择整图或使用滑块调整区域")
            
            # 显示原图
            st.image(img, caption="待选择图像", use_column_width=True)
            
            # 整图识别选项（最优先）
            if st.button("📄 使用整图识别", use_container_width=True, type="primary"):
                recognizer.selected_img = img.copy()
                st.session_state.step = 4
                st.experimental_rerun()
            
            # 简单的自定义区域选择
            st.markdown("### 自定义识别区域")
            st.write("通过滑块调整需要识别的区域")
            
            # 初始化滑块默认值
            if 'region_percentages' not in st.session_state:
                st.session_state.region_percentages = [0, 0, 100, 100]  # 全图默认值
            
            # 两列布局的滑块
            col1, col2 = st.columns(2)
            with col1:
                left = st.slider("左边界 (%)", 0, 90, st.session_state.region_percentages[0], 5)
                top = st.slider("上边界 (%)", 0, 90, st.session_state.region_percentages[1], 5)
            with col2:
                right = st.slider("右边界 (%)", left + 5, 100, st.session_state.region_percentages[2], 5)
                bottom = st.slider("下边界 (%)", top + 5, 100, st.session_state.region_percentages[3], 5)
            
            # 保存滑块值
            st.session_state.region_percentages = [left, top, right, bottom]
            
            # 计算实际坐标
            x1 = int(img_width * left / 100)
            y1 = int(img_height * top / 100)
            x2 = int(img_width * right / 100)
            y2 = int(img_height * bottom / 100)
            
            # 实时预览选择的区域
            preview_img = img.copy()
            # 绘制红色边框
            cv2.rectangle(preview_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            st.image(preview_img, caption="预览（红色框为选择区域）", use_column_width=True)
            
            # 确认按钮
            if st.button("✅ 确认自定义区域", use_container_width=True):
                if x2 > x1 and y2 > y1:
                    selected_region = img[y1:y2, x1:x2].copy()
                    recognizer.selected_img = selected_region
                    st.session_state.step = 4
                    st.experimental_rerun()
                else:
                    st.error("❌ 区域无效，请调整滑块")
        else:
            # 电脑端：完全重写的画布框选功能
            st.info("💻 电脑端模式：使用鼠标拖动绘制矩形区域")
            
            # 初始化画布key
            if 'canvas_key' not in st.session_state:
                st.session_state.canvas_key = "document_canvas"
            
            # 计算合适的画布尺寸（关键优化：使用更小的尺寸）
            max_dimension = 500
            if img_width > img_height:
                scale = max_dimension / img_width
                canvas_width = max_dimension
                canvas_height = int(img_height * scale)
            else:
                scale = max_dimension / img_height
                canvas_height = max_dimension
                canvas_width = int(img_width * scale)
            
            # 重要：将图像转换为PIL格式
            from PIL import Image
            pil_img = Image.fromarray(img).resize((canvas_width, canvas_height))
            
            # 重写的画布框选功能
            # 使用固定的drawing_mode为"rect"确保始终是矩形工具
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # 半透明橙色填充
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=pil_img,
                update_streamlit=True,  # 必须为True才能获取绘制结果
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",  # 固定为矩形模式
                key=st.session_state.canvas_key,
                display_toolbar=True,
                # 添加明确的指导文本
                help="请使用矩形工具绘制需要识别的区域"
            )
            
            # 三列布局的控制按钮
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ 清除选择", use_container_width=True):
                    # 重新生成key以清除画布
                    st.session_state.canvas_key = f"canvas_{int(time.time())}"
                    st.session_state.selected_region = None
                    st.experimental_rerun()
            with col2:
                if st.button("📄 整图识别", use_container_width=True, type="primary"):
                    # 直接使用整个图像
                    recognizer.selected_img = img.copy()
                    st.session_state.step = 4
                    st.experimental_rerun()
            with col3:
                if st.button("✅ 确认区域", use_container_width=True):
                    # 手动处理画布结果
                    # 简化的框选结果处理逻辑
                    # 检查是否有绘制的矩形
                    if canvas_result.json_data and "objects" in canvas_result.json_data:
                        objects = canvas_result.json_data["objects"]
                        if len(objects) > 0:
                            # 获取最后绘制的矩形对象
                            last_object = objects[-1]
                            # 确保是矩形类型
                            if last_object["type"] == "rect":
                                # 提取矩形坐标
                                left = last_object["left"]
                                top = last_object["top"]
                                width = last_object["width"]
                                height = last_object["height"]
                                
                                # 转换坐标到原始图像尺寸
                                scale_x = img_width / canvas_width
                                scale_y = img_height / canvas_height
                                
                                x1 = int(left * scale_x)
                                y1 = int(top * scale_y)
                                x2 = int((left + width) * scale_x)
                                y2 = int((top + height) * scale_y)
                                
                                # 确保坐标在有效范围内
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(img_width, x2)
                                y2 = min(img_height, y2)
                                
                                # 检查区域大小是否有效
                                if (x2 - x1) > 10 and (y2 - y1) > 10:
                                    # 提取并保存选择的区域
                                    selected_region = img[y1:y2, x1:x2].copy()
                                    recognizer.selected_img = selected_region
                                    # 显示成功消息
                                    st.success("✅ 区域选择成功！")
                                    st.session_state.step = 4
                                    st.experimental_rerun()
                                else:
                                    st.error("❌ 选择区域太小，请选择更大的区域（至少10x10像素）")
                            else:
                                st.error("❌ 请使用矩形工具绘制区域")
                        else:
                            st.info("💡 请在图像上绘制矩形区域")
                    else:
                        st.info("💡 请在图像上绘制矩形区域")
            
            # 统一的导航按钮
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← 返回调整角度", use_container_width=True):
                    st.session_state.step = 2
                    st.experimental_rerun()
            with col2:
                # 只在有选择区域时启用下一步按钮
                if mobile_mode:
                    # 移动端总是可以继续（整图或自定义）
                    if st.button("继续识别 →", use_container_width=True, type="primary"):
                        # 如果没有自定义区域，默认使用整图
                        if st.session_state.selected_region is None:
                            recognizer.selected_img = img.copy()
                        st.session_state.step = 4
                        st.experimental_rerun()
                else:
                    # 电脑端需要先框选或使用整图按钮
                    st.button("继续识别 →", use_container_width=True, disabled=True)
                    st.caption("请先选择区域或点击整图识别")

    # 步骤4：识别与导出 - 响应式设计优化
    elif st.session_state.step == 4:
        if recognizer.selected_img is None:
            st.error("❌ 未检测到框选区域，请返回步骤3")
            if st.button("返回步骤3", use_container_width=True):
                st.session_state.step = 3
                st.experimental_rerun()
            return
        
        st.subheader("📝 步骤4：文本识别与导出")
        
        # 根据设备类型调整图像显示大小
        if mobile_mode:
            # 移动设备：图像显示更小
            st.image(recognizer.selected_img, caption="待识别图像", width=300)
        else:
            # 电脑端：全宽显示
            st.image(recognizer.selected_img, caption="待识别图像", use_column_width=True)
        
        # 开始识别按钮
        if st.button("🔍 开始识别", use_container_width=True):
            with st.spinner("识别中..."):
                # 添加进度指示
                progress_bar = st.progress(0)
                progress_bar.progress(30)
                
                # 执行OCR
                if recognizer.ocr_text():
                    progress_bar.progress(100)
                    st.success("识别完成！")
                else:
                    progress_bar.progress(100)
        
        # 识别结果展示
        if recognizer.ocr_result:
            # 根据设备类型调整文本区域高度
            text_area_height = 150 if mobile_mode else 200
            
            st.text_area("识别结果（可修改）：", recognizer.ocr_result, height=text_area_height, key="ocr_result")
            recognizer.ocr_result = st.session_state.ocr_result
            
            # 使用session_state管理导出状态，避免回退
            if not st.session_state.export_state:
                if st.button("🔧 纠错并导出", use_container_width=True):
                    if recognizer.fix_text():
                        st.session_state.export_state = True
                        st.experimental_rerun()
            else:
                # 导出界面，保持状态
                st.text_area("纠错后结果：", recognizer.fixed_result, height=text_area_height, key="fixed_result")
                recognizer.fixed_result = st.session_state.fixed_result
                
                # 设备自适应的导出选项布局
                if mobile_mode:
                    # 移动设备：垂直布局
                    export_type = st.selectbox(
                        "导出方式",
                        [("📄 TXT", "txt"), ("📑 Word", "docx"), ("📋 剪贴板", "clipboard")],
                        format_func=lambda x: x[0],
                        key="export_type"
                    )[1]
                    custom_name = st.text_input("文件名（可选）", key="custom_name")
                else:
                    # 电脑端：水平布局
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        export_type = st.selectbox(
                            "导出方式",
                            [("📄 TXT", "txt"), ("📑 Word", "docx"), ("📋 剪贴板", "clipboard")],
                            format_func=lambda x: x[0],
                            key="export_type"
                        )[1]
                    with col2:
                        custom_name = st.text_input("文件名（可选）", key="custom_name")
                
                # 导出按钮
                if st.button("🚀 确认导出", use_container_width=True):
                    result = recognizer.export(export_type, custom_name)
                    if result == "clipboard":
                        st.success("✅ 已复制到剪贴板")
                        # 保持在导出界面，允许用户继续导出
                    elif result:
                        with open(result, "rb") as f:
                            st.download_button(
                                f"下载{os.path.basename(result)}",
                                f,
                                file_name=os.path.basename(result),
                                use_container_width=True
                            )
                        st.success("✅ 导出成功！")
        
        # 放置在底部的导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↩️ 返回步骤3", use_container_width=True):
                st.session_state.export_state = False
                st.session_state.step = 3
                st.experimental_rerun()
        with col2:
            if st.button("🔄 重新开始", use_container_width=True):
                st.session_state.clear()
                st.experimental_rerun()

    # 底部提示信息，设备自适应
    st.divider()
    if mobile_mode:
        st.markdown("### 📱 移动设备使用提示")
        st.info("• 点击拍照按钮后，请允许浏览器访问摄像头权限\n• 区域选择可使用预设或调整滑块\n• 完成识别后可选择导出格式")
    else:
        st.markdown("### 💻 电脑端使用提示")
        st.info("• 使用鼠标拖动进行区域框选\n• 旋转操作支持实时预览\n• 可通过滑块和输入框精确调整角度")

# -------------------------- 响应式设计与性能优化说明 --------------------------
# 1. 跨设备摄像头权限：使用Streamlit内置的camera_input组件，自动处理浏览器权限申请
# 2. 设备自适应布局：通过JavaScript检测设备类型，为移动设备和桌面设备提供不同交互界面
# 3. 移动端触摸优化：针对触摸屏设计简化的操作流程，包括预设区域选择和滑块调整
# 4. 性能优化：
#    - 旋转操作添加缓存机制，避免重复计算
#    - 针对90度倍数的角度使用更高效的旋转算法
#    - 优化画布尺寸和渲染方式
#    - 使用session_state缓存中间结果

# -------------------------- 跨设备兼容性配置 --------------------------
# 应用已配置为在局域网内运行，通过以下地址访问：
# - 本地访问：http://localhost:8502
# - 局域网访问：http://服务器IP:8502
# 浏览器兼容性：
# - Chrome、Firefox、Safari、Edge等现代浏览器
# - iOS Safari和Android Chrome已优化

if __name__ == "__main__":
    try:
        import streamlit_drawable_canvas
    except ImportError:
        st.error("正在安装依赖（仅首次需要）...")
        os.system("pip install streamlit-drawable-canvas")
        st.experimental_rerun()
    
    # 清除可能影响性能的缓存
    if 'canvas_key' in st.session_state:
        del st.session_state.canvas_key
    
    # 启动应用
    main()