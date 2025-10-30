import base64
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response

app = Flask(__name__)


def _strip_data_uri(b64: str) -> str:
    if isinstance(b64, str) and b64.startswith('data:image/'):
        return b64.split(',', 1)[1]
    return b64


def decode_image(b64: str) -> np.ndarray:
    """Decode base64 to BGR image (cv2)."""
    if not isinstance(b64, str) or not b64:
        raise ValueError('image must be a non-empty base64 string')
    b64 = _strip_data_uri(b64)
    pad = len(b64) % 4
    if pad:
        b64 += '=' * (4 - pad)
    img_bytes = base64.b64decode(b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('failed to decode image')
    return img


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)  # x - y
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Auto Canny thresholds based on median of pixel intensities."""
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

# 预置：文本行倾斜角估计、自动纠偏和图像编码（在路由前定义，避免运行时 NameError）

def estimate_skew_angle(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny(gray)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=max(30, int(0.05 * max(h, w))), maxLineGap=10)
    angles = []
    weights = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx = x2 - x1
            dy = y2 - y1
            length = float((dx * dx + dy * dy) ** 0.5)
            if length < 15:
                continue
            ang = float(np.degrees(np.arctan2(dy, dx)))
            if ang > 90:
                ang -= 180
            if ang < -90:
                ang += 180
            if -30 <= ang <= 30:
                angles.append(ang)
                weights.append(length)
    if angles:
        return float(np.average(angles, weights=weights))
    small = cv2.resize(gray, (max(300, int(w * 0.5)), max(300, int(h * 0.5))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    def score_angle(a: float) -> float:
        M = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), -a, 1.0)
        rot = cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rot = cv2.GaussianBlur(rot, (3, 3), 0)
        _, bin_img = cv2.threshold(rot, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        proj = bin_img.mean(axis=1)
        return float(np.var(proj))
    best_a, best_s = 0.0, -1.0
    for a in np.linspace(-10.0, 10.0, 41):
        s = score_angle(a)
        if s > best_s:
            best_s, best_a = s, a
    if abs(best_a) > 0.1:
        return float(best_a)
    _, bin_img_full = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img_full = cv2.morphologyEx(bin_img_full, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    contours, _ = cv2.findContours(bin_img_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        ang = float(rect[2])
        if ang < -45:
            ang += 90
        return ang
    # Fallback: FFT 频域方向估计（参考CSDN文章思路）
    try:
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)
        _, bin_img = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 将文字置为白色以增强频域条纹
        if bin_img.mean() > 127:
            bin_img = 255 - bin_img
        f = np.fft.fft2(bin_img)
        fshift = np.fft.fftshift(f)
        mag = np.log(np.abs(fshift) + 1.0)
        mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)
        h2, w2 = mag.shape
        cv2.circle(mag, (w2 // 2, h2 // 2), int(min(w2, h2) * 0.06), 0, -1)
        edges2 = auto_canny(mag)
        lines2 = cv2.HoughLines(edges2, 1, np.pi / 180, threshold=80)
        thetas = []
        if lines2 is not None:
            for rho, theta in lines2[:, 0, :]:
                ang = float(theta * 180.0 / np.pi)
                # 将角度映射到[-90,90]
                while ang > 90:
                    ang -= 180
                while ang < -90:
                    ang += 180
                # 频域条纹与文本行方向垂直，故纠偏角应接近ang-90或ang+90附近的微小偏差
                # 取接近水平的分量，映射到[-30,30]
                if -30 <= ang <= 30:
                    thetas.append(ang)
        if thetas:
            return float(np.median(thetas))
    except Exception:
        pass
    return 0.0


def deskew_image(img: np.ndarray):
    ang = float(estimate_skew_angle(img))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -ang, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, ang


def encode_image(img: np.ndarray, fmt: str = '.jpg', quality: int = 90) -> str:
    if img is None:
        raise ValueError('image is None')
    ext = (fmt or '.jpg').lower()
    if ext not in ('.jpg', '.jpeg', '.png'):
        ext = '.jpg'
    params = []
    if ext in ('.jpg', '.jpeg'):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif ext == '.png':
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError('failed to encode image')
    return base64.b64encode(buf).decode('ascii')


def is_border_rect(pts: np.ndarray, img_shape) -> bool:
    """Return True if rectangle lies too close to image border (likely whole-image box)."""
    h, w = img_shape[:2]
    x_min = pts[:, 0].min()
    x_max = pts[:, 0].max()
    y_min = pts[:, 1].min()
    y_max = pts[:, 1].max()
    margin_x = w * 0.005
    margin_y = h * 0.005
    near_left = x_min <= margin_x
    near_right = (w - x_max) <= margin_x
    near_top = y_min <= margin_y
    near_bottom = (h - y_max) <= margin_y
    ptsf = pts.astype(np.float32)
    rect = order_points(ptsf)
    area_img = float(h * w)
    area_rect = float(cv2.contourArea(rect))
    too_large = (area_rect / (area_img + 1e-6)) > 0.99
    return (near_left and near_right and near_top and near_bottom) or too_large

# 新增：矩形度评分（角度接近90°、对边平行度、对边长度平衡）
def rectangle_score(pts: np.ndarray) -> float:
    pts = pts.astype(np.float32)
    # 保证顺序一致
    rect = order_points(pts)
    # 边向量
    v01 = rect[1] - rect[0]
    v12 = rect[2] - rect[1]
    v23 = rect[3] - rect[2]
    v30 = rect[0] - rect[3]
    def norm(v):
        n = np.linalg.norm(v) + 1e-6
        return v / n, n
    u01, l01 = norm(v01)
    u12, l12 = norm(v12)
    u23, l23 = norm(v23)
    u30, l30 = norm(v30)
    # 角度接近90°（四角）
    def angle_score(a, b):
        cosang = np.clip(np.dot(a, b), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        return max(0.0, 1.0 - abs(ang - 90.0) / 90.0)
    s_ang = (angle_score(u01, -u30) + angle_score(u01, u12) + angle_score(u23, -u12) + angle_score(u23, u30)) / 4.0
    # 对边平行度（两组）
    s_par = (abs(np.dot(u01, -u23)) + abs(np.dot(u12, -u30))) / 2.0  # 平行时接近1
    # 对边长度平衡
    s_len = (1.0 - abs(l01 - l23) / (max(l01, l23) + 1e-6) + 1.0 - abs(l12 - l30) / (max(l12, l30) + 1e-6)) / 2.0
    # 综合评分
    return float((s_ang * 0.5) + (s_par * 0.3) + (s_len * 0.2))

# 新增：角点子像素精修

def refine_corners(gray: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype(np.float32))
    # cornerSubPix 需要浮点灰度图
    g = gray.astype(np.float32)
    # 初始角点
    corners = rect.reshape(-1, 1, 2).astype(np.float32)
    # 搜索窗口与终止条件
    win_size = (5, 5)
    zero_zone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    try:
        cv2.cornerSubPix(g, corners, win_size, zero_zone, criteria)
        rect_refined = corners.reshape(4, 2)
        return rect_refined.astype(np.float32)
    except Exception:
        return rect

# 新增：在原图上绘制识别到的文档边界

def draw_boundary_overlay(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    vis = img.copy()
    rect = order_points(pts.astype(np.float32)).astype(int)
    # 多边形
    cv2.polylines(vis, [rect], isClosed=True, color=(0, 255, 0), thickness=3)
    # 角点
    for i, (x, y) in enumerate(rect):
        cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(vis, str(i+1), (int(x)+8, int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return vis


# 新增：直线几何与交点求解

def _line_from_points(p1: np.ndarray, p2: np.ndarray):
    # 返回标准直线参数 ax + by + c = 0
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = (a**2 + b**2) ** 0.5 + 1e-6
    return a / norm, b / norm, c / norm


def _intersection(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / d
    y = (c1 * a2 - c2 * a1) / d
    return np.array([x, y], dtype=np.float32)


def _segment_midpoint(seg):
    x1, y1, x2, y2 = seg
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _segment_angle(seg):
    x1, y1, x2, y2 = seg
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))  # [-180,180]
    ang = (ang + 180) % 180  # [0,180]
    return ang


def edge_contrast_score(gray: np.ndarray, rect: np.ndarray) -> float:
    rect = order_points(rect.astype(np.float32))
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = rect.astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    border = cv2.bitwise_xor(mask, inner)
    edges = auto_canny(gray)
    border_mean = float(edges[border > 0].mean() / 255.0) if np.any(border) else 0.0
    inner_mean = float(edges[inner > 0].mean() / 255.0) if np.any(inner) else 0.0
    s = max(0.0, border_mean - inner_mean)
    return s

# 安全导入 rembg；不可用时优雅回退
try:
    from rembg import remove, new_session
    try:
        _rembg_session = new_session('u2net')
        _rembg_model = 'u2net'
    except Exception:
        _rembg_session = new_session('u2netp')
        _rembg_model = 'u2netp'
except Exception:
    remove = None
    _rembg_session = None
    _rembg_model = None


def find_document_corners_rembg(img: np.ndarray):
    if remove is None or _rembg_session is None:
        return None, {'reason': 'rembg unavailable'}
    h, w = img.shape[:2]
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ok, png_buf = cv2.imencode('.png', rgb)
        if not ok:
            return None, {'strategy': 'saliency-rembg', 'reason': 'png encode failed'}
        mask_bytes = remove(png_buf.tobytes(), session=_rembg_session, only_mask=True)
        mask_arr = np.frombuffer(mask_bytes, np.uint8)
        mask_img = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            return None, {'strategy': 'saliency-rembg', 'reason': 'mask decode failed'}
        _, th = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        diag = {'strategy': 'saliency-rembg', 'contours_checked': int(len(contours)), 'candidates': []}
        min_area = (h * w) * 0.05
        best_rect, best_score = None, -1.0
        for cnt in contours[:5]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.01, 0.02, 0.03, 0.04):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                diag['candidates'].append({'area': float(area), 'pts': len(approx), 'eps': eps})
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                else:
                    box = cv2.boxPoints(cv2.minAreaRect(cnt))
                    pts = box.astype(np.float32)
                if is_border_rect(pts, img.shape):
                    continue
                s = rectangle_score(pts)
                if s > best_score:
                    best_rect = order_points(pts)
                    best_score = s
        if best_rect is None:
            diag['reason'] = 'no quad from mask'
            return None, diag
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        best_rect = refine_corners(gray, best_rect)
        diag['best_score'] = float(best_score)
        return best_rect, diag
    except Exception as e:
        return None, {'strategy': 'saliency-rembg', 'reason': str(e)}

def find_document_corners_color_mask(img: np.ndarray):
    """基于颜色亮度掩膜（白纸特征：高亮、低饱和）提取外轮廓并拟合四边形。"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    # 自适应阈值：按图像分布设定低饱和 + 高亮的界线，更稳健
    sat_thr = int(min(80, max(20, np.percentile(S, 40))))
    val_thr = int(min(230, max(140, np.percentile(V, 60))))
    mask = cv2.inRange(hsv, (0, 0, val_thr), (179, sat_thr, 255))
    # 后处理：开闭 + 最大连通域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    diag = {'strategy': 'color-mask', 'contours_checked': int(len(contours)), 'candidates': []}
    min_area = (h * w) * 0.05
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best_rect, best_score = None, -1.0
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps in (0.01, 0.02, 0.03, 0.04):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            diag['candidates'].append({'area': float(area), 'pts': len(approx), 'eps': eps})
            if len(approx) != 4:
                box = cv2.boxPoints(cv2.minAreaRect(cnt))
                pts = box.astype(np.float32)
            else:
                pts = approx.reshape(4, 2).astype(np.float32)
            if is_border_rect(pts, img.shape):
                continue
            rect_o = order_points(pts)
            s_geom = rectangle_score(rect_o)
            s_edge = edge_contrast_score(gray, rect_o)
            s = (s_geom * 0.6) + (s_edge * 0.4)
            if s > best_score:
                best_rect = rect_o
                best_score = s
    if best_rect is None:
        diag['reason'] = 'no quad from color mask'
        return None, diag
    best_rect = refine_corners(gray, best_rect)
    diag['best_score'] = float(best_score)
    return best_rect, diag

def find_document_corners_hough(img: np.ndarray):
    """基于霍夫直线的文档边界检测。返回 (rect, diag_update) 或 (None, {})."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    min_len = max(40, int(0.10 * max(h, w)))
    max_gap = max(12, int(0.03 * max(h, w)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=min_len, maxLineGap=max_gap)
    diag = {
        'hough_lines_checked': 0,
        'hough_groups': {'horizontal': 0, 'vertical': 0},
        'strategy': None,
        'best_score': None,
    }
    if lines is None or len(lines) == 0:
        return None, diag
    lines = lines.reshape(-1, 4)
    diag['hough_lines_checked'] = int(len(lines))
    horiz, vert = [], []
    for seg in lines:
        ang = _segment_angle(seg)
        if ang < 20 or ang > 160:
            horiz.append(seg)
        elif 70 <= ang <= 110:
            vert.append(seg)
    diag['hough_groups']['horizontal'] = int(len(horiz))
    diag['hough_groups']['vertical'] = int(len(vert))
    if len(horiz) < 2 or len(vert) < 2:
        return None, diag
    horiz_sorted = sorted(horiz, key=lambda s: _segment_midpoint(s)[1])
    top_seg = horiz_sorted[0]
    bot_seg = horiz_sorted[-1]
    vert_sorted = sorted(vert, key=lambda s: _segment_midpoint(s)[0])
    left_seg = vert_sorted[0]
    right_seg = vert_sorted[-1]
    l_top = _line_from_points(np.array(top_seg[:2]), np.array(top_seg[2:]))
    l_bot = _line_from_points(np.array(bot_seg[:2]), np.array(bot_seg[2:]))
    l_left = _line_from_points(np.array(left_seg[:2]), np.array(left_seg[2:]))
    l_right = _line_from_points(np.array(right_seg[:2]), np.array(right_seg[2:]))
    tl = _intersection(l_top, l_left)
    tr = _intersection(l_top, l_right)
    br = _intersection(l_bot, l_right)
    bl = _intersection(l_bot, l_left)
    pts = [tl, tr, br, bl]
    if any(p is None for p in pts):
        return None, diag
    rect = order_points(np.vstack(pts))
    area = cv2.contourArea(rect.astype(np.int32))
    min_area = (h * w) * 0.02
    if area < min_area or is_border_rect(rect, img.shape):
        return None, diag
    s_geom = rectangle_score(rect)
    s_edge = edge_contrast_score(gray, rect)
    s = (s_geom * 0.6) + (s_edge * 0.4)
    if s < 0.45:
        return None, diag
    diag['strategy'] = 'hough-lines'
    diag['best_score'] = float(s)
    rect_refined = refine_corners(gray, rect)
    return rect_refined, diag

def find_document_corners_adaptive(img: np.ndarray):
    """基于自适应阈值/OTSU + 形态学的鲁棒四边形提取。
    - 在较小分辨率上做阈值与轮廓，降低噪声影响；
    - 使用凸包 + 多边形拟合/最小外接矩形作为候选；
    - 结合几何矩形度与边缘对比度评分选优；
    返回 (4x2 float32 或 None, 诊断信息)。
    """
    try:
        h, w = img.shape[:2]
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 降采样做粗分割
        max_side = max(h, w)
        scale = 1.0
        small = img
        gray = gray_full
        if max_side > 1200:
            scale = 1200.0 / float(max_side)
            sw = int(round(w * scale))
            sh = int(round(h * scale))
            small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # OTSU 阈值 + 形态学
        gb = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 反相，让主体为白（255）以便取最大连通域
        mask = cv2.bitwise_not(th)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        diag = {'strategy': 'adaptive-threshold', 'contours_checked': int(len(contours)), 'candidates': []}
        if not contours:
            diag['reason'] = 'no contours'
            return None, diag
        sh, sw = small.shape[:2]
        min_area = (sh * sw) * 0.05
        best_rect, best_score = None, -1.0
        # 仅考察前5大候选
        for cnt in contours[:5]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
            else:
                box = cv2.boxPoints(cv2.minAreaRect(hull))
                pts = box.astype(np.float32)
            if is_border_rect(pts, small.shape):
                continue
            s_geom = rectangle_score(pts)
            s_edge = edge_contrast_score(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), order_points(pts))
            s = (s_geom * 0.6) + (s_edge * 0.4)
            diag['candidates'].append({'area': float(area), 'pts': int(len(approx)), 'score': float(s)})
            if s > best_score:
                best_score = s
                best_rect = order_points(pts)
        if best_rect is None:
            diag['reason'] = 'no quad'
            return None, diag
        # 映射回原始分辨率
        rect_orig = (best_rect / scale).astype(np.float32)
        rect_refined = refine_corners(gray_full, rect_orig)
        diag['best_score'] = float(best_score)
        return rect_refined, diag
    except Exception as e:
        return None, {'strategy': 'adaptive-threshold', 'reason': str(e)}

# 新增：基于 Canny + 最大外轮廓 的四角检测（与文章/示例代码一致）
def find_document_corners_canny_largest(img: np.ndarray):
    """Canny 边缘 -> 最大外轮廓 -> 多边形近似 -> 四点排序；不满4点时回退 minAreaRect。
    返回 (rect 4x2 float32, diagnostics)。
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny（自适应阈值） + 闭操作稳定外轮廓
    edges = auto_canny(gray_blur)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # 回退：Otsu 二值 + 闭操作
        _, bin_img = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, {'strategy': 'canny-largest', 'reason': 'no external contours'}

    # 选择最大轮廓
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    diag = {'strategy': None, 'contours_checked': int(len(contours)), 'candidate_area': float(cv2.contourArea(largest))}

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        if is_border_rect(pts, img.shape):
            return None, {'strategy': 'canny-largest', 'reason': 'border-like rectangle'}
        rect = order_points(pts)
        rect = refine_corners(gray, rect)
        diag['strategy'] = 'canny-largest-approx4'
        diag['best_score'] = float(rectangle_score(rect))
        return rect, diag
    else:
        # 回退：最小外接矩形
        box = cv2.boxPoints(cv2.minAreaRect(largest))
        pts = box.astype(np.float32)
        if is_border_rect(pts, img.shape):
            return None, {'strategy': 'canny-largest', 'reason': 'border-like minAreaRect'}
        rect = order_points(pts)
        rect = refine_corners(gray, rect)
        diag['strategy'] = 'canny-largest-minAreaRect'
        diag['best_score'] = float(rectangle_score(rect))
        return rect, diag

def find_document_corners(img: np.ndarray):
    """Find document corners with robust multi-strategy approach.
    Returns (points 4x2 float32 or None, diagnostics dict).
    """
    diag = {
        'strategy': None,
        'contours_checked': 0,
        'candidates': [],
        'reason': ''
    }
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 0) 先走 Canny + 最大外轮廓（与文章推荐路径一致，优先级最高）
    rect0, diag0 = find_document_corners_canny_largest(img)
    if rect0 is not None:
        diag.update(diag0)
        return rect0, diag

    # 1) U²Netp 显著性（rembg）
    rect_r, diag_r = find_document_corners_rembg(img)
    if rect_r is not None:
        diag.update(diag_r)
        return rect_r, diag

    # 2) 颜色亮度掩膜（白纸）
    rect_c, diag_c = find_document_corners_color_mask(img)
    if rect_c is not None:
        diag.update(diag_c)
        return rect_c, diag

    # 3) 自适应阈值 + 形态学
    rect_a, diag_a = find_document_corners_adaptive(img)
    if rect_a is not None:
        diag.update(diag_a)
        return rect_a, diag

    # 4) 霍夫直线
    rect_h, diag_h = find_document_corners_hough(img)
    if rect_h is not None:
        diag.update(diag_h)
        return rect_h, diag

    # Strategy 1: auto canny + external contours
    edges = auto_canny(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    diag['contours_checked'] = len(contours)
    h, w = img.shape[:2]
    min_area = (h * w) * 0.02
    def try_approx(cnt):
        peri = cv2.arcLength(cnt, True)
        best = None
        best_s = -1.0
        for eps_ratio in (0.01, 0.02, 0.03, 0.04):
            approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True)
            area = float(cv2.contourArea(cnt))
            diag['candidates'].append({'area': area, 'pts': len(approx), 'eps': eps_ratio})
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                if is_border_rect(pts, img.shape):
                    continue
                s = rectangle_score(pts)
                if s > best_s:
                    best = order_points(pts)
                    best_s = s
        return best, best_s
    best_rect = None
    best_score = -1.0
    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect, s = try_approx(cnt)
        if rect is not None and s > best_score:
            best_rect = rect
            best_score = s
    if best_rect is not None:
        diag['strategy'] = 'canny-external-approx'
        diag['best_score'] = float(best_score)
        best_rect = refine_corners(gray, best_rect)
        return best_rect, diag
    # Strategy 2: Otsu threshold -> external contours
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    best_rect = None
    best_score = -1.0
    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect, s = try_approx(cnt)
        if rect is not None and s > best_score:
            best_rect = rect
            best_score = s
    if best_rect is not None:
        diag['strategy'] = 'otsu-external-approx'
        diag['best_score'] = float(best_score)
        best_rect = refine_corners(gray, best_rect)
        return best_rect, diag
    # Strategy 3: minAreaRect fallback
    best_rect = None
    best_score = -1.0
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        pts = box.astype(np.float32)
        if is_border_rect(pts, img.shape):
            continue
        rect = order_points(pts)
        s = rectangle_score(rect)
        if s > best_score:
            best_rect = rect
            best_score = s
    if best_rect is not None:
        diag['strategy'] = 'minAreaRect'
        diag['best_score'] = float(best_score)
        best_rect = refine_corners(gray, best_rect)
        return best_rect, diag
    # 终极兜底
    edges2 = auto_canny(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours_e, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_o, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_cnts = sorted(list(contours_e) + list(contours_o), key=cv2.contourArea, reverse=True)
    best_rect = None
    best_score = -1.0
    for cnt in all_cnts[:10]:
        area = cv2.contourArea(cnt)
        if area < (h * w) * 0.01:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        pts = box.astype(np.float32)
        if is_border_rect(pts, img.shape):
            continue
        rect = order_points(pts)
        s = rectangle_score(rect)
        if s > best_score:
            best_rect = rect
            best_score = s
    if best_rect is not None:
        diag['strategy'] = 'ultimate-minAreaRect'
        diag['best_score'] = float(best_score)
        best_rect = refine_corners(gray, best_rect)
        return best_rect, diag
    diag['reason'] = 'no suitable quadrilateral found'
    return None, diag


def warp_perspective(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp image using four corner points (tl, tr, br, bl)."""
    rect = order_points(pts.astype(np.float32))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    maxW = max(10, maxW)
    maxH = max(10, maxH)

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))
    return warped


@app.route('/api/perspective', methods=['POST'])
def api_perspective():
    data = request.json or {}
    if 'image' not in data:
        return jsonify({'error': 'missing image'}), 400
    try:
        img = decode_image(data['image'])
        user_points = data.get('points')
        used_points = None
        mode = 'auto'
        diag = {}
        if user_points and isinstance(user_points, list) and len(user_points) == 4:
            pts = np.array([[float(p['x']), float(p['y'])] for p in user_points], dtype=np.float32)
            used_points = pts.tolist()
            mode = 'manual'
        else:
            pts, diag = find_document_corners(img)
            if pts is None:
                # 自动纠偏兜底：即使没找到边框，也返回水平化结果
                try:
                    rotated, ang = deskew_image(img)
                except Exception:
                    ang = 0.0
                    rotated = img.copy()
                b64 = encode_image(rotated, fmt='.jpg')
                overlay_b64 = b64
                diag['strategy'] = diag.get('strategy') or 'deskew-only'
                diag['skew_angle'] = float(ang)
                return jsonify({
                    'success': True,
                    'image': 'data:image/jpeg;base64,' + b64,
                    'overlay': 'data:image/jpeg;base64,' + overlay_b64,
                    'points': [],
                    'mode': 'deskew-only',
                    'diagnostics': diag,
                    'size': {'width': int(rotated.shape[1]), 'height': int(rotated.shape[0])}
                })
            used_points = pts.tolist()
            if diag.get('strategy'):
                mode = diag['strategy']
        # 透视矫正结果
        warped = warp_perspective(img, np.array(used_points, dtype=np.float32))
        b64 = encode_image(warped, fmt='.jpg')
        # 原图叠加边界
        overlay_img = draw_boundary_overlay(img, np.array(used_points, dtype=np.float32))
        overlay_b64 = encode_image(overlay_img, fmt='.jpg')
        return jsonify({
            'success': True,
            'image': 'data:image/jpeg;base64,' + b64,
            'overlay': 'data:image/jpeg;base64,' + overlay_b64,
            'points': used_points,
            'mode': mode,
            'diagnostics': diag,
            'size': {'width': int(warped.shape[1]), 'height': int(warped.shape[0])}
        })
    except Exception as e:
        return jsonify({'error': f'perspective failed: {str(e)}'}), 500


@app.route('/')
def index_page():
    """Minimal test page for perspective correction."""
    html = r'''
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>透视矫正测试</title>
<style>
body { font-family: system-ui, sans-serif; margin: 20px; }
#canvas { border: 1px solid #ccc; max-width: 100%; cursor: crosshair; }
#points { margin-top: 8px; font-size: 12px; color: #555; }
button { margin-right: 8px; }
img { max-width: 100%; border: 1px solid #eee; margin-top: 10px; }
#info { font-size: 12px; color: #555; margin-top: 8px; }
</style>
</head>
<body>
<h2>透视矫正测试</h2>
<input type="file" id="file" accept="image/*" />
<button id="autoBtn" disabled>自动检测并矫正</button>
<button id="manualBtn" disabled>按所选四点矫正</button>
<div id="points">请在图片上依次点击 四个角点（左上、右上、右下、左下）。</div>
<canvas id="canvas"></canvas>
<h3>结果</h3>
<div id="info"></div>
<img id="overlayImg" />
<img id="resultImg" />
<script>
const fileInput = document.getElementById('file');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const pointsEl = document.getElementById('points');
const autoBtn = document.getElementById('autoBtn');
const manualBtn = document.getElementById('manualBtn');
const resultImg = document.getElementById('resultImg');
const overlayImg = document.getElementById('overlayImg');
const info = document.getElementById('info');
let imgBitmap = null;
let scaleX = 1, scaleY = 1;
let points = [];

function drawPoint(x, y, i) {
  ctx.fillStyle = '#ff3b3b';
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#333';
  ctx.fillText(String(i+1), x + 6, y - 6);
}

fileInput.addEventListener('change', async () => {
  const f = fileInput.files[0];
  if (!f) return;
  const blobUrl = URL.createObjectURL(f);
  imgBitmap = await createImageBitmap(f);
  const maxW = 900;
  const ratio = imgBitmap.width > maxW ? maxW / imgBitmap.width : 1;
  canvas.width = Math.round(imgBitmap.width * ratio);
  canvas.height = Math.round(imgBitmap.height * ratio);
  scaleX = imgBitmap.width / canvas.width;
  scaleY = imgBitmap.height / canvas.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imgBitmap, 0, 0, canvas.width, canvas.height);
  points = [];
  pointsEl.textContent = '请在图片上依次点击 四个角点（左上、右上、右下、左下）。';
  info.textContent = '';
  autoBtn.disabled = false;
  manualBtn.disabled = true;
  overlayImg.src = '';
  resultImg.src = '';
});

canvas.addEventListener('click', (e) => {
  if (!imgBitmap) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  drawPoint(cx, cy, points.length);
  points.push({ x: Math.round(cx * scaleX), y: Math.round(cy * scaleY) });
  pointsEl.textContent = '已选择 ' + points.length + ' 个点';
  if (points.length === 4) {
    manualBtn.disabled = false;
  }
});

async function postPerspective(payload) {
  const res = await fetch('/api/perspective', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!data.success) { alert(data.error || data.message || 'failed'); return; }
  overlayImg.src = data.overlay;
  resultImg.src = data.image;
  const d = data.diagnostics || {}; const cands = d.candidates || [];
  info.textContent = '模式: ' + (data.mode || 'unknown') + ', 角点: ' + JSON.stringify(data.points) + ', 评分: ' + (d.best_score!==undefined?d.best_score.toFixed(3):'N/A') + ', 轮廓数: ' + (d.contours_checked||0) + ', 候选: ' + cands.length + (d.reason?(', 原因: '+d.reason):'');
}

autoBtn.addEventListener('click', async () => {
  if (!imgBitmap) return;
  const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
  await postPerspective({ image: dataUrl.split(',')[1] });
});

manualBtn.addEventListener('click', async () => {
  if (!imgBitmap || points.length !== 4) return;
  const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
  await postPerspective({ image: dataUrl.split(',')[1], points });
});
</script>
</body>
</html>
'''
    return Response(html, mimetype='text/html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'perspective-test'})


if __name__ == '__main__':
    # Run on a different port to avoid conflict with the main app
    app.run(host='0.0.0.0', port=5001, debug=True)


# 已上移至路由前：estimate_skew_angle, deskew_image, encode_image