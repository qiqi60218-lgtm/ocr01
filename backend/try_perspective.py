import argparse
import os
import sys
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def detect_and_correct_perspective(input_path: str, output_path: str, show: bool = False) -> bool:
    image = cv2.imread(input_path)
    if image is None:
        print(f"无法加载图像，请检查路径: {os.path.abspath(input_path)}")
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测（自适应阈值版本的 Canny）
    edges = auto_canny(blurred)

    # 形态学闭操作，连接边缘，增强外轮廓稳定性
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("未检测到外轮廓，尝试 Otsu 二值回退...")
        _, bin_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("未检测到四边形轮廓，无法进行透视变换")
            return False

    largest_contour = max(contours, key=cv2.contourArea)

    # 多边形近似
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 可视化（叠加）
    vis = image.copy()
    cv2.drawContours(vis, [largest_contour], -1, (0, 255, 0), 2)
    cv2.drawContours(vis, [approx], -1, (0, 0, 255), 2)

    # 组装四边形顶点；不满足 4 点时用最小外接矩形回退
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype("float32")
    else:
        box = cv2.boxPoints(cv2.minAreaRect(largest_contour))
        pts = box.astype("float32")

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    maxWidth = max(10, maxWidth)
    maxHeight = max(10, maxHeight)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ok = cv2.imwrite(output_path, warped)
    if ok:
        print(f"结果已保存到: {os.path.abspath(output_path)}")
    else:
        print("保存失败，请检查输出路径是否有效")

    if show:
        cv2.imshow("Original + Contours", vis)
        cv2.imshow("Perspective Corrected", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ok


def main():
    parser = argparse.ArgumentParser(description="文档自动透视矫正（简版）")
    parser.add_argument("--input", required=True, help="输入图像路径")
    parser.add_argument("--output", required=True, help="输出图像路径")
    parser.add_argument("--show", action="store_true", help="是否显示窗口预览")
    args = parser.parse_args()

    success = detect_and_correct_perspective(args.input, args.output, show=args.show)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()