import cv2
import numpy as np
import math

# ----- 配置 -----
NUM_CALIPERS_DEFAULT = 10  # 默认卡尺数量
CALIPER_WIDTH_DEFAULT = 100  # 默认卡尺宽度，None 表示自动计算
CALIPER_LENGTH_DEFAULT = 50  # 默认卡尺长度，None 表示自动计算
#image_path = 'Pic_2023_07_30_143321_4.bmp'
image_path = 'Pic_2023_07_30_143321_4.bmp'
image_path = 'Pic_2023_07_30_143321_4.bmp'
#image_path = 'Pic_2023_07_30_143321_4.bmp'
# 读入图片，替换成你的路径
image_path = 'Pic_2023_07_30_143321_4.bmp'
img = cv2.imread(image_path)
if img is None:
    raise RuntimeError(f'Cannot load image: {image_path}')
img_original = img.copy()

# ----- 状态变量 -----
start_pt = None
end_pt = None
num_calipers = NUM_CALIPERS_DEFAULT
caliper_width = CALIPER_WIDTH_DEFAULT
caliper_length = CALIPER_LENGTH_DEFAULT

import cv2
import numpy as np
import math


def draw_circular_calipers(img, p1, p2, angle_range=(0, 360), num=50, width=50, length=300, scale=5):
    center = p1
    radius = np.linalg.norm(np.array(p1)-np.array(p2))
    img_draw = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    cx, cy = center

    # 角度设置
    start_angle, end_angle = np.deg2rad(angle_range[0]), np.deg2rad(angle_range[1])
    total_angle = end_angle - start_angle

    # 默认卡尺宽度和长度
    if width is None:
        width = radius * total_angle / num  # 沿圆弧方向分段
    if length is None:
        length = width * 3  # 径向长度

    for i in range(num):
        theta = start_angle + (i + 0.5) / num * total_angle  # 当前角度

        # 圆弧上卡尺中心位置
        arc_x = cx + radius * np.cos(theta)
        arc_y = cy + radius * np.sin(theta)

        # 方向向量：沿径向（指向外侧）
        ux = np.cos(theta)
        uy = np.sin(theta)

        # 垂直方向：沿圆弧方向（逆时针切线方向）
        vx = -uy
        vy = ux

        # 矩形四个点
        ptA = (arc_x + vx * width / 2 - ux * length / 2,
               arc_y + vy * width / 2 - uy * length / 2)
        ptB = (arc_x - vx * width / 2 - ux * length / 2,
               arc_y - vy * width / 2 - uy * length / 2)
        ptC = (arc_x - vx * width / 2 + ux * length / 2,
               arc_y - vy * width / 2 + uy * length / 2)
        ptD = (arc_x + vx * width / 2 + ux * length / 2,
               arc_y + vy * width / 2 + uy * length / 2)

        # 绘制卡尺矩形
        pts = np.array([[ptA, ptB, ptC, ptD]], dtype=np.float32) * scale
        pts_int = np.round(pts).astype(np.int32)
        cv2.polylines(img_draw, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3 * scale, lineType=cv2.LINE_AA)

    # 绘制原始圆弧范围
    cv2.circle(img_draw, (int(cx * scale), int(cy * scale)), int(radius * scale), (255, 0, 0), 1 * scale,
               lineType=cv2.LINE_AA)

    # 缩放回原始尺寸
    img_draw = cv2.resize(img_draw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    return img_draw


from scipy.ndimage import gaussian_filter1d

def extract_edge_along_line(gray, pt1, pt2, gradient_thresh=10, gaussian_sigma=1):
    """
    沿着一条线提取灰度并计算边缘点

    参数:
        gray: 输入灰度图
        pt1, pt2: 线段起点、终点 (x, y)
        gradient_thresh: 阈值，筛选强边缘
        gaussian_sigma: 高斯平滑的σ（可设为0关闭）

    返回:
        (x, y) 边缘点，或 None（找不到）
    """
    x1, y1 = pt1
    x2, y2 = pt2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length < 2:
        return None

    # 在 pt1 到 pt2 间插值出所有整数像素点
    x_vals = np.linspace(x1, x2, length)
    y_vals = np.linspace(y1, y2, length)

    # 提取灰度值
    gray_values = []
    coords = []
    h, w = gray.shape
    for x, y in zip(x_vals, y_vals):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            gray_values.append(gray[yi, xi])
            coords.append((xi, yi))

    if len(gray_values) < 3:
        return None

    gray_values = np.array(gray_values, dtype=np.float32)
    print(gray_values)
    # 可选的高斯滤波
    if gaussian_sigma > 0:
        gray_values = gaussian_filter1d(gray_values, sigma=gaussian_sigma)

    # 一阶差分
    gradient = np.diff(gray_values)
    max_idx = np.argmax(np.abs(gradient))

    if abs(gradient[max_idx]) >= gradient_thresh:
        return coords[max_idx]
    else:
        return None


def draw_calipers(img, p1, p2, num=NUM_CALIPERS_DEFAULT, width=None, length=None, scale=5):
    img_draw = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    line_len = math.hypot(dx, dy)
    if line_len == 0:
        return img_draw

    # 单位向量
    ux = dx / line_len
    uy = dy / line_len
    vx = -uy
    vy = ux

    # 默认设置
    if width is None:
        width = line_len / num
    if length is None:
        length = width * 3

    # 调整为最大长度不超过直线长度
    total_width = width * num
    if total_width > line_len:
        width = line_len / num  # 自动压缩为可重叠
        total_width = width * num

    # 计算两条共线导轨
    rail1_offset = length / 2
    rail2_offset = -length / 2
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(num):
        center_x = p1[0] + (i + 0.5) / num * dx
        center_y = p1[1] + (i + 0.5) / num * dy

        # 在导轨1和导轨2上，找左右顶点
        ptA = (center_x + vx * rail1_offset - ux * width / 2,
               center_y + vy * rail1_offset - uy * width / 2)
        ptB = (center_x + vx * rail1_offset + ux * width / 2,
               center_y + vy * rail1_offset + uy * width / 2)
        ptC = (center_x + vx * rail2_offset + ux * width / 2,
               center_y + vy * rail2_offset + uy * width / 2)
        ptD = (center_x + vx * rail2_offset - ux * width / 2,
               center_y + vy * rail2_offset - uy * width / 2)
        mid_ab = ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)
        mid_cd = ((ptC[0] + ptD[0]) / 2, (ptC[1] + ptD[1]) / 2)
        print("卡尺中心点（AB 中点）: ", mid_ab)
        print("卡尺底部中心点（CD 中点）: ", mid_cd)
        print(extract_edge_along_line(gray,mid_cd,mid_ab))
        # 乘以缩放比例并四舍五入，转成整数
        mid_ab= tuple(np.round(np.array(mid_ab) * scale).astype(int))
        mid_cd = tuple(np.round(np.array(mid_cd) * scale).astype(int))
        cv2.line(img_draw, mid_ab, mid_cd, (255, 0, 255), 1, lineType=cv2.LINE_AA)
        # 绘制卡尺矩形
        pts = np.array([[ptA, ptB, ptC, ptD]], dtype=np.float32) * scale
        pts_int = np.round(pts).astype(np.int32)
        cv2.polylines(img_draw, [pts_int], isClosed=True, color=(0, 255, 0), thickness=1 * scale, lineType=cv2.LINE_AA)

    # 绘制原始直线
    cv2.arrowedLine(img_draw, (int(round(p1[0] * scale)), int(round(p1[1] * scale))),
                    (int(round(p2[0] * scale)), int(round(p2[1] * scale))), (0, 0, 255), 2 * scale, tipLength=0.02)
    img_draw = cv2.resize(img_draw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    return img_draw


def mouse_callback(event, x, y, flags, param):
    global start_pt, end_pt, img

    if event == cv2.EVENT_LBUTTONDOWN:
        if start_pt is None:
            start_pt = (x, y)
        elif end_pt is None:
            end_pt = (x, y)
            # 画图
            #img = draw_calipers(img_original, start_pt, end_pt, num_calipers, caliper_width, caliper_length)
            img=draw_circular_calipers(img_original,start_pt,end_pt)
        else:
            # 重置开始点，清除结束点和图像
            start_pt = (x, y)
            end_pt = None
            img = img_original.copy()


cv2.namedWindow('Caliper Tool',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Caliper Tool', mouse_callback)

print("鼠标左键点击两点绘制直线，ESC退出")

while True:
    cv2.imshow('Caliper Tool', img)
    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC退出
        break

cv2.destroyAllWindows()
