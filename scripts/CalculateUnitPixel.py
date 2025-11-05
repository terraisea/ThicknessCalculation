import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import math

def extract_elev_by_slope(slope_vals, slope_idxs, y):
    """
    slope_vals : 一维数组或列表，坡度角（度）
    slope_idxs : 与 slope_vals 一一对应的 y 索引（原始 y 数组上的索引）
    y          : 原始高程数组（numpy array）
    threshold  : 阈值（度），返回第一个 < threshold 到最后一个 < threshold 之间的 y 值（含端点）
    返回：y[start_idx : end_idx+1]（如果无满足条件，返回空数组）
    """
    threshold = 3
    slope_vals = np.asarray(slope_vals)
    slope_idxs = np.asarray(slope_idxs, dtype=int)

    # 过滤 NaN 的坡度（若坡度里有 NaN）
    valid = ~np.isnan(slope_vals)  # 标记非NaN的有效坡度值
    cond = (slope_vals < threshold) & valid  # 同时满足:坡度值 < 3度、坡度值不是NaN

    idxs = np.where(cond)[0]  # 在 slope_vals 中的索引位置（不是 y 的索引）
    # idxs = np.where(cond) 其中是一个包含单个一维数组的元组，因此[0]可以降维成一维数组
    if idxs.size == 0:
        return np.array([])

    # 找到坡度数组中的索引，不是高程数组的索引
    first_pos = idxs[0]
    last_pos = idxs[-1]

    # 对应到 y 数组上的真实索引
    start_idx = int(slope_idxs[first_pos])
    end_idx = int(slope_idxs[last_pos])

    # 确保 start_idx <= end_idx，然后切片返回 y 的子数组
    if start_idx <= end_idx:
        return y[start_idx:end_idx + 1]
    else:
        return y[end_idx:start_idx + 1]


def compare_length_1d(a, b):
    """
    比较两个一维数组的长度。
    返回 'a>b'、'a<b' 或 'equal'
    """
    la, lb = len(a), len(b)
    c = np.hstack((a, b))
    if la > lb:
        return a
    elif la < lb:
        return b
    else:
        return c

# === 读取 DEM 元信息（保持你原来的方式） ===
with rasterio.open(r"E:\code\geo_processing\database\DEM\ZhangDEM1.tif") as src:
    transform = src.transform
    xres = abs(transform.a)  # 每像元宽度
    yres = abs(transform.e)  # 每像元高度
    lat = transform.f  # tif的纬度
    lat_rad = math.radians(lat)

x_pixel_size = abs(xres * 30334 * math.cos(lat_rad))  # 30.3300m是月球单位纬度
y_pixel_size = yres * 30334

# ========== 新增函数：单文件平均高程计算 ========== #
def calc_mean_elevation(csv_path):
    """
    输入单个剖面CSV（例如 *_Col.csv 或 *_Row.csv）
    返回 mean_elevation
    """
    data = pd.read_csv(csv_path)
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    n = len(y)
    mid = n // 2

    # 向左、右搜索最大值（保留 mid）
    left_max_idx = np.nanargmax(y[:mid + 1])  # 包含中点
    right_max_idx = mid + np.nanargmax(y[mid:])  # 从中点开始向右

    # 计算坡度时同时记录对应的 y 索引
    left_slope_vals = []
    left_slope_idxs = []

    # 左侧：从 left_max_idx 向左扫描
    for i in range(left_max_idx, 0, -1):
        slope = math.atan((y[i + 1] - y[i - 1]) / (2 * y_pixel_size))
        slope_deg = abs(math.degrees(slope))
        left_slope_vals.append(slope_deg)
        left_slope_idxs.append(i)

    # 右侧：从 right_max_idx 向右扫描
    right_slope_vals = []
    right_slope_idxs = []
    for i in range(right_max_idx, len(y) - 1):
        slope = math.atan((y[i + 1] - y[i - 1]) / (2 * y_pixel_size))
        slope_deg = abs(math.degrees(slope))
        right_slope_vals.append(slope_deg)
        right_slope_idxs.append(i)

    # 用函数提取平缓区高程
    left_flat_y = extract_elev_by_slope(left_slope_vals, left_slope_idxs, y)
    right_flat_y = extract_elev_by_slope(right_slope_vals, right_slope_idxs, y)

    # 合并最长的平缓段并求平均高程
    flat_y = compare_length_1d(left_flat_y, right_flat_y)
    if len(flat_y) == 0:
        return np.nan  # 避免空数组除0错误
    mean_elevation = np.nanmean(flat_y)
    return mean_elevation

# ========== 自动识别 Col 与 Row CSV 并分别计算 ========== #
base_dir = Path(r"E:\code\geo_processing\database\testdata\mask\show2")
prefix = "0.61232_dem"  # 公共前缀

col_path = base_dir / f"{prefix}_Col.csv"
row_path = base_dir / f"{prefix}_Row.csv"

mean_col = calc_mean_elevation(col_path)
print(f"{col_path.name} 的平均高程: {mean_col}")
mean_row = calc_mean_elevation(row_path)
print(f"{row_path.name} 的平均高程: {mean_row}")

