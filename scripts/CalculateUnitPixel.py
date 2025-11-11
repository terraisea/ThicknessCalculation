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

# === 读取 DEM 元信息（保持你原来的方式） ===
with rasterio.open(r"E:\code\geo_processing\database\marius\dem.tif") as src:
    transform = src.transform
    xres = abs(transform.a)  # 每像元宽度
    yres = abs(transform.e)  # 每像元高度
    lat = transform.f        # tif的纬度
    lat_rad = math.radians(lat)
    x_pixel_size = abs(xres * 30334 * math.cos(lat_rad))  # 30.3300m是月球单位纬度
    y_pixel_size = yres * 30334

# ========== 新增函数：读取csv ==========
def read_profile(csv_path):
    data = pd.read_csv(csv_path)
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    n = len(y)
    mid = n // 2
    return x, y, n, mid

# ========== 修改函数：用于左右最大高程计算 ==========
def get_max_elevation(csv_path):
    """ 返回左右半段的最大高程 """
    data = pd.read_csv(csv_path)
    y = data["y"].to_numpy()
    n = len(y)
    mid = n // 2

    left_max = np.nanmax(y[:mid + 1])
    right_max = np.nanmax(y[mid:])

    print(f"{Path(csv_path).name} 左侧最大高程: {left_max}")
    print(f"{Path(csv_path).name} 右侧最大高程: {right_max}")

    return left_max, right_max

# ========== 修改函数：单文件平均高程计算 ==========
def calc_mean_elevation(csv_path, cf_path=None):
    """ 输入单个剖面CSV（例如 *_Col.csv 或 *_Row.csv）
        返回 mean_elevation, h1, h2
    """
    x, y, n, mid = read_profile(csv_path)

    # 左半段最高点
    left_peak_idx = np.nanargmax(y[:mid + 1])
    left_peak_elev = y[left_peak_idx]

    # 右半段最高点
    right_peak_idx = np.nanargmax(y[mid:]) + mid
    right_peak_elev = y[right_peak_idx]

    # 计算坡度并记录索引
    # 左侧：从左峰向左扫描
    left_slope_vals = []
    left_slope_idxs = []
    for i in range(left_peak_idx, 0, -1):
        slope = math.atan((y[i] - y[i - 1]) / y_pixel_size)
        slope_deg = abs(math.degrees(slope))
        left_slope_vals.append(slope_deg)
        left_slope_idxs.append(i)

    # 右侧：从右峰向右扫描
    right_slope_vals = []
    right_slope_idxs = []
    for i in range(right_peak_idx, len(y) - 1):
        slope = math.atan((y[i + 1] - y[i]) / y_pixel_size)
        slope_deg = abs(math.degrees(slope))
        right_slope_vals.append(slope_deg)
        right_slope_idxs.append(i)

    # 用函数提取平缓区高程（h1）
    left_flat_y = extract_elev_by_slope(left_slope_vals, left_slope_idxs, y)
    right_flat_y = extract_elev_by_slope(right_slope_vals, right_slope_idxs, y)

    # 分别计算左右平缓区平均高程
    mean_left = np.nanmean(left_flat_y) if len(left_flat_y) > 0 else np.nan
    mean_right = np.nanmean(right_flat_y) if len(right_flat_y) > 0 else np.nan

    print(f"{Path(csv_path).name} 左侧平缓区平均高程: {mean_left}")
    print(f"{Path(csv_path).name} 右侧平缓区平均高程: {mean_right}")

    # ========== 新增部分：计算 h1 ==========
    h1_left = left_peak_elev - mean_left if not np.isnan(mean_left) else np.nan
    h1_right = right_peak_elev - mean_right if not np.isnan(mean_right) else np.nan

    print(f"{Path(csv_path).name} 左侧 h1 : {h1_left}")
    print(f"{Path(csv_path).name} 右侧 h1 : {h1_right}")

    # ========== 新增部分：计算 h2 ==========
    # h2 = 峰顶高程 - cf值<8.2处的高程（按左右分组，从最外侧向中心寻找）
    h2_left = np.nan
    h2_right = np.nan
    if cf_path and Path(cf_path).exists():
        cf_data = pd.read_csv(cf_path)
        cf_vals = cf_data["y"].to_numpy()
        n_cf = len(cf_vals)
        mid_cf = n_cf // 2

        # 左侧：从最左端向中间扫描
        for i in range(0, mid_cf + 1):
            if cf_vals[i] < 8.2:
                h2_left = left_peak_elev - y[i]  # 对应 DEM 高程
                break

        # 右侧：从最右端向中间扫描
        for i in range(n_cf - 1, mid_cf - 1, -1):
            if cf_vals[i] < 8.2:
                h2_right = right_peak_elev - y[i]  # 对应 DEM 高程
                break

    print(f"{Path(csv_path).name} 左侧 h2 : {h2_left}")
    print(f"{Path(csv_path).name} 右侧 h2 : {h2_right}")


    # ========== 新增部分：计算最终 h ==========
    h_left = 0.8 * (h2_left - 0.2 * h1_left) if not np.isnan(h1_left) and not np.isnan(h2_left) else np.nan
    h_right = 0.8 * (h2_right - 0.2 * h1_right) if not np.isnan(h1_right) and not np.isnan(h2_right) else np.nan

    print(f"{Path(csv_path).name} 左侧 h : {h_left}")
    print(f"{Path(csv_path).name} 右侧 h : {h_right}")

    return mean_left, mean_right, h1_left, h1_right, h2_left, h2_right, h_left, h_right

# ========== 自动识别 Col 与 Row CSV 并分别计算 ==========
base_dir = Path(r"E:\code\geo_processing\database\marius\show")
prefix = "0.791346"  # 公共前缀

col_path = base_dir / f"{prefix}_DEM_Col.csv"
row_path = base_dir / f"{prefix}_DEM_Row.csv"
col_CF_path = base_dir / f"{prefix}_CF_Col.csv"
row_CF_path = base_dir / f"{prefix}_CF_Row.csv"

# 平均高程输出 + h1 + h2
calc_mean_elevation(col_path, col_CF_path)
calc_mean_elevation(row_path, row_CF_path)

# 最大高程分左右输出
get_max_elevation(col_path)
get_max_elevation(row_path)
