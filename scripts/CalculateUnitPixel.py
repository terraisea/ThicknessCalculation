import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import math
import geopandas as gpd
from shapely.geometry import Point

# ===============================================
#  坡度筛选函数
# ===============================================
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
    valid = ~np.isnan(slope_vals)   # 标记非NaN的有效坡度值
    cond = (slope_vals < threshold) & valid  # 同时满足:坡度值 < 3度、坡度值不是NaN
    idxs = np.where(cond)[0]        # 在 slope_vals 中的索引位置（不是 y 的索引）

    if idxs.size == 0:
        return np.array([])

    # 找到坡度数组中的索引，不是高程数组的索引
    first_pos = idxs[0]
    last_pos  = idxs[-1]
    # 对应到 y 数组上的真实索引
    start_idx = int(slope_idxs[first_pos])
    end_idx   = int(slope_idxs[last_pos])

    if start_idx <= end_idx:
        return y[start_idx:end_idx + 1]
    else:
        return y[end_idx:start_idx + 1]

# ===============================================
#  读取 DEM 元信息
# ===============================================
with rasterio.open(r"E:\code\geo_processing\database\marius\dem.tif") as src:
    transform = src.transform
    xres = abs(transform.a) # 每像元宽度
    yres = abs(transform.e) # 每像元高度
    lat = transform.f       # tif的纬度
    lat_rad = math.radians(lat)
    x_pixel_size = abs(xres * 30334 * math.cos(lat_rad))    # 30.3300m是月球单位纬度
    y_pixel_size = yres * 30334

# ===============================================
#  辅助函数：读取csv
# ===============================================
def read_profile(csv_path):
    data = pd.read_csv(csv_path)
    values = data["value"].to_numpy()
    lon = data["lon"].to_numpy()
    lat = data["lat"].to_numpy()
    n = len(values)
    mid = n // 2
    return values, lon, lat, n, mid

# ===============================================
#  左右最大高程计算
# ===============================================
def get_max_elevation(csv_path):
    data = pd.read_csv(csv_path)
    y = data["value"].to_numpy()
    n = len(y)
    mid = n // 2
    left_max = np.nanmax(y[:mid + 1])
    right_max = np.nanmax(y[mid:])
    return left_max, right_max

# ===============================================
#  单文件平均高程与厚度计算
# ===============================================
def calc_mean_elevation(csv_path, cf_path=None):
    x, lon, lat, n, mid = read_profile(csv_path)

    # 左半段最高点
    left_peak_idx = np.nanargmax(x[:mid + 1])
    left_peak_val = x[left_peak_idx]
    left_peak_lon = lon[left_peak_idx]
    left_peak_lat = lat[left_peak_idx]

    # 右半段最高点
    right_peak_idx = np.nanargmax(x[mid:]) + mid
    right_peak_val = x[right_peak_idx]
    right_peak_lon = lon[right_peak_idx]
    right_peak_lat = lat[right_peak_idx]

    # ---------- 左右坡度计算 ----------
    left_slope_vals, left_slope_idxs = [], []
    for i in range(left_peak_idx, 0, -1):
        slope = math.atan((x[i] - x[i - 1]) / y_pixel_size)
        slope_deg = abs(math.degrees(slope))
        left_slope_vals.append(slope_deg)
        left_slope_idxs.append(i)

    right_slope_vals, right_slope_idxs = [], []
    for i in range(right_peak_idx, len(x) - 1):
        slope = math.atan((x[i + 1] - x[i]) / y_pixel_size)
        slope_deg = abs(math.degrees(slope))
        right_slope_vals.append(slope_deg)
        right_slope_idxs.append(i)

    # ---------- 提取平缓区 ----------
    left_flat_y  = extract_elev_by_slope(left_slope_vals, left_slope_idxs, x)
    right_flat_y = extract_elev_by_slope(right_slope_vals, right_slope_idxs, x)

    mean_left  = np.nanmean(left_flat_y)  if len(left_flat_y) > 0 else np.nan
    mean_right = np.nanmean(right_flat_y) if len(right_flat_y) > 0 else np.nan

    # ---------- 计算 h1 ----------
    h1_left  = left_peak_val  - mean_left   if not np.isnan(mean_left)  else np.nan
    h1_right = right_peak_val - mean_right  if not np.isnan(mean_right) else np.nan

    # ---------- 计算 h2 ----------
    # 新要求：
    #  - 完全不检查 x 与 cf 的长度关系
    #  - 从最高点开始向内（左峰→右走；右峰→左走）
    #  - 第一处 cf < 8.2 即使用
    h2_left = np.nan
    h2_right = np.nan

    if cf_path and Path(cf_path).exists():
        cf_vals = pd.read_csv(cf_path)["value"].to_numpy()

        # 左边：从左峰顶往右
        for idx in range(left_peak_idx, len(cf_vals)):
            if cf_vals[idx] < 8.2:
                h2_left = left_peak_val - x[idx]
                break

        # 右边：从右峰顶往左
        for idx in range(right_peak_idx, -1, -1):
            if cf_vals[idx] < 8.2:
                h2_right = right_peak_val - x[idx]
                break

    # ---------- 计算最终 h ----------
    h_left  = 0.8 * (h2_left  - 0.2 * h1_left)  if not np.isnan(h1_left)  and not np.isnan(h2_left)  else np.nan
    h_right = 0.8 * (h2_right - 0.2 * h1_right) if not np.isnan(h1_right) and not np.isnan(h2_right) else np.nan

    return {
        "left_peak": (left_peak_val, left_peak_lon, left_peak_lat),
        "right_peak": (right_peak_val, right_peak_lon, right_peak_lat),
        "mean_left": mean_left, "mean_right": mean_right,
        "h1_left": h1_left, "h1_right": h1_right,
        "h2_left": h2_left, "h2_right": h2_right,
        "h_left": h_left, "h_right": h_right
    }

# ===============================================
#  批处理部分
# ===============================================
shp_dir = Path(r"E:\code\geo_processing\database\marius\shp")
base_dir = Path(r"E:\code\geo_processing\database\marius\show")
output_dir = Path(r"E:\code\geo_processing\database\marius\thickness")
output_dir.mkdir(parents=True, exist_ok=True)

prefixes = [p.stem for p in shp_dir.glob("*.shp")]

crs_wkt = 'GEOGCS["unknown",DATUM["unnamed",SPHEROID["unnamed",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'

all_records = []

for prefix in prefixes:
    col_path = base_dir / f"{prefix}_DEM_Col.csv"
    row_path = base_dir / f"{prefix}_DEM_Row.csv"
    col_CF_path = base_dir / f"{prefix}_CF_Col.csv"
    row_CF_path = base_dir / f"{prefix}_CF_Row.csv"

    if not col_path.exists() or not row_path.exists():
        continue

    col_res = calc_mean_elevation(col_path, col_CF_path)
    row_res = calc_mean_elevation(row_path, row_CF_path)
    col_max = get_max_elevation(col_path)
    row_max = get_max_elevation(row_path)

    # ---------- 写入 TXT 日志 ----------
    out_txt = output_dir / f"{prefix}_thickness.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"文件前缀: {prefix}\n")
        f.write("=== 列方向 (Col) ===\n")
        for key, val in col_res.items():
            if isinstance(val, tuple):
                continue
            f.write(f"  {key:<10}: {val:.4f}\n")
        f.write(f"  left_max  : {col_max[0]:.4f}\n")
        f.write(f"  right_max : {col_max[1]:.4f}\n\n")

        f.write("=== 行方向 (Row) ===\n")
        for key, val in row_res.items():
            if isinstance(val, tuple):
                continue
            f.write(f"  {key:<10}: {val:.4f}\n")
        f.write(f"  left_max  : {row_max[0]:.4f}\n")
        f.write(f"  right_max : {row_max[1]:.4f}\n")

    print(f"✅ Saved {prefix}_thickness.txt")

    # ---------- 构建点记录 ----------
    if col_res["h_left"] >= 0:
        val, lon, lat = col_res["left_peak"]
        all_records.append({"prefix": prefix, "thickness": col_res["h_left"], "geometry": Point(lon, lat)})
    if col_res["h_right"] >= 0:
        val, lon, lat = col_res["right_peak"]
        all_records.append({"prefix": prefix, "thickness": col_res["h_right"], "geometry": Point(lon, lat)})
    if row_res["h_left"] >= 0:
        val, lon, lat = row_res["left_peak"]
        all_records.append({"prefix": prefix, "thickness": row_res["h_left"], "geometry": Point(lon, lat)})
    if row_res["h_right"] >= 0:
        val, lon, lat = row_res["right_peak"]
        all_records.append({"prefix": prefix, "thickness": row_res["h_right"], "geometry": Point(lon, lat)})

# ---------- 合并生成一个 SHP ----------
if all_records:
    gdf = gpd.GeoDataFrame(all_records, crs=crs_wkt)
    out_shp = output_dir / "all_prefixes_points1120.shp"
    gdf.to_file(out_shp)
    print(f"✅ SHP 已生成: {out_shp}")
else:
    print("⚠ 没有有效点生成 SHP")

print("🎯 所有文件处理完成！")
