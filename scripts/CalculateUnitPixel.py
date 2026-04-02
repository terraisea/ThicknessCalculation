import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import math
import geopandas as gpd
from shapely.geometry import Point

#python scripts\CalculateUnitPixel.py --dataset-dir Database\CE5 --shp yolo
# ===============================================
#  剖面无效值清洗函数
# ===============================================
def clean_profile_values(values, invalid_low=-3e10, invalid_zero=True):
    """
    对剖面值进行清洗：
    1. 小于 invalid_low 的值视为 nodata
    2. 等于 0 的值视为无效背景值（可选）
    3. inf / -inf 也转成 nan
    """
    values = np.asarray(values, dtype=float).copy()

    # 先处理 inf
    values[~np.isfinite(values)] = np.nan

    # 处理 nodata
    values[values < invalid_low] = np.nan

    # 处理 0 值背景
    if invalid_zero:
        values[values == 0] = np.nan

    return values


# ===============================================
#  安全求最大值索引
# ===============================================
def safe_nanargmax(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return int(np.nanargmax(arr))


# ===============================================
#  月球球面两点距离（米）
# ===============================================
def moon_distance_m(lon1, lat1, lon2, lat2, radius=1737400.0):
    """
    根据经纬度计算月球表面两点距离（单位：米）
    lon/lat 输入单位：度
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


# ===============================================
#  坡度筛选函数
# ===============================================
def extract_elev_by_slope(slope_vals, slope_point_idxs, y, peak_val, threshold=8.0, min_drop=1e-6):
    """
    slope_vals       : 一维数组，坡度角（度），顺序必须是“从峰顶向外”
    slope_point_idxs : 与 slope_vals 一一对应的 y 索引，表示该坡度对应的“外侧像元”
    y                : 原始高程数组（numpy array）
    peak_val         : 峰顶高程，用于避免把峰顶平台误判为 pre-impact surface
    threshold        : 平缓区坡度阈值（度）
    min_drop         : 候选平缓区平均高程至少要低于峰顶该值，才认为是有效 surface

    返回规则：
    1. 明确排除峰顶本身，只允许使用峰顶外侧像元；
    2. 从峰顶向外搜索第一个“连续的低坡段”；
    3. 若该低坡段平均高程与峰顶几乎相同，则继续向外搜索下一段；
    4. 若没有找到合理平缓区，返回空数组。
    """
    slope_vals = np.asarray(slope_vals, dtype=float)
    slope_point_idxs = np.asarray(slope_point_idxs, dtype=int)
    y = np.asarray(y, dtype=float)

    if slope_vals.size == 0 or slope_point_idxs.size == 0:
        return np.array([])

    point_ok = (
        (slope_point_idxs >= 0) &
        (slope_point_idxs < y.size) &
        np.isfinite(y[slope_point_idxs])
    )
    slope_ok = np.isfinite(slope_vals) & (slope_vals < threshold)
    valid = point_ok & slope_ok

    i = 0
    n = len(valid)
    while i < n:
        if not valid[i]:
            i += 1
            continue

        j = i
        while j + 1 < n and valid[j + 1]:
            j += 1

        candidate_idxs = slope_point_idxs[i:j + 1]
        out = y[candidate_idxs]
        out = clean_profile_values(out, invalid_low=-3e10, invalid_zero=True)
        out = out[~np.isnan(out)]

        if out.size > 0:
            mean_out = np.nanmean(out)
            if np.isfinite(mean_out) and (mean_out < peak_val - min_drop):
                return out

        i = j + 1

    return np.array([])


# ===============================================
#  辅助函数：读取csv
# ===============================================
def read_profile(csv_path):
    data = pd.read_csv(csv_path)

    values = data["value"].to_numpy(dtype=float)
    lon = data["lon"].to_numpy(dtype=float)
    lat = data["lat"].to_numpy(dtype=float)

    # ---------- 屏蔽 0 值和 nodata ----------
    values = clean_profile_values(values, invalid_low=-3e10, invalid_zero=True)

    n = len(values)
    mid = n // 2
    return values, lon, lat, n, mid


# ===============================================
#  左右最大高程计算
# ===============================================
def get_max_elevation(csv_path):
    data = pd.read_csv(csv_path)
    y = data["value"].to_numpy(dtype=float)

    # ---------- 屏蔽 0 值和 nodata ----------
    y = clean_profile_values(y, invalid_low=-3e10, invalid_zero=True)

    n = len(y)
    mid = n // 2

    left = y[:mid + 1]
    right = y[mid:]

    left_max = np.nan if left.size == 0 or np.all(np.isnan(left)) else np.nanmax(left)
    right_max = np.nan if right.size == 0 or np.all(np.isnan(right)) else np.nanmax(right)
    return left_max, right_max


# ===============================================
#  生成不重名输出文件
# ===============================================
def make_unique_filepath(output_folder, stem, suffix):
    output_folder = Path(output_folder)
    candidate = output_folder / f"{stem}{suffix}"

    if not candidate.exists():
        return candidate

    counter = 2
    while True:
        candidate = output_folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ===============================================
#  处理同名前缀计数
#  例如 shp 里有两个 id 都叫 A，则分别匹配:
#  A, A_2, A_3 ...
# ===============================================
def build_prefix_instance(row, prefix_field, seen_counter):
    raw_prefix = str(row[prefix_field]).strip()
    if raw_prefix == "" or raw_prefix.lower() == "nan":
        raw_prefix = "feat"

    # Windows 文件名安全处理
    invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for ch in invalid_chars:
        raw_prefix = raw_prefix.replace(ch, "_")

    count = seen_counter.get(raw_prefix, 0) + 1
    seen_counter[raw_prefix] = count

    if count == 1:
        return raw_prefix
    else:
        return f"{raw_prefix}_{count}"


# ===============================================
#  找到 mafic 区间（不是单点命中就停止）
#  仍然沿用“两条剖面 + 中点断开”的风格
# ===============================================
def find_mafic_interval(dem_vals, cf_vals, start_idx, end_idx, threshold=8.2, min_count=2):
    """
    在指定索引范围内查找 CF < threshold 的所有像元，
    返回 first / last mafic pixel 的索引与高程。
    同时屏蔽 0 / nodata / nan。
    """
    result = {
        "first_idx": np.nan,
        "last_idx": np.nan,
        "first_elev": np.nan,
        "last_elev": np.nan,
        "count": 0,
        "mafic_range": np.nan
    }

    if cf_vals is None or len(cf_vals) == 0:
        return result

    dem_vals = clean_profile_values(dem_vals, invalid_low=-3e10, invalid_zero=True)
    cf_vals = clean_profile_values(cf_vals, invalid_low=-3e10, invalid_zero=True)

    n_dem = len(dem_vals)
    n_cf = len(cf_vals)
    n_use = min(n_dem, n_cf)

    if n_use == 0:
        return result

    # 防止索引越界
    start_idx = int(max(0, min(start_idx, n_use - 1)))
    end_idx   = int(max(0, min(end_idx,   n_use - 1)))

    if start_idx <= end_idx:
        search_idx = np.arange(start_idx, end_idx + 1)
    else:
        search_idx = np.arange(start_idx, end_idx - 1, -1)

    mafic_idx = []
    for idx in search_idx:
        if np.isfinite(dem_vals[idx]) and np.isfinite(cf_vals[idx]) and (cf_vals[idx] < threshold):
            mafic_idx.append(idx)

    if len(mafic_idx) < min_count:
        return result

    first_idx = mafic_idx[0]
    last_idx  = mafic_idx[-1]

    first_elev = dem_vals[first_idx]
    last_elev  = dem_vals[last_idx]

    # first and last mafic pixels 的 vertical range
    mafic_range = abs(first_elev - last_elev)

    result["first_idx"] = int(first_idx)
    result["last_idx"] = int(last_idx)
    result["first_elev"] = float(first_elev)
    result["last_elev"] = float(last_elev)
    result["count"] = int(len(mafic_idx))
    result["mafic_range"] = float(mafic_range)
    return result


# ===============================================
#  判断 rim 为 mafic 还是 felsic
# ===============================================
def classify_rim_composition(cf_vals, peak_idx, threshold=8.2, win=2):
    """
    用峰顶附近一个小窗口判断 rim 成分类型
    CF < threshold  -> mafic
    CF >= threshold -> felsic
    """
    cf_vals = clean_profile_values(cf_vals, invalid_low=-3e10, invalid_zero=True)

    lo = max(0, peak_idx - win)
    hi = min(len(cf_vals), peak_idx + win + 1)

    local_cf = cf_vals[lo:hi]
    local_cf = local_cf[np.isfinite(local_cf)]

    if local_cf.size == 0:
        return "unknown"

    return "mafic" if np.nanmedian(local_cf) < threshold else "felsic"


# ===============================================
#  单文件平均高程与厚度计算
#  - 仍然是左半段/右半段
#  - 仍然是峰顶 + 外侧平缓区 + 内侧 basalt
#  - 公式部分改成按原文执行
# ===============================================
def calc_mean_elevation(csv_path, cf_path=None):
    x, lon, lat, n, mid = read_profile(csv_path)

    # 再做一层保护
    x = clean_profile_values(x, invalid_low=-3e10, invalid_zero=True)

    # 左半段最高点
    left_idx_local = safe_nanargmax(x[:mid + 1])
    if left_idx_local is None:
        return None
    left_peak_idx = left_idx_local
    left_peak_val = x[left_peak_idx]
    left_peak_lon = lon[left_peak_idx]
    left_peak_lat = lat[left_peak_idx]

    # 右半段最高点
    right_idx_local = safe_nanargmax(x[mid:])
    if right_idx_local is None:
        return None
    right_peak_idx = right_idx_local + mid
    right_peak_val = x[right_peak_idx]
    right_peak_lon = lon[right_peak_idx]
    right_peak_lat = lat[right_peak_idx]

    if not np.isfinite(left_peak_val) or not np.isfinite(right_peak_val):
        return None

    # ---------- 左右坡度计算 ----------
    left_slope_vals, left_slope_idxs = [], []
    for i in range(left_peak_idx, 0, -1):
        if np.isfinite(x[i]) and np.isfinite(x[i - 1]):
            dist_m = moon_distance_m(lon[i], lat[i], lon[i - 1], lat[i - 1])
            if np.isfinite(dist_m) and dist_m > 0:
                slope = math.atan((x[i] - x[i - 1]) / dist_m)
                slope_deg = abs(math.degrees(slope))
                left_slope_vals.append(slope_deg)
                left_slope_idxs.append(i - 1)

    right_slope_vals, right_slope_idxs = [], []
    for i in range(right_peak_idx, len(x) - 1):
        if np.isfinite(x[i]) and np.isfinite(x[i + 1]):
            dist_m = moon_distance_m(lon[i], lat[i], lon[i + 1], lat[i + 1])
            if np.isfinite(dist_m) and dist_m > 0:
                slope = math.atan((x[i + 1] - x[i]) / dist_m)
                slope_deg = abs(math.degrees(slope))
                right_slope_vals.append(slope_deg)
                right_slope_idxs.append(i + 1)

    # ---------- 提取平缓区 ----------
    left_flat_y  = extract_elev_by_slope(left_slope_vals, left_slope_idxs, x, left_peak_val)
    right_flat_y = extract_elev_by_slope(right_slope_vals, right_slope_idxs, x, right_peak_val)

    mean_left  = np.nanmean(left_flat_y)  if len(left_flat_y) > 0 else np.nan
    mean_right = np.nanmean(right_flat_y) if len(right_flat_y) > 0 else np.nan

    # ---------- 计算 h1 ----------
    # h1 = H_DEMmax - H_surface
    h1_left  = left_peak_val  - mean_left   if np.isfinite(mean_left)  else np.nan
    h1_right = right_peak_val - mean_right  if np.isfinite(mean_right) else np.nan

    # ---------- 初始化 ----------
    h2_left = np.nan
    h2_right = np.nan

    h_left = np.nan
    h_right = np.nan

    left_first_idx = np.nan
    left_last_idx = np.nan
    right_first_idx = np.nan
    right_last_idx = np.nan

    left_first_elev = np.nan
    left_last_elev = np.nan
    right_first_elev = np.nan
    right_last_elev = np.nan

    left_mafic_count = 0
    right_mafic_count = 0
    mafic_range_left = np.nan
    mafic_range_right = np.nan

    # 新增：公式中的 H_basalt
    h_basalt_left = np.nan
    h_basalt_right = np.nan

    # 新增：rim 类型
    rim_type_left = "unknown"
    rim_type_right = "unknown"

    # 新增：对于 felsic rim，再计算真实厚度
    t_basalt = np.nan

    # 额外输出：first/last mafic pixel 对应 burial depth
    h_begin = np.nan
    h_end = np.nan

    if cf_path and Path(cf_path).exists():
        cf_vals = pd.read_csv(cf_path)["value"].to_numpy(dtype=float)
        cf_vals = clean_profile_values(cf_vals, invalid_low=-3e10, invalid_zero=True)

        # 左侧：从左 rim 往中点
        left_mafic = find_mafic_interval(
            dem_vals=x,
            cf_vals=cf_vals,
            start_idx=left_peak_idx,
            end_idx=mid,
            threshold=8.2,
            min_count=2
        )

        # 右侧：从右 rim 往中点
        right_mafic = find_mafic_interval(
            dem_vals=x,
            cf_vals=cf_vals,
            start_idx=right_peak_idx,
            end_idx=mid,
            threshold=8.2,
            min_count=2
        )

        left_first_idx = left_mafic["first_idx"]
        left_last_idx = left_mafic["last_idx"]
        left_first_elev = left_mafic["first_elev"]
        left_last_elev = left_mafic["last_elev"]
        left_mafic_count = left_mafic["count"]
        mafic_range_left = left_mafic["mafic_range"]

        right_first_idx = right_mafic["first_idx"]
        right_last_idx = right_mafic["last_idx"]
        right_first_elev = right_mafic["first_elev"]
        right_last_elev = right_mafic["last_elev"]
        right_mafic_count = right_mafic["count"]
        mafic_range_right = right_mafic["mafic_range"]

        # ---------- 判断左右 rim 类型 ----------
        rim_type_left = classify_rim_composition(cf_vals, left_peak_idx, threshold=8.2, win=2)
        rim_type_right = classify_rim_composition(cf_vals, right_peak_idx, threshold=8.2, win=2)

        # ---------- 按原文确定公式里的 H_basalt ----------
        # 1) mafic dominated rim：H_basalt 取该侧最内侧 basaltic layer（last mafic pixel）
        # 2) felsic rim：公式算 burial depth，因此取 first/last mafic pixel 来构造 begin/end
        if rim_type_left == "mafic":
            h_basalt_left = left_last_elev
        elif rim_type_left == "felsic":
            h_basalt_left = left_first_elev

        if rim_type_right == "mafic":
            h_basalt_right = right_last_elev
        elif rim_type_right == "felsic":
            h_basalt_right = right_first_elev

        # ---------- 各侧先按公式计算 ----------
        # h2 = H_DEMmax - H_basalt
        if np.isfinite(h_basalt_left):
            h2_left = left_peak_val - h_basalt_left

        if np.isfinite(h_basalt_right):
            h2_right = right_peak_val - h_basalt_right

        # h = 0.8 * (h2 - 0.2*h1)
        if np.isfinite(h1_left) and np.isfinite(h2_left):
            h_left = 0.8 * (h2_left - 0.2 * h1_left)

        if np.isfinite(h1_right) and np.isfinite(h2_right):
            h_right = 0.8 * (h2_right - 0.2 * h1_right)

        # ---------- 对 felsic rim，按原文用 first / last mafic pixels 的 vertical range 算真实厚度 ----------
        # 在当前“两边分开”的框架下：
        # left_first_elev  作为 begin
        # right_first_elev 作为 end
        # 两侧各自先算 burial depth，再做差值
        if (rim_type_left == "felsic") and (rim_type_right == "felsic"):
            if np.isfinite(left_first_elev):
                h2_begin = left_peak_val - left_first_elev
                if np.isfinite(h1_left):
                    h_begin = 0.8 * (h2_begin - 0.2 * h1_left)

            if np.isfinite(right_first_elev):
                h2_end = right_peak_val - right_first_elev
                if np.isfinite(h1_right):
                    h_end = 0.8 * (h2_end - 0.2 * h1_right)

            if np.isfinite(h_begin) and np.isfinite(h_end):
                t_basalt = abs(h_end - h_begin)

    # ---------- 简单物理约束 ----------
    if np.isfinite(h_left) and h_left < 0:
        h_left = np.nan
    if np.isfinite(h_right) and h_right < 0:
        h_right = np.nan
    if np.isfinite(t_basalt) and t_basalt < 0:
        t_basalt = np.nan
    if np.isfinite(h_begin) and h_begin < 0:
        h_begin = np.nan
    if np.isfinite(h_end) and h_end < 0:
        h_end = np.nan

    return {
        "left_peak": (left_peak_val, left_peak_lon, left_peak_lat),
        "right_peak": (right_peak_val, right_peak_lon, right_peak_lat),

        "mean_left": mean_left,
        "mean_right": mean_right,

        "h1_left": h1_left,
        "h1_right": h1_right,

        "rim_type_left": rim_type_left,
        "rim_type_right": rim_type_right,

        "h_basalt_left": h_basalt_left,
        "h_basalt_right": h_basalt_right,

        "h2_left": h2_left,
        "h2_right": h2_right,

        # mafic rim 时，h_left/h_right = thickness
        # felsic rim 时，h_left/h_right = burial depth
        "h_left": h_left,
        "h_right": h_right,

        # felsic 情况下补充输出
        "h_begin": h_begin,
        "h_end": h_end,
        "t_basalt": t_basalt,

        # mafic 区间信息
        "left_first_idx": left_first_idx,
        "left_last_idx": left_last_idx,
        "right_first_idx": right_first_idx,
        "right_last_idx": right_last_idx,

        "left_first_elev": left_first_elev,
        "left_last_elev": left_last_elev,
        "right_first_elev": right_first_elev,
        "right_last_elev": right_last_elev,

        "left_maf_cnt": left_mafic_count,
        "right_maf_cnt": right_mafic_count,
        "m_rng_left": mafic_range_left,
        "m_rng_right": mafic_range_right
    }


# ===============================================
#  根据 dataset_dir 和 shp_source 自动推断 shp 路径
#  与前两个脚本风格统一：
#  Database/CE5/manual_ce5.shp
#  Database/CE5/yolo_ce5.shp
# ===============================================
def infer_shp_path(dataset_dir, shp_source):
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_dir.name.lower()
    shp_file = dataset_dir / f"{shp_source}_{dataset_name}.shp"
    return shp_file


# ===============================================
#  主函数
# ===============================================
def main(dataset_dir, shp_source):
    dataset_dir = Path(dataset_dir)

    shp_file = infer_shp_path(dataset_dir, shp_source)
    base_dir = dataset_dir / shp_source / "show"
    output_dir = dataset_dir / shp_source / "thickness"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"未找到数据集目录：{dataset_dir}")

    if not shp_file.exists():
        raise FileNotFoundError(f"未找到 shp 文件：{shp_file}")

    if not base_dir.exists():
        raise FileNotFoundError(f"未找到 show 目录：{base_dir}")

    gdf_src = gpd.read_file(shp_file)

    print(f"\n当前数据集目录: {dataset_dir}")
    print(f"当前使用的 shp: {shp_file}")
    print(f"CSV 输入目录: {base_dir}")
    print(f"结果输出目录: {output_dir}")

    moon2000_wkt = (
        'GEOGCS["GCS_Moon_2000",'
        'DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
        'PRIMEM["Reference_Meridian",0],'
        'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
        'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'
    )

    if gdf_src.crs is None:
        print("⚠ 输入 shp 坐标系无效，自动指定为 Moon 2000")
        crs_wkt = moon2000_wkt
    else:
        crs_wkt = gdf_src.crs.to_wkt()

    if "id" in gdf_src.columns:
        prefix_field = "id"
    elif "name" in gdf_src.columns:
        prefix_field = "name"
    else:
        raise KeyError(f"{shp_file.name} 中既没有 'id' 字段，也没有 'name' 字段")

    seen_counter = {}
    all_records = []

    for idx, row in gdf_src.iterrows():
        prefix = build_prefix_instance(row, prefix_field, seen_counter)

        col_path = base_dir / f"{prefix}_DEM_Col.csv"
        row_path = base_dir / f"{prefix}_DEM_Row.csv"
        col_CF_path = base_dir / f"{prefix}_CF_Col.csv"
        row_CF_path = base_dir / f"{prefix}_CF_Row.csv"

        if not col_path.exists() or not row_path.exists():
            print(f"⚠ 未找到 CSV，跳过：{prefix}")
            continue

        col_res = calc_mean_elevation(col_path, col_CF_path)
        row_res = calc_mean_elevation(row_path, row_CF_path)

        if col_res is None or row_res is None:
            print(f"⚠ 剖面全为无效值，跳过：{prefix}")
            continue

        col_max = get_max_elevation(col_path)
        row_max = get_max_elevation(row_path)

        out_txt = make_unique_filepath(output_dir, f"{prefix}_thickness", ".txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"文件前缀: {prefix}\n")
            f.write("=== 列方向 (Col) ===\n")
            for key, val in col_res.items():
                if isinstance(val, tuple):
                    continue
                if isinstance(val, (int, float, np.integer, np.floating)):
                    if np.isnan(val):
                        f.write(f"  {key:<14}: nan\n")
                    else:
                        f.write(f"  {key:<14}: {val:.4f}\n")
                else:
                    f.write(f"  {key:<14}: {val}\n")
            f.write(f"  {'left_max':<14}: {col_max[0]:.4f}\n" if np.isfinite(col_max[0]) else f"  {'left_max':<14}: nan\n")
            f.write(f"  {'right_max':<14}: {col_max[1]:.4f}\n\n" if np.isfinite(col_max[1]) else f"  {'right_max':<14}: nan\n\n")

            f.write("=== 行方向 (Row) ===\n")
            for key, val in row_res.items():
                if isinstance(val, tuple):
                    continue
                if isinstance(val, (int, float, np.integer, np.floating)):
                    if np.isnan(val):
                        f.write(f"  {key:<14}: nan\n")
                    else:
                        f.write(f"  {key:<14}: {val:.4f}\n")
                else:
                    f.write(f"  {key:<14}: {val}\n")
            f.write(f"  {'left_max':<14}: {row_max[0]:.4f}\n" if np.isfinite(row_max[0]) else f"  {'left_max':<14}: nan\n")
            f.write(f"  {'right_max':<14}: {row_max[1]:.4f}\n" if np.isfinite(row_max[1]) else f"  {'right_max':<14}: nan\n")

        print(f"✅ Saved {out_txt.name}")

        if np.isfinite(col_res["h_left"]) and col_res["h_left"] >= 0:
            val, lon_pt, lat_pt = col_res["left_peak"]
            all_records.append({
                "prefix": prefix,
                "dir": "col_left",
                "thickness": col_res["h_left"],
                "m_range": col_res["m_rng_left"],
                "geometry": Point(lon_pt, lat_pt)
            })

        if np.isfinite(col_res["h_right"]) and col_res["h_right"] >= 0:
            val, lon_pt, lat_pt = col_res["right_peak"]
            all_records.append({
                "prefix": prefix,
                "dir": "col_right",
                "thickness": col_res["h_right"],
                "m_range": col_res["m_rng_right"],
                "geometry": Point(lon_pt, lat_pt)
            })

        if np.isfinite(row_res["h_left"]) and row_res["h_left"] >= 0:
            val, lon_pt, lat_pt = row_res["left_peak"]
            all_records.append({
                "prefix": prefix,
                "dir": "row_left",
                "thickness": row_res["h_left"],
                "m_range": row_res["m_rng_left"],
                "geometry": Point(lon_pt, lat_pt)
            })

        if np.isfinite(row_res["h_right"]) and row_res["h_right"] >= 0:
            val, lon_pt, lat_pt = row_res["right_peak"]
            all_records.append({
                "prefix": prefix,
                "dir": "row_right",
                "thickness": row_res["h_right"],
                "m_range": row_res["m_rng_right"],
                "geometry": Point(lon_pt, lat_pt)
            })

    if all_records:
        gdf = gpd.GeoDataFrame(all_records, crs=crs_wkt)
        out_shp = output_dir / f"all_{shp_source}.shp"
        gdf.to_file(out_shp)
        print(f"✅ SHP 已生成: {out_shp}")
    else:
        print("⚠ 没有有效点生成 SHP")

    print("🎯 所有文件处理完成！")


# ===============================================
#  终端入口
# ===============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="厚度计算脚本")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help=r"数据集目录，例如：Database\CE5"
    )
    parser.add_argument(
        "--shp",
        choices=["manual", "yolo"],
        required=True,
        help="选择使用哪个 shp：manual / yolo"
    )

    args = parser.parse_args()
    main(args.dataset_dir, args.shp)