# MODIFIED_FOR_RIM_PEAK_2026_05_12: adaptive 1/3/5 rim-height scoring; core-boundary preference cancelled; SHP points use rim peaks.
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================
# Easy-to-edit input/output paths
# =========================
DEFAULT_DEM_PATH = Path(r"Database\CE5\CE5_dem.tif")
DEFAULT_CF_PATH = Path(r"Database\CE5\CE5_CF.tif")
DEFAULT_SHP_PATH = Path(r"Database/CE5/exce5/yolo_ce5.shp")
DEFAULT_STEP1_REJECT_CSV = Path(r"Database\CE5\yolo\chickin\step1\all_yolo_h123_reject.csv")
DEFAULT_OUTPUT_DIR = Path(r"Database\CE5\yolo\chickin\step2")
PLOT_DIRNAME = "plots"
CORE_SHP_NAME = "yolo_core_boxes.shp"
EXPAND_SHP_NAME = "yolo_expand_boxes.shp"
VALID_CSV_NAME = "all_yolo_h123.csv"
REJECT_CSV_NAME = "all_yolo_h123_reject.csv"

# These fields are produced by CalculateH123.py and are only inherited here.
# ExtendCalculateH123.py does not re-classify filled craters.
FILLED_FLAG_COLS = [
    'crater_filled_flag', 'profile_filled_flag', 'filled_checked',
    'filled_slope_ratio_le1', 'filled_slope_le1_count', 'filled_slope_valid_count',
    'filled_slope_threshold_deg', 'filled_ratio_threshold', 'filled_reason',
]

# These fields are produced by CalculateH123.py and are inherited or refreshed here.
BASALT_FLAG_COLS = [
    'basalt_cf_threshold', 'peak_cf', 'rim_type_new',
    'profile_between_valid_cf_count', 'profile_between_basalt_count', 'profile_between_nonbasalt_count',
    'profile_both_rims_basalt', 'profile_all_between_rims_basalt',
    'basalt_only_unpenetrated_flag', 'basalt_profile_reason', 'left_rim_cf', 'right_rim_cf',
    'contact_idx', 'contact_global', 'contact_row', 'contact_col', 'contact_cf',
    'h_basalt_elev', 'h2_basalt', 'thickness_or_depth',
    'thickness_mode', 'basalt_contact_status', 'contact_note',
    'cf_class', 'thk_ok', 'unpenetrated_flag', 'cf_class_note',
]

# =========================
# Parameters (keep original Extend core/expand logic unchanged)
# =========================
INVALID_LOW = -3e10
INVALID_ZERO = True
MOON_RADIUS_M = 1737400.0

OUTER_DEPTH_RATIO = 1.50
OUTER_DEPTH_MIN_PX = 20
OUTER_DEPTH_MAX_PX = 200

SLOPE_THRESHOLD_DEG = 3.0
OVERALL_THRESHOLD = 8.0
WINDOW_SIZES = (3, 5)
MIN_INNER_OFFSET = 1
MAX_RIM_CANDIDATES = 10
MAX_OUTER_SEGMENTS_PER_PROFILE = 6
FIG_DPI = 180

# step2 fallback
V_NOTCH_MIN_DEPTH_M = 5.0
V_NOTCH_MAX_CANDIDATES = 3
OUTER_REF_TYPE_STRICT_FLAT = 'strict_flat'
OUTER_REF_TYPE_V_NOTCH = 'v_notch_higher_peak'
OUTER_REF_TYPE_LOWEST_POINT = 'lowest_point_between_peaks'

# Current CE5 basalt rule: CF > 8.2 indicates basalt.
BASALT_CF_THRESHOLD = 8.2


# -------------------------
# basic utils
# -------------------------
def clip_int(v: float, vmin: int, vmax: int) -> int:
    return int(max(vmin, min(vmax, round(v))))


def clean_profile_values(values, invalid_low=INVALID_LOW, invalid_zero=INVALID_ZERO):
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    values[values < invalid_low] = np.nan
    if invalid_zero:
        values[values == 0] = np.nan
    return values


def moon_distance_m(lon1, lat1, lon2, lat2, radius=MOON_RADIUS_M):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def calc_point_slopes(values, lon, lat):
    values = np.asarray(values, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = len(values)
    slopes = np.full(n, np.nan, dtype=float)
    for i in range(1, n - 1):
        if not (np.isfinite(values[i - 1]) and np.isfinite(values[i]) and np.isfinite(values[i + 1])):
            continue
        if not (np.isfinite(lon[i - 1]) and np.isfinite(lat[i - 1]) and np.isfinite(lon[i + 1]) and np.isfinite(lat[i + 1])):
            continue
        dist_m = moon_distance_m(lon[i - 1], lat[i - 1], lon[i + 1], lat[i + 1])
        if not np.isfinite(dist_m) or dist_m <= 0:
            continue
        dz = values[i + 1] - values[i - 1]
        slopes[i] = abs(math.degrees(math.atan(dz / dist_m)))
    return slopes


def crs_body_type(crs) -> str:
    if crs is None:
        return 'none'
    s = str(crs).lower()
    if 'moon' in s or 'selen' in s or '1737400' in s or 'iau' in s:
        return 'moon'
    if 'wgs84' in s or '4326' in s or 'greenwich' in s or '6378137' in s or '298.257223563' in s:
        return 'earth'
    return 'other'


def harmonize_vector_raster_crs(gdf, ds):
    if ds.crs is None:
        raise ValueError('输入 DEM 没有 CRS，无法与 shp 对齐。')
    print(f'shp CRS: {gdf.crs}')
    print(f'DEM CRS: {ds.crs}')

    if gdf.crs is None:
        print('shp 没有 CRS，直接赋值为 DEM 的 CRS，不做坐标变换。')
        gdf = gdf.set_crs(ds.crs)
        return gdf, ds.crs
    if gdf.crs == ds.crs:
        print('shp 与 DEM CRS 一致，无需转换。')
        return gdf, gdf.crs

    shp_body = crs_body_type(gdf.crs)
    dem_body = crs_body_type(ds.crs)
    shp_txt = str(gdf.crs).lower()
    dem_txt = str(ds.crs).lower()
    shp_is_geographic = 'degree' in shp_txt or 'geogcs' in shp_txt or 'geographic' in shp_txt
    dem_is_geographic = 'degree' in dem_txt or 'geogcs' in dem_txt or 'geographic' in dem_txt

    if shp_body == 'moon' and dem_body == 'earth' and shp_is_geographic and dem_is_geographic:
        print('检测到 shp 是月球坐标，而 DEM 被错误标成地球 geographic CRS。')
        print('不进行坐标变换，直接沿用 shp 的月球 CRS 作为工作 CRS。')
        return gdf, gdf.crs

    if shp_body == 'earth' and dem_body == 'moon' and shp_is_geographic and dem_is_geographic:
        print('检测到 shp 是地球 geographic CRS，而 DEM 是月球坐标。')
        print('不进行坐标变换，直接覆盖 shp CRS 为 DEM CRS。')
        gdf = gdf.set_crs(ds.crs, allow_override=True)
        return gdf, ds.crs

    print('执行真正的 CRS 转换：gdf.to_crs(ds.crs)')
    gdf = gdf.to_crs(ds.crs)
    return gdf, ds.crs


def infer_name_field(gdf: gpd.GeoDataFrame) -> str:
    for field in ['name', 'Name', 'NAME', 'id', 'ID', 'fid', 'FID']:
        if field in gdf.columns:
            return field
    return '__index__'


def safe_feature_name(row, field_name: str, idx: int) -> str:
    if field_name == '__index__':
        name = f'feat_{idx}'
    else:
        try:
            name = str(row[field_name]).strip()
            if name == '' or name.lower() == 'nan':
                name = f'feat_{idx}'
        except Exception:
            name = f'feat_{idx}'
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, '_')
    return name


def geom_bounds_to_rc(src, geom) -> Tuple[int, int, int, int]:
    minx, miny, maxx, maxy = geom.bounds
    r0, c0 = src.index(minx, maxy)
    r1, c1 = src.index(maxx, miny)
    rmin, rmax = sorted((r0, r1))
    cmin, cmax = sorted((c0, c1))
    rmin = max(0, rmin)
    cmin = max(0, cmin)
    rmax = min(src.height - 1, rmax)
    cmax = min(src.width - 1, cmax)
    return rmin, rmax, cmin, cmax


def make_boxes_from_original(rmin: int, rmax: int, cmin: int, cmax: int, src) -> Dict:
    # keep original Extend core / expand algorithm unchanged
    core_h = max(1, rmax - rmin + 1)
    core_w = max(1, cmax - cmin + 1)
    out_dx = clip_int(core_w * OUTER_DEPTH_RATIO, OUTER_DEPTH_MIN_PX, OUTER_DEPTH_MAX_PX)
    out_dy = clip_int(core_h * OUTER_DEPTH_RATIO, OUTER_DEPTH_MIN_PX, OUTER_DEPTH_MAX_PX)
    exp_rmin = max(0, rmin - out_dy)
    exp_rmax = min(src.height - 1, rmax + out_dy)
    exp_cmin = max(0, cmin - out_dx)
    exp_cmax = min(src.width - 1, cmax + out_dx)
    return {
        'core': {'rmin': rmin, 'rmax': rmax, 'cmin': cmin, 'cmax': cmax},
        'expanded': {'rmin': exp_rmin, 'rmax': exp_rmax, 'cmin': exp_cmin, 'cmax': exp_cmax},
        'core_h': core_h,
        'core_w': core_w,
        'expand_dx': out_dx,
        'expand_dy': out_dy,
    }


def rc_box_to_geom(src, rmin: int, rmax: int, cmin: int, cmax: int):
    x_left, y_top = src.transform * (cmin, rmin)
    x_right, y_bottom = src.transform * (cmax + 1, rmax + 1)
    xmin, xmax = sorted((x_left, x_right))
    ymin, ymax = sorted((y_top, y_bottom))
    return box(xmin, ymin, xmax, ymax)


def read_window(src, rmin: int, rmax: int, cmin: int, cmax: int):
    window = Window(cmin, rmin, cmax - cmin + 1, rmax - rmin + 1)
    arr = src.read(1, window=window).astype(float)
    arr = clean_profile_values(arr)
    transform = src.window_transform(window)
    return arr, transform


def adaptive_rim_height_window(core_diameter_px):
    """
    Adaptive local height window for rim candidate ranking.
      <=12 px : single-pixel elevation for small craters;
      13-40 px: 3-pixel local median for medium craters;
      >40 px : 5-pixel local median for larger craters.
    """
    try:
        dpx = int(round(float(core_diameter_px)))
    except Exception:
        dpx = 0
    if dpx <= 12:
        return 1
    if dpx <= 40:
        return 3
    return 5


def local_median_height(values, idx, win):
    idx = int(idx)
    win = int(max(1, win))
    if win <= 1:
        return float(values[idx]) if 0 <= idx < len(values) and np.isfinite(values[idx]) else np.nan
    half = win // 2
    lo = max(0, idx - half)
    hi = min(len(values) - 1, idx + half)
    seg = np.asarray(values[lo:hi + 1], dtype=float)
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return np.nan
    return float(np.nanmedian(seg))


# -------------------------
# morphology helpers reused from current step2
# -------------------------
def flat_window_ok(window_slopes, slope_threshold=SLOPE_THRESHOLD_DEG, overall_threshold=OVERALL_THRESHOLD):
    ws = np.asarray(window_slopes, dtype=float)
    ws = ws[np.isfinite(ws)]
    if ws.size == 0:
        return False
    need = int(np.ceil(ws.size / 2.0))
    count_lt = int(np.sum(ws < slope_threshold))
    return (count_lt >= need) and (np.nanmax(ws) <= overall_threshold)


def valid_sequence(order_indices, slopes, values=None):
    seq = []
    n = len(slopes)
    for idx in order_indices:
        if not (0 <= idx < n):
            continue
        if not np.isfinite(slopes[idx]):
            continue
        if values is not None and not np.isfinite(values[idx]):
            continue
        seq.append(int(idx))
    return seq


def find_strict_outer_flat(order_indices, slopes, values,
                           consecutive=3,
                           slope_threshold=SLOPE_THRESHOLD_DEG):
    """
    严格平坦区判定：必须连续 3 个坡度点都 < 3°。
    这里的判据本身就是“3 个连续坡度点”，不额外附加“5 个像元”条件。
    搜索顺序按 outward_order 从靠近 rim 到远离 rim 进行，返回最近的一段平坦区。
    """
    seq = valid_sequence(order_indices, slopes, values)
    if len(seq) < consecutive:
        return None

    run_start = None
    for i in range(len(seq) - consecutive + 1):
        w = seq[i:i + consecutive]
        # 必须是原生索引连续
        step = -1 if len(w) >= 2 and w[1] < w[0] else 1
        if any(w[j + 1] - w[j] != step for j in range(len(w) - 1)):
            continue
        s = np.asarray([slopes[idx] for idx in w], dtype=float)
        if np.all(np.isfinite(s)) and np.all(s < slope_threshold):
            run_start = i
            break
    if run_start is None:
        return None

    step = -1 if len(seq) >= 2 and seq[1] < seq[0] else 1
    start_slope = seq[run_start]
    end_slope = seq[run_start + consecutive - 1]

    j = run_start + consecutive
    while j < len(seq):
        prev_idx = seq[j - 1]
        this_idx = seq[j]
        if this_idx - prev_idx != step:
            break
        if not (np.isfinite(slopes[this_idx]) and slopes[this_idx] < slope_threshold and np.isfinite(values[this_idx])):
            break
        end_slope = this_idx
        j += 1

    val_start = max(0, min(start_slope, end_slope) - 1)
    val_end = min(len(values) - 1, max(start_slope, end_slope) + 1)
    seg_vals = clean_profile_values(values[val_start:val_end + 1])
    seg_slopes = np.asarray([slopes[idx] for idx in seq[run_start:j] if np.isfinite(slopes[idx])], dtype=float)
    if seg_vals.size == 0 or seg_slopes.size == 0:
        return None

    return {
        'kind': OUTER_REF_TYPE_STRICT_FLAT,
        'first_idx': int(val_start),
        'last_idx': int(val_end),
        'indices': list(range(int(min(start_slope, end_slope)), int(max(start_slope, end_slope)) + 1)),
        'mean': float(np.nanmean(seg_vals)),
        'median': float(np.nanmedian(seg_vals)),
        'min': float(np.nanmin(seg_vals)),
        'max': float(np.nanmax(seg_vals)),
        'elev_std': float(np.nanstd(seg_vals)),
        'relief': float(np.nanmax(seg_vals) - np.nanmin(seg_vals)),
        'slope_mean': float(np.nanmean(seg_slopes)),
        'slope_max': float(np.nanmax(seg_slopes)),
        'n': int(seg_vals.size),
    }


def find_outer_v_notches(order_indices, values, max_candidates=V_NOTCH_MAX_CANDIDATES, min_depth=V_NOTCH_MIN_DEPTH_M):
    seq = [idx for idx in order_indices if 0 <= idx < len(values) and np.isfinite(values[idx])]
    if len(seq) < 5:
        return []
    z = np.asarray([values[i] for i in seq], dtype=float)
    peak_pos = []
    for p in range(1, len(seq) - 1):
        if z[p] >= z[p - 1] and z[p] >= z[p + 1]:
            peak_pos.append(p)
    if len(peak_pos) < 2:
        return []
    results = []
    for a, b in zip(peak_pos[:-1], peak_pos[1:]):
        if b - a < 2:
            continue
        valley_rel = int(np.argmin(z[a:b + 1]))
        valley_pos = a + valley_rel
        if valley_pos <= a or valley_pos >= b:
            continue
        valley_idx = int(seq[valley_pos])
        z_left = float(z[a])
        z_right = float(z[b])
        z_valley = float(z[valley_pos])
        depth = min(z_left, z_right) - z_valley
        if not np.isfinite(depth) or depth < min_depth:
            continue
        results.append({
            'kind': OUTER_REF_TYPE_V_NOTCH,
            'first_idx': valley_idx,
            'last_idx': valley_idx,
            'indices': [valley_idx],
            'mean': z_valley,
            'median': z_valley,
            'min': z_valley,
            'elev_std': 0.0,
            'relief': depth,
            'slope_mean': np.nan,
            'slope_max': np.nan,
            'n': 1,
            'anchor_idx': valley_idx,
            'near_rank': valley_pos,
            'left_peak_elev': z_left,
            'right_peak_elev': z_right,
        })
        if len(results) >= max_candidates:
            break
    results.sort(key=lambda d: (d['near_rank'], -d['relief']))
    return results


def find_outward_local_peaks(order_indices, values):
    seq = [idx for idx in order_indices if 0 <= idx < len(values) and np.isfinite(values[idx])]
    if len(seq) < 3:
        return []
    z = np.asarray([values[i] for i in seq], dtype=float)
    peaks = []
    for p in range(1, len(seq) - 1):
        if z[p] >= z[p - 1] and z[p] >= z[p + 1]:
            peaks.append({
                'idx': int(seq[p]),
                'elev': float(z[p]),
                'near_rank': int(p),
            })
    return peaks


def build_lowest_point_between_peaks(rim_idx, lower_peak_idx, values):
    lo = int(min(rim_idx, lower_peak_idx))
    hi = int(max(rim_idx, lower_peak_idx))
    if hi - lo < 2:
        return None
    cand = [i for i in range(lo + 1, hi) if np.isfinite(values[i])]
    if not cand:
        return None
    local_vals = np.asarray([values[i] for i in cand], dtype=float)
    pos = int(np.argmin(local_vals))
    idx = int(cand[pos])
    val = float(local_vals[pos])
    return {
        'kind': OUTER_REF_TYPE_LOWEST_POINT,
        'first_idx': idx,
        'last_idx': idx,
        'indices': [idx],
        'mean': val,
        'median': val,
        'min': val,
        'max': val,
        'elev_std': 0.0,
        'relief': float(max(values[rim_idx], values[lower_peak_idx]) - val) if np.isfinite(values[rim_idx]) and np.isfinite(values[lower_peak_idx]) else 0.0,
        'slope_mean': np.nan,
        'slope_max': np.nan,
        'n': 1,
        'near_rank': abs(idx - rim_idx),
        'lower_peak_idx': int(lower_peak_idx),
        'lower_peak_elev': float(values[lower_peak_idx]) if np.isfinite(values[lower_peak_idx]) else np.nan,
    }


def _is_peak_like_within_range(values: np.ndarray, idx: int, lo: int, hi: int) -> bool:
    idx = int(idx)
    lo = int(lo)
    hi = int(hi)
    if idx < lo or idx > hi or not np.isfinite(values[idx]):
        return False

    left_candidates = [j for j in (idx - 1, idx - 2) if lo <= j <= hi and np.isfinite(values[j])]
    right_candidates = [j for j in (idx + 1, idx + 2) if lo <= j <= hi and np.isfinite(values[j])]

    left_ok = True if not left_candidates else all(values[idx] >= values[j] for j in left_candidates)
    right_ok = True if not right_candidates else all(values[idx] >= values[j] for j in right_candidates)
    return bool(left_ok and right_ok)


def find_local_peak_candidates(values: np.ndarray, start_idx: int, end_idx: int, center_idx: int,
                               max_candidates: int = MAX_RIM_CANDIDATES,
                               priority_boundary_idx: Optional[int] = None,
                               side: Optional[str] = None,
                               boundary_window: int = 3,
                               core_diameter_px: Optional[int] = None) -> List[int]:
    """
    在当前侧 core 半边内找坑顶候选。

    本版取消“靠近 core 边界优先”的逻辑；priority_boundary_idx、boundary_window
    参数仅为兼容旧调用而保留，不参与排序。候选优先满足局部峰形（^），排序依据为
    自适应局部高程：小坑 1 像元，中坑 3 像元，大坑 5 像元。
    """
    lo = int(min(start_idx, end_idx))
    hi = int(max(start_idx, end_idx))
    lo = max(0, lo)
    hi = min(len(values) - 1, hi)
    if hi < lo:
        return []

    if core_diameter_px is None:
        core_diameter_px = hi - lo + 1
    height_win = adaptive_rim_height_window(core_diameter_px)

    peak_like = []
    if hi - lo + 1 >= 3:
        for i in range(lo + 1, hi):
            if not (np.isfinite(values[i - 1]) and np.isfinite(values[i]) and np.isfinite(values[i + 1])):
                continue
            if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                peak_like.append(i)

    if peak_like:
        candidates = peak_like
    else:
        # 小坑或低分辨率剖面可能没有严格局部峰，退化为半边内有限高程点。
        candidates = [i for i in range(lo, hi + 1) if np.isfinite(values[i])]

    ranked = []
    for i in candidates:
        h = local_median_height(values, i, height_win)
        if np.isfinite(h):
            ranked.append((int(i), float(h), abs(int(i) - int(center_idx))))

    ranked.sort(key=lambda t: (-t[1], -t[2]))
    return [int(t[0]) for t in ranked[:max_candidates]]


def find_nearest_peak_outside_core(values: np.ndarray, side: str,
                                   core_lo: int, core_hi: int,
                                   ext_lo: int, ext_hi: int) -> Optional[int]:
    """
    当前侧 core 半边没有候选时，才在 core 外同侧补充寻找候选。
    本版取消“离 core 边界最近优先”，改为按自适应局部高程最高优先。
    """
    n = len(values)
    ext_lo = max(0, int(ext_lo))
    ext_hi = min(n - 1, int(ext_hi))
    core_lo = max(0, int(core_lo))
    core_hi = min(n - 1, int(core_hi))

    if side == 'left':
        lo, hi = ext_lo, core_lo - 1
    else:
        lo, hi = core_hi + 1, ext_hi

    if hi < lo:
        return None

    core_diameter_px = int(abs(core_hi - core_lo) + 1)
    height_win = adaptive_rim_height_window(core_diameter_px)

    peaks = []
    if hi - lo + 1 >= 3:
        for i in range(lo + 1, hi):
            if not (np.isfinite(values[i - 1]) and np.isfinite(values[i]) and np.isfinite(values[i + 1])):
                continue
            if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                peaks.append(i)
    if not peaks:
        peaks = [i for i in range(lo, hi + 1) if np.isfinite(values[i])]
    if not peaks:
        return None

    peaks = sorted(peaks, key=lambda i: -local_median_height(values, i, height_win))
    return int(peaks[0])


def find_rim_candidates_core_then_nearest(values: np.ndarray, side: str,
                                          core_lo: int, core_hi: int, core_center: int,
                                          ext_lo: int, ext_hi: int,
                                          max_candidates: int = MAX_RIM_CANDIDATES) -> List[int]:
    """
    step2 坑顶候选：
    1) 先在当前侧 core 半边内找局部峰/最高点；
    2) 取消“core 边界点第一优先”和“离 core 边界最近优先”；
    3) 候选按自适应局部高程排序：小坑 1 像元，中坑 3 像元，大坑 5 像元；
    4) 只有 core 半边内完全没有候选时，才到 core 外同侧补充寻找。
    """
    if side == 'left':
        search_lo, search_hi = core_lo, core_center
    else:
        search_lo, search_hi = core_center, core_hi

    core_diameter_px = int(abs(int(core_hi) - int(core_lo)) + 1)
    core_candidates = find_local_peak_candidates(
        values, search_lo, search_hi, core_center,
        max_candidates=max_candidates,
        core_diameter_px=core_diameter_px,
    )
    if core_candidates:
        return core_candidates

    nearest_outside = find_nearest_peak_outside_core(values, side, core_lo, core_hi, ext_lo, ext_hi)
    if nearest_outside is not None:
        return [int(nearest_outside)]
    return []


def pit_class_by_side_len(side_len: int) -> str:
    if side_len <= 10:
        return 'small'
    if side_len <= 18:
        return 'medium'
    return 'large'


def find_floor_segments(order_indices, slopes, values, slope_threshold=SLOPE_THRESHOLD_DEG):
    seq = valid_sequence(order_indices, slopes, values)
    if len(seq) == 0:
        return []
    runs = []
    i = 0
    while i < len(seq):
        idx = seq[i]
        s = slopes[idx]
        if not (np.isfinite(s) and s < slope_threshold):
            i += 1
            continue
        j = i
        pts = []
        while j < len(seq):
            idx2 = seq[j]
            s2 = slopes[idx2]
            if np.isfinite(s2) and s2 < slope_threshold and np.isfinite(values[idx2]):
                pts.append(idx2)
                j += 1
            else:
                break
        if pts:
            vals = np.asarray([values[p] for p in pts], dtype=float)
            runs.append({
                'first_idx': int(pts[0]), 'last_idx': int(pts[-1]), 'indices': pts,
                'n': int(len(pts)), 'min': float(np.nanmin(vals)), 'mean': float(np.nanmean(vals)),
                'median': float(np.nanmedian(vals)), 'relief': float(np.nanmax(vals) - np.nanmin(vals)),
                'slope_mean': float(np.nanmean([slopes[p] for p in pts])), 'center_idx': int(pts[len(pts) // 2]),
            })
        i = max(i + 1, j)
    return runs


def rank_floor_segments(segments: List[Dict], center_idx: int) -> List[Dict]:
    return sorted(
        segments,
        key=lambda seg: (-int(seg['n']), abs(int(seg['center_idx']) - int(center_idx)), float(seg['median']),
                         float(seg['relief']), float(seg['slope_mean']))
    )


def select_bottom_by_pit_rule(inward_order, slopes, values, core_center_idx):
    valid = [idx for idx in inward_order[MIN_INNER_OFFSET:] if np.isfinite(values[idx])]
    if not valid:
        return None, None, None, 'no_floor_candidate', '未找到坑内有效候选点'
    side_len = len(inward_order)
    pit_class = pit_class_by_side_len(side_len)
    floor_runs = find_floor_segments(valid, slopes, values, slope_threshold=SLOPE_THRESHOLD_DEG)
    pos = {idx: i for i, idx in enumerate(inward_order)}
    if pit_class == 'small':
        center_half = [idx for idx in valid if pos[idx] >= len(inward_order) // 2]
        cand = center_half if center_half else valid
        bottom_idx = sorted(cand, key=lambda idx: (values[idx], abs(idx - core_center_idx)))[0]
        return int(bottom_idx), float(values[bottom_idx]), 1, None, ''
    else:
        need = 2 if pit_class == 'medium' else 3
        qualified = [seg for seg in floor_runs if int(seg['n']) >= need]
        if qualified:
            qualified = rank_floor_segments(qualified, core_center_idx)
            seg = qualified[0]
            return int(seg['center_idx']), float(seg['median']), int(seg['n']), None, ''
        center_half = [idx for idx in valid if pos[idx] >= len(inward_order) // 2]
        if not center_half:
            center_half = valid
        bottom_idx = sorted(center_half, key=lambda idx: (values[idx], abs(idx - core_center_idx)))[0]
        return int(bottom_idx), float(values[bottom_idx]), 1, 'fallback_center_min', '坑底退化为中心半区最低点'


def inward_order_for_peak(peak_idx: int, side: str, core_center_idx: int):
    if side == 'left':
        return list(range(peak_idx, core_center_idx + 1))
    return list(range(peak_idx, core_center_idx - 1, -1))


def outer_orders_for_peak(peak_idx: int, side: str, core_lo: int, core_hi: int, ext_lo: int, ext_hi: int):
    if side == 'left':
        core_order = list(range(peak_idx - 1, core_lo - 1, -1))
        extend_order = list(range(core_lo - 1, ext_lo - 1, -1))
    else:
        core_order = list(range(peak_idx + 1, core_hi + 1))
        extend_order = list(range(core_hi + 1, ext_hi + 1))
    return core_order, extend_order


def rank_flat_segments(segments: List[Dict], peak_idx: int):
    return sorted(
        segments,
        key=lambda seg: (abs(int(seg['first_idx']) - int(peak_idx)), float(seg.get('slope_mean', np.inf)),
                         float(seg.get('elev_std', np.inf)), float(seg.get('relief', np.inf)))
    )


def classify_metrics(m: Dict) -> str:
    if not np.isfinite(m['h1']) or m['h1'] <= 0:
        return 'h1_le_0'
    if not np.isfinite(m['h2']) or m['h2'] <= m['h1']:
        return 'h2_le_h1'
    if not np.isfinite(m['h3']) or m['h3'] <= 0:
        return 'h3_le_0'
    if not np.isfinite(m['h3t']) or m['h3t'] <= 0:
        return 'h3t_le_0'
    return 'ok'


def add_global_idx_fields(rec: Dict, profile_type: str, expand_start_global: int, fixed_index_global: int):
    out = dict(rec)
    out['peak_global'] = int(expand_start_global + int(out['peak_idx'])) if np.isfinite(out.get('peak_idx', np.nan)) else np.nan
    out['outer_first_global'] = int(expand_start_global + int(out['outer_first_idx'])) if np.isfinite(out.get('outer_first_idx', np.nan)) else np.nan
    out['outer_last_global'] = int(expand_start_global + int(out['outer_last_idx'])) if np.isfinite(out.get('outer_last_idx', np.nan)) else np.nan
    out['bottom_global'] = int(expand_start_global + int(out['bottom_idx'])) if np.isfinite(out.get('bottom_idx', np.nan)) else np.nan
    if profile_type == 'row':
        out['peak_row'] = int(fixed_index_global) if np.isfinite(out['peak_global']) else np.nan
        out['peak_col'] = int(out['peak_global']) if np.isfinite(out['peak_global']) else np.nan
        out['outer_first_row'] = int(fixed_index_global) if np.isfinite(out['outer_first_global']) else np.nan
        out['outer_first_col'] = int(out['outer_first_global']) if np.isfinite(out['outer_first_global']) else np.nan
        out['outer_last_row'] = int(fixed_index_global) if np.isfinite(out['outer_last_global']) else np.nan
        out['outer_last_col'] = int(out['outer_last_global']) if np.isfinite(out['outer_last_global']) else np.nan
        out['bottom_row'] = int(fixed_index_global) if np.isfinite(out['bottom_global']) else np.nan
        out['bottom_col'] = int(out['bottom_global']) if np.isfinite(out['bottom_global']) else np.nan
    else:
        out['peak_row'] = int(out['peak_global']) if np.isfinite(out['peak_global']) else np.nan
        out['peak_col'] = int(fixed_index_global) if np.isfinite(out['peak_global']) else np.nan
        out['outer_first_row'] = int(out['outer_first_global']) if np.isfinite(out['outer_first_global']) else np.nan
        out['outer_first_col'] = int(fixed_index_global) if np.isfinite(out['outer_first_global']) else np.nan
        out['outer_last_row'] = int(out['outer_last_global']) if np.isfinite(out['outer_last_global']) else np.nan
        out['outer_last_col'] = int(fixed_index_global) if np.isfinite(out['outer_last_global']) else np.nan
        out['bottom_row'] = int(out['bottom_global']) if np.isfinite(out['bottom_global']) else np.nan
        out['bottom_col'] = int(fixed_index_global) if np.isfinite(out['bottom_global']) else np.nan
    return out


def build_metrics(base_name, profile_type, side,
                  peak_idx, peak_val, outer_seg, bottom_idx, bottom_elev,
                  line_dem_row, line_dem_col, center_row, center_col,
                  core_start_global, core_end_global,
                  expand_start_global, expand_end_global,
                  width_px, height_px, diameter_px, expand_pixels):
    outer_mean = float(outer_seg['mean'])
    h1 = float(peak_val - outer_mean)
    h2 = float(peak_val - bottom_elev)
    h3 = float(outer_mean - bottom_elev)
    h3t = float((h2 - 0.2 * h1) * 0.8)
    rec = {
        'name': base_name,
        'profile_type': profile_type,
        'side': side,
        'reason': 'ok',
        'peak_idx': int(peak_idx),
        'outer_first_idx': int(outer_seg['first_idx']),
        'outer_last_idx': int(outer_seg['last_idx']),
        'bottom_idx': int(bottom_idx),
        'peak_val': float(peak_val),
        'outer_mean': outer_mean,
        'bottom_elev': float(bottom_elev),
        'h1': h1,
        'h2': h2,
        'h3': h3,
        'h3t': h3t,
        'line_dem_row': line_dem_row,
        'line_dem_col': line_dem_col,
        'center_row': int(center_row),
        'center_col': int(center_col),
        'core_start_global': int(core_start_global),
        'core_end_global': int(core_end_global),
        'expand_start_global': int(expand_start_global),
        'expand_end_global': int(expand_end_global),
        'width_px': int(width_px),
        'height_px': int(height_px),
        'diameter_px': int(diameter_px),
        'expand_pixels': int(expand_pixels),
        'outer_kind': str(outer_seg.get('kind', OUTER_REF_TYPE_STRICT_FLAT)),
        'outer_ref_type': str(outer_seg.get('kind', OUTER_REF_TYPE_STRICT_FLAT)),
        'outer_ref_label': (
            'strict flat' if str(outer_seg.get('kind', OUTER_REF_TYPE_STRICT_FLAT)) == OUTER_REF_TYPE_STRICT_FLAT else
            'V-notch low point' if str(outer_seg.get('kind', OUTER_REF_TYPE_STRICT_FLAT)) == OUTER_REF_TYPE_V_NOTCH else
            'lowest point between peaks'
        ),
        'crater_type': '',
    }
    return rec


def evaluate_one_direction(arr, transform, boxes, reject_row: Dict, cf_arr=None):
    # globals
    core_g = boxes['core']
    exp_g = boxes['expanded']
    row_off = exp_g['rmin']
    col_off = exp_g['cmin']
    core_l = {
        'rmin': core_g['rmin'] - row_off,
        'rmax': core_g['rmax'] - row_off,
        'cmin': core_g['cmin'] - col_off,
        'cmax': core_g['cmax'] - col_off,
    }
    profile_type = str(reject_row['profile_type'])
    side = str(reject_row['side'])
    base_name = str(reject_row['name'])
    width_px = boxes['core_w']
    height_px = boxes['core_h']
    diameter_px = int(max(width_px, height_px))
    expand_pixels = int(max(boxes['expand_dx'], boxes['expand_dy']))

    if profile_type == 'row':
        fixed_global_row = int(round(float(reject_row['line_dem_row'])))
        center_col_global = int(round(float(reject_row['center_col'])))
        center_row_global = int(round(float(reject_row['center_row']))) if pd.notna(reject_row.get('center_row', np.nan)) else fixed_global_row
        local_row = fixed_global_row - row_off
        center_col_local = center_col_global - col_off
        if not (0 <= local_row < arr.shape[0]):
            return None, {'name': base_name, 'profile_type': profile_type, 'side': side, 'reason': 'line_out_of_window', 'crater_type': ''}
        values = clean_profile_values(arr[local_row, :])
        cf_values = clean_profile_values(cf_arr[local_row, :]) if cf_arr is not None else None
        cols = np.arange(arr.shape[1])
        rows = np.full(arr.shape[1], local_row, dtype=int)
        lon, lat = rasterio.transform.xy(transform, rows, cols)
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        core_lo = core_l['cmin']
        core_hi = core_l['cmax']
        core_center = center_col_local
        ext_lo = 0
        ext_hi = arr.shape[1] - 1
        fixed_index_global = fixed_global_row
        line_dem_row = fixed_global_row
        line_dem_col = np.nan
        core_start_global = core_g['cmin']
        core_end_global = core_g['cmax']
        expand_start_global = exp_g['cmin']
        expand_end_global = exp_g['cmax']
    else:
        fixed_global_col = int(round(float(reject_row['line_dem_col'])))
        center_row_global = int(round(float(reject_row['center_row'])))
        center_col_global = int(round(float(reject_row['center_col']))) if pd.notna(reject_row.get('center_col', np.nan)) else fixed_global_col
        local_col = fixed_global_col - col_off
        center_row_local = center_row_global - row_off
        if not (0 <= local_col < arr.shape[1]):
            return None, {'name': base_name, 'profile_type': profile_type, 'side': side, 'reason': 'line_out_of_window', 'crater_type': ''}
        values = clean_profile_values(arr[:, local_col])
        cf_values = clean_profile_values(cf_arr[:, local_col]) if cf_arr is not None else None
        rows = np.arange(arr.shape[0])
        cols = np.full(arr.shape[0], local_col, dtype=int)
        lon, lat = rasterio.transform.xy(transform, rows, cols)
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        core_lo = core_l['rmin']
        core_hi = core_l['rmax']
        core_center = center_row_local
        ext_lo = 0
        ext_hi = arr.shape[0] - 1
        fixed_index_global = fixed_global_col
        line_dem_row = np.nan
        line_dem_col = fixed_global_col
        core_start_global = core_g['rmin']
        core_end_global = core_g['rmax']
        expand_start_global = exp_g['rmin']
        expand_end_global = exp_g['rmax']

    slopes = calc_point_slopes(values, lon, lat)
    peak_candidates = find_rim_candidates_core_then_nearest(values, side, core_lo, core_hi, core_center, ext_lo, ext_hi, MAX_RIM_CANDIDATES)
    if not peak_candidates:
        return None, {
            'name': base_name, 'profile_type': profile_type, 'side': side, 'reason': 'no_rim_candidate',
            'line_dem_row': line_dem_row, 'line_dem_col': line_dem_col,
            'center_row': center_row_global, 'center_col': center_col_global,
            'core_start_global': core_start_global, 'core_end_global': core_end_global,
            'expand_start_global': expand_start_global, 'expand_end_global': expand_end_global,
            'width_px': width_px, 'height_px': height_px, 'diameter_px': diameter_px, 'expand_pixels': expand_pixels,
            'crater_type': '',
        }

    rejects = []
    for peak_idx in peak_candidates:
        peak_val = values[peak_idx]
        if not np.isfinite(peak_val):
            continue
        inward = inward_order_for_peak(peak_idx, side, core_center)
        if len(inward) < 2:
            continue
        bottom_idx, bottom_elev, _, floor_reason, _ = select_bottom_by_pit_rule(inward, slopes, values, core_center)
        if bottom_idx is None or not np.isfinite(bottom_elev):
            rejects.append({
                'name': base_name, 'profile_type': profile_type, 'side': side, 'reason': floor_reason or 'no_inner_flat',
                'line_dem_row': line_dem_row, 'line_dem_col': line_dem_col,
                'center_row': center_row_global, 'center_col': center_col_global,
                'core_start_global': core_start_global, 'core_end_global': core_end_global,
                'expand_start_global': expand_start_global, 'expand_end_global': expand_end_global,
                'width_px': width_px, 'height_px': height_px, 'diameter_px': diameter_px, 'expand_pixels': expand_pixels,
                'crater_type': '',
            })
            continue

        core_outer_order, extend_outer_order = outer_orders_for_peak(peak_idx, side, core_lo, core_hi, ext_lo, ext_hi)

        # 新规则：
        # 1) 先在扩大后的 outward 范围里找严格平坦区，且平坦区最高点不能超过坑顶；
        # 2) 若没有平坦区，但 outward 存在高于坑顶的顶点，则找离坑顶最近的 V 字拐点最低点；
        # 3) 若没有平坦区，但 outward 存在低于坑顶的顶点，则取坑顶与该较低顶点之间的最低点。
        outer_candidates = []
        combined_order = core_outer_order + extend_outer_order
        combined_flat = find_strict_outer_flat(combined_order, slopes, values, consecutive=3, slope_threshold=SLOPE_THRESHOLD_DEG)
        if combined_flat is not None and np.isfinite(combined_flat.get('max', np.nan)) and combined_flat['max'] <= peak_val:
            outer_candidates.append(combined_flat)
        else:
            outward_peaks = find_outward_local_peaks(combined_order, values)
            higher_peaks = [pk for pk in outward_peaks if np.isfinite(pk.get('elev', np.nan)) and pk['elev'] > peak_val]
            lower_peaks = [pk for pk in outward_peaks if np.isfinite(pk.get('elev', np.nan)) and pk['elev'] < peak_val]
            if higher_peaks:
                v_candidates = find_outer_v_notches(combined_order, values)
                v_candidates = [seg for seg in v_candidates if max(seg.get('left_peak_elev', -np.inf), seg.get('right_peak_elev', -np.inf)) > peak_val]
                outer_candidates.extend(v_candidates)
            if not outer_candidates and lower_peaks:
                nearest_lower_peak = sorted(lower_peaks, key=lambda pk: pk['near_rank'])[0]
                low_seg = build_lowest_point_between_peaks(peak_idx, nearest_lower_peak['idx'], values)
                if low_seg is not None and np.isfinite(low_seg.get('mean', np.nan)):
                    outer_candidates.append(low_seg)

        if not outer_candidates:
            rejects.append({
                'name': base_name, 'profile_type': profile_type, 'side': side, 'reason': 'no_outer_flat',
                'peak_idx': int(peak_idx), 'peak_val': float(peak_val),
                'bottom_idx': int(bottom_idx), 'bottom_elev': float(bottom_elev),
                'line_dem_row': line_dem_row, 'line_dem_col': line_dem_col,
                'center_row': center_row_global, 'center_col': center_col_global,
                'core_start_global': core_start_global, 'core_end_global': core_end_global,
                'expand_start_global': expand_start_global, 'expand_end_global': expand_end_global,
                'width_px': width_px, 'height_px': height_px, 'diameter_px': diameter_px, 'expand_pixels': expand_pixels,
                'crater_type': '',
            })
            continue

        if any(seg.get('kind') == OUTER_REF_TYPE_STRICT_FLAT for seg in outer_candidates):
            outer_candidates = rank_flat_segments([seg for seg in outer_candidates if seg.get('kind') == OUTER_REF_TYPE_STRICT_FLAT], peak_idx)
        else:
            outer_candidates = sorted(outer_candidates, key=lambda seg: (seg.get('near_rank', np.inf), -seg.get('relief', 0.0)))

        for outer_seg in outer_candidates:
            rec = build_metrics(
                base_name, profile_type, side, peak_idx, float(peak_val), outer_seg, bottom_idx, float(bottom_elev),
                line_dem_row, line_dem_col, center_row_global, center_col_global,
                core_start_global, core_end_global,
                expand_start_global, expand_end_global,
                width_px, height_px, diameter_px, expand_pixels,
            )
            rec['reason'] = classify_metrics(rec)
            rec = add_global_idx_fields(rec, profile_type, expand_start_global, fixed_index_global)
            attach_basalt_contact_columns([rec], cf_values, values, profile_type, fixed_index_global, expand_start_global, inherited_basalt_profile_info(reject_row))
            if rec['reason'] == 'ok':
                return rec, None
            rejects.append(rec)

    if rejects:
        priority = {'h2_le_h1': 0, 'h1_le_0': 1, 'no_outer_flat': 2, 'no_inner_flat': 3, 'no_valid_combination': 4}
        rejects.sort(key=lambda d: (priority.get(d.get('reason', 'no_valid_combination'), 99), 0 if d.get('outer_kind') == 'strict_flat' else 1))
        return None, rejects[0]

    return None, {
        'name': base_name, 'profile_type': profile_type, 'side': side, 'reason': 'no_valid_combination',
        'line_dem_row': line_dem_row, 'line_dem_col': line_dem_col,
        'center_row': center_row_global, 'center_col': center_col_global,
        'core_start_global': core_start_global, 'core_end_global': core_end_global,
        'expand_start_global': expand_start_global, 'expand_end_global': expand_end_global,
        'width_px': width_px, 'height_px': height_px, 'diameter_px': diameter_px, 'expand_pixels': expand_pixels,
        'crater_type': '',
    }


def draw_show_figure(show_png: Path, arr: np.ndarray, boxes_local: Dict,
                     row_rec: Optional[Dict], col_rec: Optional[Dict], base_name: str):
    core = boxes_local['core']
    expanded = boxes_local['expanded']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    img = np.array(arr, dtype=float)
    finite = np.isfinite(img)
    if np.any(finite):
        vmin = float(np.nanpercentile(img[finite], 5))
        vmax = float(np.nanpercentile(img[finite], 95))
        ax.imshow(img, cmap='gray', origin='upper', vmin=vmin, vmax=vmax)
    else:
        ax.imshow(np.zeros_like(img), cmap='gray', origin='upper')
    ax.plot([core['cmin'], core['cmax'], core['cmax'], core['cmin'], core['cmin']],
            [core['rmin'], core['rmin'], core['rmax'], core['rmax'], core['rmin']], color='b', lw=1.6)
    ax.plot([expanded['cmin'], expanded['cmax'], expanded['cmax'], expanded['cmin'], expanded['cmin']],
            [expanded['rmin'], expanded['rmin'], expanded['rmax'], expanded['rmax'], expanded['rmin']], color='g', lw=1.2)
    if row_rec is not None and np.isfinite(row_rec.get('line_dem_row', np.nan)):
        ax.axhline(int(row_rec['line_dem_row']) - boxes_local['row_off'], color='y', lw=1.0)
    if col_rec is not None and np.isfinite(col_rec.get('line_dem_col', np.nan)):
        ax.axvline(int(col_rec['line_dem_col']) - boxes_local['col_off'], color='c', lw=1.0)
    ax.set_title(base_name)

    axr = axes[1]
    if row_rec is not None and np.isfinite(row_rec.get('line_dem_row', np.nan)):
        rr_local = int(row_rec['line_dem_row']) - boxes_local['row_off']
        vals = arr[rr_local, :]
        x = np.arange(len(vals)) + boxes_local['col_off']
        axr.plot(x, vals, 'k-', lw=1.5)
        axr.axvspan(row_rec['core_start_global'], row_rec['core_end_global'], color='royalblue', alpha=0.15)
        axr.axvspan(row_rec['expand_start_global'], row_rec['expand_end_global'], color='green', alpha=0.05)
        if np.isfinite(row_rec.get('peak_global', np.nan)):
            axr.scatter([row_rec['peak_global']], [row_rec['peak_val']], c='r', s=60)
        if np.isfinite(row_rec.get('bottom_global', np.nan)):
            axr.scatter([row_rec['bottom_global']], [row_rec['bottom_elev']], c='orange', s=60)
        if np.isfinite(row_rec.get('outer_first_global', np.nan)) and np.isfinite(row_rec.get('outer_last_global', np.nan)):
            x1 = int(row_rec['outer_first_global']); x2 = int(row_rec['outer_last_global'])
            axr.plot(np.arange(x1, x2 + 1), vals[int(x1 - boxes_local['col_off']):int(x2 - boxes_local['col_off']) + 1], color='lime', lw=3)
        axr.set_title(f"Row ({row_rec.get('side','')}, {row_rec.get('reason','')}, {row_rec.get('outer_ref_type','')})")

    axc = axes[2]
    if col_rec is not None and np.isfinite(col_rec.get('line_dem_col', np.nan)):
        cc_local = int(col_rec['line_dem_col']) - boxes_local['col_off']
        vals = arr[:, cc_local]
        y = np.arange(len(vals)) + boxes_local['row_off']
        axc.plot(y, vals, 'k-', lw=1.5)
        axc.axvspan(col_rec['core_start_global'], col_rec['core_end_global'], color='royalblue', alpha=0.15)
        axc.axvspan(col_rec['expand_start_global'], col_rec['expand_end_global'], color='green', alpha=0.05)
        if np.isfinite(col_rec.get('peak_global', np.nan)):
            axc.scatter([col_rec['peak_global']], [col_rec['peak_val']], c='r', s=60)
        if np.isfinite(col_rec.get('bottom_global', np.nan)):
            axc.scatter([col_rec['bottom_global']], [col_rec['bottom_elev']], c='orange', s=60)
        if np.isfinite(col_rec.get('outer_first_global', np.nan)) and np.isfinite(col_rec.get('outer_last_global', np.nan)):
            y1 = int(col_rec['outer_first_global']); y2 = int(col_rec['outer_last_global'])
            axc.plot(np.arange(y1, y2 + 1), vals[int(y1 - boxes_local['row_off']):int(y2 - boxes_local['row_off']) + 1], color='lime', lw=3)
        axc.set_title(f"Col ({col_rec.get('side','')}, {col_rec.get('reason','')}, {col_rec.get('outer_ref_type','')})")

    plt.tight_layout()
    show_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(show_png, dpi=FIG_DPI)
    plt.close(fig)






def _safe_int(v):
    try:
        if v is None or pd.isna(v):
            return None
        vv = float(v)
        if not np.isfinite(vv):
            return None
        return int(round(vv))
    except Exception:
        return None


def inherited_basalt_profile_info(src_rec):
    """Inherit profile-level basalt-only fields from step1 reject CSV."""
    defaults = {
        'profile_between_valid_cf_count': 0,
        'profile_between_basalt_count': 0,
        'profile_between_nonbasalt_count': 0,
        'profile_both_rims_basalt': 0,
        'profile_all_between_rims_basalt': 0,
        'basalt_only_unpenetrated_flag': 0,
        'basalt_profile_reason': '',
        'left_rim_cf': np.nan,
        'right_rim_cf': np.nan,
    }
    out = {}
    for k, default in defaults.items():
        out[k] = src_rec[k] if hasattr(src_rec, 'index') and k in src_rec.index else default
    return out


def attach_basalt_contact_columns(records, cf_values, dem_values, profile_type, fixed_index_global,
                                  expand_start_global, profile_basalt_info,
                                  threshold=BASALT_CF_THRESHOLD):
    """
    Add CF-contact fields to step2 records using the remedial peak/bottom indices.
    This does not change the original Extend H123 logic.
    """
    for rec in records:
        for k, v in profile_basalt_info.items():
            rec[k] = v
        rec['basalt_cf_threshold'] = float(threshold)
        rec['peak_cf'] = np.nan
        rec['rim_type_new'] = 'unknown'
        rec['contact_idx'] = np.nan
        rec['contact_global'] = np.nan
        rec['contact_row'] = np.nan
        rec['contact_col'] = np.nan
        rec['contact_cf'] = np.nan
        rec['h_basalt_elev'] = np.nan
        rec['h2_basalt'] = np.nan
        rec['thickness_or_depth'] = np.nan
        rec['thickness_mode'] = 'not_calculated'
        rec['basalt_contact_status'] = 'not_started'
        rec['contact_note'] = ''
        # CF-based classification fields.
        # thk_ok=1 only means this side has a computable basalt thickness/burial-depth value.
        # unpenetrated_flag=1 means basalt was detected but no lower non-basalt boundary was reached.
        rec['cf_class'] = 'unknown'
        rec['thk_ok'] = 0
        rec['unpenetrated_flag'] = 0
        rec['cf_class_note'] = ''

        if cf_values is None or len(cf_values) == 0:
            rec['basalt_contact_status'] = 'missing_cf_profile'
            continue

        peak_idx = _safe_int(rec.get('peak_idx', np.nan))
        bottom_idx = _safe_int(rec.get('bottom_idx', np.nan))
        if peak_idx is None or peak_idx < 0 or peak_idx >= len(cf_values):
            rec['basalt_contact_status'] = 'invalid_peak_idx'
            continue

        peak_cf = float(cf_values[peak_idx]) if np.isfinite(cf_values[peak_idx]) else np.nan
        rec['peak_cf'] = peak_cf
        if not np.isfinite(peak_cf):
            rec['basalt_contact_status'] = 'invalid_peak_cf'
            continue

        rim_is_basalt = bool(peak_cf > threshold)
        rec['rim_type_new'] = 'basalt_rim' if rim_is_basalt else 'nonbasalt_rim'

        if int(profile_basalt_info.get('basalt_only_unpenetrated_flag', 0)) == 1:
            # This is a valid CF class, not a calculation error:
            # both rims are basalt and all valid pixels between the two rims are basalt.
            # The crater/profile reached basalt but did not penetrate to a non-basalt lower boundary,
            # so no deterministic thickness is reported here.
            rec['cf_class'] = 'unpenetrated_all_basalt'
            rec['thk_ok'] = 0
            rec['unpenetrated_flag'] = 1
            rec['thickness_mode'] = 'unpenetrated_all_basalt_no_thickness'
            rec['basalt_contact_status'] = 'unpenetrated_all_basalt'
            rec['contact_note'] = 'kept as CF class: all valid between-rim CF pixels are basalt; no non-basalt lower boundary'
            rec['cf_class_note'] = 'all basalt between rims; unpenetrated; not used as deterministic thickness'
            continue

        if bottom_idx is None or bottom_idx < 0 or bottom_idx >= len(cf_values):
            rec['basalt_contact_status'] = 'invalid_bottom_idx'
            continue
        if peak_idx == bottom_idx:
            rec['basalt_contact_status'] = 'peak_equals_bottom'
            continue

        step = 1 if bottom_idx > peak_idx else -1
        idxs = list(range(peak_idx, bottom_idx + step, step))
        if len(idxs) < 2:
            rec['basalt_contact_status'] = 'too_few_peak_to_bottom_pixels'
            continue

        contact_i = None
        if rim_is_basalt:
            for ii in idxs[1:]:
                if np.isfinite(cf_values[ii]) and cf_values[ii] <= threshold:
                    contact_i = int(ii)
                    break
            if contact_i is None:
                rec['cf_class'] = 'unpenetrated_basalt_rim'
                rec['thk_ok'] = 0
                rec['unpenetrated_flag'] = 1
                rec['thickness_mode'] = 'basalt_rim_contact_not_found'
                rec['basalt_contact_status'] = 'basalt_rim_no_nonbasalt_contact_before_bottom'
                rec['contact_note'] = 'basalt rim, but no inward non-basalt contact was found before the calculation bottom'
                rec['cf_class_note'] = 'basalt detected but lower boundary not reached; not used as deterministic thickness'
                continue
            rec['cf_class'] = 'penetrated_basalt_rim'
            rec['thickness_mode'] = 'basalt_rim_thickness'
            rec['contact_note'] = 'first non-basalt pixel after basalt rim; next pixel after last basalt pixel'
        else:
            for ii in idxs[1:]:
                if np.isfinite(cf_values[ii]) and cf_values[ii] > threshold:
                    contact_i = int(ii)
                    break
            if contact_i is None:
                rec['cf_class'] = 'no_basalt_detected'
                rec['thk_ok'] = 0
                rec['unpenetrated_flag'] = 0
                rec['thickness_mode'] = 'no_basalt_detected'
                rec['basalt_contact_status'] = 'no_basalt_detected_from_rim_to_bottom'
                rec['cf_class_note'] = 'non-basalt rim and no basalt pixel detected inward'
                continue
            rec['cf_class'] = 'buried_basalt_detected'
            rec['thickness_mode'] = 'nonbasalt_rim_burial_depth'
            rec['contact_note'] = 'first basalt pixel from non-basalt rim inward'

        if contact_i < 0 or contact_i >= len(dem_values) or not np.isfinite(dem_values[contact_i]):
            rec['basalt_contact_status'] = 'invalid_contact_dem'
            continue

        h1 = rec.get('h1', np.nan)
        peak_val = rec.get('peak_val', np.nan)
        try:
            h1 = float(h1)
            peak_val = float(peak_val)
        except Exception:
            h1 = np.nan
            peak_val = np.nan
        if not (np.isfinite(h1) and np.isfinite(peak_val)):
            rec['basalt_contact_status'] = 'invalid_h1_or_peak_val'
            continue

        h_basalt_elev = float(dem_values[contact_i])
        h2_basalt = float(peak_val - h_basalt_elev)
        thickness_or_depth = float(0.8 * (h2_basalt - 0.2 * h1))
        rec['contact_idx'] = int(contact_i)
        rec['contact_global'] = int(expand_start_global + contact_i)
        if profile_type == 'row':
            rec['contact_row'] = int(fixed_index_global)
            rec['contact_col'] = int(expand_start_global + contact_i)
        else:
            rec['contact_row'] = int(expand_start_global + contact_i)
            rec['contact_col'] = int(fixed_index_global)
        rec['contact_cf'] = float(cf_values[contact_i]) if np.isfinite(cf_values[contact_i]) else np.nan
        rec['h_basalt_elev'] = h_basalt_elev
        rec['h2_basalt'] = h2_basalt
        rec['thickness_or_depth'] = thickness_or_depth if thickness_or_depth >= 0 else np.nan
        rec['thk_ok'] = int(thickness_or_depth >= 0)
        rec['basalt_contact_status'] = 'ok' if thickness_or_depth >= 0 else 'negative_result_set_nan'
        if thickness_or_depth < 0:
            rec['cf_class_note'] = 'negative calculated value; thickness_or_depth set to NaN'

def inherit_filled_columns(out_rec: Optional[Dict], src_rec) -> Optional[Dict]:
    """
    Copy filled-crater fields from step1 reject CSV into step2 records.
    This keeps ExtendCalculateH123.py as a remedial H123 calculator only.
    """
    if out_rec is None:
        return None
    for c in FILLED_FLAG_COLS:
        if c in src_rec.index:
            out_rec[c] = src_rec[c]
        else:
            if c in ('crater_filled_flag', 'profile_filled_flag', 'filled_checked',
                     'filled_slope_le1_count', 'filled_slope_valid_count'):
                out_rec[c] = 0
            elif c in ('filled_slope_ratio_le1', 'filled_slope_threshold_deg', 'filled_ratio_threshold'):
                out_rec[c] = np.nan
            else:
                out_rec[c] = ''
    for c in BASALT_FLAG_COLS:
        if c in src_rec.index:
            out_rec[c] = src_rec[c]
        elif c in ('profile_between_valid_cf_count', 'profile_between_basalt_count', 'profile_between_nonbasalt_count',
                   'profile_both_rims_basalt', 'profile_all_between_rims_basalt', 'basalt_only_unpenetrated_flag'):
            out_rec[c] = 0
        elif c in ('basalt_cf_threshold', 'peak_cf', 'left_rim_cf', 'right_rim_cf', 'contact_idx', 'contact_global',
                   'contact_row', 'contact_col', 'contact_cf', 'h_basalt_elev', 'h2_basalt', 'thickness_or_depth'):
            out_rec[c] = np.nan
        elif c in ('rim_type_new', 'thickness_mode', 'basalt_contact_status', 'contact_note', 'basalt_profile_reason',
                   'cf_class', 'cf_class_note'):
            out_rec[c] = ''
        elif c in ('thk_ok', 'unpenetrated_flag'):
            out_rec[c] = 0
        else:
            out_rec[c] = np.nan
    return out_rec

def process_one_feature(src, cf_src, geom, base_name: str, reject_rows: pd.DataFrame, output_dir: Path):
    rmin, rmax, cmin, cmax = geom_bounds_to_rc(src, geom)
    boxes = make_boxes_from_original(rmin, rmax, cmin, cmax, src)
    arr, transform = read_window(src, boxes['expanded']['rmin'], boxes['expanded']['rmax'], boxes['expanded']['cmin'], boxes['expanded']['cmax'])
    cf_arr, _ = read_window(cf_src, boxes['expanded']['rmin'], boxes['expanded']['rmax'], boxes['expanded']['cmin'], boxes['expanded']['cmax'])
    row_off = boxes['expanded']['rmin']
    col_off = boxes['expanded']['cmin']
    boxes_local = {
        'core': {
            'rmin': boxes['core']['rmin'] - row_off,
            'rmax': boxes['core']['rmax'] - row_off,
            'cmin': boxes['core']['cmin'] - col_off,
            'cmax': boxes['core']['cmax'] - col_off,
        },
        'expanded': {'rmin': 0, 'rmax': arr.shape[0] - 1, 'cmin': 0, 'cmax': arr.shape[1] - 1},
        'row_off': row_off,
        'col_off': col_off,
    }

    valid_records = []
    reject_records = []
    row_plot = None
    col_plot = None

    # one record per rejected direction
    dedup = reject_rows.drop_duplicates(subset=['name', 'profile_type', 'side'], keep='first')
    for _, rec in dedup.iterrows():
        valid_rec, reject_rec = evaluate_one_direction(arr, transform, boxes, rec, cf_arr=cf_arr)
        if valid_rec is not None:
            valid_rec = inherit_filled_columns(valid_rec, rec)
            valid_records.append(valid_rec)
            chosen = valid_rec
        else:
            reject_rec = inherit_filled_columns(reject_rec, rec)
            reject_records.append(reject_rec)
            chosen = reject_rec
        if rec['profile_type'] == 'row' and row_plot is None:
            row_plot = chosen
        if rec['profile_type'] == 'col' and col_plot is None:
            col_plot = chosen

    if row_plot is not None or col_plot is not None:
        draw_show_figure(output_dir / PLOT_DIRNAME / f'{base_name}_show.png', arr, boxes_local, row_plot, col_plot, base_name)

    out_geoms = {
        'expanded_geom': rc_box_to_geom(src, boxes['expanded']['rmin'], boxes['expanded']['rmax'], boxes['expanded']['cmin'], boxes['expanded']['cmax']),
        'core_geom': rc_box_to_geom(src, boxes['core']['rmin'], boxes['core']['rmax'], boxes['core']['cmin'], boxes['core']['cmax']),
    }
    return valid_records, reject_records, out_geoms


def run(dem_path: Path, cf_path: Path, shp_path: Path, step1_reject_csv: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not dem_path.exists():
        raise FileNotFoundError(f'未找到 DEM：{dem_path}')
    if not cf_path.exists():
        raise FileNotFoundError(f'未找到 CF：{cf_path}')
    if not shp_path.exists():
        raise FileNotFoundError(f'未找到 SHP：{shp_path}')
    if not step1_reject_csv.exists():
        raise FileNotFoundError(f'未找到 step1 reject CSV：{step1_reject_csv}')

    step1_reject = pd.read_csv(step1_reject_csv, encoding='utf-8-sig')
    required_cols = ['name', 'profile_type', 'side']
    for c in required_cols:
        if c not in step1_reject.columns:
            raise ValueError(f'step1 reject CSV 缺少必需列：{c}')

    # 只保留四个方向记录
    step1_reject = step1_reject[step1_reject['profile_type'].isin(['row', 'col']) & step1_reject['side'].isin(['left', 'right'])].copy()
    step1_reject['name'] = step1_reject['name'].astype(str)

    valid_records = []
    reject_records = []
    expanded_geoms = []
    core_geoms = []

    with rasterio.open(dem_path) as src, rasterio.open(cf_path) as cf_src:
        if src.width != cf_src.width or src.height != cf_src.height:
            raise ValueError(f'CF 与 DEM 尺寸不一致：DEM={src.height}x{src.width}, CF={cf_src.height}x{cf_src.width}。请先配准/重采样 CF 到 DEM 网格。')
        gdf = gpd.read_file(shp_path)
        gdf, work_crs = harmonize_vector_raster_crs(gdf, src)
        field_name = infer_name_field(gdf)
        shp_name_to_geom = {}
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            base_name = safe_feature_name(row, field_name, idx)
            shp_name_to_geom[base_name] = geom

        grouped = step1_reject.groupby('name', sort=True)
        for base_name, subdf in grouped:
            geom = shp_name_to_geom.get(str(base_name))
            if geom is None:
                for _, rec in subdf.iterrows():
                    reject_records.append({
                        'name': str(base_name), 'profile_type': rec['profile_type'], 'side': rec['side'],
                        'reason': 'name_not_found_in_shp', 'crater_type': ''
                    })
                continue
            try:
                valid_i, reject_i, out_geoms = process_one_feature(src, cf_src, geom, str(base_name), subdf, output_dir)
                valid_records.extend(valid_i)
                reject_records.extend(reject_i)
                expanded_geoms.append({'name': str(base_name), 'geometry': out_geoms['expanded_geom']})
                core_geoms.append({'name': str(base_name), 'geometry': out_geoms['core_geom']})
            except Exception as e:
                for _, rec in subdf.iterrows():
                    reject_records.append({
                        'name': str(base_name), 'profile_type': rec['profile_type'], 'side': rec['side'],
                        'reason': f'exception: {e}', 'crater_type': ''
                    })

    valid_cols = [
        'name', 'profile_type', 'side', 'reason',
        'peak_val', 'outer_mean', 'bottom_elev',
        'h1', 'h2', 'h3', 'h3t', 'crater_type',
        'crater_filled_flag', 'profile_filled_flag', 'filled_checked',
        'filled_slope_ratio_le1', 'filled_slope_le1_count', 'filled_slope_valid_count',
        'filled_slope_threshold_deg', 'filled_ratio_threshold', 'filled_reason',
        'basalt_cf_threshold', 'peak_cf', 'rim_type_new',
        'profile_between_valid_cf_count', 'profile_between_basalt_count', 'profile_between_nonbasalt_count',
        'profile_both_rims_basalt', 'profile_all_between_rims_basalt',
        'basalt_only_unpenetrated_flag', 'basalt_profile_reason', 'left_rim_cf', 'right_rim_cf',
        'contact_idx', 'contact_global', 'contact_row', 'contact_col', 'contact_cf',
        'h_basalt_elev', 'h2_basalt', 'thickness_or_depth',
        'thickness_mode', 'basalt_contact_status', 'contact_note',
        'cf_class', 'thk_ok', 'unpenetrated_flag', 'cf_class_note',
        'line_dem_row', 'line_dem_col',
        'center_row', 'center_col',
        'core_start_global', 'core_end_global',
        'expand_start_global', 'expand_end_global',
        'peak_idx', 'peak_global', 'peak_row', 'peak_col',
        'outer_first_idx', 'outer_last_idx',
        'outer_first_global', 'outer_first_row', 'outer_first_col',
        'outer_last_global', 'outer_last_row', 'outer_last_col',
        'bottom_idx', 'bottom_global', 'bottom_row', 'bottom_col',
        'width_px', 'height_px', 'diameter_px', 'expand_pixels', 'outer_kind', 'outer_ref_type', 'outer_ref_label'
    ]
    reject_cols = valid_cols

    valid_df = pd.DataFrame(valid_records)
    reject_df = pd.DataFrame(reject_records)
    for c in valid_cols:
        if c not in valid_df.columns:
            valid_df[c] = np.nan
        if c not in reject_df.columns:
            reject_df[c] = np.nan
    valid_df = valid_df[valid_cols]
    reject_df = reject_df[reject_cols]

    out_csv = output_dir / VALID_CSV_NAME
    out_reject_csv = output_dir / REJECT_CSV_NAME
    valid_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    reject_df.to_csv(out_reject_csv, index=False, encoding='utf-8-sig')

    gpd.GeoDataFrame(core_geoms, geometry='geometry', crs=work_crs).to_file(output_dir / CORE_SHP_NAME, driver='ESRI Shapefile', encoding='utf-8')
    gpd.GeoDataFrame(expanded_geoms, geometry='geometry', crs=work_crs).to_file(output_dir / EXPAND_SHP_NAME, driver='ESRI Shapefile', encoding='utf-8')

    print(f'DEM 路径      : {dem_path}')
    print(f'CF 路径       : {cf_path}')
    print(f'SHP 路径      : {shp_path}')
    print(f'step1 reject  : {step1_reject_csv}')
    print(f'输出目录      : {output_dir}')
    print(f'有效结果 CSV  : {out_csv}')
    print(f'剔除结果 CSV  : {out_reject_csv}')
    print(f'core SHP      : {output_dir / CORE_SHP_NAME}')
    print(f'expanded SHP  : {output_dir / EXPAND_SHP_NAME}')
    print(f'plots 目录    : {output_dir / PLOT_DIRNAME}')
    print(f'处理方向数    : {len(valid_df) + len(reject_df)}')
    print(f'通过记录数    : {len(valid_df)}')
    print(f'剔除记录数    : {len(reject_df)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ExtendCalculateH123 step2：只读取 step1 reject 的方向，保留原 Extend core/扩大算法，outer 仅允许 平坦区 or V型最低点')
    parser.add_argument('--dem', default=str(DEFAULT_DEM_PATH), help='DEM 路径')
    parser.add_argument('--cf', default=str(DEFAULT_CF_PATH), help='CF 路径')
    parser.add_argument('--shp', default=str(DEFAULT_SHP_PATH), help='SHP 路径')
    parser.add_argument('--step1-reject', default=str(DEFAULT_STEP1_REJECT_CSV), help='step1 的 reject CSV 路径')
    parser.add_argument('--out', default=str(DEFAULT_OUTPUT_DIR), help='输出目录')
    args = parser.parse_args()
    run(Path(args.dem), Path(args.cf), Path(args.shp), Path(args.step1_reject), Path(args.out))
