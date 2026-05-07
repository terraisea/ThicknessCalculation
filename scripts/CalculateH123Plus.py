
import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

# =========================
# Default paths
# =========================
PLUS_DIR = Path(r"Database/CE5/yolo/chickin/step1/plus")
DEFAULT_CSV = None  # auto-detect first csv in PLUS_DIR if not specified
DEFAULT_DEM_DIR = PLUS_DIR
DEFAULT_OUTPUT_DIR = PLUS_DIR / "h123_from_center"
PLOT_DIRNAME = "plots"

# =========================
# Parameters (keep consistent with CalculateH123.py as much as possible)
# =========================
SLOPE_THRESHOLD_DEG = 3.0
OUTER_FLAT_CONSECUTIVE = 3
MIN_EXPAND_PIXELS = 10
MAX_EXPAND_PIXELS = 180
EXPAND_FRAC_OF_DIAMETER = 1.0
MIN_INNER_OFFSET = 1
INNER_WINDOW_SIZES = (3, 5)
INNER_OVERALL_THRESHOLD = 8.0
INVALID_LOW = -3e10
FIG_DPI = 180
SHOW_CORE_BOX = False   # 是否在png中显示由中心点+直径生成的core框

CENTRAL_UPLIFT_MIN_DIAMETER_M = 4000.0
CENTRAL_UPLIFT_LEVEL_FRAC = 0.5
CENTRAL_UPLIFT_MIN_INTERSECTIONS = 4
MARE_HCP_A = 0.075
MARE_HCP_B = 0.614

# Wall-to-floor reference parameters for central-uplift / uneven-floor cases.
# Only normal / flat craters use the minimum between two rims.
INNER_FLAT_CONSECUTIVE_NEW = 3
TYPE3_SLOPE_BREAK_BEFORE = 3
TYPE3_SLOPE_BREAK_AFTER = 3
TYPE3_MIN_INNER_FRAC = 0.15
TYPE3_MAX_INNER_FRAC = 0.90
TYPE3_DEPTH_FRAC_WEIGHT = 5.0

VERBOSE_RUN = True


# =========================
# Basic utilities
# =========================
def clean_profile_values(values, nodata=None, invalid_low=INVALID_LOW):
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    if nodata is not None and np.isfinite(nodata):
        values[values == nodata] = np.nan
    values[values < invalid_low] = np.nan
    return values


def clip_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def moon_distance_m(lon1, lat1, lon2, lat2, radius=1737400.0):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def point_distance_m(x1, y1, x2, y2, is_geographic=True):
    if is_geographic:
        return moon_distance_m(x1, y1, x2, y2)
    return float(math.hypot(x2 - x1, y2 - y1))


def calc_point_slopes(values, xs, ys, is_geographic=True):
    values = np.asarray(values, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(values)
    slopes = np.full(n, np.nan, dtype=float)

    for i in range(1, n - 1):
        if not (np.isfinite(values[i - 1]) and np.isfinite(values[i]) and np.isfinite(values[i + 1])):
            continue
        if not (np.isfinite(xs[i - 1]) and np.isfinite(ys[i - 1]) and np.isfinite(xs[i + 1]) and np.isfinite(ys[i + 1])):
            continue
        dist_m = point_distance_m(xs[i - 1], ys[i - 1], xs[i + 1], ys[i + 1], is_geographic=is_geographic)
        if not np.isfinite(dist_m) or dist_m <= 0:
            continue
        dz = values[i + 1] - values[i - 1]
        slopes[i] = abs(math.degrees(math.atan(dz / dist_m)))
    return slopes


def pixel_center_xy(transform, row, col):
    x, y = xy(transform, row, col, offset="center")
    return float(x), float(y)


def line_coords(transform, profile_type, fixed_index, start_idx, end_idx):
    xs, ys = [], []
    if profile_type == "row":
        row = fixed_index
        for col in range(start_idx, end_idx + 1):
            x, y = pixel_center_xy(transform, row, col)
            xs.append(x)
            ys.append(y)
    else:
        col = fixed_index
        for row in range(start_idx, end_idx + 1):
            x, y = pixel_center_xy(transform, row, col)
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def estimate_pixel_size_m(ds, row, col):
    row = clip_int(int(row), 0, ds.height - 1)
    col = clip_int(int(col), 0, ds.width - 1)

    x0, y0 = pixel_center_xy(ds.transform, row, col)
    if col + 1 < ds.width:
        x1, y1 = pixel_center_xy(ds.transform, row, col + 1)
    else:
        x1, y1 = pixel_center_xy(ds.transform, row, max(0, col - 1))
    if row + 1 < ds.height:
        x2, y2 = pixel_center_xy(ds.transform, row + 1, col)
    else:
        x2, y2 = pixel_center_xy(ds.transform, max(0, row - 1), col)

    is_geographic = True
    dx = point_distance_m(x0, y0, x1, y1, is_geographic=is_geographic)
    dy = point_distance_m(x0, y0, x2, y2, is_geographic=is_geographic)

    vals = [v for v in [dx, dy] if np.isfinite(v) and v > 0]
    if not vals:
        raise ValueError("Cannot estimate pixel size from DEM transform.")
    return float(np.nanmean(vals))


# =========================
# CSV parsing
# =========================
def normalize_text(s: str) -> str:
    s = str(s).strip()
    s = s.replace("（", "(").replace("）", ")").replace("，", ",")
    return s


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        raw = str(c)
        c2 = raw.strip().lower()
        c2 = c2.replace("（", "(").replace("）", ")")
        c2 = c2.replace(" ", "").replace("-", "_")
        mapping[raw] = c2
    return df.rename(columns=mapping)


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_csv_fields(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = set(df.columns)

    name_col = first_existing(df, [
        "name", "crater", "crater_name", "坑名", "撞击坑", "坑名称", "坑name"
    ])

    lon_col = first_existing(df, [
        "lon", "longitude", "center_lon", "centerlongitude", "x", "center_x",
        "中心经度", "经度", "中心点经度", "中心x", "centerlon"
    ])
    lat_col = first_existing(df, [
        "lat", "latitude", "center_lat", "centerlatitude", "y", "center_y",
        "中心纬度", "纬度", "中心点纬度", "中心y", "centerlat"
    ])

    row_col = first_existing(df, [
        "row", "center_row", "dem_row", "centerrow", "中心行", "行号", "中心row"
    ])
    col_col = first_existing(df, [
        "col", "column", "center_col", "dem_col", "centercol", "中心列", "列号", "中心col"
    ])

    diam_col = first_existing(df, [
        "diameter", "diameter_km", "diameter_m", "diameter_px", "dia", "d",
        "直径", "坑径", "直径km", "直径m", "直径px", "diameter(km)", "diameter(m)"
    ])

    return {
        "name_col": name_col,
        "lon_col": lon_col,
        "lat_col": lat_col,
        "row_col": row_col,
        "col_col": col_col,
        "diam_col": diam_col,
    }


def infer_diameter_unit(col_name: str) -> str:
    c = str(col_name).lower()
    if "px" in c:
        return "px"
    if "(m)" in c or c.endswith("_m") or c == "直径m":
        return "m"
    if "(km)" in c or "km" in c or c == "直径" or c == "坑径" or c == "直径km":
        return "km"
    return "km"


def resolve_center_rowcol(ds, rec: pd.Series, fields: Dict[str, Optional[str]]) -> Tuple[int, int, float, float, str]:
    row_col = fields["row_col"]
    col_col = fields["col_col"]
    lon_col = fields["lon_col"]
    lat_col = fields["lat_col"]

    if row_col is not None and col_col is not None and pd.notna(rec[row_col]) and pd.notna(rec[col_col]):
        row = int(round(float(rec[row_col])))
        col = int(round(float(rec[col_col])))
        row = clip_int(row, 0, ds.height - 1)
        col = clip_int(col, 0, ds.width - 1)
        x, y = pixel_center_xy(ds.transform, row, col)
        return row, col, float(x), float(y), "pixel_rowcol"

    if lon_col is None or lat_col is None:
        raise ValueError("CSV must contain either center row/col or center lon/lat.")
    lon = float(rec[lon_col])
    lat = float(rec[lat_col])
    row, col = ds.index(lon, lat)
    row = clip_int(row, 0, ds.height - 1)
    col = clip_int(col, 0, ds.width - 1)
    return row, col, lon, lat, "xy_or_lonlat"


def resolve_diameter_px(ds, rec: pd.Series, fields: Dict[str, Optional[str]], center_row: int, center_col: int) -> Tuple[int, float, str]:
    diam_col = fields["diam_col"]
    if diam_col is None or pd.isna(rec[diam_col]):
        raise ValueError("CSV missing diameter field or diameter is NaN.")

    raw_d = float(rec[diam_col])
    unit = infer_diameter_unit(diam_col)

    if unit == "px":
        diameter_px = int(round(raw_d))
        diameter_m = raw_d * estimate_pixel_size_m(ds, center_row, center_col)
        return max(3, diameter_px), float(diameter_m), "px"

    if unit == "m":
        diameter_m = raw_d
    else:
        diameter_m = raw_d * 1000.0

    px_size_m = estimate_pixel_size_m(ds, center_row, center_col)
    diameter_px = int(round(diameter_m / px_size_m))
    return max(3, diameter_px), float(diameter_m), unit


def find_matching_dem(name: str, dem_dir: Path) -> Path:
    if not dem_dir.exists():
        raise FileNotFoundError(f"DEM directory not found: {dem_dir}")

    candidates = []
    normalized_name = normalize_text(name)
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", normalized_name)

    for ext in ["*.tif", "*.TIF", "*.tiff", "*.TIFF"]:
        candidates.extend(dem_dir.glob(ext))

    if not candidates:
        raise FileNotFoundError(f"No tif/tiff DEM found in: {dem_dir}")

    # exact stem match first
    for p in candidates:
        if p.stem == safe_name or p.stem == normalized_name:
            return p

    # case-insensitive exact stem
    for p in candidates:
        if p.stem.lower() == safe_name.lower() or p.stem.lower() == normalized_name.lower():
            return p

    # contains match
    for p in candidates:
        if normalized_name.lower() in p.stem.lower():
            return p

    raise FileNotFoundError(f"No DEM matches crater name '{name}' in {dem_dir}")


def auto_find_csv(plus_dir: Path) -> Path:
    csvs = sorted(list(plus_dir.glob("*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No csv found in: {plus_dir}")
    if len(csvs) > 1:
        print(f"发现多个 CSV，默认使用第一个: {csvs[0]}")
    return csvs[0]


# =========================
# Profile logic (ported from CalculateH123.py)
# =========================
def flat_window_ok(window_slopes, slope_threshold=3.0, overall_threshold=8.0):
    ws = np.asarray(window_slopes, dtype=float)
    ws = ws[np.isfinite(ws)]
    if ws.size == 0:
        return False
    need = int(np.ceil(ws.size / 2.0))
    count_lt = int(np.sum(ws < slope_threshold))
    return (count_lt >= need) and (np.nanmax(ws) <= overall_threshold)


def valid_sequence(order_indices, slopes, values=None):
    seq = []
    for idx in order_indices:
        if idx < 0 or idx >= len(slopes):
            continue
        if not np.isfinite(slopes[idx]):
            continue
        if values is not None and not np.isfinite(values[idx]):
            continue
        seq.append(int(idx))
    return seq


def find_flat_segments(order_indices, slopes, values, window_sizes=(3, 5), slope_threshold=3.0,
                       overall_threshold=8.0, max_segments=8):
    seq = valid_sequence(order_indices, slopes, values)
    if len(seq) < 3:
        return []

    segments = []
    seen = set()

    for k in sorted(set(int(w) for w in window_sizes if int(w) >= 3)):
        if len(seq) < k:
            continue
        good = [flat_window_ok(slopes[seq[i:i + k]], slope_threshold, overall_threshold)
                for i in range(len(seq) - k + 1)]
        i = 0
        while i < len(good):
            if not good[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(good) and good[j + 1]:
                j += 1
            pts = seq[i:j + k]
            if len(pts) >= 3:
                first_idx = int(pts[0])
                last_idx = int(pts[-1])
                key = (first_idx, last_idx)
                if key not in seen:
                    seen.add(key)
                    vals = values[min(first_idx, last_idx):max(first_idx, last_idx) + 1]
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        segments.append({
                            "first_idx": first_idx,
                            "last_idx": last_idx,
                            "indices": pts,
                            "mean": float(np.nanmean(vals)),
                            "median": float(np.nanmedian(vals)),
                            "min": float(np.nanmin(vals)),
                            "n": int(vals.size),
                        })
            i = j + 1

    if not segments:
        return []

    pos_map = {idx: p for p, idx in enumerate(order_indices)}
    segments.sort(key=lambda d: (pos_map.get(d["first_idx"], 10**9), pos_map.get(d["last_idx"], 10**9)))
    return segments[:max_segments]


def choose_bottom_idx(peak_idx, inward_order, flat_seg, values, min_inner_offset=1):
    if flat_seg is None:
        return None
    first_idx = int(flat_seg["first_idx"])
    order_pos = {idx: p for p, idx in enumerate(inward_order)}
    if first_idx not in order_pos:
        return None
    p = order_pos[first_idx]
    lo = max(0, p - 2)
    hi = min(len(inward_order) - 1, p + 1)
    cand = [idx for idx in inward_order[lo:hi + 1] if np.isfinite(values[idx])]
    if not cand:
        return None
    cand = [idx for idx in cand if abs(idx - peak_idx) >= min_inner_offset]
    if not cand:
        return None
    cand.sort(key=lambda idx: (values[idx], abs(order_pos.get(idx, 10 ** 9) - p)))
    return int(cand[0])


def make_outer_order(peak_idx, side, limit_start, limit_end):
    if side == "left":
        return [i for i in range(peak_idx - 1, limit_start - 1, -1)]
    return [i for i in range(peak_idx + 1, limit_end + 1)]


def make_inner_order(peak_idx, side, core_start_idx, core_end_idx):
    if side == "left":
        return [i for i in range(peak_idx + 1, core_end_idx + 1)]
    return [i for i in range(peak_idx - 1, core_start_idx - 1, -1)]


def find_local_peaks(values, lo, hi):
    peaks = []
    lo = int(max(0, lo))
    hi = int(min(len(values) - 1, hi))
    if hi - lo < 2:
        return peaks
    for i in range(lo + 1, hi):
        if not (np.isfinite(values[i - 1]) and np.isfinite(values[i]) and np.isfinite(values[i + 1])):
            continue
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            peaks.append(i)
    return peaks


def rim_inflection_score(values, slopes, idx, side, lo, hi, win=3):
    lo = int(max(0, lo))
    hi = int(min(len(values) - 1, hi))

    if side == "left":
        outer_idx = list(range(max(lo, idx - win), idx))
        inner_idx = list(range(idx + 1, min(hi, idx + win) + 1))
    else:
        outer_idx = list(range(idx + 1, min(hi, idx + win) + 1))
        inner_idx = list(range(max(lo, idx - win), idx))

    outer_s = np.array([slopes[j] for j in outer_idx if 0 <= j < len(slopes) and np.isfinite(slopes[j])], dtype=float)
    inner_s = np.array([slopes[j] for j in inner_idx if 0 <= j < len(slopes) and np.isfinite(slopes[j])], dtype=float)

    outer_z = np.array([values[j] for j in outer_idx if 0 <= j < len(values) and np.isfinite(values[j])], dtype=float)
    inner_z = np.array([values[j] for j in inner_idx if 0 <= j < len(values) and np.isfinite(values[j])], dtype=float)

    if outer_s.size == 0 or inner_s.size == 0:
        return -np.inf, -np.inf

    slope_break = float(np.nanmean(inner_s) - np.nanmean(outer_s))
    ref_vals = np.concatenate([outer_z, inner_z]) if outer_z.size + inner_z.size > 0 else np.array([], dtype=float)
    prominence = float(values[idx] - np.nanmean(ref_vals)) if ref_vals.size > 0 else -np.inf
    return slope_break, prominence


def find_rim_idx(values, slopes, side, core_start_idx, core_end_idx, center_idx):
    """
    简化版坑顶识别：
    直接在整条剖面的左右/上下半段内取最高点，不再限制在 core 蓝框内。
    - row-left: 从中心向左整段最高点
    - row-right: 从中心向右整段最高点
    - col-left: 从中心向上整段最高点
    - col-right: 从中心向下整段最高点
    core_start_idx/core_end_idx 保留仅为了兼容旧接口，这里不再用于限制坑顶搜索范围。
    """
    if side == "left":
        lo, hi = 0, center_idx
    else:
        lo, hi = center_idx, len(values) - 1

    lo = max(0, int(lo))
    hi = min(len(values) - 1, int(hi))
    if hi < lo:
        return None

    cand = [i for i in range(lo, hi + 1) if np.isfinite(values[i])]
    if not cand:
        return None

    cand.sort(key=lambda i: (-float(values[i]), abs(i - center_idx)))
    return int(cand[0])


def outer_flat_from_first_run(values, slopes, outward_order, slope_threshold=3.0, consecutive=3):
    seq = [idx for idx in outward_order
           if 0 <= idx < len(slopes) and np.isfinite(slopes[idx]) and np.isfinite(values[idx])]
    if len(seq) < consecutive:
        return None

    run_start = None
    for i in range(len(seq) - consecutive + 1):
        w = seq[i:i + consecutive]
        if max(w) - min(w) != consecutive - 1:
            step = -1 if w[1] < w[0] else 1
            contiguous = all(w[j + 1] - w[j] == step for j in range(len(w) - 1))
            if not contiguous:
                continue
        if np.all(slopes[w] < slope_threshold):
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
        if not (np.isfinite(slopes[this_idx]) and slopes[this_idx] < slope_threshold):
            break
        end_slope = this_idx
        j += 1

    val_start = max(0, min(start_slope, end_slope) - 1)
    val_end = min(len(values) - 1, max(start_slope, end_slope) + 1)
    seg_vals = values[val_start:val_end + 1]
    seg_vals = seg_vals[np.isfinite(seg_vals)]
    if seg_vals.size == 0:
        return None

    return {
        "first_idx": int(val_start),
        "last_idx": int(val_end),
        "mean": float(np.nanmean(seg_vals)),
        "n": int(seg_vals.size),
    }


def build_metrics(name, profile_type, side, peak_idx, outer_seg, bottom_idx, values):
    peak_val = float(values[peak_idx])
    outer_mean = float(outer_seg["mean"])
    bottom_elev = float(values[bottom_idx])
    h1 = float(peak_val - outer_mean)
    h2 = float(peak_val - bottom_elev)
    h3 = float(outer_mean - bottom_elev)
    h3t = float((h2 - 0.2 * h1) * 0.8)
    return {
        "name": name,
        "profile_type": profile_type,
        "side": side,
        "reason": "ok",
        "peak_idx": int(peak_idx),
        "outer_first_idx": int(outer_seg["first_idx"]),
        "outer_last_idx": int(outer_seg["last_idx"]),
        "bottom_idx": int(bottom_idx),
        "peak_val": peak_val,
        "outer_mean": outer_mean,
        "bottom_elev": bottom_elev,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h3t": h3t,
    }


def classify_metrics(m):
    if not np.isfinite(m["h1"]) or m["h1"] <= 0:
        return "h1_le_0"
    if not np.isfinite(m["h2"]) or m["h2"] <= m["h1"]:
        return "h2_le_h1"
    if not np.isfinite(m["h3"]) or m["h3"] <= 0:
        return "h3_le_0"
    if not np.isfinite(m["h3t"]) or m["h3t"] <= 0:
        return "h3t_le_0"
    return "ok"


def mare_hcp_km(diameter_km):
    if not np.isfinite(diameter_km) or diameter_km <= 0:
        return np.nan
    return float(MARE_HCP_A * (diameter_km ** MARE_HCP_B))


def horizontal_intersection_positions(values, level, start_idx, end_idx):
    values = np.asarray(values, dtype=float)
    start_idx = int(max(0, min(start_idx, end_idx)))
    end_idx = int(min(len(values) - 1, max(start_idx, end_idx)))
    if end_idx - start_idx < 1 or not np.isfinite(level):
        return []

    positions = []
    tol = 1e-9
    for i in range(start_idx, end_idx):
        y1 = values[i]
        y2 = values[i + 1]
        if not (np.isfinite(y1) and np.isfinite(y2)):
            continue

        d1 = y1 - level
        d2 = y2 - level

        if abs(d1) <= tol and abs(d2) <= tol:
            continue
        if abs(d1) <= tol:
            pos = float(i)
        elif abs(d2) <= tol:
            pos = float(i + 1)
        elif d1 * d2 < 0:
            frac = (level - y1) / (y2 - y1)
            pos = float(i) + float(frac)
        else:
            continue

        if not positions or abs(pos - positions[-1]) > 1e-6:
            positions.append(pos)

    return positions


def count_horizontal_intersections(values, level, start_idx, end_idx):
    return len(horizontal_intersection_positions(values, level, start_idx, end_idx))


def find_min_idx_in_range(values, start_idx, end_idx):
    values = np.asarray(values, dtype=float)
    lo = int(max(0, min(start_idx, end_idx)))
    hi = int(min(len(values) - 1, max(start_idx, end_idx)))
    if hi < lo:
        return None

    seg = values[lo:hi + 1]
    finite_local = np.where(np.isfinite(seg))[0]
    if finite_local.size == 0:
        return None
    rel = int(finite_local[np.argmin(seg[finite_local])])
    return int(lo + rel)


def find_minimum_between_positions(values, pos1, pos2):
    values = np.asarray(values, dtype=float)
    lo = int(max(0, math.floor(min(pos1, pos2))))
    hi = int(min(len(values) - 1, math.ceil(max(pos1, pos2))))
    if hi < lo:
        return None, np.nan

    seg = values[lo:hi + 1]
    finite_local = np.where(np.isfinite(seg))[0]
    if finite_local.size == 0:
        return None, np.nan
    rel = int(finite_local[np.argmin(seg[finite_local])])
    idx = int(lo + rel)
    return idx, float(values[idx])


def find_highest_inner_peak(values, start_idx, end_idx, min_level=None):
    values = np.asarray(values, dtype=float)
    lo = int(max(0, min(start_idx, end_idx)))
    hi = int(min(len(values) - 1, max(start_idx, end_idx)))
    if hi < lo:
        return None

    cand = []
    for i in range(lo, hi + 1):
        if not np.isfinite(values[i]):
            continue
        if min_level is not None and np.isfinite(min_level) and values[i] < min_level:
            continue
        left_ok = True
        right_ok = True
        if i - 1 >= lo and np.isfinite(values[i - 1]):
            left_ok = values[i] >= values[i - 1]
        if i + 1 <= hi and np.isfinite(values[i + 1]):
            right_ok = values[i] >= values[i + 1]
        if left_ok and right_ok:
            cand.append(i)

    if not cand:
        cand = [i for i in range(lo, hi + 1)
                if np.isfinite(values[i]) and (min_level is None or not np.isfinite(min_level) or values[i] >= min_level)]
    if not cand:
        return None

    center = 0.5 * (lo + hi)
    cand.sort(key=lambda i: (-values[i], abs(i - center)))
    return int(cand[0])


def analyze_central_uplift_profile_geometry(values, xs, ys, core_start_idx, core_end_idx, center_idx):
    info = {
        'geometry_ok': 0,
        'reason': 'not_checked',
        'rim_distance_m': np.nan,
        'left_peak_idx': np.nan,
        'right_peak_idx': np.nan,
        'left_dir_bottom_idx': np.nan,
        'left_dir_bottom_elev': np.nan,
        'right_dir_bottom_idx': np.nan,
        'right_dir_bottom_elev': np.nan,
    }

    slopes = calc_point_slopes(values, xs, ys, is_geographic=True)
    left_peak_idx = find_rim_idx(values, slopes, 'left', core_start_idx, core_end_idx, center_idx)
    right_peak_idx = find_rim_idx(values, slopes, 'right', core_start_idx, core_end_idx, center_idx)
    if left_peak_idx is None or right_peak_idx is None:
        info['reason'] = 'no_two_rims'
        return info
    if right_peak_idx <= left_peak_idx:
        info['reason'] = 'invalid_rim_order'
        return info

    rim_distance_m = point_distance_m(
        xs[left_peak_idx], ys[left_peak_idx],
        xs[right_peak_idx], ys[right_peak_idx],
        is_geographic=True
    )

    info.update({
        'geometry_ok': 1,
        'reason': 'ok',
        'rim_distance_m': float(rim_distance_m),
        'left_peak_idx': int(left_peak_idx),
        'right_peak_idx': int(right_peak_idx),
    })

    left_dir_bottom_idx = find_min_idx_in_range(values, left_peak_idx, center_idx)
    right_dir_bottom_idx = find_min_idx_in_range(values, center_idx, right_peak_idx)

    if left_dir_bottom_idx is not None and np.isfinite(values[left_dir_bottom_idx]):
        info['left_dir_bottom_idx'] = int(left_dir_bottom_idx)
        info['left_dir_bottom_elev'] = float(values[left_dir_bottom_idx])

    if right_dir_bottom_idx is not None and np.isfinite(values[right_dir_bottom_idx]):
        info['right_dir_bottom_idx'] = int(right_dir_bottom_idx)
        info['right_dir_bottom_elev'] = float(values[right_dir_bottom_idx])

    return info


def build_non_cu_bottom_map(values, xs, ys, core_start_idx, core_end_idx, center_idx):
    geom = analyze_central_uplift_profile_geometry(values, xs, ys, core_start_idx, core_end_idx, center_idx)
    if int(geom.get('geometry_ok', 0)) != 1:
        return None

    left_peak_idx = int(geom['left_peak_idx'])
    right_peak_idx = int(geom['right_peak_idx'])
    floor_idx = find_min_idx_in_range(values, left_peak_idx, right_peak_idx)
    if floor_idx is None or not np.isfinite(values[floor_idx]):
        return None

    return {"left": int(floor_idx), "right": int(floor_idx)}


def make_rim_to_center_order(peak_idx, center_idx):
    """
    Build an ordered index list from crater rim toward the center axis.
    The rim point itself is excluded; the center point is included.
    """
    peak_idx = int(peak_idx)
    center_idx = int(center_idx)
    if peak_idx < center_idx:
        return [i for i in range(peak_idx + 1, center_idx + 1)]
    if peak_idx > center_idx:
        return [i for i in range(peak_idx - 1, center_idx - 1, -1)]
    return []


def find_first_wall_to_flat_idx(values, slopes, order_indices,
                                slope_threshold=SLOPE_THRESHOLD_DEG,
                                consecutive=INNER_FLAT_CONSECUTIVE_NEW):
    """
    For central-uplift craters, the calculation bottom_elev should be the point
    where the crater wall first enters the floor. If a continuous <=3 deg floor
    segment exists, return the first point of that segment.
    """
    seq = [int(i) for i in order_indices
           if 0 <= int(i) < len(slopes)
           and np.isfinite(values[int(i)])
           and np.isfinite(slopes[int(i)])]

    if len(seq) < consecutive:
        return None, "no_wall_to_flat_run"

    for pos in range(0, len(seq) - consecutive + 1):
        w = seq[pos:pos + consecutive]

        # The points must be adjacent along the profile.
        step = 1 if w[-1] > w[0] else -1
        if not all(w[j + 1] - w[j] == step for j in range(len(w) - 1)):
            continue

        if np.all(np.asarray([slopes[i] for i in w], dtype=float) <= slope_threshold):
            return int(w[0]), "wall_to_flat_first_le3"

    return None, "no_wall_to_flat_run"


def find_wall_floor_kink_idx(values, slopes, order_indices, peak_idx, center_idx,
                             before_n=TYPE3_SLOPE_BREAK_BEFORE,
                             after_n=TYPE3_SLOPE_BREAK_AFTER,
                             min_inner_frac=TYPE3_MIN_INNER_FRAC,
                             max_inner_frac=TYPE3_MAX_INNER_FRAC,
                             depth_weight=TYPE3_DEPTH_FRAC_WEIGHT):
    """
    If no continuous <=3 deg floor exists, identify the wall-to-floor kink.
    This follows the previously agreed engineering rule:

        Score = slope_drop + 5.0 * depth_frac

    where slope_drop is the median slope before the candidate minus the median
    slope after the candidate, and depth_frac is clipped to 0-1.
    """
    seq = [int(i) for i in order_indices
           if 0 <= int(i) < len(slopes)
           and np.isfinite(values[int(i)])]

    if len(seq) < before_n + after_n + 1:
        fallback = find_min_idx_in_range(values, peak_idx, center_idx)
        return fallback, "kink_fallback_min_between_rim_center"

    local_min_idx = find_min_idx_in_range(values, peak_idx, center_idx)
    peak_val = values[peak_idx] if 0 <= int(peak_idx) < len(values) and np.isfinite(values[int(peak_idx)]) else np.nan
    local_min_val = values[local_min_idx] if local_min_idx is not None and np.isfinite(values[local_min_idx]) else np.nan
    denom = peak_val - local_min_val if np.isfinite(peak_val) and np.isfinite(local_min_val) else np.nan

    scores = []
    n = len(seq)

    for pos in range(before_n, n - after_n):
        idx = seq[pos]
        if not np.isfinite(values[idx]):
            continue

        inner_frac = pos / max(n - 1, 1)
        if inner_frac < min_inner_frac or inner_frac > max_inner_frac:
            continue

        before_vals = np.array([slopes[seq[j]] for j in range(pos - before_n, pos)
                                if np.isfinite(slopes[seq[j]])], dtype=float)
        after_vals = np.array([slopes[seq[j]] for j in range(pos, pos + after_n)
                               if np.isfinite(slopes[seq[j]])], dtype=float)
        if before_vals.size == 0 or after_vals.size == 0:
            continue

        before_med = float(np.nanmedian(before_vals))
        after_med = float(np.nanmedian(after_vals))
        slope_drop = before_med - after_med

        if np.isfinite(denom) and abs(denom) > 1e-9:
            depth_frac = (peak_val - values[idx]) / denom
        else:
            depth_frac = 0.0
        depth_frac = float(np.clip(depth_frac, 0.0, 1.0)) if np.isfinite(depth_frac) else 0.0

        score = slope_drop + depth_weight * depth_frac

        scores.append({
            "idx": int(idx),
            "score": float(score),
            "slope_drop": float(slope_drop),
            "depth_frac": float(depth_frac),
        })

    if not scores:
        fallback = find_min_idx_in_range(values, peak_idx, center_idx)
        return fallback, "kink_fallback_min_between_rim_center"

    scores.sort(key=lambda d: (-d["score"], -d["slope_drop"], -d["depth_frac"]))
    best = scores[0]

    if best["slope_drop"] <= 0:
        return int(best["idx"]), "weak_kink_best_available"
    return int(best["idx"]), "wall_floor_kink_score"


def find_floor_flat_near_side_min_idx(values, slopes, order_indices, side_min_idx,
                                      slope_threshold=SLOPE_THRESHOLD_DEG,
                                      consecutive=INNER_FLAT_CONSECUTIVE_NEW):
    """
    Find the floor-like flat segment within the rim-to-side-minimum descending
    interval. This is different from simply taking the first <=3 deg run, because
    a small flat bench can occur high on the crater wall.

    Strategy:
      1. Work only in the ordered interval from rim toward the side minimum.
      2. Find all contiguous runs whose slope is <= slope_threshold.
      3. Keep runs with length >= consecutive.
      4. Select the run whose inner end is closest to side_min_idx; this makes
         the selected flat run floor-side rather than rim-wall-side.
      5. Return the first point of that selected run, i.e. the wall-to-floor
         entry point.
    """
    side_min_idx = int(side_min_idx)
    seq = [int(i) for i in order_indices
           if 0 <= int(i) < len(slopes)
           and np.isfinite(values[int(i)])
           and np.isfinite(slopes[int(i)])]

    if len(seq) < consecutive:
        return None, "no_floor_flat_near_side_min"

    runs = []
    run = []

    for idx in seq:
        idx = int(idx)
        good = np.isfinite(slopes[idx]) and slopes[idx] <= slope_threshold
        if good:
            if run:
                expected_step = 1 if run[-1] < idx else -1
                if idx - run[-1] == expected_step:
                    run.append(idx)
                else:
                    if len(run) >= consecutive:
                        runs.append(run)
                    run = [idx]
            else:
                run = [idx]
        else:
            if len(run) >= consecutive:
                runs.append(run)
            run = []

    if len(run) >= consecutive:
        runs.append(run)

    if not runs:
        return None, "no_floor_flat_near_side_min"

    candidates = []
    for r in runs:
        vals = np.asarray([values[i] for i in r if np.isfinite(values[i])], dtype=float)
        if vals.size == 0:
            continue
        inner_end = int(r[-1])  # order is rim -> side_min, so last point is floor-side end
        candidates.append({
            "start_idx": int(r[0]),
            "end_idx": inner_end,
            "dist_to_side_min": abs(inner_end - side_min_idx),
            "mean_elev": float(np.nanmean(vals)),
            "min_elev": float(np.nanmin(vals)),
            "n": int(len(r)),
        })

    if not candidates:
        return None, "no_floor_flat_near_side_min"

    candidates.sort(key=lambda d: (d["dist_to_side_min"], d["mean_elev"], -d["n"]))
    best = candidates[0]
    return int(best["start_idx"]), "floor_flat_near_side_min_first_point"


def find_wall_to_floor_reference_idx(values, slopes, peak_idx, center_idx):
    """
    Bottom-elevation rule for central-uplift or uneven-floor cases.

    Normal/flat craters are handled elsewhere by taking the minimum between two
    rims. For central-uplift or uneven-floor cases, the center axis is only used
    to split the two sides. The actual wall-to-floor search is limited to the
    descending interval from rim to the side-local minimum, not all the way to
    the crater center.

    1. Find side_min_idx between peak_idx and center_idx. It is only the search
       endpoint, not the final bottom_elev by default.
    2. Search from peak_idx to side_min_idx.
    3. If a floor-like <=3 deg segment exists, select the segment nearest to
       side_min_idx and return its first point as the wall-to-floor transition.
    4. If no such flat segment exists, use the agreed kink score:
       Score = slope_drop + 5.0 * depth_frac.
    5. Only if both fail, fall back to side_min_idx.
    """
    side_min_idx = find_min_idx_in_range(values, peak_idx, center_idx)
    if side_min_idx is None or not np.isfinite(values[side_min_idx]):
        return np.nan, "no_side_min"

    order = make_rim_to_center_order(peak_idx, side_min_idx)
    if len(order) == 0:
        return int(side_min_idx), "fallback_side_min_empty_order"

    # 1. Prefer the flat-floor segment nearest to the side minimum, not the
    # first local <=3 deg bench on the upper wall.
    flat_idx, flat_reason = find_floor_flat_near_side_min_idx(
        values=values,
        slopes=slopes,
        order_indices=order,
        side_min_idx=side_min_idx,
    )
    if flat_idx is not None and np.isfinite(values[flat_idx]):
        return int(flat_idx), flat_reason

    # 2. If no stable <=3 deg floor segment exists, use the agreed kink score
    # only within the rim-to-side-minimum interval.
    kink_idx, kink_reason = find_wall_floor_kink_idx(
        values=values,
        slopes=slopes,
        order_indices=order,
        peak_idx=peak_idx,
        center_idx=side_min_idx,
    )
    if kink_idx is not None and np.isfinite(values[kink_idx]):
        return int(kink_idx), kink_reason

    return int(side_min_idx), "fallback_side_min"


def build_center_split_bottom_map(values, xs, ys, core_start_idx, core_end_idx, center_idx):
    """
    For central-uplift / uneven-floor cases, do NOT use the center-side minimum
    as bottom_elev except as the final fallback. Use the wall-to-floor transition
    on each side:

      - First find the side-local minimum between the rim and the center axis;
        this limits the search interval and removes central-peak/interior-floor
        roughness from the candidate set.
      - If a continuous <=3 deg flat-floor segment exists in this descending
        interval, choose the flat segment nearest to the side minimum and take
        its first point as the wall-to-floor transition.
      - If no <=3 deg flat segment exists, take the wall-to-floor kink determined
        by Score = slope_drop + 5.0 * depth_frac within the rim-to-side-minimum
        interval.
      - The side minimum is used only as the final fallback.

    The center axis is only used to separate the left and right search domains.
    """
    geom = analyze_central_uplift_profile_geometry(values, xs, ys, core_start_idx, core_end_idx, center_idx)
    if int(geom.get('geometry_ok', 0)) != 1:
        return None

    slopes = calc_point_slopes(values, xs, ys, is_geographic=True)

    left_peak_idx = int(geom['left_peak_idx'])
    right_peak_idx = int(geom['right_peak_idx'])

    left_idx, left_rule = find_wall_to_floor_reference_idx(
        values=values,
        slopes=slopes,
        peak_idx=left_peak_idx,
        center_idx=center_idx,
    )
    right_idx, right_rule = find_wall_to_floor_reference_idx(
        values=values,
        slopes=slopes,
        peak_idx=right_peak_idx,
        center_idx=center_idx,
    )

    # Keep values as raw indices for compatibility with evaluate_side().
    # The plotting code will still display one green marker per side.
    left_idx = int(left_idx) if np.isfinite(left_idx) else np.nan
    right_idx = int(right_idx) if np.isfinite(right_idx) else np.nan

    return {"left": left_idx, "right": right_idx}


def detect_central_uplift_profile(values, xs, ys, core_start_idx, core_end_idx, center_idx,
                                  crater_bottom_elev=None, crater_diameter_m=None):
    info = {
        'checked': 0,
        'flag': 0,
        'reason': 'not_checked',
        'rim_distance_m': np.nan,
        'left_peak_idx': np.nan,
        'right_peak_idx': np.nan,
        'bottom_idx': np.nan,
        'bottom_elev': np.nan,
        'test_level': np.nan,
        'intersections': np.nan,
        'left_floor_idx': np.nan,
        'left_floor_elev': np.nan,
        'right_floor_idx': np.nan,
        'right_floor_elev': np.nan,
        'hcp_km': np.nan,
        'test_base_elev': np.nan,
    }

    geom = analyze_central_uplift_profile_geometry(values, xs, ys, core_start_idx, core_end_idx, center_idx)
    info.update({
        'rim_distance_m': geom.get('rim_distance_m', np.nan),
        'left_peak_idx': geom.get('left_peak_idx', np.nan),
        'right_peak_idx': geom.get('right_peak_idx', np.nan),
    })

    if int(geom.get('geometry_ok', 0)) != 1:
        info['reason'] = geom.get('reason', 'invalid_geometry')
        return info

    if not np.isfinite(crater_diameter_m) or crater_diameter_m < CENTRAL_UPLIFT_MIN_DIAMETER_M:
        info['checked'] = 1
        info['reason'] = 'diameter_lt_4km'
        return info

    base_candidates = []
    for k in ['left_dir_bottom_elev', 'right_dir_bottom_elev']:
        v = geom.get(k, np.nan)
        if np.isfinite(v):
            base_candidates.append(float(v))

    if len(base_candidates) >= 2:
        test_base_elev = float(np.nanmean(base_candidates))
    elif len(base_candidates) == 1:
        test_base_elev = float(base_candidates[0])
    elif np.isfinite(crater_bottom_elev):
        test_base_elev = float(crater_bottom_elev)
    else:
        info['checked'] = 1
        info['reason'] = 'no_local_bottom'
        return info

    hcp_km = mare_hcp_km(crater_diameter_m / 1000.0)
    if not np.isfinite(hcp_km) or hcp_km <= 0:
        info['checked'] = 1
        info['reason'] = 'invalid_hcp'
        return info

    test_level = float(test_base_elev + CENTRAL_UPLIFT_LEVEL_FRAC * hcp_km * 1000.0)
    left_peak_idx = int(geom['left_peak_idx'])
    right_peak_idx = int(geom['right_peak_idx'])

    intersections_pos = horizontal_intersection_positions(values, test_level, left_peak_idx, right_peak_idx)
    intersections = len(intersections_pos)

    info.update({
        'checked': 1,
        'bottom_elev': float(test_base_elev),
        'test_base_elev': float(test_base_elev),
        'test_level': test_level,
        'intersections': int(intersections),
        'hcp_km': float(hcp_km),
    })

    if intersections < CENTRAL_UPLIFT_MIN_INTERSECTIONS:
        info['reason'] = 'lt4_intersections'
        return info

    p1, p2, p3, p4 = intersections_pos[:4]

    left_floor_idx, left_floor_elev = find_minimum_between_positions(values, p1, p2)
    right_floor_idx, right_floor_elev = find_minimum_between_positions(values, p3, p4)

    if left_floor_idx is not None:
        info['left_floor_idx'] = int(left_floor_idx)
        info['left_floor_elev'] = float(left_floor_elev)
    if right_floor_idx is not None:
        info['right_floor_idx'] = int(right_floor_idx)
        info['right_floor_elev'] = float(right_floor_elev)

    if np.isfinite(info['left_floor_elev']) and np.isfinite(info['right_floor_elev']):
        if info['left_floor_elev'] <= info['right_floor_elev']:
            info['bottom_idx'] = int(info['left_floor_idx'])
            info['bottom_elev'] = float(info['left_floor_elev'])
        else:
            info['bottom_idx'] = int(info['right_floor_idx'])
            info['bottom_elev'] = float(info['right_floor_elev'])

    info['flag'] = 1
    info['reason'] = 'ok'
    return info


def attach_uplift_columns(records, crater_has_central_uplift):
    crater_type = 'central uplift' if crater_has_central_uplift else 'non-central uplift'
    for rec in records:
        rec['crater_type'] = crater_type


def evaluate_side(values, xs, ys, profile_type, side, name,
                  core_start_idx, core_end_idx, center_idx,
                  slope_threshold=3.0, outer_consecutive=3, min_inner_offset=1,
                  forced_bottom_idx=None):
    slopes = calc_point_slopes(values, xs, ys, is_geographic=True)
    peak_idx = find_rim_idx(values, slopes, side, core_start_idx, core_end_idx, center_idx)
    if peak_idx is None:
        return None, {"name": name, "profile_type": profile_type, "side": side, "reason": "no_rim_candidate"}

    outward_order = make_outer_order(peak_idx, side, 0, len(values) - 1)
    outer_seg = outer_flat_from_first_run(values, slopes, outward_order,
                                          slope_threshold=slope_threshold,
                                          consecutive=outer_consecutive)
    if outer_seg is None:
        return None, {
            "name": name,
            "profile_type": profile_type,
            "side": side,
            "reason": "no_outer_flat",
            "peak_idx": int(peak_idx),
            "peak_val": float(values[peak_idx]) if np.isfinite(values[peak_idx]) else np.nan,
        }

    if forced_bottom_idx is not None and np.isfinite(forced_bottom_idx):
        bottom_idx = int(forced_bottom_idx)
        if bottom_idx < 0 or bottom_idx >= len(values) or not np.isfinite(values[bottom_idx]):
            return None, {
                "name": name,
                "profile_type": profile_type,
                "side": side,
                "reason": "invalid_forced_bottom",
                "peak_idx": int(peak_idx),
                "peak_val": float(values[peak_idx]) if np.isfinite(values[peak_idx]) else np.nan,
                "outer_mean": float(outer_seg["mean"]),
            }

        m = build_metrics(name, profile_type, side, peak_idx, outer_seg, bottom_idx, values)
        m["reason"] = classify_metrics(m)
        if m["reason"] == "ok":
            m["slopes"] = slopes
            return m, None
        return None, m

    inward_order = make_inner_order(peak_idx, side, core_start_idx, core_end_idx)
    inner_segs = find_flat_segments(
        inward_order, slopes, values,
        window_sizes=INNER_WINDOW_SIZES,
        slope_threshold=slope_threshold,
        overall_threshold=INNER_OVERALL_THRESHOLD,
        max_segments=6,
    )
    if not inner_segs:
        return None, {
            "name": name,
            "profile_type": profile_type,
            "side": side,
            "reason": "no_inner_flat",
            "peak_idx": int(peak_idx),
            "outer_first_idx": int(outer_seg["first_idx"]),
            "outer_last_idx": int(outer_seg["last_idx"]),
            "peak_val": float(values[peak_idx]) if np.isfinite(values[peak_idx]) else np.nan,
            "outer_mean": float(outer_seg["mean"]),
        }

    reject_snapshots = []
    for flat_seg in inner_segs:
        bottom_idx = choose_bottom_idx(
            peak_idx=peak_idx,
            inward_order=inward_order,
            flat_seg=flat_seg,
            values=values,
            min_inner_offset=min_inner_offset,
        )
        if bottom_idx is None:
            continue
        if not np.isfinite(values[bottom_idx]):
            continue
        if values[bottom_idx] >= values[peak_idx]:
            continue

        m = build_metrics(name, profile_type, side, peak_idx, outer_seg, bottom_idx, values)
        m["reason"] = classify_metrics(m)
        if m["reason"] == "ok":
            m["slopes"] = slopes
            return m, None
        reject_snapshots.append(m)

    if reject_snapshots:
        priority = {"h3_le_0": 0, "h2_le_h1": 1, "h1_le_0": 2, "h3t_le_0": 3, "invalid_geometry": 4}
        reject_snapshots.sort(key=lambda d: priority.get(d.get("reason", "invalid_geometry"), 99))
        return None, reject_snapshots[0]

    return None, {
        "name": name,
        "profile_type": profile_type,
        "side": side,
        "reason": "no_valid_combination",
        "peak_idx": int(peak_idx),
        "peak_val": float(values[peak_idx]) if np.isfinite(values[peak_idx]) else np.nan,
        "outer_mean": float(outer_seg["mean"]) if outer_seg is not None else np.nan,
    }


def add_local_fields(record, profile_type, fixed_index, line_start_idx):
    out = dict(record)
    out["line_dem_row"] = int(fixed_index) if profile_type == "row" else np.nan
    out["line_dem_col"] = int(fixed_index) if profile_type == "col" else np.nan

    if "peak_idx" in out and np.isfinite(out.get("peak_idx", np.nan)):
        out["peak_local"] = int(line_start_idx + int(out["peak_idx"]))
    else:
        out["peak_local"] = np.nan

    if "outer_first_idx" in out and np.isfinite(out.get("outer_first_idx", np.nan)):
        out["outer_first_local"] = int(line_start_idx + int(out["outer_first_idx"]))
    else:
        out["outer_first_local"] = np.nan

    if "outer_last_idx" in out and np.isfinite(out.get("outer_last_idx", np.nan)):
        out["outer_last_local"] = int(line_start_idx + int(out["outer_last_idx"]))
    else:
        out["outer_last_local"] = np.nan

    if "bottom_idx" in out and np.isfinite(out.get("bottom_idx", np.nan)):
        out["bottom_local"] = int(line_start_idx + int(out["bottom_idx"]))
    else:
        out["bottom_local"] = np.nan

    if profile_type == "row":
        out["peak_row"] = int(fixed_index) if np.isfinite(out["peak_local"]) else np.nan
        out["peak_col"] = int(out["peak_local"]) if np.isfinite(out["peak_local"]) else np.nan
        out["outer_first_row"] = int(fixed_index) if np.isfinite(out["outer_first_local"]) else np.nan
        out["outer_first_col"] = int(out["outer_first_local"]) if np.isfinite(out["outer_first_local"]) else np.nan
        out["outer_last_row"] = int(fixed_index) if np.isfinite(out["outer_last_local"]) else np.nan
        out["outer_last_col"] = int(out["outer_last_local"]) if np.isfinite(out["outer_last_local"]) else np.nan
        out["bottom_row"] = int(fixed_index) if np.isfinite(out["bottom_local"]) else np.nan
        out["bottom_col"] = int(out["bottom_local"]) if np.isfinite(out["bottom_local"]) else np.nan
    else:
        out["peak_row"] = int(out["peak_local"]) if np.isfinite(out["peak_local"]) else np.nan
        out["peak_col"] = int(fixed_index) if np.isfinite(out["peak_local"]) else np.nan
        out["outer_first_row"] = int(out["outer_first_local"]) if np.isfinite(out["outer_first_local"]) else np.nan
        out["outer_first_col"] = int(fixed_index) if np.isfinite(out["outer_first_local"]) else np.nan
        out["outer_last_row"] = int(out["outer_last_local"]) if np.isfinite(out["outer_last_local"]) else np.nan
        out["outer_last_col"] = int(fixed_index) if np.isfinite(out["outer_last_local"]) else np.nan
        out["bottom_row"] = int(out["bottom_local"]) if np.isfinite(out["bottom_local"]) else np.nan
        out["bottom_col"] = int(fixed_index) if np.isfinite(out["bottom_local"]) else np.nan
    return out



def bottom_entry_to_idx(entry):
    """
    Return the bottom index from either a raw index or a dict entry such as {"idx": ...}.
    This keeps the plotting code compatible with both the old two-class bottom map
    and the newer three-class bottom map.
    """
    if isinstance(entry, dict):
        return entry.get("idx", None)
    return entry


def build_bottom_marker_from_entry(entry, side, values, profile_type, fixed_index, line_start_idx):
    """
    Build a lightweight plotting record for a bottom_elev point.
    This is used so that non-normal craters can show both left-side and right-side
    bottom_elev points even when only one side is selected as the representative
    plot_record.
    """
    idx = bottom_entry_to_idx(entry)
    if idx is None:
        return None
    try:
        if not np.isfinite(idx):
            return None
        idx = int(idx)
    except Exception:
        return None

    if idx < 0 or idx >= len(values) or not np.isfinite(values[idx]):
        return None

    marker = {
        "profile_type": profile_type,
        "side": side,
        "bottom_idx": int(idx),
        "bottom_elev": float(values[idx]),
        "reason": "bottom_marker",
    }
    return add_local_fields(marker, profile_type, fixed_index, line_start_idx)


def dedupe_bottom_markers(markers):
    """
    Remove duplicated bottom markers. In type-1/normal cases, left and right often
    share the same bottom point, so only one green point is drawn.
    """
    out = []
    seen = set()
    for rec in markers or []:
        if rec is None:
            continue
        if not (np.isfinite(rec.get("bottom_local", np.nan)) and np.isfinite(rec.get("bottom_elev", np.nan))):
            continue
        key = (
            int(rec.get("bottom_local")),
            round(float(rec.get("bottom_elev")), 6),
            rec.get("profile_type", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def pick_plot_record(valid_left, valid_right, reject_left, reject_right):
    valids = [r for r in [valid_left, valid_right] if r is not None]
    if valids:
        valids.sort(key=lambda d: (d.get("h3t", -np.inf), d.get("h3", -np.inf)), reverse=True)
        return valids[0]
    rejects = [r for r in [reject_left, reject_right] if r is not None]
    if rejects:
        priority = {"h3_le_0": 0, "h2_le_h1": 1, "h1_le_0": 2, "h3t_le_0": 3,
                    "no_inner_flat": 4, "no_outer_flat": 5, "no_valid_combination": 6, "no_rim_candidate": 7}
        rejects.sort(key=lambda d: priority.get(d.get("reason", "no_valid_combination"), 99))
        return rejects[0]
    return None


def draw_profile_panel(ax, values, profile_type, core_start_idx, core_end_idx,
                       plot_record, line_start_idx, uplift_info=None, bottom_markers=None):
    x_local = np.arange(len(values), dtype=int)
    x_dem = line_start_idx + x_local
    ax.axvspan(x_dem[0], x_dem[-1], color="#dfe8df", alpha=0.85, zorder=0)
    ax.axvspan(line_start_idx + core_start_idx, line_start_idx + core_end_idx, color="#c5d2e3", alpha=0.95, zorder=1)
    ax.plot(x_dem, values, color="black", linewidth=1.3, zorder=3)

    title = f"{profile_type.capitalize()} (no valid side)"
    if plot_record is not None:
        side = plot_record.get("side", "none")
        reason = plot_record.get("reason", "unknown")
        title = f"{profile_type.capitalize()} ({side}, {reason})"
        if uplift_info is not None and int(uplift_info.get("checked", 0)) == 1:
            flag_val = uplift_info.get('flag', 0)
            inter_val = uplift_info.get('intersections', np.nan)
            flag_txt = int(flag_val) if np.isfinite(flag_val) else 'nan'
            inter_txt = int(inter_val) if np.isfinite(inter_val) else 'nan'
            title += f" | CU={flag_txt}, N={inter_txt}"

        if np.isfinite(plot_record.get("peak_local", np.nan)) and np.isfinite(plot_record.get("peak_val", np.nan)):
            ax.scatter(plot_record["peak_local"], plot_record["peak_val"], color="red", s=20, zorder=5)

    # Green point(s): actual point(s) whose elevations are used as bottom_elev.
    # For type-1/normal cases, left and right usually share one bottom point after deduplication.
    # For non-normal cases, left and right are plotted separately using the center as the boundary.
    all_bottom_markers = []
    if bottom_markers:
        all_bottom_markers.extend(bottom_markers)
    elif plot_record is not None:
        all_bottom_markers.append(plot_record)

    for marker in dedupe_bottom_markers(all_bottom_markers):
        ax.scatter(marker["bottom_local"], marker["bottom_elev"], color="green", s=32, zorder=6)

    if uplift_info is not None and int(uplift_info.get("checked", 0)) == 1 and np.isfinite(uplift_info.get("test_level", np.nan)):
        ax.axhline(float(uplift_info["test_level"]), color="magenta", linestyle="--", linewidth=1.0, zorder=2)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"DEM {'col' if profile_type == 'row' else 'row'}")
    ax.set_ylabel("Elevation")
    ax.grid(True, alpha=0.25)


def draw_map_panel(ax, patch, core_start_col_local, core_start_row_local, core_w, core_h,
                   center_local_row, center_local_col,
                   row_plot_record, col_plot_record, title):
    ax.imshow(patch, cmap="gray", origin="upper")
    h, w = patch.shape
    ax.add_patch(plt.Rectangle((0, 0), w - 1, h - 1, fill=False, edgecolor="limegreen", linewidth=2.0))
    if SHOW_CORE_BOX:
        ax.add_patch(plt.Rectangle((core_start_col_local, core_start_row_local), core_w, core_h,
                                   fill=False, edgecolor="blue", linewidth=2.0))
    ax.axhline(center_local_row, color="yellow", linewidth=1.0)
    ax.axvline(center_local_col, color="cyan", linewidth=1.0)

    if row_plot_record is not None:
        if np.isfinite(row_plot_record.get("peak_col", np.nan)) and np.isfinite(row_plot_record.get("peak_row", np.nan)):
            ax.scatter(row_plot_record["peak_col"] - core_start_col_local + (core_start_col_local - core_start_col_local),
                       row_plot_record["peak_row"] - core_start_row_local + (core_start_row_local - core_start_row_local),
                       color="red", s=25, zorder=5)
            # correct position in patch coordinates
            ax.collections.pop()
            ax.scatter(row_plot_record["peak_col"] - (center_local_col - (row_plot_record["peak_col"] - row_plot_record["peak_col"])),
                       row_plot_record["peak_row"] - (center_local_row - (row_plot_record["peak_row"] - row_plot_record["peak_row"])),
                       color="none", s=1)
    # simpler and correct redraw
    if row_plot_record is not None and np.isfinite(row_plot_record.get("peak_col", np.nan)) and np.isfinite(row_plot_record.get("peak_row", np.nan)):
        pass

    ax.set_title(title, fontsize=11)


def make_plot(plot_path, patch, exp_row_min, exp_col_min,
              core_start_col_local, core_start_row_local, core_w, core_h,
              center_local_row, center_local_col,
              row_values, row_start_col, row_core_start_idx, row_core_end_idx,
              row_plot_record, row_uplift_info, row_bottom_markers,
              col_values, col_start_row, col_core_start_idx, col_core_end_idx,
              col_plot_record, col_uplift_info, col_bottom_markers,
              title):
    fig = plt.figure(figsize=(13.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(patch, cmap="gray", origin="upper")
    h, w = patch.shape
    ax0.add_patch(plt.Rectangle((0, 0), w - 1, h - 1, fill=False, edgecolor="limegreen", linewidth=2.0))
    if SHOW_CORE_BOX:
        ax0.add_patch(plt.Rectangle((core_start_col_local, core_start_row_local), core_w, core_h,
                                    fill=False, edgecolor="blue", linewidth=2.0))
    ax0.axhline(center_local_row, color="yellow", linewidth=1.0)
    ax0.axvline(center_local_col, color="cyan", linewidth=1.0)

    if row_plot_record is not None and np.isfinite(row_plot_record.get("peak_col", np.nan)) and np.isfinite(row_plot_record.get("peak_row", np.nan)):
        ax0.scatter(row_plot_record["peak_col"] - exp_col_min, row_plot_record["peak_row"] - exp_row_min,
                    color="red", s=25, zorder=5)

    if col_plot_record is not None and np.isfinite(col_plot_record.get("peak_col", np.nan)) and np.isfinite(col_plot_record.get("peak_row", np.nan)):
        ax0.scatter(col_plot_record["peak_col"] - exp_col_min, col_plot_record["peak_row"] - exp_row_min,
                    color="red", s=25, zorder=5)

    # Green bottom_elev points on the map panel.
    # Row-profile bottom points lie on the yellow center-row line; col-profile bottom points lie on the cyan center-col line.
    for marker in dedupe_bottom_markers(row_bottom_markers):
        if np.isfinite(marker.get("bottom_col", np.nan)) and np.isfinite(marker.get("bottom_row", np.nan)):
            ax0.scatter(marker["bottom_col"] - exp_col_min, marker["bottom_row"] - exp_row_min,
                        color="green", s=36, zorder=6)
    for marker in dedupe_bottom_markers(col_bottom_markers):
        if np.isfinite(marker.get("bottom_col", np.nan)) and np.isfinite(marker.get("bottom_row", np.nan)):
            ax0.scatter(marker["bottom_col"] - exp_col_min, marker["bottom_row"] - exp_row_min,
                        color="green", s=36, zorder=6)

    ax0.set_title(title, fontsize=11)

    draw_profile_panel(ax1, row_values, "row", row_core_start_idx, row_core_end_idx,
                       row_plot_record, row_start_col, uplift_info=row_uplift_info,
                       bottom_markers=row_bottom_markers)
    draw_profile_panel(ax2, col_values, "col", col_core_start_idx, col_core_end_idx,
                       col_plot_record, col_start_row, uplift_info=col_uplift_info,
                       bottom_markers=col_bottom_markers)

    fig.savefig(plot_path, dpi=FIG_DPI)
    plt.close(fig)


def evaluate_profile(name, profile_type, values, xs, ys, fixed_index,
                     center_row, center_col,
                     core_start_global, core_end_global,
                     expand_start_global, expand_end_global,
                     width_px, height_px, diameter_px, diameter_m, expand_pixels,
                     dem_path, center_x, center_y, center_source,
                     forced_bottom_map=None):
    center_global = center_col if profile_type == "row" else center_row
    core_start_idx = int(core_start_global - expand_start_global)
    core_end_idx = int(core_end_global - expand_start_global)
    center_idx = int(center_global - expand_start_global)

    left_forced_bottom = None
    right_forced_bottom = None
    if forced_bottom_map is not None:
        left_forced_bottom = forced_bottom_map.get("left", None)
        right_forced_bottom = forced_bottom_map.get("right", None)

    left_valid, left_reject = evaluate_side(
        values=values, xs=xs, ys=ys, profile_type=profile_type, side="left", name=name,
        core_start_idx=core_start_idx, core_end_idx=core_end_idx, center_idx=center_idx,
        slope_threshold=SLOPE_THRESHOLD_DEG, outer_consecutive=OUTER_FLAT_CONSECUTIVE,
        min_inner_offset=MIN_INNER_OFFSET, forced_bottom_idx=left_forced_bottom,
    )
    right_valid, right_reject = evaluate_side(
        values=values, xs=xs, ys=ys, profile_type=profile_type, side="right", name=name,
        core_start_idx=core_start_idx, core_end_idx=core_end_idx, center_idx=center_idx,
        slope_threshold=SLOPE_THRESHOLD_DEG, outer_consecutive=OUTER_FLAT_CONSECUTIVE,
        min_inner_offset=MIN_INNER_OFFSET, forced_bottom_idx=right_forced_bottom,
    )

    valid_records = []
    reject_records = []

    for rec in [left_valid, right_valid]:
        if rec is None:
            continue
        out = add_local_fields(rec, profile_type, fixed_index, expand_start_global)
        out["center_row"] = int(center_row)
        out["center_col"] = int(center_col)
        out["center_x"] = float(center_x)
        out["center_y"] = float(center_y)
        out["center_source"] = center_source
        out["core_start_global"] = int(core_start_global)
        out["core_end_global"] = int(core_end_global)
        out["expand_start_global"] = int(expand_start_global)
        out["expand_end_global"] = int(expand_end_global)
        out["width_px"] = int(width_px)
        out["height_px"] = int(height_px)
        out["diameter_px"] = int(diameter_px)
        out["diameter_m"] = float(diameter_m)
        out["expand_pixels"] = int(expand_pixels)
        out["dem_file"] = Path(dem_path).name
        valid_records.append(out)

    for rec in [left_reject, right_reject]:
        if rec is None:
            continue
        out = add_local_fields(rec, profile_type, fixed_index, expand_start_global)
        out["center_row"] = int(center_row)
        out["center_col"] = int(center_col)
        out["center_x"] = float(center_x)
        out["center_y"] = float(center_y)
        out["center_source"] = center_source
        out["core_start_global"] = int(core_start_global)
        out["core_end_global"] = int(core_end_global)
        out["expand_start_global"] = int(expand_start_global)
        out["expand_end_global"] = int(expand_end_global)
        out["width_px"] = int(width_px)
        out["height_px"] = int(height_px)
        out["diameter_px"] = int(diameter_px)
        out["diameter_m"] = float(diameter_m)
        out["expand_pixels"] = int(expand_pixels)
        out["dem_file"] = Path(dem_path).name
        reject_records.append(out)

    left_plot_record = (
        add_local_fields(left_valid, profile_type, fixed_index, expand_start_global) if left_valid else
        add_local_fields(left_reject, profile_type, fixed_index, expand_start_global) if left_reject else None
    )
    right_plot_record = (
        add_local_fields(right_valid, profile_type, fixed_index, expand_start_global) if right_valid else
        add_local_fields(right_reject, profile_type, fixed_index, expand_start_global) if right_reject else None
    )

    plot_record = pick_plot_record(
        valid_left=add_local_fields(left_valid, profile_type, fixed_index, expand_start_global) if left_valid else None,
        valid_right=add_local_fields(right_valid, profile_type, fixed_index, expand_start_global) if right_valid else None,
        reject_left=add_local_fields(left_reject, profile_type, fixed_index, expand_start_global) if left_reject else None,
        reject_right=add_local_fields(right_reject, profile_type, fixed_index, expand_start_global) if right_reject else None,
    )

    # Bottom markers should be based on the calculation bottom map itself, not only on
    # the representative plot_record. This guarantees that non-normal craters show
    # one left-side bottom_elev and one right-side bottom_elev using the center as boundary.
    bottom_markers = []
    for side_name, entry in [("left", left_forced_bottom), ("right", right_forced_bottom)]:
        marker = build_bottom_marker_from_entry(entry, side_name, values, profile_type, fixed_index, expand_start_global)
        if marker is not None:
            bottom_markers.append(marker)

    # If forced bottom markers are unavailable, fall back to records produced by evaluate_side.
    if not bottom_markers:
        bottom_markers = [r for r in [left_plot_record, right_plot_record]
                          if r is not None and np.isfinite(r.get("bottom_local", np.nan))]

    bottom_markers = dedupe_bottom_markers(bottom_markers)

    return valid_records, reject_records, plot_record, bottom_markers


# =========================
# Main workflow
# =========================
def process_one_crater(rec: pd.Series, fields: Dict[str, Optional[str]], dem_dir: Path,
                       plot_dir: Path) -> Tuple[List[dict], List[dict]]:
    if fields["name_col"] is None:
        raise ValueError("CSV must contain crater name field.")
    name = str(rec[fields["name_col"]]).strip()
    if not name:
        raise ValueError("Empty crater name in CSV.")

    dem_path = find_matching_dem(name, dem_dir)

    with rasterio.open(dem_path) as ds:
        dem = ds.read(1)
        dem = clean_profile_values(dem, nodata=ds.nodata)

        center_row, center_col, center_x, center_y, center_source = resolve_center_rowcol(ds, rec, fields)
        diameter_px, diameter_m, diameter_unit = resolve_diameter_px(ds, rec, fields, center_row, center_col)

        half_core = max(1, int(round(diameter_px / 2.0)))
        row_min = clip_int(center_row - half_core, 0, ds.height - 1)
        row_max = clip_int(center_row + half_core, 0, ds.height - 1)
        col_min = clip_int(center_col - half_core, 0, ds.width - 1)
        col_max = clip_int(center_col + half_core, 0, ds.width - 1)

        width_px = int(col_max - col_min + 1)
        height_px = int(row_max - row_min + 1)
        core_diameter_px = int(max(width_px, height_px))
        expand_pixels = clip_int(int(round(core_diameter_px * EXPAND_FRAC_OF_DIAMETER)),
                                 MIN_EXPAND_PIXELS, MAX_EXPAND_PIXELS)

        exp_row_min = clip_int(row_min - expand_pixels, 0, ds.height - 1)
        exp_row_max = clip_int(row_max + expand_pixels, 0, ds.height - 1)
        exp_col_min = clip_int(col_min - expand_pixels, 0, ds.width - 1)
        exp_col_max = clip_int(col_max + expand_pixels, 0, ds.width - 1)

        patch = dem[exp_row_min:exp_row_max + 1, exp_col_min:exp_col_max + 1]
        row_values = dem[center_row, exp_col_min:exp_col_max + 1]
        col_values = dem[exp_row_min:exp_row_max + 1, center_col]

        row_xs, row_ys = line_coords(ds.transform, "row", center_row, exp_col_min, exp_col_max)
        col_xs, col_ys = line_coords(ds.transform, "col", center_col, exp_row_min, exp_row_max)

        row_core_start_idx = col_min - exp_col_min
        row_core_end_idx = col_max - exp_col_min
        row_center_idx = center_col - exp_col_min
        col_core_start_idx = row_min - exp_row_min
        col_core_end_idx = row_max - exp_row_min
        col_center_idx = center_row - exp_row_min

        row_geom = analyze_central_uplift_profile_geometry(
            values=row_values, xs=row_xs, ys=row_ys,
            core_start_idx=row_core_start_idx, core_end_idx=row_core_end_idx, center_idx=row_center_idx
        )
        col_geom = analyze_central_uplift_profile_geometry(
            values=col_values, xs=col_xs, ys=col_ys,
            core_start_idx=col_core_start_idx, core_end_idx=col_core_end_idx, center_idx=col_center_idx
        )

        diameter_candidates = []
        for geom_info in [row_geom, col_geom]:
            if int(geom_info.get('geometry_ok', 0)) == 1 and np.isfinite(geom_info.get('rim_distance_m', np.nan)):
                diameter_candidates.append(float(geom_info['rim_distance_m']))
        crater_diameter_m = float(np.nanmean(diameter_candidates)) if diameter_candidates else np.nan

        bottom_candidates = []
        for geom_info in [row_geom, col_geom]:
            for k in ['left_dir_bottom_elev', 'right_dir_bottom_elev']:
                v = geom_info.get(k, np.nan)
                if np.isfinite(v):
                    bottom_candidates.append(float(v))
        crater_bottom_elev = float(np.nanmin(bottom_candidates)) if bottom_candidates else np.nan

        row_uplift = detect_central_uplift_profile(
            values=row_values, xs=row_xs, ys=row_ys,
            core_start_idx=row_core_start_idx, core_end_idx=row_core_end_idx, center_idx=row_center_idx,
            crater_bottom_elev=crater_bottom_elev, crater_diameter_m=crater_diameter_m
        )
        col_uplift = detect_central_uplift_profile(
            values=col_values, xs=col_xs, ys=col_ys,
            core_start_idx=col_core_start_idx, core_end_idx=col_core_end_idx, center_idx=col_center_idx,
            crater_bottom_elev=crater_bottom_elev, crater_diameter_m=crater_diameter_m
        )

        crater_has_central_uplift = bool(
            int(row_uplift.get('flag', 0)) == 1 or int(col_uplift.get('flag', 0)) == 1
        )

        if int(row_uplift.get('flag', 0)) == 1:
            # Central-uplift / non-normal case:
            # use the center axis as the boundary and force one bottom_elev
            # on the left side and one on the right side.
            row_forced_bottom_map = build_center_split_bottom_map(
                row_values, row_xs, row_ys,
                row_core_start_idx, row_core_end_idx, row_center_idx
            )
        else:
            # Normal / non-central case:
            # both sides share the same minimum between the two rims.
            row_forced_bottom_map = build_non_cu_bottom_map(
                row_values, row_xs, row_ys,
                row_core_start_idx, row_core_end_idx, row_center_idx
            )

        if int(col_uplift.get('flag', 0)) == 1:
            # Central-uplift / non-normal case:
            # use the center axis as the boundary and force one bottom_elev
            # above/left of center and one below/right of center.
            col_forced_bottom_map = build_center_split_bottom_map(
                col_values, col_xs, col_ys,
                col_core_start_idx, col_core_end_idx, col_center_idx
            )
        else:
            # Normal / non-central case:
            # both sides share the same minimum between the two rims.
            col_forced_bottom_map = build_non_cu_bottom_map(
                col_values, col_xs, col_ys,
                col_core_start_idx, col_core_end_idx, col_center_idx
            )

        row_valids, row_rejects, row_plot_record, row_bottom_markers = evaluate_profile(
            name=name,
            profile_type="row",
            values=row_values,
            xs=row_xs,
            ys=row_ys,
            fixed_index=center_row,
            center_row=center_row,
            center_col=center_col,
            core_start_global=col_min,
            core_end_global=col_max,
            expand_start_global=exp_col_min,
            expand_end_global=exp_col_max,
            width_px=width_px,
            height_px=height_px,
            diameter_px=core_diameter_px,
            diameter_m=diameter_m,
            expand_pixels=expand_pixels,
            dem_path=dem_path,
            center_x=center_x,
            center_y=center_y,
            center_source=center_source,
            forced_bottom_map=row_forced_bottom_map,
        )

        col_valids, col_rejects, col_plot_record, col_bottom_markers = evaluate_profile(
            name=name,
            profile_type="col",
            values=col_values,
            xs=col_xs,
            ys=col_ys,
            fixed_index=center_col,
            center_row=center_row,
            center_col=center_col,
            core_start_global=row_min,
            core_end_global=row_max,
            expand_start_global=exp_row_min,
            expand_end_global=exp_row_max,
            width_px=width_px,
            height_px=height_px,
            diameter_px=core_diameter_px,
            diameter_m=diameter_m,
            expand_pixels=expand_pixels,
            dem_path=dem_path,
            center_x=center_x,
            center_y=center_y,
            center_source=center_source,
            forced_bottom_map=col_forced_bottom_map,
        )

        attach_uplift_columns(row_valids, crater_has_central_uplift)
        attach_uplift_columns(col_valids, crater_has_central_uplift)
        attach_uplift_columns(row_rejects, crater_has_central_uplift)
        attach_uplift_columns(col_rejects, crater_has_central_uplift)

        # propagate original csv fields
        original_items = {f"csv_{k}": rec[k] for k in rec.index}
        for rr in row_valids + col_valids + row_rejects + col_rejects:
            rr.update(original_items)

        core_start_col_local = col_min - exp_col_min
        core_start_row_local = row_min - exp_row_min
        center_local_row = center_row - exp_row_min
        center_local_col = center_col - exp_col_min

        plot_path = plot_dir / f"{name}_show.png"
        make_plot(
            plot_path=plot_path,
            patch=patch,
            exp_row_min=exp_row_min,
            exp_col_min=exp_col_min,
            core_start_col_local=core_start_col_local,
            core_start_row_local=core_start_row_local,
            core_w=width_px,
            core_h=height_px,
            center_local_row=center_local_row,
            center_local_col=center_local_col,
            row_values=row_values,
            row_start_col=exp_col_min,
            row_core_start_idx=row_core_start_idx,
            row_core_end_idx=row_core_end_idx,
            row_plot_record=row_plot_record,
            row_uplift_info=row_uplift,
            row_bottom_markers=row_bottom_markers,
            col_values=col_values,
            col_start_row=exp_row_min,
            col_core_start_idx=col_core_start_idx,
            col_core_end_idx=col_core_end_idx,
            col_plot_record=col_plot_record,
            col_uplift_info=col_uplift,
            col_bottom_markers=col_bottom_markers,
            title=name,
        )

        return row_valids + col_valids, row_rejects + col_rejects


def main(csv_path: Optional[Path] = DEFAULT_CSV,
         dem_dir: Path = DEFAULT_DEM_DIR,
         output_dir: Path = DEFAULT_OUTPUT_DIR):
    if csv_path is None:
        csv_path = auto_find_csv(PLUS_DIR)
    csv_path = Path(csv_path)
    dem_dir = Path(dem_dir)
    output_dir = Path(output_dir)
    plot_dir = output_dir / PLOT_DIRNAME

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = standardize_columns(df)
    fields = detect_csv_fields(df)

    if fields["name_col"] is None:
        raise ValueError(f"Cannot detect crater name column from CSV columns: {list(df.columns)}")
    if fields["diam_col"] is None:
        raise ValueError(f"Cannot detect diameter column from CSV columns: {list(df.columns)}")
    if not ((fields["row_col"] is not None and fields["col_col"] is not None) or
            (fields["lon_col"] is not None and fields["lat_col"] is not None)):
        raise ValueError("CSV must contain either center row/col or center lon/lat.")

    valid_records = []
    reject_records = []
    fail_records = []

    if VERBOSE_RUN:
        print(f"CSV: {csv_path}")
        print(f"DEM dir: {dem_dir}")
        print(f"Output dir: {output_dir}")
        print(f"Plot dir: {plot_dir}")

    for i, rec in df.iterrows():
        try:
            vr, rr = process_one_crater(rec, fields, dem_dir, plot_dir)
            valid_records.extend(vr)
            reject_records.extend(rr)
            if VERBOSE_RUN:
                print(f"[{i+1}/{len(df)}] done: {rec[fields['name_col']]}")
        except Exception as e:
            fail_info = {f"csv_{k}": rec[k] for k in rec.index}
            fail_info["name"] = rec.get(fields["name_col"], f"row_{i}")
            fail_info["error"] = str(e)
            fail_records.append(fail_info)
            print(f"[{i+1}/{len(df)}] failed: {rec.get(fields['name_col'], f'row_{i}')} | {e}")

    valid_df = pd.DataFrame(valid_records)
    reject_df = pd.DataFrame(reject_records)
    fail_df = pd.DataFrame(fail_records)

    preferred_cols = [
        "name", "dem_file", "profile_type", "side", "reason",
        "peak_val", "outer_mean", "bottom_elev",
        "h1", "h2", "h3", "h3t", "crater_type",
        "center_source", "center_x", "center_y", "center_row", "center_col",
        "line_dem_row", "line_dem_col",
        "core_start_global", "core_end_global",
        "expand_start_global", "expand_end_global",
        "peak_idx", "peak_local", "peak_row", "peak_col",
        "outer_first_idx", "outer_last_idx",
        "outer_first_local", "outer_first_row", "outer_first_col",
        "outer_last_local", "outer_last_row", "outer_last_col",
        "bottom_idx", "bottom_local", "bottom_row", "bottom_col",
        "width_px", "height_px", "diameter_px", "diameter_m", "expand_pixels",
    ]

    def reorder_cols(df0: pd.DataFrame) -> pd.DataFrame:
        if df0.empty:
            return df0
        cols = [c for c in preferred_cols if c in df0.columns] + [c for c in df0.columns if c not in preferred_cols]
        return df0[cols]

    valid_df = reorder_cols(valid_df)
    reject_df = reorder_cols(reject_df)

    valid_csv = output_dir / "single_crater_h123_valid.csv"
    reject_csv = output_dir / "single_crater_h123_reject.csv"
    fail_csv = output_dir / "single_crater_h123_fail.csv"

    valid_df.to_csv(valid_csv, index=False, encoding="utf-8-sig")
    reject_df.to_csv(reject_csv, index=False, encoding="utf-8-sig")
    fail_df.to_csv(fail_csv, index=False, encoding="utf-8-sig")

    if VERBOSE_RUN:
        print(f"✅ valid saved: {valid_csv}")
        print(f"✅ reject saved: {reject_csv}")
        print(f"✅ fail saved: {fail_csv}")
        print(f"✅ plots saved in: {plot_dir}")
        print(f"valid count: {len(valid_df)}")
        print(f"reject count: {len(reject_df)}")
        print(f"fail count: {len(fail_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate row/col H123 metrics for single-crater DEMs using crater center and diameter from CSV."
    )
    parser.add_argument("--csv", type=str, default=None, help="CSV with crater name, diameter, center coordinates")
    parser.add_argument("--dem-dir", type=str, default=str(DEFAULT_DEM_DIR), help="Directory containing per-crater DEM tif/tiff")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    main(
        csv_path=Path(args.csv) if args.csv else None,
        dem_dir=Path(args.dem_dir),
        output_dir=Path(args.out_dir),
    )
