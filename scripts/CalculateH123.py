# MODIFIED_FOR_RIM_PEAK_2026_05_12: adaptive 1/3/5 rim-height scoring; core-boundary preference cancelled; SHP points use rim peaks.

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from shapely.geometry import box


# =========================
# Easy-to-edit input/output paths
# =========================
DEM_PATH = Path(r"Database/CE5/CE5_dem.tif")
CF_PATH = Path(r"Database/CE5/CE5_CF.tif")
SHP_PATH = Path(r"Database/CE5/exce5/yolo_ce5.shp")
OUTPUT_DIR = Path(r"Database/CE5/yolo/chickin/step1")
PLOT_DIR = OUTPUT_DIR / "plots"
CORE_SHP_PATH = OUTPUT_DIR / "yolo_core_boxes.shp"
EXPAND_SHP_PATH = OUTPUT_DIR / "yolo_expand_boxes.shp"
VALID_CSV_PATH = OUTPUT_DIR / "all_yolo_h123.csv"
REJECT_CSV_PATH = OUTPUT_DIR / "all_yolo_h123_reject.csv"

# =========================
# Parameters
# =========================
SLOPE_THRESHOLD_DEG = 3.0
OUTER_FLAT_CONSECUTIVE = 3      # 3 consecutive slope points < 3°
MIN_EXPAND_PIXELS = 10
MAX_EXPAND_PIXELS = 180
EXPAND_FRAC_OF_DIAMETER = 1.0
MIN_INNER_OFFSET = 1
INNER_WINDOW_SIZES = (3, 5)     # keep CalculateH123前.py inner-flat logic
INNER_OVERALL_THRESHOLD = 8.0
INVALID_LOW = -3e10
FIG_DPI = 180

CENTRAL_UPLIFT_MIN_DIAMETER_M = 4000.0
CENTRAL_UPLIFT_LEVEL_FRAC = 0.5
CENTRAL_UPLIFT_MIN_INTERSECTIONS = 4
MARE_HCP_A = 0.075
MARE_HCP_B = 0.614

# =========================
# New crater-bottom classification parameters
# =========================
# Type 1: normal crater. Bottom elevation is the minimum elevation between the two rims
#         on the row/col profile.
# Type 2: crater with central peak. From rim inward, if a <=3 deg gentle floor exists,
#         the first >3 deg point after the gentle floor, toward the crater center, is used
#         as the calculation bottom elevation.
# Type 3: no gentle floor from wall to floor. The wall-to-floor kink is used as the
#         calculation bottom elevation.
CENTER_LEVEL_HALF_HCP_FRAC = 0.25      # test levels: center elevation +/- 0.1*hcp
CENTER_LEVEL_STEPS = 21              # number of horizontal levels tested in that interval
NON_NORMAL_MIN_INTERSECTIONS = 4      # >3 intersections means non-normal
INNER_FLAT_CONSECUTIVE_NEW = 3        # consecutive slope points <=3 deg for a gentle floor
TYPE3_SLOPE_BREAK_BEFORE = 3          # slope samples before candidate kink
TYPE3_SLOPE_BREAK_AFTER = 3           # slope samples after candidate kink
TYPE3_MIN_INNER_FRAC = 0.15           # ignore candidates too close to the rim
TYPE3_MAX_INNER_FRAC = 0.90           # ignore candidates too close to the center
TYPE3_DEPTH_FRAC_WEIGHT = 5.0         #

# =========================
# Filled-crater flag parameters
# =========================
# If >=60% of valid slopes between the two rims are <=1 degree,
# the profile is marked as filled. This only adds CSV flags and does
# not change the original H123 calculation logic.
FILLED_SLOPE_THRESHOLD_DEG = 1.0
FILLED_RATIO_THRESHOLD = 0.60

# =========================
# Basalt contact extraction parameters
# =========================
# Current CE5 basalt rule: CF > 8.2 indicates basalt.
# These fields are only added to CSV and do not change the original H123 logic.
BASALT_CF_THRESHOLD = 8.2

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
    """
    Center-difference slope:
    slope at i uses i-1 and i+1.
    Therefore 3 consecutive slope points correspond to 5 DEM pixels.
    """
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




def analyze_filled_between_rims(values, xs, ys, left_peak_idx, right_peak_idx,
                                slope_threshold=FILLED_SLOPE_THRESHOLD_DEG,
                                ratio_threshold=FILLED_RATIO_THRESHOLD):
    """
    Filled-crater flag only. This function does not modify rim/bottom/H123 logic.

    Rule:
      Between the two rims, if >=60% of valid center-difference slope points are
      <=1 degree, the row/col profile is marked as filled.
    """
    info = {
        "filled_checked": 0,
        "profile_filled_flag": 0,
        "filled_slope_ratio_le1": np.nan,
        "filled_slope_le1_count": 0,
        "filled_slope_valid_count": 0,
        "filled_slope_threshold_deg": float(slope_threshold),
        "filled_ratio_threshold": float(ratio_threshold),
        "filled_reason": "not_checked",
    }

    try:
        left_peak_idx = int(left_peak_idx)
        right_peak_idx = int(right_peak_idx)
    except Exception:
        info["filled_reason"] = "invalid_rim_index"
        return info

    if left_peak_idx < 0 or right_peak_idx < 0 or left_peak_idx >= len(values) or right_peak_idx >= len(values):
        info["filled_reason"] = "rim_index_out_of_range"
        return info

    lo = int(min(left_peak_idx, right_peak_idx))
    hi = int(max(left_peak_idx, right_peak_idx))
    if hi - lo < 2:
        info["filled_reason"] = "too_few_points_between_rims"
        return info

    slopes = calc_point_slopes(values, xs, ys, is_geographic=True)
    seg = slopes[lo:hi + 1]
    seg = seg[np.isfinite(seg)]

    info["filled_checked"] = 1
    info["filled_slope_valid_count"] = int(seg.size)

    if seg.size == 0:
        info["filled_reason"] = "no_valid_slope_between_rims"
        return info

    le_count = int(np.sum(seg <= slope_threshold))
    ratio = float(le_count / seg.size)

    info.update({
        "profile_filled_flag": int(ratio >= ratio_threshold),
        "filled_slope_ratio_le1": ratio,
        "filled_slope_le1_count": le_count,
        "filled_reason": "filled_by_slope_ratio" if ratio >= ratio_threshold else "not_filled_by_slope_ratio",
    })
    return info


def analyze_filled_from_geom(values, xs, ys, geom_info):
    if not isinstance(geom_info, dict) or int(geom_info.get('geometry_ok', 0)) != 1:
        return {
            "filled_checked": 0,
            "profile_filled_flag": 0,
            "filled_slope_ratio_le1": np.nan,
            "filled_slope_le1_count": 0,
            "filled_slope_valid_count": 0,
            "filled_slope_threshold_deg": float(FILLED_SLOPE_THRESHOLD_DEG),
            "filled_ratio_threshold": float(FILLED_RATIO_THRESHOLD),
            "filled_reason": geom_info.get('reason', 'invalid_geometry') if isinstance(geom_info, dict) else "invalid_geometry",
        }
    return analyze_filled_between_rims(
        values=values,
        xs=xs,
        ys=ys,
        left_peak_idx=geom_info.get('left_peak_idx', np.nan),
        right_peak_idx=geom_info.get('right_peak_idx', np.nan),
    )


def attach_filled_columns(records, profile_filled_info, crater_filled_flag):
    for rec in records:
        rec['profile_filled_flag'] = int(profile_filled_info.get('profile_filled_flag', 0))
        rec['crater_filled_flag'] = int(crater_filled_flag)
        rec['filled_checked'] = int(profile_filled_info.get('filled_checked', 0))
        rec['filled_slope_ratio_le1'] = profile_filled_info.get('filled_slope_ratio_le1', np.nan)
        rec['filled_slope_le1_count'] = int(profile_filled_info.get('filled_slope_le1_count', 0))
        rec['filled_slope_valid_count'] = int(profile_filled_info.get('filled_slope_valid_count', 0))
        rec['filled_slope_threshold_deg'] = profile_filled_info.get('filled_slope_threshold_deg', FILLED_SLOPE_THRESHOLD_DEG)
        rec['filled_ratio_threshold'] = profile_filled_info.get('filled_ratio_threshold', FILLED_RATIO_THRESHOLD)
        rec['filled_reason'] = profile_filled_info.get('filled_reason', '')



# =========================
# Basalt contact helpers
# =========================
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


def analyze_basalt_between_rims(cf_values, geom_info, threshold=BASALT_CF_THRESHOLD):
    """
    Profile-level basalt-only/unpenetrated flag.
    It uses the two rim indices already identified by the existing H123 geometry logic.
    """
    info = {
        'profile_between_valid_cf_count': 0,
        'profile_between_basalt_count': 0,
        'profile_between_nonbasalt_count': 0,
        'profile_both_rims_basalt': 0,
        'profile_all_between_rims_basalt': 0,
        'basalt_only_unpenetrated_flag': 0,
        'basalt_profile_reason': 'not_checked',
        'left_rim_cf': np.nan,
        'right_rim_cf': np.nan,
    }
    if not isinstance(geom_info, dict) or int(geom_info.get('geometry_ok', 0)) != 1:
        info['basalt_profile_reason'] = geom_info.get('reason', 'invalid_geometry') if isinstance(geom_info, dict) else 'invalid_geometry'
        return info

    left_idx = _safe_int(geom_info.get('left_peak_idx', np.nan))
    right_idx = _safe_int(geom_info.get('right_peak_idx', np.nan))
    if left_idx is None or right_idx is None:
        info['basalt_profile_reason'] = 'invalid_rim_index'
        return info
    if left_idx < 0 or right_idx < 0 or left_idx >= len(cf_values) or right_idx >= len(cf_values):
        info['basalt_profile_reason'] = 'rim_index_out_of_range'
        return info

    lo = int(min(left_idx, right_idx))
    hi = int(max(left_idx, right_idx))
    seg = np.asarray(cf_values[lo:hi + 1], dtype=float)
    valid = np.isfinite(seg)
    valid_cf = seg[valid]
    if valid_cf.size == 0:
        info['basalt_profile_reason'] = 'no_valid_cf_between_rims'
        return info

    basalt = valid_cf > threshold
    left_cf = float(cf_values[left_idx]) if np.isfinite(cf_values[left_idx]) else np.nan
    right_cf = float(cf_values[right_idx]) if np.isfinite(cf_values[right_idx]) else np.nan
    both_rims_basalt = bool(np.isfinite(left_cf) and np.isfinite(right_cf) and left_cf > threshold and right_cf > threshold)
    all_between_basalt = bool(valid_cf.size > 0 and np.all(basalt))

    info.update({
        'profile_between_valid_cf_count': int(valid_cf.size),
        'profile_between_basalt_count': int(np.sum(basalt)),
        'profile_between_nonbasalt_count': int(np.sum(~basalt)),
        'profile_both_rims_basalt': int(both_rims_basalt),
        'profile_all_between_rims_basalt': int(all_between_basalt),
        'basalt_only_unpenetrated_flag': int(both_rims_basalt and all_between_basalt),
        'basalt_profile_reason': 'basalt_only_unpenetrated' if (both_rims_basalt and all_between_basalt) else 'checked',
        'left_rim_cf': left_cf,
        'right_rim_cf': right_cf,
    })
    return info


def attach_basalt_contact_columns(records, cf_values, dem_values, profile_type, fixed_index_global,
                                  expand_start_global, profile_basalt_info,
                                  threshold=BASALT_CF_THRESHOLD):
    """
    Add CF-contact fields to each H123 side record.
    The existing record's peak_idx/bottom_idx/h1/peak_val are used; no rim/bottom/H123 logic is changed.
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
            rec['thickness_mode'] = 'basalt_rim_all_between_rims_basalt_no_thickness'
            rec['basalt_contact_status'] = 'excluded_basalt_only_unpenetrated'
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
            # First non-basalt pixel inward; this is the next pixel after the last basalt pixel.
            for ii in idxs[1:]:
                if np.isfinite(cf_values[ii]) and cf_values[ii] <= threshold:
                    contact_i = int(ii)
                    break
            if contact_i is None:
                rec['thickness_mode'] = 'basalt_rim_contact_not_found'
                rec['basalt_contact_status'] = 'basalt_rim_no_nonbasalt_contact_before_bottom'
                continue
            rec['thickness_mode'] = 'basalt_rim_thickness'
            rec['contact_note'] = 'first non-basalt pixel after basalt rim; next pixel after last basalt pixel'
        else:
            # First basalt pixel inward; result is burial depth, not layer thickness by itself.
            for ii in idxs[1:]:
                if np.isfinite(cf_values[ii]) and cf_values[ii] > threshold:
                    contact_i = int(ii)
                    break
            if contact_i is None:
                rec['thickness_mode'] = 'no_basalt_detected'
                rec['basalt_contact_status'] = 'no_basalt_detected_from_rim_to_bottom'
                continue
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
        rec['basalt_contact_status'] = 'ok' if thickness_or_depth >= 0 else 'negative_result_set_nan'

def crs_body_type(crs) -> str:
    if crs is None:
        return "none"
    s = str(crs).lower()
    if "moon" in s or "selen" in s or "1737400" in s or "iau" in s:
        return "moon"
    if "wgs84" in s or "4326" in s or "greenwich" in s or "6378137" in s or "298.257223563" in s:
        return "earth"
    return "other"


def harmonize_vector_raster_crs(gdf, ds):
    if ds.crs is None:
        raise ValueError("输入 DEM 没有 CRS，无法与 shp 对齐。")

    print(f"shp CRS: {gdf.crs}")
    print(f"DEM CRS: {ds.crs}")

    if gdf.crs is None:
        print("shp 没有 CRS，直接赋值为 DEM 的 CRS，不做坐标变换。")
        gdf = gdf.set_crs(ds.crs)
        return gdf, ds.crs

    if gdf.crs == ds.crs:
        print("shp 与 DEM CRS 一致，无需转换。")
        return gdf, gdf.crs

    shp_body = crs_body_type(gdf.crs)
    dem_body = crs_body_type(ds.crs)
    shp_txt = str(gdf.crs).lower()
    dem_txt = str(ds.crs).lower()
    shp_is_geographic = "degree" in shp_txt or "geogcs" in shp_txt or "geographic" in shp_txt
    dem_is_geographic = "degree" in dem_txt or "geogcs" in dem_txt or "geographic" in dem_txt

    if shp_body == "moon" and dem_body == "earth" and shp_is_geographic and dem_is_geographic:
        print("检测到 shp 是月球坐标，而 DEM 被错误标成地球 geographic CRS。")
        print("不进行坐标变换，直接沿用 shp 的月球 CRS 作为工作 CRS。")
        return gdf, gdf.crs

    if shp_body == "earth" and dem_body == "moon" and shp_is_geographic and dem_is_geographic:
        print("检测到 shp 是地球 geographic CRS，而 DEM 是月球坐标。")
        print("不进行坐标变换，直接覆盖 shp CRS 为 DEM CRS。")
        gdf = gdf.set_crs(ds.crs, allow_override=True)
        return gdf, ds.crs

    print("执行真正的 CRS 转换：gdf.to_crs(ds.crs)")
    gdf = gdf.to_crs(ds.crs)
    return gdf, ds.crs


def get_name_field(gdf):
    preferred = ["name", "Name", "NAME", "id", "ID", "fid", "FID"]
    for c in preferred:
        if c in gdf.columns:
            return c
    cols = [c for c in gdf.columns if c != gdf.geometry.name]
    return cols[0] if cols else None


def pixel_center_xy(transform, row, col):
    x, y = xy(transform, row, col, offset="center")
    return float(x), float(y)


def line_coords(transform, profile_type, fixed_index, start_idx, end_idx):
    xs = []
    ys = []
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


def adaptive_rim_height_window(core_diameter_px):
    """
    Adaptive height window for rim scoring.

    DEM resolution in the CE5 area is roughly several hundred meters per pixel.
    A fixed 5-pixel median window is too large for sub-kilometer to small-kilometer
    craters, so rim height is evaluated in three levels:
      <=12 px : single-pixel elevation, preserving small crater rims;
      13-40 px: 3-pixel local median;
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
    """Return single-pixel height or local median height around idx."""
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


def rim_inflection_score(values, slopes, idx, side, lo, hi, win=3):
    """
    拐点 + 局部峰联合评分：
    rim 应满足：
    1) 自身是局部峰；
    2) 外侧平均坡度更缓，内侧平均坡度更陡；
    3) 峰值相对周围有一定突出度。
    """
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
    在 core 半边内识别坑顶 rim。

    本版取消“靠近 core 边界优先/限制”的逻辑，不再按离 core 边界远近排序。
    坑顶仍需优先满足局部峰形（^ 形）和内外坡折关系；在候选排序中加入
    自适应局部高程：小坑用单像元，中等坑用 3 像元中位数，大坑用 5 像元中位数。

    Score = 0.35*坡折强度 + 0.45*自适应相对高程 + 0.10*距中心远近 + 0.10*突出度
    """
    if side == "left":
        lo, hi = core_start_idx, center_idx
    else:
        lo, hi = center_idx, core_end_idx

    lo = max(0, int(lo))
    hi = min(len(values) - 1, int(hi))
    if hi < lo:
        return None

    core_diameter_px = int(abs(int(core_end_idx) - int(core_start_idx)) + 1)
    height_win = adaptive_rim_height_window(core_diameter_px)

    seg_idx = [i for i in range(lo, hi + 1) if np.isfinite(values[i])]
    if not seg_idx:
        return None
    seg_heights = np.asarray([local_median_height(values, i, height_win) for i in seg_idx], dtype=float)
    seg_heights = seg_heights[np.isfinite(seg_heights)]
    segment_mean = float(np.nanmean(seg_heights)) if seg_heights.size else 0.0

    peaks = find_local_peaks(values, lo, hi)
    if not peaks:
        peaks = seg_idx
    if not peaks:
        return None

    cand = []
    for i in peaks:
        if not np.isfinite(values[i]):
            continue

        smooth_height = local_median_height(values, i, height_win)
        if not np.isfinite(smooth_height):
            continue

        slope_break, prominence = rim_inflection_score(values, slopes, i, side, lo, hi, win=3)
        if not np.isfinite(slope_break):
            continue
        # 保留“内侧坡度大于外侧坡度”的 ^ 型坑顶约束；若不满足则不作为首选候选。
        if slope_break <= 0:
            continue

        center_dist = abs(i - center_idx)
        height_rel = float(smooth_height - segment_mean)

        cand.append({
            "idx": int(i),
            "slope_break": float(slope_break),
            "height_rel": float(height_rel),
            "smooth_height": float(smooth_height),
            "prominence": float(prominence),
            "dist": float(center_dist),
            "height_window_px": int(height_win),
        })

    if not cand:
        # 如果没有满足坡折约束的局部峰，则退化为当前半边内自适应局部高程最高的点。
        peaks = sorted(peaks, key=lambda i: (-local_median_height(values, i, height_win), -abs(i - center_idx)))
        return int(peaks[0])

    def norm(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return arr
        amin = np.nanmin(arr)
        amax = np.nanmax(arr)
        if (not np.isfinite(amin)) or (not np.isfinite(amax)) or abs(amax - amin) < 1e-12:
            return np.ones_like(arr)
        return (arr - amin) / (amax - amin)

    slope_n = norm([c["slope_break"] for c in cand])
    height_n = norm([c["height_rel"] for c in cand])
    dist_n   = norm([c["dist"] for c in cand])
    prom_n   = norm([c["prominence"] for c in cand])

    for k, c in enumerate(cand):
        c["score"] = (
            0.35 * slope_n[k] +
            0.45 * height_n[k] +
            0.10 * dist_n[k] +
            0.10 * prom_n[k]
        )

    cand.sort(key=lambda d: (-d["score"], -d["smooth_height"], -d["slope_break"]))
    return int(cand[0]["idx"])


def outer_flat_from_first_run(values, slopes, outward_order, slope_threshold=3.0, consecutive=3):
    """
    Search outward from the rim.
    Once the nearest run of consecutive slopes < threshold appears, keep extending
    while slopes remain < threshold. Outer-flat elevation uses the DEM pixels covered
    by those slope points. 3 consecutive slope points correspond to 5 DEM pixels.
    """
    seq = [idx for idx in outward_order
           if 0 <= idx < len(slopes) and np.isfinite(slopes[idx]) and np.isfinite(values[idx])]
    if len(seq) < consecutive:
        return None

    run_start = None
    for i in range(len(seq) - consecutive + 1):
        w = seq[i:i + consecutive]
        if max(w) - min(w) != consecutive - 1:
            # require true adjacency in native profile indexing
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

    forced_entry = forced_bottom_idx
    forced_idx = forced_bottom_to_idx(forced_entry)

    if forced_idx is not None and np.isfinite(forced_idx):
        bottom_idx = int(forced_idx)
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
        attach_forced_bottom_metadata(m, forced_entry)
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

def add_global_fields(record, profile_type, fixed_index, start_global):
    out = dict(record)
    out["line_dem_row"] = int(fixed_index) if profile_type == "row" else np.nan
    out["line_dem_col"] = int(fixed_index) if profile_type == "col" else np.nan

    if "peak_idx" in out and np.isfinite(out.get("peak_idx", np.nan)):
        out["peak_global"] = int(start_global + int(out["peak_idx"]))
    else:
        out["peak_global"] = np.nan

    if "outer_first_idx" in out and np.isfinite(out.get("outer_first_idx", np.nan)):
        out["outer_first_global"] = int(start_global + int(out["outer_first_idx"]))
    else:
        out["outer_first_global"] = np.nan

    if "outer_last_idx" in out and np.isfinite(out.get("outer_last_idx", np.nan)):
        out["outer_last_global"] = int(start_global + int(out["outer_last_idx"]))
    else:
        out["outer_last_global"] = np.nan

    if "bottom_idx" in out and np.isfinite(out.get("bottom_idx", np.nan)):
        out["bottom_global"] = int(start_global + int(out["bottom_idx"]))
    else:
        out["bottom_global"] = np.nan

    if profile_type == "row":
        out["peak_row"] = int(fixed_index) if np.isfinite(out["peak_global"]) else np.nan
        out["peak_col"] = int(out["peak_global"]) if np.isfinite(out["peak_global"]) else np.nan
        out["outer_first_row"] = int(fixed_index) if np.isfinite(out["outer_first_global"]) else np.nan
        out["outer_first_col"] = int(out["outer_first_global"]) if np.isfinite(out["outer_first_global"]) else np.nan
        out["outer_last_row"] = int(fixed_index) if np.isfinite(out["outer_last_global"]) else np.nan
        out["outer_last_col"] = int(out["outer_last_global"]) if np.isfinite(out["outer_last_global"]) else np.nan
        out["bottom_row"] = int(fixed_index) if np.isfinite(out["bottom_global"]) else np.nan
        out["bottom_col"] = int(out["bottom_global"]) if np.isfinite(out["bottom_global"]) else np.nan
    else:
        out["peak_row"] = int(out["peak_global"]) if np.isfinite(out["peak_global"]) else np.nan
        out["peak_col"] = int(fixed_index) if np.isfinite(out["peak_global"]) else np.nan
        out["outer_first_row"] = int(out["outer_first_global"]) if np.isfinite(out["outer_first_global"]) else np.nan
        out["outer_first_col"] = int(fixed_index) if np.isfinite(out["outer_first_global"]) else np.nan
        out["outer_last_row"] = int(out["outer_last_global"]) if np.isfinite(out["outer_last_global"]) else np.nan
        out["outer_last_col"] = int(fixed_index) if np.isfinite(out["outer_last_global"]) else np.nan
        out["bottom_row"] = int(out["bottom_global"]) if np.isfinite(out["bottom_global"]) else np.nan
        out["bottom_col"] = int(fixed_index) if np.isfinite(out["bottom_global"]) else np.nan
    return out




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





def mare_hcp_km(diameter_km):
    if not np.isfinite(diameter_km) or diameter_km <= 0:
        return np.nan
    return float(MARE_HCP_A * (diameter_km ** MARE_HCP_B))


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
    """
    对非中央隆起坑：
    先按当前坑顶识别逻辑确定左右坑顶，
    再直接取两个坑顶之间的最低点，作为左右两侧共同的坑底。
    """
    geom = analyze_central_uplift_profile_geometry(values, xs, ys, core_start_idx, core_end_idx, center_idx)
    if int(geom.get('geometry_ok', 0)) != 1:
        return None

    left_peak_idx = int(geom['left_peak_idx'])
    right_peak_idx = int(geom['right_peak_idx'])
    floor_idx = find_min_idx_in_range(values, left_peak_idx, right_peak_idx)
    if floor_idx is None or not np.isfinite(values[floor_idx]):
        return None

    return {
        "left": int(floor_idx),
        "right": int(floor_idx),
    }


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




def forced_bottom_to_idx(entry):
    if isinstance(entry, dict):
        return entry.get("idx", None)
    return entry


def attach_forced_bottom_metadata(record, entry):
    if record is None or not isinstance(entry, dict):
        return record
    for k, v in entry.items():
        if k == "idx":
            continue
        record[k] = v
    return record


def find_first_flat_run_in_order(order_indices, slopes, values,
                                 slope_threshold=SLOPE_THRESHOLD_DEG,
                                 consecutive=INNER_FLAT_CONSECUTIVE_NEW):
    """
    From rim inward, find the first truly adjacent run whose slopes are <=3 deg.
    This is the gentle floor segment required by the new type-2 rule.
    """
    seq = [int(i) for i in order_indices
           if 0 <= int(i) < len(slopes)
           and np.isfinite(slopes[int(i)])
           and np.isfinite(values[int(i)])]
    if len(seq) < consecutive:
        return None

    for p in range(0, len(seq) - consecutive + 1):
        w = seq[p:p + consecutive]
        step = 1 if w[-1] > w[0] else -1
        if not all(w[j + 1] - w[j] == step for j in range(len(w) - 1)):
            continue
        if np.all(np.asarray([slopes[i] for i in w], dtype=float) <= slope_threshold):
            q = p + consecutive - 1
            while q + 1 < len(seq):
                if seq[q + 1] - seq[q] != step:
                    break
                if not (np.isfinite(slopes[seq[q + 1]]) and slopes[seq[q + 1]] <= slope_threshold):
                    break
                q += 1
            pts = seq[p:q + 1]
            return {
                "start_pos": int(p),
                "end_pos": int(q),
                "first_idx": int(pts[0]),
                "last_idx": int(pts[-1]),
                "n": int(len(pts)),
            }
    return None


def find_first_steep_after_flat(order_indices, flat_run, slopes,
                                slope_threshold=SLOPE_THRESHOLD_DEG):
    """
    Type 2 rule: after the gentle floor, move toward the crater center/central peak.
    The first point with slope >3 deg is used as the calculation bottom point.
    """
    if flat_run is None:
        return None
    seq = [int(i) for i in order_indices if 0 <= int(i) < len(slopes)]
    start = int(flat_run.get("end_pos", -1)) + 1
    for pos in range(start, len(seq)):
        idx = seq[pos]
        if np.isfinite(slopes[idx]) and slopes[idx] > slope_threshold:
            return int(idx)
    return None


def find_type3_wall_floor_kink(values, slopes, inward_order, peak_idx, center_idx,
                               before_n=TYPE3_SLOPE_BREAK_BEFORE,
                               after_n=TYPE3_SLOPE_BREAK_AFTER,
                               min_inner_frac=TYPE3_MIN_INNER_FRAC,
                               max_inner_frac=TYPE3_MAX_INNER_FRAC):
    """
    Type 3 rule: no <=3 deg gentle floor exists from wall to floor.
    Use the wall-to-floor kink. Operationally this is the strongest local decrease
    in smoothed slope from a steep wall to a lower-slope floor/interior segment.
    If no stable slope break exists, fall back to the minimum elevation between rim and center.
    """
    seq = [int(i) for i in inward_order
           if 0 <= int(i) < len(slopes)
           and np.isfinite(values[int(i)])]
    if len(seq) < before_n + after_n + 1:
        fallback = find_min_idx_in_range(values, peak_idx, center_idx)
        return fallback, "type3_fallback_min_between_rim_center"

    scores = []
    n = len(seq)
    for pos in range(before_n, n - after_n):
        idx = seq[pos]
        if not np.isfinite(values[idx]):
            continue

        frac = pos / max(n - 1, 1)
        if frac < min_inner_frac or frac > max_inner_frac:
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

        peak_val = values[peak_idx] if np.isfinite(values[peak_idx]) else np.nan
        local_min_idx = find_min_idx_in_range(values, peak_idx, center_idx)
        local_min_val = values[local_min_idx] if local_min_idx is not None and np.isfinite(values[local_min_idx]) else np.nan
        denom = peak_val - local_min_val if np.isfinite(peak_val) and np.isfinite(local_min_val) else np.nan
        if np.isfinite(denom) and abs(denom) > 1e-9:
            depth_frac = (peak_val - values[idx]) / denom
        else:
            depth_frac = 0.0
        depth_frac = float(np.clip(depth_frac, 0.0, 1.0)) if np.isfinite(depth_frac) else 0.0

        score = slope_drop +  TYPE3_DEPTH_FRAC_WEIGHT * depth_frac
        scores.append({
            "idx": int(idx),
            "score": float(score),
            "slope_drop": float(slope_drop),
            "before_slope": float(before_med),
            "after_slope": float(after_med),
            "depth_frac": float(depth_frac),
        })

    if not scores:
        fallback = find_min_idx_in_range(values, peak_idx, center_idx)
        return fallback, "type3_fallback_min_between_rim_center"

    scores.sort(key=lambda d: (-d["score"], -d["slope_drop"], -d["depth_frac"]))
    best = scores[0]
    reason = "type3_wall_floor_kink"
    if best["slope_drop"] <= 0:
        reason = "type3_weak_kink_best_available"
    return int(best["idx"]), reason


def center_intersection_test(values, core_start_idx, core_end_idx, center_idx, hcp_m,
                             half_hcp_frac=CENTER_LEVEL_HALF_HCP_FRAC,
                             steps=CENTER_LEVEL_STEPS,
                             min_intersections=NON_NORMAL_MIN_INTERSECTIONS):
    """
    New non-normal test required by the adviser:
    At the row/col center-axis intersection point, test horizontal levels within
    center_elevation +/- 0.1 hcp. If any level has >3 intersections, the crater is not normal.
    """
    info = {
        "checked": 0,
        "profile_non_normal": 0,
        "center_elev": np.nan,
        "hcp_m": np.nan,
        "center_level_min": np.nan,
        "center_level_max": np.nan,
        "center_test_level": np.nan,
        "center_max_intersections": np.nan,
        "center_test_reason": "not_checked",
    }

    if center_idx < 0 or center_idx >= len(values) or not np.isfinite(values[center_idx]):
        info["center_test_reason"] = "invalid_center_elev"
        return info

    center_elev = float(values[center_idx])
    info["center_elev"] = center_elev

    if not np.isfinite(hcp_m) or hcp_m <= 0:
        local_min_idx = find_min_idx_in_range(values, core_start_idx, core_end_idx)
        if local_min_idx is not None and np.isfinite(values[local_min_idx]):
            local_relief = abs(center_elev - float(values[local_min_idx]))
            hcp_m = local_relief if local_relief > 0 else np.nan

    if not np.isfinite(hcp_m) or hcp_m <= 0:
        info["checked"] = 1
        info["center_test_reason"] = "invalid_hcp"
        return info

    half_width = float(half_hcp_frac * hcp_m)
    level_min = center_elev - half_width
    level_max = center_elev + half_width
    levels = np.linspace(level_min, level_max, int(max(3, steps)))

    max_count = -1
    best_level = np.nan
    for level in levels:
        count = count_horizontal_intersections(values, float(level), core_start_idx, core_end_idx)
        if count > max_count:
            max_count = int(count)
            best_level = float(level)

    info.update({
        "checked": 1,
        "hcp_m": float(hcp_m),
        "center_level_min": float(level_min),
        "center_level_max": float(level_max),
        "center_test_level": float(best_level),
        "center_max_intersections": int(max_count),
        "profile_non_normal": int(max_count >= min_intersections),
        "center_test_reason": "non_normal_by_intersections" if max_count >= min_intersections else "normal_by_intersections",
    })
    return info


def classify_special_side_bottom(values, slopes, side, peak_idx, center_idx,
                                 core_start_idx, core_end_idx):
    inward_order = make_inner_order(peak_idx, side, core_start_idx, core_end_idx)
    flat_run = find_first_flat_run_in_order(
        inward_order, slopes, values,
        slope_threshold=SLOPE_THRESHOLD_DEG,
        consecutive=INNER_FLAT_CONSECUTIVE_NEW,
    )

    if flat_run is not None:
        steep_idx = find_first_steep_after_flat(
            inward_order, flat_run, slopes,
            slope_threshold=SLOPE_THRESHOLD_DEG,
        )
        if steep_idx is not None and np.isfinite(values[steep_idx]):
            return {
                "idx": int(steep_idx),
                "side_crater_type": "central_peak",
                "crater_type_code": 2,
                "bottom_rule": "type2_flat_floor_then_first_gt3_to_center",
                "inner_flat_first_idx": int(flat_run["first_idx"]),
                "inner_flat_last_idx": int(flat_run["last_idx"]),
                "inner_flat_n": int(flat_run["n"]),
                "kink_slope_deg": float(slopes[steep_idx]) if np.isfinite(slopes[steep_idx]) else np.nan,
            }
        fallback_idx = int(flat_run["last_idx"])
        return {
            "idx": fallback_idx,
            "side_crater_type": "central_peak",
            "crater_type_code": 2,
            "bottom_rule": "type2_flat_floor_no_gt3_fallback_flat_inner_edge",
            "inner_flat_first_idx": int(flat_run["first_idx"]),
            "inner_flat_last_idx": int(flat_run["last_idx"]),
            "inner_flat_n": int(flat_run["n"]),
            "kink_slope_deg": float(slopes[fallback_idx]) if np.isfinite(slopes[fallback_idx]) else np.nan,
        }

    kink_idx, reason = find_type3_wall_floor_kink(
        values=values,
        slopes=slopes,
        inward_order=inward_order,
        peak_idx=peak_idx,
        center_idx=center_idx,
    )
    if kink_idx is None or not np.isfinite(values[kink_idx]):
        kink_idx = find_min_idx_in_range(values, peak_idx, center_idx)
    return {
        "idx": int(kink_idx) if kink_idx is not None else np.nan,
        "side_crater_type": "uneven_floor_no_flat",
        "crater_type_code": 3,
        "bottom_rule": reason,
        "inner_flat_first_idx": np.nan,
        "inner_flat_last_idx": np.nan,
        "inner_flat_n": 0,
        "kink_slope_deg": float(slopes[kink_idx]) if kink_idx is not None and np.isfinite(slopes[kink_idx]) else np.nan,
    }


def build_new_bottom_map(values, xs, ys, core_start_idx, core_end_idx, center_idx,
                         crater_diameter_m=np.nan, force_non_normal=False):
    """
    Build side-specific bottom indices according to the new three-class rule.
    Returns (bottom_map, profile_info).
    bottom_map['left'/'right'] is a dict with at least {'idx': ...}.
    """
    info = {
        "profile_checked": 0,
        "profile_crater_type": "invalid_geometry",
        "profile_type_code": 0,
        "profile_non_normal": 0,
        "profile_reason": "not_checked",
        "left_peak_idx_new": np.nan,
        "right_peak_idx_new": np.nan,
        "normal_floor_idx": np.nan,
    }

    slopes = calc_point_slopes(values, xs, ys, is_geographic=True)
    geom = analyze_central_uplift_profile_geometry(values, xs, ys, core_start_idx, core_end_idx, center_idx)
    if int(geom.get("geometry_ok", 0)) != 1:
        info["profile_reason"] = geom.get("reason", "invalid_geometry")
        return None, info

    left_peak_idx = int(geom["left_peak_idx"])
    right_peak_idx = int(geom["right_peak_idx"])
    info["left_peak_idx_new"] = left_peak_idx
    info["right_peak_idx_new"] = right_peak_idx

    if np.isfinite(crater_diameter_m) and crater_diameter_m > 0:
        hcp_m = mare_hcp_km(crater_diameter_m / 1000.0) * 1000.0
    else:
        hcp_m = np.nan

    center_test = center_intersection_test(
        values=values,
        core_start_idx=left_peak_idx,
        core_end_idx=right_peak_idx,
        center_idx=center_idx,
        hcp_m=hcp_m,
    )
    info.update(center_test)
    profile_non_normal = bool(int(center_test.get("profile_non_normal", 0)) == 1 or force_non_normal)
    info["profile_non_normal"] = int(profile_non_normal)

    normal_floor_idx = find_min_idx_in_range(values, left_peak_idx, right_peak_idx)
    info["normal_floor_idx"] = int(normal_floor_idx) if normal_floor_idx is not None else np.nan

    common_meta = {
        "profile_non_normal": int(profile_non_normal),
        "center_max_intersections": info.get("center_max_intersections", np.nan),
        "center_test_level": info.get("center_test_level", np.nan),
        "center_level_min": info.get("center_level_min", np.nan),
        "center_level_max": info.get("center_level_max", np.nan),
        "center_elev": info.get("center_elev", np.nan),
        "hcp_m": info.get("hcp_m", np.nan),
        "profile_reason": info.get("center_test_reason", np.nan),
    }

    if not profile_non_normal:
        if normal_floor_idx is None or not np.isfinite(values[normal_floor_idx]):
            info.update({"profile_checked": 1, "profile_reason": "no_normal_floor"})
            return None, info
        entry = {
            "idx": int(normal_floor_idx),
            "side_crater_type": "normal",
            "crater_type_code": 1,
            "bottom_rule": "type1_min_between_two_rims",
            "inner_flat_first_idx": np.nan,
            "inner_flat_last_idx": np.nan,
            "inner_flat_n": 0,
            "kink_slope_deg": float(slopes[normal_floor_idx]) if np.isfinite(slopes[normal_floor_idx]) else np.nan,
        }
        entry.update(common_meta)
        info.update({
            "profile_checked": 1,
            "profile_crater_type": "normal",
            "profile_type_code": 1,
            "profile_reason": center_test.get("center_test_reason", "normal"),
        })
        return {"left": dict(entry), "right": dict(entry)}, info

    left_entry = classify_special_side_bottom(
        values=values, slopes=slopes, side="left", peak_idx=left_peak_idx,
        center_idx=center_idx, core_start_idx=core_start_idx, core_end_idx=core_end_idx,
    )
    right_entry = classify_special_side_bottom(
        values=values, slopes=slopes, side="right", peak_idx=right_peak_idx,
        center_idx=center_idx, core_start_idx=core_start_idx, core_end_idx=core_end_idx,
    )

    left_entry.update(common_meta)
    right_entry.update(common_meta)

    side_types = [left_entry.get("side_crater_type"), right_entry.get("side_crater_type")]
    if "central_peak" in side_types:
        ptype = "central_peak"
        pcode = 2
    else:
        ptype = "uneven_floor_no_flat"
        pcode = 3

    info.update({
        "profile_checked": 1,
        "profile_crater_type": ptype,
        "profile_type_code": pcode,
        "profile_reason": "forced_non_normal" if force_non_normal and int(center_test.get("profile_non_normal", 0)) == 0 else center_test.get("center_test_reason", "non_normal"),
    })
    return {"left": left_entry, "right": right_entry}, info


def resolve_overall_crater_type(row_info, col_info, row_bottom_map=None, col_bottom_map=None):
    all_entries = []
    for bm in [row_bottom_map, col_bottom_map]:
        if isinstance(bm, dict):
            for side in ["left", "right"]:
                if isinstance(bm.get(side), dict):
                    all_entries.append(bm[side])

    if any(e.get("side_crater_type") == "central_peak" for e in all_entries):
        return "central_peak", 2
    if any(e.get("side_crater_type") == "uneven_floor_no_flat" for e in all_entries):
        return "uneven_floor_no_flat", 3

    non_normal = False
    for info in [row_info, col_info]:
        if isinstance(info, dict) and int(info.get("profile_non_normal", 0)) == 1:
            non_normal = True
    if non_normal:
        return "uneven_floor_no_flat", 3
    return "normal", 1

def attach_uplift_columns(records, crater_type, crater_type_code=None):
    for rec in records:
        rec['crater_type'] = crater_type
        rec['crater_type_code'] = crater_type_code if crater_type_code is not None else np.nan



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


def draw_profile_panel(ax, values, profile_type, core_start_idx, core_end_idx, plot_record, start_global, uplift_info=None):
    x_local = np.arange(len(values), dtype=int)
    x_global = start_global + x_local
    ax.axvspan(x_global[0], x_global[-1], color="#dfe8df", alpha=0.85, zorder=0)
    ax.axvspan(start_global + core_start_idx, start_global + core_end_idx, color="#c5d2e3", alpha=0.95, zorder=1)
    ax.plot(x_global, values, color="black", linewidth=1.3, zorder=3)

    title = f"{profile_type.capitalize()} (no valid side)"
    if plot_record is not None:
        side = plot_record.get("side", "none")
        reason = plot_record.get("reason", "unknown")
        title = f"{profile_type.capitalize()} ({side}, {reason})"
        if uplift_info is not None and int(uplift_info.get("checked", uplift_info.get("profile_checked", 0))) == 1:
            ptype = uplift_info.get("profile_crater_type", "unknown")
            inter_val = uplift_info.get("center_max_intersections", uplift_info.get("intersections", np.nan))
            inter_txt = int(inter_val) if np.isfinite(inter_val) else 'nan'
            title += f" | Type={ptype}, Nmax={inter_txt}"

        if np.isfinite(plot_record.get("peak_global", np.nan)) and np.isfinite(plot_record.get("peak_val", np.nan)):
            ax.scatter(plot_record["peak_global"], plot_record["peak_val"], color="red", s=20, zorder=5)

        if np.isfinite(plot_record.get("bottom_global", np.nan)) and np.isfinite(plot_record.get("bottom_elev", np.nan)):
            ax.scatter(plot_record["bottom_global"], plot_record["bottom_elev"], color="orange", s=20, zorder=5)

        g1 = plot_record.get("outer_first_global", np.nan)
        g2 = plot_record.get("outer_last_global", np.nan)
        if np.isfinite(g1) and np.isfinite(g2):
            lo = int(min(g1, g2) - start_global)
            hi = int(max(g1, g2) - start_global)
            lo = max(0, lo)
            hi = min(len(values) - 1, hi)
            ax.plot(x_global[lo:hi + 1], values[lo:hi + 1], color="#15d215", linewidth=2.0, zorder=4)

    if uplift_info is not None:
        line_level = uplift_info.get("center_test_level", uplift_info.get("test_level", np.nan))
        checked_val = int(uplift_info.get("checked", uplift_info.get("profile_checked", 0)))
        if checked_val == 1 and np.isfinite(line_level):
            ax.axhline(float(line_level), color="magenta", linestyle="--", linewidth=1.0, zorder=2)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"DEM {'col' if profile_type == 'row' else 'row'}")
    ax.set_ylabel("Elevation")
    ax.grid(True, alpha=0.25)


def draw_map_panel(ax, patch, core_rect, center_local_row, center_local_col,
                   row_plot_record, col_plot_record, title):
    ax.imshow(patch, cmap="gray", origin="upper")
    h, w = patch.shape
    ax.add_patch(plt.Rectangle((0, 0), w - 1, h - 1, fill=False, edgecolor="limegreen", linewidth=2.0))
    ax.add_patch(plt.Rectangle((core_rect[0], core_rect[1]), core_rect[2], core_rect[3],
                               fill=False, edgecolor="blue", linewidth=2.0))
    ax.axhline(center_local_row, color="yellow", linewidth=1.0)
    ax.axvline(center_local_col, color="cyan", linewidth=1.0)

    if row_plot_record is not None:
        if np.isfinite(row_plot_record.get("peak_col", np.nan)) and np.isfinite(row_plot_record.get("peak_row", np.nan)):
            ax.scatter(row_plot_record["peak_col"] - core_rect[4], row_plot_record["peak_row"] - core_rect[5],
                       color="red", s=25, zorder=5)
        if np.isfinite(row_plot_record.get("bottom_col", np.nan)) and np.isfinite(row_plot_record.get("bottom_row", np.nan)):
            ax.scatter(row_plot_record["bottom_col"] - core_rect[4], row_plot_record["bottom_row"] - core_rect[5],
                       color="orange", s=25, zorder=5)
        if np.isfinite(row_plot_record.get("outer_first_col", np.nan)) and np.isfinite(row_plot_record.get("outer_last_col", np.nan)):
            x1 = row_plot_record["outer_first_col"] - core_rect[4]
            x2 = row_plot_record["outer_last_col"] - core_rect[4]
            y = row_plot_record["outer_first_row"] - core_rect[5]
            ax.plot([x1, x2], [y, y], color="#15d215", linewidth=2.5)

    if col_plot_record is not None:
        if np.isfinite(col_plot_record.get("peak_col", np.nan)) and np.isfinite(col_plot_record.get("peak_row", np.nan)):
            ax.scatter(col_plot_record["peak_col"] - core_rect[4], col_plot_record["peak_row"] - core_rect[5],
                       color="red", s=25, zorder=5)
        if np.isfinite(col_plot_record.get("bottom_col", np.nan)) and np.isfinite(col_plot_record.get("bottom_row", np.nan)):
            ax.scatter(col_plot_record["bottom_col"] - core_rect[4], col_plot_record["bottom_row"] - core_rect[5],
                       color="orange", s=25, zorder=5)
        if np.isfinite(col_plot_record.get("outer_first_row", np.nan)) and np.isfinite(col_plot_record.get("outer_last_row", np.nan)):
            x = col_plot_record["outer_first_col"] - core_rect[4]
            y1 = col_plot_record["outer_first_row"] - core_rect[5]
            y2 = col_plot_record["outer_last_row"] - core_rect[5]
            ax.plot([x, x], [y1, y2], color="#15d215", linewidth=2.5)

    ax.set_title(title, fontsize=11)


def make_plot(plot_path, patch, core_start_col_local, core_start_row_local, core_w, core_h,
              center_local_row, center_local_col,
              row_values, row_start_global, row_core_start_idx, row_core_end_idx, row_plot_record, row_uplift_info,
              col_values, col_start_global, col_core_start_idx, col_core_end_idx, col_plot_record, col_uplift_info,
              title):
    fig = plt.figure(figsize=(13.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    core_rect = (core_start_col_local, core_start_row_local, core_w, core_h,
                 col_start_global, row_start_global)
    draw_map_panel(ax0, patch, core_rect, center_local_row, center_local_col, row_plot_record, col_plot_record, title)
    draw_profile_panel(ax1, row_values, "row", row_core_start_idx, row_core_end_idx, row_plot_record, row_start_global, uplift_info=row_uplift_info)
    draw_profile_panel(ax2, col_values, "col", col_core_start_idx, col_core_end_idx, col_plot_record, col_start_global, uplift_info=col_uplift_info)
    fig.savefig(plot_path, dpi=FIG_DPI)
    plt.close(fig)



def evaluate_profile(name, profile_type, values, xs, ys, fixed_index,
                     center_row, center_col,
                     core_start_global, core_end_global,
                     expand_start_global, expand_end_global,
                     width_px, height_px, diameter_px, expand_pixels,
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
        out = add_global_fields(rec, profile_type, fixed_index, expand_start_global)
        out["center_row"] = int(center_row)
        out["center_col"] = int(center_col)
        out["core_start_global"] = int(core_start_global)
        out["core_end_global"] = int(core_end_global)
        out["expand_start_global"] = int(expand_start_global)
        out["expand_end_global"] = int(expand_end_global)
        out["width_px"] = int(width_px)
        out["height_px"] = int(height_px)
        out["diameter_px"] = int(diameter_px)
        out["expand_pixels"] = int(expand_pixels)
        valid_records.append(out)

    for rec in [left_reject, right_reject]:
        if rec is None:
            continue
        out = add_global_fields(rec, profile_type, fixed_index, expand_start_global)
        out["center_row"] = int(center_row)
        out["center_col"] = int(center_col)
        out["core_start_global"] = int(core_start_global)
        out["core_end_global"] = int(core_end_global)
        out["expand_start_global"] = int(expand_start_global)
        out["expand_end_global"] = int(expand_end_global)
        out["width_px"] = int(width_px)
        out["height_px"] = int(height_px)
        out["diameter_px"] = int(diameter_px)
        out["expand_pixels"] = int(expand_pixels)
        reject_records.append(out)

    plot_record = pick_plot_record(
        valid_left=add_global_fields(left_valid, profile_type, fixed_index, expand_start_global) if left_valid else None,
        valid_right=add_global_fields(right_valid, profile_type, fixed_index, expand_start_global) if right_valid else None,
        reject_left=add_global_fields(left_reject, profile_type, fixed_index, expand_start_global) if left_reject else None,
        reject_right=add_global_fields(right_reject, profile_type, fixed_index, expand_start_global) if right_reject else None,
    )

    return valid_records, reject_records, plot_record



def main(dem_path=DEM_PATH, shp_path=SHP_PATH, cf_path=CF_PATH):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as ds, rasterio.open(cf_path) as cf_ds:
        dem = ds.read(1)
        nodata = ds.nodata
        dem = clean_profile_values(dem, nodata=nodata)

        cf = cf_ds.read(1).astype(float)
        cf = clean_profile_values(cf, nodata=cf_ds.nodata)
        if cf.shape != dem.shape:
            raise ValueError(f"CF 与 DEM 尺寸不一致：DEM={dem.shape}, CF={cf.shape}。请先配准/重采样 CF 到 DEM 网格。")

        gdf = gpd.read_file(shp_path)
        gdf, work_crs = harmonize_vector_raster_crs(gdf, ds)

        name_field = get_name_field(gdf)

        core_boxes = []
        expand_boxes = []
        valid_records = []
        reject_records = []

        print(f"DEM: {dem_path}")
        print(f"CF : {cf_path}")
        print(f"SHP: {shp_path}")
        print(f"输出目录: {OUTPUT_DIR}")
        print(f"plot目录: {PLOT_DIR}")

        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            name = str(row[name_field]) if name_field is not None else str(idx)

            minx, miny, maxx, maxy = geom.bounds
            r0, c0 = ds.index(minx, maxy)
            r1, c1 = ds.index(maxx, miny)
            row_min = clip_int(min(r0, r1), 0, ds.height - 1)
            row_max = clip_int(max(r0, r1), 0, ds.height - 1)
            col_min = clip_int(min(c0, c1), 0, ds.width - 1)
            col_max = clip_int(max(c0, c1), 0, ds.width - 1)

            width_px = int(col_max - col_min + 1)
            height_px = int(row_max - row_min + 1)
            diameter_px = int(max(width_px, height_px))
            expand_pixels = clip_int(int(round(diameter_px * EXPAND_FRAC_OF_DIAMETER)),
                                     MIN_EXPAND_PIXELS, MAX_EXPAND_PIXELS)

            center_row = int(round((row_min + row_max) / 2.0))
            center_col = int(round((col_min + col_max) / 2.0))

            exp_row_min = clip_int(row_min - expand_pixels, 0, ds.height - 1)
            exp_row_max = clip_int(row_max + expand_pixels, 0, ds.height - 1)
            exp_col_min = clip_int(col_min - expand_pixels, 0, ds.width - 1)
            exp_col_max = clip_int(col_max + expand_pixels, 0, ds.width - 1)

            core_minx, core_maxy = pixel_center_xy(ds.transform, row_min, col_min)
            core_maxx, core_miny = pixel_center_xy(ds.transform, row_max, col_max)
            exp_minx, exp_maxy = pixel_center_xy(ds.transform, exp_row_min, exp_col_min)
            exp_maxx, exp_miny = pixel_center_xy(ds.transform, exp_row_max, exp_col_max)

            core_boxes.append({"name": name, "geometry": box(min(core_minx, core_maxx), min(core_miny, core_maxy),
                                                             max(core_minx, core_maxx), max(core_miny, core_maxy))})
            expand_boxes.append({"name": name, "geometry": box(min(exp_minx, exp_maxx), min(exp_miny, exp_maxy),
                                                               max(exp_minx, exp_maxx), max(exp_miny, exp_maxy))})

            patch = dem[exp_row_min:exp_row_max + 1, exp_col_min:exp_col_max + 1]
            row_values = dem[center_row, exp_col_min:exp_col_max + 1]
            col_values = dem[exp_row_min:exp_row_max + 1, center_col]
            row_cf_values = cf[center_row, exp_col_min:exp_col_max + 1]
            col_cf_values = cf[exp_row_min:exp_row_max + 1, center_col]

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

            # Filled-crater flag only; this does not affect the original H123 calculation.
            row_filled_info = analyze_filled_from_geom(row_values, row_xs, row_ys, row_geom)
            col_filled_info = analyze_filled_from_geom(col_values, col_xs, col_ys, col_geom)
            crater_filled_flag = int(
                int(row_filled_info.get('profile_filled_flag', 0)) == 1 or
                int(col_filled_info.get('profile_filled_flag', 0)) == 1
            )

            # Profile-level basalt-only flag between the two rims.
            row_basalt_info = analyze_basalt_between_rims(row_cf_values, row_geom)
            col_basalt_info = analyze_basalt_between_rims(col_cf_values, col_geom)

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

            # New three-class bottom-elevation rule.
            # First pass: each row/col profile is tested by the center +/-0.1 hcp
            # horizontal-intersection criterion.
            row_forced_bottom_map, row_uplift = build_new_bottom_map(
                values=row_values, xs=row_xs, ys=row_ys,
                core_start_idx=row_core_start_idx, core_end_idx=row_core_end_idx,
                center_idx=row_center_idx,
                crater_diameter_m=crater_diameter_m,
                force_non_normal=False,
            )
            col_forced_bottom_map, col_uplift = build_new_bottom_map(
                values=col_values, xs=col_xs, ys=col_ys,
                core_start_idx=col_core_start_idx, core_end_idx=col_core_end_idx,
                center_idx=col_center_idx,
                crater_diameter_m=crater_diameter_m,
                force_non_normal=False,
            )

            # Adviser rule: if either row or col center-axis profile has a level with >3 intersections,
            # the crater is not normal. Then both profiles are recalculated using type-2/type-3 rules.
            crater_non_normal = bool(
                int(row_uplift.get("profile_non_normal", 0)) == 1 or
                int(col_uplift.get("profile_non_normal", 0)) == 1
            )
            if crater_non_normal:
                if int(row_uplift.get("profile_non_normal", 0)) == 0:
                    row_forced_bottom_map, row_uplift = build_new_bottom_map(
                        values=row_values, xs=row_xs, ys=row_ys,
                        core_start_idx=row_core_start_idx, core_end_idx=row_core_end_idx,
                        center_idx=row_center_idx,
                        crater_diameter_m=crater_diameter_m,
                        force_non_normal=True,
                    )
                if int(col_uplift.get("profile_non_normal", 0)) == 0:
                    col_forced_bottom_map, col_uplift = build_new_bottom_map(
                        values=col_values, xs=col_xs, ys=col_ys,
                        core_start_idx=col_core_start_idx, core_end_idx=col_core_end_idx,
                        center_idx=col_center_idx,
                        crater_diameter_m=crater_diameter_m,
                        force_non_normal=True,
                    )

            crater_type, crater_type_code = resolve_overall_crater_type(
                row_uplift, col_uplift, row_forced_bottom_map, col_forced_bottom_map
            )

            row_valids, row_rejects, row_plot_record = evaluate_profile(
                name=f"{name}",
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
                diameter_px=diameter_px,
                expand_pixels=expand_pixels,
                forced_bottom_map=row_forced_bottom_map,
            )

            col_valids, col_rejects, col_plot_record = evaluate_profile(
                name=f"{name}",
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
                diameter_px=diameter_px,
                expand_pixels=expand_pixels,
                forced_bottom_map=col_forced_bottom_map,
            )

            attach_uplift_columns(row_valids, crater_type, crater_type_code)
            attach_uplift_columns(col_valids, crater_type, crater_type_code)
            attach_uplift_columns(row_rejects, crater_type, crater_type_code)
            attach_uplift_columns(col_rejects, crater_type, crater_type_code)

            attach_filled_columns(row_valids, row_filled_info, crater_filled_flag)
            attach_filled_columns(col_valids, col_filled_info, crater_filled_flag)
            attach_filled_columns(row_rejects, row_filled_info, crater_filled_flag)
            attach_filled_columns(col_rejects, col_filled_info, crater_filled_flag)

            attach_basalt_contact_columns(row_valids, row_cf_values, row_values, 'row', center_row, exp_col_min, row_basalt_info)
            attach_basalt_contact_columns(col_valids, col_cf_values, col_values, 'col', center_col, exp_row_min, col_basalt_info)
            attach_basalt_contact_columns(row_rejects, row_cf_values, row_values, 'row', center_row, exp_col_min, row_basalt_info)
            attach_basalt_contact_columns(col_rejects, col_cf_values, col_values, 'col', center_col, exp_row_min, col_basalt_info)

            valid_records.extend(row_valids)
            valid_records.extend(col_valids)
            reject_records.extend(row_rejects)
            reject_records.extend(col_rejects)

            core_start_col_local = col_min - exp_col_min
            core_start_row_local = row_min - exp_row_min
            center_local_row = center_row - exp_row_min
            center_local_col = center_col - exp_col_min

            plot_path = PLOT_DIR / f"{name}_show.png"
            make_plot(
                plot_path=plot_path,
                patch=patch,
                core_start_col_local=core_start_col_local,
                core_start_row_local=core_start_row_local,
                core_w=width_px,
                core_h=height_px,
                center_local_row=center_local_row,
                center_local_col=center_local_col,
                row_values=row_values,
                row_start_global=exp_col_min,
                row_core_start_idx=row_core_start_idx,
                row_core_end_idx=row_core_end_idx,
                row_plot_record=row_plot_record,
                row_uplift_info=row_uplift,
                col_values=col_values,
                col_start_global=exp_row_min,
                col_core_start_idx=col_core_start_idx,
                col_core_end_idx=col_core_end_idx,
                col_plot_record=col_plot_record,
                col_uplift_info=col_uplift,
                title=name,
            )

    valid_cols = [
        "name", "profile_type", "side", "reason",
        "peak_val", "outer_mean", "bottom_elev",
        "h1", "h2", "h3", "h3t", "crater_type", "crater_type_code",
        "crater_filled_flag", "profile_filled_flag", "filled_checked",
        "filled_slope_ratio_le1", "filled_slope_le1_count", "filled_slope_valid_count",
        "filled_slope_threshold_deg", "filled_ratio_threshold", "filled_reason",
        "basalt_cf_threshold", "peak_cf", "rim_type_new",
        "profile_between_valid_cf_count", "profile_between_basalt_count", "profile_between_nonbasalt_count",
        "profile_both_rims_basalt", "profile_all_between_rims_basalt",
        "basalt_only_unpenetrated_flag", "basalt_profile_reason", "left_rim_cf", "right_rim_cf",
        "contact_idx", "contact_global", "contact_row", "contact_col", "contact_cf",
        "h_basalt_elev", "h2_basalt", "thickness_or_depth",
        "thickness_mode", "basalt_contact_status", "contact_note",
        "side_crater_type", "bottom_rule", "kink_slope_deg",
        "inner_flat_first_idx", "inner_flat_last_idx", "inner_flat_n",
        "profile_non_normal", "center_max_intersections", "center_test_level",
        "center_level_min", "center_level_max", "center_elev", "hcp_m", "profile_reason",
        "line_dem_row", "line_dem_col",
        "center_row", "center_col",
        "core_start_global", "core_end_global",
        "expand_start_global", "expand_end_global",
        "peak_idx", "peak_global", "peak_row", "peak_col",
        "outer_first_idx", "outer_last_idx",
        "outer_first_global", "outer_first_row", "outer_first_col",
        "outer_last_global", "outer_last_row", "outer_last_col",
        "bottom_idx", "bottom_global", "bottom_row", "bottom_col",
        "width_px", "height_px", "diameter_px", "expand_pixels"
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

    valid_df.to_csv(VALID_CSV_PATH, index=False, encoding="utf-8-sig")
    reject_df.to_csv(REJECT_CSV_PATH, index=False, encoding="utf-8-sig")

    core_gdf = gpd.GeoDataFrame(core_boxes, geometry="geometry", crs=work_crs)
    expand_gdf = gpd.GeoDataFrame(expand_boxes, geometry="geometry", crs=work_crs)
    core_gdf.to_file(CORE_SHP_PATH, encoding="utf-8")
    expand_gdf.to_file(EXPAND_SHP_PATH, encoding="utf-8")

    print(f"✅ 有效结果已保存: {VALID_CSV_PATH}")
    print(f"✅ 剔除记录已保存: {REJECT_CSV_PATH}")
    print(f"✅ 原始YOLO框 shp 已保存: {CORE_SHP_PATH}")
    print(f"✅ 外扩框 shp 已保存: {EXPAND_SHP_PATH}")
    print(f"✅ 剖面图目录: {PLOT_DIR}")
    print(f"通过记录数: {len(valid_df)}")
    print(f"剔除记录数: {len(reject_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="直接读取 DEM 与 yolo 坑框计算 h1/h2/h3/h3t")
    parser.add_argument("--dem", type=str, default=str(DEM_PATH), help="DEM 路径")
    parser.add_argument("--cf", type=str, default=str(CF_PATH), help="CF 路径")
    parser.add_argument("--shp", type=str, default=str(SHP_PATH), help="shp 路径")
    args = parser.parse_args()
    main(dem_path=Path(args.dem), shp_path=Path(args.shp), cf_path=Path(args.cf))
