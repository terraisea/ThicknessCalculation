import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_DEM_PATH = Path(r"Database\CE5\CE5_dem.tif")
DEFAULT_SHP_PATH = Path(r"Database\CE5\keyextend_ce5.shp")
DEFAULT_OUTPUT_DIR = Path(r"Database\CE5\yolo\chickin\extend")
OUTPUT_TAG = "yolo"

INVALID_LOW = -3e10
INVALID_ZERO = True
MOON_RADIUS_M = 1737400.0

OUTER_DEPTH_RATIO = 1.20
OUTER_DEPTH_MIN_PX = 20
OUTER_DEPTH_MAX_PX = 160

FLOAT_BAND_RATIO = 0.10
FLOAT_BAND_MIN_PX = 1
FLOAT_BAND_MAX_PX = 6

SLOPE_THRESHOLD = 3.0
OVERALL_THRESHOLD = 8.0
WINDOW_SIZES = (3, 5)

MAX_RIM_CANDIDATES = 10
MAX_OUTER_SEGMENTS_PER_PROFILE = 6
MIN_INNER_OFFSET = 1
SHOW_DIRNAME = 'show'


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


def segment_linear_trend(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return 0.0
    x = np.arange(vals.size, dtype=float)
    try:
        return float(np.polyfit(x, vals, 1)[0])
    except Exception:
        return 0.0


def flat_window_ok(window_slopes, slope_threshold=SLOPE_THRESHOLD, overall_threshold=OVERALL_THRESHOLD):
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


def find_flat_segments(order_indices, slopes, values,
                       window_sizes=WINDOW_SIZES,
                       slope_threshold=SLOPE_THRESHOLD,
                       overall_threshold=OVERALL_THRESHOLD,
                       max_segments=MAX_OUTER_SEGMENTS_PER_PROFILE):
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
            first_idx = int(pts[0]); last_idx = int(pts[-1])
            key = (first_idx, last_idx)
            if key not in seen:
                seen.add(key)
                lo = min(first_idx, last_idx); hi = max(first_idx, last_idx)
                vals = clean_profile_values(values[lo:hi + 1])
                seg_slopes = slopes[pts]
                seg_slopes = seg_slopes[np.isfinite(seg_slopes)]
                if vals.size > 0 and seg_slopes.size > 0:
                    segments.append({
                        'kind': 'flat',
                        'first_idx': first_idx,
                        'last_idx': last_idx,
                        'indices': pts,
                        'mean': float(np.nanmean(vals)),
                        'median': float(np.nanmedian(vals)),
                        'min': float(np.nanmin(vals)),
                        'elev_std': float(np.nanstd(vals)),
                        'relief': float(np.nanmax(vals) - np.nanmin(vals)),
                        'slope_mean': float(np.nanmean(seg_slopes)),
                        'slope_max': float(np.nanmax(seg_slopes)),
                        'trend': segment_linear_trend(vals),
                        'n': int(vals.size),
                    })
            i = j + 1
    return segments[:max_segments]


def find_outer_inflection_anchor(order_indices, slopes, values, max_candidates=2):
    seq = valid_sequence(order_indices, slopes, values)
    if len(seq) < 5:
        return []
    s = np.array([slopes[i] for i in seq], dtype=float)
    z = np.array([values[i] for i in seq], dtype=float)
    # smooth slope and elevation increments
    s_smooth = np.array([np.nanmean(s[max(0, i-1):min(len(s), i+2)]) for i in range(len(s))], dtype=float)
    dz = np.diff(z)
    if dz.size == 0:
        return []
    results = []
    # find first outward segment that has appreciable gradient, then drops toward flatter behavior
    for i in range(2, len(seq) - 2):
        prev_mean = np.nanmean(s_smooth[max(0, i-2):i+1])
        next_mean = np.nanmean(s_smooth[i+1:min(len(seq), i+4)])
        if not (np.isfinite(prev_mean) and np.isfinite(next_mean)):
            continue
        if prev_mean < 1.5:
            continue
        # gradient magnitude decreases markedly or enters low-slope regime
        if next_mean <= max(SLOPE_THRESHOLD, prev_mean * 0.55):
            i0 = max(0, i-1)
            i1 = min(len(seq)-1, i+1)
            pts = seq[i0:i1+1]
            vals = np.array([values[p] for p in pts], dtype=float)
            ss = np.array([slopes[p] for p in pts], dtype=float)
            results.append({
                'kind': 'inflection',
                'first_idx': int(pts[0]),
                'last_idx': int(pts[-1]),
                'indices': list(map(int, pts)),
                'mean': float(np.nanmean(vals)),
                'median': float(np.nanmedian(vals)),
                'min': float(np.nanmin(vals)),
                'elev_std': float(np.nanstd(vals)),
                'relief': float(np.nanmax(vals) - np.nanmin(vals)),
                'slope_mean': float(np.nanmean(ss[np.isfinite(ss)])) if np.any(np.isfinite(ss)) else np.inf,
                'slope_max': float(np.nanmax(ss[np.isfinite(ss)])) if np.any(np.isfinite(ss)) else np.inf,
                'trend': segment_linear_trend(vals),
                'n': int(vals.size),
                'anchor_idx': int(seq[i]),
            })
            if len(results) >= max_candidates:
                break
    return results


def terrain_rank(seg: Dict, axis_offset: int):
    trend = abs(float(seg.get('trend', np.nan))) if np.isfinite(seg.get('trend', np.nan)) else np.inf
    kind_priority = 0 if seg.get('kind') == 'flat' else 1
    return (
        kind_priority,
        round(seg['slope_mean'], 6),
        round(seg['elev_std'], 6),
        round(seg['relief'], 6),
        round(trend, 6),
        abs(int(axis_offset)),
    )


def infer_name_field(gdf: gpd.GeoDataFrame) -> str:
    for field in ['id', 'name', 'fid']:
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
    rmin = max(0, rmin); cmin = max(0, cmin)
    rmax = min(src.height - 1, rmax); cmax = min(src.width - 1, cmax)
    return rmin, rmax, cmin, cmax


def make_boxes_from_original(rmin: int, rmax: int, cmin: int, cmax: int, src) -> Dict:
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


def center_candidates_from_bounds(lo: int, hi: int) -> List[int]:
    n = hi - lo + 1
    c1 = lo + max(0, n // 2 - 1)
    c2 = lo + min(max(0, n // 2), max(0, n - 1))
    out = []
    for c in [c1, c2]:
        if lo <= c <= hi and c not in out:
            out.append(c)
    return out


def profile_depth_score(profile: np.ndarray, lo: int, hi: int) -> float:
    vals = clean_profile_values(profile[lo:hi+1])
    if vals.size < 6 or not np.any(np.isfinite(vals)):
        return -np.inf
    n = len(vals)
    left = vals[:max(2, n//4)]
    center = vals[max(0, n//3):min(n, 2*n//3)]
    right = vals[min(n-2, 3*n//4):]
    if not (np.any(np.isfinite(left)) and np.any(np.isfinite(center)) and np.any(np.isfinite(right))):
        return -np.inf
    rim = 0.5 * (np.nanmax(left) + np.nanmax(right))
    bottom = np.nanmin(center)
    return float(rim - bottom)


def refine_center_indices(data: np.ndarray, core: Dict) -> Tuple[List[int], List[int]]:
    # local DEM re-centering before building clusters
    r0s = center_candidates_from_bounds(core['rmin'], core['rmax'])
    c0s = center_candidates_from_bounds(core['cmin'], core['cmax'])
    r0 = r0s[0]
    c0 = c0s[0]
    band_r = min(clip_int((core['rmax'] - core['rmin'] + 1) * 0.15, 2, 4), core['rmax'] - core['rmin'])
    band_c = min(clip_int((core['cmax'] - core['cmin'] + 1) * 0.15, 2, 4), core['cmax'] - core['cmin'])
    row_cands = list(range(max(core['rmin'], r0 - band_r), min(core['rmax'], r0 + band_r) + 1))
    col_cands = list(range(max(core['cmin'], c0 - band_c), min(core['cmax'], c0 + band_c) + 1))
    row_scored = []
    for r in row_cands:
        row_scored.append((-(profile_depth_score(data[r, :], core['cmin'], core['cmax'])), abs(r-r0), r))
    col_scored = []
    for c in col_cands:
        col_scored.append((-(profile_depth_score(data[:, c], core['rmin'], core['rmax'])), abs(c-c0), c))
    row_scored.sort(); col_scored.sort()
    best_rows = []
    best_cols = []
    for _,_,r in row_scored[:2]:
        if r not in best_rows:
            best_rows.append(r)
    for _,_,c in col_scored[:2]:
        if c not in best_cols:
            best_cols.append(c)
    if r0 not in best_rows:
        best_rows.append(r0)
    if c0 not in best_cols:
        best_cols.append(c0)
    return best_rows, best_cols


def profile_floor_score(values: np.ndarray, slopes: np.ndarray, core_lo: int, core_hi: int) -> Tuple:
    vals = clean_profile_values(values[core_lo:core_hi + 1])
    slp = np.asarray(slopes[core_lo:core_hi + 1], dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return (np.inf, np.inf, np.inf, np.inf)
    center_min = float(np.nanmin(vals[finite]))
    low_count = int(np.sum(np.isfinite(slp) & (slp < SLOPE_THRESHOLD)))
    relief = float(np.nanmax(vals[finite]) - np.nanmin(vals[finite]))
    return (-low_count, center_min, relief, 0)


def candidate_rows(core_rmin: int, core_rmax: int, preferred_centers: Optional[List[int]] = None) -> List[int]:
    centers = preferred_centers or center_candidates_from_bounds(core_rmin, core_rmax)
    band = clip_int((core_rmax - core_rmin + 1) * FLOAT_BAND_RATIO, FLOAT_BAND_MIN_PX, FLOAT_BAND_MAX_PX)
    rows = set()
    for center in centers:
        rows.update(range(max(core_rmin, center - band), min(core_rmax, center + band) + 1))
    rows = list(rows)
    rows.sort(key=lambda r: min(abs(r - c) for c in centers))
    return rows


def candidate_cols(core_cmin: int, core_cmax: int, preferred_centers: Optional[List[int]] = None) -> List[int]:
    centers = preferred_centers or center_candidates_from_bounds(core_cmin, core_cmax)
    band = clip_int((core_cmax - core_cmin + 1) * FLOAT_BAND_RATIO, FLOAT_BAND_MIN_PX, FLOAT_BAND_MAX_PX)
    cols = set()
    for center in centers:
        cols.update(range(max(core_cmin, center - band), min(core_cmax, center + band) + 1))
    cols = list(cols)
    cols.sort(key=lambda c: min(abs(c - cc) for cc in centers))
    return cols


def build_profile_coords_row(transform, ncols: int, row_idx: int):
    cols = np.arange(ncols)
    rows = np.full(ncols, row_idx, dtype=int)
    lon, lat = rasterio.transform.xy(transform, rows, cols)
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def build_profile_coords_col(transform, nrows: int, col_idx: int):
    rows = np.arange(nrows)
    cols = np.full(nrows, col_idx, dtype=int)
    lon, lat = rasterio.transform.xy(transform, rows, cols)
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def find_local_peak_candidates(values: np.ndarray, start_idx: int, end_idx: int, center_idx: int,
                               max_candidates: int = MAX_RIM_CANDIDATES) -> List[int]:
    lo = int(min(start_idx, end_idx))
    hi = int(max(start_idx, end_idx))
    if hi - lo + 1 < 3:
        return []
    idxs = []
    for i in range(lo + 1, hi):
        if not (np.isfinite(values[i - 1]) and np.isfinite(values[i]) and np.isfinite(values[i + 1])):
            continue
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            idxs.append(i)
    if not idxs:
        valid = [i for i in range(lo, hi + 1) if np.isfinite(values[i])]
        if not valid:
            return []
        idxs = valid
    idxs = sorted(idxs, key=lambda i: (-values[i], abs(i - center_idx)))
    out = []
    seen = set()
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(int(i))
        if len(out) >= max_candidates:
            break
    return out


def pit_class_by_side_len(side_len: int) -> str:
    if side_len <= 10:
        return 'small'
    if side_len <= 18:
        return 'medium'
    return 'large'


def find_floor_segments(order_indices, slopes, values, slope_threshold=SLOPE_THRESHOLD):
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
                'median': float(np.nanmedian(vals)), 'relief': float(np.nanmax(vals)-np.nanmin(vals)),
                'slope_mean': float(np.nanmean([slopes[p] for p in pts])), 'center_idx': int(pts[len(pts)//2]),
            })
        i = max(i + 1, j)
    return runs


def rank_floor_segments(segments: List[Dict], center_idx: int) -> List[Dict]:
    return sorted(segments, key=lambda seg: (-int(seg['n']), abs(int(seg['center_idx'])-int(center_idx)), float(seg['median']), float(seg['relief']), float(seg['slope_mean'])))


def select_bottom_by_pit_rule(inward_order, slopes, values, core_center_idx):
    valid = [idx for idx in inward_order[MIN_INNER_OFFSET:] if np.isfinite(values[idx])]
    if not valid:
        return None, None, None, 'no_floor_candidate', '未找到坑内有效候选点'
    side_len = len(inward_order)
    pit_class = pit_class_by_side_len(side_len)
    floor_runs = find_floor_segments(valid, slopes, values, slope_threshold=SLOPE_THRESHOLD)
    pos = {idx: i for i, idx in enumerate(inward_order)}
    if pit_class == 'small':
        center_half = [idx for idx in valid if pos[idx] >= len(inward_order)//2]
        cand = center_half if center_half else valid
        bottom_idx = sorted(cand, key=lambda idx: (values[idx], abs(idx-core_center_idx)))[0]
        return int(bottom_idx), float(values[bottom_idx]), 1, None, ''
    else:
        need = 2 if pit_class == 'medium' else 3
        qualified = [seg for seg in floor_runs if int(seg['n']) >= need]
        if qualified:
            qualified = rank_floor_segments(qualified, core_center_idx)
            seg = qualified[0]
            return int(seg['center_idx']), float(seg['median']), int(seg['n']), None, ''
        center_half = [idx for idx in valid if pos[idx] >= len(inward_order)//2]
        if not center_half:
            center_half = valid
        bottom_idx = sorted(center_half, key=lambda idx: (values[idx], abs(idx-core_center_idx)))[0]
        return int(bottom_idx), float(values[bottom_idx]), 1, 'fallback_center_min', '坑底退化为中心半区最低点'


def outer_orders_for_peak(peak_idx: int, logic_side: str, core_lo: int, core_hi: int, ext_lo: int, ext_hi: int):
    if logic_side == 'left':
        core_order = list(range(peak_idx - 1, core_lo - 1, -1))
        extend_order = list(range(core_lo - 1, ext_lo - 1, -1))
    else:
        core_order = list(range(peak_idx + 1, core_hi + 1))
        extend_order = list(range(core_hi + 1, ext_hi + 1))
    return core_order, extend_order


def inward_order_for_peak(peak_idx: int, logic_side: str, core_center_idx: int):
    if logic_side == 'left':
        return list(range(peak_idx, core_center_idx + 1))
    return list(range(peak_idx, core_center_idx - 1, -1))


def rank_segments(segments: List[Dict], axis_offset: int):
    return sorted(segments, key=lambda seg: terrain_rank(seg, axis_offset))


def build_metrics(name, profile_type, logic_side, geo_side, outer_domain,
                  peak_idx, peak_val, outer_seg, bottom_idx, bottom_elev,
                  axis_offset, core_center_idx, profile_rc, floor_n):
    outer_mean = float(outer_seg['mean'])
    outer_median = float(outer_seg['median'])
    outer_ref = outer_median
    h1 = float(peak_val - outer_ref)
    h2 = float(peak_val - bottom_elev)
    h3 = float(outer_ref - bottom_elev)
    h3t = float((h2 - 0.2 * h1) * 0.8)
    return {
        'name': name, 'profile_type': profile_type, 'logic_side': logic_side, 'geo_side': geo_side,
        'outer_domain': outer_domain, 'profile_rc': int(profile_rc), 'axis_offset': int(axis_offset),
        'core_center_idx': int(core_center_idx), 'peak_idx': int(peak_idx),
        'outer_first_idx': int(outer_seg['first_idx']), 'outer_last_idx': int(outer_seg['last_idx']),
        'bottom_idx': int(bottom_idx), 'peak_val': float(peak_val), 'outer_mean': outer_mean,
        'outer_median': outer_median, 'bottom_elev': float(bottom_elev),
        'outer_slope_mean': float(outer_seg['slope_mean']), 'outer_elev_std': float(outer_seg['elev_std']),
        'outer_relief': float(outer_seg['relief']), 'outer_trend': float(outer_seg.get('trend', 0.0)),
        'outer_kind': str(outer_seg.get('kind', 'flat')),
        'h1': h1, 'h2': h2, 'h3': h3, 'h3t': h3t, 'floor_n': int(floor_n),
    }


def is_valid_metrics(m: Dict) -> bool:
    return np.isfinite(m['h1']) and np.isfinite(m['h2']) and np.isfinite(m['h3']) and np.isfinite(m['h3t']) and m['h1'] > 0 and m['h2'] > m['h1'] and m['h3'] > 0 and m['h3t'] > 0


def evaluate_profile_side(values, lon, lat, profile_type, logic_side, geo_side,
                          core_lo, core_hi, core_center_idx, ext_lo, ext_hi,
                          name, axis_offset, profile_rc):
    values = clean_profile_values(values)
    slopes = calc_point_slopes(values, lon, lat)
    suffix = 'Row' if profile_type == 'row' else 'Col'
    name_out = f'{name}_DEM_{suffix}_{logic_side}'
    if core_hi - core_lo + 1 < 3:
        return None, {'name': name_out, 'reason': 'core_too_small', 'reason_zh': 'core范围过小', 'profile_type': profile_type, 'logic_side': logic_side, 'geo_side': geo_side, 'axis_offset': axis_offset, 'profile_rc': profile_rc}
    peak_candidates = find_local_peak_candidates(values, core_lo, core_hi, core_center_idx, MAX_RIM_CANDIDATES)
    if not peak_candidates:
        return None, {'name': name_out, 'reason': 'no_rim_candidate', 'reason_zh': '未找到坑缘候选', 'profile_type': profile_type, 'logic_side': logic_side, 'geo_side': geo_side, 'axis_offset': axis_offset, 'profile_rc': profile_rc}
    reject_snapshots = []
    for peak_idx in peak_candidates:
        peak_val = values[peak_idx]
        if not np.isfinite(peak_val):
            continue
        inward = inward_order_for_peak(peak_idx, logic_side, core_center_idx)
        if len(inward) < 2:
            continue
        bottom_idx, bottom_elev, floor_n, floor_reason, floor_reason_zh = select_bottom_by_pit_rule(inward, slopes, values, core_center_idx)
        if bottom_idx is None or not np.isfinite(bottom_elev):
            reject_snapshots.append({'name': name_out, 'reason': floor_reason or 'no_floor_candidate', 'reason_zh': floor_reason_zh or '未找到坑底', 'profile_type': profile_type, 'logic_side': logic_side, 'geo_side': geo_side, 'axis_offset': axis_offset, 'profile_rc': profile_rc, 'peak_idx': int(peak_idx), 'peak_val': float(peak_val)})
            continue

        core_outer_order, extend_outer_order = outer_orders_for_peak(peak_idx, logic_side, core_lo, core_hi, ext_lo, ext_hi)
        outer_candidates = []
        outer_core = find_flat_segments(core_outer_order, slopes, values, max_segments=MAX_OUTER_SEGMENTS_PER_PROFILE)
        outer_inf_core = find_outer_inflection_anchor(core_outer_order, slopes, values, max_candidates=2)
        outer_extend = find_flat_segments(extend_outer_order, slopes, values, max_segments=MAX_OUTER_SEGMENTS_PER_PROFILE)
        outer_inf_extend = find_outer_inflection_anchor(extend_outer_order, slopes, values, max_candidates=2)
        outer_candidates.extend([('core', seg) for seg in rank_segments(outer_core + outer_inf_core, axis_offset)])
        outer_candidates.extend([('extend', seg) for seg in rank_segments(outer_extend + outer_inf_extend, axis_offset)])
        if not outer_candidates:
            reject_snapshots.append({'name': name_out, 'reason': 'no_outer_flat', 'reason_zh': '未找到坑外平缓区', 'profile_type': profile_type, 'logic_side': logic_side, 'geo_side': geo_side, 'axis_offset': axis_offset, 'profile_rc': profile_rc, 'peak_idx': int(peak_idx), 'peak_val': float(peak_val), 'bottom_idx': int(bottom_idx), 'bottom_elev': float(bottom_elev), 'floor_n': int(floor_n)})
            continue
        for outer_domain, outer_seg in outer_candidates:
            m = build_metrics(name_out, profile_type, logic_side, geo_side, outer_domain, peak_idx, float(peak_val), outer_seg, bottom_idx, float(bottom_elev), axis_offset, core_center_idx, profile_rc, floor_n)
            # light physical preference: try more plausible combinations first, but do not hard reject too early
            if is_valid_metrics(m):
                m['reason'] = 'ok'; m['reason_zh'] = ''
                return m, None
            if not np.isfinite(m['h1']) or m['h1'] <= 0:
                reason = 'h1_le_0'; reason_zh = 'h1小于等于0'
            elif not np.isfinite(m['h2']) or m['h2'] <= m['h1']:
                reason = 'h2_le_h1'; reason_zh = 'h2小于等于h1'
            elif not np.isfinite(m['h3']) or m['h3'] <= 0:
                reason = 'h3_le_0'; reason_zh = 'h3小于等于0'
            elif not np.isfinite(m['h3t']) or m['h3t'] <= 0:
                reason = 'h3t_le_0'; reason_zh = 'h3t小于等于0'
            else:
                reason = 'invalid_geometry'; reason_zh = '几何关系不满足'
            m['reason'] = reason; m['reason_zh'] = reason_zh
            reject_snapshots.append(m)
    if reject_snapshots:
        priority = {'h2_le_h1': 0, 'h1_le_0': 1, 'no_outer_flat': 2, 'no_floor_candidate': 3, 'invalid_geometry': 4}
        reject_snapshots.sort(key=lambda d: (priority.get(d.get('reason', 'invalid_geometry'), 99), abs(d.get('axis_offset', 999)), abs(d.get('profile_rc', 999)-d.get('core_center_idx', d.get('profile_rc', 999))), 0 if d.get('outer_kind')=='flat' else 1))
        return None, reject_snapshots[0]
    return None, {'name': name_out, 'reason': 'no_valid_combination', 'reason_zh': '未找到有效组合', 'profile_type': profile_type, 'logic_side': logic_side, 'geo_side': geo_side, 'axis_offset': axis_offset, 'profile_rc': profile_rc}


def evaluate_row_cluster(data, transform, boxes_local, base_name):
    core = boxes_local['core']; expanded = boxes_local['expanded']
    nrows, ncols = data.shape
    refined_rows, refined_cols = refine_center_indices(data, core)
    center_row_candidates = refined_rows
    center_col_candidates = refined_cols
    rows = candidate_rows(core['rmin'], core['rmax'], preferred_centers=refined_rows)
    scored_rows = []
    for r in rows:
        if not (0 <= r < nrows):
            continue
        values = clean_profile_values(data[r, :])
        lon, lat = build_profile_coords_row(transform, ncols, r)
        slopes = calc_point_slopes(values, lon, lat)
        score = profile_floor_score(values, slopes, core['cmin'], core['cmax'])
        scored_rows.append((score, r))
    scored_rows.sort(key=lambda x: x[0])
    valid, reject = [], []
    for score, r in scored_rows:
        values = data[r, :]; lon, lat = build_profile_coords_row(transform, ncols, r)
        axis_offset = min(abs(r-c) for c in center_row_candidates)
        for center_col in center_col_candidates:
            v_left, rej_left = evaluate_profile_side(values, lon, lat, 'row', 'left', 'west', core['cmin'], center_col, center_col, expanded['cmin'], expanded['cmax'], base_name, axis_offset, r)
            if v_left is not None:
                v_left['chosen_center_col'] = center_col; v_left['floor_score'] = score; valid.append(v_left)
            elif rej_left is not None:
                rej_left['chosen_center_col'] = center_col; rej_left['floor_score'] = score; reject.append(rej_left)
            v_right, rej_right = evaluate_profile_side(values, lon, lat, 'row', 'right', 'east', center_col, core['cmax'], center_col, expanded['cmin'], expanded['cmax'], base_name, axis_offset, r)
            if v_right is not None:
                v_right['chosen_center_col'] = center_col; v_right['floor_score'] = score; valid.append(v_right)
            elif rej_right is not None:
                rej_right['chosen_center_col'] = center_col; rej_right['floor_score'] = score; reject.append(rej_right)
    return valid, reject


def evaluate_col_cluster(data, transform, boxes_local, base_name):
    core = boxes_local['core']; expanded = boxes_local['expanded']
    nrows, ncols = data.shape
    refined_rows, refined_cols = refine_center_indices(data, core)
    center_row_candidates = refined_rows
    center_col_candidates = refined_cols
    cols = candidate_cols(core['cmin'], core['cmax'], preferred_centers=refined_cols)
    scored_cols = []
    for c in cols:
        if not (0 <= c < ncols):
            continue
        values = clean_profile_values(data[:, c])
        lon, lat = build_profile_coords_col(transform, nrows, c)
        slopes = calc_point_slopes(values, lon, lat)
        score = profile_floor_score(values, slopes, core['rmin'], core['rmax'])
        scored_cols.append((score, c))
    scored_cols.sort(key=lambda x: x[0])
    valid, reject = [], []
    for score, c in scored_cols:
        values = data[:, c]; lon, lat = build_profile_coords_col(transform, nrows, c)
        axis_offset = min(abs(c-cc) for cc in center_col_candidates)
        for center_row in center_row_candidates:
            v_left, rej_left = evaluate_profile_side(values, lon, lat, 'col', 'left', 'north', core['rmin'], center_row, center_row, expanded['rmin'], expanded['rmax'], base_name, axis_offset, c)
            if v_left is not None:
                v_left['chosen_center_row'] = center_row; v_left['floor_score'] = score; valid.append(v_left)
            elif rej_left is not None:
                rej_left['chosen_center_row'] = center_row; rej_left['floor_score'] = score; reject.append(rej_left)
            v_right, rej_right = evaluate_profile_side(values, lon, lat, 'col', 'right', 'south', center_row, core['rmax'], center_row, expanded['rmin'], expanded['rmax'], base_name, axis_offset, c)
            if v_right is not None:
                v_right['chosen_center_row'] = center_row; v_right['floor_score'] = score; valid.append(v_right)
            elif rej_right is not None:
                rej_right['chosen_center_row'] = center_row; rej_right['floor_score'] = score; reject.append(rej_right)
    return valid, reject


def best_result_by_name(results: List[Dict], profile_type: str, logic_side: str):
    subset = [r for r in results if r['profile_type'] == profile_type and r['logic_side'] == logic_side]
    if not subset:
        return None
    domain_priority = {'core': 0, 'extend': 1}
    subset.sort(key=lambda r: (domain_priority.get(r.get('outer_domain', 'extend'), 99), abs(r.get('axis_offset', 0)), 0 if r.get('outer_kind') == 'flat' else 1, -r.get('h3', -np.inf), r.get('outer_slope_mean', np.inf), r.get('outer_elev_std', np.inf), r.get('outer_relief', np.inf)))
    return subset[0]


def best_reject_by_name(rejects: List[Dict], profile_type: str, logic_side: str):
    subset = [r for r in rejects if r.get('profile_type') == profile_type and r.get('logic_side') == logic_side]
    if not subset:
        return None
    priority = {'no_outer_flat': 0, 'h2_le_h1': 1, 'h1_le_0': 2, 'no_floor_candidate': 3, 'no_valid_combination': 4}
    subset.sort(key=lambda r: (priority.get(r.get('reason', 'no_valid_combination'), 99), abs(r.get('axis_offset', 999)), abs(r.get('profile_rc', 999)-r.get('core_center_idx', r.get('profile_rc', 999))), 0 if r.get('outer_kind') == 'flat' else 1))
    return subset[0]


def draw_show_figure(show_png: Path, arr: np.ndarray, boxes_local: Dict, row_rec: Optional[Dict], col_rec: Optional[Dict], base_name: str):
    core = boxes_local['core']; expanded = boxes_local['expanded']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    img = np.array(arr, dtype=float)
    finite = np.isfinite(img)
    if np.any(finite):
        vmin = float(np.nanpercentile(img[finite], 5)); vmax = float(np.nanpercentile(img[finite], 95))
        ax.imshow(img, cmap='gray', origin='upper', vmin=vmin, vmax=vmax)
    else:
        ax.imshow(np.zeros_like(img), cmap='gray', origin='upper')
    ax.plot([core['cmin'], core['cmax'], core['cmax'], core['cmin'], core['cmin']], [core['rmin'], core['rmin'], core['rmax'], core['rmax'], core['rmin']], color='b', lw=1.6)
    ax.plot([expanded['cmin'], expanded['cmax'], expanded['cmax'], expanded['cmin'], expanded['cmin']], [expanded['rmin'], expanded['rmin'], expanded['rmax'], expanded['rmax'], expanded['rmin']], color='g', lw=1.2)
    # row line yellow, col line cyan to match your recent figures
    if row_rec is not None and np.isfinite(row_rec.get('profile_rc', np.nan)):
        rr = int(row_rec['profile_rc']); ax.axhline(rr, color='y', lw=1.0)
        if np.isfinite(row_rec.get('peak_idx', np.nan)): ax.scatter([int(row_rec['peak_idx'])], [rr], c='r', s=40)
        if np.isfinite(row_rec.get('bottom_idx', np.nan)): ax.scatter([int(row_rec['bottom_idx'])], [rr], c='orange', s=40)
        if np.isfinite(row_rec.get('outer_first_idx', np.nan)) and np.isfinite(row_rec.get('outer_last_idx', np.nan)):
            ax.plot([int(row_rec['outer_first_idx']), int(row_rec['outer_last_idx'])], [rr, rr], color='lime', lw=3)
    if col_rec is not None and np.isfinite(col_rec.get('profile_rc', np.nan)):
        cc = int(col_rec['profile_rc']); ax.axvline(cc, color='c', lw=1.0)
        if np.isfinite(col_rec.get('peak_idx', np.nan)): ax.scatter([cc], [int(col_rec['peak_idx'])], c='r', s=40)
        if np.isfinite(col_rec.get('bottom_idx', np.nan)): ax.scatter([cc], [int(col_rec['bottom_idx'])], c='orange', s=40)
        if np.isfinite(col_rec.get('outer_first_idx', np.nan)) and np.isfinite(col_rec.get('outer_last_idx', np.nan)):
            ax.plot([cc, cc], [int(col_rec['outer_first_idx']), int(col_rec['outer_last_idx'])], color='lime', lw=3)
    ax.set_title(base_name)
    axr = axes[1]
    if row_rec is not None and np.isfinite(row_rec.get('profile_rc', np.nan)):
        rr = int(row_rec['profile_rc']); vals = arr[rr, :]; x = np.arange(len(vals))
        axr.plot(x, vals, 'k-', lw=1.5)
        axr.axvspan(core['cmin'], core['cmax'], color='royalblue', alpha=0.15)
        axr.axvspan(expanded['cmin'], expanded['cmax'], color='green', alpha=0.05)
        if np.isfinite(row_rec.get('peak_idx', np.nan)): axr.scatter([int(row_rec['peak_idx'])], [vals[int(row_rec['peak_idx'])]], c='r', s=60)
        if np.isfinite(row_rec.get('bottom_idx', np.nan)): axr.scatter([int(row_rec['bottom_idx'])], [vals[int(row_rec['bottom_idx'])]], c='orange', s=60)
        if np.isfinite(row_rec.get('outer_first_idx', np.nan)) and np.isfinite(row_rec.get('outer_last_idx', np.nan)):
            segx = np.arange(int(row_rec['outer_first_idx']), int(row_rec['outer_last_idx']) + 1)
            axr.plot(segx, vals[segx], color='lime', lw=3)
        axr.set_title(f"Row ({row_rec.get('logic_side','')}, {row_rec.get('reason','')})")
    else:
        axr.set_title('Row')
    axc = axes[2]
    if col_rec is not None and np.isfinite(col_rec.get('profile_rc', np.nan)):
        cc = int(col_rec['profile_rc']); vals = arr[:, cc]; y = np.arange(len(vals))
        axc.plot(y, vals, 'k-', lw=1.5)
        axc.axvspan(core['rmin'], core['rmax'], color='royalblue', alpha=0.15)
        axc.axvspan(expanded['rmin'], expanded['rmax'], color='green', alpha=0.05)
        if np.isfinite(col_rec.get('peak_idx', np.nan)): axc.scatter([int(col_rec['peak_idx'])], [vals[int(col_rec['peak_idx'])]], c='r', s=60)
        if np.isfinite(col_rec.get('bottom_idx', np.nan)): axc.scatter([int(col_rec['bottom_idx'])], [vals[int(col_rec['bottom_idx'])]], c='orange', s=60)
        if np.isfinite(col_rec.get('outer_first_idx', np.nan)) and np.isfinite(col_rec.get('outer_last_idx', np.nan)):
            segx = np.arange(int(col_rec['outer_first_idx']), int(col_rec['outer_last_idx']) + 1)
            axc.plot(segx, vals[segx], color='lime', lw=3)
        axc.set_title(f"Col ({col_rec.get('logic_side','')}, {col_rec.get('reason','')})")
    else:
        axc.set_title('Col')
    plt.tight_layout(); show_png.parent.mkdir(parents=True, exist_ok=True); plt.savefig(show_png, dpi=180); plt.close(fig)


def process_feature(src, geom, base_name: str, show_dir: Path):
    rmin, rmax, cmin, cmax = geom_bounds_to_rc(src, geom)
    boxes = make_boxes_from_original(rmin, rmax, cmin, cmax, src)
    arr, transform = read_window(src, boxes['expanded']['rmin'], boxes['expanded']['rmax'], boxes['expanded']['cmin'], boxes['expanded']['cmax'])
    row_off = boxes['expanded']['rmin']; col_off = boxes['expanded']['cmin']
    boxes_local = {
        'core': {'rmin': boxes['core']['rmin'] - row_off, 'rmax': boxes['core']['rmax'] - row_off, 'cmin': boxes['core']['cmin'] - col_off, 'cmax': boxes['core']['cmax'] - col_off},
        'expanded': {'rmin': 0, 'rmax': arr.shape[0]-1, 'cmin': 0, 'cmax': arr.shape[1]-1},
    }
    row_valid, row_reject = evaluate_row_cluster(arr, transform, boxes_local, base_name)
    col_valid, col_reject = evaluate_col_cluster(arr, transform, boxes_local, base_name)
    all_valid = row_valid + col_valid
    all_reject = row_reject + col_reject
    valid_records, reject_records, chosen_map = [], [], {}
    for profile_type, logic_side in [('row','left'), ('row','right'), ('col','left'), ('col','right')]:
        best_v = best_result_by_name(all_valid, profile_type, logic_side)
        if best_v is not None:
            valid_records.append(best_v); chosen_map[(profile_type, logic_side)] = best_v
        else:
            best_r = best_reject_by_name(all_reject, profile_type, logic_side)
            if best_r is None:
                suffix = 'Row' if profile_type == 'row' else 'Col'
                best_r = {'name': f'{base_name}_DEM_{suffix}_{logic_side}', 'reason': 'no_valid_result', 'reason_zh': '未找到有效结果', 'profile_type': profile_type, 'logic_side': logic_side}
            reject_records.append(best_r); chosen_map[(profile_type, logic_side)] = best_r
    row_rec = chosen_map.get(('row','right')) or chosen_map.get(('row','left'))
    col_rec = chosen_map.get(('col','right')) or chosen_map.get(('col','left'))
    draw_show_figure(show_dir / f'{base_name}_show.png', arr, boxes_local, row_rec, col_rec, base_name)
    out_geoms = {
        'expanded_geom': rc_box_to_geom(src, boxes['expanded']['rmin'], boxes['expanded']['rmax'], boxes['expanded']['cmin'], boxes['expanded']['cmax']),
        'core_geom': rc_box_to_geom(src, boxes['core']['rmin'], boxes['core']['rmax'], boxes['core']['cmin'], boxes['core']['cmax']),
    }
    return valid_records, reject_records, out_geoms


def run(dem_path: Path, shp_path: Path, output_dir: Path, output_tag: str = OUTPUT_TAG):
    output_dir.mkdir(parents=True, exist_ok=True)
    show_dir = output_dir / SHOW_DIRNAME
    if not dem_path.exists():
        raise FileNotFoundError(f'未找到 DEM：{dem_path}')
    if not shp_path.exists():
        raise FileNotFoundError(f'未找到 SHP：{shp_path}')
    gdf = gpd.read_file(shp_path)
    field_name = infer_name_field(gdf)
    valid_records, reject_records, expanded_geoms, core_geoms = [], [], [], []
    with rasterio.open(dem_path) as src:
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            base_name = safe_feature_name(row, field_name, idx)
            try:
                valid_i, reject_i, out_geoms = process_feature(src, geom, base_name, show_dir)
                valid_records.extend(valid_i); reject_records.extend(reject_i)
                expanded_geoms.append({'name': base_name, 'geometry': out_geoms['expanded_geom']})
                core_geoms.append({'name': base_name, 'geometry': out_geoms['core_geom']})
            except Exception as e:
                for profile_type, logic_side in [('row','left'), ('row','right'), ('col','left'), ('col','right')]:
                    suffix = 'Row' if profile_type == 'row' else 'Col'
                    reject_records.append({'name': f'{base_name}_DEM_{suffix}_{logic_side}', 'reason': f'exception: {e}', 'reason_zh': '程序异常', 'profile_type': profile_type, 'logic_side': logic_side})
    valid_cols = [
        'name','profile_type','logic_side','geo_side','outer_domain','profile_rc','axis_offset',
        'chosen_center_row','chosen_center_col','floor_score','core_center_idx',
        'peak_idx','outer_first_idx','outer_last_idx','bottom_idx','peak_val',
        'outer_mean','outer_median','bottom_elev','outer_slope_mean','outer_elev_std','outer_relief','outer_trend','outer_kind',
        'h1','h2','h3','h3t','floor_n','reason','reason_zh'
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
    out_csv = output_dir / f'all_{output_tag}_h123.csv'
    out_reject_csv = output_dir / f'all_{output_tag}_h123_reject.csv'
    valid_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    reject_df.to_csv(out_reject_csv, index=False, encoding='utf-8-sig')
    shp_crs = gdf.crs
    gpd.GeoDataFrame(expanded_geoms, crs=shp_crs).to_file(output_dir / f'{output_tag}_expanded.shp', driver='ESRI Shapefile')
    gpd.GeoDataFrame(core_geoms, crs=shp_crs).to_file(output_dir / f'{output_tag}_core.shp', driver='ESRI Shapefile')
    print(f'DEM 路径      : {dem_path}')
    print(f'SHP 路径      : {shp_path}')
    print(f'输出目录      : {output_dir}')
    print(f'有效结果 CSV  : {out_csv}')
    print(f'剔除结果 CSV  : {out_reject_csv}')
    print(f'扩大后 SHP    : {output_dir / f"{output_tag}_expanded.shp"}')
    print(f'core 调试 SHP : {output_dir / f"{output_tag}_core.shp"}')
    print(f'show 目录     : {show_dir}')
    print(f'通过记录数    : {len(valid_df)}')
    print(f'剔除记录数    : {len(reject_df)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一体化 ExtendCalculateH123：输入 shp + DEM，core 保持原 shp，outer flat 先 core 后 extend，再计算 h1/h2/h3")
    parser.add_argument("-dir", "--dir", "--shp", dest="shp", default=str(DEFAULT_SHP_PATH), help="SHP 路径")
    parser.add_argument("-dem", "--dem", dest="dem", default=str(DEFAULT_DEM_PATH), help="DEM 路径")
    parser.add_argument("-out", "--out", "--output-dir", dest="output_dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("-tag", "--tag", dest="tag", default=OUTPUT_TAG, help="输出文件名前缀标签，例如 yolo")
    args = parser.parse_args()

    run(
        dem_path=Path(args.dem),
        shp_path=Path(args.shp),
        output_dir=Path(args.output_dir),
        output_tag=args.tag,
    )
