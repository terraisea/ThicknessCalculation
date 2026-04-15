import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


DEFAULT_SHOW_DIR = Path(r"Database\CE5\yolo\show")
DEFAULT_DATASET_DIR = Path(r"Database\CE5")
DEFAULT_SOURCE = "yolo"


def resolve_dataset_and_source(show_dir: Path):
    show_dir = Path(show_dir)
    if show_dir.name.lower() != "show":
        raise ValueError(f"-dir 应直接指向 show 目录，例如 Database\\CE5\\yolo\\show；当前为：{show_dir}")
    source_dir = show_dir.parent
    dataset_dir = source_dir.parent
    source = source_dir.name
    return dataset_dir, source


# python scripts\CalculateH123.py --dataset-dir Database\CE5 --source yolo


def clean_profile_values(values, invalid_low=-3e10, invalid_zero=True):
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    values[values < invalid_low] = np.nan
    if invalid_zero:
        values[values == 0] = np.nan
    return values


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


def read_profile(csv_path):
    data = pd.read_csv(csv_path)
    values = clean_profile_values(data["value"].to_numpy(dtype=float))
    lon = data["lon"].to_numpy(dtype=float)
    lat = data["lat"].to_numpy(dtype=float)
    n = len(values)
    mid = n // 2
    return values, lon, lat, n, mid


def calc_point_slopes(values, lon, lat):
    """
    该点坡度 = 该点前后两个点构成的斜率（中心差分）
    """
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


def side_orders(peak_idx: int, n: int, side: str) -> Tuple[List[int], List[int]]:
    if side == "left":
        outward = list(range(peak_idx - 1, -1, -1))
        inward = list(range(peak_idx + 1, n))
    elif side == "right":
        outward = list(range(peak_idx + 1, n))
        inward = list(range(peak_idx - 1, -1, -1))
    else:
        raise ValueError("side must be left or right")
    return outward, inward


def get_search_range(mid: int, n: int, side: str, rim_search_frac: float) -> np.ndarray:
    if side == "left":
        half = np.arange(0, mid)
        if half.size == 0:
            return np.array([], dtype=int)
        keep = max(3, int(np.ceil(half.size * rim_search_frac)))
        return np.arange(max(0, mid - keep), mid, dtype=int)
    else:
        half = np.arange(mid + 1, n)
        if half.size == 0:
            return np.array([], dtype=int)
        keep = max(3, int(np.ceil(half.size * rim_search_frac)))
        return np.arange(mid + 1, min(n, mid + 1 + keep), dtype=int)


def find_rim_candidates(values, mid, side, rim_search_frac=0.70, max_candidates=8):
    """
    在靠近中心的一段范围内找多个 rim 候选。
    排序规则：
    1) 局部峰优先；
    2) 高程高优先；
    3) 离中心近优先。
    """
    n = len(values)
    cand = get_search_range(mid, n, side, rim_search_frac)
    cand = cand[np.isfinite(values[cand])]
    if cand.size == 0:
        return []

    local_peaks = []
    for i in cand:
        left_ok = i - 1 >= 0 and np.isfinite(values[i - 1])
        right_ok = i + 1 < n and np.isfinite(values[i + 1])
        if not (left_ok and right_ok):
            continue
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            local_peaks.append(int(i))

    use = np.array(local_peaks if len(local_peaks) > 0 else cand.tolist(), dtype=int)
    if use.size == 0:
        return []

    if side == "left":
        center_dist = mid - use
    else:
        center_dist = use - mid

    order = np.lexsort((center_dist, -values[use]))
    ordered = use[order]

    out = []
    seen = set()
    for idx in ordered:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
        if len(out) >= max_candidates:
            break
    return out


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
        if not np.isfinite(slopes[idx]):
            continue
        if values is not None and not np.isfinite(values[idx]):
            continue
        seq.append(int(idx))
    return seq


def find_flat_segments(order_indices, slopes, values, window_sizes=(3, 5), slope_threshold=3.0, overall_threshold=8.0,
                       max_segments=8):
    """
    多尺度搜索 flat 段。
    允许同一剖面存在多个候选 flat 段，后续再组合筛选。
    """
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


def collect_segment_values(values, idx_a, idx_b):
    lo = min(idx_a, idx_b)
    hi = max(idx_a, idx_b)
    out = values[lo:hi + 1]
    out = out[np.isfinite(out)]
    return out


def choose_bottom_idx(peak_idx, inward_order, flat_seg, values, side, min_inner_offset=1):
    """
    坡底定义保持“由降转平处”，但为了减少误判，不直接死用平缓段首点：
    在“平缓段首点附近 + 前一两个点”的过渡区内取最低高程点。
    这样不放宽阈值，但能减少 h3<=0 的误判。
    """
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

    # 高程最低优先；若并列，取更接近平缓段首点的
    cand.sort(key=lambda idx: (values[idx], abs(order_pos.get(idx, 10**9) - p)))
    return int(cand[0])


def build_metrics(name, side, peak_idx, peak_val, outer_seg, bottom_idx, bottom_elev):
    outer_mean = outer_seg["mean"]
    h1 = float(peak_val - outer_mean)
    h2 = float(peak_val - bottom_elev)
    h3 = float(outer_mean - bottom_elev)
    h3t = float((h2 - 0.2 * h1) * 0.8)
    return {
        "name": name,
        "side": side,
        "peak_idx": int(peak_idx),
        "outer_first_idx": int(outer_seg["first_idx"]),
        "outer_last_idx": int(outer_seg["last_idx"]),
        "bottom_idx": int(bottom_idx),
        "peak_val": float(peak_val),
        "outer_mean": float(outer_mean),
        "bottom_elev": float(bottom_elev),
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h3t": h3t,
    }


def is_valid_metrics(m):
    return (
        np.isfinite(m["h1"]) and
        np.isfinite(m["h2"]) and
        np.isfinite(m["h3"]) and
        np.isfinite(m["h3t"]) and
        (m["h1"] > 0) and
        (m["h2"] > m["h1"]) and
        (m["h3"] > 0) and
        (m["h3t"] > 0)
    )


def evaluate_side(values, lon, lat, mid, side, base_name,
                  rim_search_frac=0.70,
                  max_rim_candidates=8,
                  window_sizes=(3, 5),
                  slope_threshold=3.0,
                  overall_threshold=8.0,
                  max_outer_segments=6,
                  max_inner_segments=6,
                  min_inner_offset=1):
    slopes = calc_point_slopes(values, lon, lat)
    n = len(values)
    name = f"{base_name}_{side}"

    rim_candidates = find_rim_candidates(values, mid, side, rim_search_frac=rim_search_frac,
                                         max_candidates=max_rim_candidates)
    if not rim_candidates:
        return None, {"name": name, "side": side, "reason": "no_rim_candidate"}

    reject_snapshots = []

    for peak_idx in rim_candidates:
        if not np.isfinite(values[peak_idx]):
            continue
        peak_val = float(values[peak_idx])
        outward_order, inward_order = side_orders(peak_idx, n, side)

        outer_segs = find_flat_segments(
            outward_order, slopes, values,
            window_sizes=window_sizes,
            slope_threshold=slope_threshold,
            overall_threshold=overall_threshold,
            max_segments=max_outer_segments,
        )
        if not outer_segs:
            reject_snapshots.append({
                "name": name, "side": side, "reason": "no_outer_flat",
                "peak_idx": int(peak_idx), "peak_val": peak_val,
            })
            continue

        inner_flat_segs = find_flat_segments(
            inward_order, slopes, values,
            window_sizes=window_sizes,
            slope_threshold=slope_threshold,
            overall_threshold=overall_threshold,
            max_segments=max_inner_segments,
        )
        if not inner_flat_segs:
            reject_snapshots.append({
                "name": name, "side": side, "reason": "no_inner_flat",
                "peak_idx": int(peak_idx), "peak_val": peak_val,
                "outer_first_idx": int(outer_segs[0]["first_idx"]),
                "outer_last_idx": int(outer_segs[0]["last_idx"]),
                "outer_mean": float(outer_segs[0]["mean"]),
            })
            continue

        # 按“离 rim 近优先”的顺序尝试多个组合；一旦满足几何关系，就接受。
        for outer_seg in outer_segs:
            if not np.isfinite(outer_seg["mean"]):
                continue
            if peak_val <= outer_seg["mean"]:
                # 该 outer seg 比 rim 还高，不可能得到正 h1
                continue

            for flat_seg in inner_flat_segs:
                bottom_idx = choose_bottom_idx(
                    peak_idx=peak_idx,
                    inward_order=inward_order,
                    flat_seg=flat_seg,
                    values=values,
                    side=side,
                    min_inner_offset=min_inner_offset,
                )
                if bottom_idx is None:
                    continue
                bottom_elev = float(values[bottom_idx])
                if not np.isfinite(bottom_elev):
                    continue
                if bottom_elev >= peak_val:
                    continue

                m = build_metrics(name, side, peak_idx, peak_val, outer_seg, bottom_idx, bottom_elev)
                if is_valid_metrics(m):
                    m["reason"] = "ok"
                    return m, None
                else:
                    reason = "invalid_geometry"
                    if not np.isfinite(m["h1"]) or m["h1"] <= 0:
                        reason = "h1_le_0"
                    elif not np.isfinite(m["h2"]) or m["h2"] <= m["h1"]:
                        reason = "h2_le_h1"
                    elif not np.isfinite(m["h3"]) or m["h3"] <= 0:
                        reason = "h3_le_0"
                    elif not np.isfinite(m["h3t"]) or m["h3t"] <= 0:
                        reason = "h3t_le_0"
                    m["reason"] = reason
                    reject_snapshots.append(m)

    if reject_snapshots:
        best = reject_snapshots[0]
        # 给更具体的最后原因：优先保留最接近通过的几何型失败
        priority = {"h3_le_0": 0, "h2_le_h1": 1, "h1_le_0": 2, "h3t_le_0": 3,
                    "no_inner_flat": 4, "no_outer_flat": 5, "invalid_geometry": 6}
        reject_snapshots.sort(key=lambda d: priority.get(d.get("reason", "invalid_geometry"), 99))
        best = reject_snapshots[0]
        return None, best

    return None, {"name": name, "side": side, "reason": "no_valid_combination"}


def calc_profile_metrics(csv_path,
                         rim_search_frac=0.70,
                         max_rim_candidates=8,
                         window_sizes=(3, 5),
                         slope_threshold=3.0,
                         overall_threshold=8.0,
                         max_outer_segments=6,
                         max_inner_segments=6,
                         min_inner_offset=1):
    values, lon, lat, n, mid = read_profile(csv_path)
    base_name = csv_path.stem

    left_valid, left_reject = evaluate_side(
        values, lon, lat, mid, "left", base_name,
        rim_search_frac=rim_search_frac,
        max_rim_candidates=max_rim_candidates,
        window_sizes=window_sizes,
        slope_threshold=slope_threshold,
        overall_threshold=overall_threshold,
        max_outer_segments=max_outer_segments,
        max_inner_segments=max_inner_segments,
        min_inner_offset=min_inner_offset,
    )
    right_valid, right_reject = evaluate_side(
        values, lon, lat, mid, "right", base_name,
        rim_search_frac=rim_search_frac,
        max_rim_candidates=max_rim_candidates,
        window_sizes=window_sizes,
        slope_threshold=slope_threshold,
        overall_threshold=overall_threshold,
        max_outer_segments=max_outer_segments,
        max_inner_segments=max_inner_segments,
        min_inner_offset=min_inner_offset,
    )

    return {
        "left_valid": left_valid,
        "right_valid": right_valid,
        "left_reject": left_reject,
        "right_reject": right_reject,
    }


def parse_window_sizes(s: str):
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            v = int(p)
        except Exception:
            continue
        if v >= 3:
            out.append(v)
    out = sorted(set(out))
    return tuple(out if out else [3, 5])


def main(dataset_dir, source,
         rim_search_frac=0.70,
         max_rim_candidates=8,
         window_sizes=(3, 5),
         slope_threshold=3.0,
         overall_threshold=8.0,
         max_outer_segments=6,
         max_inner_segments=6,
         min_inner_offset=1):
    dataset_dir = Path(dataset_dir)
    base_dir = dataset_dir / source / "show"
    output_dir = dataset_dir / source / "chickin"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"未找到数据集目录：{dataset_dir}")
    if not base_dir.exists():
        raise FileNotFoundError(f"未找到 show 目录：{base_dir}")

    dem_files = sorted(list(base_dir.glob("*_DEM_Col.csv")) + list(base_dir.glob("*_DEM_Row.csv")))
    if not dem_files:
        raise FileNotFoundError(f"在 {base_dir} 中未找到 *_DEM_Col.csv 或 *_DEM_Row.csv")

    valid_records = []
    reject_records = []

    print(f"\n当前数据集目录: {dataset_dir}")
    print(f"当前数据源目录: {base_dir}")
    print(f"结果输出目录: {output_dir}")

    for csv_path in dem_files:
        try:
            res = calc_profile_metrics(
                csv_path,
                rim_search_frac=rim_search_frac,
                max_rim_candidates=max_rim_candidates,
                window_sizes=window_sizes,
                slope_threshold=slope_threshold,
                overall_threshold=overall_threshold,
                max_outer_segments=max_outer_segments,
                max_inner_segments=max_inner_segments,
                min_inner_offset=min_inner_offset,
            )
        except Exception as e:
            stem = csv_path.stem
            reject_records.append({"name": f"{stem}_left", "side": "left", "reason": f"exception: {e}"})
            reject_records.append({"name": f"{stem}_right", "side": "right", "reason": f"exception: {e}"})
            print(f"⚠ 处理失败：{csv_path.name} | {e}")
            continue

        for key_valid, key_reject in [("left_valid", "left_reject"), ("right_valid", "right_reject")]:
            if res[key_valid] is not None:
                valid_records.append(res[key_valid])
            else:
                reject_records.append(res[key_reject])

    valid_cols = [
        "name", "h1", "h2", "h3", "h3t",
        "peak_idx", "outer_first_idx", "outer_last_idx", "bottom_idx",
        "peak_val", "outer_mean", "bottom_elev", "side", "reason"
    ]
    reject_cols = [
        "name", "reason", "peak_idx", "outer_first_idx", "outer_last_idx", "bottom_idx",
        "peak_val", "outer_mean", "bottom_elev", "h1", "h2", "h3", "h3t", "side"
    ]

    valid_df = pd.DataFrame(valid_records)
    reject_df = pd.DataFrame(reject_records)

    for c in valid_cols:
        if c not in valid_df.columns:
            valid_df[c] = np.nan
    for c in reject_cols:
        if c not in reject_df.columns:
            reject_df[c] = np.nan

    valid_df = valid_df[valid_cols]
    reject_df = reject_df[reject_cols]

    out_csv = output_dir / f"all_{source}_h123.csv"
    out_reject_csv = output_dir / f"all_{source}_h123_reject.csv"

    valid_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    reject_df.to_csv(out_reject_csv, index=False, encoding="utf-8-sig")

    print(f"✅ 有效结果已保存: {out_csv}")
    print(f"✅ 剔除记录已保存: {out_reject_csv}")
    print(f"通过记录数  : {len(valid_df)}")
    print(f"剔除记录数  : {len(reject_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于原始 DEM 剖面的 h1/h2/h3/h3t 计算脚本")
    parser.add_argument("-dir", "--dir", default=str(DEFAULT_SHOW_DIR),
                        help=r"直接指定 show 目录，例如：Database\CE5\yolo\show")
    parser.add_argument("--dataset-dir", default=None,
                        help=r"数据集目录，例如：Database\CE5；通常不必手填")
    parser.add_argument("--source", "--shp", dest="source", choices=["manual", "yolo"], default=None,
                        help="选择子目录：manual / yolo；通常不必手填")
    parser.add_argument("--rim-search-frac", type=float, default=0.70,
                        help="rim 搜索范围占半边比例，默认 0.70")
    parser.add_argument("--max-rim-candidates", type=int, default=8,
                        help="每侧最多尝试多少个 rim 候选，默认 8")
    parser.add_argument("--window-sizes", type=str, default="3,5",
                        help="flat 判定窗口大小，逗号分隔，默认 3,5")
    parser.add_argument("--slope-threshold", type=float, default=3.0,
                        help="多数点低坡判据，默认 3°")
    parser.add_argument("--overall-threshold", type=float, default=8.0,
                        help="窗口整体坡度上限，默认 8°")
    parser.add_argument("--max-outer-segments", type=int, default=6,
                        help="每侧最多尝试多少个外侧平缓段，默认 6")
    parser.add_argument("--max-inner-segments", type=int, default=6,
                        help="每侧最多尝试多少个内侧平缓段，默认 6")
    parser.add_argument("--min-inner-offset", type=int, default=1,
                        help="坡底点距 rim 的最小像元数，默认 1")

    args = parser.parse_args()

    if args.dataset_dir is not None and args.source is not None:
        dataset_dir = Path(args.dataset_dir)
        source = args.source
    else:
        dataset_dir, source = resolve_dataset_and_source(Path(args.dir))

    main(
        dataset_dir=dataset_dir,
        source=source,
        rim_search_frac=args.rim_search_frac,
        max_rim_candidates=args.max_rim_candidates,
        window_sizes=parse_window_sizes(args.window_sizes),
        slope_threshold=args.slope_threshold,
        overall_threshold=args.overall_threshold,
        max_outer_segments=args.max_outer_segments,
        max_inner_segments=args.max_inner_segments,
        min_inner_offset=args.min_inner_offset,
    )
