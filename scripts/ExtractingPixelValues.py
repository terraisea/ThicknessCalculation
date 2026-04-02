import argparse
import rasterio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# python scripts\ExtractingPixelValues.py --mode all --dataset-dir Database\CE5 --shp yolo


def infer_dataset_name(dataset_dir):
    """
    例如：
    Database/CE5 -> CE5
    """
    return Path(dataset_dir).name


def infer_shp_path(dataset_dir, shp_source, dataset_name):
    """
    根据参数自动推断 shp 路径：
    manual -> manual_ce5.shp
    yolo   -> yolo_ce5.shp
    """
    dataset_dir = Path(dataset_dir)
    shp_filename = f"{shp_source}_{dataset_name.lower()}.shp"
    return dataset_dir / shp_filename


def clean_raster_values(arr, invalid_low=-3e10, nodata=None, invalid_zero=True):
    """
    对栅格值做统一清洗：
    1. 非有限值 -> nan
    2. 小于 invalid_low -> nan
    3. 若存在 nodata，则 nodata -> nan
    4. 若 invalid_zero=True，则 0 -> nan
    """
    arr = np.asarray(arr, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[arr < invalid_low] = np.nan

    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan

    if invalid_zero:
        arr[arr == 0] = np.nan

    return arr


def get_center_indices(nrows, ncols):
    """
    与旧版脚本保持一致：
    对偶数尺寸，优先取中心左/上的那一条剖面。
    """
    row_idx = max(0, nrows // 2 - 1)
    col_idx = max(0, ncols // 2 - 1)
    return row_idx, col_idx


def read_profile_from_tif(tif_path, threshold=-3e10):
    """
    从单个 tif 中提取中心行/列剖面，并返回对应经纬度。
    """
    with rasterio.open(tif_path) as src:
        img = src.read(1).astype(float)
        img = clean_raster_values(
            img,
            invalid_low=threshold,
            nodata=src.nodata,
            invalid_zero=True
        )
        transform = src.transform

    nrows, ncols = img.shape
    row_idx, col_idx = get_center_indices(nrows, ncols)

    row_values = img[row_idx, :]
    col_values = img[:, col_idx]

    rows_for_col = np.arange(nrows)
    cols_for_col = np.full(nrows, col_idx, dtype=int)
    lon_col, lat_col = rasterio.transform.xy(transform, rows_for_col, cols_for_col)

    cols_for_row = np.arange(ncols)
    rows_for_row = np.full(ncols, row_idx, dtype=int)
    lon_row, lat_row = rasterio.transform.xy(transform, rows_for_row, cols_for_row)

    return {
        "row_values": np.asarray(row_values, dtype=float),
        "col_values": np.asarray(col_values, dtype=float),
        "lon_col": np.asarray(lon_col, dtype=float),
        "lat_col": np.asarray(lat_col, dtype=float),
        "lon_row": np.asarray(lon_row, dtype=float),
        "lat_row": np.asarray(lat_row, dtype=float),
    }


def save_profile_csv(csv_path, values, lon, lat):
    np.savetxt(
        csv_path,
        np.column_stack((values, lon, lat)),
        delimiter=",",
        header="value,lon,lat",
        comments=""
    )


def plot_single_mode(output_png, mode_label, col_values, row_values, color_col, color_row):
    x1 = np.arange(len(col_values))
    x2 = np.arange(len(row_values))

    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.plot(x1, col_values, label=f"{mode_label} Col", color=color_col, linewidth=2)
    plt.title(f"{mode_label} Col Profile")
    plt.xlabel("X axis")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x2, row_values, label=f"{mode_label} Row", color=color_row, linewidth=2)
    plt.title(f"{mode_label} Row Profile")
    plt.xlabel("X axis")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_all_mode(output_png, dem_profile, cf_profile):
    x1 = np.arange(len(dem_profile["col_values"]))
    x2 = np.arange(len(dem_profile["row_values"]))
    x3 = np.arange(len(cf_profile["col_values"]))
    x4 = np.arange(len(cf_profile["row_values"]))

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x1, dem_profile["col_values"], label="DEM Col", color="blue", linewidth=2)
    plt.title("DEM Col Profile")
    plt.xlabel("X axis")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x2, dem_profile["row_values"], label="DEM Row", color="blue", linewidth=2)
    plt.title("DEM Row Profile")
    plt.xlabel("X axis")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x3, cf_profile["col_values"], label="CF Col", color="green", linewidth=2)
    plt.title("CF Col Profile")
    plt.xlabel("X axis")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x4, cf_profile["row_values"], label="CF Row", color="green", linewidth=2)
    plt.title("CF Row Profile")
    plt.xlabel("X axis")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_png, dpi=200)
    plt.close()


def process_single_mode(current_dir, output_dir, mode, threshold=-3e10):
    mode_lower = mode.lower()
    mode_upper = mode.upper()
    color_col = "blue" if mode_lower == "dem" else "green"
    color_row = "blue" if mode_lower == "dem" else "green"

    tif_files = sorted(current_dir.glob("*.tif"))
    if not tif_files:
        print(f"⚠ 目录中没有 tif 文件：{current_dir}")
        return

    for tif_file in tif_files:
        filename = tif_file.stem
        print(f"▶ 正在处理: {filename} ({mode_upper})")

        profile = read_profile_from_tif(tif_file, threshold=threshold)

        plot_single_mode(
            output_png=output_dir / f"{filename}_{mode_lower}_show.png",
            mode_label=mode_upper,
            col_values=profile["col_values"],
            row_values=profile["row_values"],
            color_col=color_col,
            color_row=color_row
        )

        save_profile_csv(
            output_dir / f"{filename}_{mode_lower}_Col.csv",
            profile["col_values"],
            profile["lon_col"],
            profile["lat_col"]
        )

        save_profile_csv(
            output_dir / f"{filename}_{mode_lower}_Row.csv",
            profile["row_values"],
            profile["lon_row"],
            profile["lat_row"]
        )

        print(f"✅ {filename} 已完成 ({mode_upper})")


def process_all_mode(dem_dir, cf_dir, output_dir, threshold=-3e10):
    dem_files = sorted(dem_dir.glob("*.tif"))
    if not dem_files:
        print(f"⚠ 目录中没有 DEM tif 文件：{dem_dir}")
        return

    for dem_file in dem_files:
        filename = dem_file.stem
        cf_file = cf_dir / f"{filename}.tif"

        if not cf_file.exists():
            print(f"⚠ CF 文件不存在: {cf_file}，跳过")
            continue

        print(f"▶ 正在处理: {filename} (ALL)")

        dem_profile = read_profile_from_tif(dem_file, threshold=threshold)
        cf_profile = read_profile_from_tif(cf_file, threshold=threshold)

        plot_all_mode(
            output_png=output_dir / f"{filename}_all_show.png",
            dem_profile=dem_profile,
            cf_profile=cf_profile
        )

        save_profile_csv(
            output_dir / f"{filename}_DEM_Col.csv",
            dem_profile["col_values"],
            dem_profile["lon_col"],
            dem_profile["lat_col"]
        )
        save_profile_csv(
            output_dir / f"{filename}_DEM_Row.csv",
            dem_profile["row_values"],
            dem_profile["lon_row"],
            dem_profile["lat_row"]
        )
        save_profile_csv(
            output_dir / f"{filename}_CF_Col.csv",
            cf_profile["col_values"],
            cf_profile["lon_col"],
            cf_profile["lat_col"]
        )
        save_profile_csv(
            output_dir / f"{filename}_CF_Row.csv",
            cf_profile["row_values"],
            cf_profile["lon_row"],
            cf_profile["lat_row"]
        )

        print(f"✅ {filename} 已完成 (ALL)")


def main(mode, dataset_dir, shp_source):
    threshold = -3e10

    dataset_dir = Path(dataset_dir)
    dataset_name = infer_dataset_name(dataset_dir)
    shp_file = infer_shp_path(dataset_dir, shp_source, dataset_name)

    input_root = dataset_dir / shp_source
    dem_dir = input_root / "dem"
    cf_dir = input_root / "cf"
    output_dir = input_root / "show"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"未找到数据集文件夹：{dataset_dir}")

    if not shp_file.exists():
        raise FileNotFoundError(f"未找到 shp 文件：{shp_file}")

    if not dem_dir.exists():
        raise FileNotFoundError(f"未找到 DEM 掩膜目录：{dem_dir}")

    if mode in ["cf", "all"] and not cf_dir.exists():
        raise FileNotFoundError(f"未找到 CF 掩膜目录：{cf_dir}")

    print(f"\n当前数据集目录: {dataset_dir}")
    print(f"数据集名称: {dataset_name}")
    print(f"当前使用的 shp: {shp_file.name}")
    print(f"输入根目录: {input_root}")
    print(f"DEM 输入目录: {dem_dir}")
    print(f"CF 输入目录 : {cf_dir}")
    print(f"结果输出目录: {output_dir}")

    if mode == "dem":
        process_single_mode(dem_dir, output_dir, mode="dem", threshold=threshold)
    elif mode == "cf":
        process_single_mode(cf_dir, output_dir, mode="cf", threshold=threshold)
    elif mode == "all":
        process_all_mode(dem_dir, cf_dir, output_dir, threshold=threshold)
    else:
        print("未知任务，请使用 --mode dem / cf / all")

    print(f"🎯 所有文件处理完成！结果已保存到: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量提取 DEM / CF 中心剖面，输出 PNG 与 CSV")

    parser.add_argument(
        "--mode",
        choices=["dem", "cf", "all"],
        default="all",
        required=True,
        help="选择处理模式：dem / cf / all"
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help=r"数据集目录，例如：Database\CE5"
    )

    parser.add_argument(
        "--shp",
        choices=["manual", "yolo"],
        required=True,
        help="选择使用哪个 shp：manual 或 yolo"
    )

    args = parser.parse_args()

    main(
        mode=args.mode,
        dataset_dir=args.dataset_dir,
        shp_source=args.shp
    )