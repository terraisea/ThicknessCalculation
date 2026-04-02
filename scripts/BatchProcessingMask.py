import argparse
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from pathlib import Path

#python scripts\BatchProcessingMask.py --mode all --dataset-dir Database\CE5 --shp yolo
def make_unique_filepath(output_folder, stem, suffix=".tif"):
    """
    如果文件已存在，则自动追加 _2, _3, ...
    例如：
    xxx.tif
    xxx_2.tif
    xxx_3.tif
    """
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


def mask_raster(shp_file, tif_path, output_folder, label, field_name):
    """对单个 shapefile 执行掩膜"""
    gdf = gpd.read_file(shp_file)

    # 如果一个 shp 有多个面，则逐个处理
    for idx, row in gdf.iterrows():
        geom = [mapping(row.geometry)]

        with rasterio.open(tif_path) as src:
            out_image, out_transform = mask(src, geom, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # ★ 使用指定字段命名
        try:
            value = str(row[field_name]).strip()
            if value == "" or value.lower() == "nan":
                value = f"feat_{idx}"
        except Exception:
            value = f"feat_{idx}"

        # Windows 文件名安全处理
        invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
        for ch in invalid_chars:
            value = value.replace(ch, "_")

        output_folder.mkdir(parents=True, exist_ok=True)

        # ★ 若重名，自动生成 xxx_2.tif, xxx_3.tif ...
        output_filepath = make_unique_filepath(output_folder, value, ".tif")

        with rasterio.open(output_filepath, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"{label} 掩膜完成：{output_filepath}")


def select_field(shp_file):
    """列出字段并让用户选择作为命名字段"""
    gdf = gpd.read_file(shp_file)
    fields = list(gdf.columns)

    print("\n=== 请选择用于命名输出 TIF 的字段 ===")
    for i, f in enumerate(fields):
        print(f"{i}. {f}")

    while True:
        try:
            idx = int(input("\n输入字段编号："))
            if 0 <= idx < len(fields):
                print(f"你选择了字段：{fields[idx]}")
                return fields[idx]
        except Exception:
            pass
        print("无效输入，请重新选择。")


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


def infer_dem_cf_paths(dataset_dir, dataset_name):
    """
    根据数据集目录自动推断 DEM / CF 路径：
    CE5_dem.tif
    CE5_CF.tif
    """
    dataset_dir = Path(dataset_dir)
    dem_path = dataset_dir / f"{dataset_name}_dem.tif"
    cf_path = dataset_dir / f"{dataset_name}_CF.tif"
    return dem_path, cf_path


def main(mode, dataset_dir, shp_source):

    dataset_dir = Path(dataset_dir)
    dataset_name = infer_dataset_name(dataset_dir)   # 例如 CE5

    shp_file = infer_shp_path(dataset_dir, shp_source, dataset_name)
    tif_path_DEM, tif_path_CF = infer_dem_cf_paths(dataset_dir, dataset_name)

    # ---------- 输出目录 ----------
    # manual -> Database/CE5/manual/dem 和 Database/CE5/manual/cf
    # yolo   -> Database/CE5/yolo/dem   和 Database/CE5/yolo/cf
    output_root = dataset_dir / shp_source
    output_dem = output_root / "dem"
    output_cf = output_root / "cf"

    # ---------- 基本检查 ----------
    if not dataset_dir.exists():
        raise FileNotFoundError(f"未找到数据集文件夹：{dataset_dir}")

    if not shp_file.exists():
        raise FileNotFoundError(f"未找到 shp 文件：{shp_file}")

    if not tif_path_DEM.exists():
        raise FileNotFoundError(f"未找到 DEM 文件：{tif_path_DEM}")

    if not tif_path_CF.exists():
        raise FileNotFoundError(f"未找到 CF 文件：{tif_path_CF}")

    print(f"\n当前数据集目录: {dataset_dir}")
    print(f"数据集名称: {dataset_name}")
    print(f"当前使用的 shp: {shp_file.name}")
    print(f"DEM 数据: {tif_path_DEM}")
    print(f"CF 数据 : {tif_path_CF}")
    print(f"输出根目录: {output_root}")
    print(f"DEM 输出目录: {output_dem}")
    print(f"CF 输出目录 : {output_cf}")

    # ---------- 选择字段 ----------
    field_name = select_field(shp_file)

    if mode == "a":
        # 仅执行 DEM 掩膜
        mask_raster(
            shp_file=shp_file,
            tif_path=tif_path_DEM,
            output_folder=output_dem,
            label="DEM",
            field_name=field_name
        )

    elif mode == "b":
        # 仅执行 CF 掩膜
        mask_raster(
            shp_file=shp_file,
            tif_path=tif_path_CF,
            output_folder=output_cf,
            label="CF",
            field_name=field_name
        )

    elif mode == "all":
        # DEM 和 CF 全部执行
        mask_raster(
            shp_file=shp_file,
            tif_path=tif_path_DEM,
            output_folder=output_dem,
            label="DEM",
            field_name=field_name
        )
        mask_raster(
            shp_file=shp_file,
            tif_path=tif_path_CF,
            output_folder=output_cf,
            label="CF",
            field_name=field_name
        )

    else:
        print("未知任务，请使用 --mode a / b / all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shapefile 掩膜工具")

    parser.add_argument(
        "--mode",
        choices=["a", "b", "all"],
        required=True,
        help="选择掩膜任务：a=DEM，b=CF，all=全部"
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