import argparse
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from pathlib import Path

def mask_raster(shp_list, tif_path, output_folder, label, field_name):
    """对一组 shapefile 执行掩膜"""
    for shp in shp_list:
        gdf = gpd.read_file(shp)

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
                value = str(row[field_name])
            except Exception:
                value = f"feat_{idx}"

            shp_name = shp.stem
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            output_filepath = f"{output_folder}/{shp_name}_{value}.tif"

            with rasterio.open(output_filepath, "w", **out_meta) as dest:
                dest.write(out_image)

            print(f"{label} 掩膜完成：{output_filepath}")


def select_field(shp_list):
    """统一列出字段并让用户选择作为命名字段"""
    # 读取第一个 shp 作为字段参考
    gdf = gpd.read_file(shp_list[0])
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
        except:
            pass
        print("无效输入，请重新选择。")


def main(mode):
    folder = Path(r"E:\code\geo_processing\database\marius\marius\manual")   #shp文件的文件夹，⭐注意:路径只到文件夹
    shp_list = list(folder.glob("*.shp"))  #可以自动读取文件夹内的shp

    tif_path_DEM = r"E:\code\geo_processing\database\marius\marius\demmar.tif" #被掩膜的DEM数据的路径，⭐需要写到该文件的全部路径
    tif_path_CF = r"E:\code\geo_processing\database\marius\marius\cfmar.tif"    #被掩膜的CF数据的路径，⭐需要写到该文件的全部路径

    # ★ 新增：选择字段
    field_name = select_field(shp_list)

    if mode == "a":
        # 仅执行 DEM 掩膜
        mask_raster(
            shp_list,
            tif_path_DEM,
            r"E:\code\geo_processing\database\marius\marius\dem",#掩膜得到的DEM数据的文件夹路径，⭐注意:路径只到文件夹
            "DEM",
            field_name
        )

    elif mode == "b":
        # 仅执行 CF 掩膜
        mask_raster(
            shp_list,
            tif_path_CF,
            r"E:\code\geo_processing\database\marius\marius\CF",#掩膜得到的CF数据的文件夹路径，⭐注意:路径只到文件夹
            "CF",
            field_name
        )

    elif mode == "all":
        # 两个都执行
        mask_raster(
            shp_list,
            tif_path_DEM,
            r"E:\code\geo_processing\database\marius\marius\dem",#掩膜得到的DEM数据的文件夹路径，⭐注意:路径只到文件夹
            "DEM",
            field_name
        )
        mask_raster(
            shp_list,
            tif_path_CF,
            r"E:\code\geo_processing\database\marius\marius\cf",#掩膜得到的CF数据的文件夹路径，⭐注意:路径只到文件夹
            "CF",
            field_name
        )
    else:
        print("未知任务，请使用 --mode a / b / all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shapefile 掩膜工具")
    parser.add_argument("--mode", choices=["a", "b", "all"], required=True,
                        help="选择掩膜任务：a=DEM，b=CF，all=全部")
    args = parser.parse_args()
    main(args.mode)
