import argparse
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from pathlib import Path

def mask_raster(shp_list, tif_path, output_folder, label):
    """对一组 shapefile 执行掩膜"""
    for shp in shp_list:
        gdf = gpd.read_file(shp)
        geoms = [mapping(geom) for geom in gdf.geometry]

        with rasterio.open(tif_path) as src:
            out_image, out_transform = mask(src, geoms, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        shp_name = shp.stem
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_filepath = f"{output_folder}/{shp_name}.tif"

        with rasterio.open(output_filepath, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"{label} 掩膜完成，输出文件：{output_filepath}")


def main(mode):
    folder = Path(r"E:\code\geo_processing\database\UnifiedProject")   #shp文件的文件夹，⭐注意:路径只到文件夹
    shp_list = list(folder.glob("*.shp"))  #可以自动读取文件夹内的shp

    tif_path_DEM = r"E:\code\geo_processing\database\DEM\ZhangDEM1.tif" #被掩膜的DEM数据的路径，⭐需要写到该文件的全部路径
    tif_path_CF = r"E:\code\geo_processing\database\DEM\ZhangCF.tif"    #被掩膜的CF数据的路径，⭐需要写到该文件的全部路径

    if mode == "a":
        # 仅执行 DEM 掩膜
        mask_raster(
            shp_list,
            tif_path_DEM,
            r"E:\code\geo_processing\database\testdata\mask\DEMargparse",#掩膜得到的DEM数据的文件夹路径，⭐注意:路径只到文件夹
            "DEM"
        )

    elif mode == "b":
        # 仅执行 CF 掩膜
        mask_raster(
            shp_list,
            tif_path_CF,
            r"E:\code\geo_processing\database\testdata\mask\CFargparse",#掩膜得到的CF数据的文件夹路径，⭐注意:路径只到文件夹
            "CF"
        )

    elif mode == "all":
        # 两个都执行
        mask_raster(
            shp_list,
            tif_path_DEM,
            r"E:\code\geo_processing\database\testdata\mask\DEMargparse",#掩膜得到的DEM数据的文件夹路径，⭐注意:路径只到文件夹
            "DEM"
        )
        mask_raster(
            shp_list,
            tif_path_CF,
            r"E:\code\geo_processing\database\testdata\mask\CFargparse",#掩膜得到的CF数据的文件夹路径，⭐注意:路径只到文件夹
            "CF"
        )
    else:
        print("未知任务，请使用 --mode a / b / all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shapefile 掩膜工具")
    parser.add_argument("--mode", choices=["a", "b", "all"], required=True,
                        help="选择掩膜任务：a=DEM，b=CF，all=全部")
    args = parser.parse_args()
    main(args.mode)
