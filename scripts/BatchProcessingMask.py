import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from pathlib import Path

# 读取 shapefile
folder = Path(r"E:\code\geo_processing\database\UnifiedProject")#路径到文件夹
# 获取所有 .shp 文件
shp_list = list(folder.glob("*.shp"))
# 读取 shapefile
shp_path = shp_list

# 读取栅格
tif_path_DEM = r"E:\code\geo_processing\database\DEM\ZhangDEM1.tif"
tif_path_CF = r"E:\code\geo_processing\database\DEM\ZhangCF.tif"

for shp in shp_path:
    gdf = gpd.read_file(shp)
    geoms = [mapping(geom) for geom in gdf.geometry] #单个与多个不冲突

    #DEM
    # 打开 tif 并掩膜
    with rasterio.open(tif_path_DEM) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

    # 更新元数据并写出结果
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    shp_name = shp.stem  # 只获取文件名（不含路径、不含后缀）
    output_folder = r"E:\code\geo_processing\database\testdata\mask\DEM"
    Path(output_folder).mkdir(parents=True, exist_ok=True)# ✅ 如果文件夹不存在则自动创建
    output_filepath = f"{output_folder}/{shp_name}.tif"

    with rasterio.open(output_filepath, "w", **out_meta) as dest:
        dest.write(out_image)

    print("DEM掩膜完成，输出文件：", output_filepath)

    # cf
    # 打开 tif 并掩膜
    with rasterio.open(tif_path_CF) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

    # 更新元数据并写出结果
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    shp_name = shp.stem  # 只获取文件名（不含路径、不含后缀）
    output_folder = r"E:\code\geo_processing\database\testdata\mask\CF"
    Path(output_folder).mkdir(parents=True, exist_ok=True)  # ✅ 如果文件夹不存在则自动创建
    output_filepath = f"{output_folder}/{shp_name}.tif"

    with rasterio.open(output_filepath, "w", **out_meta) as dest:
        dest.write(out_image)

    print("CF掩膜完成，输出文件：", output_filepath)

