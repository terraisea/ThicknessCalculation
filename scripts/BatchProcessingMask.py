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
for shp in shp_path:
    gdf = gpd.read_file(shp)
    geoms = [mapping(geom) for geom in gdf.geometry] #单个与多个不冲突

    # 读取栅格
    tif_path = r"E:\code\geo_processing\database\zhangzhongjing1.tif"
    # 打开 tif 并掩膜
    with rasterio.open(tif_path) as src:
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
    output_folder = r"E:\code\geo_processing\database\testdata\mask"
    output_filepath = f"{output_folder}/{shp_name}.tif"

    with rasterio.open(output_filepath, "w", **out_meta) as dest:
        dest.write(out_image)

    print("掩膜完成，输出文件：", output_filepath)
