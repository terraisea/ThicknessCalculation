import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping

# 读取 shapefile
shp_path = r"E:\code\geo_processing\database\UnifiedProject\0.61232.shp"
gdf = gpd.read_file(shp_path)
geoms = [mapping(geom) for geom in gdf.geometry] #单个与多个不冲突

# 读取栅格
tif_path = r"E:\code\geo_processing\database\DEM\ZhangDEM1.tif"
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

output_path = r"E:\code\geo_processing\database\testdata\0.61232.tif"

with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(out_image)

print("掩膜完成，输出文件：", output_path)



