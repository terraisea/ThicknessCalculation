from rasterio import open
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
import numpy as np

with open(r"E:\code\geo_processing\database\ce6\yolo\region\rb.tif") as dataset:
    width = dataset.width
    height = dataset.height

data = np.loadtxt(r"E:\code\geo_processing\database\ce6\yolo\rb.txt") #YOLO生成的坐标txt
rows = len(data)

results = []
output_folder = r"E:\code\geo_processing\database\ce6\yolo\shp"#输出文件夹路径，注意是文件夹

# ✅ 同时创建 ExRegion 子文件夹
Path(output_folder).mkdir(parents=True, exist_ok=True)
Path(f"{output_folder}/ExRegion").mkdir(parents=True, exist_ok=True)

# ================================
# 收集合并到一个 SHP 的列表
# ================================
poly_list = []
poly_list_ex = []
name_list = []
name_list_ex = []

for i in range(rows):
    x_center = data[i,1]
    y_center = data[i,2]
    box_width = data[i,3]
    box_height = data[i,4]

    x_center_coords = width * x_center
    y_center_coords = height * y_center
    width_coords = box_width * width / 2
    height_coords = box_height * height / 2

    X_LT = x_center_coords - width_coords
    Y_LT = y_center_coords - height_coords
    X_LB = x_center_coords + width_coords
    Y_LB = y_center_coords - height_coords
    X_RT = x_center_coords - width_coords
    Y_RT = y_center_coords + height_coords
    X_RB = x_center_coords + width_coords
    Y_RB = y_center_coords + height_coords

    xlt, ylt = dataset.transform * (X_LT, Y_LT)
    xlb, ylb = dataset.transform * (X_LB, Y_LB)
    xrt, yrt = dataset.transform * (X_RT, Y_RT)
    xrb, yrb = dataset.transform * (X_RB, Y_RB)

    poly_coords = [(xlt, ylt), (xlb, ylb), (xrb, yrb), (xrt, yrt), (xlt, ylt)]
    polygon = Polygon(poly_coords)

    # 外扩
    Pixel = 10   #由于DEM与WAC分辨率相差0.4224倍，此处应该需要考虑一下。wac ✖ 0.4224 = DEM
    XE_LT = X_LT - Pixel; YE_LT = Y_LT - Pixel
    XE_LB = X_LB + Pixel; YE_LB = Y_LB - Pixel
    XE_RT = X_RT - Pixel; YE_RT = Y_RT + Pixel
    XE_RB = X_RB + Pixel; YE_RB = Y_RB + Pixel

    xelt, yelt = dataset.transform * (XE_LT, YE_LT)
    xelb, yelb = dataset.transform * (XE_LB, YE_LB)
    xert, yert = dataset.transform * (XE_RT, YE_RT)
    xerb, yerb = dataset.transform * (XE_RB, YE_RB)

    poly_expanded = [(xelt, yelt), (xelb, yelb), (xerb, yerb), (xert, yert), (xelt, yelt)]
    polygon_expanded = Polygon(poly_expanded)

    # name 使用 YOLO txt 第二列作为字符串
    name_val = str(data[i][1])

    poly_list.append(polygon)
    name_list.append(name_val)

    poly_list_ex.append(polygon_expanded)
    name_list_ex.append(name_val)

    print(f"已加入 {name_val}")

# ================================
# 生成一个 SHP（原框）
# ================================
moon2000_wkt = (
    'GEOGCS["GCS_Moon_2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'
)

gdf_all = gpd.GeoDataFrame({"name": name_list, "geometry": poly_list}, crs=moon2000_wkt)
gdf_all.to_file(f"{output_folder}/rb_all_boxes.shp", driver="ESRI Shapefile") #⭐记得改路径名

# ================================
# 生成一个 SHP（ExRegion 扩大框）
# ================================
gdf_ex_all = gpd.GeoDataFrame({"name": name_list_ex, "geometry": poly_list_ex}, crs=moon2000_wkt)
gdf_ex_all.to_file(f"{output_folder}/ExRegion/rb_all_expanded.shp", driver="ESRI Shapefile") #⭐记得改路径名

print("🎉 所有 Polygon 已成功合并输出到两个 SHP 文件！")
