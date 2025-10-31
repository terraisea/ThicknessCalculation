from rasterio import open
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
import numpy as np

with open("E:\code\geo_processing\database\zhangzhongjing1.tif") as dataset:
    width = dataset.width
    height = dataset.height

# 读取空格分隔的 TXT 文件
data = np.loadtxt("E:\code\geo_processing\database\zhangzhongjing1020.txt")  # 默认按空格分割
rows = len(data)

results = []
output_folder = f"E:\code\geo_processing/database/UnifiedProject2"  # 建立到文件夹即可

for i in range(rows):
    x_center = data[i,1]
    y_center = data[i,2]
    box_width = data[i,3]
    box_height = data[i,4]

    x_center_coords = width*x_center
    y_center_coords = height*y_center

    width_coords=box_width*width/2
    height_coords=box_height*height/2

    X_LT=x_center_coords-width_coords
    Y_LT=y_center_coords-height_coords
    X_LB=x_center_coords+width_coords
    Y_LB=y_center_coords-height_coords
    X_RT = x_center_coords - width_coords
    Y_RT = y_center_coords + height_coords
    X_RB=x_center_coords+width_coords
    Y_RB=y_center_coords+height_coords

    #外扩像素,以得到环境平缓高程
    Pixel = 10
    XE_LT = X_LT-Pixel ; YE_LT = Y_LT-Pixel
    XE_LB = X_LB+Pixel ; YE_LB = Y_LB-Pixel
    XE_RT = X_RT-Pixel ; YE_RT = Y_RT+Pixel
    XE_RB = X_RB+Pixel ; YE_RB = Y_RB+Pixel
    # 保存为一行
    results.append([XE_LT, YE_LT, XE_LB, YE_LB, XE_RT, YE_RT, XE_RB, YE_RB])
    Path(output_folder).mkdir(parents=True, exist_ok=True)  # ✅ 如果文件夹不存在则自动创建
    output_path = f"{output_folder}/expanded_coordinates.txt"
    np.savetxt(output_path, results, fmt="%.6f", delimiter="\t",
               header="XE_LT\tYE_LT\tXE_LB\tYE_LB\tXE_RT\tYE_RT\tXE_RB\tYE_RB", comments='')

    #dataset.transform可将像素坐标转换为同tif同坐标系的下的经纬度
    xlt, ylt = dataset.transform * (X_LT, Y_LT)
    xlb, ylb = dataset.transform * (X_LB, Y_LB)
    xrt, yrt = dataset.transform * (X_RT, Y_RT)
    xrb, yrb = dataset.transform * (X_RB, Y_RB)

 #如果要导入arcmap需要360-x轴坐标，这里不再展示见https://github.com/terraisea/LearningRasterio的YOLOtoSHP
 #原因是arcmap没有moon2000的参考CRS
    poly_coords = [(xlt,ylt), (xlb, ylb), (xrb, yrb),(xrt, yrt), (xlt,ylt)]#注意闭合

    polygon = Polygon(poly_coords)

    # Moon 2000 GEOGCS WKT
    moon2000_wkt = 'GEOGCS["GCS_Moon_2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'

    gdf_poly = gpd.GeoDataFrame({'name': ['Region1'], 'geometry': [polygon]}, crs=moon2000_wkt)

    # 获取第一列的值作为文件名
    filename = str(data[i][1])

    # 构造完整路径
    Path(output_folder).mkdir(parents=True, exist_ok=True)  # ✅ 如果文件夹不存在则自动创建
    output_filepath = f"{output_folder}/{filename}.shp"
    gdf_poly.to_file(output_filepath, driver="ESRI Shapefile")

    print(f"Saved {filename}")