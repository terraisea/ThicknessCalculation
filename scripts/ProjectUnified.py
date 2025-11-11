from rasterio import open
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
import numpy as np

with open(r"E:\code\geo_processing\database\marius\mar2.tif") as dataset:
    width = dataset.width
    height = dataset.height

data = np.loadtxt(r"E:\code\geo_processing\database\marius\marius.txt") #YOLO生成的坐标txt
rows = len(data)

results = []
output_folder = r"E:\code\geo_processing\database\marius\shp"#输出文件夹路径，注意是文件夹

# ✅ 同时创建 ExRegion 子文件夹
Path(output_folder).mkdir(parents=True, exist_ok=True)
Path(f"{output_folder}/ExRegion").mkdir(parents=True, exist_ok=True)

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

    moon2000_wkt = (
        'GEOGCS["GCS_Moon_2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
        'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
        'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'
    )

    gdf_poly = gpd.GeoDataFrame({'name': ['Region1'], 'geometry': [polygon]}, crs=moon2000_wkt)
    gdf_poly_ex = gpd.GeoDataFrame({'name': ['ExRegion'], 'geometry': [polygon_expanded]}, crs=moon2000_wkt)

    filename = str(data[i][1])

    output_filepath = f"{output_folder}/{filename}.shp"
    output_filepath_Ex = f"{output_folder}/ExRegion/{filename}.shp"

    gdf_poly.to_file(output_filepath, driver="ESRI Shapefile")
    gdf_poly_ex.to_file(output_filepath_Ex, driver="ESRI Shapefile")

    print(f"✅ Saved {filename}.shp and ExRegion/{filename}.shp")
