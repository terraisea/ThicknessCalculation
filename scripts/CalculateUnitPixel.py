import rasterio

with rasterio.open(r"E:\code\geo_processing\database\DEM\ZhangDEM1.tif") as src:
    transform = src.transform
    xres = abs(transform.a)  # 每像元宽度
    yres = abs(transform.e)  # 每像元高度
    print(f"X方向像元大小: {xres} °, Y方向像元大小: {yres} °")