import rasterio
import numpy as np
import matplotlib.pyplot as plt

# 1. 打开 DEM
tif_path = r"E:\code\geo_processing\database\testdata\mask\DEM\0.61232.tif"
with rasterio.open(tif_path) as src:
    img = src.read()         # numpy 数组，形状 (bands, height, width)
    profile = src.profile    # 保存元数据（投影、分辨率等）

row_median = img.shape[2] // 2
col_median = img.shape[1] // 2
row_values = img[0,col_median-1,:]  #0表示第一波段，row_median-1是因为从0行/列开始，行的值是中间列
# 定义一个极小阈值，例如 -3e10
threshold = -3e10
# 判断首尾是否为异常值
if row_values[0] < threshold:   #边缘值可能为nodata=-3.402823e+38，舍弃
    row_values = row_values[1:]  # 去掉第一个
if row_values[-1] < threshold:
    row_values = row_values[:-1]  # 去掉最后一个

col_values = img[0, :, row_median-1]
# 判断首尾是否为异常值
if col_values[0] < threshold:   #边缘值可能为nodata=-3.402823e+38，舍弃
    col_values = col_values[1:]  # 去掉第一个
if col_values[-1] < threshold:
    col_values = col_values[:-1]  # 去掉最后一个

x1 = np.arange(0, len(col_values), 1)
y1 = col_values
x2 = np.arange(0, len(row_values), 1)
y2 = row_values

# 2. 打开 CF
tif_path2 = r"E:\code\geo_processing\database\testdata\mask\CF\0.61232.tif"
with rasterio.open(tif_path2) as src2:
    img2 = src2.read()         # numpy 数组，形状 (bands, height, width)
    profile2 = src2.profile    # 保存元数据（投影、分辨率等）

row_median2 = img2.shape[2] // 2
col_median2 = img2.shape[1] // 2
row_values2 = img2[0,col_median2-1,:]  #0表示第一波段
# 判断首尾是否为异常值
if row_values2[0] < threshold:   #边缘值可能为nodata=-3.402823e+38，舍弃
    row_values2 = row_values2[1:]  # 去掉第一个
if row_values2[-1] < threshold:
    row_values2 = row_values2[:-1]  # 去掉最后一个

col_values2 = img2[0, :, row_median2-1]
# 判断首尾是否为异常值
if col_values2[0] < threshold:   #边缘值可能为nodata=-3.402823e+38，舍弃
    col_values2 = col_values2[1:]  # 去掉第一个
if col_values2[-1] < threshold:
    col_values2 = col_values2[:-1]  # 去掉最后一个


x3 = np.arange(0, len(col_values2), 1)
y3 = col_values2
x4 = np.arange(0, len(row_values2), 1)
y4 = row_values2

plt.subplot(2, 2, 1)
plt.plot(x1, y1, label='elevation', color='blue', linewidth=2)
plt.title("ColCalculateThickness")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

plt.subplot(2, 2, 2)
plt.plot(x2, y2, label='elevation', color='blue', linewidth=2)
plt.title("RowCalculateThickness")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.tight_layout(pad=2.0)
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

plt.subplot(2, 2, 3)
plt.plot(x3, y3, label='elevation', color='blue', linewidth=2)
plt.title("ColCF")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

plt.subplot(2, 2, 4)
plt.plot(x4, y4, label='elevation', color='blue', linewidth=2)
plt.title("RowCF")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

plt.savefig(r"E:\code\geo_processing\database\testdata\show\show.png")
plt.show()

np.savetxt(r"E:\code\geo_processing\database\testdata\show\ColCalculateThickness_xy_values.csv", np.column_stack((x1, y1)), delimiter=",", header="x,y", comments='')#去掉文件头部前面的表示去除csv的“#”
np.savetxt(r"E:\code\geo_processing\database\testdata\show\RowCalculateThickness_xy_values.csv", np.column_stack((x2, y2)), delimiter=",", header="x,y", comments='')
np.savetxt(r"E:\code\geo_processing\database\testdata\show\ColCF_xy_values.csv", np.column_stack((x3, y3)), delimiter=",", header="x,y", comments='')
np.savetxt(r"E:\code\geo_processing\database\testdata\show\RowCF_xy_values.csv", np.column_stack((x4, y4)), delimiter=",", header="x,y", comments='')
