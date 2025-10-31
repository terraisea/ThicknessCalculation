import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#路径设置#
base_dir = Path(r"E:\code\geo_processing\database\testdata\mask")
dem_dir = base_dir / "DEM"
cf_dir = base_dir / "CF"
output_dir = base_dir / "show"
output_dir.mkdir(exist_ok=True, parents=True)
#常数设置
# 定义一个极小阈值，例如 -3e10
threshold = -3e10

# 1. 打开 DEM
for dem_file in dem_dir.glob("*.tif"):
    filename = dem_file.stem  # 不带后缀的文件名，比如 "0.61232"
    cf_file = cf_dir / f"{filename}.tif"  # 查找 CF 下的同名文件

    # ========== 打开 DEM ==========
    with rasterio.open(dem_file) as src:
        img = src.read().astype(float)
        img[img < threshold] = np.nan  # 将 nodata 转为 nan
    row_median = img.shape[2] // 2
    col_median = img.shape[1] // 2
    row_values = img[0, col_median - 1, :]
    col_values = img[0, :, row_median - 1]

    x1 = np.arange(len(col_values))
    y1 = col_values
    x2 = np.arange(len(row_values))
    y2 = row_values

    # ========== 打开 CF ==========
    with rasterio.open(cf_file) as src2:
        img2 = src2.read().astype(float)
        img2[img2 < threshold] = np.nan
    row_median2 = img2.shape[2] // 2
    col_median2 = img2.shape[1] // 2
    row_values2 = img2[0, col_median2 - 1, :]
    col_values2 = img2[0, :, row_median2 - 1]

    x3 = np.arange(len(col_values2))
    y3 = col_values2
    x4 = np.arange(len(row_values2))
    y4 = row_values2

    # ========== 绘图 ==========
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(x1, y1, label='DEM Col', color='blue', linewidth=2)
    plt.title("ColCalculateThickness")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x2, y2, label='DEM Row', color='blue', linewidth=2)
    plt.title("RowCalculateThickness")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x3, y3, label='CF Col', color='green', linewidth=2)
    plt.title("ColCF")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x4, y4, label='CF Row', color='green', linewidth=2)
    plt.title("RowCF")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.legend()
    plt.grid(True)

    plt.tight_layout(pad=2.0)

    # 保存图像
    out_img = output_dir / f"{filename}_show.png"
    plt.savefig(out_img, dpi=200)
    plt.close()

    # ========== 保存数据 ==========
    np.savetxt(output_dir / f"{filename}_ColCalculateThickness.csv",
               np.column_stack((x1, y1)), delimiter=",", header="x,y", comments='')
    np.savetxt(output_dir / f"{filename}_RowCalculateThickness.csv",
               np.column_stack((x2, y2)), delimiter=",", header="x,y", comments='')
    np.savetxt(output_dir / f"{filename}_ColCF.csv",
               np.column_stack((x3, y3)), delimiter=",", header="x,y", comments='')
    np.savetxt(output_dir / f"{filename}_RowCF.csv",
               np.column_stack((x4, y4)), delimiter=",", header="x,y", comments='')
    print(filename,"已完成")

print("✅ 所有文件处理完成！结果已保存到:", output_dir)