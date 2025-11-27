import argparse
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================================
#  主处理函数
# ===============================================
def process_files(mode):
    threshold = -3e10  # 固定阈值，用于后面过滤rasterio进行掩膜得到的tif的nodata区域

    # ========== 路径设置（保持原样） ==========
    base_dir = Path(r"E:\code\geo_processing\database\marius\marius\YOLO")
    #BatchProcessingMask.py运行后得到的文件夹，⭐注意:路径只到文件夹
    dem_dir = base_dir / "dem"  #读取文件夹下的DEM
    cf_dir = base_dir / "cf"    #读取文件夹下的DEM
    output_dir = base_dir / "show"  #输出文件的路径，⭐注意:路径只到文件夹
    output_dir.mkdir(exist_ok=True, parents=True)

    # ========== 处理 dem 或 cf 文件 ==========
    if mode in ["dem", "cf"]:
        current_dir = dem_dir if mode == "dem" else cf_dir
        color_col = "blue" if mode == "dem" else "green"
        color_row = "blue" if mode == "dem" else "green"

        for file in current_dir.glob("*.tif"):
            filename = file.stem
            print(f"▶ 正在处理: {filename} ({mode.upper()})")

            with rasterio.open(file) as src:
                img = src.read().astype(float)
                img[img < threshold] = np.nan

                transform = src.transform  # 获取仿射变换，用于经纬度计算

            row_mid = img.shape[2] // 2
            col_mid = img.shape[1] // 2
            row_values = img[0, col_mid - 1, :]
            col_values = img[0, :, row_mid - 1]

            x1 = np.arange(len(col_values))
            y1 = col_values
            x2 = np.arange(len(row_values))
            y2 = row_values

            # ---------- 经纬度计算 ----------
            # 列方向剖面
            rows = np.arange(img.shape[1])
            cols = np.full_like(rows, row_mid - 1)
            lon_col, lat_col = rasterio.transform.xy(transform, rows, cols)

            # 行方向剖面
            cols2 = np.arange(img.shape[2])
            rows2 = np.full_like(cols2, col_mid - 1)
            lon_row, lat_row = rasterio.transform.xy(transform, rows2, cols2)

            # ---------- 绘图 2x1 ----------
            plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            plt.plot(x1, y1, label=f"{mode.upper()} Col", color=color_col, linewidth=2)
            plt.title(f"{mode.upper()} Col Profile")
            plt.xlabel("X axis")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(x2, y2, label=f"{mode.upper()} Row", color=color_row, linewidth=2)
            plt.title(f"{mode.upper()} Row Profile")
            plt.xlabel("X axis")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            plt.tight_layout(pad=2.0)
            plt.savefig(output_dir / f"{filename}_{mode}_show.png", dpi=200)
            plt.close()

            # ---------- 保存 CSV（value, lon, lat） ----------
            np.savetxt(output_dir / f"{filename}_{mode}_Col.csv",
                       np.column_stack((y1, lon_col, lat_col)),
                       delimiter=",", header="value,lon,lat", comments='')

            np.savetxt(output_dir / f"{filename}_{mode}_Row.csv",
                       np.column_stack((y2, lon_row, lat_row)),
                       delimiter=",", header="value,lon,lat", comments='')

            print(f"✅ {filename} 已完成 ({mode.upper()})")

    # ========== 处理 all 模式 ==========
    elif mode == "all":
        dem_files = list(dem_dir.glob("*.tif"))
        for dem_file in dem_files:
            filename = dem_file.stem
            cf_file = cf_dir / f"{filename}.tif"
            if not cf_file.exists():
                print(f"⚠ CF 文件不存在: {cf_file}, 跳过")
                continue
            print(f"▶ 正在处理: {filename} (ALL)")

            # ---------- DEM ----------
            with rasterio.open(dem_file) as src:
                dem_img = src.read().astype(float)
                dem_img[dem_img < threshold] = np.nan
                transform = src.transform

            dem_row_mid = dem_img.shape[2] // 2
            dem_col_mid = dem_img.shape[1] // 2
            dem_row_values = dem_img[0, dem_col_mid - 1, :]
            dem_col_values = dem_img[0, :, dem_row_mid - 1]

            x1 = np.arange(len(dem_col_values))
            y1 = dem_col_values
            x2 = np.arange(len(dem_row_values))
            y2 = dem_row_values

            # ---------- DEM 经纬度 ----------
            rows = np.arange(dem_img.shape[1])
            cols = np.full_like(rows, dem_row_mid - 1)
            lon_col, lat_col = rasterio.transform.xy(transform, rows, cols)
            cols2 = np.arange(dem_img.shape[2])
            rows2 = np.full_like(cols2, dem_col_mid - 1)
            lon_row, lat_row = rasterio.transform.xy(transform, rows2, cols2)

            # ---------- CF ----------
            with rasterio.open(cf_file) as src2:
                cf_img = src2.read().astype(float)
                cf_img[cf_img < threshold] = np.nan
                transform_cf = src2.transform

            cf_row_mid = cf_img.shape[2] // 2
            cf_col_mid = cf_img.shape[1] // 2
            cf_row_values = cf_img[0, cf_col_mid - 1, :]
            cf_col_values = cf_img[0, :, cf_row_mid - 1]

            x3 = np.arange(len(cf_col_values))
            y3 = cf_col_values
            x4 = np.arange(len(cf_row_values))
            y4 = cf_row_values

            # ---------- CF 经纬度 ----------
            rows_cf = np.arange(cf_img.shape[1])
            cols_cf = np.full_like(rows_cf, cf_row_mid - 1)
            lon_col_cf, lat_col_cf = rasterio.transform.xy(transform_cf, rows_cf, cols_cf)
            cols2_cf = np.arange(cf_img.shape[2])
            rows2_cf = np.full_like(cols2_cf, cf_col_mid - 1)
            lon_row_cf, lat_row_cf = rasterio.transform.xy(transform_cf, rows2_cf, cols2_cf)

            # ---------- 绘图 2x2 ----------
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.plot(x1, y1, label="DEM Col", color="blue", linewidth=2)
            plt.title("DEM Col Profile")
            plt.xlabel("X axis")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.plot(x2, y2, label="DEM Row", color="blue", linewidth=2)
            plt.title("DEM Row Profile")
            plt.xlabel("X axis")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.plot(x3, y3, label="CF Col", color="green", linewidth=2)
            plt.title("CF Col Profile")
            plt.xlabel("X axis")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot(x4, y4, label="CF Row", color="green", linewidth=2)
            plt.title("CF Row Profile")
            plt.xlabel("X axis")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            plt.tight_layout(pad=2.0)
            plt.savefig(output_dir / f"{filename}_all_show.png", dpi=200)
            plt.close()

            # ---------- 保存 CSV（value, lon, lat） ----------
            np.savetxt(output_dir / f"{filename}_DEM_Col.csv",
                       np.column_stack((y1, lon_col, lat_col)),
                       delimiter=",", header="value,lon,lat", comments='')

            np.savetxt(output_dir / f"{filename}_DEM_Row.csv",
                       np.column_stack((y2, lon_row, lat_row)),
                       delimiter=",", header="value,lon,lat", comments='')

            np.savetxt(output_dir / f"{filename}_CF_Col.csv",
                       np.column_stack((y3, lon_col_cf, lat_col_cf)),
                       delimiter=",", header="value,lon,lat", comments='')

            np.savetxt(output_dir / f"{filename}_CF_Row.csv",
                       np.column_stack((y4, lon_row_cf, lat_row_cf)),
                       delimiter=",", header="value,lon,lat", comments='')

            print(f"✅ {filename} 已完成 (ALL)")

    print(f"🎯 所有文件处理完成！结果已保存到: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量处理 DEM / CF 文件，输出剖面图与CSV")
    parser.add_argument("--mode", type=str, choices=["dem", "cf", "all"], default="all",
                        help="选择处理模式: dem / cf / all")
    args = parser.parse_args()
    process_files(mode=args.mode)
