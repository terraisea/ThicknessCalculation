# split_tif_and_png_fixed.py
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.transform import Affine
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

def write_png_auxxml(png_path: Path, srs_wkt: str, gt_list):
    aux_path = png_path.with_suffix(png_path.suffix + ".aux.xml")
    root = ET.Element("PAMDataset")
    srs_el = ET.SubElement(root, "SRS")
    srs_el.set("dataAxisToSRSAxisMapping", "2,1")
    srs_el.text = srs_wkt
    gt_el = ET.SubElement(root, "GeoTransform")
    gt_el.text = " " + ", ".join([format(v, ".15g") for v in gt_list])
    tree = ET.ElementTree(root)
    tree.write(aux_path, encoding="utf-8", xml_declaration=True)

def scale_to_uint8(arr):
    """线性拉伸 numpy 数组到 uint8 (0-255)。接受 2D array 或 3D (C,H,W)"""
    if arr.dtype == np.uint8:
        return arr
    # For 3D, scale per band
    if arr.ndim == 3:
        bands = []
        for b in range(arr.shape[0]):
            band = arr[b].astype("float64")
            mask = np.isfinite(band)
            if mask.any():
                mn = band[mask].min()
                mx = band[mask].max()
                if mx > mn:
                    scaled = (band - mn) / (mx - mn) * 255.0
                else:
                    scaled = np.zeros_like(band)
            else:
                scaled = np.zeros_like(band)
            bands.append(np.clip(scaled, 0, 255).astype(np.uint8))
        return np.stack(bands, axis=0)
    else:
        # 2D
        band = arr.astype("float64")
        mask = np.isfinite(band)
        if mask.any():
            mn = band[mask].min()
            mx = band[mask].max()
            if mx > mn:
                scaled = (band - mn) / (mx - mn) * 255.0
            else:
                scaled = np.zeros_like(band)
        else:
            scaled = np.zeros_like(band)
        return np.clip(scaled, 0, 255).astype(np.uint8)

def arr_to_pil_image(arr):
    """
    输入 arr:
      - shape (3,H,W)  或 (H,W,3) -> 返回 RGB PIL
      - shape (1,H,W) 或 (H,W) -> 转为灰度再 convert('RGB')
    注意：期望输入为 uint8
    """
    if arr.ndim == 3 and arr.shape[0] == 3:
        # CHW -> HWC
        imarr = np.transpose(arr, (1,2,0))
        return Image.fromarray(imarr)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        imarr = arr[0]
        return Image.fromarray(imarr).convert("RGB")
    elif arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr)
    else:
        # try to collapse first bands into 3
        if arr.ndim == 3:
            ch = arr.shape[0]
            if ch >= 3:
                imarr = np.transpose(arr[:3], (1,2,0))
                return Image.fromarray(imarr)
        raise ValueError("无法将数组转为 PIL.Image，数组形状: " + str(arr.shape))

def split_tif_and_png(input_tif, nx, ny, output_folder):
    input_tif = Path(input_tif)
    outdir = Path(output_folder)
    outdir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_tif) as src:
        W = src.width
        H = src.height
        # tile size (整数除法，右/下边可能会被截掉以保持均分)
        tile_w = W // nx
        tile_h = H // ny

        crs = src.crs
        srs_wkt = crs.to_wkt() if crs is not None else ""
        dtype = src.dtypes[0]
        nodata = src.nodata

        for row in range(ny):
            for col in range(nx):
                left = col * tile_w
                top = row * tile_h
                win = Window(left, top, tile_w, tile_h)

                # accurate transform for this window
                tile_transform = window_transform(win, src.transform)

                # read data (bands, h, w)
                data = src.read(window=win)
                # If read returns smaller window at right/bottom (shouldn't if exact div)
                h_win = data.shape[1]
                w_win = data.shape[2]

                # ---------------- write GeoTIFF (keep original dtype, nodata, count)
                out_tif = outdir / f"tile_{row}_{col}.tif"
                profile = src.profile.copy()
                profile.update({
                    "driver": "GTiff",
                    "height": h_win,
                    "width": w_win,
                    "transform": tile_transform,
                    "crs": src.crs
                })
                with rasterio.open(out_tif, "w", **profile) as dst:
                    dst.write(data)

                # ---------------- prepare PNG for YOLO (8-bit RGB)
                # Convert to uint8 properly
                if data.shape[0] >= 3:
                    # use first 3 bands as RGB
                    rgb = data[:3]
                    if rgb.dtype != np.uint8:
                        rgb_u8 = scale_to_uint8(rgb)
                    else:
                        rgb_u8 = rgb
                else:
                    # single band: scale to uint8
                    single = data[0]
                    single_u8 = scale_to_uint8(single)
                    rgb_u8 = np.expand_dims(single_u8, 0)

                # create PIL image and save
                pil_img = arr_to_pil_image(rgb_u8)
                out_png = outdir / f"tile_{row}_{col}.png"
                pil_img.save(out_png)

                # ---------------- write aux.xml for png (use tile_transform to compute GT list)
                # rasterio Affine: a, b, c, d, e, f -> corresponds to:
                # transform = Affine(a, b, c, d, e, f)
                # We want GT list: [GT0, GT1, GT2, GT3, GT4, GT5] matching earlier convention:
                # GT0 = c, GT1 = a, GT2 = b, GT3 = f, GT4 = d, GT5 = e
                a = tile_transform.a
                b = tile_transform.b
                c = tile_transform.c
                d = tile_transform.d
                e = tile_transform.e
                f = tile_transform.f
                gt_list = [c, a, b, f, d, e]
                write_png_auxxml(out_png, srs_wkt, gt_list)

    print("完成：输出在", outdir)

# ------------- 仅改这里运行 -------------
if __name__ == "__main__":
    split_tif_and_png(
        input_tif=r"E:\ArcGISProject\ce51205\CE5_WAC.tif",  # 你的输入 GeoTIFF（必须带投影）
        nx=2,   # 横向切成几块
        ny=2,   # 纵向切成几块
        output_folder=r"E:\ArcGISProject\ce51205\split"  # 输出目录
    )
