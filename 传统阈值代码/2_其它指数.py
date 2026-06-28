"""
生成其他指数特征图（MNDWI、NDBI等）
功能：读取 Landsat TIF 影像，计算 MNDWI 和 NDBI，保存为 GeoTIFF
"""

import numpy as np
from osgeo import gdal
import os

# ==================== 配置路径 ====================
# 输入 TIF 文件（与之前计算 NDVI/NDWI 使用同一影像）
tif_file = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\Nepal_Landsat\scale_along_median_2016nepal2.tif"

# 输出目录（建议单独新建一个文件夹存放其他指数）
output_dir = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\指数特征图+经验阈值"
os.makedirs(output_dir, exist_ok=True)

# 输出文件名
output_mndwi = os.path.join(output_dir, "MNDWI.tif")
output_ndbi = os.path.join(output_dir, "NDBI.tif")
# 可选：如果需要 SAVI，取消下方注释
# output_savi = os.path.join(output_dir, "SAVI.tif")
# ================================================


def normalized_index(b1, b2, constant=0):
    """
    计算归一化指数 (b1 - b2) / (b1 + b2 + constant)，防止除零
    constant 用于 SAVI 等需要土壤调节因子的指数
    """
    denominator = b1 + b2 + constant
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (b1 - b2) / denominator
        result = np.where(np.isnan(result), 0, result)
        result = np.where(~np.isfinite(result), 0, result)
    return result


def compute_other_indices(tif_path, output_mndwi, output_ndbi):
    """
    读取 TIF 影像，计算 MNDWI 和 NDBI，并保存
    假设波段顺序（与原始代码一致）：
    Band1: Coastal/Blue (未使用)
    Band2: Green
    Band3: Red
    Band4: NIR
    Band5: SWIR1
    Band6: SWIR2 (未使用)
    """
    # 1. 打开影像
    ds = gdal.Open(tif_path)
    if ds is None:
        raise IOError(f"无法打开文件: {tif_path}")

    # 2. 获取地理信息和投影
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    # 3. 读取所需波段（索引从1开始）
    band_green = ds.GetRasterBand(2).ReadAsArray()   # Green
    band_red   = ds.GetRasterBand(3).ReadAsArray()   # Red (SAVI需要)
    band_nir   = ds.GetRasterBand(4).ReadAsArray()   # NIR
    band_swir1 = ds.GetRasterBand(5).ReadAsArray()   # SWIR1 (第5波段)

    # 4. 计算指数
    print("正在计算 MNDWI (Green - SWIR1)/(Green + SWIR1)...")
    mndwi = normalized_index(band_green, band_swir1)

    print("正在计算 NDBI (SWIR1 - NIR)/(SWIR1 + NIR)...")
    ndbi = normalized_index(band_swir1, band_nir)

    # 可选：计算 SAVI (土壤调节植被指数，L=0.5)
    # L = 0.5
    # savi = normalized_index(band_nir, band_red, constant=L) * (1 + L)

    # 5. 保存 GeoTIFF
    rows, cols = mndwi.shape
    driver = gdal.GetDriverByName('GTiff')

    # 保存 MNDWI
    out_ds_mndwi = driver.Create(output_mndwi, cols, rows, 1, gdal.GDT_Float32)
    out_ds_mndwi.SetGeoTransform(geotransform)
    out_ds_mndwi.SetProjection(projection)
    out_band = out_ds_mndwi.GetRasterBand(1)
    out_band.WriteArray(mndwi)
    out_band.SetNoDataValue(np.nan)
    out_band.ComputeStatistics(False)
    out_ds_mndwi = None

    # 保存 NDBI
    out_ds_ndbi = driver.Create(output_ndbi, cols, rows, 1, gdal.GDT_Float32)
    out_ds_ndbi.SetGeoTransform(geotransform)
    out_ds_ndbi.SetProjection(projection)
    out_band = out_ds_ndbi.GetRasterBand(1)
    out_band.WriteArray(ndbi)
    out_band.SetNoDataValue(np.nan)
    out_band.ComputeStatistics(False)
    out_ds_ndbi = None

    # 可选：保存 SAVI
    # out_ds_savi = driver.Create(output_savi, cols, rows, 1, gdal.GDT_Float32)
    # ... 类似操作

    ds = None
    print(f"MNDWI 已保存: {output_mndwi}")
    print(f"NDBI 已保存: {output_ndbi}")


def main():
    print("=" * 50)
    print("生成 MNDWI 和 NDBI 指数特征图")
    print("=" * 50)
    try:
        compute_other_indices(tif_file, output_mndwi, output_ndbi)
        print("\n任务完成！")
    except Exception as e:
        print(f"\n错误: {e}")
        raise


if __name__ == "__main__":
    main()