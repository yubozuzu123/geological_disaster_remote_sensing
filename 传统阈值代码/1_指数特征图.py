"""
生成 NDVI 和 NDWI 指数特征图
功能：读取单一时相或多时相 Landsat TIF 影像，计算 NDVI 和 NDWI，保存为 GeoTIFF
"""

import numpy as np
from osgeo import gdal, osr
import os

# ==================== 配置路径 ====================
# 输入 TIF 文件（可以是单一时相，如 2015 年或 2016 年）
tif_file = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\Nepal_Landsat\scale_along_median_2015nepal2.tif"

# 输出目录（自动创建）
output_dir = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\指数特征图+经验阈值_2015"
os.makedirs(output_dir, exist_ok=True)

# 输出文件名（可根据需要修改）
output_ndvi = os.path.join(output_dir, "NDVI.tif")
output_ndwi = os.path.join(output_dir, "NDWI.tif")
# ================================================


def normalized_index(b1, b2):
    """计算归一化指数，防止除零"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (b1 - b2) / (b1 + b2)
        result = np.where(np.isnan(result), 0, result)
    return result


def compute_indices(tif_path, output_ndvi, output_ndwi):
    """
    读取 TIF 影像，计算 NDVI 和 NDWI，并保存
    """
    # 1. 打开影像
    ds = gdal.Open(tif_path)
    if ds is None:
        raise IOError(f"无法打开文件: {tif_path}")
    
    # 2. 获取地理信息和投影
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    
    # 3. 读取波段（假设顺序：B1, B2, B3, B4, B5, B6）
    # 索引对应：B1=0, B2=1, B3=2, B4=3, B5=4, B6=5
    band2 = ds.GetRasterBand(2).ReadAsArray()  # Green
    band3 = ds.GetRasterBand(3).ReadAsArray()  # Red
    band4 = ds.GetRasterBand(4).ReadAsArray()  # NIR
    
    # 4. 计算 NDVI 和 NDWI
    print("正在计算 NDVI...")
    ndvi = normalized_index(band4, band3)
    
    print("正在计算 NDWI...")
    ndwi = normalized_index(band2, band4)
    
    # 5. 保存为 GeoTIFF
    rows, cols = ndvi.shape
    
    # 保存 NDVI
    driver = gdal.GetDriverByName('GTiff')
    out_ds_ndvi = driver.Create(output_ndvi, cols, rows, 1, gdal.GDT_Float32)
    out_ds_ndvi.SetGeoTransform(geotransform)
    out_ds_ndvi.SetProjection(projection)
    out_band_ndvi = out_ds_ndvi.GetRasterBand(1)
    out_band_ndvi.WriteArray(ndvi)
    out_band_ndvi.SetNoDataValue(np.nan)
    out_band_ndvi.ComputeStatistics(False)
    out_ds_ndvi = None
    
    # 保存 NDWI
    out_ds_ndwi = driver.Create(output_ndwi, cols, rows, 1, gdal.GDT_Float32)
    out_ds_ndwi.SetGeoTransform(geotransform)
    out_ds_ndwi.SetProjection(projection)
    out_band_ndwi = out_ds_ndwi.GetRasterBand(1)
    out_band_ndwi.WriteArray(ndwi)
    out_band_ndwi.SetNoDataValue(np.nan)
    out_band_ndwi.ComputeStatistics(False)
    out_ds_ndwi = None
    
    # 关闭原始影像
    ds = None
    
    print(f"NDVI 已保存: {output_ndvi}")
    print(f"NDWI 已保存: {output_ndwi}")


def main():
    print("=" * 50)
    print("生成 NDVI 和 NDWI 指数特征图")
    print("=" * 50)
    
    try:
        compute_indices(tif_file, output_ndvi, output_ndwi)
        print("\n任务完成！")
    except Exception as e:
        print(f"\n错误: {e}")
        raise


if __name__ == "__main__":
    main()