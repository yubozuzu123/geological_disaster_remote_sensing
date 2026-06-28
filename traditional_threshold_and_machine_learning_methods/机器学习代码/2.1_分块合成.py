"""
分块结果合成脚本
功能：读取指定目录下所有分块GeoTIFF，拼接成完整的滑坡预测结果，并保存为最终GeoTIFF。
"""
import numpy as np
from osgeo import gdal, osr
import os
import re

# ==================== 配置路径 ====================
blocks_dir = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\预测分块"  # 分块文件目录
reference_tif = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\Nepal_Landsat\scale_along_median_2016nepal2.tif"  # 参考影像（获取尺寸和投影）
output_tif = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\prediction_result.tif"  # 最终输出路径
# ================================================


def parse_block_filename(filename):
    """解析分块文件名，返回 (row_start, col_start)"""
    # 文件名格式：block_{row_start}_{col_start}.tif
    match = re.match(r"block_(\d+)_(\d+)\.tif", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        raise ValueError(f"文件名格式不正确: {filename}")


def main():
    print("=" * 50)
    print("分块结果合成")
    print("=" * 50)
    
    # 1. 读取参考影像，获取尺寸、地理信息
    print("\n[1/4] 读取参考影像...")
    ref_ds = gdal.Open(reference_tif)
    rows = ref_ds.RasterYSize
    cols = ref_ds.RasterXSize
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    print(f"  参考影像尺寸: {rows} x {cols}")
    
    # 2. 创建结果数组
    print("\n[2/4] 创建结果数组...")
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    # 3. 遍历所有分块文件
    print("\n[3/4] 遍历分块文件...")
    block_files = [f for f in os.listdir(blocks_dir) if f.endswith('.tif') and f.startswith('block_')]
    print(f"  找到 {len(block_files)} 个分块文件")
    
    for filename in block_files:
        block_path = os.path.join(blocks_dir, filename)
        row_start, col_start = parse_block_filename(filename)
        
        # 读取分块数据
        ds = gdal.Open(block_path)
        block_data = ds.GetRasterBand(1).ReadAsArray()
        block_height, block_width = block_data.shape
        ds = None
        
        # 计算放置位置
        row_end = row_start + block_height
        col_end = col_start + block_width
        
        # 边界检查（防止越界）
        if row_end > rows or col_end > cols:
            print(f"  警告: 块 {filename} 超出参考影像范围，将裁剪")
            block_data = block_data[:rows-row_start, :cols-col_start]
            row_end = min(row_end, rows)
            col_end = min(col_end, cols)
        
        # 填入结果数组
        result[row_start:row_end, col_start:col_end] = block_data
        print(f"  已合成: {filename} -> 位置 ({row_start}, {col_start})")
    
    # 4. 保存最终结果
    print("\n[4/4] 保存最终结果...")
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, cols, rows, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    band = out_ds.GetRasterBand(1)
    band.WriteArray(result)
    band.SetNoDataValue(0)
    band.ComputeStatistics(False)
    out_ds = None
    
    # 统计结果
    landslide_count = np.sum(result == 1)
    total = rows * cols
    print(f"\n  最终结果: 滑坡像素 {landslide_count} ({landslide_count/total*100:.2f}%)")
    print(f"  保存至: {output_tif}")
    
    print("\n" + "=" * 50)
    print("合成完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()