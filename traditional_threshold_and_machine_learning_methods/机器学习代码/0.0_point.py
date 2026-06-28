import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from shapely.geometry import Point
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 文件路径 ====================
image_2014_path = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\数据与模型文件\2014\scale_along_median_2014nepal2.tif"
image_2015_path = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\数据与模型文件\2015\scale_along_median_2015nepal2.tif"
shp_path = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\数据与模型文件\2015\shp\2015.shp"

# ==================== 参数设置（与第一个代码一致）====================
use_all_landslide_points = False  # True表示全部，False表示采样
num_landslide_points = 100        # 滑坡点数量
background_multiplier = 20        # 背景点是滑坡点的倍数

print("="*60)
print("开始数据验证...")
print("="*60)

# ==================== 1. 读取2015年影像 ====================
print("\n[1] 读取2015年影像...")
dataset_2015 = gdal.Open(image_2015_path)
if dataset_2015 is None:
    print("❌ 无法打开2015年影像！")
    exit()
else:
    cols_2015 = dataset_2015.RasterXSize
    rows_2015 = dataset_2015.RasterYSize
    bands_2015 = dataset_2015.RasterCount
    geotransform_2015 = dataset_2015.GetGeoTransform()
    print(f"✅ 2015年影像信息:")
    print(f"   - 尺寸: {rows_2015}行 × {cols_2015}列")
    print(f"   - 波段数: {bands_2015}")
    print(f"   - 地理变换: {geotransform_2015[:6]}")

# ==================== 2. 读取2014年影像 ====================
print("\n[2] 读取2014年影像...")
dataset_2014 = gdal.Open(image_2014_path)
if dataset_2014 is None:
    print("❌ 无法打开2014年影像！")
    exit()
else:
    cols_2014 = dataset_2014.RasterXSize
    rows_2014 = dataset_2014.RasterYSize
    bands_2014 = dataset_2014.RasterCount
    geotransform_2014 = dataset_2014.GetGeoTransform()
    print(f"✅ 2014年影像信息:")
    print(f"   - 尺寸: {rows_2014}行 × {cols_2014}列")
    print(f"   - 波段数: {bands_2014}")
    print(f"   - 地理变换: {geotransform_2014[:6]}")

# ==================== 3. 检查两期影像是否对齐 ====================
print("\n[3] 检查两期影像对齐情况...")
if rows_2014 == rows_2015 and cols_2014 == cols_2015:
    print("✅ 两期影像尺寸一致")
else:
    print("❌ 两期影像尺寸不一致！")
    print(f"   2014年: {rows_2014}×{cols_2014}")
    print(f"   2015年: {rows_2015}×{cols_2015}")
    exit()

if bands_2014 == bands_2015:
    print(f"✅ 两期影像波段数一致: {bands_2014}个波段")
else:
    print(f"⚠️ 两期影像波段数不同: 2014年{bands_2014}个, 2015年{bands_2015}个")

# 检查地理变换是否一致
if np.allclose(geotransform_2014, geotransform_2015, rtol=1e-5):
    print("✅ 两期影像地理变换一致")
else:
    print("⚠️ 两期影像地理变换不同，可能存在偏移")
    print(f"   2014年: {geotransform_2014[:6]}")
    print(f"   2015年: {geotransform_2015[:6]}")

# ==================== 4. 读取shapefile ====================
print("\n[4] 读取shapefile...")
try:
    shapefile = gpd.read_file(shp_path)
    print(f"✅ Shapefile读取成功")
    print(f"   - 总记录数: {len(shapefile)}")
    print(f"   - 几何类型: {shapefile.geometry.geom_type.iloc[0]}")
    print(f"   - CRS: {shapefile.crs}")
    
    # 检查是否有空几何
    null_geom = shapefile.geometry.isna().sum()
    if null_geom > 0:
        print(f"   ⚠️ 有{null_geom}个空几何")
        shapefile = shapefile.dropna(subset=['geometry'])
    
    # 检查几何类型
    geom_types = shapefile.geometry.geom_type.unique()
    print(f"   - 几何类型: {geom_types}")
    
    # 将多边形转换为质心点
    print("\n   将多边形转换为质心点...")
    shapefile = shapefile.copy()
    shapefile['geometry'] = shapefile['geometry'].centroid
    print(f"   ✅ 转换完成，几何类型: {shapefile.geometry.geom_type.iloc[0]}")
    
except Exception as e:
    print(f"❌ 读取shapefile失败: {e}")
    exit()

# ==================== 5. 采样滑坡点 ====================
print("\n[5] 滑坡点采样...")
total_landslide_points = len(shapefile)
print(f"   - shapefile中总点数: {total_landslide_points}")

if use_all_landslide_points or num_landslide_points >= total_landslide_points:
    sampled_points = shapefile.copy()
    print(f"✅ 使用全部滑坡点: {len(sampled_points)}个")
else:
    sampled_points = shapefile.sample(n=num_landslide_points, random_state=1)
    print(f"✅ 随机采样 {num_landslide_points} 个滑坡点 (从{total_landslide_points}个中)")

# ==================== 6. 计算像素坐标 ====================
print("\n[6] 计算像素坐标...")

# 获取地理变换参数
x_origin = geotransform_2015[0]
pixel_width = geotransform_2015[1]
y_origin = geotransform_2015[3]
pixel_height = geotransform_2015[5]  # 注意：通常是负值

def get_pixel_coords(geom, x_origin, pixel_width, y_origin, pixel_height):
    """根据地理坐标计算像素坐标"""
    if geom is None or geom.is_empty:
        return None, None
    
    x_geo = geom.x
    y_geo = geom.y
    
    # 根据地理变换计算像素坐标
    pixel_x = int((x_geo - x_origin) / pixel_width)
    pixel_y = int((y_geo - y_origin) / pixel_height)
    
    return pixel_x, pixel_y

# 添加像素坐标
sampled_points = sampled_points.copy()
pixel_coords = sampled_points.geometry.apply(
    lambda g: get_pixel_coords(g, x_origin, pixel_width, y_origin, pixel_height)
)
sampled_points['pixel_x'] = pixel_coords.apply(lambda x: x[0] if x is not None else None)
sampled_points['pixel_y'] = pixel_coords.apply(lambda x: x[1] if x is not None else None)

# 移除无法计算像素坐标的点
valid_mask = ~(sampled_points['pixel_x'].isna() | sampled_points['pixel_y'].isna())
sampled_points = sampled_points[valid_mask].copy()
print(f"✅ 有效滑坡点: {len(sampled_points)}个")

# ==================== 7. 检查采样点是否在影像范围内 ====================
print("\n[7] 检查滑坡点是否在影像范围内...")
in_range = ((sampled_points['pixel_x'] >= 0) & 
            (sampled_points['pixel_x'] < cols_2015) &
            (sampled_points['pixel_y'] >= 0) & 
            (sampled_points['pixel_y'] < rows_2015))

in_range_count = in_range.sum()
out_range_count = len(sampled_points) - in_range_count
print(f"   - 在影像范围内: {in_range_count} 个")
print(f"   - 超出影像范围: {out_range_count} 个")

if out_range_count > 0:
    print(f"⚠️ 有 {out_range_count} 个点不在影像范围内，将被移除")
    sampled_points = sampled_points[in_range].copy()
    print(f"   ✅ 剩余有效点: {len(sampled_points)}个")

# ==================== 8. 读取影像数据（仅读取小范围测试） ====================
print("\n[8] 读取影像数据（验证模式）...")

# 只读取几个点周围的数据进行验证
test_indices = range(min(5, len(sampled_points)))

try:
    # 读取2015年影像数据
    bands_data_2015 = []
    for i in range(bands_2015):
        band = dataset_2015.GetRasterBand(i + 1)
        band_data = band.ReadAsArray()
        bands_data_2015.append(band_data)
    bands_data_2015 = np.stack(bands_data_2015, axis=-1)
    print(f"✅ 2015年影像数据读取成功: {bands_data_2015.shape}")
    
    # 读取2014年影像数据
    bands_data_2014 = []
    for i in range(bands_2014):
        band = dataset_2014.GetRasterBand(i + 1)
        band_data = band.ReadAsArray()
        bands_data_2014.append(band_data)
    bands_data_2014 = np.stack(bands_data_2014, axis=-1)
    print(f"✅ 2014年影像数据读取成功: {bands_data_2014.shape}")
    
except Exception as e:
    print(f"❌ 读取影像数据失败: {e}")
    print("   可能影像太大，内存不足。建议使用块读取方式。")
    exit()

# ==================== 9. 验证部分采样点的数据 ====================
print("\n[9] 验证采样点数据...")
print("\n前5个采样点信息:")
for i in test_indices:
    row = sampled_points.iloc[i]
    x = int(row['pixel_x'])
    y = int(row['pixel_y'])
    
    print(f"\n点 {i+1}:")
    print(f"   - 像素坐标: ({x}, {y})")
    print(f"   - 地理位置: ({row.geometry.x:.6f}, {row.geometry.y:.6f})")
    
    # 检查2015年影像值
    try:
        values_2015 = bands_data_2015[y, x, :]
        print(f"   - 2015年波段值: {values_2015[:3]}... (共{len(values_2015)}个波段)")
        print(f"   - 2015年均值: {np.mean(values_2015):.4f}")
        print(f"   - 2015年标准差: {np.std(values_2015):.4f}")
    except Exception as e:
        print(f"   ❌ 2015年数据访问失败: {e}")
    
    # 检查2014年影像值
    try:
        values_2014 = bands_data_2014[y, x, :]
        print(f"   - 2014年波段值: {values_2014[:3]}... (共{len(values_2014)}个波段)")
        print(f"   - 2014年均值: {np.mean(values_2014):.4f}")
        print(f"   - 2014年标准差: {np.std(values_2014):.4f}")
    except Exception as e:
        print(f"   ❌ 2014年数据访问失败: {e}")

# ==================== 10. 生成非滑坡点采样方案 ====================
print("\n[10] 非滑坡点采样计划...")
num_background = len(sampled_points) * background_multiplier
print(f"   需要生成 {num_background} 个非滑坡点")
print(f"   (滑坡点{len(sampled_points)}个 × {background_multiplier}倍)")
print(f"   - 使用与第一个代码相同的随机生成方法")

# 检查是否有足够的非滑坡像素
total_pixels = rows_2015 * cols_2015
landslide_pixels = len(sampled_points)
available_pixels = total_pixels - landslide_pixels
print(f"   - 影像总像素: {total_pixels}")
print(f"   - 可用非滑坡像素: {available_pixels}")
if available_pixels >= num_background:
    print(f"   ✅ 有足够的非滑坡像素")
else:
    print(f"   ⚠️ 非滑坡像素不足！建议减少采样数量")

# ==================== 11. 输出验证总结 ====================
print("\n" + "="*60)
print("验证总结:")
print("="*60)
print(f"✅ 2014年影像: {rows_2014}×{cols_2014}×{bands_2014}")
print(f"✅ 2015年影像: {rows_2015}×{cols_2015}×{bands_2015}")
print(f"✅ 两期影像对齐: {'是' if rows_2014==rows_2015 and cols_2014==cols_2015 else '否'}")
print(f"✅ 有效滑坡点: {len(sampled_points)}个")
print(f"✅ 非滑坡点: {len(sampled_points)*background_multiplier}个")
print(f"✅ 总样本数: {len(sampled_points)*(background_multiplier+1)}个")
print("="*60)
print("\n🎉 所有验证通过！可以开始生成2015feature.csv")

# 释放内存
dataset_2014 = None
dataset_2015 = None
try:
    del bands_data_2014, bands_data_2015
except:
    pass

print("\n提示: 如需继续生成特征文件，请运行完整版代码。")