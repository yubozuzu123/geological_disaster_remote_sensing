import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from skimage.feature import graycomatrix, graycoprops, hog
from shapely.geometry import Point
import random
import warnings
warnings.filterwarnings('ignore')

# ==================== 文件路径 ====================
image_2014_path = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\数据与模型文件\2014\scale_along_median_2014nepal2.tif"
image_2015_path = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\数据与模型文件\2015\scale_along_median_2015nepal2.tif"
shp_path = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\数据与模型文件\2015\shp\2015.shp"
output_csv = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\机器学习部分\2015feature.csv"

# ==================== 参数设置 ====================
use_all_landslide_points = False
num_landslide_points = 100
background_multiplier = 20
random_state = 42

print("="*60)
print("开始提取特征...")
print("="*60)

# ==================== 1. 读取影像 ====================
print("\n[1] 读取影像数据...")

# 读取2015年影像
dataset_2015 = gdal.Open(image_2015_path)
cols = dataset_2015.RasterXSize
rows = dataset_2015.RasterYSize
bands = dataset_2015.RasterCount
geotransform = dataset_2015.GetGeoTransform()

# 读取所有波段
bands_data_2015 = []
for i in range(bands):
    band = dataset_2015.GetRasterBand(i + 1)
    bands_data_2015.append(band.ReadAsArray())
bands_data_2015 = np.stack(bands_data_2015, axis=-1)
print(f"✅ 2015年影像: {bands_data_2015.shape}")

# 读取2014年影像
dataset_2014 = gdal.Open(image_2014_path)
bands_data_2014 = []
for i in range(bands):
    band = dataset_2014.GetRasterBand(i + 1)
    bands_data_2014.append(band.ReadAsArray())
bands_data_2014 = np.stack(bands_data_2014, axis=-1)
print(f"✅ 2014年影像: {bands_data_2014.shape}")

# ==================== 2. 读取并采样滑坡点 ====================
print("\n[2] 采样滑坡点...")

# 读取shapefile
shapefile = gpd.read_file(shp_path)
shapefile = shapefile.dropna(subset=['geometry'])

# 多边形转质心
shapefile = shapefile.copy()
shapefile['geometry'] = shapefile['geometry'].centroid

# 采样
total_points = len(shapefile)
if use_all_landslide_points or num_landslide_points >= total_points:
    sampled_landslide = shapefile.copy()
else:
    sampled_landslide = shapefile.sample(n=num_landslide_points, random_state=1)

# 计算像素坐标
x_origin = geotransform[0]
pixel_width = geotransform[1]
y_origin = geotransform[3]
pixel_height = geotransform[5]

def get_pixel_coords(geom):
    if geom is None or geom.is_empty:
        return None, None
    pixel_x = int((geom.x - x_origin) / pixel_width)
    pixel_y = int((geom.y - y_origin) / pixel_height)
    return pixel_x, pixel_y

sampled_landslide = sampled_landslide.copy()
pixel_coords = sampled_landslide.geometry.apply(get_pixel_coords)
sampled_landslide['pixel_x'] = pixel_coords.apply(lambda x: x[0] if x is not None else None)
sampled_landslide['pixel_y'] = pixel_coords.apply(lambda x: x[1] if x is not None else None)

# 移除无效点
valid_mask = ~(sampled_landslide['pixel_x'].isna() | sampled_landslide['pixel_y'].isna())
sampled_landslide = sampled_landslide[valid_mask].copy()

# 检查是否在影像范围内
in_range = ((sampled_landslide['pixel_x'] >= 0) & 
            (sampled_landslide['pixel_x'] < cols) &
            (sampled_landslide['pixel_y'] >= 0) & 
            (sampled_landslide['pixel_y'] < rows))
sampled_landslide = sampled_landslide[in_range].copy()

print(f"✅ 滑坡点: {len(sampled_landslide)}个")

# ==================== 3. 生成非滑坡点 ====================
print("\n[3] 生成非滑坡点...")

def pixel2coord(x, y, geotransform):
    """将像素坐标转换为地理坐标"""
    px = geotransform[0] + x * geotransform[1] + y * geotransform[2]
    py = geotransform[3] + x * geotransform[4] + y * geotransform[5]
    return px, py

def generate_background_points(num_points, width, height, landslide_points, geotransform):
    """生成非滑坡点"""
    points = []
    landslide_coords = set(zip(landslide_points['pixel_x'], landslide_points['pixel_y']))
    
    # 获取所有滑坡点的几何用于判断
    landslide_geoms = landslide_points.geometry.tolist()
    
    max_attempts = num_points * 10
    attempts = 0
    
    while len(points) < num_points and attempts < max_attempts:
        attempts += 1
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        
        # 检查是否与滑坡点重叠
        if (x, y) in landslide_coords:
            continue
        
        # 转换为地理坐标
        px, py = pixel2coord(x, y, geotransform)
        point = Point(px, py)
        
        # 检查是否在滑坡多边形内
        is_landslide = False
        for geom in landslide_geoms:
            if geom.contains(point):
                is_landslide = True
                break
        
        if not is_landslide:
            points.append((x, y, point))
    
    return points

# 生成非滑坡点
num_background = len(sampled_landslide) * background_multiplier
background_points = generate_background_points(
    num_background, cols, rows, sampled_landslide, geotransform
)

# 创建GeoDataFrame
background_gdf = gpd.GeoDataFrame({
    'geometry': [pt[2] for pt in background_points],
    'pixel_x': [pt[0] for pt in background_points],
    'pixel_y': [pt[1] for pt in background_points]
}, crs=shapefile.crs)

print(f"✅ 非滑坡点: {len(background_gdf)}个")

# ==================== 4. 特征提取函数 ====================
print("\n[4] 定义特征提取函数...")

def calculate_ndvi(nir_band, red_band):
    """计算NDVI"""
    return (nir_band - red_band) / (nir_band + red_band + 1e-10)

def calculate_ndwi(green_band, nir_band):
    """计算NDWI"""
    return (green_band - nir_band) / (green_band + nir_band + 1e-10)

def calculate_glcm_features(window):
    """计算GLCM纹理特征"""
    if window.shape[0] < 2 or window.shape[1] < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    if window.max() == window.min():
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # 归一化到0-255
    window = (255 * (window - window.min()) / (window.max() - window.min())).astype(np.uint8)
    
    try:
        glcm = graycomatrix(window, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        return contrast, dissimilarity, homogeneity, energy, correlation
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan

def calculate_hog_features(window):
    """计算HOG特征"""
    if window.shape[0] < 16 or window.shape[1] < 16:
        return np.full(32, np.nan)
    
    try:
        hog_features = hog(window, orientations=8, pixels_per_cell=(8, 8),
                          cells_per_block=(1, 1), block_norm='L2-Hys',
                          feature_vector=True)
        return hog_features
    except:
        return np.full(32, np.nan)

def extract_features_for_point(x, y, bands_2015, bands_2014):
    """提取单个点的所有特征"""
    
    # 光谱特征
    spectrum_2015 = bands_2015[y, x, :]
    spectrum_2014 = bands_2014[y, x, :]
    
    # 统计特征
    mean_2015 = np.mean(spectrum_2015)
    std_2015 = np.std(spectrum_2015)
    mean_2014 = np.mean(spectrum_2014)
    std_2014 = np.std(spectrum_2014)
    
    # 植被指数
    ndvi_2015 = calculate_ndvi(spectrum_2015[4], spectrum_2015[3])
    ndwi_2015 = calculate_ndwi(spectrum_2015[2], spectrum_2015[4])
    ndvi_2014 = calculate_ndvi(spectrum_2014[4], spectrum_2014[3])
    ndwi_2014 = calculate_ndwi(spectrum_2014[2], spectrum_2014[4])
    
    # GLCM纹理特征 (3x3窗口)
    window_size_glcm = 3
    half_glcm = window_size_glcm // 2
    
    if (y - half_glcm >= 0 and y + half_glcm < rows and
        x - half_glcm >= 0 and x + half_glcm < cols):
        window_2015 = bands_2015[y - half_glcm:y + half_glcm + 1,
                                 x - half_glcm:x + half_glcm + 1, 0]
        window_2014 = bands_2014[y - half_glcm:y + half_glcm + 1,
                                 x - half_glcm:x + half_glcm + 1, 0]
        glcm_2015 = calculate_glcm_features(window_2015)
        glcm_2014 = calculate_glcm_features(window_2014)
    else:
        glcm_2015 = (np.nan, np.nan, np.nan, np.nan, np.nan)
        glcm_2014 = (np.nan, np.nan, np.nan, np.nan, np.nan)
    
    # HOG特征 (16x16窗口)
    window_size_hog = 16
    half_hog = window_size_hog // 2
    
    if (y - half_hog >= 0 and y + half_hog < rows and
        x - half_hog >= 0 and x + half_hog < cols):
        window_hog_2015 = bands_2015[y - half_hog:y + half_hog + 1,
                                     x - half_hog:x + half_hog + 1, 0]
        window_hog_2014 = bands_2014[y - half_hog:y + half_hog + 1,
                                     x - half_hog:x + half_hog + 1, 0]
        hog_2015 = calculate_hog_features(window_hog_2015)
        hog_2014 = calculate_hog_features(window_hog_2014)
    else:
        hog_2015 = np.full(32, np.nan)
        hog_2014 = np.full(32, np.nan)
    
    # 拼接所有特征
    features = (list(spectrum_2015) +           # 6个波段
                list(spectrum_2014) +           # 6个波段
                [mean_2015, std_2015, ndwi_2015, ndvi_2015] +  # 2015统计+指数
                [mean_2014, std_2014, ndwi_2014, ndvi_2014] +  # 2014统计+指数
                list(glcm_2015) +               # 5个纹理特征
                list(glcm_2014) +               # 5个纹理特征
                list(hog_2015) +                # 32个HOG特征
                list(hog_2014))                 # 32个HOG特征
    
    return features

print("✅ 特征提取函数定义完成")

# ==================== 5. 提取所有样本的特征 ====================
print("\n[5] 开始提取特征...")

# 准备样本数据 - 使用字符串标签
landslide_samples = sampled_landslide[['pixel_x', 'pixel_y']].copy()
landslide_samples['label'] = 'landslide'

background_samples = background_gdf[['pixel_x', 'pixel_y']].copy()
background_samples['label'] = 'non-landslide'

all_samples = pd.concat([landslide_samples, background_samples], ignore_index=True)
print(f"总样本数: {len(all_samples)}个")

# 提取特征
results = []
total_samples = len(all_samples)

for idx, row in all_samples.iterrows():
    x = int(row['pixel_x'])
    y = int(row['pixel_y'])
    label = row['label']
    
    if idx % 100 == 0:
        print(f"   进度: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
    
    try:
        features = extract_features_for_point(x, y, bands_data_2015, bands_data_2014)
        results.append(features + [label])
    except Exception as e:
        print(f"   ⚠️ 点({x},{y})提取失败: {e}")
        nan_features = [np.nan] * (6 + 6 + 4 + 4 + 5 + 5 + 32 + 32)
        results.append(nan_features + [label])

print(f"✅ 特征提取完成，共{len(results)}个样本")

# ==================== 6. 保存结果 ====================
print("\n[6] 保存结果...")

# 构建特征名称
feature_names = (
    ['Band_0', 'Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5'] +
    ['Band_0_last_year', 'Band_1_last_year', 'Band_2_last_year', 
     'Band_3_last_year', 'Band_4_last_year', 'Band_5_last_year'] +
    ['Mean_spectrum', 'Std_spectrum', 'NDWI', 'NDVI'] +
    ['Mean_spectrum_last_year', 'Std_spectrum_last_year', 'NDWI_last_year', 'NDVI_last_year'] +
    ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'] +
    ['Contrast_last_year', 'Dissimilarity_last_year', 
     'Homogeneity_last_year', 'Energy_last_year', 'Correlation_last_year'] +
    [f'HOG_feature_{i}' for i in range(32)] +
    [f'HOG_feature_last_year_{i}' for i in range(32)] +
    ['Label']
)

# 创建DataFrame
df_results = pd.DataFrame(results, columns=feature_names)

# ==================== 7. 删除包含空值的行 ====================
print("\n[7] 删除包含空值的行...")
original_count = len(df_results)
print(f"   删除前样本数: {original_count}")

df_results_clean = df_results.dropna()
clean_count = len(df_results_clean)
removed_count = original_count - clean_count
print(f"   删除后样本数: {clean_count}")
print(f"   删除了 {removed_count} 行 ({removed_count/original_count*100:.1f}%)")

# 检查标签分布
print(f"\n   滑坡样本: {df_results_clean[df_results_clean['Label']=='landslide'].shape[0]}个")
print(f"   非滑坡样本: {df_results_clean[df_results_clean['Label']=='non-landslide'].shape[0]}个")

# ==================== 8. 保存清洗后的数据 ====================
print(f"\n[8] 保存清洗后的数据到: {output_csv}")
df_results_clean.to_csv(output_csv, index=False)
print(f"✅ 结果已保存")
print(f"   文件大小: {len(df_results_clean)}行 × {len(df_results_clean.columns)}列")

# ==================== 9. 数据统计总结 ====================
print("\n" + "="*60)
print("数据统计总结:")
print("="*60)
print(f"原始样本数: {original_count}")
print(f"清洗后样本数: {clean_count}")
print(f"删除样本数: {removed_count} ({removed_count/original_count*100:.1f}%)")
print(f"特征数量: {len(df_results_clean.columns)-1}个")
print(f"滑坡:非滑坡 = {df_results_clean[df_results_clean['Label']==1].shape[0]}:{df_results_clean[df_results_clean['Label']==0].shape[0]}")

print("\n前5行数据:")
print(df_results_clean.head())

print("\n" + "="*60)
print("🎉 特征提取完成！")
print("="*60)