"""
随机森林滑坡预测（分块独立保存版）
功能：对双时相TIF影像进行滑坡预测，将每个分块的预测结果保存为独立GeoTIFF，支持断点续传。
"""
import numpy as np
import pandas as pd
from osgeo import gdal, osr
from skimage.feature import graycomatrix, graycoprops, hog
from joblib import load
import os

# ==================== 配置路径 ====================
tif_2015 = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\Nepal_Landsat\scale_along_median_2015nepal2.tif"
tif_2016 = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\Nepal_Landsat\scale_along_median_2016nepal2.tif"
model_path = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\训练结果\best_rf_model2.pkl"
output_dir = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\预测分块"  # 分块文件保存目录
block_size = 500  # 分块大小
# ================================================


def get_hog_feature_length(window_size=16):
    """计算 HOG 特征的实际长度"""
    dummy = np.zeros((window_size, window_size), dtype=np.float32)
    hog_feat = hog(dummy, orientations=8, pixels_per_cell=(8, 8),
                   cells_per_block=(1, 1), block_norm='L2-Hys',
                   feature_vector=True)
    return len(hog_feat)


HOG_FEATURE_LEN = get_hog_feature_length()


def normalized_index(b1, b2):
    """计算归一化指数，防止除零"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (b1 - b2) / (b1 + b2)
        result = np.where(np.isnan(result), 0, result)
    return result


def calculate_glcm_features(window, distances=[1], angles=[0]):
    """计算GLCM纹理特征"""
    if window.max() == window.min():
        return np.nan, np.nan, np.nan, np.nan, np.nan
    window = (255 * (window - window.min()) / (window.max() - window.min())).astype(np.uint8)
    glcm = graycomatrix(window, distances=distances, angles=angles, 
                        symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return contrast, dissimilarity, homogeneity, energy, correlation


def calculate_hog_features(window):
    """计算HOG特征"""
    hog_features = hog(window, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(1, 1), block_norm='L2-Hys',
                       feature_vector=True)
    return hog_features


def extract_features_for_pixel(bands_year, bands_last_year, y, x, 
                                rows, cols, window_hog=16, window_glcm=3):
    """
    为单个像素提取特征（与训练时一致）
    """
    half_hog = window_hog // 2
    half_glcm = window_glcm // 2
    
    # 边界检查
    boundary_ok = (y - half_hog >= 0 and y + half_hog < rows and
                   x - half_hog >= 0 and x + half_hog < cols)
    
    # ===== 2016年特征 =====
    spectrum_year = bands_year[y, x, :]
    mean_year = np.mean(spectrum_year)
    std_year = np.std(spectrum_year)
    ndwi_year = normalized_index(spectrum_year[1], spectrum_year[3])
    ndvi_year = normalized_index(spectrum_year[3], spectrum_year[2])
    
    if boundary_ok:
        # GLCM特征
        window_glcm_year = bands_year[y-half_glcm:y+half_glcm+1, 
                                       x-half_glcm:x+half_glcm+1, 0]
        glcm_year = calculate_glcm_features(window_glcm_year)
        
        # HOG特征
        window_hog_year = bands_year[y-half_hog:y+half_hog+1, 
                                      x-half_hog:x+half_hog+1, 0]
        hog_year = calculate_hog_features(window_hog_year)
    else:
        glcm_year = (np.nan,) * 5
        hog_year = np.array([np.nan] * HOG_FEATURE_LEN)
    
    # ===== 2015年特征 =====
    spectrum_last = bands_last_year[y, x, :]
    mean_last = np.mean(spectrum_last)
    std_last = np.std(spectrum_last)
    ndwi_last = normalized_index(spectrum_last[1], spectrum_last[3])
    ndvi_last = normalized_index(spectrum_last[3], spectrum_last[2])
    
    if boundary_ok:
        window_glcm_last = bands_last_year[y-half_glcm:y+half_glcm+1,
                                           x-half_glcm:x+half_glcm+1, 0]
        glcm_last = calculate_glcm_features(window_glcm_last)
        
        window_hog_last = bands_last_year[y-half_hog:y+half_hog+1,
                                          x-half_hog:x+half_hog+1, 0]
        hog_last = calculate_hog_features(window_hog_last)
    else:
        glcm_last = (np.nan,) * 5
        hog_last = np.array([np.nan] * HOG_FEATURE_LEN)
    
    # ===== 合并所有特征（顺序必须与训练时一致）=====
    features = (list(spectrum_year) + list(spectrum_last) +
                [mean_year, std_year, ndwi_year, ndvi_year] +
                list(glcm_year) + list(hog_year) +
                [mean_last, std_last, ndwi_last, ndvi_last] +
                list(glcm_last) + list(hog_last))
    
    return np.array(features)


def extract_features_block(bands_year, bands_last_year, 
                           row_start, row_end, col_start, col_end):
    """
    分块提取特征（逐像素循环）
    """
    features_list = []
    total_pixels = (row_end - row_start) * (col_end - col_start)
    processed = 0
    
    for y in range(row_start, row_end):
        for x in range(col_start, col_end):
            features = extract_features_for_pixel(
                bands_year, bands_last_year, y, x,
                bands_year.shape[0], bands_year.shape[1]
            )
            features_list.append(features)
            processed += 1
        
        if (y - row_start + 1) % 100 == 0:
            print(f"  进度: {processed}/{total_pixels} 像素")
    
    return np.array(features_list)


def predict_block(model, features):
    """对特征进行预测，处理NaN"""
    col_means = np.nanmean(features, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    features_clean = np.where(np.isnan(features), col_means, features)
    prediction = model.predict(features_clean)
    return prediction


def save_block_geotiff(output_path, data, geotransform, projection, 
                       x_offset, y_offset, block_width, block_height):
    """
    保存分块为GeoTIFF（仅包含块内数据，但GeoTransform基于原始影像全局坐标）
    data: 2D数组，shape (block_height, block_width)
    geotransform: 原始影像的GeoTransform
    x_offset, y_offset: 块左上角在原始影像中的像素坐标（列、行）
    block_width, block_height: 块的实际宽度和高度
    """
    # 计算块左上角的地理坐标
    origin_x = geotransform[0] + x_offset * geotransform[1]
    origin_y = geotransform[3] + y_offset * geotransform[5]  # geotransform[5]为负值
    
    block_geotransform = (origin_x, geotransform[1], geotransform[2],
                          origin_y, geotransform[4], geotransform[5])
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, block_width, block_height, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(block_geotransform)
    out_ds.SetProjection(projection)
    
    band = out_ds.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(0)
    band.ComputeStatistics(False)
    out_ds = None


def main():
    print("=" * 50)
    print("随机森林滑坡预测（分块独立保存版）")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载模型
    print("\n[1/5] 加载模型...")
    model = load(model_path)
    print(f"  模型加载成功: {model_path}")
    print(f"  模型期望特征数: {model.n_features_in_}")
    
    # 2. 读取TIF文件
    print("\n[2/5] 读取TIF影像...")
    ds_2015 = gdal.Open(tif_2015)
    ds_2016 = gdal.Open(tif_2016)
    
    bands_2015 = np.stack([ds_2015.GetRasterBand(i+1).ReadAsArray() 
                          for i in range(ds_2015.RasterCount)], axis=-1)
    bands_2016 = np.stack([ds_2016.GetRasterBand(i+1).ReadAsArray() 
                          for i in range(ds_2016.RasterCount)], axis=-1)
    
    rows, cols, num_bands = bands_2016.shape
    print(f"  影像尺寸: {rows} x {cols}, 波段数: {num_bands}")
    
    # 获取地理信息
    geotransform = ds_2016.GetGeoTransform()
    projection = ds_2016.GetProjection()
    
    # 3. 分块预测并保存
    print("\n[3/5] 分块预测...")
    block_count = 0
    total_blocks = ((rows + block_size - 1) // block_size) * ((cols + block_size - 1) // block_size)
    
    for row_start in range(0, rows, block_size):
        row_end = min(row_start + block_size, rows)
        block_height = row_end - row_start
        for col_start in range(0, cols, block_size):
            col_end = min(col_start + block_size, cols)
            block_width = col_end - col_start
            block_count += 1
            
            # 生成块文件名
            block_filename = f"block_{row_start}_{col_start}.tif"
            block_path = os.path.join(output_dir, block_filename)
            
            # 断点续传：如果文件已存在，跳过该块
            if os.path.exists(block_path):
                print(f"\n  块 {block_count}/{total_blocks}: 行{row_start}-{row_end}, 列{col_start}-{col_end} -> 已存在，跳过")
                continue
            
            print(f"\n  处理块 {block_count}/{total_blocks}: 行{row_start}-{row_end}, 列{col_start}-{col_end}")
            
            # 提取该块的特征
            features = extract_features_block(
                bands_2016, bands_2015,
                row_start, row_end, col_start, col_end
            )
            
            # 预测
            preds = predict_block(model, features)
            
            # 将预测结果重塑为二维数组
            block_pred = preds.reshape((block_height, block_width))
            
            # 保存为GeoTIFF
            save_block_geotiff(block_path, block_pred, geotransform, projection,
                               col_start, row_start, block_width, block_height)
            print(f"  保存至: {block_path}")
    
    print("\n" + "=" * 50)
    print("分块预测完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()