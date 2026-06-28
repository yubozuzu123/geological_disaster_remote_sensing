import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import os

def extract_landslide_candidates_multi_index(
    index_dir,                          # 指数文件所在文件夹
    output_path,
    # NDVI 阈值
    ndvi_lower=0.05, ndvi_upper=0.2,   # NDVI 区间 [lower, upper)
    # NDWI 阈值
    ndwi_upper=0.0,                    # NDWI < upper
    # MNDWI 阈值
    mndwi_upper=0.0,                   # MNDWI < upper
    # NDBI 阈值
    ndbi_upper=0.0,                    # NDBI < upper
    # 坡度约束（可选）
    slope_path=None, slope_threshold=15.0,
    mask_nodata=True
):
    """
    基于多个指数阈值提取滑坡候选区（二值图）
    
    指数文件命名约定（与 NDVI/NDWI 同一目录）：
        - NDVI.tif
        - NDWI.tif
        - MNDWI.tif   （需要提前计算）
        - NDBI.tif     （需要提前计算）
    
    参数：
        index_dir : str       存放所有指数 GeoTIFF 的文件夹路径
        output_path : str     输出二值栅格路径
        ndvi_lower, ndvi_upper : float  NDVI 有效区间（排除阴影/高植被）
        ndwi_upper : float                NDWI 上界（排除水体/湿润区）
        mndwi_upper : float               MNDWI 上界（进一步抑制阴影）
        ndbi_upper : float                NDBI 上界（排除建筑/道路）
        slope_path : str                 坡度文件路径（可选）
        slope_threshold : float          坡度阈值（大于该值视为候选）
        mask_nodata : bool               是否忽略 NoData
    """
    
    # 构建指数文件完整路径
    ndvi_path = os.path.join(index_dir, "NDVI.tif")
    ndwi_path = os.path.join(index_dir, "NDWI.tif")
    mndwi_path = os.path.join(index_dir, "MNDWI.tif")
    ndbi_path = os.path.join(index_dir, "NDBI.tif")
    
    # 检查文件是否存在
    for p in [ndvi_path, ndwi_path, mndwi_path, ndbi_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少指数文件: {p}")
    
    # 1. 读取 NDVI
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        ndvi_nodata = src.nodata
        valid_mask = (ndvi != ndvi_nodata) if ndvi_nodata is not None else np.isfinite(ndvi)
    
    # 2. 读取 NDWI
    with rasterio.open(ndwi_path) as src:
        ndwi = src.read(1).astype(np.float32)
        nodata = src.nodata
        valid = (ndwi != nodata) if nodata is not None else np.isfinite(ndwi)
        valid_mask &= valid
        if src.transform != transform or src.crs != crs:
            raise ValueError("NDWI 几何不一致")
    
    # 3. 读取 MNDWI
    with rasterio.open(mndwi_path) as src:
        mndwi = src.read(1).astype(np.float32)
        nodata = src.nodata
        valid = (mndwi != nodata) if nodata is not None else np.isfinite(mndwi)
        valid_mask &= valid
        if src.transform != transform or src.crs != crs:
            raise ValueError("MNDWI 几何不一致")
    
    # 4. 读取 NDBI
    with rasterio.open(ndbi_path) as src:
        ndbi = src.read(1).astype(np.float32)
        nodata = src.nodata
        valid = (ndbi != nodata) if nodata is not None else np.isfinite(ndbi)
        valid_mask &= valid
        if src.transform != transform or src.crs != crs:
            raise ValueError("NDBI 几何不一致")
    
    # 5. 组合阈值条件
    condition = (
        # (ndvi >= ndvi_lower) & 
        (ndvi < ndvi_upper) &
        (ndwi < ndwi_upper) &
        (mndwi < mndwi_upper) &
        (ndbi < ndbi_upper)
    )
    
    landslide_mask = np.zeros(ndvi.shape, dtype=np.uint8)
    landslide_mask[condition & valid_mask] = 1
    
    # 6. 坡度约束（可选）
    if slope_path is not None:
        with rasterio.open(slope_path) as src_slope:
            if src_slope.transform != transform or src_slope.crs != crs:
                slope_reproj = np.zeros(ndvi.shape, dtype=np.float32)
                reproject(
                    source=rasterio.band(src_slope, 1),
                    destination=slope_reproj,
                    src_transform=src_slope.transform,
                    src_crs=src_slope.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear
                )
                slope = slope_reproj
            else:
                slope = src_slope.read(1).astype(np.float32)
            
            slope_nodata = src_slope.nodata
            if slope_nodata is not None:
                valid_mask &= (slope != slope_nodata)
            else:
                valid_mask &= np.isfinite(slope)
            
            slope_condition = (slope > slope_threshold)
            landslide_mask = np.where(slope_condition & ( landslide_mask == 1), 1, 0)
    
    # 7. 无效区域置0
    landslide_mask[~valid_mask] = 0
    
    # 8. 保存结果
    meta.update({
        'dtype': 'uint8',
        'nodata': 0,
        'count': 1,
        'compress': 'lzw'
    })
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(landslide_mask, 1)
    
    print(f"多指数阈值提取完成，结果保存至：{output_path}")
    print(f"候选区像元数：{np.sum(landslide_mask == 1)}")
    print(f"阈值设置：NDVI ∈ [{ndvi_lower}, {ndvi_upper}), NDWI < {ndwi_upper}, MNDWI < {mndwi_upper}, NDBI < {ndbi_upper}")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 指数文件所在文件夹（与 NDVI.tif 同一目录）
    index_folder = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\指数特征图+经验阈值"
    output_file = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\指数特征图+经验阈值\自动寻找阈值\landslide_candidates_multi_index.tif"
    
    # 坡度文件（可选，若无则注释掉 slope_path）
    # slope_file = r"路径\坡度.tif"
    
    extract_landslide_candidates_multi_index(
        index_dir=index_folder,
        output_path=output_file,
        ndvi_lower=0.05,      # 排除阴影（NDVI < 0.05）
        ndvi_upper=0.086,       # 排除高植被
        ndwi_upper=0.064,       # 排除水体/湿润区
        mndwi_upper=-0.070,      # 进一步抑制阴影
        ndbi_upper=0.177,       # 排除建筑/道路
        # slope_path=slope_file,
        # slope_threshold=15.0,
        mask_nodata=True
    )