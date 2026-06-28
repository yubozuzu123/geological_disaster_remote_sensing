import rasterio
from rasterio.features import shapes
import geopandas as gpd
from pathlib import Path
from shapely.geometry import shape

def to_tif_geo(dir, out_path, min_area=3600):
    """
    将TIF栅格转换为矢量，并筛选指定最小面积以上的要素
    
    Parameters:
    dir: 输入TIF文件路径
    out_path: 输出矢量文件路径
    min_area: 最小面积阈值（平方米），默认5000
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 1. 打开刚才生成的"编号"栅格
    with rasterio.open(dir) as src:
        image = src.read(1)          # int16 数组，0=背景，1,2,3…=滑坡ID
        mask = image != 0            # 只转非 0 区域
        transform = src.transform
        crs = src.crs

    # 2. 栅格→矢量（返回生成器）
    results = []
    for geom, value in shapes(image, mask=mask, transform=transform):
        if value != 0 and value != 255:  # 过滤背景
            # 将GeoJSON字典转换为Shapely几何对象
            shapely_geom = shape(geom)
            results.append((shapely_geom, value))

    # 3. 装进 GeoDataFrame
    if results:  # 确保有结果
        gdf = gpd.GeoDataFrame(
            results,
            columns=['geometry', 'Id'],
            geometry='geometry',
            crs=crs)

        # 4. 计算面积
        try:
            utm_crs = gdf.estimate_utm_crs()
            gdf = gdf.to_crs(utm_crs)  # 转换到UTM坐标系
            gdf['area_m2'] = gdf.area   # 计算面积
            gdf = gdf.to_crs(crs)  # 转回原始坐标系
        except Exception as e:
            print(f"警告：无法转换到UTM坐标系: {e}")
            print("使用原始CRS计算面积...")
            gdf['area_m2'] = gdf.area  # 使用原始CRS的面积
        
        # 5. 筛选面积 >= min_area 的要素
        original_count = len(gdf)
        gdf = gdf[gdf['area_m2'] >= min_area].copy()
        filtered_count = len(gdf)
        
        # 6. 写出 Shapefile / GeoPackage
        if not gdf.empty:
            gdf.to_file(out_path)
            print(f"成功生成: {out_path}")
            print(f"  原始要素数: {original_count}")
            print(f"  筛选后要素数: {filtered_count}")
            print(f"  过滤掉 {original_count - filtered_count} 个面积<{min_area}的要素")
        else:
            print(f"警告：筛选后没有要素保留: {dir}")
            # 可以选择创建一个空的shapefile
            gdf.to_file(out_path)  # 保存空文件
    else:
        print(f"警告：没有找到任何非零要素: {dir}")


if __name__ == '__main__':
    # dir = r"F:\滑坡\12月31日\训练数据\16x16分割\predictions_v2\合成\_landsat_{T}__大图_geo_bayesian.tif"
    # out = r"F:\滑坡\12月31日\训练数据\16x16分割\predictions_v2\结果2\_landsat_{T}__大图_geo__bayesian.shp"
    
    
    # min_area_threshold = 3600  # 设置最小面积阈值
    
    # for T in range(2021, 2025):
    #     dir_path = dir.format(T=T)
    #     out_path = out.format(T=T)
        
    #     # 检查文件是否存在
    #     if not Path(dir_path).exists():
    #         print(f"文件不存在: {dir_path}")
    #         continue
            
    #     print(f"处理 {T} 年数据 (最小面积: {min_area_threshold}m²)...")
    #     to_tif_geo(dir_path, out_path, min_area=min_area_threshold)
    #     print("-" * 50)


    dir_path = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\指数特征图+经验阈值\自动寻找阈值\landslide_full_optimal_fid_210_f0.5.tif"
    out_path = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\指数特征图+经验阈值\自动寻找阈值\shp_multi_index_fid_210"
    min_area_threshold = 3600
    to_tif_geo(dir_path, out_path, min_area=min_area_threshold)