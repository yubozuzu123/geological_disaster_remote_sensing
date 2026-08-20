import geopandas as gpd
import pandas as pd
import json
import logging
import os.path
import numpy as np
from shapely.geometry import LineString
from rasterio.windows import Window, intersection


''' A.1 计算坡度的函数 '''
def compute_slope_factor(slope_degree):
    # 坡度影响因子 (经验公式)
    return np.exp(3.533 * np.tan(np.radians(slope_degree)))


''' A.2 计算风的函数 '''
def compute_wind_factor(wind_spd, wind_dir, cell_dir):
    wind_spd = wind_spd / 3.0    # 风影响因子 (根据风速和风向对蔓延方向的夹角调整) 4
    wind_to = (wind_dir + 180) % 360    # 把风向转换为去向
    angle_diff = min(abs(wind_to - cell_dir), 360 - abs(wind_to - cell_dir))  # 计算最小夹角
    return np.exp(0.1783 * wind_spd * np.cos(np.radians(angle_diff)))


''' A.3 计算方向的函数 '''
def compute_direction_angle(dy, dx):
    if dy == 0 and dx == 0:
        return 0  # 自身不计算方向
    angle = (90 - np.degrees(np.arctan2(-dy, dx))) % 360
    return angle


''' A.4 解析以GeoJson格式传过来的防火隔离带Polyline '''
def parse_polyline_geojson(geojson_polyline_str):
    """
        从 GeoJSON 字符串中提取所有 LineString 或 MultiLineString 的坐标。

        参数:
            geojson_polyline_str (str): GeoJSON 字符串，类型应为 FeatureCollection。

        返回:
            gpd.GeoDataFrame: 所有线条组成的 GeoDataFrame
    """
    try:
        if not geojson_polyline_str:
            raise ValueError("传入的防火隔离带字符串为空值")

        geojson_polyline = json.loads(geojson_polyline_str)

        if geojson_polyline.get("type") != "FeatureCollection":
            raise ValueError("GeoJSON类型必须是FeatureCollection")

        features = geojson_polyline.get("features")
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("FeatureCollection中未包含有效的features")

        coord_list = []
        proper_list = []

        for feature in features:
            geometry = feature.get("geometry")
            properties = feature.get("properties", {}).get("name")
            if not geometry:
                continue

            geom_type = geometry.get("type")
            coords = geometry.get("coordinates")

            if geom_type == "LineString":
                coord_list.append(LineString(coords))
                proper_list.append(properties)

            elif geom_type == "MultiLineString":
                # coord_list.append(MultiLineString(coords))  # 多条线：每一条 append 进 all_lines
                for crds in coords:
                    coord_list.append(LineString(crds))
                    proper_list.append(properties)

            else:
                print(f"跳过不支持的geometry类型: {geom_type}")

        if coord_list:
            attribute = pd.DataFrame({"name": proper_list})
            line_vector = gpd.GeoDataFrame(attribute, geometry=coord_list, crs="EPSG:4326")
            return line_vector

        else:
            return None

    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
        print(f"GeoJSON解析异常: {e}")
        return None


''' A.5 A.4 解析以GeoJson格式传过来的火点polygon '''
def parse_polygon_geojson(geojson_polygon_str, target_crs='EPSG:32650', logger=None):
    """
    专门负责解析火源数据，并统一转换到DEM的坐标系
    Args:
        geojson_polygon_str: 字典(Dict) 或 JSON字符串(Str) 或 None
        target_crs: 目标坐标系 (通常是 dem_src.crs)
        logger: 日志记录器 (可选)

    Returns:
        geopandas.GeoDataFrame 或 None
    """

    if not geojson_polygon_str:
        return None

    # 如果没有传 logger，就用默认的 print 或者临时 logger，防止报错
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        input_list = geojson_polygon_str if isinstance(geojson_polygon_str, list) else [geojson_polygon_str]

        gdf_list = []  # 用于暂存解析好的小表格

        for item in input_list:
            data = item

            # A. 如果是字符串，尝试解析为 Dict
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("跳过无效的 JSON 字符串，无法解析")
                    continue
            # B. 如果是字典，转 GeoDataFrame
            if isinstance(data, dict):
                temp_gdf = None
                # 判断是 FeatureCollection 还是 单个 Feature
                if "features" in data:
                    temp_gdf = gpd.GeoDataFrame.from_features(data["features"])
                else:
                    # 容错：尝试直接解析单个对象
                    temp_gdf = gpd.GeoDataFrame.from_features([data])

                # 只有解析成功且不为空时，才加入待合并列表
                if temp_gdf is not None and not temp_gdf.empty:
                    gdf_list.append(temp_gdf)

        # 3. 合并
        if not gdf_list:
            return None

        geojson_polygon_gdf = pd.concat(gdf_list, ignore_index=True)
        # 4. 投影转换
        if geojson_polygon_gdf is not None and not geojson_polygon_gdf.empty:
            if geojson_polygon_gdf.crs is None:
                geojson_polygon_gdf.set_crs(epsg=4326, inplace=True)
            if target_crs is not None:
                geojson_polygon_gdf = geojson_polygon_gdf.to_crs(target_crs)
                logger.info(f"成功加载{len(geojson_polygon_gdf)}个火源数据")

            return geojson_polygon_gdf

    except Exception as e:
        logger.error(f"解析火源GeoJSON数据失败: {e}")

    return None


def get_dynamic_window(dem_src, fire_source_gdf, window_size, scale=2):
    dem_height, dem_width = dem_src.shape
    dem_window = Window(0, 0, dem_width, dem_height)

    x_res = abs(dem_src.res[0])
    y_res = abs(dem_src.res[1])

    # 1. 计算【默认最小】窗口像素 (防止火点太小时窗口缩成一个点)
    # 假设参数 5.0 对应 1250 像素 (即 1单位=250像素)
    extent_height, extent_width = window_size
    min_h_pixels, min_w_pixels = int(extent_height * 400), int(extent_width * 400)

    # 初始化变量
    req_h_pixels = 0
    req_w_pixels = 0
    center_row = dem_height // 2
    center_col = dem_width // 2

    # 2. 如果有火源数据，基于火源计算窗口需求
    if fire_source_gdf is not None and not fire_source_gdf.empty:
        try:
            # 获取火场矢量的边界
            minx, miny, maxx, maxy = fire_source_gdf.total_bounds
            row_top, col_left = dem_src.index(minx, maxy)
            row_bottom, col_right = dem_src.index(maxx, miny)
            # 计算火场本体的长宽 (像素)
            fire_h = abs(row_bottom - row_top)
            fire_w = abs(col_right - col_left)
            # 计算中心点
            center_row = int((row_top + row_bottom) / 2)
            center_col = int((col_left + col_right) / 2)
            # 【关键】应用缓冲系数
            req_h_pixels = int(fire_h * scale)
            req_w_pixels = int(fire_w * scale)

        except Exception as e:
            # 如果坐标系转换出错或越界，打个日志，后续会使用默认坐标
            # logger.warning(f"Window calculation error: {e}")
            pass

    # 3. 决策最终窗口大小 (取“默认最小”和“火场需求”的最大值)
    final_h = max(min_h_pixels, req_h_pixels)
    final_w = max(min_w_pixels, req_w_pixels)
    # 4. 根据中心点推算左上角偏移量
    col_off = center_col - int(final_w / 2)
    row_off = center_row - int(final_h / 2)
    # 5. 构造理想窗口 (可能含负数)
    ideal_window = Window(col_off, row_off, final_w, final_h)
    # 6. 取交集 (裁剪掉超出 DEM 边界的部分)
    final_window = intersection(dem_window, ideal_window)
    # 7. 生成对应的 Transform
    transform = dem_src.window_transform(final_window)
    win_shape = (int(final_window.height), int(final_window.width))

    return final_window, transform, win_shape


if __name__ == "__main__":
    # 这里是你自己的测试代码，用于验证函数是否能正确运行
    polyline_geojson_str = """
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "name": "Line 1"
          },
          "geometry": {
            "type": "LineString",
            "coordinates": [
              [102.0, 0.0],
              [103.0, 1.0],
              [104.0, 0.0],
              [105.0, 1.0]
            ]
          }
        }
      ]
    }
    """

    polygon_geojson_str = """
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "value": 1
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [
              [
                [115.74, 39.80],
                [115.75, 39.80],
                [115.75, 39.81],
                [115.74, 39.81],
                [115.74, 39.80]
              ]
            ]
          }
        },
        {
          "type": "Feature",
          "properties": {
            "value": 2
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [
              [
                [115.76, 39.82],
                [115.77, 39.82],
                [115.77, 39.83],
                [115.76, 39.83],
                [115.76, 39.82]
              ]
            ]
          }
        }
      ]
    }
    """

    # 调用你写的函数，比如 parse_feature_from_geojson
    geojson_string = ""
    # vector = parse_polyline_geojson(polyline_geojson_str)
    vector = parse_polygon_geojson(polygon_geojson_str)
    if vector is not None:
        geojson_path = os.path.join('C:/Users/H2O/Desktop/test/polygons.shp')
        vector.to_file(geojson_path, driver='ESRI Shapefile', encoding='utf-8', index=False)
    else:
        print("解析结果为None")
