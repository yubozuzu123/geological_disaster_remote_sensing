import geopandas as gpd
import os
import logging
import os.path
import shapely
import rasterio
import numpy as np
import FireUtils as Utils
from rasterio import features
from shapely.geometry import shape
from shapely.ops import unary_union
from rasterio.features import shapes


def fire_simulate(simulate_params, stop_event=None):
    logger = logging.getLogger(__name__)

    """ A. Load parameters or set default values """
    # fire_extent = tuple(float(x) for x in simulate_params.get("geo_coords", [124.345, 50.308]))
    fire_extent = simulate_params.get("geo_coords", None)
    wind_speed = float(simulate_params.get("wind_speed", 10.0))
    wind_direction = float(simulate_params.get("wind_direction", 315.0))
    air_temperature = float(simulate_params.get("air_temperature", 25.0))  # 新增：空气温度(℃)
    relative_humidity = float(simulate_params.get("relative_humidity", 35.0))  # 新增：相对湿度(%)
    isolate_belts = str(simulate_params.get("Isolation_Belts", ""))
    window_size = tuple(float(x) for x in simulate_params.get("window_shape", [5.0, 5.0]))
    spread_time = float(simulate_params.get("spread_time", 48.0))
    refresh_interval = float(simulate_params.get("refresh_interval", 15.0))
    fire_area_id = int(simulate_params.get("fire_area_id", 0))
    fire_extent_id = int(simulate_params.get("fire_extent_id", 0))
    source_folder = str(simulate_params.get("source_folder", '/'))
    target_folder = str(simulate_params.get("target_folder", '/'))


    """ B. Paths Definition and Open Files """
    # B.1 DEM
    input_dem_path = os.path.join(source_folder, 'dem.tif')
    if not os.path.exists(input_dem_path):
        logger.error(f"DEM文件不存在: {input_dem_path}")
        return
    dem_src = rasterio.open(input_dem_path)
    meta = dem_src.meta.copy()

    # B.2 Land Cover
    input_land_cover_path = os.path.join(source_folder, 'land_cover.tif')
    land_cover_src = rasterio.open(input_land_cover_path)

    # B.3 解析防火隔离带
    firebreak = Utils.parse_polyline_geojson(isolate_belts)
    # *****************************************************************
    # input_json_path = os.path.join('C:/Users/H2O/Desktop/box.geojson')
    # firebreak = gpd.GeoDataFrame.from_file(input_json_path)
    # geojson_path = os.path.join('C:/Users/H2O/Desktop/test.shp')
    # firebreak.to_file(geojson_path, driver='ESRI Shapefile', encoding='utf-8')
    # base_folder = os.path.join('G:/Beijing/WildFire/Assign/')
    # input_json_path = os.path.join(base_folder, 'line.shp')
    # geojson = gpd.GeoDataFrame.from_file(input_json_path)
    # *****************************************************************

    # B.4 解析GeoJSON内容为GeoDataFrame
    fire_extent_gdf = Utils.parse_polygon_geojson(fire_extent, target_crs=dem_src.crs, logger=logger)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)


    ''' C. Parameters Definition '''
    # dem 数据网格大小
    decay = 0.02
    cell_res = round((dem_src.res[0] + dem_src.res[1]), 1) / 2.0
    time_res = 15
    time_scale = max(int(refresh_interval / time_res), 1)
    interval = time_scale * int(time_res)

    window, transform, win_shape = Utils.get_dynamic_window(dem_src, fire_extent_gdf, window_size)

    current_steps = 0

    # 1. 初始化网格
    simulate_grid = np.zeros(win_shape, dtype='uint8')

    # 2. 将前端传入的基准面 (fire_extent_gdf) 栅格化到 grid 中
    if fire_extent_gdf is not None and not fire_extent_gdf.empty:
        shapes_to_burn = [(geom, 1) for geom in fire_extent_gdf.geometry]
        features.rasterize(shapes=shapes_to_burn,  out_shape=win_shape, out=simulate_grid,
                           transform=transform, all_touched=True, default_value=1, dtype='uint8')
    else:
        if fire_extent_id == 0:     # 只有在既没有GeoJSON又需要初始化时，才使用单点
            ctr_row, ctr_col = dem_src.index(761104.054, 3392557.975)
            if 0 <= ctr_row < win_shape[0] and 0 <= ctr_col < win_shape[1]:
                simulate_grid[ctr_row, ctr_col] = 1

    # 3. accumulate_grid = np.zeros(win_shape, dtype='float32')
    # simulate_grid[ctr_row, ctr_col] = 1   # 初始化火源 # 0=未燃, 1=燃烧中, 2=燃尽
    # accumulate_grid[ctr_row, ctr_col] = 1.0
    fire_grid = simulate_grid
    heat_grid = simulate_grid.copy().astype('float16')
    # 给初始火点赋予初始热量，直接起火
    # heat_grid[fire_grid == 1] = ignition_threshold + 0.5

    # 4. 设置时间步和热量逻辑
    if fire_extent_id == 0:
        current_steps = 0
        # heat_grid[fire_grid == 1] = ignition_threshold + 0.5

    else:
        # 更新模式：计算接续的时间步
        current_steps = (fire_extent_id + 1) * interval
        if fire_extent_gdf is not None and not fire_extent_gdf.empty:
            file_name = str(fire_area_id) + '-' + str(fire_extent_id)
            current_union = fire_extent_gdf
            current_union = current_union.to_crs('EPSG:4326')
            current_json_path = os.path.join(target_folder, f'{str(file_name)}.geojson')
            current_union.to_file(current_json_path, driver='GeoJSON', encoding='utf-8')


    ''' D. Load Rasters and Define transform '''
    dem = dem_src.read(1, window=window)
    land_cover = land_cover_src.read(1, window=window)
    meta.update({'height': window.height, 'width': window.width, 'transform': transform, 'dtype': 'uint8'})

    if firebreak is not None and not firebreak.empty:
        strip_vector = firebreak.to_crs(dem_src.crs)
        geometry = strip_vector.unary_union.buffer(cell_res)
        strip_band = features.rasterize(shapes=[geometry],
                                        out_shape=win_shape,
                                        transform=transform,
                                        all_touched=True,
                                        fill=1,
                                        default_value=0,
                                        dtype='uint8'
                                        )
        land_cover = land_cover * strip_band
        # *****************************************************************
        # strip_line_output = os.path.join("C:/Users/H2O/Desktop/strip_line.tif")
        # with rasterio.open(strip_line_output, 'w', **meta) as strip_dst:
        #     strip_dst.write(strip_band, indexes=1)
        #     strip_dst.write_colormap(1, {0: (255, 165, 0, 255), 1: (0, 0, 0, 255), 2: (255, 165, 0, 255)})
        # *****************************************************************


    ''' E. Fire Spread '''
    heat_gain = 0.4             # 0.4  3
    cooling_factor = 0.95       # 每步热量衰减比例
    ignition_threshold = 2.0    # 点燃阈值 2.0  4
    time_steps = int(spread_time * 60 + 1)

    # 新增：温湿度环境修正
    # 空气温度越高，燃料预热越快；相对湿度越高，可燃物越不易点燃
    temperature_factor = 1.0 + 0.02 * (air_temperature - 25.0)
    humidity_factor = 1.0 - 0.01 * (relative_humidity - 35.0)
    env_factor = np.clip(temperature_factor * humidity_factor, 0.25, 1.5)

    neighbors = [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if not (dy == 0 and dx == 0)]
    update_mask = np.random.rand(window.height, window.width)

    # 循环模拟
    for t in range(current_steps, time_steps + 1, int(time_res / 3)):
        if stop_event is not None and stop_event.is_set():
            logger.info(f"[{fire_area_id}] 模拟被中断（t={t}）.")
            return      # 中断并结束任务

        burning_cells = np.argwhere(fire_grid == 1)
        np.random.shuffle(burning_cells)  # 打乱顺序

        for cell in burning_cells:
            y, x = cell
            fire_grid[y, x] = 2  # 燃尽
            if t > 15:
                if update_mask[y, x] < 0.35:
                    continue  # 跳过这个像素 0.35

            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < win_shape[0] and 0 <= nx < win_shape[1]:
                    if fire_grid[ny, nx] == 0:

                        # 1. 计算坡度
                        dz = dem[ny, nx] - dem[y, x]
                        distance = np.sqrt((dy * cell_res / 1) ** 2 + (dx * cell_res / 1) ** 2) + 1e-6
                        slope_deg = np.degrees(
                            np.arctan(dz / distance)
                        )
                        slope_factor = Utils.compute_slope_factor(slope_deg)

                        # 2. 计算风影响
                        cell_angle = Utils.compute_direction_angle(dy, dx)
                        wind_factor = Utils.compute_wind_factor(wind_speed, wind_direction, cell_angle)

                        # 3. 计算土地利用影响
                        land_cover_factor = land_cover[ny, nx]

                        # 4. 热量增加（加入扰动）
                        heat_gain_value = heat_gain * land_cover_factor * slope_factor * wind_factor * env_factor
                        heat_gain_value = np.clip(heat_gain_value, 0, 3)    # 0， 3
                        heat_grid[ny, nx] += heat_gain_value

                        # 4. 计算传播概率 (确保范围在 [0,1] 之间)
                        # spread_prob =initial_spread_rate * slope_factor * wind_factor * land_cover_factor
                        # spread_prob = np.clip(spread_prob, 0, 1)
                        # 5. 随机传播# noise = np.random.normal(0, 0.1) + noise
                        # r = 1 + 0.5 * np.random.rand()
                        # if r < spread_prob:
                        #     fire_grid[ny, nx] = 1

        # 衰减热量（模拟冷却）
        heat_grid *= cooling_factor

        # 点燃热量足够的单元格
        new_fires = (fire_grid == 0) & (heat_grid >= ignition_threshold)
        fire_grid[new_fires] = 1

        if (t >= interval) & (t % interval == 0):
            min_time = int(t // interval)
            file_name = str(fire_area_id) + '-' + str(min_time)

            target_raster_path = os.path.join(target_folder, f'{file_name}.tif')
            with rasterio.open(target_raster_path, 'w', **meta) as dst:
                dst.write(fire_grid, indexes=1)
                dst.write_colormap(1, {0: (0, 0, 0, 255), 1: (255, 165, 0, 255), 2: (255, 0, 0, 255)})

            # 矢量化输出
            mask = fire_grid > 0
            geoms_list = [shape(geom) for geom, val in shapes(fire_grid, mask=mask, transform=transform)]
            if geoms_list:
                geom_union = unary_union(geoms_list)
                flat_polys = shapely.get_parts(geom_union)
                spread_gdf = gpd.GeoDataFrame({'geometry': flat_polys}, crs=dem_src.crs)
                # spread_gdf['min_time'] = min_time

                if fire_extent_gdf is not None:
                    joined = gpd.sjoin(spread_gdf, fire_extent_gdf, how='inner', predicate='intersects')
                    spread_vector = joined.groupby(joined.index).agg({
                        'geometry': 'first',
                        'AreaName': lambda x: ','.join(sorted(set(x.dropna().astype(str))))
                    })
                    spread_union = gpd.GeoDataFrame(spread_vector, geometry='geometry', crs=dem_src.crs)
                    spread_union = spread_union.to_crs('EPSG:4326')

                    # target_shp_path = os.path.join(target_folder, f'{str(file_name)}.shp')
                    # spread_union.to_file(target_shp_path, driver='ESRI Shapefile', encoding='utf-8')
                    target_json_path = os.path.join(target_folder, f'{str(file_name)}.geojson')
                    spread_union.to_file(target_json_path, driver='GeoJSON', encoding='utf-8')

    logger.info(f'[{fire_area_id}]模拟完成.')
