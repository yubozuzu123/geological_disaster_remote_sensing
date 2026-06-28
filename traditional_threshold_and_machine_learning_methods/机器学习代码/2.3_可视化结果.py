import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from rasterio.windows import Window

def visualize_landslide_highres(truth_shp, pred_shp, tif_path, output_png,
                                 col_range=(3000, 4000), row_range=(500, 1500)):
    """
    基于指定的影像行列范围生成高分辨率滑坡可视化图。

    参数:
        truth_shp: 真实滑坡面 shapefile 路径
        pred_shp:  预测滑坡面 shapefile 路径
        tif_path:  Landsat 6波段影像路径
        output_png: 输出 PNG 路径
        col_range:  (col_start, col_end) 列范围（像素坐标，起始0）
        row_range:  (row_start, row_end) 行范围（像素坐标，起始0）
    """
    # 1. 读取影像子集（只读取所需行列）
    col_start, col_end = col_range
    row_start, row_end = row_range
    width = col_end - col_start
    height = row_end - row_start

    with rasterio.open(tif_path) as src:
        # 使用 window 读取波段4、3、2
        window = Window(col_start, row_start, width, height)
        red = src.read(4, window=window).astype(np.float32)
        green = src.read(3, window=window).astype(np.float32)
        blue = src.read(2, window=window).astype(np.float32)

        # 处理无效值
        if src.nodata is not None:
            red[red == src.nodata] = 0
            green[green == src.nodata] = 0
            blue[blue == src.nodata] = 0
        red = np.nan_to_num(red, nan=0, posinf=0, neginf=0)
        green = np.nan_to_num(green, nan=0, posinf=0, neginf=0)
        blue = np.nan_to_num(blue, nan=0, posinf=0, neginf=0)

        # 百分位数拉伸
        def percent_stretch(band):
            p_low, p_high = np.percentile(band, (2, 98))
            if p_high == p_low:
                return np.zeros_like(band, dtype=np.uint8)
            stretched = (band - p_low) / (p_high - p_low) * 255
            stretched = np.clip(stretched, 0, 255).astype(np.uint8)
            return stretched

        rgb = np.stack([percent_stretch(red),
                        percent_stretch(green),
                        percent_stretch(blue)], axis=-1)

        # 计算该窗口对应的地理范围
        transform = src.transform
        # 窗口左上角地理坐标
        x_min = transform[2] + col_start * transform[0] + row_start * transform[1]
        y_max = transform[5] + col_start * transform[3] + row_start * transform[4]
        # 窗口右下角地理坐标
        x_max = x_min + width * transform[0]   # transform[0] 是 x 方向像素宽度
        y_min = y_max + height * transform[4]  # transform[4] 是 y 方向像素高度（通常为负）
        # 由于 origin='upper'，y 方向递减，所以 y_min 小于 y_max
        extent = [x_min, x_max, y_min, y_max]

        crs = src.crs

    # 2. 读取 shapefile 并投影到影像 CRS
    truth = gpd.read_file(truth_shp)
    pred = gpd.read_file(pred_shp)
    if truth.crs != crs:
        truth = truth.to_crs(crs)
    if pred.crs != crs:
        pred = pred.to_crs(crs)

    # 3. 计算 TP、FN、FP（在影像 CRS 下）
    truth_union = truth.geometry.unary_union
    pred_union = pred.geometry.unary_union
    tp_geom = truth_union.intersection(pred_union)
    fn_geom = truth_union.difference(pred_union)
    fp_geom = pred_union.difference(truth_union)

    # 4. 设置绘图范围（即窗口的地理范围）
    xmin, xmax, ymin, ymax = extent

    # 5. 绘图，输出尺寸为 (height, width) 像素
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)  # 使实际像素 = width × height
    # 显示真彩色底图
    ax.imshow(rgb, extent=extent, origin='upper', zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # 6. 添加 TN：灰色半透明矩形覆盖整个显示范围
    rect = mpatches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                              facecolor='gray', alpha=0.5, zorder=1)
    ax.add_patch(rect)

    # 7. 添加 TP、FN、FP 面
    if not tp_geom.is_empty:
        gpd.GeoSeries([tp_geom]).plot(ax=ax, color='yellow', alpha=0.7, zorder=2)
    if not fn_geom.is_empty:
        gpd.GeoSeries([fn_geom]).plot(ax=ax, color='red', alpha=0.7, zorder=2)
    if not fp_geom.is_empty:
        gpd.GeoSeries([fp_geom]).plot(ax=ax, color='blue', alpha=0.7, zorder=2)

    # 8. 图例
    legend_elements = [
        mpatches.Patch(facecolor='gray', alpha=0.5, label='TN (Background)'),
        mpatches.Patch(facecolor='yellow', alpha=0.7, label='TP'),
        mpatches.Patch(facecolor='red', alpha=0.7, label='FN'),
        mpatches.Patch(facecolor='blue', alpha=0.7, label='FP')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # 9. 去掉坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    # 10. 保存图片
    plt.tight_layout(pad=0)
    plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"高分辨率可视化结果已保存至: {output_png}")

if __name__ == "__main__":
    # 请根据实际文件路径和行列范围修改
    truth_shp = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\2016\2016.shp"
    pred_shp = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\指数特征图+经验阈值\自动寻找阈值\shp_multi_index_fid_210\shp_multi_index_fid_210.shp"
    tif_path = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\Nepal_Landsat\scale_along_median_2016nepal2.tif" 
    output_png = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\复现\指数特征图+经验阈值\自动寻找阈值\shp_multi_index_fid_210\result.png" 
    # 指定行列范围（列 3000-4000，行 500-1500）
    visualize_landslide_highres(truth_shp, pred_shp, tif_path, output_png,
                                col_range=(3500, 4500), row_range=(500, 1500))
                                # col_range=(1500, 2500), row_range=(0, 1000))


