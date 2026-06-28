import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import random
import shutil


# ============================================================
# 配置参数
# ============================================================
# 裁剪参数
PATCH_SIZE = 256      # 裁剪图像块大小（像素）
STRIDE = 128          # 滑动窗口步长（小于PATCH_SIZE时有重叠）
MIN_LS_RATIO = 0.005  # 最小滑坡占比
# 数据划分
VAL_RATIO = 0.2      # 验证集比例
# 原始数据路径（请根据实际情况修改）
RAW_DATA_DIR = './数据'
# 原始数据配置：TIF图像 + SHP矢量文件
DATASETS = [
    {
        'name': '横断_2008',
        'tif': f'{RAW_DATA_DIR}/hd/2008.tif',
        'shp': f'{RAW_DATA_DIR}/hd/2008shp/2008.shp',
    },
    {
        'name': '横断_2018',
        'tif': f'{RAW_DATA_DIR}/hd/2018.tif',
        'shp': f'{RAW_DATA_DIR}/hd/2018shp/2018.shp',
    },
    {
        'name': '汶川',
        'tif': f'{RAW_DATA_DIR}/WC/WC.tif',
        'shp': f'{RAW_DATA_DIR}/WC/SHP/WC.shp',
    },
]
# 输出目录
OUTPUT_DIR = './landslide_data'


# ============================================================

def percentile_stretch(img, low=2, high=98):
    # 提取有效像素（排除 nodata 的 0 值）
    valid_mask = img > 0
    valid_pixels = img[valid_mask]

    if len(valid_pixels) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    # 仅对有效像素计算百分位
    low_val = np.percentile(valid_pixels, low)
    high_val = np.percentile(valid_pixels, high)
    # 对全图做 clip 和线性拉伸
    img_clipped = np.clip(img, low_val, high_val)
    if high_val > low_val:
        img_stretched = (img_clipped - low_val) / (high_val - low_val) * 255
    else:
        img_stretched = np.zeros_like(img, dtype=np.float64)
    # nodata 区域保持为 0（黑色背景）
    result = img_stretched.astype(np.uint8)
    result[~valid_mask] = 0
    return result


def create_mask_from_shp(shp_path, bounds, out_shape):
    from rasterio.features import rasterize
    # 读取矢量文件
    gdf = gpd.read_file(shp_path)
    # 获取图像边界
    img_box = box(*bounds)
    # 创建空白掩膜
    height, width = out_shape
    mask_array = np.zeros((height, width), dtype=np.uint8)
    # 计算地理坐标到像素坐标的缩放比例
    geo_width = bounds[2] - bounds[0]
    geo_height = bounds[3] - bounds[1]
    scale_x = width / geo_width if geo_width > 0 else 1
    scale_y = height / geo_height if geo_height > 0 else 1

    # 批量裁剪几何形状到图像边界
    clipped_geoms = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        try:
            clipped_geom = geom.intersection(img_box)
            if not clipped_geom.is_empty and clipped_geom.is_valid:
                clipped_geoms.append(clipped_geom)
        except:
            continue

    if not clipped_geoms:
        return mask_array

    # 批量转换为像素坐标
    pixel_geoms = []
    for geom in clipped_geoms:
        # 简单的缩放变换（忽略旋转和倾斜）
        def transform_fn(x, y, z=None):
            return ((x - bounds[0]) * scale_x, (bounds[3] - y) * scale_y)

        from shapely.ops import transform as shapely_transform
        pixel_geom = shapely_transform(transform_fn, geom)
        pixel_geoms.append(pixel_geom)
    # 批量栅格化
    if pixel_geoms:
        try:
            mask_array = rasterize(
                [(geom, 1) for geom in pixel_geoms],
                out_shape=mask_array.shape,
                default_value=0
            )
        except Exception as e:
            print(f"      栅格化失败: {e}")

    return mask_array


def clip_patches(image, mask, patch_size, stride):
    h, w = image.shape[:2]
    patches = []
    mask_patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image[y:y+patch_size, x:x+patch_size]
            msk_patch = mask[y:y+patch_size, x:x+patch_size]
            # 计算滑坡占比
            ls_ratio = np.sum(msk_patch > 0) / (patch_size * patch_size)
            # 过滤掉滑坡占比太低的样本
            if ls_ratio >= MIN_LS_RATIO:
                patches.append(img_patch)
                mask_patches.append(msk_patch)
    return patches, mask_patches


def rgb_to_uint8(band_data):
    # 处理无效值
    band_data = np.nan_to_num(band_data, nan=0, posinf=0, neginf=0)

    # 百分位拉伸
    band_data = percentile_stretch(band_data, low=2, high=98)

    return band_data


# ============================================================
# 主处理函数
# ============================================================

def process_dataset(ds_config):
    name = ds_config['name']
    tif_path = ds_config['tif']
    shp_path = ds_config['shp']
    print(f"\n{'='*60}")
    print(f"处理数据集: {name}")
    print(f"{'='*60}")
    # 检查文件是否存在
    if not os.path.exists(tif_path):
        print(f"  警告: TIF文件不存在 - {tif_path}")
        return 0
    if not os.path.exists(shp_path):
        print(f"  警告: SHP文件不存在 - {shp_path}")
        return 0
    # 创建输出目录
    generate_train = ds_config.get('generate_train', True)
    if generate_train:
        for split in ['train', 'val']:
            os.makedirs(f'{OUTPUT_DIR}/{name}/{split}/images', exist_ok=True)
            os.makedirs(f'{OUTPUT_DIR}/{name}/{split}/labels', exist_ok=True)
    else:
        # 只生成验证集
        os.makedirs(f'{OUTPUT_DIR}/{name}/val/images', exist_ok=True)
        os.makedirs(f'{OUTPUT_DIR}/{name}/val/labels', exist_ok=True)

    # 读取 TIF 图像
    print(f"  读取图像: {tif_path}")
    with rasterio.open(tif_path) as src:
        # 读取所有波段
        bands = []
        for i in range(1, src.count + 1):
            band = src.read(i)
            band = rgb_to_uint8(band)
            bands.append(band)
        # 取前3个波段作为RGB（如果是多波段）
        if len(bands) >= 3:
            image = np.stack([bands[0], bands[1], bands[2]], axis=-1)
        else:
            # 单波段图像复制3次
            image = np.stack([bands[0]] * 3, axis=-1)
        bounds = src.bounds
        print(f"  图像尺寸: {image.shape[1]}x{image.shape[0]}, 波段数: {src.count}")

    # 创建滑坡掩膜
    print(f"  创建滑坡掩膜: {shp_path}")
    try:
        mask = create_mask_from_shp(shp_path, (bounds.left, bounds.bottom,
                                                 bounds.right, bounds.top),
                                    out_shape=(image.shape[0], image.shape[1]))
    except Exception as e:
        print(f"  警告: 创建掩膜失败 - {e}")
        # 尝试使用简单的边界框掩膜
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # 裁剪图像块
    print(f"  裁剪图像块 (大小={PATCH_SIZE}, 步长={STRIDE})...")
    patches, mask_patches = clip_patches(image, mask, PATCH_SIZE, STRIDE)
    if len(patches) == 0:
        print(f"  警告: 没有找到满足条件的样本！")
        print(f"  提示: 请检查SHP文件是否与TIF图像对齐，或降低 MIN_LS_RATIO")
        return 0
    print(f"  生成 {len(patches)} 个样本")

    # 划分训练集/验证集
    random.seed(42)
    indices = list(range(len(patches)))
    random.shuffle(indices)

    # 检查是否只生成测试集
    generate_train = ds_config.get('generate_train', True)

    if generate_train:
        val_count = int(len(patches) * VAL_RATIO)
        val_indices = set(indices[:val_count])
        train_indices = set(indices[val_count:])
        print(f"  训练集: {len(train_indices)} 样本")
        print(f"  验证集: {len(val_indices)} 样本")
    else:
        # 只生成验证集（全部样本作为测试）
        val_indices = set(indices)
        train_indices = set()
        print(f"  验证集（全部）: {len(val_indices)} 样本")

    # 保存图像和标签
    for idx, (img_patch, msk_patch) in enumerate(zip(patches, mask_patches)):
        # 确定是训练集还是验证集
        if idx in val_indices:
            split = 'val'
        elif generate_train:
            split = 'train'
        else:
            continue  # 不生成训练集时跳过

        # 保存图像 (JPG)
        img_pil = Image.fromarray(img_patch)
        img_path = f'{OUTPUT_DIR}/{name}/{split}/images/{name}_{idx:05d}.jpg'
        img_pil.save(img_path, quality=95)

        # 保存标签 (PNG, 二值图像)
        msk_patch = (msk_patch > 0).astype(np.uint8) * 255
        msk_pil = Image.fromarray(msk_patch)
        lbl_path = f'{OUTPUT_DIR}/{name}/{split}/labels/{name}_{idx:05d}.png'
        msk_pil.save(lbl_path)
    return len(patches)


def main():
    """主函数"""
    print("=" * 60)
    print("滑坡数据集生成工具")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"图像块大小: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"最小滑坡占比: {MIN_LS_RATIO*100:.1f}%")
    print(f"验证集比例: {VAL_RATIO*100:.0f}%")

    total_samples = 0

    for ds_config in DATASETS:
        count = process_dataset(ds_config)
        total_samples += count

    print("\n" + "=" * 60)
    print("生成完成!")
    print(f"总计生成 {total_samples} 个样本")
    print(f"数据保存在: {OUTPUT_DIR}/")
    print("=" * 60)

    # 打印目录结构
    print("\n目录结构:")
    for name in ['横断_2008', '横断_2018', '汶川', '尼泊尔_2015', '尼泊尔_2016']:
        for split in ['train', 'val']:
            img_dir = f'{OUTPUT_DIR}/{name}/{split}/images'
            if os.path.exists(img_dir):
                count = len(os.listdir(img_dir))
                print(f"  {OUTPUT_DIR}/{name}/{split}/images/  ({count} 张)")


if __name__ == '__main__':
    main()
