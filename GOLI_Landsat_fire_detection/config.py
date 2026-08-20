from pathlib import Path


# =============================================================================
# 1. 输入与输出路径
# =============================================================================
# ROOT_DIR 下的每个一级子文件夹视为一景 Landsat 数据。
ROOT_DIR = Path(r"data")

# 所有结果统一输出到该目录。
OUTPUT_DIR = Path(r"result")

# 是否递归查找包含 Landsat 波段的文件夹。
SEARCH_RECURSIVELY = False

# 批处理汇总表。
SUMMARY_CSV = OUTPUT_DIR / "GOLI_batch_summary.csv"


# =============================================================================
# 2. 反射率输入模式
# =============================================================================
# "AUTO"：
#   根据文件名自动识别。
#   *_SR_B*.TIF -> Landsat Collection 2 Level-2 Surface Reflectance
#   *_B*.TIF    -> Landsat Level-1 TOA Reflectance
#
# "L1_TOA"：
#   强制使用 Level-1 DN + MTL 计算太阳天顶角校正后的 TOA 反射率。
#
# "L2_SR"：
#   强制使用 Level-2 地表反射率缩放公式。
REFLECTANCE_MODE = "AUTO"

# 阈值是基于 Landsat-8 Level-1 TOA 反射率建立的。
# 当输入为 Level-2 SR 时，仍允许使用论文阈值进行试验。
ALLOW_LEVEL2_SR_ADAPTATION = True


# =============================================================================
# 3. QA 掩膜
# =============================================================================
# 如实际数据中云误检较多，可改为 True。
USE_QA_CLOUD_MASK = False

# 是否通过 QA_PIXEL 删除水体。
# 默认不使用 QA 水体掩膜。
MASK_QA_WATER = False


# =============================================================================
# 4. GOLI 固定阈值
# =============================================================================

# Red <= 0.53 * SWIR2 - 0.214
UNAMBIGUOUS_RED_SWIR2_SLOPE = 0.53
UNAMBIGUOUS_RED_SWIR2_INTERCEPT = -0.214

# 与前面火点相邻，且 Red <= 0.35 * SWIR1 - 0.044
RECOVERY_RED_SWIR1_SLOPE = 0.35
RECOVERY_RED_SWIR1_INTERCEPT = -0.044

# Red <= 0.53 * SWIR2 - 0.125
POTENTIAL_RED_SWIR2_SLOPE = 0.53
POTENTIAL_RED_SWIR2_INTERCEPT = -0.125

# SWIR1 <= 1.08 * SWIR2 - 0.048
POTENTIAL_SWIR1_SWIR2_SLOPE = 1.08
POTENTIAL_SWIR1_SWIR2_INTERCEPT = -0.048


# =============================================================================
# 5. GOLI 上下文筛选
# =============================================================================
# 窗口从 5×5 开始，以 2 像元递增，最大到 61×61。
CONTEXT_MIN_WINDOW = 5
CONTEXT_MAX_WINDOW = 61
CONTEXT_WINDOW_STEP = 2

# 要求至少 25% 的周围像元可用。
CONTEXT_MIN_VALID_FRACTION = 0.25

# SWIR2/NIR > mean + max(3*std, 0.8)
CONTEXT_RATIO_STD_FACTOR = 3.0
CONTEXT_RATIO_MIN_INCREMENT = 0.8

# SWIR2 > mean + max(3*std, 0.08)
CONTEXT_SWIR2_STD_FACTOR = 3.0
CONTEXT_SWIR2_MIN_INCREMENT = 0.08


# =============================================================================
# 6. 可选空间清理
# =============================================================================
# 没有要求删除单像元火点，默认保留。
MIN_COMPONENT_PIXELS = 1

# None 表示不设置最大斑块限制。
MAX_COMPONENT_PIXELS = None


# =============================================================================
# 7. 输出控制
# =============================================================================
SAVE_REFLECTANCE = True
SAVE_FIRE_TYPE = True
SAVE_FIRE_POINTS = True

# 输出各级判定结果以及上下文统计，便于检查。
SAVE_DEBUG_RASTERS = True


# =============================================================================
# 8. Landsat Collection 2 Level-2 SR 缩放系数
# =============================================================================
SR_SCALE = 0.0000275
SR_OFFSET = -0.2
