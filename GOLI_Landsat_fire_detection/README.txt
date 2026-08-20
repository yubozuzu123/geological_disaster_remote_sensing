GOLI Landsat-8 反射率火点检测算法：单景核心试验版
======================================================================

一、来源
----------------------------------------------------------------------
本程序基于：

Kumar, S. S. & Roy, D. P.
Global operational land imager Landsat-8 reflectance-based active fire
detection algorithm.
International Journal of Digital Earth, 2018, 11(2): 154-178.
DOI: 10.1080/17538947.2017.1391341


二、当前实现的论文内容
----------------------------------------------------------------------
本版本实现论文单景核心部分：

1. 公式（12）：明确火点
   Red <= 0.53 * SWIR2 - 0.214

2. 公式（13）：相邻强火/饱和火点恢复
   与公式（12）火点相邻，且：
   Red <= 0.35 * SWIR1 - 0.044

3. 公式（14）和（15）：潜在火点初选
   Red <= 0.53 * SWIR2 - 0.125
   或：
   SWIR1 <= 1.08 * SWIR2 - 0.048

4. 公式（8）和（9）：潜在火点上下文筛选
   SWIR2/NIR > 背景均值 + max(3×标准差, 0.8)
   并且：
   SWIR2 > 背景均值 + max(3×标准差, 0.08)

5. 自适应背景窗口
   窗口从 5×5、7×7、9×9……增长到 61×61。
   使用第一个至少包含 25% 有效背景像元的窗口。

6. 公式（16）：水体背景排除
   Blue > Green > Red > NIR


三、尚未实现的论文部分
----------------------------------------------------------------------
论文公式（17）和（18）使用前六个月的 Landsat 时间序列减少建筑、
永久高亮目标等误检。

本试验版先实现可以直接运行的单景核心算法，未启用六个月多时相过滤。


四、输入数据
----------------------------------------------------------------------
支持两种输入：

1. Level-1：
   *_B2.TIF 至 *_B7.TIF
   *_MTL.txt

   程序根据 MTL 计算太阳天顶角校正后的 TOA 反射率。
   这是最接近论文原始方法的使用方式。

2. Level-2：
   *_SR_B2.TIF 至 *_SR_B7.TIF

   程序使用：
   reflectance = DN × 0.0000275 - 0.2

   注意：
   论文阈值是依据 TOA 反射率建立的。
   将其直接用于 Level-2 地表反射率属于适配测试，不是严格复现。


五、文件夹结构
----------------------------------------------------------------------
GOLI_Landsat_fire_detection/
│
├── Data/
│   ├── Scene_1/
│   │   ├── LC09_..._SR_B2.TIF
│   │   ├── LC09_..._SR_B3.TIF
│   │   ├── LC09_..._SR_B4.TIF
│   │   ├── LC09_..._SR_B5.TIF
│   │   ├── LC09_..._SR_B6.TIF
│   │   ├── LC09_..._SR_B7.TIF
│   │   └── LC09_..._QA_PIXEL.TIF
│   │
│   ├── Scene_2/
│   │   └── ...
│   │
│   └── Scene_3/
│       └── ...
│
├── Result/
│   ├── GOLI_batch_summary.csv
│   ├── LC09_...*GOLI_fire_mask.tif
│   └── ...
│
├── config.py
├── goli_detection.py
├── landsat_io.py
├── outputs.py
├── main.py
├── environment.yml
└── README.txt


六、运行
----------------------------------------------------------------------
1. 修改 config.py：

   ROOT_DIR
   OUTPUT_DIR

2. 安装依赖：

   # 用miniconda创建环境，然后用pip安装可以避免rasterio和GDAL等常见的依赖问题
   conda create -n active_fire_detection python=3.13.9 pip
   conda activate active_fire_detection
   # 需要进入requirement.txt文件所在目录
   python -m pip install -r requirements.txt    

3. 运行：

   python main.py


七、主要输出
----------------------------------------------------------------------
*_GOLI_fire_mask.tif

    0 = 非火点
    1 = 火点

*_GOLI_fire_type.tif

    0 = 非火点
    1 = 上下文检验通过的潜在火点
    2 = 公式（12）明确火点
    3 = 公式（13）相邻强火恢复像元

*_GOLI_fire_pixels.shp

    每个火点像元输出一个点，包含：
    经度、纬度、火点类型、Red、NIR、SWIR1、SWIR2、
    SWIR2/NIR、上下文窗口大小和成像日期。


八、调试输出
----------------------------------------------------------------------
当 SAVE_DEBUG_RASTERS = True 时，输出：

unambiguous_eq12
recovered_eq13
potential_eq14
potential_eq15
potential_raw
contextual_potential
water_background
background_valid
selected_window
swir2_nir_ratio
context_ratio_mean
context_ratio_std
context_swir2_mean
context_swir2_std


九、关于速度
----------------------------------------------------------------------
上下文检验只对潜在火点计算，并使用积分图快速统计窗口均值和标准差。
如果逐像元写出 Shapefile 较慢，可设置：

SAVE_FIRE_POINTS = False
