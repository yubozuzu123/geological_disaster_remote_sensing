传统阈值滑坡提取

利用年度合成影像计算指数特征，并基于阈值条件进行滑坡提取，输出指数特征图和阈值提取结果图。

一、环境准备

1.1 软件准备

(1). Python 3.10 及以上版本

(2). Anaconda

(3). VSCode、PyCharm 等 Python 编辑器


1.2 安装步骤

步骤1：创建 Python 环境

打开 Anaconda Prompt，依次执行：

(1). 创建环境

conda create -n landslide python=3.10

(2). 激活环境

conda activate landslide

步骤2：安装依赖包

激活环境后，执行：

安装核心依赖

pip install numpy gdal

依赖版本清单
1. Python          版本 >= 3.10
2. numpy           版本 >= 1.26
3. GDAL            版本 >= 3.6


二、数据说明

2.1 数据下载

本仓库使用 Git LFS（Large File Storage） 管理大文件（如 .tif 等）

若直接使用 GitHub “Download ZIP” 下载，大文件可能只会显示为约 1KB 的指针文件

(1). 环境准备

① 安装 Git

进入官网下载安装：https://git-scm.com/downloads

下载后一路点击 Next 安装即可（默认配置即可）

安装完成后验证：git --version

如果显示版本号（如 git version 2.xx），说明安装成功。

②安装 Git LFS

下载并安装：https://git-lfs.com/

安装完成后，在命令行执行：git lfs install

(2). 下载数据

克隆仓库：git clone git@github.com:yubozuzu123/geological_disaster_remote_sensing.git

进入项目目录：cd geological_disaster_remote_sensing

下载大文件：git lfs pull


2.2 原始数据目录结构

本部分代码所需数据位于 `../数据与模型文件/` 中，主要使用年度合成影像数据，其中，遥感年度影像合成及下载方式参考年度影像合成.docx，目录结构如下：

../数据与模型文件/
├── 2014年影像/

│   └── scale_along_median_2014nepal2.tif

├── 2015年影像/

│   └── scale_along_median_2015nepal2.tif

└── 2016年影像/

│   └── scale_along_median_2016nepal2.tif


2.3 数据用途说明

(1). `2014年影像/`、`2015年影像/`、`2016年影像/` 中存放年度合成影像数据，格式为 `.tif`

(2). 本部分代码主要读取年度合成影像，计算指数特征，并进行阈值判别


三、运行流程

总体流程：

激活环境后，先计算指数特征图，再根据阈值条件进行滑坡提取。

步骤：第一步：计算 NDVI、NDWI → 第二步：计算 MNDWI、NDBI → 第三步：阈值提取

1_指数特征图.py →  2_其它指数.py →  3_阈值提取.py


3.1 计算指数特征图（运行 `1_指数特征图.py`）

(1). 作用：根据年度合成影像计算 NDVI 和 NDWI 指数图

(2). 输入：年度合成影像 `.tif`

(3). 输出：NDVI 图、NDWI 图

(4). 可调参数（在 `1_指数特征图.py` 中修改）：

tif_file：输入的年度合成影像路径
	
output_dir：输出文件夹路径
	
output_ndvi：NDVI 输出文件路径
	
output_ndwi：NDWI 输出文件路径


3.2 计算其它指数（运行 `2_其它指数.py`）

(1). 作用：根据年度合成影像计算 MNDWI 和 NDBI 指数图

(2). 输入：年度合成影像 `.tif`

(3). 输出：MNDWI 图、NDBI 图

(4). 可调参数（在 `2_其它指数.py` 中修改）：

tif_file：输入的年度合成影像路径
	
output_dir：输出文件夹路径
	
output_mndwi：MNDWI 输出文件路径
	
output_ndbi：NDBI 输出文件路径
	
3.3 阈值提取（运行 `3_阈值提取.py`）

(1). 作用：根据多个指数阈值组合判断，输出滑坡提取结果图

(2). 输入：NDVI 图、NDWI 图、MNDWI 图、NDBI 图，以及可选的坡度图

(3). 输出：阈值提取结果图

(4). 可调参数（在 `3_阈值提取.py` 中修改）：

index_dir：指数文件所在文件夹路径

output_path：阈值提取结果输出路径

ndvi_lower：NDVI 下限

ndvi_upper：NDVI 上限

ndwi_upper：NDWI 上限

mndwi_upper：MNDWI 上限

ndbi_upper：NDBI 上限

slope_path：坡度文件路径

slope_threshold：坡度阈值

mask_nodata：是否忽略 NoData 区域

(5). 参数影响说明：

阈值越严格 → 误判可能减少，但漏判风险会增加

阈值越宽松 → 提取范围可能增大，但误判也可能增加
