基于机器学习的滑坡提取

利用年度合成影像和滑坡矢量数据生成样本与特征，训练随机森林模型，实现滑坡提取、测试集评价及结果后处理。

一、环境准备

1.1 软件准备

(1). Python 3.10 及以上版本，建议使用 Python 3.10

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

安装依赖

pip install numpy pandas gdal rasterio geopandas fiona shapely pyproj scikit-image scikit-learn scipy joblib matplotlib openpyxl

依赖版本清单

1. Python          版本 >= 3.10
2. numpy           版本 >= 1.26
3. pandas          版本 >= 2.0
4. GDAL            版本 >= 3.6
5. rasterio        版本 >= 1.3
6. geopandas       版本 >= 0.14
7. Fiona           版本 >= 1.9
8. shapely         版本 >= 2.0
9. pyproj          版本 >= 3.6
10. scikit-image   版本 >= 0.25
11. scikit-learn   版本 >= 1.7
12. scipy          版本 >= 1.15
13. joblib         版本 >= 1.5
14. matplotlib     版本 >= 3.10
15. openpyxl       版本 >= 3.1

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

本部分代码所需数据统一存放在 `../数据与模型文件/` 中，结构如下：

../数据与模型文件/

├── 2014年影像/

│   └── scale_along_median_2014nepal2.tif

├── 2015年影像/

│   ├── scale_along_median_2015nepal2.tif

│   └── shp/

│       ├── 2015.shp

│       ├── 2015.shx

│       ├── 2015.dbf

│       ├── 2015.prj

│       └── ...

├── 2016年影像/

│   ├── scale_along_median_2016nepal2.tif

│   └── shp/

│       ├── 2016.shp

│       ├── 2016.shx

│       ├── 2016.dbf

│       ├── 2016.prj

│       └── ...

├── 生成特征/

│   ├── 2015feature.csv

│   ├── train.xlsx

│   └── test.xlsx

└── 训练结果/

│   ├── best_rf_model2.pkl
	
│   └── best_rf_params2.json

2.3 各类数据说明

(1). `2014年影像/`

存放 2014 年年度合成影像，用于与 2015 年年度合成影像共同构建样本特征。

(2). `2015年影像/`

存放 2015 年年度合成影像及对应滑坡矢量数据（shp），其中影像用于特征提取，滑坡矢量数据（shp）用于生成滑坡样本点。

(3). `2016年影像/`

存放 2016 年年度合成影像及对应滑坡矢量数据（shp），其中影像可用于后续滑坡提取，滑坡矢量数据（shp）可用于结果对比与可视化。

(4). `生成特征/`

存放机器学习流程中的中间表格文件，包括特征表、训练集和测试集。

(5). `训练结果/`

存放训练完成后的模型文件及最优参数文件。

2.4 中间文件与结果文件说明

(1). `2015feature.csv`

特征提取后的样本特征表，是后续划分训练集和测试集的基础数据。

(2). `train.xlsx`

划分后的训练集样本表，用于模型训练。

(3). `test.xlsx`

划分后的测试集样本表，用于测试集评价。

(4). `best_rf_model2.pkl`

训练完成后的随机森林模型文件，可直接用于滑坡提取。

(5). `best_rf_params2.json`

模型训练得到的最优参数记录文件。

2.5 数据关系说明

(1). `2014年影像/` 与 `2015年影像/` 中的年度合成影像共同用于提取样本特征，并生成 `2015feature.csv`

(2). `2015年影像/` 中的滑坡矢量数据（shp）用于生成滑坡样本点

(3). `2015feature.csv` 进一步划分为 `train.xlsx` 和 `test.xlsx`

(4). `train.xlsx` 和 `test.xlsx` 用于随机森林模型训练及测试集评价

(5). 模型训练完成后，结果保存到 `训练结果/` 中

(6). `2015年影像/`、`2016年影像/` 中的年度合成影像以及训练好的模型文件可用于后续滑坡提取、结果合成、栅格转矢量和可视化

2.6 代码文件说明

(1). `0.0_point.py`

作用：根据滑坡矢量数据（shp）生成滑坡点和背景点。

(2). `0.1_feature.py`

作用：提取采样点在两期年度合成影像上的特征，生成特征表。

(3). `0.2_split.py`

作用：将特征样本划分为训练集和测试集。

(4). `1_RF_train.py`

作用：训练随机森林模型，并在测试集上进行评价。

(5). `2.0_predict.py`

作用：使用训练好的模型进行滑坡提取。

(6). `2.1_分块合成.py`

作用：将分块提取结果合成为完整栅格。

(7). `2.2_tif2shp.py`

作用：将提取结果栅格转换为矢量面，并按面积阈值筛选结果。

(8). `2.3_可视化结果.py`

作用：叠加年度合成影像、真实滑坡和提取结果，生成可视化结果。


三、运行流程

总体流程：

激活环境后，先生成样本点和特征，再划分训练测试集，随后训练随机森林模型，最后进行滑坡提取与结果后处理。


步骤：第一步：生成样本点 → 第二步：提取特征 → 第三步：划分训练测试集 → 第四步：训练模型并测试集评价 → 第五步：滑坡提取 → 第六步：分块合成 → 第七步：栅格转矢量 → 第八步：结果可视化

python 0.0_point.py → python 0.1_feature.py → python 0.2_split.py → python 1_RF_train.py → python 2.0_predict.py → python 2.1_分块合成.py → python 2.2_tif2shp.py → python 2.3_可视化结果.py

3.1 生成样本点（运行 `0.0_point.py`）

(1). 作用：根据滑坡矢量数据（shp）生成滑坡点和背景点，为后续特征提取提供采样位置。

(2). 输入：滑坡矢量数据（shp）

(3). 输出：采样点数据

(4). 可调参数（在 `0.0_point.py` 中修改）：

use_all_landslide_points：是否使用全部滑坡点
	
num_landslide_points：滑坡采样点数量
	
background_multiplier：背景点数量相对于滑坡点数量的倍数
	
random_state：随机种子

3.2 提取特征（运行 `0.1_feature.py`）

(1). 作用：提取采样点在两期年度合成影像上的特征，形成机器学习输入样本。

(2). 输入：采样点数据、两期年度合成影像

(3). 输出：`2015feature.csv`

(4). 可调参数（在 `0.1_feature.py` 中修改）：

image_2014_path：前一期年度合成影像路径
	
image_2015_path：后一期年度合成影像路径
	
window_size：纹理特征提取窗口大小
	
hog_window_size：HOG 特征提取窗口大小

3.3 划分训练测试集（运行 `0.2_split.py`）

(1). 作用：将特征样本划分为训练集和测试集。

(2). 输入：`2015feature.csv`

(3). 输出：`train.xlsx`、`test.xlsx`

(4). 可调参数（在 `0.2_split.py` 中修改）：

test_size：测试集比例
	
random_state：随机种子

3.4 训练模型并测试集评价（运行 `1_RF_train.py`）

(1). 作用：训练随机森林模型，并在测试集上进行效果评价。

(2). 输入：`train.xlsx`、`test.xlsx`

(3). 输出：`best_rf_model2.pkl`、`best_rf_params2.json` 以及测试集评价结果

(4). 可调参数（在 `1_RF_train.py` 中修改）：

param_grid：随机森林参数搜索范围

cv：交叉验证折数
	
scoring：模型评价指标
	
n_jobs：并行任务数

3.5 滑坡提取（运行 `2.0_predict.py`）

(1). 作用：使用训练好的随机森林模型对整幅影像进行分块提取，得到滑坡提取结果。

(2). 输入：两期年度合成影像、训练好的模型文件 `best_rf_model2.pkl`

(3). 输出：分块提取结果 `.tif`

(4). 可调参数（在 `2.0_predict.py` 中修改）：

block_size：分块大小
	
model_path：模型文件路径
	
image_2015_path：前一期年度合成影像路径
	
image_2016_path：后一期年度合成影像路径
	
output_dir：分块提取结果输出文件夹

3.6 分块合成（运行 `2.1_分块合成.py`）

(1). 作用：将多个分块提取结果合成为完整提取栅格。

(2). 输入：分块提取结果 `.tif`

(3). 输出：完整提取结果图 `.tif`

(4). 可调参数（在 `2.1_分块合成.py` 中修改）：

input_folder：分块结果文件夹路径
	
output_file：合成结果输出路径

3.7 栅格转矢量（运行 `2.2_tif2shp.py`）

(1). 作用：将提取栅格转换为滑坡矢量结果，并根据面积阈值筛选结果。

(2). 输入：完整提取结果图 `.tif`

(3). 输出：滑坡提取结果矢量数据（shp）

(4). 可调参数（在 `2.2_tif2shp.py` 中修改）：

input_raster：输入栅格路径
	
output_shp：输出矢量路径
	
min_area_threshold：最小面积阈值

3.8 结果可视化（运行 `2.3_可视化结果.py`）

(1). 作用：将提取结果与年度合成影像、真实滑坡范围进行叠加显示，生成可视化结果。

(2). 输入：年度合成影像、真实滑坡矢量数据（shp）、提取滑坡矢量数据（shp）

(3). 输出：可视化结果

(4). 可调参数（在 `2.3_可视化结果.py` 中修改）：

image_path：底图年度合成影像路径
	
true_shp：真实滑坡矢量路径

pred_shp：提取滑坡矢量路径
