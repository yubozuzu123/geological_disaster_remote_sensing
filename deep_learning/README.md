基于深度学习的滑坡提取

利用遥感影像和滑坡矢量样本生成训练数据集，训练 MobileUNet 模型实现滑坡自动提取，并输出滑坡提取结果及精度评价指标。

一、环境准备

1.1 软件准备

(1). Python   下载地址：https://www.python.org/downloads/ 

(2). Anaconda  下载地址：https://www.anaconda.com/download 

(3). VSCode、PyCharm等Python编辑器

1.2 安装步骤

步骤1：安装Anaconda

(1). 下载并安装Anaconda（Windows版本）

(2). 安装完成后，在开始菜单找到 Anaconda Prompt (anaconda3)

(3). 后续所有命令在 Anaconda Prompt 中运行

步骤2：创建Python环境

打开 Anaconda Prompt，依次执行：

(1). 创建名为 landslide 的环境，Python版本3.10

conda create -n landslide python=3.10

(2). 激活环境

conda activate landslide

步骤3：安装依赖包

激活环境后，依次执行：

(1). 安装PyTorch（GPU版本，需NVIDIA显卡）

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

(2). 安装其他依赖（指定版本避免兼容性问题）

pip install numpy==1.26.4 rasterio==1.3.11 geopandas==0.14.4 matplotlib==3.8.5 scikit-learn==1.5.1 Pillow==10.4.0 opencv-python==4.10.0.84

依赖版本清单
1. Python          版本 3.10.x 
2. PyTorch         版本 2.1.0 
3. torchvision     版本 0.16.0 
4. numpy           版本 1.26.4 
5. rasterio        版本 1.3.11 
6. geopandas       版本 0.14.4 
7. matplotlib      版本 3.8.5 
8. scikit-learn    版本 1.5.1 
9. Pillow          版本 10.4.0 
10. opencv-python  版本 4.10.0.84 

提示：
1. `numpy` 使用 `1.26.4`（PyTorch 2.1.0 不兼容 numpy 2.x）
2. GPU训练需要 NVIDIA 显卡和 CUDA 驱动（CUDA 11.8）
3. 如无GPU，改为安装CPU版本：`pip install torch==2.1.0 torchvision==0.16.0`


二、数据说明

2.1 数据下载

本仓库使用 Git LFS（Large File Storage） 管理大文件（如 .tif 等）

若直接使用 GitHub “Download ZIP” 下载，大文件可能只会显示为约 1KB 的指针文件。

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

原始数据位于 `./数据/` 目录：

./数据/

├── hd/                      # 横断山区

│   ├── 2008.tif            # 2008年遥感影像（TIF格式）

│   ├── 2008shp/2008.shp   # 2008年滑坡矢量（Shapefile格式）

│   ├── 2018.tif            # 2018年遥感影像

│   └── 2018shp/2018.shp   # 2018年滑坡矢量

└── WC/                      # 汶川地区

│   ├── WC.tif              # 遥感影像

│   └── SHP/WC.shp          # 滑坡矢量

其中，横断山区的遥感影像合成及下载方式参考年度影像合成.docx

2.3 生成数据集目录结构

运行 `generate_dataset.py` 后自动生成：

./landslide_data/

├── 横断_2008/

│   ├── train/images/        # 训练图像

│   ├── train/labels/       # 训练标签

│   ├── val/images/         # 验证图像

│   └── val/labels/         # 验证标签

├── 横断_2018/              # 同上

│   ├── train/images/        

│   ├── val/images/          

│   └── ...

└── 汶川/

│   ├── train/images/     
	
│   ├── val/images/       
	
│   └── ...

三、运行流程

总体流程：

激活先前创建的conda环境

conda activate landslide

步骤：第一步：生成数据集 → 第二步：训练模型 → 第三步：模型测试

python generate_dataset.py  → python train.py  → python test.py

3.1 生成数据集（运行generate_dataset.py）

(1). 作用：从原始TIF遥感影像和SHP矢量文件裁剪生成256×256的训练样本。

(2). 输出：`./landslide_data/` 目录

(3). 可调参数（在 `generate_dataset.py` 中修改）：

PATCH_SIZE：  256      #裁剪块大小（像素）

STRIDE：  128          #滑动步长，128表示50%重叠  

MIN_LS_RATIO：  0.005  #最小滑坡占比（过滤无效样本）

VAL_RATIO：  0.2       #训练/验证集划分比例

参数影响说明：

STRIDE： 越小 → 重叠越多 → 数据量越大 → 训练效果越好，但生成越慢

MIN_LS_RATIO： 越大 → 过滤越多 → 样本质量越高，但数量越少

3.2 训练模型（运行train.py）

(1). 作用：训练MobileUNet模型，自动保存最佳权重。

(2). 输出：`./checkpoints/best_final.pth`

(3). 可调参数：   

ochs：150      #训练轮数
	
lr：  0.0001   #学习率
	
bs：  8        #批次大小（显存不够时调小）

(4). 数据集配置（在 `train.py` 中修改）：

	train_dataset = LandslideDataset({
	    './landslide_data/横断_2008': ['train'],        # 仅用train部分
	    './landslide_data/横断_2018': ['train', 'val'], # 用全部
	    './landslide_data/汶川': ['train', 'val'],      # 用全部
	}, augment=True)

	val_dataset = LandslideDataset({
	    './landslide_data/横断_2008': ['val'],  # 仅验证
	}, augment=False)

3.3 模型测试（运行test.py）

(1). 作用：加载模型，在测试集上评估模型性能，生成可视化结果。

(2). 输出：`./可视化结果/` 目录下的结果图

终端打印：P（精确率）、R（召回率）、F1分数、IoU
	
(3). 可调参数（在 `test.py` 中修改）：

THRESHOLD:  0.5  # 提取阈值（>此值判定为滑坡）
	
NUM_VIZ:  10  # 生成可视化图的数量

阈值影响说明：
 
THRESHOLD： 越高 →漏提多、错提少

THRESHOLD： 越低 →错提多、漏提少

(4). 生成结果图：

左 ： 原始影像  #遥感图像原图

中 ： 标签      # 黑白二值图（白色=滑坡）

右 ： 提取结果  # 白色=正确，红色=错提，绿色=漏提
