WildfireSpread 林火蔓延模拟程序
======================================================================

一、程序说明
----------------------------------------------------------------------

本程序基于元胞自动机思想进行林火蔓延模拟。

程序综合考虑地形坡度、风速和风向、土地覆盖类型、空气温度、相对湿度和防火隔离带等因素，通过逐时间步更新元胞热量和燃烧状态，实现林火蔓延过程模拟，并按照指定时间间隔输出火场范围 GeoJSON 文件。


二、当前实现的主要内容
----------------------------------------------------------------------

1. 坡度影响

根据 DEM 计算相邻元胞之间的坡度，并通过坡度权重体现上山火蔓延加速效应。

坡度越大，火势沿上坡方向传播越快。


2. 风场影响

根据输入的风速和风向，计算火势向不同方向传播时的风场影响权重。

顺风方向促进火势传播，逆风方向抑制火势传播。


3. 土地覆盖影响

land_cover.tif 中的像元值作为不同土地覆盖类型的林火传播权重。

不同土地覆盖类型可设置不同的林火传播能力。


4. 温湿度影响

空气温度和相对湿度共同修正林火传播环境条件。

温度升高可增强燃料预热作用，相对湿度升高则抑制燃烧和传播。


5. 热量累积与衰减

燃烧元胞向周围未燃元胞传递热量。

每个时间步中，热量按照设定的 cooling_factor 进行衰减。

当未燃元胞累积热量达到 ignition_threshold 后，该元胞被点燃。


6. 元胞燃烧状态

模拟网格中包含三种主要状态：

0 = 未燃烧
1 = 燃烧中
2 = 已燃尽


7. 防火隔离带

支持通过 GeoJSON LineString 输入防火隔离带。

程序将防火隔离带转换到 DEM 坐标系并栅格化，用于限制林火穿越隔离带传播。


8. 动态模拟窗口

根据输入火源范围自动确定模拟区域。

当火源范围较小时，使用默认最小模拟窗口；当火源范围较大时，根据火场范围自动扩大模拟窗口。


9. 动态更新

支持在模拟过程中根据新的火场范围、风速、风向等参数重新启动当前火灾事件的模拟，实现火场状态更新。


三、输入数据
----------------------------------------------------------------------

程序主要使用以下两类栅格数据：

1. DEM数据

文件名：dem.tif

用于计算相邻元胞之间的高程差和坡度影响。


2. 土地覆盖数据

文件名：land_cover.tif

每个像元值代表对应土地覆盖类型的林火传播权重。

dem.tif 与 land_cover.tif 应具有一致的：

坐标系、分辨率、空间范围、行列数、像元网格


四、模拟参数
----------------------------------------------------------------------

Simulate_API.py 中主要设置以下参数：

1. geo_coords: 初始火源范围。使用 GeoJSON Polygon 或 FeatureCollection 输入。

2. wind_speed: 风速，单位为 m/s。

3. wind_direction: 风向，单位为 °。

4. air_temperature: 空气温度，单位为 ℃。

5. relative_humidity: 相对湿度，单位为 %。

6. Isolation_Belts: 防火隔离带。使用 GeoJSON LineString 或 MultiLineString 输入。

7. window_shape: 默认模拟窗口大小。

8. spread_time: 预设模拟时长，单位为小时。

9. refresh_interval: 模拟结果输出的时间间隔，单位为分钟。

10. fire_area_id: 火灾事件编号。

11. fire_extent_id: 当前火场模拟阶段编号。

12. source_folder: DEM 和土地覆盖数据所在目录。

13. target_folder: 林火蔓延结果输出目录。


五、GeoJSON 输入格式
----------------------------------------------------------------------

1. 火源范围

火源使用 GeoJSON Polygon 格式，例如:

{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "AreaName": 1
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [115.640262, 39.744646],
            [115.640496, 39.744649],
            [115.640499, 39.744469],
            [115.640266, 39.744466],
            [115.640262, 39.744646]
          ]
        ]
      }
    }
  ]
}


2. 防火隔离带

防火隔离带使用 GeoJSON LineString 格式，例如:

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
          [115.63735, 39.74681],
          [115.63937, 39.74681],
          [115.64139, 39.74681],
          [115.64341, 39.74681]
        ]
      }
    }
  ]
}

注意: GeoJSON 经纬度坐标顺序为: [经度, 纬度]


六、文件夹结构
----------------------------------------------------------------------

WildfireSpread/
    Simulate/
        FireCore.py
        FireSpread.py
        FireUtils.py
        Simulate_API.py
        Update_API.py
        config.env
        requirements.txt
        README.txt
    Source/
        dem.tif
        land_cover.tif
    Target/


各文件功能:

FireCore.py: 林火蔓延核心模拟程序，包括坡度、风场、土地覆盖、温湿度、热量累积和元胞状态更新。

FireUtils.py: 公共工具函数，包括坡度权重、风场权重、方向计算、GeoJSON 解析和动态模拟窗口计算。

FireSpread.py: Flask 服务程序，提供林火初始模拟和动态更新两个 API。

Simulate_API.py: 林火初始模拟测试程序。用于设置火源、风速、风向、温湿度、防火隔离带等参数，并向 FireSpread.py 发送初始模拟请求。

Update_API.py: 林火动态更新测试程序。用于在已有模拟任务基础上更新火场范围及相关参数。

config.env: Flask 服务配置文件。

requirements.txt: Python 第三方依赖安装文件。

README.txt: 程序使用说明文件。

Source/: 输入数据目录。

Target/: 模拟结果输出目录。


七、环境安装
----------------------------------------------------------------------

推荐使用 Miniconda 创建独立 Python 环境，再使用 pip 安装第三方依赖。

1. 创建环境：

conda create -n wildfire python=3.13.9 pip


2. 激活环境：

conda activate wildfire


3. 安装依赖

进入 requirements.txt 所在的 Simulate 文件夹，在已激活的 wildfire 环境中执行：

python -m pip install -r requirements.txt


requirements.txt 主要包括：

numpy、pandas、rasterio、geopandas、shapely、pyogrio、Flask、python-dotenv、requests、pyinstaller


八、程序运行与测试
----------------------------------------------------------------------

1. 准备输入数据

确保以下文件位于 Source 文件夹：

dem.tif
land_cover.tif


2. 设置数据路径

根据实际文件夹位置修改 Simulate_API.py 中的：

source_folder
target_folder


3. PyCharm 测试运行 （先启动服务才能进行蔓延模拟）

使用 PyCharm 打开工程，并选择已经安装好依赖的 wildfire Python 环境。

由于程序采用 Flask 服务方式运行，测试时需要同时运行服务端和客户端，因此需要两个独立的运行终端。

终端1：启动 Flask 服务（先运行）

运行：

FireSpread.py

保持该程序持续运行，用于监听并接收林火蔓延模拟请求。


终端2：发送模拟请求（后运行）

运行：

Simulate_API.py

向已经启动的 Flask 服务发送初始林火蔓延模拟请求。

如果程序能够正常接收请求，并在 Target 文件夹生成模拟结果，则说明环境配置和程序运行正常。

如需测试动态更新，可在终端2进一步运行：

Update_API.py


九、程序打包与部署
----------------------------------------------------------------------

程序测试正常后，可使用 PyInstaller 将 Flask 服务入口程序 FireSpread.py 打包为独立 EXE。

1. 激活 wildfire 环境

打开 Conda 终端，执行：

conda activate wildfire


2. 进入 Simulate 文件夹

在 Conda 终端进入 WildfireSpread 工程中的 Simulate 文件夹。


3. 执行打包命令

确认当前终端已经处于 wildfire 环境，并且当前目录为 Simulate 后，执行：

pyinstaller --onefile --clean --noconfirm --collect-all rasterio --collect-all pyogrio --collect-all geopandas --collect-all shapely --collect-all pyproj --hidden-import pyogrio._geometry FireSpread.py


4. 获取 EXE

打包完成后，新生成的 FireSpread.exe 位于 dist 文件夹。

将 FireSpread.exe 从 dist 文件夹复制到项目 Simulate 文件夹，与 config.env 保持同级，使 FireSpread.exe 能够正常读取配置文件。


5. 运行打包后的程序

打包后的程序仍采用“服务端 + 客户端”的运行方式，因此测试时同样需要两个运行终端。

终端1：启动 Flask 服务

双击运行：

FireSpread.exe

保持程序窗口运行，用于监听并接收林火蔓延模拟请求。


终端2：发送模拟请求

运行：

Simulate_API.py

向 FireSpread.exe 启动的 Flask 服务发送模拟请求。

如需进行动态更新，可运行：

Update_API.py

注意：FireSpread.exe 已包含 Python 解释器及服务端主要程序依赖；如果客户端仍通过 Simulate_API.py 或 Update_API.py 运行，则客户端需要可用的 Python 环境及 requests 等依赖。


十、API
----------------------------------------------------------------------

程序提供两个主要接口：

初始模拟：/fire_spread_simulation

动态更新：/fire_spread_update

默认服务地址：http://127.0.0.1:5000


十一、主要输出
----------------------------------------------------------------------

程序按照设定的 refresh_interval 输出阶段性火场模拟结果。

结果保存在：Target/

输出文件示例：

3-1.geojson
3-1.tif
3-2.geojson
3-2.tif
3-3.geojson
3-3.tif
...

GeoJSON 文件表示对应模拟阶段的林火蔓延范围，GeoTIFF 文件表示对应阶段的火场状态栅格。


十二、注意事项
----------------------------------------------------------------------

1. dem.tif 与 land_cover.tif 必须具有一致的坐标系、分辨率、空间范围和像元网格。

2. 火源和防火隔离带使用 GeoJSON 输入时，经纬度坐标顺序必须为：[经度, 纬度]

3. 火源 GeoJSON 在程序内部会自动转换到 DEM 坐标系。

4. 防火隔离带 GeoJSON 也会自动转换到 DEM 坐标系后进行栅格化。

5. Source 和 Target 路径应正确设置，并确保程序具有 Target 目录的写入权限。

6. 使用源代码运行时，FireSpread.py 必须保持运行；使用打包程序运行时，FireSpread.exe 必须保持运行，Simulate_API.py 和 Update_API.py 才能正常发送模拟请求。

7. 当前程序主要用于林火蔓延算法试验、参数分析和教学演示。
