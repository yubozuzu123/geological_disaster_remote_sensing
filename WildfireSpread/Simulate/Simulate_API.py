import requests

simulation_url = "http://127.0.0.1:5000/fire_spread_simulation"

simulation_data = {
    "geo_coords": ["""{
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
    }"""],                                                                         # 火场范围（m/s），字符串
    "wind_speed": 21,                                                              # 火场风速（m/s），浮点型，数值
    "wind_direction": 180,                                                         # 火场风向（°），浮点型，数值
    "air_temperature": 30,                                                         # 火场温度（℃），浮点型，数值
    "relative_humidity": 30,                                                       # 火场湿度（%），浮点型，数值
    "Isolation_Belts": """{
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
    }""",                                                      # 防火隔离带，字符串
    "window_shape": [5.0, 5.0],                                                    # 火场窗口大小（km），浮点型，list
    "spread_time": 24,                                                              # 预设蔓延时间（小时），浮点型，数值
    "refresh_interval": 15.0,                                                      # 矢量产出的间隔时间（小时），浮点型，数值
    "fire_area_id": 3,                                                             # 蔓延阶段编号，整型，数值
    "fire_extent_id": 0,                                                           # 火灾事件编号，整型，数值
    "source_folder": "D:/CoursePPT/Wildfire/WildfireSpread/Source/",                         # tif数据存放位置，字符串
    "target_folder": "D:/CoursePPT/Wildfire/WildfireSpread/Target/"                          # geojson数据的输出位置，字符串
}

simulation_response = requests.post(simulation_url, json=simulation_data)
print("Status Code:", simulation_response.status_code)
print("Response JSON:", simulation_response.text)
