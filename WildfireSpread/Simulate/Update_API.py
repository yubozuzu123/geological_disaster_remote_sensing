import requests


update_url = "http://127.0.0.1:5000/fire_spread_update"

update_data = {
    "geo_coords": ["""
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
                [115.639594, 39.732764],
                [115.639599, 39.732764],
                [115.639599, 39.732769],
                [115.639594, 39.732769],
                [115.639594, 39.732764]
              ]
            ]
          }
        },
        {
          "type": "Feature",
          "properties": {
            "AreaName": 2
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [
              [
                [115.649594, 39.742764],
                [115.649599, 39.742764],
                [115.649599, 39.742769],
                [115.649594, 39.742769],
                [115.649594, 39.742764]
              ]
            ]
          }
        }
      ]
    }
    """],                  # 火场范围（m/s），字符串
    "wind_speed": 10,                     # 火场风速（m/s），浮点型，数值
    "wind_direction": 315,                # 火场风向（°），浮点型，数值
    "fire_area_id": 3,                    # 火灾事件名称，整型，数值
    "fire_extent_id": 25                  # 火灾模拟步数，整型，数值
}

update_response = requests.post(update_url, json=update_data)
print("Status Code:", update_response.status_code)
print("Response JSON:", update_response.text)