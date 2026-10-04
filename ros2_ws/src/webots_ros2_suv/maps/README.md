# Карты

`nav2.launch.py` по умолчанию грузит `maps/robocross.yaml` (переопределяется
аргументом `map:=/abs/path.yaml`).

## Как записать карту

Карту надо записывать, стартуя из той же точки спавна в `robocross.wbt`: тогда
начало карты совпадает с началом `odom` (первый GPS-фикс), и `map -> odom` —
единичный статический TF. Если точка спавна другая — задайте смещение
аргументами `map_odom_x/y/yaw` у `nav2.launch.py`.

```bash
ros2 launch webots_ros2_suv robot.launch.py
ros2 launch webots_ros2_suv mapping.launch.py   # pointcloud_to_laserscan + slam_toolbox
# проехать трассу (teleop), затем:
ros2 service call /map_saver/save_map nav2_msgs/srv/SaveMap \
    "{map_topic: map, map_url: $PWD/src/webots_ros2_suv/maps/robocross, image_format: pgm,
      map_mode: trinary, free_thresh: 0.25, occupied_thresh: 0.65}"
```

После сохранения пересоберите пакет (`colcon build`), чтобы карта попала в share/.
