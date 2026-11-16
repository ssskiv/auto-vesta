# Auto Vesta — sim

Симуляция автономной Lada Vesta на Webots + ROS 2 (Humble). Ветка `sim`
содержит только симулятор (мир, модель машины, драйвер, Docker-окружение) —
без железа и остального автопилота.

## Требования

- Docker + Docker Compose (с правами текущего пользователя, см. `docker info`)
- Linux: X11 и `xhost` (пакет `x11-xserver-utils`) для вывода окна Webots.
  Windows: WSL2 с WSLg (графика идёт через встроенный Wayland/X11 сокет).
- Опционально GPU: NVIDIA (с `nvidia-container-toolkit`), AMD/Intel (Mesa,
  `/dev/dri`) — иначе используется программный рендер на CPU.

## Быстрый старт

```bash
make                       # справка
make image                 # проверить кэш или скачать готовый образ
make up                    # контейнер, инкрементальная сборка, bash
```

Внутри контейнера:

```bash
ros2 launch webots_ros2_suv robot.launch.py
```

Общий образ среды — `ghcr.io/ssskiv/vesta-sim:nav2ai` (`linux/amd64`);
его имя не выбирает ветку исходников. Ветка `sim` собирает только симулятор.
Исходники монтируются из `ros2_ws/src`, сборка хранится в `.build/{build,install,log}`.
Чекпойнты нейросетей для симулятора не нужны.

`make image` скачивает образ только при отсутствии в кэше, `make pull` обновляет
его явно. Для закрепления версии используйте `VESTA_IMAGE=ghcr.io/ssskiv/vesta-sim@sha256:...`.
Локальная сборка: `make image-build`, запуск: `VESTA_IMAGE=vesta-sim:latest make up`.
`make rebuild` собирает локальный образ без кэша и очищает `.build`.

| Команда | Назначение |
|---|---|
| `make up` | постоянный контейнер `vesta_sim`, сборка и оболочка |
| `make shell` | временный контейнер, удаляется при выходе |
| `make attach` / `make exec` | ещё одна оболочка в работающем контейнере |
| `make stop` | остановить контейнеры Compose |
| `make check` / `make gpu` | проверить Docker / показать выбранную GPU |
| `make headless` | временная оболочка без настройки графики |

Параметры: `GPU=nvidia|amd|intel|cpu`, `HEADLESS=1`, `CMD="команда"`.
Даже с `gui:=false` камеры Webots требуют X display и OpenGL.
Для прямого Compose: `source docker/image.sh`, затем базовый файл и один GPU overlay.
Прежние `scripts/start.sh` и Windows-скрипты сохранены; основной проверенный путь — Linux/Make.

## Проверки

```bash
bash docker/test_image.sh                            # контракты CLI без Docker и сети
bash docker/smoke-image.sh ghcr.io/ssskiv/vesta-sim:nav2ai
```

В контейнере после сборки и `source install/setup.bash`:

```bash
python3 -m pytest -q src/webots_ros2_suv/test/test_referee.py src/webots_ros2_suv/test/test_runtime_assets.py
python3 src/webots_ros2_suv/tools/vendor_webots_assets.py --check
python3 src/webots_ros2_suv/tools/check_barrels_proto.py --seeds 100  # нужен Node.js
```

Для проверки работающего симулятора запустите launch с логом:
`ros2 launch webots_ros2_suv robot.launch.py gui:=false > /tmp/sim.log 2>&1`.
В другой оболочке того же контейнера выполните:

```bash
python3 src/webots_ros2_suv/tools/smoke_sim.py
python3 src/webots_ros2_suv/tools/smoke_sim.py --gate --log /tmp/sim.log
```

Смоук проверяет данные датчиков, `/clock`, `/odom` и TF; гейт дополнительно
меняет раскладку бочек трижды, телепортирует машину и проверяет движение
вперёд/назад. Выполняйте гейт в отдельном тестовом запуске: он меняет состояние мира.

## Что внутри

- `ros2_ws/src/webots_ros2_suv` — ROS 2 пакет: мир `worlds/robocross.wbt`,
  PROTO-модель машины (`LadaVesta*.proto`), драйвер `vesta_driver.py`,
  launch-файл `launch/robot.launch.py`.
- `models/vesta` — 3D-модель кузова (Blender/OBJ/Collada) и текстуры.
- `docker/` — `Dockerfile` (ROS humble + Webots) и compose-оверлеи по GPU.
- `Makefile` — основной запуск; `scripts/` — совместимые прежние точки входа.
- `oscc`, `ros2_ws/src/ros2cco` — git-подмодули (сторонние зависимости).
  Загружаются командой:
  ```bash
  git submodule update --init --recursive
  ```

## Управление и топики

- `vesta_driver.py` принимает `geometry_msgs/Twist` на `/cmd_vel` (режим
  скорости, подходит для `teleop_twist_keyboard` и Nav2) либо сырые
  throttle/brake команды (режим torque).
- Публикует: `/imu/data`, `/lidar`, `/radar`, `/joint_states`, `/odom`,
  RGB `/camera/image_raw/*`, стерео `/stereo_raw/{left,right}/*`,
  TF `odom -> base_link`.
- Протестировано телеуправление на Windows с DualShock-геймпадом.

## Перенос общего из `nav2ai` в `sim`

Сравнение выполнено с `nav2ai` на `01acdfc`; основа `sim` — `4a5caf5`.

| Область | Результат |
|---|---|
| Модели, текстуры и пути | Локальный комплект Webots, обновлённые текстуры знаков/ворот, runtime-ссылки на контроллеры и плагины текущего `WEBOTS_HOME` |
| Мир и положение | Камера обзора следует за `vesta`; геометрия трассы, spawn, ворота и сдвиг карты `(2.2, -1.6)` сохранены |
| Сенсоры | Добавлены стереопара с базой 0.20 м, optical frames и `base_footprint`; исправлен GPS mount; RGB и лидар `sim` сохранены |
| Драйвер | Время Webots для драйверных сообщений, `/odom`, `/vesta/teleport`, отключаемый odom TF, опциональное состояние светофора, исправление руля при заднем ходе |
| Бочки и судья | Уже совпадали в ветках; генератор, PROTO и правила не менялись |
| Общая среда | Make, получение/проверка готового образа, правильный mount исходников, ожидание entrypoint, последовательная первая сборка при RAM <24 ГиБ, NVIDIA PRIME |
| Алгоритмы | `offroad_nav2`, нейросети, маршруты и Nav2-конфигурация RViz не перенесены; общий опубликованный образ содержит их зависимости |

Проверено 2026-11-16 на NVIDIA GeForce RTX 5070 Ti Laptop GPU,
OpenGL 4.6 NVIDIA 595.99.02, с отдельной свежей сборкой и изолированной сетью:

- Сборка пакета: успешно; `colcon test`: 12 тестов, 0 ошибок/падений.
- Контракты получения образа и CPU dependency smoke готового образа: успешно.
- Runtime smoke: реальные сообщения RGB, стерео, IMU, лидара, радара, `/clock`, `/odom`, TF.
- Runtime gate: seed `7 → 19 → 7`, телепорт, движение вперёд/назад, стабильная высота машины.
- Генератор: 100 extended + 5 basic + 1 максимальная парная раскладка, все проверки пройдены.
- Vendored assets: 0 сетевых ссылок; shell/Python syntax и `git diff --check`: успешно.

Логи локального прогона: `.build/sim-transfer-evidence/` (не входят в git).
Проверка использовала имеющиеся локальные крупные ассеты; Windows-скрипт
обновлён для выбора готового образа, но на Windows не запускался.

## Известные ограничения

У топиков камер время берётся из реального мира; драйверные IMU/лидар/радар/одометрия используют время Webots. Если нужна будет синхронизация, надо будет сделать прослойку, которая будет менять время на время симуляции.

На винде не тестировано. Если будут проблемы - создавайте pull requests, а лучше - поставьте ubuntu и установите докер(`rootless`) ;)

По любым вопросам к автору репозитория, в профиле контакты есть.

Ветка `sim` останется как ветка, в которой хранится **шаблон**, на котором можно строить свои алгоритмы. Nav2, PyTorch и все такое добавляйте по необходимости, но в этой ветке их нет и не будет.

Возможно есть неактуальные комментарии от клода в коде, так что больше внимания уделяйте самому коду, чем комментам. После клода приходилось переписывать часть кода, тк он убивает производительность, тыкая костыли туда, где они не нужны.

Если нужен пример реализации другой машины на похожей карте: [вот репозиторий, на который ориентировался автор этого репозитория](https://github.com/ulstu/robocross.virtual)

Ветка `main` не будет обновляться, пока алгоритм не будет отлажен в симуляции. Основной смысл этого репозитория - автономная езда реального автомобиля Lada Vesta Cross.
