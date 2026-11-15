# =========================================================================
# RU: Точка входа для запуска симуляции (Linux + Docker). Просто `make` покажет справку.
# EN: Entry point for the simulation (Linux + Docker). Plain `make` shows the help.
# =========================================================================

SHELL        := /bin/bash
.SHELLFLAGS  := -c
.ONESHELL:
.DEFAULT_GOAL := help
.PHONY: help shell start up image pull image-build onnx headless rebuild stop attach exec demo demo-stop gpu check

# RU: GPU=nvidia|amd|intel|cpu задаёт видеокарту вручную, иначе автоопределение.
# EN: GPU=nvidia|amd|intel|cpu forces the GPU, otherwise it is auto-detected.
GPU      ?=
# RU: HEADLESS=1 — без графики (не открывать окно). EN: HEADLESS=1 — no window.
HEADLESS ?= 0
# RU: CMD="..." — выполнить команду в контейнере и выйти (без TTY). EN: CMD="..." — run a command and exit (no TTY).
CMD      ?=
export GPU HEADLESS CMD VESTA_IMAGE

# -------------------------------------------------------------------------
# RU: Общая часть: проверка Docker, compose, определение видеокарты.
# EN: Common part: Docker check, compose detection, GPU detection.
# -------------------------------------------------------------------------
define PREAMBLE
set -uo pipefail
cd "$(CURDIR)"
source docker/image.sh || exit $$?
say() { echo -e "  $$*"; }

if ! command -v docker >/dev/null 2>&1; then
    say "RU: Docker не установлен."
    say "EN: Docker is not installed."
    say ""
    say "    https://docs.docker.com/engine/install/"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    say "RU: Docker не запущен или нет прав. Попробуйте:"
    say "EN: Docker is not running or permissions are missing. Try:"
    say ""
    say "    sudo systemctl start docker"
    say "    sudo usermod -aG docker \$$USER   # RU: затем перелогиньтесь"
    exit 1
fi

# RU: В новых версиях это плагин docker compose, в старых отдельный бинарник.
# EN: Newer versions ship a docker compose plugin, older ones a separate binary.
if docker compose version >/dev/null 2>&1; then
    DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DC="docker-compose"
else
    say "RU: Не найден docker compose. | EN: docker compose not found."
    exit 1
fi

detect_gpu() {
    # RU: NVIDIA годится, только если установлен container toolkit.
    # EN: NVIDIA only counts if the container toolkit is installed.
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        if docker info 2>/dev/null | grep -qi nvidia || [ -f /usr/bin/nvidia-container-runtime ]; then
            echo "nvidia"; return
        fi
        say "RU: NVIDIA найдена, но нет nvidia-container-toolkit." >&2
        say "EN: NVIDIA found, but nvidia-container-toolkit is missing." >&2
        say "    https://docs.nvidia.com/datacenter/cloud-native/" >&2
    fi
    if [ -d /dev/dri ]; then
        if lspci 2>/dev/null | grep -qiE 'vga|3d' && lspci 2>/dev/null | grep -qi 'amd\|radeon'; then
            echo "amd"; return
        fi
        if lspci 2>/dev/null | grep -qi 'intel.*graphics'; then
            echo "intel"; return
        fi
        echo "amd"; return   # RU: /dev/dri есть — пробуем Mesa | EN: /dev/dri exists, try Mesa
    fi
    echo "cpu"
}

GPU="$${GPU:-}"
[ -z "$$GPU" ] && GPU="$$(detect_gpu)"
OVERLAY="docker/compose.$$GPU.yml"
if [ ! -f "$$OVERLAY" ]; then
    say "RU: Нет конфигурации для '$$GPU' | EN: no configuration for '$$GPU'"
    exit 1
fi
COMPOSE="$$DC -f docker/docker-compose.yml -f $$OVERLAY"
endef

# RU: Общий выбор, скачивание и проверка образа без автоматической сборки.
# EN: Shared image selection, acquisition and validation without automatic builds.
define ENSURE_IMAGE
vesta_image_acquire || exit $$?
endef

# -------------------------------------------------------------------------
# RU: Подготовка к запуску: дисплей, каталоги сборки, образ (только если его нет), осиротевшие контейнеры.
# EN: Launch preparation: display, build dirs, image (only if missing), orphaned containers.
# -------------------------------------------------------------------------
define PREPARE
HEADLESS="$${HEADLESS:-0}"
if [ "$$HEADLESS" = 1 ]; then
    say "RU: Режим без графики | EN: headless mode"
elif [ -z "$${DISPLAY:-}" ]; then
    say "RU: DISPLAY не задан, перехожу в режим без графики."
    say "EN: DISPLAY is unset, falling back to headless."
    HEADLESS=1
elif command -v xhost >/dev/null 2>&1; then
    # RU: xhost открывает доступ к X-серверу для локальных контейнеров, иначе окно Webots не откроется.
    # EN: xhost grants X server access to local containers, otherwise the Webots window cannot open.
    xhost +local:docker >/dev/null 2>&1 || xhost +local:root >/dev/null 2>&1 || true
else
    say "RU: Нет xhost, окно может не открыться: apt install x11-xserver-utils"
    say "EN: xhost missing, the window may not open: apt install x11-xserver-utils"
fi

mkdir -p .build/build .build/install .build/log

# RU: Нейросети: чекпойнты (/models) и каталог Models/ из autoware.off-road (/offroad). Если каталога нет, монтируем пустой.
# EN: Networks: checkpoints (/models) and the Models/ directory of autoware.off-road (/offroad). If a directory is missing, mount an empty one.
export VESTA_MODELS="$${VESTA_MODELS:-$$HOME/autoware_data/ml_models_snapshot}"
export OFFROAD_REPO="$${OFFROAD_REPO:-$$HOME/Programming/autoware.off-road}"
for v in VESTA_MODELS OFFROAD_REPO; do
    if [ ! -d "$${!v}" ]; then
        say "RU: нет каталога $$v=$${!v}, восприятие работать не будет (задайте $$v=...)"
        say "EN: directory $$v=$${!v} is missing, perception will not work (set $$v=...)"
        mkdir -p ".build/missing_$$v"
        declare -gx "$$v=$(CURDIR)/.build/missing_$$v"
    fi
done

$(ENSURE_IMAGE)

# RU: Осиротевшие контейнеры от прерванных запусков держат порт 8080 и не дают подняться новому.
# EN: Orphaned containers from interrupted runs hold port 8080 and block the next start.
docker rm -f $$(docker ps -aq --filter "name=docker-sim-run") 2>/dev/null || true

if [ ! -f "$$VESTA_MODELS/onnx/seg.onnx" ]; then
    say "RU: нет $$VESTA_MODELS/onnx/*.onnx: сначала выполните make onnx (экспорт нейросетей для C++ узла perception_node)"
    say "EN: $$VESTA_MODELS/onnx/*.onnx is missing: run make onnx first (exports the networks for the C++ perception_node)"
fi

echo
echo "  ========================================================"
echo "   Simulation"
echo "  ========================================================"
echo "   GPU:      $$GPU"
echo "   Display:  $$([ "$$HEADLESS" = 1 ] && echo headless || echo "$${DISPLAY}")"
echo "  ========================================================"
echo
endef

# -------------------------------------------------------------------------
help:
	@cat <<'EOF'
	  Auto Vesta: симуляция Lada Vesta в Webots (+ восприятие и Nav2) | Webots simulation (+ perception and Nav2)

	  make                показать эту справку | show this help
	  make up             скачать образ (если нужно), поднять контейнер в фоне и зайти в bash | pull (if needed), start in background, open bash
	  make image          получить и проверить выбранный образ | acquire and validate the selected image
	  make pull           обновить выбранный образ из реестра | refresh the selected registry image
	  make image-build    собрать локальный vesta-sim:latest | build local vesta-sim:latest incrementally
	  make shell          одноразовый контейнер с bash (удаляется при выходе) | throw-away container with bash (removed on exit)
	  make headless       то же без графики | same without a window                [= make shell HEADLESS=1]
	  make attach         зайти в уже запущенный контейнер | open another shell in the running container   [= make exec]
	  make exec           то же, что make attach | same as make attach
	  make stop           остановить контейнеры | stop the containers
	  make onnx           экспорт нейросетей в ONNX для C++ узла восприятия (после смены чекпойнтов) | export the networks to ONNX for the C++ perception node
	  make rebuild        пересобрать локальный vesta-sim:latest с нуля и очистить .build | rebuild local vesta-sim:latest from scratch, clear .build
	  make demo           собрать и запустить Nav2 + восприятие (offroad_nav2), ждать готовности | build and start Nav2 + perception
	  make demo-stop      остановить демо | stop the demo
	  make gpu            показать определённую видеокарту | show the detected GPU
	  make check          проверить Docker и compose | check Docker and compose

	  Параметры | options:
	    GPU=nvidia|amd|intel|cpu   задать видеокарту вручную | force the GPU          (make up GPU=amd)
	    HEADLESS=1                 без графики | no window
	    CMD="команда"              выполнить команду в контейнере и выйти | run a command in the container and exit
	    ARGS="..."                 аргументы launch для make demo | launch arguments for make demo  (ARGS="gui:=true barrels_seed:=3")
	    VESTA_IMAGE=...             образ из реестра (можно @sha256:...) или локальный vesta-sim:latest | registry image (tag or digest) or local vesta-sim:latest

	  Прямой Compose | direct Compose: source docker/image.sh; docker compose -f docker/docker-compose.yml -f docker/compose.cpu.yml config

	  Внутри контейнера | inside the container:   ros2 launch webots_ros2_suv robot.launch.py
	  Демо Nav2 + восприятие | Nav2 + perception demo:   см. ros2_ws/src/offroad_nav2/docs/LAUNCH_GUIDE.md
	EOF

check:
	@$(PREAMBLE)
	say "RU: Docker и $$DC в порядке | EN: Docker and $$DC are fine"

gpu:
	@$(PREAMBLE)
	say "GPU: $$GPU"

# -------------------------------------------------------------------------
# RU: Запуск | EN: Launch
# -------------------------------------------------------------------------
# RU: up = скачать образ (если нужно), поднять контейнер vesta_sim в фоне, дождаться сборки воркспейса, зайти в bash.
#     Контейнер продолжает работать после выхода из bash: повторный `make up` просто заходит в него, `make stop` его останавливает.
# EN: up = pull the image (if needed), start container vesta_sim in the background, wait for the workspace build, open bash.
#     The container keeps running after you leave bash: another `make up` just re-enters it, `make stop` stops it.
up:
	@$(PREAMBLE)
	if [ -n "$$(docker ps -q -f name=^webots_vesta$$)" ]; then
	    say "RU: Уже запущен одноразовый контейнер webots_vesta (make shell): make attach или make stop"
	    say "EN: a throw-away container webots_vesta (make shell) is already running: make attach or make stop"
	    exit 1
	fi
	$(PREPARE)
	# RU: up -d идемпотентен: ничего не делает, если контейнер актуален, и пересоздаёт его, если изменился образ или конфигурация.
	# EN: up -d is idempotent: it does nothing if the container is current and recreates it if the image or configuration changed.
	$$COMPOSE up --no-build --pull never -d sim || exit 1
	# RU: Точка входа может собирать воркспейс при старте контейнера: ждём её, чтобы два colcon не мешали друг другу.
	# EN: The entrypoint may build the workspace at container start: wait for it so two colcon runs do not collide.
	n=0
	until docker exec vesta_sim test -f /tmp/.entrypoint_ready 2>/dev/null; do
	    if [ -z "$$(docker ps -q -f name=^vesta_sim$$)" ]; then say "RU: контейнер завершился, см. docker logs vesta_sim | EN: the container exited, see docker logs vesta_sim"; exit 1; fi
	    [ $$n -eq 0 ] && say "RU: Жду сборку воркспейса при старте контейнера | EN: waiting for the workspace build at container start"
	    n=$$((n+1))
	    if [ $$n -gt 360 ]; then say "RU: Сборка не завершилась, см. docker logs vesta_sim | EN: the build did not finish, see docker logs vesta_sim"; exit 1; fi
	    sleep 5
	done
	# RU: Каждый вход: colcon build (инкрементальный, symlink-install) и source install/setup.bash.
	# EN: On every entry: colcon build (incremental, symlink-install) and source install/setup.bash.
	BUILD='source /opt/ros/humble/setup.bash; cd /ros2_ws; echo "[make up] colcon build --symlink-install"; colcon build --symlink-install 2>&1 | tail -n 6; source /ros2_ws/install/setup.bash'
	if [ -n "$${CMD:-}" ]; then
	    exec docker exec -i vesta_sim bash -lc "$$BUILD; $$CMD"
	fi
	exec docker exec -it vesta_sim bash -lc "$$BUILD; exec bash"

# RU: Получить образ без запуска контейнера. EN: Acquire the image without starting a container.
image:
	@$(PREAMBLE)
	$(ENSURE_IMAGE)

pull:
	@$(PREAMBLE)
	vesta_image_acquire pull || exit $$?

# RU: Локальная сборка всегда имеет отдельный тег. EN: Local builds always use their own tag.
image-build:
	@$(PREAMBLE)
	VESTA_IMAGE=vesta-sim:latest $$COMPOSE build sim || exit $$?

shell start:
	@$(PREAMBLE)
	if [ -n "$$(docker ps -q -f name=^vesta_sim$$)" ]; then
	    say "RU: Уже запущен контейнер vesta_sim (make up), порт 8080 занят: make attach или make stop"
	    say "EN: container vesta_sim (make up) is already running and holds port 8080: make attach or make stop"
	    exit 1
	fi
	$(PREPARE)
	if [ -n "$${CMD:-}" ]; then
	    exec $$COMPOSE run --pull never -T --rm --service-ports sim bash -lc "$$CMD"
	fi
	exec $$COMPOSE run --pull never --name webots_vesta --rm --service-ports sim bash

headless:
	@$(MAKE) --no-print-directory shell HEADLESS=1

# RU: Экспорт трёх нейросетей (чекпойнты *.pth -> ONNX fp16) для C++ узла восприятия. Нужен после обновления чекпойнтов.
# EN: Export the three networks (checkpoints *.pth -> fp16 ONNX) for the C++ perception node. Needed after the checkpoints change.
onnx:
	@$(PREAMBLE)
	$(PREPARE)
	mkdir -p "$$VESTA_MODELS/onnx"
	exec $$COMPOSE run --pull never -T --rm -v "$$VESTA_MODELS/onnx:/models/onnx:rw" sim bash -lc 'cd /ros2_ws/src/offroad_nav2 && python3 tools/export_onnx.py --models-root /offroad/Models --ckpt-dir /models --out /models/onnx --check --image docs/results/domain_gap/06_rgb.png 2>&1 | grep -E "exported|agrees|Error|error"'

rebuild:
	@$(PREAMBLE)
	say "RU: Пересборка образа с нуля | EN: rebuilding image from scratch"
	VESTA_IMAGE=vesta-sim:latest $$COMPOSE build --no-cache sim || exit $$?
	# RU: Артефакты сборки colcon лежат на хосте в .build (root-владелец): чистим через контейнер.
	# EN: colcon artifacts live on the host in .build (root-owned): clean them through a container.
	if [ -d .build ]; then
	    docker run --pull=never --rm -v "$(CURDIR)/.build:/b" vesta-sim:latest bash -c 'rm -rf /b/build /b/install /b/log' 2>/dev/null || rm -rf .build
	fi
	mkdir -p .build/build .build/install .build/log
	say "RU: Готово. Запуск: make up | EN: Done. Start with: make up"

stop:
	@$(PREAMBLE)
	say "RU: Останавливаю | EN: stopping"
	$$COMPOSE down --remove-orphans   # RU: заодно убирает контейнеры удалённых сервисов (CARLA, Autoware) | EN: also removes containers of removed services (CARLA, Autoware)

attach exec:
	@c=$$(docker ps --format '{{.Names}}' | grep -E '^(vesta_sim|webots_vesta)$$' | head -1)
	if [ -n "$$c" ]; then exec docker exec -it $$c bash -l; else echo "  RU: контейнер не запущен (make up) | EN: no container is running (make up)"; exit 1; fi

# -------------------------------------------------------------------------
# RU: Демо Nav2 + восприятие | EN: Nav2 + perception demo
# -------------------------------------------------------------------------
demo: image
	@ros2_ws/src/offroad_nav2/docker/restart.sh $(ARGS)

demo-stop:
	@if [ -n "$$(docker ps -aq -f name=^nav_sim$$)" ]; then docker stop nav_sim >/dev/null; docker rm nav_sim >/dev/null; echo "  nav_sim stopped"; else echo "  nav_sim is not running"; fi
