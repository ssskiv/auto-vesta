# =========================================================================
# RU: Точка входа для запуска симуляции (Linux + Docker). Просто `make` покажет справку.
# EN: Entry point for the simulation (Linux + Docker). Plain `make` shows the help.
# =========================================================================

SHELL        := /bin/bash
.SHELLFLAGS  := -c
.ONESHELL:
.DEFAULT_GOAL := help
.PHONY: help shell start headless rebuild stop attach demo demo-stop gpu check

# RU: GPU=nvidia|amd|intel|cpu задаёт видеокарту вручную, иначе автоопределение.
# EN: GPU=nvidia|amd|intel|cpu forces the GPU, otherwise it is auto-detected.
GPU      ?=
# RU: HEADLESS=1 — без графики (не открывать окно). EN: HEADLESS=1 — no window.
HEADLESS ?= 0
# RU: CMD="..." — выполнить команду в контейнере и выйти (без TTY). EN: CMD="..." — run a command and exit (no TTY).
CMD      ?=
export GPU HEADLESS CMD

# -------------------------------------------------------------------------
# RU: Общая часть: проверка Docker, compose, определение видеокарты.
# EN: Common part: Docker check, compose detection, GPU detection.
# -------------------------------------------------------------------------
define PREAMBLE
set -uo pipefail
cd "$(CURDIR)"
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

# -------------------------------------------------------------------------
help:
	@cat <<'EOF'
	  Auto Vesta: симуляция Lada Vesta в Webots (+ восприятие и Nav2) | Webots simulation (+ perception and Nav2)

	  make                запустить контейнер (оболочка внутри) | start the container (shell inside)   [= make shell]
	  make shell          то же самое | same
	  make headless       без графики | no window                                [= make shell HEADLESS=1]
	  make rebuild        пересобрать образ с нуля и очистить .build | rebuild the image from scratch, clear .build
	  make stop           остановить контейнеры | stop the containers
	  make attach         зайти в уже запущенный контейнер | open another shell in the running container
	  make demo           собрать и запустить Nav2 + восприятие (offroad_nav2), ждать готовности | build and start Nav2 + perception
	  make demo-stop      остановить демо | stop the demo
	  make gpu            показать определённую видеокарту | show the detected GPU
	  make check          проверить Docker и compose | check Docker and compose

	  Параметры | options:
	    GPU=nvidia|amd|intel|cpu   задать видеокарту вручную | force the GPU          (make shell GPU=amd)
	    HEADLESS=1                 без графики | no window
	    CMD="команда"              выполнить команду в контейнере и выйти | run a command in the container and exit
	    ARGS="..."                 аргументы launch для make demo | launch arguments for make demo  (ARGS="gui:=true barrels_seed:=3")

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
shell start:
	@$(PREAMBLE)
	HEADLESS="$${HEADLESS:-0}"
	# RU: Доступ к дисплею | EN: Display access
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

	if ! docker image inspect vesta-sim:latest >/dev/null 2>&1; then
	    say ""
	    say "RU: Первая сборка образа. Это 15-25 минут, дальше запуск за секунды."
	    say "EN: First image build. Takes 15-25 minutes; later starts are instant."
	    say ""
	    $$COMPOSE build || exit 1
	fi

	# RU: Осиротевшие контейнеры от прерванных запусков держат порт 8080 и не дают подняться новому.
	# EN: Orphaned containers from interrupted runs hold port 8080 and block the next start.
	docker rm -f $$(docker ps -aq --filter "name=docker-sim-run") 2>/dev/null || true

	echo
	echo "  ========================================================"
	echo "   Simulation"
	echo "  ========================================================"
	echo "   GPU:      $$GPU"
	echo "   Display:  $$([ "$$HEADLESS" = 1 ] && echo headless || echo "$${DISPLAY}")"
	echo "  ========================================================"
	echo

	VOLS=(-v "$(CURDIR)/ros2_ws/src:/ros2_ws/src" -v "$(CURDIR)/ros2_ws/src/rviz:/root/.rviz2/")
	if [ -n "$${CMD:-}" ]; then
	    exec $$COMPOSE run -T "$${VOLS[@]}" --rm --service-ports sim bash -lc "$$CMD"
	fi
	exec $$COMPOSE run --name webots_vesta "$${VOLS[@]}" --rm --service-ports sim bash

headless:
	@$(MAKE) --no-print-directory shell HEADLESS=1

rebuild:
	@$(PREAMBLE)
	say "RU: Пересборка образа с нуля | EN: rebuilding image from scratch"
	$$COMPOSE build --no-cache
	# RU: Артефакты сборки colcon лежат на хосте в .build (root-владелец): чистим через контейнер.
	# EN: colcon artifacts live on the host in .build (root-owned): clean them through a container.
	if [ -d .build ]; then
	    docker run --rm -v "$(CURDIR)/.build:/b" vesta-sim:latest bash -c 'rm -rf /b/build /b/install /b/log' 2>/dev/null || rm -rf .build
	fi
	mkdir -p .build/build .build/install .build/log
	say "RU: Готово. Запуск: make | EN: Done. Start with: make"

stop:
	@$(PREAMBLE)
	say "RU: Останавливаю | EN: stopping"
	$$COMPOSE down --remove-orphans   # RU: заодно убирает контейнеры удалённых сервисов (CARLA, Autoware) | EN: also removes containers of removed services (CARLA, Autoware)

attach:
	@docker exec -it webots_vesta bash || { echo "  RU: контейнер webots_vesta не запущен (make) | EN: container webots_vesta is not running (make)"; exit 1; }

# -------------------------------------------------------------------------
# RU: Демо Nav2 + восприятие | EN: Nav2 + perception demo
# -------------------------------------------------------------------------
demo:
	@ros2_ws/src/offroad_nav2/docker/restart.sh $(ARGS)

demo-stop:
	@docker rm -f nav_sim >/dev/null 2>&1 && echo "  nav_sim stopped" || echo "  nav_sim is not running"
