#!/usr/bin/env bash
# =========================================================================
# RU: Запуск симуляции робота одной командой.
#     Скрипт сам определяет видеокарту, настраивает доступ к дисплею,
#     собирает образ при первом запуске и стартует симуляцию.
#
#     ./start.sh              запустить симуляцию с окном Gazebo
#     ./start.sh --headless   без графики, только API (быстрее)
#     ./start.sh --shell      оболочка внутри контейнера
#     ./start.sh --rebuild    пересобрать образ с нуля
#     ./start.sh --stop       остановить
#     ./start.sh --gpu amd    задать видеокарту вручную
#
# EN: Launch the simulation with a single command.
#     The script detects the GPU, configures display access, builds the
#     image on first run and starts the simulation.
#
#     ./start.sh              run the simulation with a Gazebo window
#     ./start.sh --headless   no graphics, API only (faster)
#     ./start.sh --shell      shell inside the container
#     ./start.sh --rebuild    rebuild the image from scratch
#     ./start.sh --stop       stop everything
#     ./start.sh --gpu amd    select the GPU manually
# =========================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$HERE/../docker"

MODE="shell"
GPU=""
HEADLESS=false

while [ $# -gt 0 ]; do
    case "$1" in
        --headless) HEADLESS=true; shift ;;
        --shell)    MODE="shell"; shift ;;
        --rebuild)  MODE="rebuild"; shift ;;
        --stop)     MODE="stop"; shift ;;
        --build-world) MODE="world"; shift ;;
        --attach)   MODE="attach"; shift ;;
        --gpu)      GPU="$2"; shift 2 ;;
        -h|--help)  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "RU: неизвестный аргумент | EN: unknown argument: $1"; exit 1 ;;
    esac
done

say() { echo -e "  $*"; }

# -------------------------------------------------------------------------
# RU: Проверка Docker | EN: Docker check
# -------------------------------------------------------------------------
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
    say "    sudo usermod -aG docker \$USER   # RU: затем перелогиньтесь"
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

# -------------------------------------------------------------------------
# RU: Определение видеокарты | EN: GPU detection
# -------------------------------------------------------------------------
detect_gpu() {
    # RU: NVIDIA годится, только если установлен container toolkit —
    #     без него контейнер не получит доступ к устройству.
    # EN: NVIDIA only counts if the container toolkit is installed —
    #     without it the container cannot access the device.
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        if docker info 2>/dev/null | grep -qi nvidia || \
           [ -f /usr/bin/nvidia-container-runtime ]; then
            echo "nvidia"; return
        fi
        say "RU: NVIDIA найдена, но нет nvidia-container-toolkit."
        say "EN: NVIDIA found, but nvidia-container-toolkit is missing."
        say "    https://docs.nvidia.com/datacenter/cloud-native/"
    fi

    if [ -d /dev/dri ]; then
        if lspci 2>/dev/null | grep -qiE 'vga|3d' && \
           lspci 2>/dev/null | grep -qi 'amd\|radeon'; then
            echo "amd"; return
        fi
        if lspci 2>/dev/null | grep -qi 'intel.*graphics'; then
            echo "intel"; return
        fi
        echo "amd"; return   # RU: /dev/dri есть — пробуем Mesa | EN: /dev/dri exists, try Mesa
    fi

    echo "cpu"
}

if [ -z "$GPU" ]; then
    GPU="$(detect_gpu)"
fi

OVERLAY="$DOCKER_DIR/compose.${GPU}.yml"
if [ ! -f "$OVERLAY" ]; then
    say "RU: Нет конфигурации для '$GPU' | EN: no configuration for '$GPU'"
    exit 1
fi

COMPOSE="$DC -f $DOCKER_DIR/docker-compose.yml -f $OVERLAY"

# -------------------------------------------------------------------------
# RU: Остановка | EN: Stop
# -------------------------------------------------------------------------
if [ "$MODE" = "stop" ]; then
    say "RU: Останавливаю | EN: stopping"
    $COMPOSE down
    exit 0
fi

# -------------------------------------------------------------------------
# RU: Доступ к дисплею | EN: Display access
# -------------------------------------------------------------------------
setup_display() {
    if [ "$HEADLESS" = true ]; then
        say "RU: Режим без графики | EN: headless mode"
        return
    fi

    if [ -z "${DISPLAY:-}" ]; then
        say "RU: DISPLAY не задан, перехожу в режим без графики."
        say "EN: DISPLAY is unset, falling back to headless."
        HEADLESS=true
        return
    fi

    # RU: xhost открывает доступ к X-серверу для локальных контейнеров.
    #     Без этого Gazebo не откроет окно.
    # EN: xhost grants X server access to local containers.
    #     Without it Gazebo cannot open a window.
    if command -v xhost >/dev/null 2>&1; then
        xhost +local:docker >/dev/null 2>&1 || \
            xhost +local:root >/dev/null 2>&1 || true
    else
        say "RU: Нет xhost, окно может не открыться: apt install x11-xserver-utils"
        say "EN: xhost missing, the window may not open: apt install x11-xserver-utils"
    fi
}

setup_display

mkdir -p "$HERE/../.build/build" "$HERE/../.build/install" "$HERE/../.build/log"

# -------------------------------------------------------------------------
# RU: Сборка образа | EN: Image build
# -------------------------------------------------------------------------
if [ "$MODE" = "rebuild" ]; then
    say "RU: Пересборка образа с нуля | EN: rebuilding image from scratch"
    $COMPOSE build --no-cache
    rm -rf "$HERE/../.build"
    mkdir -p "$HERE/../.build"/{build,install,log}
    MODE="shell"
elif ! docker image inspect vesta-sim:latest >/dev/null 2>&1; then
    say ""
    say "RU: Первая сборка образа. Это 15-25 минут, дальше запуск за секунды."
    say "EN: First image build. Takes 15-25 minutes; later starts are instant."
    say ""
    $COMPOSE build || exit 1
fi

# -------------------------------------------------------------------------
# RU: Запуск | EN: Launch
# -------------------------------------------------------------------------
# RU: Осиротевшие контейнеры от прерванных запусков держат порт 8080
#     и не дают подняться новому.
# EN: Orphaned containers from interrupted runs hold port 8080 and block
#     the next start.
docker rm -f $(docker ps -aq --filter "name=docker-sim-run") 2>/dev/null || true

cat <<INFO

  ========================================================
   Simulation
  ========================================================
   GPU:      $GPU
   Display:  $([ "$HEADLESS" = true ] && echo "headless" || echo "${DISPLAY}")
  ========================================================

INFO


if [ "$MODE" = "shell" ]; then
    $COMPOSE run --name webots_vesta -v "$HERE/../ros2_ws/src:/ros2_ws/src" -v "$HERE/../ros2_ws/src/rviz:/root/.rviz2/" --rm --service-ports sim bash
    exit $?
fi

MODE_ARG="realtime"
[ "$HEADLESS" = true ] && MODE_ARG="fast"

say "RU: Запуск. Панель поднимется примерно через минуту."
say "EN: Starting. The panel will be up in about a minute."
say ""

$COMPOSE run -v "$HERE/../ros2_ws/src:/ros2_ws/src" -v "$HERE/../ros2_ws/src/rviz:/root/.rviz2/" --rm --service-ports sim bash -lc "bash"