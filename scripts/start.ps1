<#
=========================================================================
 RU: Запуск симуляции робота на Windows.
     Скрипт проверяет Docker Desktop, определяет видеокарту, настраивает
     графику через WSLg и запускает симуляцию.

     .\start.ps1              запустить с окном Gazebo
     .\start.ps1 -Headless    без графики, только API (быстрее)
     .\start.ps1 -Shell       оболочка внутри контейнера
     .\start.ps1 -Rebuild     пересобрать образ с нуля
     .\start.ps1 -Stop        остановить
     .\start.ps1 -Gpu cpu     задать видеокарту вручную

 EN: Launch the simulation on Windows.
     The script checks Docker Desktop, detects the GPU, sets up graphics
     through WSLg and starts the simulation.

     .\start.ps1              run with a Gazebo window
     .\start.ps1 -Headless    no graphics, API only (faster)
     .\start.ps1 -Shell       shell inside the container
     .\start.ps1 -Rebuild     rebuild the image from scratch
     .\start.ps1 -Stop        stop everything
     .\start.ps1 -Gpu cpu     select the GPU manually

 RU: Если PowerShell блокирует запуск скриптов, выполните один раз:
 EN: If PowerShell blocks scripts, run once:
     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
=========================================================================
#>
param(
    [switch]$Headless,
    [switch]$Shell,
    [switch]$Rebuild,
    [switch]$Stop,
    [string]$Gpu = ""
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$DockerDir = Join-Path $Here "docker"

function Say($text) { Write-Host "  $text" }

# -------------------------------------------------------------------------
# RU: Проверка Docker | EN: Docker check
# -------------------------------------------------------------------------
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Say "RU: Docker Desktop не установлен."
    Say "EN: Docker Desktop is not installed."
    Say ""
    Say "    https://www.docker.com/products/docker-desktop/"
    exit 1
}

try { docker info 2>&1 | Out-Null } catch {
    Say "RU: Docker Desktop не запущен. Запустите его и повторите."
    Say "EN: Docker Desktop is not running. Start it and try again."
    exit 1
}

# RU: WSL 2 обязателен: без него нет ни производительности, ни WSLg.
# EN: WSL 2 is required: without it there is neither performance nor WSLg.
$wslOk = $false
try {
    $wslOut = wsl --status 2>&1 | Out-String
    if ($wslOut -match "2") { $wslOk = $true }
} catch { }

if (-not $wslOk) {
    Say "RU: WSL 2 не найден. Docker Desktop должен работать на WSL 2."
    Say "EN: WSL 2 not found. Docker Desktop must run on the WSL 2 backend."
    Say ""
    Say "    wsl --install"
    Say "    RU: затем в Docker Desktop: Settings > General > Use WSL 2"
    Say "    EN: then in Docker Desktop: Settings > General > Use WSL 2"
    Say ""
}

# -------------------------------------------------------------------------
# RU: Определение видеокарты | EN: GPU detection
# -------------------------------------------------------------------------
function Get-GpuKind {
    # RU: WSLg пробрасывает GPU автоматически, но NVIDIA требует
    #     совместимого драйвера на хосте.
    # EN: WSLg forwards the GPU automatically, but NVIDIA needs a
    #     compatible host driver.
    try {
        $video = Get-CimInstance Win32_VideoController -ErrorAction Stop
        $names = ($video | ForEach-Object { $_.Name }) -join " "

        if ($names -match "NVIDIA") {
            if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
                return "nvidia"
            }
            Say "RU: NVIDIA найдена, но nvidia-smi недоступна — программный режим."
            Say "EN: NVIDIA found, but nvidia-smi is unavailable — software mode."
            return "cpu"
        }
        if ($names -match "AMD|Radeon") { return "amd" }
        if ($names -match "Intel")      { return "intel" }
    } catch { }
    return "cpu"
}

if ([string]::IsNullOrEmpty($Gpu)) { $Gpu = Get-GpuKind }

$Overlay = Join-Path $DockerDir "compose.$Gpu.yml"
if (-not (Test-Path $Overlay)) {
    Say "RU: Нет конфигурации для '$Gpu' | EN: no configuration for '$Gpu'"
    exit 1
}

$BaseCompose = Join-Path $DockerDir "docker-compose.yml"
$ComposeArgs = @("compose", "-f", $BaseCompose, "-f", $Overlay)

# -------------------------------------------------------------------------
# RU: Остановка | EN: Stop
# -------------------------------------------------------------------------
if ($Stop) {
    Say "RU: Останавливаю | EN: stopping"
    & docker @ComposeArgs down
    exit 0
}

# -------------------------------------------------------------------------
# RU: Графика через WSLg | EN: Graphics through WSLg
# -------------------------------------------------------------------------
# RU: WSLg в Windows 11 даёт готовый X-сервер по адресу :0 и монтирует
#     сокет в /tmp/.X11-unix, поэтому отдельный VcXsrv не нужен.
#     На Windows 10 без WSLg окно не откроется — используйте -Headless.
# EN: WSLg on Windows 11 provides a ready X server at :0 and mounts the
#     socket at /tmp/.X11-unix, so a separate VcXsrv is unnecessary.
#     On Windows 10 without WSLg the window will not open — use -Headless.
if (-not $Headless) {
    $env:DISPLAY = ":0"
    $build = [System.Environment]::OSVersion.Version.Build
    if ($build -lt 22000) {
        Say "RU: Windows 10 — WSLg недоступен, окно Gazebo может не открыться."
        Say "EN: Windows 10 — WSLg is unavailable, the Gazebo window may not open."
        Say "    RU: при проблемах используйте -Headless"
        Say "    EN: use -Headless if it fails"
        Say ""
    }
} else {
    $env:DISPLAY = ""
}

# RU: Каталоги сборки на хосте: пересборка не нужна при каждом запуске.
# EN: Host-side build directories: no rebuild on every start.
foreach ($d in @("build", "install", "log")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Here ".build\$d") | Out-Null
}

# -------------------------------------------------------------------------
# RU: Сборка образа | EN: Image build
# -------------------------------------------------------------------------
$imageExists = $false
try {
    docker image inspect vesta-sim:latest 2>&1 | Out-Null
    $imageExists = $true
} catch { }

if ($Rebuild) {
    Say "RU: Пересборка образа с нуля | EN: rebuilding image from scratch"
    & docker @ComposeArgs build --no-cache
    Remove-Item -Recurse -Force (Join-Path $Here ".build") -ErrorAction SilentlyContinue
    foreach ($d in @("build", "install", "log")) {
        New-Item -ItemType Directory -Force -Path (Join-Path $Here ".build\$d") | Out-Null
    }
} elseif (-not $imageExists) {
    Say ""
    Say "RU: Первая сборка образа. Это 15-25 минут, дальше запуск за секунды."
    Say "EN: First image build. Takes 15-25 minutes; later starts are instant."
    Say ""
    & docker @ComposeArgs build
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

# -------------------------------------------------------------------------
# RU: Запуск | EN: Launch
# -------------------------------------------------------------------------
$displayInfo = if ($Headless) { "headless" } else { $env:DISPLAY }

Write-Host @"

  ========================================================
   Simulation
  ========================================================
   GPU:      $Gpu
   Display:  $displayInfo
  ========================================================

"@

if ($Shell) {
    & docker @ComposeArgs run --rm --service-ports sim bash
    exit $LASTEXITCODE
}

$guiArg = if ($Headless) { "false" } else { "true" }

Say "RU: Запуск. Панель поднимется примерно через минуту."
Say "EN: Starting. The panel will be up in about a minute."
Say ""

