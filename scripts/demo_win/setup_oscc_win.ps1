<#
.SYNOPSIS
    Sets up and diagnoses the OSCC DS4 teleop environment on Windows 11.

.DESCRIPTION
    Installs the Python packages, then checks every link in the chain that has
    to work before `py ds4_teleop_win.py` can talk to the car:
    Python architecture, PCANBasic.dll, the PCAN-USB device, python-can's view
    of the channels, and the gamepad.

    Nothing here touches the vehicle. Safe to run repeatedly.

.PARAMETER SkipInstall
    Only diagnose; do not install or upgrade Python packages.

.PARAMETER OpenDownloads
    Open the PEAK driver download page in the browser if the driver is missing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_oscc_win.ps1
#>

[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$OpenDownloads
)

$ErrorActionPreference = 'Continue'
$script:Problems = @()

function Write-Head($text) {
    Write-Host ""
    Write-Host "== $text " -ForegroundColor Cyan -NoNewline
    Write-Host ("=" * [Math]::Max(0, 60 - $text.Length)) -ForegroundColor Cyan
}

function Write-Ok($text)   { Write-Host "  [ ok ] $text" -ForegroundColor Green }
function Write-Warn($text) { Write-Host "  [warn] $text" -ForegroundColor Yellow }
function Write-Bad($text)  {
    Write-Host "  [fail] $text" -ForegroundColor Red
    $script:Problems += $text
}
function Write-Info($text) { Write-Host "         $text" -ForegroundColor DarkGray }

$PeakUrl = 'https://www.peak-system.com/Drivers.523.0.html'

# ---------------------------------------------------------------------------
Write-Head "Python"

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }

if (-not $py) {
    Write-Bad "No Python launcher found on PATH."
    Write-Info "Install from python.org and tick 'Add python.exe to PATH'."
    Write-Host ""
    Write-Host "Cannot continue without Python." -ForegroundColor Red
    exit 1
}

$exe = $py.Source
$pyVersion = & $exe -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
$pyBits    = & $exe -c "import struct; print(struct.calcsize('P') * 8)" 2>$null

Write-Ok "$exe -> Python $pyVersion ($pyBits-bit)"

if ($pyBits -ne '64') {
    Write-Bad "Python is $pyBits-bit."
    Write-Info "The PCAN-Basic DLL loaded by python-can must match. A 32-bit"
    Write-Info "Python needs the SysWOW64 copy; mixing them is the single most"
    Write-Info "common reason detect_available_configs returns an empty list."
}

# ---------------------------------------------------------------------------
Write-Head "Python packages"

if ($SkipInstall) {
    Write-Info "skipped (-SkipInstall)"
} else {
    # `uptime` only silences python-can's timestamp warning, but it is one line.
    $packages = @('pygame', 'python-can', 'uptime')
    Write-Info ("installing: " + ($packages -join ', '))
    & $exe -m pip install --upgrade --quiet @packages
    if ($LASTEXITCODE -ne 0) {
        Write-Bad "pip install failed (exit $LASTEXITCODE). Re-run without --quiet to see why."
    }
}

foreach ($module in @('pygame', 'can')) {
    # pygame prints a banner to stdout on import; swallow it so only the
    # version reaches the console.
    $probe = "import os,contextlib,io; os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'`n" +
             "buf=io.StringIO()`n" +
             "with contextlib.redirect_stdout(buf): import $module`n" +
             "print(getattr($module,'__version__','unknown'))"
    $ver = ($probe -replace '`n', "`n") | & $exe - 2>$null
    $ver = ($ver | Select-Object -Last 1)
    if ($LASTEXITCODE -eq 0 -and $ver) { Write-Ok "$module $ver" }
    else { Write-Bad "$module not importable" }
}

# ---------------------------------------------------------------------------
Write-Head "PCAN-Basic driver"

$dll64 = Join-Path $env:WINDIR 'System32\PCANBasic.dll'
$dll32 = Join-Path $env:WINDIR 'SysWOW64\PCANBasic.dll'
$has64 = Test-Path $dll64
$has32 = Test-Path $dll32

if ($has64) { Write-Ok  "System32\PCANBasic.dll present (for 64-bit Python)" }
else        { Write-Warn "System32\PCANBasic.dll missing" }

if ($has32) { Write-Ok  "SysWOW64\PCANBasic.dll present (for 32-bit Python)" }
else        { Write-Warn "SysWOW64\PCANBasic.dll missing" }

$needed = if ($pyBits -eq '64') { $has64 } else { $has32 }
if (-not $needed) {
    Write-Bad "No PCANBasic.dll matching your $pyBits-bit Python."
    Write-Info "Neither copy exists, so the PEAK software was never installed."
    Write-Info "Download 'PCAN-Drivers for Windows' (Device Driver Setup) and run"
    Write-Info "it; during setup tick the PCAN-Basic API component. It installs"
    Write-Info "both the kernel driver for the adapter and the DLL python-can"
    Write-Info "loads. Copying the DLL alone is not enough -- the device needs"
    Write-Info "the driver too. Re-plug the adapter afterwards."
    Write-Info $PeakUrl
    if ($OpenDownloads) { Start-Process $PeakUrl }
}

# Confirm the DLL actually loads, not just that the file exists. A DLL built
# for the other architecture is present but unloadable.
if ($needed) {
    $loadTest = @'
import ctypes, sys
try:
    ctypes.WinDLL("PCANBasic.dll")
    print("LOADED")
except OSError as exc:
    print("FAILED", exc)
'@
    $result = $loadTest | & $exe - 2>$null
    if ($result -match 'LOADED') { Write-Ok "PCANBasic.dll loads into this Python" }
    else {
        Write-Bad "PCANBasic.dll exists but will not load: $result"
        Write-Info "Almost always an architecture mismatch. Reinstall the driver,"
        Write-Info "or switch to a Python matching the DLL you have."
    }
}

# ---------------------------------------------------------------------------
Write-Head "PCAN-USB hardware"

$devices = @()
try {
    # -cmatch, not -match: the default is case-insensitive and "Speakers"
    # contains "peak". -PresentOnly drops ghost entries for hardware that was
    # once attached and is now gone.
    $devices = @(Get-PnpDevice -PresentOnly -ErrorAction Stop | Where-Object {
        $_.FriendlyName -cmatch 'PCAN' -or
        $_.Manufacturer -cmatch 'PEAK' -or
        $_.InstanceId   -cmatch 'VID_0C72'
    })
} catch {
    Write-Warn "Get-PnpDevice unavailable: $($_.Exception.Message)"
}

if ($devices.Count -eq 0) {
    Write-Bad "No PCAN/PEAK device found by Windows."
    Write-Info "Plug the adapter into USB and re-run. If it is plugged in,"
    Write-Info "open Device Manager and look for an unknown device: that means"
    Write-Info "the driver is missing, not the hardware."
} else {
    foreach ($device in $devices) {
        # A driverless device has no FriendlyName at all, so fall back to
        # something that identifies it.
        $label = $device.FriendlyName
        if ([string]::IsNullOrWhiteSpace($label)) { $label = $device.Name }
        if ([string]::IsNullOrWhiteSpace($label)) { $label = $device.InstanceId }
        if ([string]::IsNullOrWhiteSpace($label)) { $label = '(unnamed device)' }

        $problem = $null
        try {
            $problem = (Get-PnpDeviceProperty -InstanceId $device.InstanceId `
                -KeyName 'DEVPKEY_Device_ProblemCode' -ErrorAction Stop).Data
        } catch { }

        if ($device.Status -eq 'OK') {
            Write-Ok "$label [OK]"
        } else {
            Write-Bad "$label [$($device.Status)$(if ($problem) { ", problem $problem" })]"
            Write-Info "InstanceId: $($device.InstanceId)"

            switch ($problem) {
                28 {
                    Write-Info "Problem 28 = no driver installed. The adapter is"
                    Write-Info "physically fine; Windows simply has nothing to bind"
                    Write-Info "to it. Installing the PEAK driver package fixes this."
                }
                43 { Write-Info "Problem 43 = the device reported a failure. Try another port or cable." }
                default {
                    Write-Info "Sitting under 'Other devices' in Device Manager with"
                    Write-Info "a yellow mark means the driver is missing."
                }
            }
        }
    }
}

# PCAN-View holds channels exclusively; python-can then sees nothing.
$viewer = Get-Process -Name 'PcanView*' -ErrorAction SilentlyContinue
if ($viewer) {
    Write-Bad "PCAN-View is running and holds the channel exclusively."
    Write-Info "Close it before starting the teleop."
}

# ---------------------------------------------------------------------------
Write-Head "python-can channel detection"

$detect = @'
import warnings
warnings.filterwarnings("ignore")
try:
    import can
    found = can.detect_available_configs(["pcan"])
except Exception as exc:
    print("ERROR", type(exc).__name__, exc)
else:
    if not found:
        print("EMPTY")
    for config in found:
        print("CHANNEL", config.get("channel"), config.get("interface"))
'@

$channels = $detect | & $exe - 2>$null
$found = @($channels | Where-Object { $_ -match '^CHANNEL' })

if ($found.Count -gt 0) {
    foreach ($line in $found) { Write-Ok $line }
} elseif ($channels -match 'ERROR') {
    Write-Bad "detection raised: $channels"
} else {
    if (-not $needed) {
        Write-Warn "No PCAN channels -- expected, the driver is not installed yet."
        Write-Info "This line will resolve itself once the PEAK package is in."
    } else {
        Write-Bad "No PCAN channels detected."
        Write-Info "The DLL loads and the device is present, so this usually means"
        Write-Info "the adapter is bound to another process, or it enumerated after"
        Write-Info "the driver install and has not been re-plugged."
    }
}

# ---------------------------------------------------------------------------
Write-Head "Gamepad"

$pad = @'
import os, warnings
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
warnings.filterwarnings("ignore")
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import pygame
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    info = []
    for i in range(count):
        j = pygame.joystick.Joystick(i); j.init()
        info.append((j.get_name(), j.get_numaxes(), j.get_numbuttons()))
if not info:
    print("NONE")
for name, axes, buttons in info:
    print(f"PAD {name} | axes={axes} buttons={buttons}")
'@

$pads = $pad | & $exe - 2>$null
if ($pads -match '^PAD') {
    foreach ($line in @($pads | Where-Object { $_ -match '^PAD' })) { Write-Ok $line }
    Write-Info "Run 'py ds4_teleop_win.py --probe' to confirm axis indices."
} else {
    Write-Warn "No gamepad detected."
    Write-Info "USB: plug the DS4 in. Bluetooth: hold SHARE+PS until the bar"
    Write-Info "flashes, then pair from Settings > Bluetooth & devices."
}

# ---------------------------------------------------------------------------
Write-Head "Summary"

if ($script:Problems.Count -eq 0) {
    Write-Host "  Everything checks out." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Next:" -ForegroundColor White
    Write-Host "    py ds4_teleop_win.py --probe     confirm axis and button indices"
    Write-Host "    py ds4_teleop_win.py --no-can    full loop, bus untouched"
    Write-Host "    py ds4_teleop_win.py             live"
} else {
    Write-Host "  $($script:Problems.Count) problem(s):" -ForegroundColor Red
    foreach ($problem in $script:Problems) { Write-Host "    - $problem" -ForegroundColor Red }
    Write-Host ""
    Write-Host "  The gamepad half works without any of this:" -ForegroundColor White
    Write-Host "    py ds4_teleop_win.py --no-can"
}
Write-Host ""
