@echo off
REM Double-click launcher for setup_oscc_win.ps1.
REM Bypasses the execution policy for this one process only -- nothing is
REM changed system-wide.

setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_oscc_win.ps1" %*

echo.
pause
