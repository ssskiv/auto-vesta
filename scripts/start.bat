@echo off
REM RU: Обёртка для запуска из проводника двойным кликом.
REM EN: Wrapper so the simulation can be started by double-click.
REM
REM RU: Аргументы передаются дальше: start.bat -Headless
REM EN: Arguments are forwarded: start.bat -Headless
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
pause
