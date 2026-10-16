@echo off
rem Always-on НИР7 data collection (Windows). Restarts the service if Python exits.
rem Set PYTHON to the interpreter with requirements.txt installed (default: python).
chcp 65001 >nul
if "%PYTHON%"=="" set PYTHON=python
cd /d "%~dp0"
if not exist data\raw mkdir data\raw
:loop
"%PYTHON%" -u -m collector.service >> data\raw\service.log 2>&1
echo %date% %time% service exited, restarting in 30 s >> data\raw\service.log
timeout /t 30 /nobreak >nul
goto loop
