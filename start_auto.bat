@echo off
REM ============================================================
REM start_auto.bat - Auto-Jaoyi hands-off trader launcher
REM Console -> auto_console.log; auto_run.py writes auto_trade.log.
REM Keep them separate (bat redirect + python FileHandler on same
REM file -> Windows PermissionError).
REM ============================================================
cd /d "D:\LICAISOFT\Auto-Jaoyi"

REM Prefer the managed Python (has akshare/baostock/pandas/numpy)
set "PY=C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "CON=auto_console.log"

REM start keep_awake.py in background (prevents system sleep / screen-off)
start "AutoJaoyi_KeepAwake" /min "%PY%" keep_awake.py

REM periodic param-review hint (throttled; logs to console file)
"%PY%" review_hint.py >> "%CON%" 2>&1

echo [%date% %time%] launching auto_run.py >> "%CON%"
"%PY%" auto_run.py >> "%CON%" 2>&1
echo [%date% %time%] auto_run.py exited >> "%CON%"

REM stop keep_awake now that trading is done
powershell -NoProfile -ExecutionPolicy Bypass -Command "&{ Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*keep_awake.py*' } | Stop-Process -Force }"