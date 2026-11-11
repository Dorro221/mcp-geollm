@echo off
setlocal

rem Builds the ENVIROO analysis schema from the source inventory:
rem   1. schema  2. shelters  3. population layer
rem Safe to re-run: each step rebuilds its own tables and touches nothing else.
rem The source inventory is only ever read.
set "PROJECT_DIR=%~dp0"
set "PY=%PROJECT_DIR%env\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
cd /d "%PROJECT_DIR%"

echo [1/3] schema
"%PY%" -m db.apply_schema || exit /b 1
echo [2/3] shelters
"%PY%" -m db.etl_shelters || exit /b 1
echo [3/3] population layer
"%PY%" -m db.etl_population || exit /b 1
echo Done.
endlocal
