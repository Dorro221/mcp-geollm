@echo off
setlocal

rem Starts the ENVIROO web server in its own CMD window.
set "PROJECT_DIR=%~dp0"

if not exist "%PROJECT_DIR%env\Scripts\python.exe" (
    echo Error: no virtual environment at "%PROJECT_DIR%env".
    echo Create it and install the dependencies with: python -m venv env ^&^& env\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

start "ENVIROO web :8090" /D "%PROJECT_DIR%" cmd /k "call env\Scripts\activate.bat && python -m app.web.web_server"

echo ENVIROO web server window launched (http://localhost:8090).
endlocal
