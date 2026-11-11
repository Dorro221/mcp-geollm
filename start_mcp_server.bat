@echo off
setlocal

rem Starts the ENVIROO shelter MCP server in its own CMD window.
rem Runs from any directory; %~dp0 is the project root.
set "PROJECT_DIR=%~dp0"

if not exist "%PROJECT_DIR%env\Scripts\python.exe" (
    echo Error: no virtual environment at "%PROJECT_DIR%env".
    echo Create it and install the dependencies with: python -m venv env ^&^& env\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

start "ENVIROO MCP :9100" /D "%PROJECT_DIR%" cmd /k "call env\Scripts\activate.bat && python -m app.mcp.server"

echo ENVIROO MCP server window launched (http://localhost:9100/mcp).
endlocal
