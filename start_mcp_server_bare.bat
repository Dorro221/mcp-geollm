@echo off
setlocal

rem Starts the BARE-description arm of the shelter MCP server, on its own port.
rem Same tools, same data; the tool name is the only documentation.
rem
rem The three arms publish the same tool names, so enable ONLY ONE of them in
rem the UI panel at a time.
set "PROJECT_DIR=%~dp0"

if not exist "%PROJECT_DIR%env\Scripts\python.exe" (
    echo Error: no virtual environment at "%PROJECT_DIR%env".
    exit /b 1
)

start "ENVIROO MCP BARE :9102" /D "%PROJECT_DIR%" cmd /k "call env\Scripts\activate.bat && python -m app.mcp.server_bare"

echo ENVIROO MCP bare-description server launched (http://localhost:9102/mcp).
endlocal
