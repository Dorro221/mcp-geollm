@echo off
setlocal

rem Starts the MINIMAL-description arm of the shelter MCP server, on its own
rem port, in its own CMD window. Same tools, same data, shorter descriptions.
rem
rem The two arms publish the same tool names, so register ONLY ONE of them in
rem mcp_servers.json at a time - the harness keeps the first server that claims
rem a name and warns about the rest.
set "PROJECT_DIR=%~dp0"

if not exist "%PROJECT_DIR%env\Scripts\python.exe" (
    echo Error: no virtual environment at "%PROJECT_DIR%env".
    exit /b 1
)

start "ENVIROO MCP MINIMAL :9101" /D "%PROJECT_DIR%" cmd /k "call env\Scripts\activate.bat && python -m app.mcp.server_minimal"

echo ENVIROO MCP minimal-description server launched (http://localhost:9101/mcp).
echo Register it in the UI panel, and remove the rich one at :9100 first.
endlocal
