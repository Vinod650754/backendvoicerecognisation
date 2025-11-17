@echo off
REM Start the backend server locally

setlocal enabledelayedexpansion

echo.
echo ======================================================================
echo                   BACKEND SERVER STARTUP
echo ======================================================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Check if virtual environment exists
if not exist "%SCRIPT_DIR%..\.venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found
    echo Please run from the parent directory with .venv activated
    exit /b 1
)

echo [1/3] Checking environment...
echo Backend Directory: %SCRIPT_DIR%
echo.

echo [2/3] Creating data directories...
for %%D in (wakeword command open_door close_door) do (
    if not exist "%SCRIPT_DIR%data\%%D" mkdir "%SCRIPT_DIR%data\%%D"
    echo   - data\%%D created/exists
)
echo.

echo [3/3] Starting backend server...
echo.
echo Listening on: http://127.0.0.1:8000
echo Documentation: http://127.0.0.1:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

endlocal
