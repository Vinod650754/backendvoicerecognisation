@echo off
REM Master startup script for Voice App

echo.
echo ======================================================================
echo                         VOICE APP STARTUP
echo ======================================================================
echo.

REM Check Python venv
set PYTHON=C:\vscodeprojects\home_app\.venv\Scripts\python.exe
if not exist "%PYTHON%" (
    echo ERROR: Python virtual environment not found
    echo Expected: %PYTHON%
    exit /b 1
)

REM Start backend server
echo [STEP 1] Starting backend server on http://127.0.0.1:8000...
echo.
start "Backend Server" cmd /k "%PYTHON% -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"

echo [STEP 2] Waiting 5 seconds for backend to initialize...
timeout /t 5 /nobreak

echo.
echo [STEP 3] Starting Flutter app on emulator...
echo (This may take 1-2 minutes on first run)
echo.

flutter run -d emulator-5554

echo.
echo ======================================================================
echo If you see the app running on the emulator, everything is working!
echo Backend is running on http://127.0.0.1:8000
echo ======================================================================
pause
