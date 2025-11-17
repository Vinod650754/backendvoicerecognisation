# Quick Start Guide - Voice App

## Prerequisites
- ✓ Python 3.11+ with virtual environment (`.venv`)
- ✓ Flutter SDK installed
- ✓ Android emulator set up and ready
- ✓ All dependencies installed

## How to Run the Application

### Option 1: Using the Batch Script (Easiest - Windows)
```batch
1. Open PowerShell or CMD in the project root: C:\vscodeprojects\home_app
2. Run: .\START_APP.bat
3. This will:
   - Start the backend server (http://127.0.0.1:8000) in a new window
   - Wait 5 seconds for initialization
   - Launch the Flutter app on the emulator
4. Watch for the app to launch on your emulator screen
```

### Option 2: Manual Two-Step (if batch script doesn't work)

**Terminal 1 - Start the Backend:**
```powershell
cd C:\vscodeprojects\home_app
C:/.venv/Scripts/python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Wait for: `Uvicorn running on http://127.0.0.1:8000`

**Terminal 2 - Start the Frontend:**
```powershell
cd C:\vscodeprojects\home_app
flutter run -d emulator-5554
```

## Backend Details

**Running Endpoint:** http://127.0.0.1:8000

**Available Endpoints:**
- `GET /health` - Check backend status
- `POST /collect_sample` - Upload audio samples for training
- `POST /detect_wakeword` - Detect wakeword in audio
- `POST /detect_command` - Detect command intent
- `POST /verify_voice` - Verify voice biometric

**Data Stored At:** `backend/data/`
- `backend/data/wakeword/` - Wakeword training samples
- `backend/data/command/` - Command training samples
- `backend/data/open_door/` - Open door command samples
- `backend/data/close_door/` - Close door command samples

**Models:**
- WakeWordModel - Binary classifier (detect wakeword vs noise)
- CommandModel - Multi-class classifier (detect which command)
- VoiceVerifier - Biometric voice verification

## Frontend Details (Flutter App)

**Device:** Android Emulator (emulator-5554)

**Two Tabs:**
1. **Voice Auth Tab** - Continuous wakeword detection and voice verification
2. **Training Samples Tab** - Record and upload training samples

**Backend Connection:** 
- Frontend connects to `http://10.0.2.2:8000` (emulator's way of reaching host localhost)
- This maps to the backend running at `http://127.0.0.1:8000`

## Troubleshooting

### Backend Won't Start
- **Issue:** `ModuleNotFoundError` or import errors
- **Fix:** Run `flutter pub get` to ensure venv is set up, then try again

### Backend Starts But Frontend Can't Connect
- **Issue:** `TimeoutException` in app logs
- **Fix:** 
  - Ensure backend is running: `curl http://127.0.0.1:8000/health`
  - Check emulator can reach host: `ping 10.0.2.2` in emulator
  - Emulator may need to be restarted

### Flutter App Won't Build
- **Issue:** Build errors or dependencies missing
- **Fix:** 
  - Run: `flutter clean`
  - Run: `flutter pub get`
  - Run: `flutter pub upgrade`

### No Emulator Found
- **Issue:** `emulator-5554 not found`
- **Fix:**
  - Open Android Studio → AVD Manager
  - Create/start an Android emulator
  - Ensure it shows up in `flutter devices`

## Testing the Connection

**Test 1: Backend Health**
```powershell
$response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health"
$response.Content | ConvertFrom-Json
```

**Test 2: Upload a Sample**
```powershell
$file = Get-ChildItem backend/data/wakeword/*.wav | Select-Object -First 1
$form = @{
    file = $file
    label = "wakeword"
}
# (requires multipart form data - use test_client.py instead)
```

**Test 3: Using test_client.py**
```powershell
python backend/test_client.py
```

## What Should Happen

1. ✓ Backend starts and prints: `INFO: Uvicorn running on http://127.0.0.1:8000`
2. ✓ Flutter app loads on emulator showing two tabs
3. ✓ **Training Samples Tab**: Can record and upload samples
4. ✓ **Voice Auth Tab**: Shows detection status and voice auth buttons
5. ✓ When you use the app, backend logs show requests: `GET /health`, `POST /detect_wakeword`, etc.
6. ✓ Audio samples are saved to `backend/data/<label>/`

## Performance Notes

- First build may take 3-5 minutes
- Audio processing uses PyTorch (backend is CPU-heavy)
- Wakeword detection may be 1-2 seconds per audio clip
- Voice verification is mock-based (add real model as needed)

## Next Steps

1. Test sample upload on Training Samples tab
2. Test voice auth pipeline on Voice Auth tab
3. Integrate with real BLE device (ESP32)
4. Add real TFLite models for on-device inference
5. Implement email alerts via Gmail SMTP
