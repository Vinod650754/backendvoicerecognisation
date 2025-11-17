# Postman API Testing Guide

## Backend URL
```
http://127.0.0.1:8000
```

---

## Endpoint 1: Health Check (GET)

**URL:** `http://127.0.0.1:8000/health`

**Method:** GET

**Headers:** None needed

**Body:** None

**Expected Response (200 OK):**
```json
{
    "status": "ok",
    "message": "Backend is running"
}
```

---

## Endpoint 2: Test Endpoint (POST)

**URL:** `http://127.0.0.1:8000/test`

**Method:** POST

**Headers:** None needed

**Body:** None (or empty)

**Expected Response (200 OK):**
```json
{
    "status": "ok",
    "message": "Backend is working"
}
```

---

## Endpoint 3: Collect Sample (Upload Audio)

**URL:** `http://127.0.0.1:8000/collect_sample`

**Method:** POST

**Headers:** 
- Content-Type: multipart/form-data (Postman sets this automatically)

**Body:**
1. Click on **Body** tab
2. Select **form-data**
3. Add two fields:

| Key | Type | Value |
|-----|------|-------|
| file | File | Select a .wav file from your computer |
| label | Text | `wakeword` or `command` or `open_door` or `close_door` |

**Example:**
- file: (select `C:\vscodeprojects\home_app\backend\data\wakeword\sample_1763282265861.wav`)
- label: `wakeword`

**Expected Response (200 OK):**
```json
{
    "success": true,
    "message": "Sample saved as sample_1700000000000.wav",
    "path": "C:\\vscodeprojects\\home_app\\backend\\data\\wakeword\\sample_1700000000000.wav",
    "label": "wakeword"
}
```

---

## Endpoint 4: Detect Wakeword

**URL:** `http://127.0.0.1:8000/detect_wakeword`

**Method:** POST

**Headers:**
- Content-Type: multipart/form-data

**Body:**
1. Click on **Body** tab
2. Select **form-data**
3. Add one field:

| Key | Type | Value |
|-----|------|-------|
| file | File | Select a .wav file |

**Example:**
- file: (select any .wav file)

**Expected Response (200 OK):**
```json
{
    "wakeword_detected": true,
    "confidence": 0.87,
    "model_status": "mock_model"
}
```

OR:

```json
{
    "wakeword_detected": false,
    "confidence": 0.42,
    "model_status": "mock_model"
}
```

---

## Endpoint 5: Detect Command

**URL:** `http://127.0.0.1:8000/detect_command`

**Method:** POST

**Headers:**
- Content-Type: multipart/form-data

**Body:**
1. Click on **Body** tab
2. Select **form-data**
3. Add one field:

| Key | Type | Value |
|-----|------|-------|
| file | File | Select a .wav file |

**Example:**
- file: (select any .wav file)

**Expected Response (200 OK):**
```json
{
    "intent": "open",
    "intent_idx": 0,
    "confidence": 0.92,
    "model_status": "mock_model"
}
```

**Possible intents:** `lock`, `unlock`, `open`, `close`

---

## Endpoint 6: Verify Voice (Biometric)

**URL:** `http://127.0.0.1:8000/verify_voice`

**Method:** POST

**Headers:**
- Content-Type: multipart/form-data

**Body:**
1. Click on **Body** tab
2. Select **form-data**
3. Add one field:

| Key | Type | Value |
|-----|------|-------|
| file | File | Select a .wav file |

**Example:**
- file: (select any .wav file)

**Expected Response (200 OK):**
```json
{
    "verified": true,
    "confidence": 0.85,
    "owner": "HomeOwner",
    "threshold": 0.7,
    "model_status": "mock_model"
}
```

OR:

```json
{
    "verified": false,
    "confidence": 0.62,
    "owner": "HomeOwner",
    "threshold": 0.7,
    "model_status": "mock_model"
}
```

---

## Step-by-Step: How to Test in Postman

### 1. Download and Open Postman
- Get it from: https://www.postman.com/downloads/

### 2. Create a New Request
- Click **+ New** → **HTTP Request**

### 3. Test Health Endpoint
```
GET http://127.0.0.1:8000/health
Click Send
```

### 4. Test Collect Sample
```
POST http://127.0.0.1:8000/collect_sample
Body → form-data
  - file: (upload a .wav file)
  - label: wakeword
Click Send
```

### 5. Test Wakeword Detection
```
POST http://127.0.0.1:8000/detect_wakeword
Body → form-data
  - file: (upload a .wav file)
Click Send
```

### 6. Test Command Detection
```
POST http://127.0.0.1:8000/detect_command
Body → form-data
  - file: (upload a .wav file)
Click Send
```

### 7. Test Voice Verification
```
POST http://127.0.0.1:8000/verify_voice
Body → form-data
  - file: (upload a .wav file)
Click Send
```

---

## Sample WAV File for Testing

You can use the sample already collected:
```
C:\vscodeprojects\home_app\backend\data\wakeword\sample_1763282265861.wav
```

Or record your own using the Flutter app's "Training Samples" tab!

---

## Backend Status Check

Make sure backend is running before testing:

**Terminal:**
```powershell
cd C:\vscodeprojects\home_app
C:/.venv/Scripts/python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Should show:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Connection refused | Make sure backend is running on port 8000 |
| 404 Not Found | Check the URL spelling (should be `/health`, `/detect_wakeword`, etc.) |
| 400 Bad Request | File might be missing or label is wrong |
| Timeout | Backend might be processing (takes 1-2 seconds for AI) |

---

## Data Directory

All uploaded samples are saved to:
```
backend/data/
├── wakeword/
├── command/
├── open_door/
└── close_door/
```

Check here to see what the app and Postman have uploaded!
