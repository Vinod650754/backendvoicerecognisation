# Smart Door Lock Backend

This folder contains the FastAPI service that powers the Flutter **Smart Door Lock** app. The backend exposes three ML endpoints and a data ingestion endpoint so you can keep collecting fresh samples from the mobile app.

## Folder layout

```
backend/
├── app.py                # FastAPI application
├── run_server.py         # Simple uvicorn launcher
├── requirements.txt      # Python dependencies
├── models/               # Drop your trained checkpoints here
│   ├── wakeword_model.pth          (PyTorch, optional placeholder)
│   ├── command_model.pth           (PyTorch, optional placeholder)
│   ├── voice_biometric_model.h5    (Keras)
│   └── voice_biometric_labels.json (label → index map)
└── data/
    ├── wakeword/                   # Wakeword samples from the app
    └── command/
        ├── open_door/
        └── close_door/
```

- `models/` currently contains the voice biometric model shipped with the project. The PyTorch checkpoints for wakeword and command detection are optional; if you haven’t trained them yet, the backend falls back to placeholder weights but will log a warning.
- `data/` is kept empty by default and is where `/collect_sample` writes incoming audio clips from the Flutter training tab.

## Endpoints

| Method | Path              | Description |
|--------|-------------------|-------------|
| GET    | `/health`         | Quick readiness probe with model/data status. |
| POST   | `/collect_sample` | Multipart upload (`file`, `label`). Label must be `wakeword`, `open_door`, or `close_door`. Samples are stored under `backend/data`. |
| POST   | `/detect_wakeword`| Multipart audio (`file`). Runs the wakeword PyTorch model and returns `{ wakeword_detected, confidence }`. |
| POST   | `/detect_command` | Multipart audio (`file`). Classifies between `open_door` and `close_door`. |
| POST   | `/verify_voice`   | Multipart audio (`file`). Uses the TensorFlow/Keras biometric model to verify the speaker. |
| POST   | `/report_command` | Optional debug hook to log the final command emitted by the app. |

All endpoints are CORS-enabled for local development so the Flutter emulator (or the web frontend) can call them via `http://10.0.2.2:8000`.

## Running locally (PowerShell)

```powershell
cd C:\vscodeprojects\home_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Alternatively, you can run `python backend/run_server.py` which launches uvicorn with the same settings.

## Adding/Updating models

1. Copy your trained checkpoints into `backend/models/`:
   - `wakeword_model.pth`
   - `command_model.pth`
   - `voice_biometric_model.h5` and `voice_biometric_labels.json`
2. (Optional) Provide `command_labels.json` if your command model uses a different label map. See `DEFAULT_COMMAND_LABELS` in `app.py` for the expected shape.
3. Restart the backend. The `/health` endpoint will tell you which models were loaded (`wake_model_source`, `command_model_source`, `biometric_model_source`).

The backend automatically falls back to placeholder PyTorch weights when the `.pth` files are missing, so the Flutter app can still exercise the full pipeline while you gather samples. Replace those placeholders with real checkpoints before deploying to production hardware.
