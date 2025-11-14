# Voice Assistant - Starter

This workspace contains starter scripts to train two simple audio models and a FastAPI backend to serve them.

Files added:
- `train_wakeword.py` - train a simple wake word detector
- `train_command_model.py` - train a simple speech-to-intent classifier
- `backend/app.py` - FastAPI server exposing `/detect_wakeword` and `/detect_command`
- `frontend/index.html` - minimal UI to upload audio and call backend endpoints
- `requirements.txt` - dependencies

Quick start (Windows PowerShell):

1. Create and activate a virtual environment (you already have one in `speechbrain-env/` in this workspace):

```powershell
python -m venv voiceassistant-env; .\voiceassistant-env\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Prepare data folders:

- `data/wakewords/train/wakeword` and `data/wakewords/train/noise` with WAV files
- `data/wakewords/test/...` for evaluation
- `data/commands/train/<intent_name>/*.wav`

4. Train models:

```powershell
python train_wakeword.py --data data/wakewords --epochs 10 --batch 8
python train_command_model.py --data data/commands/train --epochs 12 --batch 8
```

5. Start backend (from repository root):

```powershell
uvicorn backend.app:app --reload
```

6. Open `frontend/index.html` in a browser (or serve it via a simple static server) and upload audio files to test.

Notes:
- These are starter models. For production use, collect more data, add augmentation, and refine architectures.
- The model shapes assume small mel-spectrograms; you may need to adjust pooling or linear sizes depending on input length.
