"""FastAPI backend for the Smart Door Lock voice pipeline.

The backend exposes three ML-powered services:
1. Wake-word detection
2. Command intent classification (open / close door)
3. Voice biometric verification (shared across wake-word + command flow)

It also provides a sample collection endpoint so the Flutter app can upload
fresh data to `backend/data/` for retraining.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import torch
import torch.nn as nn
import torchaudio
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_COMMAND_LABELS: Dict[str, int] = {"open_door": 0, "close_door": 1}
BIOMETRIC_THRESHOLD = 0.7
ALLOWED_SAMPLE_LABELS = {"wakeword", "open_door", "close_door"}


def _ensure_data_layout() -> None:
    (DATA_DIR / "wakeword").mkdir(parents=True, exist_ok=True)
    for command_label in DEFAULT_COMMAND_LABELS.keys():
        (DATA_DIR / "command" / command_label).mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


_ensure_data_layout()


def _resolve_model_path(env_key: str, default: Path) -> Optional[Path]:
    candidates = [os.environ.get(env_key), str(default)]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            print(f"[models] Using {env_key or 'default'} -> {path}")
            return path
    return None


class WakeWordModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CommandModel(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _load_wake_model() -> Tuple[WakeWordModel, str]:
    path = _resolve_model_path("WAKE_MODEL_PATH", MODELS_DIR / "wakeword_model.pth")
    model = WakeWordModel()
    source = "placeholder"
    if path:
        try:
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state)
            source = "file"
        except Exception as exc:
            print(f"[models] Failed to load wakeword model ({path}): {exc}")
    model.eval()
    return model, source


def _load_command_model() -> Tuple[CommandModel, Dict[str, int], str]:
    path = _resolve_model_path("COMMAND_MODEL_PATH", MODELS_DIR / "command_model.pth")
    labels_map = DEFAULT_COMMAND_LABELS.copy()
    label_path = _resolve_model_path("COMMAND_LABELS_PATH", MODELS_DIR / "command_labels.json")
    if label_path:
        try:
            labels_map = json.loads(label_path.read_text())
            labels_map = {str(k): int(v) for k, v in labels_map.items()}
            print(f"[models] Loaded command labels -> {labels_map}")
        except Exception as exc:
            print(f"[models] Failed to parse command labels {label_path}: {exc}")
    model = CommandModel(num_classes=max(labels_map.values()) + 1)
    model.fc2 = nn.Linear(model.fc1.out_features, max(labels_map.values()) + 1)
    source = "placeholder"
    if path:
        try:
            checkpoint = torch.load(path, map_location="cpu")
            if "labels_map" in checkpoint:
                labels_map = {str(k): int(v) for k, v in checkpoint["labels_map"].items()}
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            source = "file"
        except Exception as exc:
            print(f"[models] Failed to load command model ({path}): {exc}")
    model.eval()
    return model, labels_map, source


def _load_biometric_model() -> Tuple[Optional[tf.keras.Model], Optional[Dict[str, int]], str]:
    model_path = _resolve_model_path(
        "VOICE_BIOMETRIC_MODEL_PATH", MODELS_DIR / "voice_biometric_model.h5"
    )
    labels_path = _resolve_model_path(
        "VOICE_BIOMETRIC_LABELS_PATH", MODELS_DIR / "voice_biometric_labels.json"
    )
    if model_path and labels_path:
        try:
            model = tf.keras.models.load_model(model_path)
            labels = json.loads(labels_path.read_text())
            labels = {str(k): int(v) for k, v in labels.items()}
            print(f"[models] Loaded biometric model -> {model_path}")
            return model, labels, "file"
        except Exception as exc:
            print(f"[models] Failed to load biometric model: {exc}")
    return None, None, "missing"


WAKE_MODEL, WAKE_MODEL_SOURCE = _load_wake_model()
COMMAND_MODEL, COMMAND_LABELS, COMMAND_MODEL_SOURCE = _load_command_model()
BIOMETRIC_MODEL, BIOMETRIC_LABELS, BIOMETRIC_MODEL_SOURCE = _load_biometric_model()


def preprocess_audio_bytes(
    audio_bytes: bytes, target_sr: int = 16000
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        waveform = torch.from_numpy(data).unsqueeze(0)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_mels=64, n_fft=400, hop_length=160
        )(waveform)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = mel.unsqueeze(0)
        return mel, target_sr
    except Exception as exc:
        print(f"[audio] Failed to preprocess audio: {exc}")
        return None, None


def _extract_mfcc(audio_bytes: bytes, sr: int = 16000, n_mfcc: int = 13) -> Optional[np.ndarray]:
    try:
        data, src_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        data = data / 32768.0
        src_sr = 48000
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if src_sr != sr:
        data = librosa.resample(data, orig_sr=src_sr, target_sr=sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0, keepdims=True)


def _label_from_index(mapping: Dict[str, int], idx: int) -> Optional[str]:
    for label, value in mapping.items():
        if value == idx:
            return label
    return None


app = FastAPI(title="Smart Door Lock Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> Dict[str, object]:
    return {
        "status": "ok",
        "wake_model_source": WAKE_MODEL_SOURCE,
        "command_model_source": COMMAND_MODEL_SOURCE,
        "biometric_model_source": BIOMETRIC_MODEL_SOURCE,
        "data_dirs": {
            "wakeword": str((DATA_DIR / "wakeword").resolve()),
            "command": str((DATA_DIR / "command").resolve()),
        },
    }


@app.post("/collect_sample")
async def collect_sample(file: UploadFile = File(...), label: str = Form(...)) -> Dict[str, object]:
    normalized_label = label.strip().lower()
    if normalized_label not in ALLOWED_SAMPLE_LABELS:
        return JSONResponse(
            {"error": f'label must be one of {sorted(ALLOWED_SAMPLE_LABELS)}'},
            status_code=400,
        )

    target_dir = DATA_DIR / "wakeword" if normalized_label == "wakeword" else DATA_DIR / "command" / normalized_label
    target_dir.mkdir(parents=True, exist_ok=True)

    contents = await file.read()
    if len(contents) < 200:
        return JSONResponse({"error": "audio file too small"}, status_code=400)

    try:
        sf.read(io.BytesIO(contents))
    except Exception as exc:
        return JSONResponse({"error": f"invalid audio: {exc}"}, status_code=400)

    filename = f"sample_{int(time.time() * 1000)}.wav"
    destination = target_dir / filename
    destination.write_bytes(contents)
    print(f"[data] Saved {normalized_label} sample -> {destination}")
    return {"saved": str(destination), "label": normalized_label}


@app.post("/detect_wakeword")
async def detect_wakeword(file: UploadFile = File(...)) -> Dict[str, object]:
    audio_bytes = await file.read()
    mel, _ = preprocess_audio_bytes(audio_bytes)
    if mel is None:
        return JSONResponse({"error": "invalid audio"}, status_code=400)
    with torch.no_grad():
        logits = WAKE_MODEL(mel)
        probs = torch.softmax(logits, dim=1)
        confidence = float(probs[0, 1].item())
    detected = confidence >= 0.5
    return {
        "wakeword_detected": bool(detected),
        "confidence": confidence,
        "model_source": WAKE_MODEL_SOURCE,
    }


@app.post("/detect_command")
async def detect_command(file: UploadFile = File(...)) -> Dict[str, object]:
    audio_bytes = await file.read()
    mel, _ = preprocess_audio_bytes(audio_bytes)
    if mel is None:
        return JSONResponse({"error": "invalid audio"}, status_code=400)
    with torch.no_grad():
        logits = COMMAND_MODEL(mel)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item())
    intent = _label_from_index(COMMAND_LABELS, pred_idx)
    return {
        "intent": intent,
        "intent_idx": pred_idx,
        "confidence": confidence,
        "model_source": COMMAND_MODEL_SOURCE,
    }


@app.post("/verify_voice")
async def verify_voice(file: UploadFile = File(...)) -> Dict[str, object]:
    if BIOMETRIC_MODEL is None or BIOMETRIC_LABELS is None:
        return JSONResponse({"error": "voice biometric model not available"}, status_code=500)
    audio_bytes = await file.read()
    features = _extract_mfcc(audio_bytes)
    if features is None:
        return JSONResponse({"error": "invalid audio"}, status_code=400)
    predictions = BIOMETRIC_MODEL.predict(features, verbose=0)
    confidence = float(np.max(predictions[0]))
    predicted_idx = int(np.argmax(predictions[0]))
    owner = _label_from_index(BIOMETRIC_LABELS, predicted_idx)
    verified = confidence >= BIOMETRIC_THRESHOLD
    return {
        "verified": bool(verified),
        "confidence": confidence,
        "owner": owner,
        "threshold": BIOMETRIC_THRESHOLD,
        "model_source": BIOMETRIC_MODEL_SOURCE,
    }


@app.post("/report_command")
async def report_command(request: Request) -> Dict[str, object]:
    payload = await request.json()
    print(f"[command] {payload}")
    return {"received": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
