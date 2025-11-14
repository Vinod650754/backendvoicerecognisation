import io
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form
from fastapi import Request
import torch
import torchaudio
import soundfile as sf
import torch.nn as nn
import librosa
import numpy as np
import tensorflow as tf
import json

app = FastAPI()

# Enable CORS for the frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models must match definitions used in training scripts
class WakeWordModel(nn.Module):
    def __init__(self):
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

    def forward(self, x):
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
    def __init__(self, num_classes):
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
        self.fc2 = nn.Linear(128, 2)  # placeholder, will be resized when loading

    def forward(self, x):
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


def load_wake_model(path):
    if not os.path.exists(path):
        return None
    model = WakeWordModel()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def load_command_model(path):
    if not os.path.exists(path):
        return None, None
    ckpt = torch.load(path, map_location='cpu')
    labels = ckpt.get('labels_map', None)
    num_classes = len(labels) if labels else 2
    model = CommandModel(num_classes=num_classes)
    # Resize final layer if needed
    model.fc2 = nn.Linear(model.fc1.out_features, num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, labels


# Try to load models from expected paths
def find_first_existing(paths):
    for p in paths:
        if p is None:
            continue
        try:
            ap = os.path.abspath(p)
        except Exception:
            ap = p
        if os.path.exists(ap):
            print(f"Found checkpoint: {ap}")
            return ap
    return None


# Candidate paths (environment variables take precedence)
WAKE_ENV = os.environ.get('WAKE_MODEL_PATH')
CMD_ENV = os.environ.get('COMMAND_MODEL_PATH')

cwd = os.getcwd()
wake_candidates = [WAKE_ENV,
                   os.path.join(cwd, 'data', 'wakewords', 'wakeword_model.pth'),
                   os.path.join(cwd, 'data', 'wakewords', 'wakeword_model.pth'),
                   os.path.join(cwd, '..', 'data', 'wakewords', 'wakeword_model.pth')]

cmd_candidates = [CMD_ENV,
                  os.path.join(cwd, 'data', 'commands', 'train', 'command_model.pth'),
                  os.path.join(cwd, 'data', 'commands', 'command_model.pth'),
                  os.path.join(cwd, '..', 'data', 'commands', 'train', 'command_model.pth')]

WAKE_PATH = find_first_existing(wake_candidates)
CMD_PATH = find_first_existing(cmd_candidates)

print('WAKE_PATH ->', WAKE_PATH)
print('CMD_PATH  ->', CMD_PATH)

wake_model = load_wake_model(WAKE_PATH) if WAKE_PATH else None
command_model, command_labels = load_command_model(CMD_PATH) if CMD_PATH else (None, None)

# Load voice biometric model
biometric_model = None
biometric_labels = None
biometric_threshold = 0.7  # Confidence threshold for voice verification

try:
    # Get the parent directory (project root) from the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    biometric_model_path = os.path.join(project_root, 'models', 'voice_biometric_model.h5')
    biometric_labels_path = os.path.join(project_root, 'models', 'voice_biometric_labels.json')
    
    if os.path.exists(biometric_model_path) and os.path.exists(biometric_labels_path):
        biometric_model = tf.keras.models.load_model(biometric_model_path)
        with open(biometric_labels_path, 'r') as f:
            biometric_labels = json.load(f)
        print(f"✓ Loaded voice biometric model from {biometric_model_path}")
        print(f"  Voice biometric labels: {biometric_labels}")
    else:
        print("⚠ Voice biometric model not found. Path:", biometric_model_path)
except Exception as e:
    print(f"✗ Error loading voice biometric model: {e}")

# Serve frontend static files under /static and expose index at /
frontend_dir = os.path.abspath(os.path.join(os.getcwd(), 'frontend'))
index_path = os.path.join(frontend_dir, 'index.html')
if os.path.isdir(frontend_dir):
    app.mount('/static', StaticFiles(directory=frontend_dir), name='static')
    print('Serving frontend static from', frontend_dir)


@app.get('/')
def root():
    # Return the frontend index.html if it exists
    if os.path.exists(index_path):
        from fastapi.responses import FileResponse
        return FileResponse(index_path, media_type='text/html')
    return {'message': 'Frontend not available'}


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'wake_model_loaded': bool(wake_model),
        'command_model_loaded': bool(command_model),
    }



@app.post('/collect_sample')
async def collect_sample(file: UploadFile = File(...), sample_type: str = Form(...), label: str = Form(...), dataset: str = Form('train')):
    """Save an uploaded audio sample to disk under data/<sample_type>/<dataset>/<label>/filename.wav
    sample_type: 'wakewords' or 'commands'
    label: for wakewords use 'wakeword' or 'noise'; for commands use 'open_door' or 'close_door'
    dataset: 'train' or 'test'
    """
    # basic validation
    if sample_type not in ('wakewords', 'commands'):
        return JSONResponse({'error': 'sample_type must be wakewords or commands'}, status_code=400)
    if dataset not in ('train', 'test'):
        dataset = 'train'

    # build target dir
    target_dir = os.path.join(os.getcwd(), 'data', sample_type, dataset)
    if sample_type == 'wakewords':
        # labels expected: 'wakeword' or 'noise'
        target_dir = os.path.join(target_dir, label)
    else:
        # commands: label like 'open_door' or 'close_door'
        target_dir = os.path.join(target_dir, label)

    os.makedirs(target_dir, exist_ok=True)

    # save file with a timestamped name
    import time
    filename = f"sample_{int(time.time()*1000)}.wav"
    dest_path = os.path.join(target_dir, filename)
    contents = await file.read()
    with open(dest_path, 'wb') as f:
        f.write(contents)

    return {'saved': dest_path}


def preprocess_audio_bytes(audio_bytes):
    # torchaudio accepts a file-like object
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)
    except Exception:
        return None, None
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0), sr


@app.post('/detect_wakeword')
async def detect_wakeword(file: UploadFile = File(...)):
    if wake_model is None:
        return JSONResponse({'error': 'wake model not found on server'}, status_code=500)
    audio = await file.read()
    mel, sr = preprocess_audio_bytes(audio)
    if mel is None:
        return JSONResponse({'error': 'invalid audio'}, status_code=400)
    with torch.no_grad():
        out = wake_model(mel)
        pred = int(out.argmax(dim=1).item())
    return {'wakeword_detected': bool(pred)}


@app.post('/detect_command')
async def detect_command(file: UploadFile = File(...)):
    if command_model is None:
        return JSONResponse({'error': 'command model not found on server'}, status_code=500)
    audio = await file.read()
    mel, sr = preprocess_audio_bytes(audio)
    if mel is None:
        return JSONResponse({'error': 'invalid audio'}, status_code=400)
    with torch.no_grad():
        out = command_model(mel)
        idx = int(out.argmax(dim=1).item())
    intent = None
    if command_labels:
        intent = [k for k, v in command_labels.items() if v == idx][0]
    return {'intent': intent, 'intent_idx': idx}


def extract_mfcc_for_biometric(audio_bytes, n_mfcc=13, sr=16000):
    """Extract MFCC features from audio bytes."""
    try:
        print(f"[DEBUG] Received audio bytes: {len(audio_bytes)} bytes, first 20: {audio_bytes[:20]}")
        data, sr_loaded = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        print(f"[DEBUG] Loaded audio: shape={data.shape}, sr={sr_loaded}")
        if sr_loaded != sr:
            # Resample if needed
            print(f"[DEBUG] Resampling from {sr_loaded} to {sr}")
            data = librosa.resample(data, orig_sr=sr_loaded, target_sr=sr)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        print(f"[DEBUG] MFCC shape: {mfcc_mean.reshape(1, -1).shape}")
        return mfcc_mean.reshape(1, -1)  # Shape (1, n_mfcc) for prediction
    except Exception as e:
        print(f"Error extracting MFCC: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.post('/verify_voice')
async def verify_voice(file: UploadFile = File(...)):
    """Verify if the voice in the audio matches the owner's voice.
    Returns: {verified: bool, confidence: float, owner: str or None}
    """
    if biometric_model is None or biometric_labels is None:
        return JSONResponse({'error': 'voice biometric model not loaded on server'}, status_code=500)
    
    try:
        print(f"[VERIFY_VOICE] Received file: {file.filename}, content_type={file.content_type}")
        audio = await file.read()
        print(f"[VERIFY_VOICE] Audio data received: {len(audio)} bytes")
        
        mfcc_feature = extract_mfcc_for_biometric(audio)
        
        if mfcc_feature is None:
            print(f"[VERIFY_VOICE] Failed to extract MFCC from audio")
            return JSONResponse({'error': 'invalid audio - could not extract MFCC'}, status_code=400)
        
        # Predict
        predictions = biometric_model.predict(mfcc_feature, verbose=0)
        confidence = float(np.max(predictions[0]))
        predicted_label_idx = int(np.argmax(predictions[0]))
        
        # Map label index to owner name
        owner_name = None
        for label, idx in biometric_labels.items():
            if idx == predicted_label_idx:
                owner_name = label
                break
        
        verified = confidence >= biometric_threshold
        print(f"[BIOMETRIC CHECK] owner={owner_name} confidence={confidence:.4f} verified={verified}")
        
        return {
            'verified': verified,
            'confidence': confidence,
            'owner': owner_name,
            'threshold': biometric_threshold
        }
    except Exception as e:
        print(f"Error in voice verification: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)


@app.post('/test_audio_upload')
async def test_audio_upload(file: UploadFile = File(...)):
    """Debug endpoint - just echo back what was received"""
    print(f"[TEST_AUDIO] Filename: {file.filename}")
    print(f"[TEST_AUDIO] Content-Type: {file.content_type}")
    audio = await file.read()
    print(f"[TEST_AUDIO] Received {len(audio)} bytes")
    print(f"[TEST_AUDIO] First 50 bytes (hex): {audio[:50].hex()}")
    print(f"[TEST_AUDIO] First 50 bytes (text): {repr(audio[:50])}")
    
    # Try to read as WAV
    try:
        data, sr = sf.read(io.BytesIO(audio), dtype='float32')
        print(f"[TEST_AUDIO] ✓ Parsed as WAV: shape={data.shape}, sr={sr}")
        return {'status': 'ok', 'shape': str(data.shape), 'sample_rate': sr}
    except Exception as e:
        print(f"[TEST_AUDIO] ✗ Failed to parse: {type(e).__name__}: {e}")
        return JSONResponse({'status': 'error', 'error': str(e)}, status_code=400)


@app.post('/report_command')
async def report_command(request: Request):
    """Receive a JSON command event from the frontend and log it to the server terminal.
    Expected JSON: {"command": "open_door", "raw": "open"}
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({'error': 'invalid json'}, status_code=400)
    command = payload.get('command')
    raw = payload.get('raw')
    ts = payload.get('timestamp')
    print(f"[COMMAND EVENT] command={command} raw={raw} timestamp={ts}")
    return {'received': True}
