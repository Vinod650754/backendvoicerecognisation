"""
Configuration settings for the backend application
"""
import os
from pathlib import Path
from typing import Optional

# App Settings
APP_NAME = "Voice Authentication Backend"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Backend for voice-based door authentication system"

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure data directories exist
for data_subdir in ["wakeword", "command", "open_door", "close_door"]:
    (DATA_DIR / data_subdir).mkdir(parents=True, exist_ok=True)

# Server Settings
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
RELOAD = os.getenv("RELOAD", "false").lower() == "true"

# CORS Settings
CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://10.0.2.2",  # Android emulator
    "http://10.0.2.2:8000",
]

# Audio Settings
TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_SECONDS = 0.5
MAX_AUDIO_DURATION_SECONDS = 30
MIN_AUDIO_FILE_SIZE = 100  # bytes

# Model Settings
WAKEWORD_CONFIDENCE_THRESHOLD = 0.7
COMMAND_CONFIDENCE_THRESHOLD = 0.7
VOICE_VERIFICATION_THRESHOLD = 0.7

# Audio Processing
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"
