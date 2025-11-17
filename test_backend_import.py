#!/usr/bin/env python3
"""Test if backend imports work correctly"""

import sys
import traceback

print("=" * 60)
print("TESTING BACKEND IMPORTS")
print("=" * 60)

try:
    print("\n[1/7] Importing FastAPI...")
    from fastapi import FastAPI
    print("✓ FastAPI imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[2/7] Importing torch...")
    import torch
    print(f"✓ PyTorch imported successfully (version: {torch.__version__})")
except Exception as e:
    print(f"✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[3/7] Importing torchaudio...")
    import torchaudio
    print(f"✓ torchaudio imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[4/7] Importing soundfile...")
    import soundfile as sf
    print("✓ soundfile imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[5/7] Importing librosa...")
    import librosa
    print("✓ librosa imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[6/7] Importing uvicorn...")
    import uvicorn
    print("✓ uvicorn imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[7/7] Importing backend.app...")
    from backend.app import app
    print("✓ backend.app imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL IMPORTS SUCCESSFUL - Backend should work!")
print("=" * 60)
