#!/usr/bin/env python3
"""
Voice Biometric System Verification & Setup Check
Verifies all components are installed and ready to use
"""

import os
import sys
import json

def check_files():
    print("\n" + "="*70)
    print("📋 VOICE BIOMETRIC SYSTEM SETUP VERIFICATION")
    print("="*70)
    
    files_to_check = [
        ('Frontend', 'frontend/index.html'),
        ('Backend API', 'backend/app.py'),
        ('Training Script', 'train_voice_biometric.py'),
        ('Biometric Model (H5)', 'models/voice_biometric_model.h5'),
        ('Biometric Labels', 'models/voice_biometric_labels.json'),
        ('TFLite Model (Embedded)', 'models/voice_biometric_model.tflite'),
        ('Owner Wakeword Samples', 'data/wakeword/owner'),
        ('Owner Command Samples', 'data/command/owner'),
    ]
    
    print("\n✅ FILE VERIFICATION:")
    all_exist = True
    for name, path in files_to_check:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name:40s} {path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n  All required files found! ✅")
    else:
        print("\n  ⚠ Some files missing. Run setup steps.")
    
    return all_exist

def check_imports():
    print("\n✅ PYTHON DEPENDENCIES:")
    deps = [
        ('librosa', 'Audio feature extraction'),
        ('numpy', 'Numerical computing'),
        ('tensorflow', 'Deep learning (MFCC + model)'),
        ('soundfile', 'Audio I/O'),
        ('fastapi', 'Backend API'),
        ('uvicorn', 'ASGI server'),
        ('torch', 'PyTorch (for backend components)'),
        ('torchaudio', 'Audio processing'),
    ]
    
    all_ok = True
    for package, description in deps:
        try:
            __import__(package)
            print(f"  ✓ {package:15s} - {description}")
        except ImportError:
            print(f"  ✗ {package:15s} - {description} [MISSING]")
            all_ok = False
    
    if all_ok:
        print("\n  All dependencies installed! ✅")
    else:
        print("\n  ⚠ Install missing packages: pip install librosa tensorflow")
    
    return all_ok

def check_model_metadata():
    print("\n✅ MODEL METADATA:")
    labels_path = 'models/voice_biometric_labels.json'
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        print(f"  Voice Biometric Labels: {labels}")
        print(f"  Owner Identity: {list(labels.keys())[0] if labels else 'UNKNOWN'}")
        print(f"  Number of owners: {len(labels)}")
        return True
    else:
        print("  ✗ Labels file not found (run training)")
        return False

def check_urls():
    print("\n✅ ACCESS ENDPOINTS:")
    print(f"  Frontend URL: http://127.0.0.1:8000/")
    print(f"  Backend Health: http://127.0.0.1:8000/health")
    print(f"  Biometric Verify: POST http://127.0.0.1:8000/verify_voice")
    print(f"  Command Report: POST http://127.0.0.1:8000/report_command")

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    
    files_ok = check_files()
    deps_ok = check_imports()
    model_ok = check_model_metadata()
    check_urls()
    
    print("\n" + "="*70)
    if files_ok and deps_ok and model_ok:
        print("✅ SYSTEM READY! All components verified.")
        print("\nNext steps:")
        print("  1. Start backend: python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000")
        print("  2. Open frontend: http://127.0.0.1:8000/")
        print("  3. Say 'Hey Jarvis' to trigger biometric check")
        print("  4. On verified, say 'open' or 'close'")
        return 0
    else:
        print("⚠ SETUP INCOMPLETE. Run setup steps first.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
