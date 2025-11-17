#!/usr/bin/env python3
"""Master startup script: Run backend, then frontend"""

import subprocess
import time
import requests
import sys
import os

def check_backend_health():
    """Check if backend is responding"""
    try:
        response = requests.get('http://127.0.0.1:8000/health', timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    print("=" * 70)
    print(" " * 20 + "VOICE APP STARTUP")
    print("=" * 70)
    
    # Check if emulator is running
    print("\n[STEP 1] Checking if emulator is running...")
    result = subprocess.run(
        ['cmd', '/c', 'flutter', 'devices'],
        capture_output=True,
        text=True,
        cwd=r'C:\vscodeprojects\home_app',
        shell=True
    )
    
    if 'emulator-5554' not in result.stdout:
        print("✗ Emulator 'emulator-5554' not found!")
        print("Available devices:")
        print(result.stdout)
        print("\nTo start emulator:")
        print("  - Open Android Studio")
        print("  - Click AVD Manager")
        print("  - Launch the emulator")
        return False
    print("✓ Emulator 'emulator-5554' is running")
    
    # Start backend server
    print("\n[STEP 2] Starting backend server...")
    backend_proc = subprocess.Popen(
        [
            sys.executable, '-m', 'uvicorn',
            'backend.app:app',
            '--host', '127.0.0.1',
            '--port', '8000'
        ],
        cwd=r'C:\vscodeprojects\home_app',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for backend to start
    print("  Waiting for backend to initialize...")
    for i in range(10):  # Try for 10 seconds
        time.sleep(1)
        if check_backend_health():
            print("✓ Backend is running on http://127.0.0.1:8000")
            break
    else:
        print("✗ Backend failed to start!")
        print("Error output:")
        stdout, stderr = backend_proc.communicate(timeout=2)
        if stderr:
            print(stderr[:500])
        return False
    
    # Test backend endpoints
    print("\n[STEP 3] Testing backend endpoints...")
    endpoints = ['/health', '/test']
    for endpoint in endpoints:
        try:
            r = requests.post(f'http://127.0.0.1:8000{endpoint}', timeout=2)
            if r.status_code == 200:
                print(f"  ✓ {endpoint} is working")
            else:
                print(f"  ✗ {endpoint} returned {r.status_code}")
        except Exception as e:
            print(f"  ✗ {endpoint} failed: {e}")
    
    # Data directory structure
    print("\n[STEP 4] Checking data directory structure...")
    backend_dir = r'C:\vscodeprojects\home_app\backend'
    data_labels = ['wakeword', 'command', 'open_door', 'close_door']
    for label in data_labels:
        data_path = os.path.join(backend_dir, 'data', label)
        os.makedirs(data_path, exist_ok=True)
        if os.path.exists(data_path):
            print(f"  ✓ data/{label}/ directory ready")
    
    # Start Flutter app
    print("\n[STEP 5] Starting Flutter app on emulator...")
    print("  (This may take a minute...)\n")
    
    flutter_proc = subprocess.Popen(
        ['cmd', '/c', 'flutter', 'run', '-d', 'emulator-5554'],
        cwd=r'C:\vscodeprojects\home_app',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True
    )
    
    # Read and print flutter output
    try:
        for line in flutter_proc.stdout:
            print(line.rstrip())
    except KeyboardInterrupt:
        print("\n\n[STOPPING] Shutting down...")
        flutter_proc.terminate()
    finally:
        flutter_proc.wait()
        backend_proc.terminate()
        print("\n✓ Backend and Flutter app stopped")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
