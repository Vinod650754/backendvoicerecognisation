#!/usr/bin/env python3
"""Run backend server and test endpoints"""

import subprocess
import time
import requests
import sys

print("=" * 60)
print("STARTING BACKEND SERVER")
print("=" * 60)

# Start server in background
print("\n[1] Starting FastAPI server on http://127.0.0.1:8000...")
proc = subprocess.Popen(
    [
        sys.executable, '-m', 'uvicorn',
        'backend.app:app',
        '--host', '127.0.0.1',
        '--port', '8000',
        '--log-level', 'info'
    ],
    cwd=r'C:\vscodeprojects\home_app',
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Wait for server to start
print("[2] Waiting 3 seconds for server to initialize...")
time.sleep(3)

# Test health endpoint
print("\n[3] Testing /health endpoint...")
try:
    response = requests.get('http://127.0.0.1:8000/health', timeout=5)
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Response: {response.json()}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    # Print server logs
    print("\n--- SERVER OUTPUT ---")
    print(proc.stdout.read())
    print("\n--- SERVER ERRORS ---")
    print(proc.stderr.read())
    proc.terminate()
    sys.exit(1)

print("\n[4] Testing /test endpoint...")
try:
    response = requests.post('http://127.0.0.1:8000/test', timeout=5)
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Response: {response.json()}")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 60)
print("✓ BACKEND IS WORKING CORRECTLY!")
print("=" * 60)
print("\nServer is running on http://127.0.0.1:8000")
print("Press Ctrl+C to stop the server...")

try:
    proc.wait()
except KeyboardInterrupt:
    print("\n[STOPPED] Terminating server...")
    proc.terminate()
    proc.wait()
    print("Server stopped.")
