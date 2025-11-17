#!/usr/bin/env python3
"""Test all backend endpoints using Python requests library"""

import requests
import os
import time
from pathlib import Path

BASE_URL = 'http://127.0.0.1:8000'
BACKEND_DIR = r'C:\vscodeprojects\home_app\backend'

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(endpoint, method, status, response):
    print(f"\n[{method} {endpoint}]")
    print(f"Status: {status}")
    print(f"Response: {response}")

def test_health():
    """Test GET /health"""
    print_section("TEST 1: Health Check")
    print("GET /health")
    print("Testing if backend is running...")
    
    try:
        response = requests.get(f'{BASE_URL}/health', timeout=5)
        print_result('/health', 'GET', response.status_code, response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_test_endpoint():
    """Test POST /test"""
    print_section("TEST 2: Test Endpoint")
    print("POST /test")
    print("Testing if backend responds to POST...")
    
    try:
        response = requests.post(f'{BASE_URL}/test', timeout=5)
        print_result('/test', 'POST', response.status_code, response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_collect_sample():
    """Test POST /collect_sample"""
    print_section("TEST 3: Collect Sample (Upload Audio)")
    
    # Find an existing sample or create a dummy one
    sample_path = None
    sample_dir = os.path.join(BACKEND_DIR, 'data', 'wakeword')
    
    if os.path.exists(sample_dir):
        wav_files = list(Path(sample_dir).glob('*.wav'))
        if wav_files:
            sample_path = str(wav_files[0])
    
    if not sample_path:
        print("No sample WAV file found to test upload")
        print(f"Expected location: {sample_dir}/*.wav")
        return False
    
    print(f"Using sample file: {sample_path}")
    
    try:
        with open(sample_path, 'rb') as f:
            files = {'file': f}
            data = {'label': 'wakeword'}
            response = requests.post(
                f'{BASE_URL}/collect_sample',
                files=files,
                data=data,
                timeout=10
            )
        
        print_result('/collect_sample', 'POST', response.status_code, response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_detect_wakeword():
    """Test POST /detect_wakeword"""
    print_section("TEST 4: Detect Wakeword")
    
    sample_path = None
    sample_dir = os.path.join(BACKEND_DIR, 'data', 'wakeword')
    
    if os.path.exists(sample_dir):
        wav_files = list(Path(sample_dir).glob('*.wav'))
        if wav_files:
            sample_path = str(wav_files[0])
    
    if not sample_path:
        print("No sample WAV file found to test wakeword detection")
        return False
    
    print(f"Using sample file: {sample_path}")
    
    try:
        with open(sample_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f'{BASE_URL}/detect_wakeword',
                files=files,
                timeout=10
            )
        
        print_result('/detect_wakeword', 'POST', response.status_code, response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_detect_command():
    """Test POST /detect_command"""
    print_section("TEST 5: Detect Command")
    
    sample_path = None
    sample_dir = os.path.join(BACKEND_DIR, 'data', 'wakeword')
    
    if os.path.exists(sample_dir):
        wav_files = list(Path(sample_dir).glob('*.wav'))
        if wav_files:
            sample_path = str(wav_files[0])
    
    if not sample_path:
        print("No sample WAV file found to test command detection")
        return False
    
    print(f"Using sample file: {sample_path}")
    
    try:
        with open(sample_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f'{BASE_URL}/detect_command',
                files=files,
                timeout=10
            )
        
        print_result('/detect_command', 'POST', response.status_code, response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_verify_voice():
    """Test POST /verify_voice"""
    print_section("TEST 6: Verify Voice (Biometric)")
    
    sample_path = None
    sample_dir = os.path.join(BACKEND_DIR, 'data', 'wakeword')
    
    if os.path.exists(sample_dir):
        wav_files = list(Path(sample_dir).glob('*.wav'))
        if wav_files:
            sample_path = str(wav_files[0])
    
    if not sample_path:
        print("No sample WAV file found to test voice verification")
        return False
    
    print(f"Using sample file: {sample_path}")
    
    try:
        with open(sample_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f'{BASE_URL}/verify_voice',
                files=files,
                timeout=10
            )
        
        print_result('/verify_voice', 'POST', response.status_code, response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "BACKEND API ENDPOINT TESTER" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print(f"\nBackend URL: {BASE_URL}")
    print(f"Backend Dir: {BACKEND_DIR}")
    
    results = {}
    
    # Test all endpoints
    results['Health Check'] = test_health()
    time.sleep(0.5)
    
    results['Test Endpoint'] = test_test_endpoint()
    time.sleep(0.5)
    
    results['Collect Sample'] = test_collect_sample()
    time.sleep(1)
    
    results['Detect Wakeword'] = test_detect_wakeword()
    time.sleep(1)
    
    results['Detect Command'] = test_detect_command()
    time.sleep(1)
    
    results['Verify Voice'] = test_verify_voice()
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED! Backend is working correctly.")
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
    
    print("\n")

if __name__ == '__main__':
    main()
