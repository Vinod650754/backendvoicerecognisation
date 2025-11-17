#!/usr/bin/env python3
"""
Comprehensive test client for the voice authentication backend
Tests all endpoints and provides detailed output
"""

import os
import sys
import time
import requests
from pathlib import Path
from typing import Optional
import json

# Backend URL
BASE_URL = "http://127.0.0.1:8000"

# Test data
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.YELLOW}→ {text}{Colors.END}")


def test_health_check() -> bool:
    """Test the /health endpoint"""
    print_header("TEST 1: Health Check")
    
    try:
        print_info(f"Requesting: GET {BASE_URL}/health")
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Backend is healthy!")
            print(f"  Status: {data.get('status')}")
            print(f"  Message: {data.get('message')}")
            print(f"  Version: {data.get('version')}")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Failed to connect to backend")
        print_info(f"Make sure backend is running at {BASE_URL}")
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_sample_upload(audio_file: Optional[Path] = None) -> bool:
    """Test the /collect_sample endpoint"""
    print_header("TEST 2: Sample Upload")
    
    # Try to find a test audio file
    if audio_file is None:
        possible_files = [
            Path("backend/data/wakeword/sample_1763282265861.wav"),
            Path("../backend/data/wakeword/sample_1763282265861.wav"),
            Path("test_audio.wav"),
        ]
        for pf in possible_files:
            if pf.exists():
                audio_file = pf
                break
    
    if audio_file is None or not audio_file.exists():
        print_error("No test audio file found")
        print_info("Generate an audio file or provide one as argument")
        return False
    
    try:
        print_info(f"Uploading audio file: {audio_file}")
        
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {'label': 'wakeword'}
            
            response = requests.post(
                f"{BASE_URL}/collect_sample",
                files=files,
                data=data,
                timeout=10
            )
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Sample uploaded successfully!")
            print(f"  Message: {result.get('message')}")
            print(f"  Path: {result.get('path')}")
            print(f"  Label: {result.get('label')}")
            return True
        else:
            print_error(f"Upload failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_wakeword_detection(audio_file: Optional[Path] = None) -> bool:
    """Test the /detect_wakeword endpoint"""
    print_header("TEST 3: Wakeword Detection")
    
    # Try to find a test audio file
    if audio_file is None:
        possible_files = [
            Path("backend/data/wakeword/sample_1763282265861.wav"),
            Path("../backend/data/wakeword/sample_1763282265861.wav"),
        ]
        for pf in possible_files:
            if pf.exists():
                audio_file = pf
                break
    
    if audio_file is None or not audio_file.exists():
        print_error("No test audio file found")
        return False
    
    try:
        print_info(f"Testing wakeword detection with: {audio_file}")
        
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/detect_wakeword",
                files=files,
                timeout=10
            )
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Wakeword detection completed!")
            print(f"  Detected: {result.get('wakeword_detected')}")
            print(f"  Confidence: {result.get('confidence'):.2%}")
            print(f"  Processing Time: {result.get('processing_time_ms'):.2f}ms")
            return True
        else:
            print_error(f"Detection failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_command_detection(audio_file: Optional[Path] = None) -> bool:
    """Test the /detect_command endpoint"""
    print_header("TEST 4: Command Detection")
    
    # Try to find a test audio file
    if audio_file is None:
        possible_files = [
            Path("backend/data/wakeword/sample_1763282265861.wav"),
            Path("../backend/data/wakeword/sample_1763282265861.wav"),
        ]
        for pf in possible_files:
            if pf.exists():
                audio_file = pf
                break
    
    if audio_file is None or not audio_file.exists():
        print_error("No test audio file found")
        return False
    
    try:
        print_info(f"Testing command detection with: {audio_file}")
        
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/detect_command",
                files=files,
                timeout=10
            )
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Command detection completed!")
            print(f"  Intent: {result.get('intent')}")
            print(f"  Confidence: {result.get('confidence'):.2%}")
            print(f"  Processing Time: {result.get('processing_time_ms'):.2f}ms")
            return True
        else:
            print_error(f"Detection failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_voice_verification(audio_file: Optional[Path] = None) -> bool:
    """Test the /verify_voice endpoint"""
    print_header("TEST 5: Voice Verification")
    
    # Try to find a test audio file
    if audio_file is None:
        possible_files = [
            Path("backend/data/wakeword/sample_1763282265861.wav"),
            Path("../backend/data/wakeword/sample_1763282265861.wav"),
        ]
        for pf in possible_files:
            if pf.exists():
                audio_file = pf
                break
    
    if audio_file is None or not audio_file.exists():
        print_error("No test audio file found")
        return False
    
    try:
        print_info(f"Testing voice verification with: {audio_file}")
        
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {'user_id': 'TestUser'}
            response = requests.post(
                f"{BASE_URL}/verify_voice",
                files=files,
                data=data,
                timeout=10
            )
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Voice verification completed!")
            print(f"  Verified: {result.get('verified')}")
            print(f"  Confidence: {result.get('confidence'):.2%}")
            print(f"  Threshold: {result.get('threshold'):.1%}")
            print(f"  Processing Time: {result.get('processing_time_ms'):.2f}ms")
            return True
        else:
            print_error(f"Verification failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def main():
    """Run all tests"""
    print_header("BACKEND TEST SUITE")
    print_info(f"Backend URL: {BASE_URL}")
    print_info(f"Starting tests in 2 seconds...")
    time.sleep(2)
    
    results = {
        "Health Check": test_health_check(),
        "Sample Upload": test_sample_upload(),
        "Wakeword Detection": test_wakeword_detection(),
        "Command Detection": test_command_detection(),
        "Voice Verification": test_voice_verification(),
    }
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.END}" if result else f"{Colors.RED}FAILED{Colors.END}"
        print(f"  {test_name}: {status}")
    
    print(f"\n  Total: {passed}/{total} passed")
    
    if passed == total:
        print_success("All tests passed! Backend is working correctly.")
        return 0
    else:
        print_error(f"{total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
