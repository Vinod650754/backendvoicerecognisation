import os
import math
import wave
import struct
import random

# Generates short WAV files (mono, 16kHz) containing simple sine tones and noise

def write_wav(path, samples, sr=16000):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = b''.join(struct.pack('<h', int(max(-32767, min(32767, s*32767)))) for s in samples)
        wf.writeframes(frames)


def sine(freq, length_sec, sr=16000, amp=0.5):
    return [amp * math.sin(2*math.pi*freq*(i/sr)) for i in range(int(sr*length_sec))]


def noise(length_sec, sr=16000, amp=0.2):
    return [amp * (random.random()*2 - 1) for _ in range(int(sr*length_sec))]


def overlay(a, b):
    L = max(len(a), len(b))
    out = []
    for i in range(L):
        va = a[i] if i < len(a) else 0
        vb = b[i] if i < len(b) else 0
        out.append(va + vb)
    return out


def make_dirs():
    os.makedirs('data/wakewords/train/wakeword', exist_ok=True)
    os.makedirs('data/wakewords/train/noise', exist_ok=True)
    os.makedirs('data/wakewords/test/wakeword', exist_ok=True)
    os.makedirs('data/wakewords/test/noise', exist_ok=True)
    os.makedirs('data/commands/train/turn_on', exist_ok=True)
    os.makedirs('data/commands/train/turn_off', exist_ok=True)
    os.makedirs('data/commands/test/turn_on', exist_ok=True)
    os.makedirs('data/commands/test/turn_off', exist_ok=True)


def generate():
    sr = 16000
    make_dirs()
    # Wakeword positives: a 700 Hz tone + low noise
    for i in range(6):
        s = sine(700 + i*5, 1.0, sr, amp=0.6)
        s = overlay(s, noise(1.0, sr, amp=0.1))
        write_wav(f'data/wakewords/train/wakeword/wake_{i}.wav', s, sr)
    # Wakeword negatives: noise or other tones
    for i in range(6):
        s = overlay(sine(300 + i*10, 1.0, sr, amp=0.4), noise(1.0, sr, amp=0.2))
        write_wav(f'data/wakewords/train/noise/noise_{i}.wav', s, sr)

    # small test set
    write_wav('data/wakewords/test/wake_test.wav', overlay(sine(700,1.0,sr,0.6), noise(1.0,sr,0.1)), sr)
    write_wav('data/wakewords/test/noise_test.wav', overlay(sine(350,1.0,sr,0.4), noise(1.0,sr,0.2)), sr)

    # Commands: turn_on and turn_off – different tone combos
    for i in range(6):
        s = overlay(sine(800 + i*10, 1.0, sr, amp=0.6), noise(1.0, sr, amp=0.05))
        write_wav(f'data/commands/train/turn_on/on_{i}.wav', s, sr)
    for i in range(6):
        s = overlay(sine(500 + i*10, 1.0, sr, amp=0.6), noise(1.0, sr, amp=0.05))
        write_wav(f'data/commands/train/turn_off/off_{i}.wav', s, sr)

    write_wav('data/commands/test/turn_on/on_test.wav', overlay(sine(800,1.0,sr,0.6), noise(1.0,sr,0.05)), sr)
    write_wav('data/commands/test/turn_off/off_test.wav', overlay(sine(500,1.0,sr,0.6), noise(1.0,sr,0.05)), sr)

if __name__ == '__main__':
    generate()
    print('Synthetic data generated under data/')
