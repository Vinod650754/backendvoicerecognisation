#!/usr/bin/env python3
"""
Generate synthetic owner voice samples for biometric training.
This creates demo audio files with distinct characteristics for training.
"""

import os
import numpy as np
import soundfile as sf
from scipy import signal

def generate_voice_sample(sample_rate=16000, duration=2.0, voice_id=0, label_type='wakeword'):
    """
    Generate a synthetic voice sample with characteristics unique to the owner.
    voice_id: unique identifier (0 = owner, others = non-owner for testing)
    label_type: 'wakeword' or 'command'
    """
    t = np.arange(0, duration, 1/sample_rate)
    
    # Create a complex signal with multiple frequency components (simulating speech)
    # Owner voice has unique frequency fingerprint
    if voice_id == 0:  # Owner
        if label_type == 'wakeword':
            # Wakeword: 'hey jarvis' - unique freq pattern
            f1, f2, f3 = 200 + np.random.randint(-20, 20), 800 + np.random.randint(-50, 50), 2500 + np.random.randint(-100, 100)
        else:  # command
            # Command: 'open/close' - different pattern
            f1, f2, f3 = 150 + np.random.randint(-20, 20), 900 + np.random.randint(-50, 50), 2200 + np.random.randint(-100, 100)
    else:  # Non-owner (for testing)
        f1, f2, f3 = 300, 1200, 3000
    
    # Generate multi-frequency signal
    signal_part = (0.3 * np.sin(2 * np.pi * f1 * t) +
                   0.25 * np.sin(2 * np.pi * f2 * t) +
                   0.15 * np.sin(2 * np.pi * f3 * t))
    
    # Add envelope (fade in/out)
    envelope = np.ones_like(t)
    fade_samples = int(0.1 * sample_rate)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    # Apply envelope
    signal_part = signal_part * envelope
    
    # Add small random noise
    noise = 0.02 * np.random.randn(len(t))
    final_signal = signal_part + noise
    
    # Normalize
    final_signal = final_signal / (np.max(np.abs(final_signal)) + 1e-6)
    
    return final_signal.astype(np.float32)


def create_dataset(num_wakeword_samples=15, num_command_samples=15):
    """Create synthetic owner voice dataset."""
    sr = 16000
    
    # Create directory structure
    os.makedirs('data/wakeword/owner', exist_ok=True)
    os.makedirs('data/command/owner', exist_ok=True)
    
    print("Generating owner voice samples for training...")
    
    # Generate wakeword samples
    print(f"Generating {num_wakeword_samples} wakeword samples...")
    for i in range(num_wakeword_samples):
        audio = generate_voice_sample(sample_rate=sr, duration=2.0, voice_id=0, label_type='wakeword')
        path = f'data/wakeword/owner/sample_{i:02d}.wav'
        sf.write(path, audio, sr)
        print(f"  Created {path}")
    
    # Generate command samples
    print(f"Generating {num_command_samples} command samples...")
    for i in range(num_command_samples):
        audio = generate_voice_sample(sample_rate=sr, duration=1.5, voice_id=0, label_type='command')
        path = f'data/command/owner/sample_{i:02d}.wav'
        sf.write(path, audio, sr)
        print(f"  Created {path}")
    
    print(f"\nDataset created successfully!")
    print(f"Total samples: {num_wakeword_samples + num_command_samples}")
    print(f"Location: ./data/wakeword/owner/ and ./data/command/owner/")


if __name__ == '__main__':
    create_dataset(num_wakeword_samples=15, num_command_samples=15)
