"""
Utility functions for audio preprocessing
"""
import io
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
from typing import Tuple, Optional

from app.config.settings import (
    TARGET_SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH,
    MIN_AUDIO_FILE_SIZE, MIN_AUDIO_DURATION_SECONDS, MAX_AUDIO_DURATION_SECONDS
)
from app.config.logger import logger


def preprocess_audio_bytes(
    audio_bytes: bytes,
    target_sr: int = TARGET_SAMPLE_RATE
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Convert audio bytes to mel spectrogram tensor
    
    Args:
        audio_bytes: Raw audio bytes (WAV format)
        target_sr: Target sample rate for resampling
        
    Returns:
        Tuple of (mel_spectrogram_tensor, sample_rate) or (None, None) on error
    """
    try:
        # Validate file size
        if len(audio_bytes) < MIN_AUDIO_FILE_SIZE:
            logger.warning(f"Audio file too small: {len(audio_bytes)} bytes")
            return None, None
        
        # Read audio bytes
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        
        # Convert to mono if needed
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        
        # Check duration
        duration = len(data) / sr
        if duration < MIN_AUDIO_DURATION_SECONDS or duration > MAX_AUDIO_DURATION_SECONDS:
            logger.warning(f"Audio duration {duration:.2f}s out of range [{MIN_AUDIO_DURATION_SECONDS}, {MAX_AUDIO_DURATION_SECONDS}]")
            return None, None
        
        # Resample if needed
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        
        # Create mel spectrogram
        waveform = torch.from_numpy(data).unsqueeze(0)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )(waveform)
        
        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        # Add batch and channel dims: [1, 1, freq, time]
        mel = mel.unsqueeze(0)
        
        logger.debug(f"Audio preprocessed: {mel.shape}, duration={duration:.2f}s")
        return mel, target_sr
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        return None, None


def validate_audio_file(audio_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate if audio file is valid
    
    Args:
        audio_bytes: Raw audio bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if len(audio_bytes) < MIN_AUDIO_FILE_SIZE:
            return False, f"Audio file too small: {len(audio_bytes)} bytes"
        
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        
        duration = len(data) / sr
        
        if duration < MIN_AUDIO_DURATION_SECONDS:
            return False, f"Audio too short: {duration:.2f}s"
        
        if duration > MAX_AUDIO_DURATION_SECONDS:
            return False, f"Audio too long: {duration:.2f}s"
        
        if len(data) == 0:
            return False, "Audio file is empty"
        
        return True, ""
        
    except Exception as e:
        return False, f"Invalid audio format: {str(e)}"
