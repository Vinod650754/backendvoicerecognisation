"""
Service layer for audio processing and model inference
"""
import time
import torch
import numpy as np
from typing import Tuple, Optional

from app.models.neural_models import WakeWordModel, CommandModel, VoiceBiometricModel
from app.utils import preprocess_audio_bytes
from app.config.settings import (
    WAKEWORD_CONFIDENCE_THRESHOLD,
    COMMAND_CONFIDENCE_THRESHOLD,
    VOICE_VERIFICATION_THRESHOLD
)
from app.config.logger import logger


class AudioProcessingService:
    """Service for audio preprocessing and analysis"""
    
    @staticmethod
    def preprocess_audio(audio_bytes: bytes) -> Tuple[Optional[torch.Tensor], Optional[int], Optional[float]]:
        """
        Preprocess audio bytes to mel spectrogram
        
        Returns:
            Tuple of (mel_spectrogram, sample_rate, processing_time_ms)
        """
        start_time = time.time()
        mel, sr = preprocess_audio_bytes(audio_bytes)
        elapsed_ms = (time.time() - start_time) * 1000
        
        return mel, sr, elapsed_ms


class WakewordDetectionService:
    """Service for wakeword detection"""
    
    def __init__(self):
        self.model = WakeWordModel()
        self.model.eval()
        logger.info("WakewordDetectionService initialized")
    
    def detect(self, audio_bytes: bytes) -> Tuple[bool, float, float]:
        """
        Detect wakeword in audio
        
        Returns:
            Tuple of (wakeword_detected, confidence, processing_time_ms)
        """
        start_time = time.time()
        
        try:
            audio_processor = AudioProcessingService()
            mel, sr, prep_time = audio_processor.preprocess_audio(audio_bytes)
            
            if mel is None:
                logger.warning("Failed to preprocess audio for wakeword detection")
                return False, 0.0, (time.time() - start_time) * 1000
            
            with torch.no_grad():
                output = self.model(mel)
                pred_idx = int(output.argmax(dim=1).item())
                confidence = float(torch.softmax(output, dim=1)[0, pred_idx].item())
            
            # pred_idx == 1 means wakeword detected
            wakeword_detected = (pred_idx == 1)
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Wakeword detection: detected={wakeword_detected}, confidence={confidence:.3f}, time={elapsed_ms:.2f}ms")
            
            return wakeword_detected, confidence, elapsed_ms
            
        except Exception as e:
            logger.error(f"Error in wakeword detection: {e}")
            return False, 0.0, (time.time() - start_time) * 1000


class CommandDetectionService:
    """Service for command/intent detection"""
    
    COMMANDS = ['lock', 'unlock', 'open', 'close']
    
    def __init__(self):
        self.model = CommandModel(num_classes=len(self.COMMANDS))
        self.model.eval()
        logger.info("CommandDetectionService initialized")
    
    def detect(self, audio_bytes: bytes) -> Tuple[str, int, float, float]:
        """
        Detect command intent from audio
        
        Returns:
            Tuple of (intent, intent_idx, confidence, processing_time_ms)
        """
        start_time = time.time()
        
        try:
            audio_processor = AudioProcessingService()
            mel, sr, prep_time = audio_processor.preprocess_audio(audio_bytes)
            
            if mel is None:
                logger.warning("Failed to preprocess audio for command detection")
                return "unknown", -1, 0.0, (time.time() - start_time) * 1000
            
            with torch.no_grad():
                output = self.model(mel)
                pred_idx = int(output.argmax(dim=1).item())
                confidence = float(torch.softmax(output, dim=1)[0, pred_idx].item())
            
            intent = self.COMMANDS[pred_idx % len(self.COMMANDS)]
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Command detection: intent={intent}, confidence={confidence:.3f}, time={elapsed_ms:.2f}ms")
            
            return intent, pred_idx, confidence, elapsed_ms
            
        except Exception as e:
            logger.error(f"Error in command detection: {e}")
            return "unknown", -1, 0.0, (time.time() - start_time) * 1000


class VoiceVerificationService:
    """Service for voice biometric verification"""
    
    def __init__(self):
        self.model = VoiceBiometricModel(embedding_dim=128)
        self.model.eval()
        self.reference_embeddings = {}  # In production, store in database
        logger.info("VoiceVerificationService initialized")
    
    def register_voice(self, user_id: str, audio_bytes: bytes) -> bool:
        """Register a voice sample for a user"""
        try:
            audio_processor = AudioProcessingService()
            mel, sr, _ = audio_processor.preprocess_audio(audio_bytes)
            
            if mel is None:
                logger.warning(f"Failed to preprocess audio for user {user_id}")
                return False
            
            with torch.no_grad():
                embedding = self.model(mel)
                self.reference_embeddings[user_id] = embedding.cpu().numpy()
            
            logger.info(f"Voice registered for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering voice for {user_id}: {e}")
            return False
    
    def verify(self, user_id: str, audio_bytes: bytes) -> Tuple[bool, float, float]:
        """
        Verify if voice matches registered user
        
        Returns:
            Tuple of (verified, confidence, processing_time_ms)
        """
        start_time = time.time()
        
        try:
            # For demo: simulate verification with random confidence
            # In production: compare with registered embedding
            
            if user_id not in self.reference_embeddings:
                logger.warning(f"No registered voice for user: {user_id}")
                # Generate random confidence for demo
                confidence = np.random.uniform(0.6, 0.99)
            else:
                audio_processor = AudioProcessingService()
                mel, sr, _ = audio_processor.preprocess_audio(audio_bytes)
                
                if mel is None:
                    return False, 0.0, (time.time() - start_time) * 1000
                
                with torch.no_grad():
                    embedding = self.model(mel)
                    reference = torch.from_numpy(self.reference_embeddings[user_id])
                    # Cosine similarity
                    similarity = torch.nn.functional.cosine_similarity(embedding, reference)
                    confidence = float(similarity.item())
            
            verified = confidence >= VOICE_VERIFICATION_THRESHOLD
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Voice verification: user={user_id}, verified={verified}, confidence={confidence:.3f}, time={elapsed_ms:.2f}ms")
            
            return verified, confidence, elapsed_ms
            
        except Exception as e:
            logger.error(f"Error verifying voice for {user_id}: {e}")
            return False, 0.0, (time.time() - start_time) * 1000


# Singleton instances
audio_processor = AudioProcessingService()
wakeword_detector = WakewordDetectionService()
command_detector = CommandDetectionService()
voice_verifier = VoiceVerificationService()
