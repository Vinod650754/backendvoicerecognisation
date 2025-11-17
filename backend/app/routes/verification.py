"""
Command detection and voice verification endpoints
"""
import time
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.utils import validate_audio_file
from app.services import command_detector, voice_verifier
from app.models.schemas import CommandDetectionResponse, VoiceVerificationResponse, ErrorResponse
from app.config.logger import logger

router = APIRouter(tags=["Detection"])


@router.post("/detect_command", response_model=CommandDetectionResponse)
async def detect_command(file: UploadFile = File(...)):
    """
    Detect command intent from audio
    
    Args:
        file: Audio file (WAV format)
        
    Returns:
        CommandDetectionResponse with detected command and confidence
    """
    try:
        audio_bytes = await file.read()
        
        # Validate audio
        is_valid, error_msg = validate_audio_file(audio_bytes)
        if not is_valid:
            logger.warning(f"Invalid audio for command detection: {error_msg}")
            return JSONResponse(
                {
                    "error": error_msg,
                    "details": None,
                    "status_code": 400
                },
                status_code=400
            )
        
        # Detect command
        intent, intent_idx, confidence, elapsed_ms = command_detector.detect(audio_bytes)
        
        return CommandDetectionResponse(
            intent=intent,
            intent_idx=intent_idx,
            confidence=confidence,
            processing_time_ms=elapsed_ms
        )
    
    except Exception as e:
        logger.error(f"Error in detect_command: {e}")
        return JSONResponse(
            {
                "error": "Command detection failed",
                "details": str(e),
                "status_code": 500
            },
            status_code=500
        )


@router.post("/verify_voice", response_model=VoiceVerificationResponse)
async def verify_voice(file: UploadFile = File(...), user_id: str = "HomeOwner"):
    """
    Verify if voice matches owner (biometric verification)
    
    Args:
        file: Audio file (WAV format)
        user_id: User identifier (default: HomeOwner)
        
    Returns:
        VoiceVerificationResponse with verification result
    """
    try:
        audio_bytes = await file.read()
        
        # Validate audio
        is_valid, error_msg = validate_audio_file(audio_bytes)
        if not is_valid:
            logger.warning(f"Invalid audio for voice verification: {error_msg}")
            return JSONResponse(
                {
                    "error": error_msg,
                    "details": None,
                    "status_code": 400
                },
                status_code=400
            )
        
        # Verify voice
        verified, confidence, elapsed_ms = voice_verifier.verify(user_id, audio_bytes)
        
        return VoiceVerificationResponse(
            verified=verified,
            confidence=confidence,
            threshold=0.7,
            processing_time_ms=elapsed_ms
        )
    
    except Exception as e:
        logger.error(f"Error in verify_voice: {e}")
        return JSONResponse(
            {
                "error": "Voice verification failed",
                "details": str(e),
                "status_code": 500
            },
            status_code=500
        )
