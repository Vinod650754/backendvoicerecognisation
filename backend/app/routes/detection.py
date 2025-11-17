"""
Wakeword detection endpoints
"""
import time
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.utils import validate_audio_file
from app.services import wakeword_detector
from app.models.schemas import WakewordDetectionResponse, ErrorResponse
from app.config.logger import logger

router = APIRouter(tags=["Detection"])


@router.post("/detect_wakeword", response_model=WakewordDetectionResponse)
async def detect_wakeword(file: UploadFile = File(...)):
    """
    Detect if wakeword is present in audio
    
    Args:
        file: Audio file (WAV format)
        
    Returns:
        WakewordDetectionResponse with detection result and confidence
    """
    try:
        audio_bytes = await file.read()
        
        # Validate audio
        is_valid, error_msg = validate_audio_file(audio_bytes)
        if not is_valid:
            logger.warning(f"Invalid audio for wakeword detection: {error_msg}")
            return JSONResponse(
                {
                    "error": error_msg,
                    "details": None,
                    "status_code": 400
                },
                status_code=400
            )
        
        # Detect wakeword
        detected, confidence, elapsed_ms = wakeword_detector.detect(audio_bytes)
        
        return WakewordDetectionResponse(
            wakeword_detected=detected,
            confidence=confidence,
            processing_time_ms=elapsed_ms
        )
    
    except Exception as e:
        logger.error(f"Error in detect_wakeword: {e}")
        return JSONResponse(
            {
                "error": "Wakeword detection failed",
                "details": str(e),
                "status_code": 500
            },
            status_code=500
        )
