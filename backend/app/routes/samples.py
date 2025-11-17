"""
Audio sample collection endpoints
"""
import os
import time
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

from app.config.settings import DATA_DIR
from app.utils import validate_audio_file
from app.models.schemas import SampleUploadResponse, ErrorResponse
from app.config.logger import logger

router = APIRouter(tags=["Samples"])


@router.post("/collect_sample", response_model=SampleUploadResponse)
async def collect_sample(file: UploadFile = File(...), label: str = Form(...)):
    """
    Save an uploaded audio sample to backend/data/<label>/ folder
    
    Args:
        file: Audio file (WAV format)
        label: Label for the sample (wakeword, command, open_door, close_door)
        
    Returns:
        SampleUploadResponse with save location and metadata
    """
    try:
        # Validate label
        valid_labels = ['wakeword', 'command', 'open_door', 'close_door']
        if label not in valid_labels:
            logger.warning(f"Invalid label: {label}")
            return JSONResponse(
                {
                    "error": f"label must be one of: {', '.join(valid_labels)}",
                    "details": None,
                    "status_code": 400
                },
                status_code=400
            )
        
        # Read audio bytes
        audio_bytes = await file.read()
        
        # Validate audio
        is_valid, error_msg = validate_audio_file(audio_bytes)
        if not is_valid:
            logger.warning(f"Invalid audio: {error_msg}")
            return JSONResponse(
                {
                    "error": error_msg,
                    "details": None,
                    "status_code": 400
                },
                status_code=400
            )
        
        # Create directory
        target_dir = DATA_DIR / label
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        timestamp = int(time.time() * 1000)
        filename = f"sample_{timestamp}.wav"
        dest_path = target_dir / filename
        
        with open(dest_path, 'wb') as f:
            f.write(audio_bytes)
        
        logger.info(f"Saved sample to: {dest_path}")
        
        return SampleUploadResponse(
            success=True,
            message=f"Sample saved as {filename}",
            path=str(dest_path),
            label=label,
            timestamp=timestamp
        )
    
    except Exception as e:
        logger.error(f"Error in collect_sample: {e}")
        return JSONResponse(
            {
                "error": "Failed to save sample",
                "details": str(e),
                "status_code": 500
            },
            status_code=500
        )
