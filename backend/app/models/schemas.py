"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Status of the backend")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")


class SampleUploadResponse(BaseModel):
    """Response from sample upload endpoint"""
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Response message")
    path: str = Field(..., description="Path where sample was saved")
    label: str = Field(..., description="Label of the sample")
    timestamp: int = Field(..., description="Unix timestamp in milliseconds")


class WakewordDetectionResponse(BaseModel):
    """Response from wakeword detection"""
    wakeword_detected: bool = Field(..., description="Whether wakeword was detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class CommandDetectionResponse(BaseModel):
    """Response from command detection"""
    intent: str = Field(..., description="Detected command (lock, unlock, open, close)")
    intent_idx: int = Field(..., ge=0, description="Command index")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class VoiceVerificationResponse(BaseModel):
    """Response from voice verification"""
    verified: bool = Field(..., description="Whether voice was verified")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    threshold: float = Field(..., description="Verification threshold")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    status_code: int = Field(..., description="HTTP status code")
