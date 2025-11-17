"""
Health check and utility endpoints
"""
from fastapi import APIRouter
from app.config.settings import APP_NAME, APP_VERSION, APP_DESCRIPTION
from app.models.schemas import HealthResponse
from app.config.logger import logger

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if backend is running and healthy
    
    Returns:
        HealthResponse with status, message, and version
    """
    logger.info("Health check endpoint called")
    return HealthResponse(
        status="ok",
        message="Backend is running and healthy",
        version=APP_VERSION
    )


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - redirects to API info"""
    return HealthResponse(
        status="ok",
        message=f"{APP_NAME} - {APP_DESCRIPTION}",
        version=APP_VERSION
    )
