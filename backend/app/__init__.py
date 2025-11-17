"""
Main FastAPI application factory
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import (
    APP_NAME, APP_VERSION, APP_DESCRIPTION,
    CORS_ORIGINS, DEBUG
)
from app.config.logger import logger
from app.routes import (
    health_router,
    samples_router,
    detection_router,
    verification_router
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title=APP_NAME,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        debug=DEBUG
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health_router)
    app.include_router(samples_router)
    app.include_router(detection_router)
    app.include_router(verification_router)
    
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
        logger.info(f"CORS origins: {CORS_ORIGINS}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info(f"Shutting down {APP_NAME}")
    
    return app


# Create the application instance
app = create_app()
