"""Routes package"""
from .health import router as health_router
from .samples import router as samples_router
from .detection import router as detection_router
from .verification import router as verification_router

__all__ = ["health_router", "samples_router", "detection_router", "verification_router"]
