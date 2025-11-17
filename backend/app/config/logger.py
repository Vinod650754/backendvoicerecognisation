"""
Logging configuration
"""
import logging
import sys
from app.config.settings import LOG_LEVEL, LOG_FORMAT


def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("backend.log"),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()
