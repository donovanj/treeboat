"""Centralized logging configuration for the financial prediction system."""

import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler
import traceback

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Configure the root logger
log_file_path = os.path.join(logs_dir, "financial_prediction_system.log")

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create file handler with daily rotation
file_handler = TimedRotatingFileHandler(
    log_file_path, when="midnight", backupCount=7
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Create a logger specific to the financial prediction system
# This prevents double logging by disabling propagation to the root logger
logger = logging.getLogger("financial_prediction_system")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Don't propagate to the root logger

# Add handlers directly to our logger instead of the root logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Add exception handler
def log_exception(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )

# Set up exception handler
sys.excepthook = log_exception

# Export the logger for use in other modules
__all__ = ["logger"]