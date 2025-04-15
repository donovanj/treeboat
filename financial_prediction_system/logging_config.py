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

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Create a logger specific to the financial prediction system
logger = logging.getLogger("financial_prediction_system")

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