"""Logging configuration for the preprocessing pipeline.

Provides consistent logging setup across all pipeline modules.
Supports both console and optional file output.

Usage:
    from logging_config import setup_logging

    # Basic setup (console only)
    setup_logging()

    # With file output
    setup_logging(log_file="pipeline.log")

    # Debug level
    setup_logging(level=logging.DEBUG)
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """Configure logging for the preprocessing pipeline.

    Sets up the root logger with consistent formatting for all modules.
    Call this once at application startup (e.g., in GUI or main script).

    Args:
        level: Logging level (default: INFO).
        log_file: Optional path to log file. If provided, logs go to both
                 console and file.
        log_format: Custom log format string. If None, uses default.
        date_format: Custom date format string. If None, uses default.

    Returns:
        The configured root logger.

    Example:
        >>> setup_logging(level=logging.DEBUG, log_file="output/pipeline.log")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Pipeline started")
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {log_path}")

    return root_logger


def get_module_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Convenience function for getting a logger with the module name.
    The logger inherits settings from the root logger configured by
    setup_logging().

    Args:
        name: Module name (typically __name__).

    Returns:
        Logger instance for the module.

    Example:
        >>> logger = get_module_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


def set_level(level: int) -> None:
    """Change the logging level at runtime.

    Useful for enabling debug output temporarily.

    Args:
        level: New logging level (e.g., logging.DEBUG).

    Example:
        >>> set_level(logging.DEBUG)
        >>> # ... do something with verbose output
        >>> set_level(logging.INFO)  # back to normal
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers:
        handler.setLevel(level)
