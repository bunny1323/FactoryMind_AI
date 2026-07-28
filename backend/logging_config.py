"""Structured logging configuration for FactoryMind AI."""
from __future__ import annotations

import logging
import sys
from typing import Any
from backend.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    console_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    loggers = [
        "factorymind",
        "uvicorn",
        "uvicorn.access",
        "qdrant_client",
        "fastembed",
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
        logger.addHandler(console_handler)


class StructuredLogger:
    """Structured logger with context support."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context: dict[str, Any] = {}
    
    def with_context(self, **kwargs: Any) -> "StructuredLogger":
        """Add context to logger."""
        new_logger = StructuredLogger(self.logger.name)
        new_logger._context = {**self._context, **kwargs}
        return new_logger
    
    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self._context:
            context_str = " | ".join(f"{k}={v}" for k, v in self._context.items())
            return f"{message} | {context_str}"
        return message
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message), **kwargs)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message), **kwargs)
    
    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self.logger.exception(self._format_message(message), **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)
