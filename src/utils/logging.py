"""
Logging configuration for TPE system.

Provides readable console output and detailed JSON file logging.
"""

import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger

from config.settings import get_settings


class JSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for file logging."""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)

        # Add timestamp
        if not log_record.get("timestamp"):
            log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add log level
        if log_record.get("level"):
            log_record["level"] = log_record["level"].upper()
        else:
            log_record["level"] = record.levelname

        # Add correlation ID if available
        if hasattr(record, "correlation_id"):
            log_record["correlation_id"] = record.correlation_id

        # Add request ID if available
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id

        # Add agent type if available
        if hasattr(record, "agent_type"):
            log_record["agent_type"] = record.agent_type

        # Add session ID if available
        if hasattr(record, "session_id"):
            log_record["session_id"] = record.session_id


class ReadableFormatter(logging.Formatter):
    """Readable formatter for console output."""

    # Colors for different log levels
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )

    def format(self, record):
        # Format the base message
        formatted = super().format(record)

        # Add colors for console output
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Check if we should apply colors
        should_color = (
            # Check if this is a StreamHandler (console output)
            hasattr(self, 'stream') or
            # Check if stdout is a TTY (interactive terminal)
            hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() or
            # Environment variable override
            os.environ.get('FORCE_COLOR', '').lower() in ('1', 'true', 'yes')
        )

        if should_color:
            # Color the level name
            colored_level = f"{color}{record.levelname}{reset}"
            formatted = formatted.replace(record.levelname, colored_level, 1)

        # Simplify module names for readability
        formatted = formatted.replace("src.", "").replace("data_generation.", "dgen.")
        formatted = formatted.replace("evaluation.", "eval.").replace("agents.", "agent.")

        # Add context info if available
        context_parts = []
        if hasattr(record, 'agent_type') and record.agent_type:
            context_parts.append(f"agent={record.agent_type}")
        if hasattr(record, 'episode_id') and record.episode_id:
            context_parts.append(f"ep={record.episode_id[:8]}...")  # Shorten episode ID
        if hasattr(record, 'request_id') and record.request_id:
            context_parts.append(f"req={record.request_id[:8]}...")  # Shorten request ID

        if context_parts:
            formatted = f"{formatted} [{', '.join(context_parts)}]"

        return formatted


def setup_logging() -> None:
    """Configure structured logging for the application."""

    settings = get_settings()

    # Create formatters
    readable_formatter = ReadableFormatter()  # For console
    json_formatter = JSONFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )  # For files

    # Console handler with readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(readable_formatter)
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))

    # Setup handlers list
    handlers = [console_handler]

    # File handler with JSON format (if configured)
    if settings.log_file:
        log_file_path = Path(settings.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all logs, let handlers filter
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Removed structlog configuration to avoid duplication
    # Using standard Python logging with custom formatters


class LoggerMixin:
    """Mixin class to add logging capabilities with context."""

    @property
    def logger(self) -> logging.Logger:
        """Get a logger with class context."""
        return logging.getLogger(self.__class__.__name__)

    def log_with_context(
        self,
        message: str,
        level: str = "info",
        **context: Any,
    ) -> None:
        """Log a message with additional context."""
        # Add context as extra fields
        logger = self.logger
        logger_method = getattr(logger, level.lower())

        # Create a LogRecord with custom attributes for the formatter
        # This allows our formatter to access the context
        for key, value in context.items():
            setattr(logger_method.__self__, key, value)

        logger_method(message)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a standard logger instance with our custom formatting."""
    return logging.getLogger(name)


class CorrelationFilter(logging.Filter):
    """Filter to add correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to record if not present."""
        if not hasattr(record, "correlation_id"):
            record.correlation_id = str(uuid.uuid4())
        return True


# Initialize logging on import
setup_logging()


# Export main logger
logger = get_logger("tpe")
