"""Centralized logging configuration.

Every module gets its logger via `logging.getLogger(__name__)` as usual;
this module just configures the root handler/format once at startup so
log output is consistent across the API, services, and AI client layers.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.core.config import get_settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> None:
    """Configure the root logger. Call this once, at process startup."""
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level,
        format=_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
    )

    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        logging.getLogger().addHandler(file_handler)
