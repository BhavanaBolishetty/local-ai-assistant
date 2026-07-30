"""Centralized logging configuration.

Every module gets its logger via `logging.getLogger(__name__)` as usual;
this module just configures the root handler/format once at startup so
log output is consistent across the API, services, and AI client layers.
"""

import logging
import sys

from src.core.config import get_settings


def configure_logging() -> None:
    """Configure the root logger. Call this once, at process startup."""
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
