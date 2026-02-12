"""Lightweight logging configuration for CLI and library use.

Avoids configuring root logging in library import. CLI commands can call
`setup_logging()` to set a level and format; library code should use
`get_logger(__name__)`.
"""
from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a sensible default level."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Do not add handlers by default; leave to CLI or host application.
        logger.propagate = True
    return logger


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging for CLI execution with a simple format."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
