"""Lightweight timing/context utilities with structured logging.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def time_block(name: str) -> Iterator[None]:
    """Context manager to time a code block and log the duration.

    Args:
        name: Descriptive name for the timed block.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info("[timing] %s: %.3fs", name, dt)
