"""Lightweight timing/context utilities.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def time_block(name: str) -> Iterator[None]:
    """Context manager to time a code block and print the duration.

    Args:
        name: Descriptive name for the timed block.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[timing] {name}: {dt:.3f}s")
