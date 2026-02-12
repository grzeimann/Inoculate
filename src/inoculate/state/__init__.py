"""State management and checkpointing utilities for Inoculate.

Modules here provide a thin abstraction for saving and resuming pipeline runs.
Google-style docstrings are used throughout.
"""
from __future__ import annotations

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
]

from .checkpoint import load_checkpoint, save_checkpoint  # re-export
