"""Lightweight bookkeeping for resume points and stage completion.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class StageProgress:
    """Record of completion for a named pipeline stage.

    Attributes:
        stage: Unique stage name.
        completed: Whether the stage successfully completed.
        message: Optional human-readable note.
    """

    stage: str
    completed: bool
    message: str = ""


def update_progress(progress: Dict[str, StageProgress], stage: str, completed: bool, message: str = "") -> None:
    """Update or add progress for a stage in-place.

    Args:
        progress: Mapping of stage name to StageProgress.
        stage: Stage name.
        completed: Completion flag.
        message: Optional note.
    """
    progress[stage] = StageProgress(stage=stage, completed=completed, message=message)
