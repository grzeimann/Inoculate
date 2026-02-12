"""Resume helpers that integrate with the checkpointing system.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..state import load_checkpoint
from ..utils.exceptions import CheckpointError


def maybe_resume(checkpoint_path: str | Path) -> Optional[Dict[str, Any]]:
    """Load a checkpoint if it exists; return None if not present.

    Args:
        checkpoint_path: Path to a checkpoint JSON file.

    Returns:
        The loaded state mapping if the file exists, otherwise None.

    Raises:
        CheckpointError: If the file exists but fails to load.
    """
    p = Path(checkpoint_path)
    if not p.exists():
        return None
    return load_checkpoint(p)
