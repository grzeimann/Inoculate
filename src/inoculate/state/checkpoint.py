"""Checkpoint save/load utilities for Inoculate runs.

These stubs define the API expected by the CLI and pipeline layers. Real
implementations will use structured manifests and robust atomic writes.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..utils.exceptions import CheckpointError


def save_checkpoint(path: str | Path, state: Dict[str, Any]) -> None:
    """Save a minimal JSON checkpoint to the given path.

    This placeholder writes a UTF-8 JSON file atomically via a temporary file
    and rename. In production, we will include content hashing and schema
    validation.

    Args:
        path: Filesystem path for the checkpoint file.
        state: Serializable mapping representing pipeline state.

    Raises:
        CheckpointError: If writing fails.
    """
    import json
    import os
    from tempfile import NamedTemporaryFile

    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
            json.dump(state, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, path)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        raise CheckpointError(f"Failed to save checkpoint to {path}: {exc}") from exc


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    """Load a minimal JSON checkpoint from the given path.

    Args:
        path: Filesystem path for the checkpoint file.

    Returns:
        Mapping containing the loaded pipeline state.

    Raises:
        CheckpointError: If reading or decoding fails, or file is missing.
    """
    import json

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise CheckpointError(f"Checkpoint not found: {path}") from exc
    except Exception as exc:  # pragma: no cover - filesystem dependent
        raise CheckpointError(f"Failed to load checkpoint from {path}: {exc}") from exc
