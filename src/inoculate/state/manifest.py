"""Manifests describing inputs, outputs, and configuration snapshots.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class FileManifest:
    """Snapshot of files and configuration used for a run.

    Attributes:
        inputs: List of input file paths.
        outputs: List of produced file paths.
        config: Arbitrary configuration mapping captured at runtime.
        work_dir: Working directory path for the run.
    """

    inputs: List[Path]
    outputs: List[Path]
    config: Dict[str, Any]
    work_dir: Path
