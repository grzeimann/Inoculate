"""Standardized product containers for shot-level outputs.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ShotProducts:
    """Container for high-level shot products.

    Attributes:
        metadata: Arbitrary metadata about the shot processing.
        diagnostics: Mapping of diagnostic metrics computed during processing.
        artifacts_dir: Optional path to directory with large artifacts.
    """

    metadata: Dict[str, Any]
    diagnostics: Dict[str, Any]
    artifacts_dir: Optional[str] = None
