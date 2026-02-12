"""Base data structures for model specification.

Defines simple dataclasses for model pieces (placeholders for now).

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelSpec:
    """Specification for the shot-level sky model (placeholder).

    Attributes:
        components: List of component names (e.g., ["mult", "additive"]).
        metadata: Arbitrary metadata mapping.
    """

    components: List[str]
    metadata: Dict[str, Any]


@dataclass
class Priors:
    """Container for model priors (placeholder)."""

    params: Dict[str, Any]


@dataclass
class Constraints:
    """Container for model parameter constraints (placeholder)."""

    bounds: Dict[str, Any]
