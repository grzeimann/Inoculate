"""Project-specific exception hierarchy for Inoculate.

All public modules should raise these exceptions instead of generic Exception
for predictable error handling by the CLI and pipeline layers.

Google-style docstrings are used throughout.
"""
from __future__ import annotations


class SchemaError(Exception):
    """Raised when inputs or internal tables violate expected schemas."""


class CheckpointError(Exception):
    """Raised when checkpoint save/load operations fail."""


class ModelSpecError(Exception):
    """Raised when a model specification is invalid or inconsistent."""


class FitFailureError(Exception):
    """Raised when a numerical fit fails to converge or is ill-posed."""
