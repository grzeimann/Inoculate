"""Survey-informed priors for multiplicative shapes and amplifier behavior.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from typing import Any, Dict


def amplifier_health_prior() -> Dict[str, Any]:
    """Return a placeholder prior on amplifier health states.

    Returns:
        Mapping of default prior probabilities for amp labels.
    """
    return {
        "zeros": 0.01,
        "defective": 0.02,
        "blank": 0.6,
        "filled": 0.35,
        "outlier": 0.02,
    }
