"""Registry of known-bad amplifiers and IFUs.

Placeholder for a persistent store of components with historical health issues.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from typing import Dict, Set


class HealthRegistry:
    """In-memory registry of known-bad components (placeholder).

    Attributes:
        bad_amps: Set of amplifier identifiers.
        bad_ifus: Set of IFU identifiers.
    """

    def __init__(self) -> None:
        self.bad_amps: Set[str] = set()
        self.bad_ifus: Set[str] = set()

    def mark_bad_amp(self, amp_id: str) -> None:
        """Mark an amplifier as bad.

        Args:
            amp_id: Amplifier identifier string.
        """
        self.bad_amps.add(amp_id)

    def mark_bad_ifu(self, ifu_id: str) -> None:
        """Mark an IFU as bad.

        Args:
            ifu_id: IFU identifier string.
        """
        self.bad_ifus.add(ifu_id)
