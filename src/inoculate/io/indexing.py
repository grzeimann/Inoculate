"""Indexing helpers for VIRUS amplifier and exposure layout.

These functions avoid loading entire fiber arrays by returning indices or slices
for the requested amplifier/exposure blocks.

All functions use Google-style docstrings.
"""
from __future__ import annotations

from ..constants import EXPOSURES_PER_SHOT, NUM_FIBERS_PER_AMP


def amp_block_start(amp_index: int, *, fibers_per_amp: int = NUM_FIBERS_PER_AMP,
                    exposures_per_shot: int = EXPOSURES_PER_SHOT) -> int:
    """Compute the starting row index for an amplifier block.

    Args:
        amp_index: Zero-based amplifier index.
        fibers_per_amp: Number of fibers per amplifier block.
        exposures_per_shot: Number of exposures per shot.

    Returns:
        The starting (inclusive) row index of the amplifier block.

    Raises:
        ValueError: If ``amp_index`` is negative.
    """
    if amp_index < 0:
        raise ValueError("amp_index must be non-negative")
    return amp_index * (exposures_per_shot * fibers_per_amp)


def amp_exposure_range(amp_index: int, exposure_index: int, *,
                       fibers_per_amp: int = NUM_FIBERS_PER_AMP,
                       exposures_per_shot: int = EXPOSURES_PER_SHOT) -> tuple[int, int]:
    """Return the [start, stop) row range for an amp/exposure.

    Args:
        amp_index: Zero-based amplifier index.
        exposure_index: Zero-based exposure index within the shot.
        fibers_per_amp: Number of fibers per amplifier.
        exposures_per_shot: Number of exposures per shot.

    Returns:
        A tuple ``(start, stop)`` indices for slicing.

    Raises:
        ValueError: If ``exposure_index`` is out of range or ``amp_index`` is negative.
    """
    if not (0 <= exposure_index < exposures_per_shot):
        raise ValueError("exposure_index out of range")
    start = amp_block_start(amp_index, fibers_per_amp=fibers_per_amp,
                            exposures_per_shot=exposures_per_shot)
    start += exposure_index * fibers_per_amp
    return start, start + fibers_per_amp
