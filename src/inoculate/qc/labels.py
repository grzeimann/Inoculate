"""Rule-based amplifier labeling for SingleShot workflow.

Provides a minimal set of rules to derive a good/bad mask for PCA/mult stages.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


class Label:
    """Simple container for a QC label.

    Attributes:
        name: Label name (e.g., "good", "zeros", "defective").
        confidence: Confidence score in [0, 1].
        reasons: Optional list of reason codes.
    """

    def __init__(self, name: str, confidence: float = 0.0, reasons: list[str] | None = None) -> None:
        self.name = name
        self.confidence = float(confidence)
        self.reasons = reasons or []


def label_amps(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Assign labels and a good-amp mask using simple threshold rules.

    Rules (placeholder, conservative):
      - zeros if zero_frac > 0.5
      - defective if mad_resid > 10 * median(mad_resid)
      - otherwise good

    Args:
        df_features: Feature table with at least [amp, exp, med_ratio, mad_resid, zero_frac].

    Returns:
        Tuple of (df_labels, good_mask) where df_labels has columns [amp, label]
        aggregated across exposures by worst-case, and good_mask is (n_amp,) bool.
    """
    if df_features.empty:
        return pd.DataFrame(columns=["amp", "label"]).astype({"amp": int, "label": str}), np.array([], dtype=bool)

    grouped = df_features.groupby("amp")
    med_mad = df_features["mad_resid"].median() if np.isfinite(df_features["mad_resid"]).any() else 0.0
    labels = []
    good_mask = np.ones(grouped.ngroups, dtype=bool)

    for i, (amp, g) in enumerate(grouped):
        reasons: list[str] = []
        name = "good"
        if (g["zero_frac"] > 0.5).any():
            name = "zeros"
            reasons.append("zero_frac>0.5")
        elif (g["mad_resid"] > max(1e-6, 10.0 * med_mad)).any():
            name = "defective"
            reasons.append("mad_resid>10xmedian")
        labels.append({"amp": int(amp), "label": name, "reasons": ",".join(reasons)})
        if name != "good":
            good_mask[int(amp)] = False

    df_labels = pd.DataFrame(labels).sort_values("amp").reset_index(drop=True)
    return df_labels[["amp", "label"]], good_mask