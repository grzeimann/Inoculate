"""Iterative, stateful labeling for amplifier QC in SingleShot workflow.

Implements a LabelSet that evolves as the pipeline computes intermediate
products. Labeling is not a one-shot classifier; it updates per-amp/per-exp
scores and masks at multiple stages (features → mult → poly2d → ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
from pathlib import Path

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


@dataclass
class LabelSet:
    """Stateful labels that evolve across pipeline stages.

    Attributes:
        n_amp: Number of amplifiers.
        n_exp: Number of exposures.
        score: Per-amp/per-exp score in [0, 1], higher is better. Shape (n_amp, n_exp).
        mask: Per-amp/per-exp boolean mask (True=use). Shape (n_amp, n_exp).
        reasons: Optional per-amp/per-exp comma-separated reason strings. Shape (n_amp, n_exp) object array.
    """

    n_amp: int
    n_exp: int
    score: np.ndarray
    mask: np.ndarray
    reasons: np.ndarray

    @staticmethod
    def initialize(n_amp: int, n_exp: int) -> "LabelSet":
        score = np.ones((n_amp, n_exp), dtype=float)
        mask = np.ones((n_amp, n_exp), dtype=bool)
        reasons = np.empty((n_amp, n_exp), dtype=object)
        reasons[:] = ""
        return LabelSet(n_amp=n_amp, n_exp=n_exp, score=score, mask=mask, reasons=reasons)

    @staticmethod
    def from_features(df_features: pd.DataFrame) -> "LabelSet":
        if df_features.empty:
            return LabelSet.initialize(0, 0)
        n_amp = int(df_features["amp"].max()) + 1
        n_exp = int(df_features["exp"].max()) + 1
        ls = LabelSet.initialize(n_amp, n_exp)
        med_mad = df_features["mad_resid"].median() if np.isfinite(df_features["mad_resid"]).any() else 0.0
        for _, row in df_features.iterrows():
            a = int(row["amp"]) ; e = int(row["exp"]) ; zf = float(row.get("zero_frac", 0.0)) ; mad = float(row.get("mad_resid", np.nan))
            if zf > 0.5:
                ls.mask[a, e] = False
                ls.score[a, e] *= 0.0
                ls.reasons[a, e] = ",".join([r for r in [str(ls.reasons[a, e]) if ls.reasons[a, e] else "", "zero_frac>0.5"] if r])
            if np.isfinite(mad) and mad > max(1e-6, 10.0 * med_mad):
                ls.mask[a, e] = False
                ls.score[a, e] *= 0.25
                ls.reasons[a, e] = ",".join([r for r in [str(ls.reasons[a, e]) if ls.reasons[a, e] else "", "mad_resid>10xmedian"] if r])
        # ensure consistency
        ls._update_mask_from_score()
        return ls

    def update_with_mult(self, mult: np.ndarray, bounds: Tuple[float, float] | None = None) -> None:
        if mult.size == 0 or self.n_amp == 0:
            return
        low, high = (bounds if bounds is not None else (0.2, 5.0))
        for a in range(self.n_amp):
            for e in range(self.n_exp):
                m = float(mult[a, e]) if np.isfinite(mult[a, e]) else np.nan
                if not np.isfinite(m):
                    self.score[a, e] *= 0.25
                    self.mask[a, e] = False
                    self._append_reason(a, e, "mult=nan")
                    continue
                if m < low or m > high:
                    self.score[a, e] *= 0.25
                    self._append_reason(a, e, "mult_out_of_bounds")
                    # mask only if far outside (20% beyond)
                    margin = 0.2 * (high - low)
                    if m < (low - margin) or m > (high + margin):
                        self.mask[a, e] = False
        self._update_mask_from_score()

    def update_with_poly2d(self, mult: np.ndarray, pred: np.ndarray, rel_thresh: Tuple[float, float] = (3.0, 5.0)) -> None:
        """Update labels using residuals to the per-exposure poly2d fit.

        Computes residuals r_e = |mult - pred| normalized by the per-exposure
        robust dispersion (MAD-based std) of residuals across amplifiers:
            scale_e = 1.4826 * median(|resid_e - median(resid_e)|)
            r = |m - p| / max(eps, scale_e)
        rel_thresh defines the soft and hard cut thresholds in units of this
        robust sigma (e.g., (3, 5) or (5, 10)).
        """
        if mult.size == 0 or self.n_amp == 0:
            return
        t_soft, t_hard = rel_thresh
        eps = 1e-12
        # Compute per-exposure robust scales (MAD-based std) over amps
        # Handle NaNs gracefully by ignoring non-finite values
        if pred.shape != mult.shape:
            # shape mismatch; nothing to do
            return
        resid = mult - pred  # (n_amp, n_exp)
        for e in range(self.n_exp):
            re = resid[:, e]
            re_finite = re[np.isfinite(re)]
            if re_finite.size == 0:
                scale_e = np.nan
            else:
                med = np.median(re_finite)
                mad = np.median(np.abs(re_finite - med))
                scale_e = 1.4826 * mad
            scale = scale_e if (np.isfinite(scale_e) and scale_e > 0) else eps
            for a in range(self.n_amp):
                m = float(mult[a, e])
                p = float(pred[a, e])
                if not (np.isfinite(m) and np.isfinite(p)):
                    continue
                r = abs(m - p) / max(eps, scale)
                if r > t_soft:
                    self.score[a, e] *= 0.8
                    self._append_reason(a, e, "poly2d_resid>soft")
                if r > t_hard:
                    self.mask[a, e] = False
                    self._append_reason(a, e, "poly2d_resid>hard")
        self._update_mask_from_score()

    def per_amp_summary(self) -> pd.DataFrame:
        if self.n_amp == 0:
            return pd.DataFrame(columns=["amp", "label"]).astype({"amp": int, "label": str})
        labels = []
        for a in range(self.n_amp):
            # worst-case across exposures
            if not self.mask[a, :].all():
                name = "bad"
            else:
                name = "good"
            labels.append({"amp": int(a), "label": name})
        return pd.DataFrame(labels)

    def good_mask(self) -> np.ndarray:
        if self.n_amp == 0:
            return np.array([], dtype=bool)
        # conservative: amp is good only if all exposures are good
        return self.mask.all(axis=1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_amp": int(self.n_amp),
            "n_exp": int(self.n_exp),
            "score": self.score.tolist(),
            "mask": self.mask.tolist(),
            "reasons": self._reasons_to_list(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LabelSet":
        n_amp = int(d.get("n_amp", 0))
        n_exp = int(d.get("n_exp", 0))
        score = np.array(d.get("score", []), dtype=float).reshape((n_amp, n_exp)) if n_amp and n_exp else np.zeros((0, 0))
        mask = np.array(d.get("mask", []), dtype=bool).reshape((n_amp, n_exp)) if n_amp and n_exp else np.zeros((0, 0), dtype=bool)
        reasons_list = d.get("reasons", [[""] * n_exp for _ in range(n_amp)])
        reasons = np.empty((n_amp, n_exp), dtype=object)
        for a in range(n_amp):
            for e in range(n_exp):
                reasons[a, e] = reasons_list[a][e] if a < len(reasons_list) and e < len(reasons_list[a]) else ""
        return LabelSet(n_amp=n_amp, n_exp=n_exp, score=score, mask=mask, reasons=reasons)

    def _append_reason(self, a: int, e: int, reason: str) -> None:
        prev = str(self.reasons[a, e]) if self.reasons[a, e] else ""
        self.reasons[a, e] = ",".join([r for r in [prev, reason] if r])

    def _reasons_to_list(self) -> list[list[str]]:
        out: list[list[str]] = []
        for a in range(self.n_amp):
            row: list[str] = []
            for e in range(self.n_exp):
                row.append(str(self.reasons[a, e]) if self.reasons[a, e] else "")
            out.append(row)
        return out

    def _update_mask_from_score(self, thresh: float = 0.1) -> None:
        # any score that has decayed below threshold is masked off
        self.mask = np.logical_and(self.mask, self.score >= thresh)


def save_labelset(path: str | Path, labels: LabelSet) -> None:
    p = Path(path)
    p.write_text(pd.Series(labels.to_dict()).to_json(indent=2), encoding="utf-8")


def load_labelset(path: str | Path) -> LabelSet:
    p = Path(path)
    if not p.exists():
        return LabelSet.initialize(0, 0)
    d = pd.read_json(p, typ="series").to_dict()
    return LabelSet.from_dict(d)


# Backward-compatible simple masking API retained

def label_amps(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Assign per-amp summary labels and good mask using conservative rules.

    This mirrors previous behavior for backward compatibility and is also used
    to seed the iterative LabelSet via LabelSet.from_features(df_features).
    """
    if df_features.empty:
        return pd.DataFrame(columns=["amp", "label"]).astype({"amp": int, "label": str}), np.array([], dtype=bool)

    ls = LabelSet.from_features(df_features)
    df_labels = ls.per_amp_summary()
    return df_labels[["amp", "label"]], ls.good_mask()