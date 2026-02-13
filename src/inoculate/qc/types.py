"""QC types and data structures.

Defines LabelState, an explicit per-amp/per-exp label container that keeps
separate masks for downstream actions. This follows the Iterative Labeling
Architecture document and complements the lightweight LabelSet used in
qc.labels for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "amp_id",
    "exp_id",
    "label",
    "mask_good_for_mult",
    "mask_good_for_illum",
    "mask_good_for_pca",
    "mask_good_for_fit",
    "confidence",
    "reasons",
]


@dataclass
class LabelState:
    """Explicit per-amp/per-exp label state with action-specific masks.

    Attributes:
        df: pandas DataFrame with required columns and one row per (amp_id, exp_id).
        frozen: If True, prevents further updates (advisory only).
    """

    df: pd.DataFrame
    frozen: bool = False

    @staticmethod
    def initialize(n_amp: int, n_exp: int) -> "LabelState":
        idx = [(a, e) for a in range(int(n_amp)) for e in range(int(n_exp))]
        if not idx:
            df = pd.DataFrame(columns=REQUIRED_COLUMNS)
            return LabelState(df=df)
        base = pd.DataFrame(idx, columns=["amp_id", "exp_id"]).astype({"amp_id": int, "exp_id": int})
        base["label"] = "good"
        base["mask_good_for_mult"] = True
        base["mask_good_for_illum"] = True
        base["mask_good_for_pca"] = True
        base["mask_good_for_fit"] = True
        base["confidence"] = 1.0
        base["reasons"] = ""
        return LabelState(df=base[REQUIRED_COLUMNS])

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        # store as records for human-readability and robustness
        records = self.df.to_dict(orient="records")
        pd.Series({"frozen": self.frozen, "records": records}).to_json(p, indent=2)

    @staticmethod
    def from_json(path: str | Path) -> "LabelState":
        p = Path(path)
        if not p.exists():
            return LabelState.initialize(0, 0)
        s = pd.read_json(p, typ="series")
        records = s.get("records", [])
        df = pd.DataFrame(records)
        # ensure all required columns exist
        for c in REQUIRED_COLUMNS:
            if c not in df.columns:
                if c.startswith("mask_good_"):
                    df[c] = False
                elif c == "confidence":
                    df[c] = 0.0
                elif c == "label":
                    df[c] = "unknown"
                elif c in ("amp_id", "exp_id"):
                    # try to recover from any present columns
                    if c == "amp_id" and "amp" in df.columns:
                        df[c] = df["amp"].astype(int)
                    elif c == "exp_id" and "exp" in df.columns:
                        df[c] = df["exp"].astype(int)
                    else:
                        df[c] = 0
                else:
                    df[c] = ""
        df = df[REQUIRED_COLUMNS]
        frozen = bool(s.get("frozen", False))
        return LabelState(df=df, frozen=frozen)

    # Convenience helpers for iterative updates
    def set_mask(self, col: str, mask_updates: Iterable[tuple[int, int, bool]], reason: str | None = None) -> None:
        if self.frozen:
            return
        if col not in self.df.columns:
            raise KeyError(f"Unknown mask column: {col}")
        for a, e, val in mask_updates:
            sel = (self.df["amp_id"] == int(a)) & (self.df["exp_id"] == int(e))
            if not np.any(sel):
                continue
            self.df.loc[sel, col] = bool(val)
            if reason:
                self._append_reason(a, e, reason)

    def _append_reason(self, a: int, e: int, reason: str) -> None:
        sel = (self.df["amp_id"] == int(a)) & (self.df["exp_id"] == int(e))
        if not np.any(sel):
            return
        prev = self.df.loc[sel, "reasons"].astype(str).fillna("").values[0]
        new = ",".join([r for r in [prev, reason] if r])
        self.df.loc[sel, "reasons"] = new

    # Derive per-amp masks
    def mask_for_mult(self) -> np.ndarray:
        return self._per_amp_all_true("mask_good_for_mult")

    def mask_for_pca(self) -> np.ndarray:
        return self._per_amp_all_true("mask_good_for_pca")

    def mask_for_fit(self) -> np.ndarray:
        return self._per_amp_all_true("mask_good_for_fit")

    def _per_amp_all_true(self, col: str) -> np.ndarray:
        if self.df.empty:
            return np.array([], dtype=bool)
        n_amp = int(self.df["amp_id"].max()) + 1
        good = np.ones(n_amp, dtype=bool)
        for a, g in self.df.groupby("amp_id"):
            if not bool(np.all(g[col].to_numpy(dtype=bool))):
                good[int(a)] = False
        return good

    # Round-trip with the lightweight LabelSet (optional use)
    @staticmethod
    def from_labelset(ls: Any) -> "LabelState":
        # late import to avoid hard dependency in this module
        n_amp = getattr(ls, "n_amp", 0)
        n_exp = getattr(ls, "n_exp", 0)
        state = LabelState.initialize(n_amp, n_exp)
        if n_amp == 0:
            return state
        mask = getattr(ls, "mask", np.ones((n_amp, n_exp), dtype=bool))
        reasons = getattr(ls, "reasons", np.empty((n_amp, n_exp), dtype=object))
        for a in range(n_amp):
            for e in range(n_exp):
                use = bool(mask[a, e])
                state.set_mask("mask_good_for_mult", [(a, e, use)])
                state.set_mask("mask_good_for_pca", [(a, e, use)])
                state.set_mask("mask_good_for_fit", [(a, e, use)])
                r = str(reasons[a, e]) if reasons.size else ""
                if r:
                    state._append_reason(a, e, r)
        return state

    def to_labelset(self) -> Any:
        # produce a minimal compatible structure used by pipeline
        n_amp = int(self.df["amp_id"].max()) + 1 if not self.df.empty else 0
        n_exp = int(self.df["exp_id"].max()) + 1 if not self.df.empty else 0
        score = np.ones((n_amp, n_exp), dtype=float)
        mask = np.ones((n_amp, n_exp), dtype=bool)
        reasons = np.empty((n_amp, n_exp), dtype=object)
        reasons[:] = ""
        for _, row in self.df.iterrows():
            a = int(row["amp_id"]) ; e = int(row["exp_id"]) ;
            ok = bool(row.get("mask_good_for_mult", True))
            mask[a, e] = ok
            conf = float(row.get("confidence", 1.0))
            score[a, e] = conf
            r = str(row.get("reasons", ""))
            reasons[a, e] = r
        # lightweight object with identical attributes
        class _LS:
            def __init__(self, n_amp, n_exp, score, mask, reasons) -> None:
                self.n_amp = n_amp ; self.n_exp = n_exp
                self.score = score ; self.mask = mask ; self.reasons = reasons
            def good_mask(self) -> np.ndarray:
                return self.mask.all(axis=1)
            def per_amp_summary(self) -> pd.DataFrame:
                if self.n_amp == 0:
                    return pd.DataFrame(columns=["amp", "label"]).astype({"amp": int, "label": str})
                labels = []
                for a in range(self.n_amp):
                    labels.append({"amp": int(a), "label": "good" if self.mask[a, :].all() else "bad"})
                return pd.DataFrame(labels)
        return _LS(n_amp, n_exp, score, mask, reasons)
