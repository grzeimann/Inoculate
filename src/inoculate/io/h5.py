"""
Lightweight HDF5 accessors tailored to VIRUS HETDEX shot files.

The functions here avoid loading full fiber×wavelength arrays by providing
slice helpers that operate amplifier-by-amplifier and exposure-by-exposure.

This module depends optionally on PyTables. If it's not installed, an
ImportError with a helpful message is raised upon use.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, Optional, Tuple

import numpy as np

from ..constants import EXPOSURES_PER_SHOT, NUM_FIBERS_PER_AMP
from ..utils import SchemaError

try:  # Optional dependency
    import tables  # type: ignore
except Exception:  # pragma: no cover - handled at call time
    tables = None  # type: ignore


@contextmanager
def open_h5(path: str):
    """Context manager that opens an HDF5 file with PyTables if available.

    Parameters
    ----------
    path:
        File path to the VIRUS spectral HDF5 file.

    Yields
    ------
    tables.File
        An open PyTables handle. Closes automatically on exit.
    """
    if tables is None:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTables is required to read HDF5 inputs. Install with `pip install tables`"
        )
    h5 = tables.open_file(path)
    try:
        yield h5
    finally:
        h5.close()


def amp_exposure_slice(amp_index: int, exposure_index: int, fibers_per_amp: int = NUM_FIBERS_PER_AMP,
                       exposures_per_shot: int = EXPOSURES_PER_SHOT) -> slice:
    """Return a Python slice for rows corresponding to a given amp and exposure.

    The Goals document specifies 112 fibers per amplifier and fibers ordered as
    [exp1(112), exp2(112), exp3(112)] within each amplifier block.

    Parameters
    ----------
    amp_index:
        Zero-based amplifier index within the file.
    exposure_index:
        Zero-based exposure index within the shot (0..2).
    fibers_per_amp:
        Number of fibers per amplifier (default 112).
    exposures_per_shot:
        Number of exposures per shot (default 3).

    Returns
    -------
    slice
        A slice(start, stop) selecting the fibers for the requested amp and exposure.
    """
    if not (0 <= exposure_index < exposures_per_shot):
        raise ValueError("exposure_index out of range")
    if amp_index < 0:
        raise ValueError("amp_index must be non-negative")

    block = exposures_per_shot * fibers_per_amp
    start = amp_index * block + exposure_index * fibers_per_amp
    stop = start + fibers_per_amp
    return slice(start, stop)


@dataclass
class H5VIRUS:
    """Memory-safe reader for VIRUS HDF5 shot files.

    Attributes:
        path: Path to the HDF5 file.
        table_path: Optional path inside HDF5 to the spectral table. If None,
            the reader will auto-detect among common locations ("/Spectra",
            "/Fibers").
    """

    path: str | Path
    table_path: str | None = None
    _resolved_table_path: str | None = None  # cached after first detection

    def _require_tables(self) -> None:
        if tables is None:  # pragma: no cover - env dependent
            raise ImportError(
                "PyTables is required to read HDF5 inputs. Install with `pip install tables`"
            )

    def _open(self):  # returns tables.File
        self._require_tables()
        return tables.open_file(str(self.path))

    def _get_spectra_node(self, h5) -> "tables.Table":  # type: ignore[name-defined]
        """Return the spectral table node, auto-detecting its path if needed.

        Tries, in order: explicit ``table_path`` if provided, cached resolved
        path from a prior call, then common candidates: "/Spectra", "/Fibers".
        Raises SchemaError if none exist.
        """
        # Use cached path if available
        if self._resolved_table_path is not None:
            try:
                return h5.get_node(self._resolved_table_path)
            except Exception:
                # fall through to re-detect in case file changed
                self._resolved_table_path = None
        # Try explicit path first
        candidates: list[str] = []
        if self.table_path:
            candidates.append(self.table_path)
        # Common defaults
        for p in ("/Spectra", "/Fibers"):
            if p not in candidates:
                candidates.append(p)
        last_exc: Exception | None = None
        for p in candidates:
            try:
                node = h5.get_node(p)
                # Basic sanity: require it has column access (PyTables Table or EArray with cols)
                if hasattr(node, "cols"):
                    self._resolved_table_path = p
                    return node  # type: ignore[return-value]
            except Exception as exc:
                last_exc = exc
        raise SchemaError(
            f"Could not locate spectral table. Tried paths: {candidates}. Last error: {last_exc}"
        )

    def read_info(self) -> Dict[str, object]:
        """Read file metadata (date, ifuslots, amps, shapes) with validation.

        Returns:
            Mapping with keys: date (int|str), ifuslots (np.ndarray[str]),
            amps (np.ndarray[str]), n_rows (int), n_wave (int), n_amp (int),
            exposures (int), fibers_per_amp (int).

        Raises:
            SchemaError: If required nodes/columns are missing or layout invalid.
        """
        import os

        self._require_tables()
        with self._open() as h5:
            # Get Info node (metadata)
            try:
                info = h5.root.Info
            except Exception as exc:  # pragma: no cover - file dependent
                raise SchemaError(f"Missing required /Info table: {exc}") from exc

            # Locate spectral table (auto-detects /Spectra or /Fibers)
            try:
                spectra = self._get_spectra_node(h5)
            except SchemaError:
                raise
            except Exception as exc:  # pragma: no cover
                raise SchemaError(f"Failed to access spectral table: {exc}") from exc

            try:
                ifuslots = info.cols.ifuslot[:]
                amps_raw = info.cols.amp[:]
                amps = np.array([a.decode("utf-8") if isinstance(a, (bytes, bytearray)) else str(a) for a in amps_raw])
            except Exception as exc:  # pragma: no cover
                raise SchemaError(f"/Info table is missing required columns (ifuslot, amp): {exc}") from exc

            # Validate spectral columns existence minimally
            required_cols = ["spectrum", "skyspectrum", "error"]
            for c in required_cols:
                if not hasattr(spectra.cols, c):
                    raise SchemaError(f"Spectral table missing required column: {c}")

            n_rows = spectra.nrows
            # Infer wavelength length from first row spectrum
            try:
                first = spectra.cols.spectrum[0]
                n_wave = int(len(first))
            except Exception as exc:  # pragma: no cover
                raise SchemaError(f"Unable to read spectrum length: {exc}") from exc

            fibers_per_amp = NUM_FIBERS_PER_AMP
            exposures = EXPOSURES_PER_SHOT
            block = fibers_per_amp * exposures
            if n_rows % fibers_per_amp != 0:
                raise SchemaError("Row count is not divisible by fibers_per_amp=112")
            if (n_rows // fibers_per_amp) % exposures != 0:
                raise SchemaError("Row count does not match multiple of (exposures * fibers_per_amp)")
            n_amp = (n_rows // block)
            if n_amp <= 0:
                raise SchemaError("Computed zero amplifiers from row count")

            # Optional: basic ordering check by sampling slice starts
            # We ensure that row indices for exp slices are increasing and in-bounds
            for a in (0, max(0, n_amp - 1)):
                for e in range(exposures):
                    s = amp_exposure_slice(a, e, fibers_per_amp, exposures)
                    if not (0 <= s.start < n_rows and 0 < s.stop <= n_rows and s.stop - s.start == fibers_per_amp):
                        raise SchemaError("Computed amp/exp slice out of bounds")

            # Derive date from filename (YYYYMMDD* prefix typical); keep as string if not int
            base = os.path.basename(str(self.path))
            date_token = base.split("_")[0]
            try:
                date: int | str = int(date_token)
            except Exception:
                date = date_token

            return {
                "date": date,
                "ifuslots": ifuslots,
                "amps": amps,
                "n_rows": int(n_rows),
                "n_wave": int(n_wave),
                "n_amp": int(n_amp),
                "exposures": int(exposures),
                "fibers_per_amp": int(fibers_per_amp),
            }

    def iter_amp_blocks(self, cols: list[str]) -> Generator[Tuple[int, int, slice, Dict[str, np.ndarray]], None, None]:
        """Iterate amplifier/exposure blocks lazily.

        Args:
            cols: Column names to read from the spectral table.

        Yields:
            Tuples of (amp_idx, exp_idx, row_slice, arrays) where arrays is a
            mapping from requested column name to numpy array for that slice.
        """
        self._require_tables()
        info = self.read_info()
        n_amp = int(info["n_amp"])  # type: ignore[arg-type]
        exposures = int(info["exposures"])  # type: ignore[arg-type]
        fibers_per_amp = int(info["fibers_per_amp"])  # type: ignore[arg-type]

        with self._open() as h5:
            spectra = self._get_spectra_node(h5)
            for a in range(n_amp):
                for e in range(exposures):
                    s = amp_exposure_slice(a, e, fibers_per_amp, exposures)
                    out: Dict[str, np.ndarray] = {}
                    for c in cols:
                        if not hasattr(spectra.cols, c):
                            raise SchemaError(f"Spectral table missing required column: {c}")
                        # Read only the rows we need; avoid loading entire column
                        out[c] = spectra.cols._f_col(c)[s]  # type: ignore[attr-defined]
                    yield a, e, s, out
