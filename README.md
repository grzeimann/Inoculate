# Inoculate

Inoculate constructs a structured model of the sky signal across fibers, amplifiers, visits, and time. By exposing the data to a controlled representation of the contaminant signal, the pipeline builds resistance to systematic residuals that typically limit faint emission-line science.

This repository currently provides the foundational package and documentation scaffolding, modeled after Panacea's structure. Scientific methods and data models will be added iteratively.

## Install

Option A — Conda (recommended)

```bash
# Create the environment from the YAML file
conda env create -f environment.yml
conda activate inoculate
# (Optional) If the file changes later, update with:
# conda env update -f environment.yml --prune
```

Option B — venv (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -U .[docs]
```

## Run the Single-Shot workflow

- CLI (recommended):
  - Basic run
    
    ```bash
    inoculate-shot /path/to/shot.h5 --outdir /desired/output/dir --resume --log-level INFO
    ```
  - With diagnostic plots written to OUTDIR/plots (optional)
    
    ```bash
    inoculate-shot /path/to/shot.h5 --outdir /desired/output/dir --resume --make-plots --max-plots 12 --log-level INFO
    ```
  - Example
    
    ```bash
    inoculate-shot /data/virus/20240101_shot.h5 --outdir ./inoculate_out --resume --log-level INFO
    ```

- Python API:

  ```python
  from inoculate.shot.pipeline import run_shot
  result = run_shot("/path/to/shot.h5", "./inoculate_out", modelspec=None, resume=True)
  print(result["manifest"])  # path to stage_07_model_start_manifest.json
  ```

Notes
- Resume: With --resume (or resume=True) the pipeline skips any stage whose artifact already exists.
- Produced artifacts (in --outdir):
  - stage_00_info.json
  - stage_01_bw_amp.npz
  - stage_02_bw_full.npz
  - stage_03_amp_qc.parquet
  - stage_04_mult.npz
  - stage_0425_mult_poly2d.npz
  - stage_045_poly.npz
  - stage_05_pca.npz
  - stage_06_amp_fits.parquet
  - stage_07_model_start_manifest.json
- Optional helper: Show amp/exp slice calculation without running the pipeline
  
  ```bash
  inoculate-shot --show-slice --amp 0 --exp 1
  ```

Environment reminder
- Activate your env first:
  
  ```bash
  conda activate inoculate
  # Ensure dependencies from environment.yml are installed.
  ```

## Diagnostics plots

You can visualize key outputs from a completed run using the plotting utilities.
Make sure matplotlib is installed (it is included in environment.yml).

Python examples:

```python
from inoculate.plot import (
    plot_mult_by_amp,       # NEW: mult summary across amps
    plot_fit_example,       # NEW: 2-panel: initial residual+components; final residual
    plot_bw_amp_vs_full,    # single amp/exp comparison
)

outdir = "./inoculate_out"

# 1) Multiplicative summary across all amps (x=amp index, y=mult per exposure)
plot_mult_by_amp(outdir, show_labels=True, show=True)

# 2) One comprehensive fit example (initial residual, model parts, final residual)
plot_fit_example(outdir, show=True)  # auto-selects a good amp/exp

# 3) Optional: compare BW_amp to scaled BW_full for the same amp/exp
plot_bw_amp_vs_full(outdir, amp=0, exp=0, show=True)
```

To save figures instead of showing interactively, pass a file path via `save`:

```python
plot_fit_example(outdir, save="figs/fit_example.png")
```

Notes:
- The previous per-amp wavelength ratio function `plot_amp_ratio` is kept for
  backward compatibility but the intended multiplicative overview is now provided
  by `plot_mult_by_amp`.

## Build the docs (Sphinx + MyST)

```bash
SPHINX_THEME=sphinx_rtd_theme sphinx-build -b html docs docs/_build/html
# macOS
open docs/_build/html/index.html
# Linux
# xdg-open docs/_build/html/index.html
```

If the build warns about 'linkify', either install linkify-it-py or remove 'linkify' from myst_enable_extensions in docs/conf.py.

## Testing

```bash
pytest
```

## Links

- Docs landing page source: docs/index.md
- Getting started: docs/getting-started/
- Citation: CITATION.cff
- License: BSD-3-Clause (LICENSE)
