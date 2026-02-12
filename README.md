# Inoculate

Inoculate constructs a structured model of the sky signal across fibers, amplifiers, visits, and time. By exposing the data to a controlled representation of the contaminant signal, the pipeline builds resistance to systematic residuals that typically limit faint emission-line science.

This repository currently provides the foundational package and documentation scaffolding, modeled after Panacea's structure. Scientific methods and data models will be added iteratively.

## Install

Option A — Conda (recommended)

```bash
conda create -y -n inoculate python=3.11
conda activate inoculate
pip install -U .[docs]
```

Option B — venv (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -U .[docs]
```

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
