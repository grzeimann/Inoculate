# Installation

These instructions follow the Panacea docs style: build a clean environment, install the package, then build docs locally.

## Option A — Conda (recommended)

```bash
# In the repo root
git clone https://github.com/<your-org>/Inoculate.git
cd Inoculate

# Create and activate the environment
conda create -y -n inoculate python=3.11
conda activate inoculate

# Install package plus docs extras
pip install -U .[docs]
```

## Option B — venv (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -U .[docs]
```

## Build the documentation (Sphinx + MyST)

```bash
SPHINX_THEME=sphinx_rtd_theme sphinx-build -b html docs docs/_build/html
# macOS
open docs/_build/html/index.html
# Linux
# xdg-open docs/_build/html/index.html
```

If the build warns about 'linkify', either install linkify-it-py or remove 'linkify' from myst_enable_extensions in docs/conf.py.
