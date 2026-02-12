# Configuration file for the Sphinx documentation builder.
# This follows a minimal Panacea-like setup using Markdown via MyST.

import os
import sys
from datetime import datetime

# Ensure package is importable
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

project = "Inoculate"
author = "Inoculate Developers"
current_year = str(datetime.now().year)
copyright = f"{current_year}, {author}"

# Read version from the package
try:
    from inoculate import __version__ as release  # type: ignore
except Exception:
    release = "0.0.1"

# -- General configuration ------------------------------------------------
extensions = [
    "myst_parser",
]

# Allow both .md and .rst files
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# Use docs/index.md as the master document
master_doc = "index"

# MyST configuration (optional niceties)
myst_heading_anchors = 3
myst_enable_extensions = [
    "linkify",
]

# -- Options for HTML output ----------------------------------------------
html_theme = os.getenv("SPHINX_THEME", "alabaster")
html_title = project
