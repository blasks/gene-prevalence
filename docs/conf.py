project = "gene-prevalence"
author = "blasks"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "numpydoc",
]

numpydoc_class_members_toctree = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
