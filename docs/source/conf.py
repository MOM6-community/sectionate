# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as get_version
from pathlib import Path
import shutil

# The master toctree document.
master_doc = "index"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'xbudget'
copyright = '2026, Henri Drake'
author = 'Henri Drake'

release = get_version(project)
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx"
]

templates_path = ['_templates']
exclude_patterns = []

# Don't execute notebooks on RTD unless you want that
nbsphinx_execute = "never"

# -- Copy notebooks from repo root/examples into docs/source/examples --------

HERE = Path(__file__).resolve()
DOCS_SOURCE = HERE.parent                       # docs/source
REPO_ROOT = HERE.parents[2]                     # up two levels from conf.py
EXAMPLES_SRC = REPO_ROOT / "examples"
EXAMPLES_DST = DOCS_SOURCE / "examples"

def _sync_examples():
    if not EXAMPLES_SRC.exists():
        return

    # fresh copy so removed notebooks don't linger
    if EXAMPLES_DST.exists():
        shutil.rmtree(EXAMPLES_DST)
    EXAMPLES_DST.mkdir(parents=True, exist_ok=True)

    for nb in EXAMPLES_SRC.rglob("*.ipynb"):
        rel = nb.relative_to(EXAMPLES_SRC)
        out = EXAMPLES_DST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nb, out)

_sync_examples()

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = f"{project} v{version} documentation"

html_theme = "alabaster"

html_theme_options = {
    "sidebar_width": "270px",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

latex_documents = [
    ("index")
]