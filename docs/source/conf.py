# Configuration file for the Sphinx documentation builder.

import os
import sys
import inspect

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'GraphToolbox'
copyright = '2025, Eloi Campagne'
author = 'Eloi Campagne'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # docstrings Google/Numpy
    "sphinx.ext.viewcode",   # liens internes vers le code
    "sphinx.ext.linkcode",   # liens externes vers GitHub
    "myst_parser",           # lecture des fichiers .md
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ['_static']

html_theme_options = {
    "light_logo": "logo_light.png",
    "dark_logo": "logo_dark.png",
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
}

# -- Autodoc config ----------------------------------------------------------
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

suppress_warnings = ["autodoc.noindex"]

# -- Link to GitHub source code ----------------------------------------------
# ⚙️ Change the repository URL below if needed
github_user = "eloicampagne"
github_repo = "GraphToolbox"
github_branch = "main"  # or "master" if applicable

def linkcode_resolve(domain, info):
    """
    Return the GitHub URL corresponding to the object being documented.
    """
    if domain != 'py' or not info['module']:
        return None

    try:
        module = sys.modules[info['module']]
        obj = module
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        # Try to get the source file and line numbers
        fn = inspect.getsourcefile(obj)
        if not fn:
            return None
        fn = os.path.relpath(fn, start=os.path.dirname(__file__) + "/../../")
        source, lineno = inspect.getsourcelines(obj)
        return f"https://github.com/{github_user}/{github_repo}/blob/{github_branch}/{fn}#L{lineno}-L{lineno + len(source) - 1}"
    except Exception:
        return None

def setup(app):
    # Pour un fichier local
    app.add_css_file("styles.css")