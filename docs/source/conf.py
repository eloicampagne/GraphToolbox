import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'GraphToolbox'
copyright = '2025, Eloi Campagne'
author = 'Eloi Campagne'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# Furo gère les .rst et les .md
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_title = "GraphToolbox Documentation"

# Personnalisation du thème
html_theme_options = {
    "light_logo": "logo_light.png",   # optionnel
    "dark_logo": "logo_dark.png",     # optionnel
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",     # affiche un bouton "Edit on GitHub"
    "source_repository": "https://github.com/eloicampagne/GraphToolbox/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# -- Autodoc options ---------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': True,
}
autodoc_typehints = "description"

# -- Napoleon (docstrings style NumPy / Google) ------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
