import sys
sys.setrecursionlimit(1500) 

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GraphToolbox'
copyright = '2024, Eloi Campagne'
author = 'Eloi Campagne'

release = '0.1'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
]

mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = 'graphtoolbox.png'
html_theme_options = {
    "repository_url": "https://github.com/Exion35/GraphToolbox",
    "use_repository_button": True,
    "collapse_navigation": False,
    "logo_only": True,
    "extra_navbar": f"<p>Version: {release}</p>",
    "fixed_sidebar": True,
}
html_static_path = ['_static']

add_module_names = False
numpydoc_show_class_members = False

autodoc_member_order = "bysource"

# -- Options for EPUB output
epub_show_urls = 'footnote'