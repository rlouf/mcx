# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "mcx"
copyright = "2020, Rémi Louf"
author = "Rémi Louf"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

pygments_style = "nord"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

html_theme_options = {
    "description": "MCX (pronounced /miks'/) is a python probabilistic programming library based on source code transformation.",
    "show_powered_by": False,
    "github_user": "rlouf",
    "github_repo": "mcx",
    "github_banner": True,
    "show_related": False,
    "github_button": True,
    "github_type": "star",
    "github_count": False,
    "note_bg": "#FFF59C",
    "pre_bg": "#3B4252",
    "page_width": 1140,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
