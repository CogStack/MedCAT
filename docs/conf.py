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
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'MedCAT'
copyright = '2022, CogStack Org'
author = 'CogStack Org'

# The full version, including alpha/beta/rc tags
release = ':latest'  # where is the version retrievable from?


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'myst_parser',
    'sphinx.ext.napoleon',
    'autoapi.extension',
    'sphinx.ext.inheritance_diagram',
]

autoapi_type = 'python'
autoapi_dirs = ['../medcat']
autodoc_typehints = 'description'
autoapi_template_dir = '_templates/autoapi_templates'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/overrides.css'
]

html_logo = '../media/cat-logo.svg'

html_theme_options = {
    'display_version': True,
    'logo_only': True,
}

# Render multi-type Returns blocks correctly
napoleon_custom_sections = [('Returns', 'params_style')]


def autoapi_skip_member(app, what, name, obj, skip, options):
    # skip:
    #   log class attributes
    #   'private' methods, attributes, functions
    exclude = (what == 'attribute' and name == 'log') or \
        (name.startswith('_') and not name.startswith('__'))
    return exclude


def setup(app):
    """Add autoapi-skip-member."""
    app.connect('autoapi-skip-member', autoapi_skip_member)
