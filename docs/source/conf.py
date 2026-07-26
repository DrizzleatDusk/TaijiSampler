# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

project = 'TaijiSampler'
copyright = '2026, DrizzleatDusk'
author = 'DrizzleatDusk'
release = '0.1.0'

extensions = [
    'myst_parser',
    'sphinx.ext.mathjax',
]
myst_enable_extensions = [
    'dollarmath',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_theme_options = {
   "logo": {
        "image_light": "_static/taiji_logo_light.png",
        "image_dark": "_static/taiji_logo_dark.png",
   }
}