# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import the module to verify it's accessible
try:
    import outboxml.automl_manager
    print(f"Successfully imported outboxml.automl_manager from {project_root}")
except ImportError as e:
    print(f"Warning: Could not import outboxml.automl_manager: {e}")

project = 'OutBoxML'
copyright = '2025, OutBoxML Contributors'
author = 'OutBoxML Team'
release = '0.9.5'
version = '0.9.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None
html_favicon = None

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google/NumPy style docstrings
# Note: Napoleon is primarily for Google/NumPy style, but we use reST format
# So we configure it to be more permissive
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False  # Hide private methods even if they have docstrings
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
# These options control what is included in the generated documentation
autodoc_default_options = {
    'members': True,              # Show all public members (methods, attributes)
    'member-order': 'bysource',   # Order members by source code order
    'special-members': '__init__', # Show __init__ method (but not other special methods)
    'undoc-members': True,         # Show members without docstrings (but only public ones)
    'exclude-members': '__weakref__',  # Exclude specific members
    'show-inheritance': True,     # Show inheritance information
    'inherited-members': False,    # Don't show inherited members
    'private-members': False,      # Hide private methods/attributes (starting with _)
    # Note: Methods starting with _ (single underscore) are considered private
    # Methods starting with __ (double underscore) are also hidden unless in special-members
}

# Explicitly exclude private methods from documentation
# This ensures that methods starting with _ are never shown
autodoc_member_order = 'bysource'

def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip private members (starting with _) from documentation."""
    # Skip private methods and attributes (starting with single underscore)
    # but allow __init__ and other special methods if explicitly requested
    if name.startswith('_') and not name.startswith('__'):
        return True
    # Skip double-underscore methods except __init__ if it's in special-members
    if name.startswith('__') and name != '__init__':
        return True
    return skip

def setup(app):
    """Setup function for Sphinx extensions."""
    app.connect('autodoc-skip-member', autodoc_skip_member)

autodoc_mock_imports = [
    'mlflow', 'loguru', 'pandas', 'numpy', 'sklearn', 'plotly', 
    'optuna', 'catboost', 'xgboost', 'dotenv', 'pydantic',
    'sqlalchemy', 'psycopg2', 'lightgbm', 'scikit-learn', 'environs'
]
autodoc_inherit_docstrings = True
autodoc_typehints = 'description'

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True
