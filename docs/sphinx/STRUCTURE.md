# Sphinx Documentation Structure

This document describes the structure of the Sphinx documentation for OutBoxML.

## Directory Structure

```
docs/sphinx/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation index
├── Makefile             # Build script for Linux/Mac
├── make.bat             # Build script for Windows
├── requirements.txt     # Python dependencies
├── README.md            # Documentation overview
├── BUILD.md             # Build instructions
├── STRUCTURE.md         # This file
├── .gitignore           # Git ignore rules
│
├── _static/             # Static files (CSS, images)
├── _templates/          # Custom templates
│
├── installation.rst     # Installation guide
├── quickstart.rst      # Quick start guide
├── contributing.rst    # Contributing guidelines
├── changelog.rst       # Changelog
│
├── modules/             # API reference
│   ├── index.rst
│   ├── automl_manager.rst
│   ├── datadrift.rst
│   ├── datasets_manager.rst
│   ├── extractors.rst
│   ├── feature_selection.rst
│   ├── hyperparameter_tuning.rst
│   ├── metrics.rst
│   ├── models.rst
│   ├── monitoring_manager.rst
│   ├── plots.rst
│   ├── export_results.rst
│   └── core/
│       ├── index.rst
│       ├── config_builders.rst
│       ├── data_prepare.rst
│       ├── enums.rst
│       ├── pydantic_models.rst
│       ├── utils.rst
│       ├── validators.rst
│       ├── predict.rst
│       └── monitoring_factory.rst
│
├── examples/            # Example documentation
│   ├── index.rst
│   ├── titanic_basic.rst
│   ├── titanic_extractor.rst
│   ├── titanic_full.rst
│   ├── energy_efficiency_basic.rst
│   ├── energy_efficiency_extractor.rst
│   ├── energy_efficiency_custom.rst
│   └── housing_pricing.rst
│
└── api/                 # API index
    └── index.rst
```

## Key Files

### conf.py
Main Sphinx configuration file. Contains:
- Project metadata
- Extension configuration
- Autodoc settings
- Theme configuration
- Path setup for importing modules

### index.rst
Main entry point for documentation. Contains:
- Welcome message
- Table of contents
- Links to all major sections

### Module RST Files
Each module has a corresponding `.rst` file that uses `.. automodule::` to automatically generate documentation from docstrings.

## Building Documentation

See `BUILD.md` for detailed build instructions.

Quick build:
```bash
cd docs/sphinx
make html  # or make.bat html on Windows
```

## Adding New Documentation

1. **Add a new module:**
   - Create `modules/your_module.rst`
   - Add `.. automodule:: outboxml.your_module`
   - Add entry to `modules/index.rst`

2. **Add a new example:**
   - Create `examples/your_example.rst`
   - Add entry to `examples/index.rst`

3. **Add a new page:**
   - Create `your_page.rst`
   - Add entry to `index.rst` toctree

## Documentation Style

- All docstrings use reStructuredText format
- Examples are included in docstrings using `Example::`
- Parameters use `:param:` and `:type:`
- Return values use `:return:` and `:rtype:`
- Attributes use `:var:` and `:vartype:`
