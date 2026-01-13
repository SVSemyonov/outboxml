# Sphinx Documentation

This directory contains the Sphinx documentation source files for OutBoxML.

## Building the Documentation

### Prerequisites

Install Sphinx and required extensions:

```bash
pip install -r requirements.txt
```

### Building HTML Documentation

On Linux/Mac:

```bash
make html
```

On Windows:

```bash
make.bat html
```

The generated HTML documentation will be in `_build/html/`.

### Building PDF Documentation

On Linux/Mac:

```bash
make latexpdf
```

On Windows:

```bash
make.bat latexpdf
```

### Viewing the Documentation

After building, open `_build/html/index.html` in your web browser.

## Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index
- `modules/` - API reference documentation
- `examples/` - Example documentation
- `_templates/` - Custom templates (optional)
- `_static/` - Static files (CSS, images, etc.)

## Adding New Documentation

1. Create a new `.rst` file in the appropriate directory
2. Add it to the relevant `toctree` in the index file
3. Rebuild the documentation
