# Building Sphinx Documentation

## Quick Start

1. **Install dependencies:**

   ```bash
   cd docs/sphinx
   pip install -r requirements.txt
   ```

2. **Build HTML documentation:**

   On Linux/Mac:
   ```bash
   make html
   ```

   On Windows:
   ```bash
   make.bat html
   ```

3. **View the documentation:**

   Open `_build/html/index.html` in your web browser.

## Available Build Targets

- `html` - Build HTML documentation (default)
- `latexpdf` - Build PDF documentation (requires LaTeX)
- `clean` - Remove build files
- `help` - Show available targets

## Troubleshooting

### Import Errors

If you encounter import errors when building, ensure that:
- The project root is in your Python path (configured in `conf.py`)
- All required packages are installed
- Mock imports are configured correctly in `conf.py`

### Missing Modules

If Sphinx cannot find modules:
- Check that `sys.path.insert(0, str(project_root))` in `conf.py` points to the correct directory
- Verify that the `outboxml` package is importable

### Theme Issues

If the Read the Docs theme is not found:
```bash
pip install sphinx-rtd-theme
```
