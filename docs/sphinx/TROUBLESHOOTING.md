# Troubleshooting Sphinx Documentation

## Problem: Empty Pages / No Docstrings

If you see empty pages in the generated documentation, here are steps to diagnose and fix:

### 1. Check Module Import

Test if the module can be imported:

```bash
cd docs/sphinx
python -c "import sys; sys.path.insert(0, '..'); from outboxml import automl_manager; print('Import successful')"
```

If this fails, check:
- All dependencies are installed or mocked in `conf.py`
- The project root path is correct in `conf.py`
- Python path includes the project root

### 2. Check Autodoc Configuration

In `conf.py`, ensure:
- `autodoc_default_options` includes `'members': True`
- `autodoc_mock_imports` includes all external dependencies
- `sys.path.insert(0, str(project_root))` is set correctly

### 3. Verify Docstrings Format

Docstrings should be in reStructuredText format. Check that:
- Parameters use `:param:` and `:type:`
- Returns use `:return:` and `:rtype:`
- Examples use `.. code-block:: python`

### 4. Build with Verbose Output

Build with verbose output to see errors:

```bash
sphinx-build -v -b html . _build/html
```

### 5. Check for Import Errors

Look for warnings in the build output about:
- Missing modules
- Import errors
- Syntax errors in docstrings

### 6. Test Individual Classes

Try documenting a single class:

```rst
.. autoclass:: outboxml.automl_manager.AutoMLResult
   :members:
```

If this works, the issue is with the module-level documentation.

### Common Fixes

1. **Add missing mocks**: Add any missing packages to `autodoc_mock_imports`
2. **Fix path issues**: Ensure `project_root` points to the correct directory
3. **Check docstring format**: Ensure docstrings use proper reST syntax
4. **Enable undoc-members**: Set `'undoc-members': True` in `autodoc_default_options` to see if members exist but aren't documented
