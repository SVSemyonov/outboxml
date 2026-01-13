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

## Automated Builds (CI/CD)

Documentation is automatically built when:

- **Pull Requests** are created targeting the `develop` branch (GitHub)
- **Merge Requests** are created targeting the `develop` branch (GitLab)
- Code changes are pushed to the `develop` branch

### GitHub Actions

The workflow (`.github/workflows/docs.yml`) will:
- Build documentation on PRs and pushes to `develop`
- Upload build artifacts for review
- Deploy to GitHub Pages when pushed to `develop`

### GitLab CI

The pipeline (`.gitlab-ci.yml`) will:
- Build documentation on merge requests and pushes to `develop`
- Store build artifacts for download
- Deploy to GitLab Pages when pushed to `develop`

### Manual Trigger

To manually trigger a build in CI/CD:
- **GitHub**: Push changes or create a PR
- **GitLab**: Push changes or create a merge request

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
