# Release Process

This document outlines the automated release process for publishing the `openmed` package to PyPI.

## Quick Start

### Option 1: Using Make (Recommended)

```bash
# Patch release (0.1.1 → 0.1.2)
make patch

# Minor release (0.1.1 → 0.2.0)
make minor

# Major release (0.1.1 → 1.0.0)
make major

# Just build and publish current version
make release
```

### Option 2: Using Python Script

```bash
# Patch release
python scripts/release/release.py patch

# Minor release
python scripts/release/release.py minor

# Major release
python scripts/release/release.py major
```

### Option 3: Using Shell Script

```bash
# Patch release
./scripts/release/release.sh patch

# Minor release
./scripts/release/release.sh minor

# Major release
./scripts/release/release.sh major
```

## Available Commands

### Make Commands

- `make help` - Show all available commands
- `make build` - Build the package only
- `make publish` - Publish to PyPI only
- `make release` - Full release cycle (clean + build + publish)
- `make clean` - Clean build artifacts
- `make install` - Install locally for testing
- `make test-build` - Test build without publishing

### Version Bumping

- `make patch` - Bump patch version and release (0.1.1 → 0.1.2)
- `make minor` - Bump minor version and release (0.1.1 → 0.2.0)
- `make major` - Bump major version and release (0.1.1 → 1.0.0)

## Manual Process

If you prefer to do things manually:

1. **Update version numbers** in:
   - `pyproject.toml`
   - `openmed/__init__.py`

2. **Build the package**:

   ```bash
   python -m build
   ```

3. **Publish to PyPI**:

   ```bash
   hatch publish
   # or
   python -m twine upload dist/*
   ```

## Credentials Setup

The scripts use your existing PyPI credentials from:

- `.pypirc` file (recommended for Hatch)
- `creds.txt` file (fallback)

Make sure your `.pypirc` contains:

```ini
[pypi]
username = __token__
password = pypi-AgEIcHl...
```

## Git Integration

The scripts automatically:

- Commit changes with message "Release: bump {type} version"
- Create git tags (e.g., `v0.1.2`)
- Push to remote (if configured)

## Troubleshooting

### Build Issues

```bash
# Clean and rebuild
make clean
make build
```

### Publish Issues

```bash
# Check your credentials
cat ~/.pypirc

# Test Hatch publishing
hatch publish --dry-run
```

### Version Conflicts

If you get "File already exists" error:

1. Update the version number manually in `pyproject.toml`
2. Run the release process again

## Advanced Usage

### Custom Version

To set a specific version instead of auto-bumping:

```bash
# Edit pyproject.toml manually
version = "1.2.3"

# Then run
make release
```

### Pre-release Versions

For beta/rc versions:

```toml
version = "1.0.0-beta.1"
```

### Multiple Repositories

To publish to test PyPI first:

```bash
# Publish to test PyPI
hatch publish -r test

# Publish to main PyPI
hatch publish
```

## Future Enhancements

Consider setting up:

- GitHub Actions for automated releases on tag push
- Pre-commit hooks for version consistency
- Automated testing before release
- Changelog generation

## Files Overview

- `Makefile` - Main automation interface
- `scripts/release/release.py` - Python script for version bumping and release
- `scripts/release/release.sh` - Simple shell script alternative
- `docs/RELEASE.md` - This documentation
- `pyproject.toml` - Package configuration with Hatch settings
- `.pypirc` - PyPI credentials (keep secure!)
- `.gitignore` - Excludes sensitive files from git
