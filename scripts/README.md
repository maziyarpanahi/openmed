# Release Scripts

This directory contains automated scripts for publishing the `openmed` package to PyPI.

## Files

- `release/release.py` - Main Python script for version bumping and automated releases
- `release/release.sh` - Simple shell script alternative for releases

## Usage

See the main documentation at `docs/RELEASE.md` for detailed usage instructions.

## Quick Commands

```bash
# From project root
make patch    # Bump patch version and release
make minor    # Bump minor version and release
make major    # Bump major version and release
```
