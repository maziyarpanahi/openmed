# Release Process

This document outlines the release process for publishing the `openmed` package to PyPI.

## Source of truth

OpenMed versioning is dynamic and stored in:

- `openmed/__about__.py` (`__version__ = "X.Y.Z"`)

Do not edit `pyproject.toml` for version bumps.

## Quick start

### Option 1: Make targets (recommended)

```bash
# Bump only
make bump-patch
make bump-minor
make bump-major

# Build only
make build

# Full local publish (manual)
make release
```

### Option 2: Bump script directly

```bash
python3 scripts/release/release.py patch
python3 scripts/release/release.py minor
python3 scripts/release/release.py major
```

The bump script updates only `openmed/__about__.py`.

## Tag-driven CI publish (recommended)

1. Update changelog and commit.
2. Push a release tag:

```bash
git tag v0.6.0
git push origin v0.6.0
```

3. `.github/workflows/publish.yml` builds and publishes to PyPI.

## Manual local publish

```bash
python3 -m build
hatch publish
```

## Notes

- Keep `CHANGELOG.md` aligned with released tags.
- Use `uv run mkdocs build --strict` before tagging to keep docs links healthy.
- Prefer CI publishing over local publish for traceability.
