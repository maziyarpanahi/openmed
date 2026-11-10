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
2. Run the release preflight for the exact tag you plan to publish:

```bash
VERSION=$(python3 -c "from openmed import __version__; print(__version__)")
python3 scripts/release/check_release_version.py --version "$VERSION"
```

This confirms that `openmed.__version__`, the top changelog entry, public docs, and Swift app versions all agree, and
that the release tag is not already used locally or on `origin`.

3. Push the matching release tag:

```bash
git tag "v$VERSION"
git push origin "v$VERSION"
```

4. `.github/workflows/publish.yml` builds and publishes to PyPI.

## Manual local publish

```bash
python3 -m build
hatch publish
```

## Notes

- Keep `CHANGELOG.md` aligned with released tags.
- Use `uv run mkdocs build --strict` before tagging to keep docs links healthy.
- Prefer CI publishing over local publish for traceability.
