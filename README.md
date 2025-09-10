# openmed

A placeholder package to reserve the name on PyPI. Real library coming soon.

## Installation

```bash
pip install openmed
```

## Usage

```python
import openmed
print(openmed.__version__)
```

## Development

### Project Structure

- `scripts/` - Release automation scripts
- `docs/` - Documentation and release guides
- `.github/workflows/` - CI/CD workflows

### Releasing

See [docs/RELEASE.md](docs/RELEASE.md) for the complete release process.

Quick release commands:

```bash
make patch    # Patch release (0.1.1 → 0.1.2)
make minor    # Minor release (0.1.1 → 0.2.0)
make major    # Major release (0.1.1 → 1.0.0)
```
