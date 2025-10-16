"""Command-line entry points for the OpenMed toolkit."""

from . import main as _main_module


def main(argv=None):
    """Proxy to :func:`openmed.cli.main.main` for convenience."""
    return _main_module.main(argv)


# Expose attributes used by tests/consumers for easy patching
main.analyze_text = _main_module.analyze_text  # type: ignore[attr-defined]
main.list_models = _main_module.list_models  # type: ignore[attr-defined]

__all__ = ["main"]
