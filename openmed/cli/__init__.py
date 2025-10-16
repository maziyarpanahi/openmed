"""Command-line entry point wiring for the OpenMed toolkit."""

from . import main as main_module


def main(argv=None):
    """Proxy to :func:`openmed.cli.main.main` for convenience."""
    return main_module.main(argv)


__all__ = ["main", "main_module"]
