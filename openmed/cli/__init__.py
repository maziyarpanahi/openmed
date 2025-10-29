"""Command-line entry point wiring for the OpenMed toolkit."""

from . import main as main_module

# Some test cases patch these attributes on ``openmed.cli.main_module`` directly.
# Ensure they always exist even if the implementation defers importing heavy
# dependencies until runtime.
for _attr in ("analyze_text", "list_models", "get_model_max_length"):
    if not hasattr(main_module, _attr):
        setattr(main_module, _attr, None)


def main(argv=None):
    """Proxy to :func:`openmed.cli.main.main` for convenience."""
    return main_module.main(argv)


__all__ = ["main", "main_module"]
