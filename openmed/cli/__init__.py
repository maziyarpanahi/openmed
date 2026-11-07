"""Command-line entry point wiring for the OpenMed toolkit."""

from . import main as main_module

COMPLIANCE_CAVEAT = main_module.COMPLIANCE_CAVEAT

# Some test cases patch these attributes on ``openmed.cli.main_module`` directly.
# Ensure they always exist even if the implementation defers importing heavy
# dependencies until runtime.
for _attr in ("analyze_text", "list_models", "get_model_max_length"):
    if not hasattr(main_module, _attr):
        setattr(main_module, _attr, None)


def main(argv=None, *, governance_service=None):
    """Proxy to :func:`openmed.cli.main.main` for convenience."""
    if governance_service is None:
        return main_module.main(argv)
    return main_module.main(argv, governance_service=governance_service)


__all__ = ["COMPLIANCE_CAVEAT", "main", "main_module"]
