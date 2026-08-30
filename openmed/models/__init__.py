"""Model lifecycle helpers."""

from importlib import import_module
from typing import Any

__all__ = [
    "BootstrapReport",
    "DiagnosticCategory",
    "bootstrap_check",
    "check_bootstrap",
    "format_human",
    "render_json",
    "run_bootstrap_check",
]


def __getattr__(name: str) -> Any:
    """Load bootstrap helpers lazily, including for ``python -m`` execution."""

    exports = {
        "BootstrapReport",
        "DiagnosticCategory",
        "check_bootstrap",
        "format_human",
        "render_json",
        "run_bootstrap_check",
    }
    if name == "bootstrap_check" or name in exports:
        module = import_module(".bootstrap_check", __name__)
        if name == "bootstrap_check":
            return module
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
