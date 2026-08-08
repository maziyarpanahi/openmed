"""Local privacy guards for host integrations."""

from importlib import import_module
from typing import Any

__all__ = ["SessionScrubResult", "SessionTraceError", "scrub_trace"]


def __getattr__(name: str) -> Any:
    """Load session-hook exports without importing the executable eagerly."""
    if name not in __all__:
        raise AttributeError(name)
    module = import_module(".session_hook", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
