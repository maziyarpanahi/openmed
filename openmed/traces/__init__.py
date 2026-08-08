"""Privacy-safe discovery helpers for local agent trace stores."""

from .discovery import (
    SUPPORTED_TRACE_ROOT_RULES,
    TRACE_DISCOVERY_ENV_VAR,
    TRACE_ROOTS_ENV_VAR,
    TraceRootRule,
    TraceStore,
    TraceStoreSummary,
    discover_trace_stores,
)

__all__ = [
    "TRACE_DISCOVERY_ENV_VAR",
    "TRACE_ROOTS_ENV_VAR",
    "SUPPORTED_TRACE_ROOT_RULES",
    "TraceRootRule",
    "TraceStore",
    "TraceStoreSummary",
    "discover_trace_stores",
]
