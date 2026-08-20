"""Local-first helpers for privacy-safe agent traces."""

from .discovery import (
    SUPPORTED_TRACE_ROOT_RULES,
    TRACE_DISCOVERY_ENV_VAR,
    TRACE_ROOTS_ENV_VAR,
    TraceRootRule,
    TraceStore,
    TraceStoreSummary,
    discover_trace_stores,
)
from .tool_calls import (
    DEFAULT_CONTENT_PATHS,
    ContentPath,
    TextRedactor,
    ToolCallRedactionError,
    ToolCallRedactionReport,
    ToolCallRedactionResult,
    redact_tool_call,
    redact_tool_call_with_report,
    redact_tool_calls,
)

__all__ = [
    "ContentPath",
    "DEFAULT_CONTENT_PATHS",
    "SUPPORTED_TRACE_ROOT_RULES",
    "TRACE_DISCOVERY_ENV_VAR",
    "TRACE_ROOTS_ENV_VAR",
    "TextRedactor",
    "ToolCallRedactionError",
    "ToolCallRedactionReport",
    "ToolCallRedactionResult",
    "TraceRootRule",
    "TraceStore",
    "TraceStoreSummary",
    "discover_trace_stores",
    "redact_tool_call",
    "redact_tool_call_with_report",
    "redact_tool_calls",
]
