"""Fail-closed guards for bounded evidence queries and read-only SQL."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Sequence
from typing import Final

MAX_QUERY_TEXT_BYTES: Final = 4096
MAX_SQL_BYTES: Final = 8192
MAX_QUERY_ROWS: Final = 100

DEFAULT_READ_ONLY_VIEWS: Final = frozenset(
    {
        "cohort_runs",
        "dataset_manifests",
        "journey_artifacts",
        "journey_current_facts",
        "journey_events",
        "journey_evidence",
        "journey_facts",
        "journey_mappings",
        "measure_results",
        "registry_cases",
        "trial_reviews",
    }
)

_MUTATION_KEYWORDS = frozenset(
    {
        "alter",
        "attach",
        "call",
        "copy",
        "create",
        "delete",
        "detach",
        "do",
        "drop",
        "execute",
        "grant",
        "insert",
        "merge",
        "prepare",
        "pragma",
        "refresh",
        "replace",
        "revoke",
        "truncate",
        "update",
        "upsert",
        "vacuum",
    }
)
_BANNED_FUNCTIONS = frozenset(
    {
        "dblink",
        "lo_export",
        "nextval",
        "pg_ls_dir",
        "pg_read_file",
        "pg_sleep",
        "set_config",
        "setval",
    }
)
_ADVICE_PATTERNS = (
    re.compile(r"\bdiagnos(?:e|is|ing)\b"),
    re.compile(r"\bprescrib(?:e|ing)\b"),
    re.compile(r"\brecommend\s+(?:a\s+)?(?:dose|medication|treatment)\b"),
    re.compile(r"\bwhat\s+(?:drug|medication|treatment)\s+should\b"),
    re.compile(r"\bshould\s+(?:i|the patient|they)\s+(?:take|stop|start)\b"),
)
_STATE_CHANGE_PATTERNS = (
    re.compile(r"\b(?:delete|modify|overwrite)\s+(?:the\s+)?(?:record|fact|case)\b"),
    re.compile(r"\b(?:enroll|contact|message)\s+(?:the\s+)?patient\b"),
    re.compile(r"\b(?:place|send)\s+(?:an?\s+)?(?:order|referral)\b"),
)


class QuerySafetyError(ValueError):
    """Raised with a stable code when a query violates a safety boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def classify_query_text(query_text: str) -> str | None:
    """Return a refusal code for obvious advice or state-changing requests."""

    normalized = normalize_query_text(query_text)
    if any(pattern.search(normalized) for pattern in _ADVICE_PATTERNS):
        return "unsupported_clinical_advice"
    if any(pattern.search(normalized) for pattern in _STATE_CHANGE_PATTERNS):
        return "state_change_requested"
    return None


def normalize_query_text(query_text: str) -> str:
    """Validate and normalize transient query text without echoing failures."""

    if not isinstance(query_text, str) or not query_text.strip():
        raise QuerySafetyError("query_text_required")
    if len(query_text.encode("utf-8")) > MAX_QUERY_TEXT_BYTES:
        raise QuerySafetyError("query_text_too_large")
    if "\x00" in query_text:
        raise QuerySafetyError("query_text_invalid")
    return " ".join(unicodedata.normalize("NFKC", query_text).casefold().split())


def validate_bounded_read_only_sql(
    sql: str,
    *,
    max_rows: int,
    allowed_views: Sequence[str] = tuple(sorted(DEFAULT_READ_ONLY_VIEWS)),
) -> str:
    """Return normalized SQL after proving it is one bounded read-only query."""

    if not isinstance(sql, str) or not sql.strip():
        raise QuerySafetyError("sql_required")
    if len(sql.encode("utf-8")) > MAX_SQL_BYTES or "\x00" in sql:
        raise QuerySafetyError("sql_invalid")
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_QUERY_ROWS:
        raise QuerySafetyError("row_limit_invalid")
    allowed = frozenset(_normalize_view_name(value) for value in allowed_views)
    if not allowed:
        raise QuerySafetyError("sql_view_allowlist_empty")

    normalized = unicodedata.normalize("NFKC", sql).strip()
    structural = _strip_literals_and_comments(normalized).casefold().strip()
    if structural.endswith(";"):
        structural = structural[:-1].rstrip()
    if ";" in structural:
        raise QuerySafetyError("sql_multiple_statements")
    if not re.match(r"^(?:select|with)\b", structural):
        raise QuerySafetyError("sql_not_read_only")
    tokens = set(re.findall(r"[a-z_]+", structural))
    if tokens.intersection(_MUTATION_KEYWORDS):
        raise QuerySafetyError("sql_mutation_rejected")
    if re.search(r"\bfor\s+(?:no\s+key\s+)?update\b", structural):
        raise QuerySafetyError("sql_locking_rejected")
    if re.search(r"\bselect\s+.+?\binto\b", structural, flags=re.DOTALL):
        raise QuerySafetyError("sql_mutation_rejected")
    if "*" in structural:
        raise QuerySafetyError("sql_wildcard_rejected")
    functions = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", structural))
    if functions.intersection(_BANNED_FUNCTIONS):
        raise QuerySafetyError("sql_function_rejected")

    raw_references = tuple(
        match.group(1)
        for match in re.finditer(r"\b(?:from|join)\s+([a-z_][a-z0-9_.]*)\b", structural)
    )
    referenced: list[str] = []
    for reference in raw_references:
        parts = reference.split(".")
        if len(parts) == 1:
            referenced.append(parts[0])
        elif len(parts) == 2 and parts[0] == "openmed":
            referenced.append(parts[1])
        else:
            raise QuerySafetyError("sql_view_not_allowed")
    cte_names = frozenset(
        match.group(1)
        for match in re.finditer(
            r"(?:\bwith\b|,)\s*([a-z_][a-z0-9_]*)\s+as\s*\(", structural
        )
    )
    if (
        not referenced
        or not any(view in allowed for view in referenced)
        or any(view not in allowed and view not in cte_names for view in referenced)
    ):
        raise QuerySafetyError("sql_view_not_allowed")

    limit = re.search(r"\blimit\s+([0-9]+)(?:\s+offset\s+([0-9]+))?\s*$", structural)
    if limit is None:
        raise QuerySafetyError("sql_unbounded")
    row_limit = int(limit.group(1))
    if not 1 <= row_limit <= max_rows:
        raise QuerySafetyError("sql_row_limit_exceeded")
    if limit.group(2) is not None and int(limit.group(2)) > max_rows:
        raise QuerySafetyError("sql_offset_exceeded")
    return normalized


def quote_untrusted_scalar(value: str | int | float | bool | None) -> str:
    """Render one tool value as inert, single-line JSON data."""

    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_QUERY_TEXT_BYTES or "\x00" in value:
            raise QuerySafetyError("tool_value_invalid")
    elif isinstance(value, float) and not math.isfinite(value):
        raise QuerySafetyError("tool_value_invalid")
    elif value is not None and not isinstance(value, (int, float, bool)):
        raise QuerySafetyError("tool_value_invalid")
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    replacements = {
        "&": "\\u0026",
        "<": "\\u003c",
        ">": "\\u003e",
        "[": "\\u005b",
        "]": "\\u005d",
        "`": "\\u0060",
        "\u2028": "\\u2028",
        "\u2029": "\\u2029",
    }
    for source, replacement in replacements.items():
        rendered = rendered.replace(source, replacement)
    return rendered


def _normalize_view_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) is None
    ):
        raise QuerySafetyError("sql_view_allowlist_invalid")
    return value


def _strip_literals_and_comments(sql: str) -> str:
    """Replace string literals and comments while retaining SQL structure."""

    output: list[str] = []
    index = 0
    length = len(sql)
    state = "code"
    while index < length:
        char = sql[index]
        nxt = sql[index + 1] if index + 1 < length else ""
        if state == "code":
            if char == "'":
                state = "single"
                output.append(" ")
            elif char == '"':
                state = "double"
                output.append(" ")
            elif char == "-" and nxt == "-":
                state = "line_comment"
                output.extend((" ", " "))
                index += 1
            elif char == "/" and nxt == "*":
                state = "block_comment"
                output.extend((" ", " "))
                index += 1
            else:
                output.append(char)
        elif state == "single":
            output.append(" ")
            if char == "'" and nxt == "'":
                output.append(" ")
                index += 1
            elif char == "'":
                state = "code"
        elif state == "double":
            output.append(" ")
            if char == '"' and nxt == '"':
                output.append(" ")
                index += 1
            elif char == '"':
                state = "code"
        elif state == "line_comment":
            output.append("\n" if char == "\n" else " ")
            if char == "\n":
                state = "code"
        else:
            output.append(" ")
            if char == "*" and nxt == "/":
                output.append(" ")
                index += 1
                state = "code"
        index += 1
    if state in {"single", "double", "block_comment"}:
        raise QuerySafetyError("sql_unterminated_input")
    return "".join(output)


__all__ = [
    "DEFAULT_READ_ONLY_VIEWS",
    "MAX_QUERY_ROWS",
    "MAX_QUERY_TEXT_BYTES",
    "MAX_SQL_BYTES",
    "QuerySafetyError",
    "classify_query_text",
    "normalize_query_text",
    "quote_untrusted_scalar",
    "validate_bounded_read_only_sql",
]
