"""Offline, allowlisted normalization of caller-supplied DQD result files.

Only the OHDSI DataQualityDashboard 2.9.0 JSON export is supported. No DQD
code, database connection, R runtime, or terminology assets are required.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openmed.clinical.journey_contracts import canonical_json

from .fact_projection import OMOP_FACT_TABLES
from .quality import (
    OMOP_QUALITY_CATEGORIES,
    OMOP_QUALITY_COMPATIBILITY_POLICY,
    OMOP_QUALITY_REPORT_SCHEMA_VERSION,
    OMOP_QUALITY_REQUEST_ARTIFACT,
    OmopQualityCheck,
    OmopQualityConflictError,
    OmopQualityError,
    OmopQualityInput,
    OmopQualityProtocolError,
    OmopQualityUnsupportedError,
    build_omop_quality_tool_output,
)

DQD_SUPPORTED_VERSIONS = frozenset({"2.9.0"})
_MAX_RESULTS_BYTES = 64 * 1024 * 1024
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_OUTPUT_BYTES = 1_000_000
_MAX_EXACT_COUNT = 2**53 - 1


def normalize_dqd_results_file(
    path: str | os.PathLike[str], *, quality_input: OmopQualityInput
) -> dict[str, Any]:
    """Read a local DQD export and build digest-bound aggregate tool output.

    Args:
        path: Caller-owned DQD 2.9.0 JSON results file, at most 64 MiB.
        quality_input: Caller-supplied custody manifest for the evaluated projection.

    Returns:
        Local tool output accepted by ``normalize_omop_quality_output``. Only
        categories, statuses, counts, allowlisted tables and generated codes survive.

    Raises:
        OmopQualityUnsupportedError: The version, category or table is unsupported.
        OmopQualityProtocolError: The file is unreadable, malformed or exceeds bounds.
    """

    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(_MAX_RESULTS_BYTES + 1)
    except (OSError, ValueError, TypeError):
        raise OmopQualityProtocolError("dqd_results_unreadable") from None
    payload = _decode_json(raw, _MAX_RESULTS_BYTES)
    if not isinstance(payload, Mapping):
        raise OmopQualityProtocolError("dqd_results_invalid")
    metadata = payload.get("Metadata")
    if (
        not isinstance(metadata, list)
        or len(metadata) != 1
        or not isinstance(metadata[0], Mapping)
    ):
        raise OmopQualityUnsupportedError("dqd_version_unsupported")
    version = metadata[0].get("dqdVersion")
    if not isinstance(version, str) or version not in DQD_SUPPORTED_VERSIONS:
        raise OmopQualityUnsupportedError("dqd_version_unsupported")
    rows = payload.get("CheckResults")
    if not isinstance(rows, list):
        raise OmopQualityProtocolError("dqd_checks_invalid")
    # IDs are source offsets, never caller-controlled names or free text. Row order
    # is part of custody; editing ignored fields cannot change the output digest.
    checks = tuple(_normalize_check(row, index) for index, row in enumerate(rows))
    output = build_omop_quality_tool_output(
        quality_input,
        checks,
        tool_name="ohdsi.dqd_results",
        tool_version=version,
        execution_mode="local",
    )
    if len(canonical_json(output).encode("utf-8")) + 1 > _MAX_OUTPUT_BYTES:
        raise OmopQualityProtocolError("dqd_output_too_large")
    return output


def _normalize_check(row: Any, index: int) -> OmopQualityCheck:
    if not isinstance(row, Mapping):
        raise OmopQualityProtocolError("dqd_check_invalid")
    category = row.get("category")
    if not isinstance(category, str) or category.lower() not in OMOP_QUALITY_CATEGORIES:
        raise OmopQualityUnsupportedError("dqd_category_unsupported")
    table = row.get("cdmTableName")
    if not isinstance(table, str) or table.lower() not in OMOP_FACT_TABLES:
        raise OmopQualityUnsupportedError("dqd_table_unsupported")
    required = {"failed", "passed", "isError", "notApplicable", "numViolatedRows"}
    if not required <= row.keys():
        raise OmopQualityProtocolError("dqd_check_fields_missing")
    flags = tuple(
        row[name] for name in ("failed", "passed", "isError", "notApplicable")
    )
    if any(
        flag is not None and (type(flag) not in (int, bool) or flag not in (0, 1))
        for flag in flags
    ):
        raise OmopQualityProtocolError("dqd_flags_invalid")
    count = row["numViolatedRows"]
    if count is not None and (
        type(count) not in (int, float)
        or count < 0
        or count > _MAX_EXACT_COUNT
        or not math.isfinite(count)
        or int(count) != count
    ):
        raise OmopQualityProtocolError("dqd_count_invalid")
    affected_rows = 0 if count is None else int(count)
    failed, passed, errored, not_applicable = flags
    status = "unknown"
    if errored == 1:
        code = "dqd_check_error"
    elif not_applicable == 1:
        code = "dqd_not_applicable"
    elif any(flag is None for flag in flags):
        code = "dqd_flags_unknown"
    elif failed == passed:
        code = "dqd_status_unknown"
    elif failed == 1:
        status, code = "fail", "dqd_check_failed"
    elif count is None:
        code = "dqd_count_unknown"
    elif affected_rows:
        # DQD threshold passes may contain violations; the bridge defines a pass
        # as zero affected rows. Preserve the count and abstain instead of coercing.
        code = "dqd_threshold_pass_requires_review"
    else:
        status, code = "pass", "dqd_check_passed"
    return OmopQualityCheck(
        check_id=f"dqd.check.{index}",
        category=category.lower(),
        status=status,
        severity="info" if status == "pass" else "error",
        code=code,
        remediation_code="dqd_review_check",
        affected_rows=affected_rows,
        table=table.lower(),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OmopQualityProtocolError("dqd_json_duplicate_key")
        result[key] = value
    return result


def _invalid_constant(_value: str) -> Any:
    raise OmopQualityProtocolError("dqd_json_invalid")


def _decode_json(raw: bytes, limit: int) -> Any:
    if len(raw) > limit:
        raise OmopQualityProtocolError("dqd_json_too_large")
    try:
        return json.loads(
            raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
    except (ValueError, UnicodeError, RecursionError):
        raise OmopQualityProtocolError("dqd_json_invalid") from None


def _read_request() -> OmopQualityInput:
    request = _decode_json(
        sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1), _MAX_REQUEST_BYTES
    )
    if not isinstance(request, Mapping) or set(request) != {
        "artifact_type",
        "compatibility_policy",
        "input",
        "input_digest",
        "schema_version",
    }:
        raise OmopQualityProtocolError("dqd_request_invalid")
    if (
        request["artifact_type"] != OMOP_QUALITY_REQUEST_ARTIFACT
        or request["schema_version"] != OMOP_QUALITY_REPORT_SCHEMA_VERSION
        or request["compatibility_policy"] != OMOP_QUALITY_COMPATIBILITY_POLICY
    ):
        raise OmopQualityUnsupportedError("dqd_request_unsupported")
    quality_input = OmopQualityInput.from_dict(request["input"])
    if request["input_digest"] != quality_input.digest:
        raise OmopQualityConflictError("dqd_request_digest_conflict")
    return quality_input


def main() -> int:
    """Normalize one file using the bridge request on stdin and JSON on stdout.

    Returns:
        Zero on success, or two with a controlled, value-free diagnostic on stderr.
    """

    if sys.argv[1:] == ["--help"]:
        sys.stdout.write(
            "Usage: python -m openmed.interop.omop.dqd RESULTS.json\n"
            "Reads an OMOP quality request from stdin; writes aggregate JSON.\n"
        )
        return 0
    try:
        if len(sys.argv) != 2:
            raise OmopQualityProtocolError("dqd_arguments_invalid")
        output = normalize_dqd_results_file(sys.argv[1], quality_input=_read_request())
        sys.stdout.write(canonical_json(output) + "\n")
        return 0
    except OmopQualityUnsupportedError:
        state, code = "unsupported", "dqd_results_unsupported"
    except OmopQualityConflictError:
        state, code = "conflict", "dqd_request_conflict"
    except (OmopQualityError, OSError, TypeError, ValueError, RecursionError):
        state, code = "failure", "dqd_results_invalid"
    sys.stderr.write(canonical_json({"state": state, "code": code}) + "\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
