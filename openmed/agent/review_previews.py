"""Strict offline decoding of the supported native OMOP review preview.

This adapter accepts metadata only. It never accepts row values, retrieves a
snapshot, binds an approval, commits a batch, or authenticates a preview.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from openmed.interop.omop.mutation_batch import (
    MAX_BATCH_MUTATIONS,
    MUTATION_BATCH_SCHEMA,
    MutationOperation,
    OmopMutationPreview,
    OmopMutationSummary,
    OmopReferenceIssue,
    _digest,
    _validate_digest,
    _validate_identifier,
)

from .approval_evidence import ApprovalEvidenceError

_MAX_BYTES = 1_048_576
_FIELDS = frozenset(
    {
        "schema",
        "batch_digest",
        "preview_digest",
        "reference_snapshot_digest",
        "mutation_count",
        "is_valid",
        "mutations",
        "issues",
        "operation_counts",
    }
)
_MUTATION_FIELDS = frozenset(
    {"ordinal", "operation", "table", "field_names", "reference_count", "row_digest"}
)
_ISSUE_FIELDS = frozenset({"ordinal", "code", "table", "field_name"})
_ISSUE_CODES = frozenset(
    {"duplicate_insert", "missing_target", "referenced_tombstone", "missing_reference"}
)


def parse_omop_review_preview(payload: str | bytes | bytearray) -> OmopMutationPreview:
    """Decode exact native preview fields and recompute their binding digest.

    Args:
        payload: Bounded JSON produced by OmopMutationPreview.to_json().

    Returns:
        Native value-free preview metadata; a digest match grants no authority.

    Raises:
        ApprovalEvidenceError: On malformed, unsupported or changed metadata.
    """
    if not isinstance(payload, (str, bytes, bytearray)):
        raise ApprovalEvidenceError("invalid_json")
    try:
        raw = payload.encode("utf-8") if isinstance(payload, str) else payload
        if len(raw) > _MAX_BYTES:
            raise ApprovalEvidenceError("json_too_large")
        fields = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    except ApprovalEvidenceError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ApprovalEvidenceError("invalid_json") from None
    _exact(fields, _FIELDS)
    if fields["schema"] != MUTATION_BATCH_SCHEMA:
        raise ApprovalEvidenceError("unsupported_version")
    try:
        for key in ("batch_digest", "preview_digest", "reference_snapshot_digest"):
            _validate_digest(fields[key], key)
        count = _count(fields["mutation_count"], minimum=1, maximum=MAX_BATCH_MUTATIONS)
        if type(fields["is_valid"]) is not bool:
            raise ApprovalEvidenceError("invalid_metadata")
        if type(fields["mutations"]) is not list or len(fields["mutations"]) != count:
            raise ApprovalEvidenceError("invalid_metadata")
        mutations = []
        counts: Counter[str] = Counter()
        for index, item in enumerate(fields["mutations"]):
            _exact(item, _MUTATION_FIELDS)
            if _count(item["ordinal"], maximum=MAX_BATCH_MUTATIONS - 1) != index:
                raise ApprovalEvidenceError("invalid_metadata")
            operation = MutationOperation(item["operation"])
            counts[operation.value] += 1
            table = _validate_identifier(item["table"], "table")
            names = item["field_names"]
            if type(names) is not list or len(names) > 200_000:
                raise ApprovalEvidenceError("invalid_metadata")
            for name in names:
                _validate_identifier(name, "field_names")
            if names != sorted(set(names)):
                raise ApprovalEvidenceError("invalid_metadata")
            references = _count(item["reference_count"])
            digest = _validate_digest(item["row_digest"], "row_digest")
            mutations.append(
                OmopMutationSummary(
                    index, operation, table, tuple(names), references, digest
                )
            )
        _exact(fields["operation_counts"], frozenset(counts))
        for name, value in fields["operation_counts"].items():
            if _count(value, minimum=1, maximum=MAX_BATCH_MUTATIONS) != counts[name]:
                raise ApprovalEvidenceError("invalid_metadata")
        if type(fields["issues"]) is not list or len(fields["issues"]) > 200_000:
            raise ApprovalEvidenceError("invalid_metadata")
        if fields["is_valid"] != (not fields["issues"]):
            raise ApprovalEvidenceError("invalid_metadata")
        issues = []
        for item in fields["issues"]:
            _exact(item, _ISSUE_FIELDS)
            ordinal = _count(item["ordinal"], maximum=count - 1)
            if (
                item["table"] != mutations[ordinal].table
                or item["code"] not in _ISSUE_CODES
            ):
                raise ApprovalEvidenceError("invalid_metadata")
            if item["field_name"] is not None:
                _validate_identifier(item["field_name"], "field_name")
            issues.append(
                OmopReferenceIssue(
                    ordinal, item["code"], item["table"], item["field_name"]
                )
            )
    except ApprovalEvidenceError:
        raise
    except (ValueError, TypeError, KeyError):
        raise ApprovalEvidenceError("invalid_metadata") from None
    unsigned = {key: value for key, value in fields.items() if key != "preview_digest"}
    if _digest(unsigned) != fields["preview_digest"]:
        raise ApprovalEvidenceError("digest_mismatch")
    if (
        _digest(
            {
                "schema": MUTATION_BATCH_SCHEMA,
                "row_digests": [item.row_digest for item in mutations],
            }
        )
        != fields["batch_digest"]
    ):
        raise ApprovalEvidenceError("digest_mismatch")
    return OmopMutationPreview(
        batch_digest=fields["batch_digest"],
        preview_digest=fields["preview_digest"],
        reference_snapshot_digest=fields["reference_snapshot_digest"],
        mutations=tuple(mutations),
        issues=tuple(issues),
        operation_counts=tuple(sorted(counts.items())),
    )


def _exact(values: Any, expected: frozenset[str]) -> None:
    if type(values) is not dict or set(values) != expected:
        raise ApprovalEvidenceError("invalid_fields")


def _count(value: Any, *, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ApprovalEvidenceError("invalid_metadata")
    return value


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in pairs:
        if key in values:
            raise ApprovalEvidenceError("duplicate_field")
        values[key] = value
    return values


def _constant(value: str) -> None:
    del value
    raise ApprovalEvidenceError("invalid_json")


__all__ = ["parse_omop_review_preview"]
