"""Deterministic audit reports for de-identification runs."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

_HASH_ALGORITHM = "sha256"
_SIGNATURE_ALGORITHM = "HMAC-SHA256"
_MISSING_KEY_ERROR = (
    "A non-empty HMAC release key is required for audit signing and verification"
)
_AUDIT_RAW_VALUE_KEYS = frozenset(
    {
        "deidentified_text",
        "original",
        "original_text",
        "normalized_value",
        "path",
        "record_id",
        "raw",
        "replacement",
        "source_path",
        "surface",
        "text",
        "value",
        "word",
    }
)
_AUDIT_RAW_COLLECTION_KEYS = frozenset(
    {
        "equivalence_classes",
        "members",
        "per_record",
        "records",
        "suppressed_records",
    }
)
_AUDIT_DROPPED_DETAIL_KEYS = frozenset(
    {
        "normalized_value",
        "path",
        "quasi_identifier_key",
        "record_hash",
        "record_id",
        "record_index",
        "source_path",
    }
)
_AUDIT_ABHA_ADDRESS_RE = re.compile(
    r"(?<![\w.])[A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)*@[A-Z][A-Z0-9-]{1,31}(?![\w.-])",
    re.IGNORECASE,
)
_AUDIT_PAN_RE = re.compile(
    r"(?<![A-Z0-9])[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z](?![A-Z0-9])"
)
_AUDIT_LONG_INDIA_ID_RE = re.compile(r"(?<!\d)(?:\d[ -]?){11,13}\d(?!\d)")
_AUDIT_METADATA_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_AUDIT_RISK_CATEGORIES = frozenset(
    {
        "age",
        "date",
        "geography",
        "provider_institution",
        "rare_condition",
        "stable_surrogate",
    }
)


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(data: bytes) -> str:
    return f"{_HASH_ALGORITHM}:{hashlib.sha256(data).hexdigest()}"


def hash_text(text: str) -> str:
    """Return a stable SHA-256 hash for text content."""
    return _sha256_bytes(text.encode("utf-8"))


def _contains_sensitive_india_identifier(value: str) -> bool:
    return bool(
        _AUDIT_ABHA_ADDRESS_RE.search(value)
        or _AUDIT_PAN_RE.search(value.upper())
        or _AUDIT_LONG_INDIA_ID_RE.search(value)
    )


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _safe_count(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    return 0


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [
        item
        for item in value
        if isinstance(item, str) and _AUDIT_METADATA_IDENTIFIER_RE.fullmatch(item)
    ]


def _safe_size_distribution(value: Any) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    if not isinstance(value, (list, tuple)):
        return result
    for item in value:
        if isinstance(item, Mapping):
            size = _safe_number(item.get("size"))
            count = _safe_number(item.get("class_count"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            size = _safe_number(item[0])
            count = _safe_number(item[1])
        else:
            continue
        if isinstance(size, int) and isinstance(count, int):
            result.append({"size": size, "class_count": count})
    return result


def _safe_numeric_mapping(
    value: Any,
    *,
    allowed_keys: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in allowed_keys:
        item = value.get(key)
        if isinstance(item, bool):
            result[key] = item
            continue
        number = _safe_number(item)
        if number is not None:
            result[key] = number
    return result


def _safe_policy_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "target_k",
        "target_l",
        "target_t",
        "suppression_limit",
        "suppression_rate",
        "max_lattice_nodes",
        "max_suppression_subsets",
    ):
        number = _safe_number(value.get(key))
        if number is not None:
            result[key] = number
    l_metric = value.get("l_metric")
    if l_metric in {"distinct", "entropy"}:
        result["l_metric"] = l_metric
    t_distance = value.get("t_distance")
    if t_distance == "variational":
        result["t_distance"] = t_distance
    for key in (
        "quasi_identifiers",
        "sensitive_attributes",
        "direct_identifiers",
        "non_sensitive_attributes",
        "excluded_attributes",
    ):
        result[key] = _safe_string_list(value.get(key))
    privacy_unit = value.get("privacy_unit")
    if privacy_unit is None or (
        isinstance(privacy_unit, str)
        and _AUDIT_METADATA_IDENTIFIER_RE.fullmatch(privacy_unit)
    ):
        result["privacy_unit"] = privacy_unit
    return result


def _safe_attribute_disclosure(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        entry: dict[str, Any] = {}
        attribute = item.get("attribute")
        if isinstance(attribute, str) and _AUDIT_METADATA_IDENTIFIER_RE.fullmatch(
            attribute
        ):
            entry["attribute"] = attribute
        for source_key, allowed_keys in (
            (
                "l_diversity",
                ("achieved", "threshold", "violating_classes", "meets_target"),
            ),
            (
                "t_closeness",
                ("achieved", "target", "violating_classes", "meets_target"),
            ),
        ):
            summary = _safe_numeric_mapping(
                item.get(source_key),
                allowed_keys=allowed_keys,
            )
            source = item.get(source_key)
            if isinstance(source, Mapping):
                metric = source.get("metric")
                if metric in {"distinct", "entropy"}:
                    summary["metric"] = metric
                distance = source.get("distance")
                if distance == "variational":
                    summary["distance"] = distance
            entry[source_key] = summary
        result.append(entry)
    return result


def _sanitize_release_assessment(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only the aggregate release-assessment evidence schema."""

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "deidentification_release_assessment",
        "not_an_expert_determination": True,
        "qualified_expert_review_required": True,
    }
    for key in ("row_count", "privacy_unit_count"):
        number = _safe_number(value.get(key))
        if number is not None:
            result[key] = number
    privacy_unit = value.get("privacy_unit_column")
    if privacy_unit is None or (
        isinstance(privacy_unit, str)
        and _AUDIT_METADATA_IDENTIFIER_RE.fullmatch(privacy_unit)
    ):
        result["privacy_unit_column"] = privacy_unit
    result["quasi_identifiers"] = _safe_string_list(value.get("quasi_identifiers"))
    result["sensitive_attributes"] = _safe_string_list(
        value.get("sensitive_attributes")
    )
    result["policy"] = _safe_policy_mapping(value.get("policy"))

    k_anonymity = value.get("k_anonymity")
    safe_k = _safe_numeric_mapping(
        k_anonymity,
        allowed_keys=(
            "achieved_k",
            "class_count",
            "singleton_class_count",
            "singleton_privacy_unit_count",
            "k_violating_class_count",
            "l_violating_class_count",
            "t_violating_class_count",
            "violating_class_count",
            "violating_privacy_unit_count",
            "meets_target",
        ),
    )
    if isinstance(k_anonymity, Mapping):
        safe_k["class_size_distribution"] = _safe_size_distribution(
            k_anonymity.get("class_size_distribution")
        )
    result["k_anonymity"] = safe_k

    identity_risk = _safe_numeric_mapping(
        value.get("sample_identity_risk"),
        allowed_keys=(
            "max",
            "mean",
            "median",
            "p95",
            "population_risk_estimated",
        ),
    )
    identity_risk["attacker_model"] = "prosecutor_exact_match_on_declared_qis"
    result["sample_identity_risk"] = identity_risk
    result["attribute_disclosure"] = _safe_attribute_disclosure(
        value.get("attribute_disclosure")
    )
    if isinstance(value.get("meets_policy"), bool):
        result["meets_policy"] = value["meets_policy"]
    for key in ("dataset_digest", "policy_digest"):
        digest = value.get(key)
        if _is_sha256_hash(digest):
            result[key] = digest
    warnings = value.get("warnings")
    result["warning_count"] = _safe_count(warnings)
    return result


def _safe_generalization_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    levels: list[dict[str, Any]] = []
    raw_levels = value.get("levels")
    if isinstance(raw_levels, (list, tuple)):
        level_items = raw_levels
    elif isinstance(raw_levels, Mapping):
        level_items = [
            {"attribute": attribute, **dict(level)}
            for attribute, level in raw_levels.items()
            if isinstance(attribute, str) and isinstance(level, Mapping)
        ]
    else:
        level_items = []
    for item in level_items:
        if not isinstance(item, Mapping):
            continue
        level: dict[str, Any] = {}
        attribute = item.get("attribute")
        if isinstance(attribute, str) and _AUDIT_METADATA_IDENTIFIER_RE.fullmatch(
            attribute
        ):
            level["attribute"] = attribute
        for key in ("level", "loss", "affected_privacy_unit_count"):
            number = _safe_number(item.get(key))
            if number is not None:
                level[key] = number
        levels.append(level)

    result: dict[str, Any] = {"levels": levels}
    for key in (
        "information_loss",
        "generalization_loss",
        "suppression_loss",
        "suppressed_privacy_units",
        "suppressed_rows",
        "search_space_size",
        "nodes_evaluated",
        "max_lattice_nodes",
        "suppression_subsets_evaluated",
        "suppression_subsets_possible",
        "max_suppression_subsets",
    ):
        number = _safe_number(value.get(key))
        if number is not None:
            result[key] = number
    search = value.get("search")
    if isinstance(search, Mapping):
        safe_search: dict[str, Any] = _safe_numeric_mapping(
            search,
            allowed_keys=(
                "search_space_size",
                "nodes_evaluated",
                "max_lattice_nodes",
                "suppression_subsets_evaluated",
                "suppression_subsets_possible",
                "max_suppression_subsets",
                "complete",
                "optimum_proven",
            ),
        )
        strategy = search.get("strategy")
        if strategy in {"full-domain exhaustive lattice", "exhaustive_lattice"}:
            safe_search["strategy"] = strategy
        suppression_strategy = search.get("suppression_strategy")
        if suppression_strategy == "exhaustive equivalence-class subsets":
            safe_search["suppression_strategy"] = suppression_strategy
        result["search"] = safe_search
    elif isinstance(search, str):
        if search == "full-domain exhaustive lattice":
            result["search"] = search
    return result


def _sanitize_anonymization_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy aggregate anonymization evidence while never traversing records."""

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "deidentification_anonymization_summary",
        "not_an_expert_determination": True,
        "qualified_expert_review_required": True,
        "policy": _safe_policy_mapping(value.get("policy")),
        "generalization": _safe_generalization_summary(value.get("generalization")),
    }
    for key in ("before", "after"):
        assessment = value.get(key)
        if isinstance(assessment, Mapping):
            result[key] = _sanitize_release_assessment(assessment)
    utility = value.get("utility")
    if isinstance(utility, Mapping):
        result["utility"] = _safe_numeric_mapping(
            utility,
            allowed_keys=(
                "source_rows",
                "released_rows",
                "source_privacy_units",
                "released_privacy_units",
                "row_suppression_rate",
                "privacy_unit_suppression_rate",
                "quasi_identifier_cells_compared",
                "quasi_identifier_cells_changed",
                "quasi_identifier_cell_change_rate",
                "direct_identifier_cells_removed",
                "missing_qi_cells_before",
                "missing_qi_cells_after",
                "mean_qi_distribution_shift",
            ),
        )
    for key in (
        "source_dataset_digest",
        "released_dataset_digest",
        "released_schema_digest",
        "hierarchy_digest",
    ):
        digest = value.get(key)
        if _is_sha256_hash(digest):
            result[key] = digest
    return result


def _looks_like_detailed_risk_report(value: Mapping[str, Any]) -> bool:
    return (
        "singleton_records" in value
        and "quasi_identifiers" in value
        and any(key in value for key in ("leakage_rate", "reid_rate", "k_min"))
    )


def _sanitize_detailed_risk_report(value: Mapping[str, Any]) -> dict[str, Any]:
    singletons = value.get("singleton_records")
    quasi_identifiers = value.get("quasi_identifiers")
    if not singletons and not quasi_identifiers:
        # Preserve the long-standing safe empty shape for signed-report
        # compatibility while still dropping all unknown fields.
        return {
            "leakage_rate": _safe_number(value.get("leakage_rate")),
            "reid_rate": _safe_number(value.get("reid_rate")),
            "k_min": _safe_number(value.get("k_min")),
            "singleton_records": [],
            "quasi_identifiers": [],
        }

    category_counts: dict[str, int] = {}
    if isinstance(quasi_identifiers, (list, tuple)):
        for item in quasi_identifiers:
            if not isinstance(item, Mapping):
                continue
            category = item.get("category")
            name = (
                category
                if isinstance(category, str) and category in _AUDIT_RISK_CATEGORIES
                else "unknown"
            )
            category_counts[name] = category_counts.get(name, 0) + 1
    record_count = _safe_number(value.get("record_count"))
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "deidentification_risk_summary",
        "detail_level": "aggregate_phi_safe",
        "record_count": record_count if record_count is not None else 0,
        "singleton_record_count": _safe_count(singletons),
        "quasi_identifier_count": _safe_count(quasi_identifiers),
        "quasi_identifier_categories": dict(sorted(category_counts.items())),
    }
    for source, target in (
        ("leakage_rate", "leakage_rate"),
        ("reid_rate", "reidentification_rate"),
        ("k_min", "minimum_k"),
    ):
        number = _safe_number(value.get(source))
        if number is not None:
            result[target] = number
    return result


def _looks_like_kanon_report(value: Mapping[str, Any]) -> bool:
    return (
        "equivalence_classes" in value
        and "class_size_distribution" in value
        and any(key in value for key in ("k", "class_count", "record_count"))
    )


def _sanitize_kanon_report(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "deidentification_kanon_summary",
        "detail_level": "aggregate_phi_safe",
        "quasi_identifiers": _safe_string_list(value.get("quasi_identifiers")),
        "sensitive_attributes": _safe_string_list(value.get("sensitive_attributes")),
        "class_size_distribution": _safe_size_distribution(
            value.get("class_size_distribution")
        ),
    }
    for key in ("record_count", "k", "class_count"):
        number = _safe_number(value.get(key))
        if number is not None:
            result[key] = number
    for key in ("l_metric", "t_distance"):
        item = value.get(key)
        if isinstance(item, str):
            result[key] = item
    l_summary = value.get("l")
    if isinstance(l_summary, Mapping):
        result["l"] = {
            str(attribute): number
            for attribute, metric in l_summary.items()
            if isinstance(attribute, str)
            and _AUDIT_METADATA_IDENTIFIER_RE.fullmatch(attribute)
            and (number := _safe_number(metric)) is not None
        }
    l_diversity = value.get("l_diversity")
    if isinstance(l_diversity, Mapping):
        result["l_diversity"] = {
            str(attribute): _safe_numeric_mapping(
                metrics,
                allowed_keys=("min_distinct", "min_entropy"),
            )
            for attribute, metrics in l_diversity.items()
            if isinstance(attribute, str)
            and _AUDIT_METADATA_IDENTIFIER_RE.fullmatch(attribute)
            and isinstance(metrics, Mapping)
        }
    t_closeness = value.get("t_closeness")
    if isinstance(t_closeness, Mapping):
        result["t_closeness"] = {
            str(attribute): number
            for attribute, metric in t_closeness.items()
            if isinstance(attribute, str)
            and _AUDIT_METADATA_IDENTIFIER_RE.fullmatch(attribute)
            and (number := _safe_number(metric)) is not None
        }
    result["equivalence_class_detail_count"] = _safe_count(
        value.get("equivalence_classes")
    )
    return result


def _looks_like_kanon_enforcement(value: Mapping[str, Any]) -> bool:
    return isinstance(value.get("kanon"), Mapping) and any(
        key in value for key in ("target_k", "generalization", "records")
    )


def _sanitize_kanon_enforcement(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "deidentification_kanon_enforcement_summary",
        "detail_level": "aggregate_phi_safe",
        "quasi_identifiers": _safe_string_list(value.get("quasi_identifiers")),
        "sensitive_attributes": _safe_string_list(value.get("sensitive_attributes")),
    }
    for key in (
        "record_count",
        "released_count",
        "suppressed_count",
        "suppression_limit",
        "target_k",
        "target_l",
        "target_t",
    ):
        number = _safe_number(value.get(key))
        if number is not None:
            result[key] = number
    l_metric = value.get("l_metric")
    if isinstance(l_metric, str):
        result["l_metric"] = l_metric
    kanon = value.get("kanon")
    if isinstance(kanon, Mapping):
        result["kanon"] = _sanitize_kanon_report(kanon)
    result["generalization"] = _safe_generalization_summary(value.get("generalization"))

    bounds = value.get("bounds")
    if isinstance(bounds, Mapping):
        safe_bounds: dict[str, Any] = _safe_numeric_mapping(
            bounds,
            allowed_keys=(
                "target_reidentification_upper_bound",
                "max_reidentification_upper_bound",
                "target_k",
                "target_l",
                "target_t",
                "suppressed_count",
            ),
        )
        self_check = bounds.get("numeric_self_check")
        if isinstance(self_check, Mapping):
            safe_check = _safe_numeric_mapping(
                self_check,
                allowed_keys=(
                    "passed",
                    "l_diversity_satisfied",
                    "t_closeness_satisfied",
                ),
            )
            safe_check["identity_bound_violation_count"] = _safe_count(
                self_check.get("identity_bound_violations")
            )
            safe_bounds["numeric_self_check"] = safe_check
        result["bounds"] = safe_bounds
    return result


def _safe_release_object(value: Any) -> dict[str, Any] | None:
    value_type = type(value)
    if value_type.__module__ != "openmed.risk.release":
        return None
    if value_type.__name__ == "ReleaseAssessment":
        payload = value.to_dict()
        return (
            _sanitize_release_assessment(payload)
            if isinstance(payload, Mapping)
            else None
        )
    if value_type.__name__ == "AnonymizationResult":
        payload = value.to_safe_dict()
        return (
            _sanitize_anonymization_summary(payload)
            if isinstance(payload, Mapping)
            else None
        )
    return None


def _sanitize_audit_value(value: Any) -> Any:
    """Return a deterministic audit-safe value with risk details aggregated."""

    safe_release = _safe_release_object(value)
    if safe_release is not None:
        return safe_release

    if isinstance(value, Mapping):
        artifact = value.get("artifact")
        if artifact == "deidentification_release_assessment":
            return _sanitize_release_assessment(value)
        if artifact == "deidentification_anonymization_summary":
            return _sanitize_anonymization_summary(value)
        if _looks_like_kanon_enforcement(value):
            return _sanitize_kanon_enforcement(value)
        if _looks_like_detailed_risk_report(value):
            return _sanitize_detailed_risk_report(value)
        if _looks_like_kanon_report(value):
            return _sanitize_kanon_report(value)

        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            normalized_name = name.strip().casefold()
            if normalized_name in _AUDIT_DROPPED_DETAIL_KEYS:
                continue
            if normalized_name in _AUDIT_RAW_VALUE_KEYS:
                if isinstance(item, str):
                    serialized = item
                else:
                    try:
                        serialized = _canonical_json(_sanitize_audit_value(item))
                    except (TypeError, ValueError):
                        serialized = str(item)
                sanitized[f"{name}_hash"] = hash_text(serialized)
                continue
            if normalized_name in _AUDIT_RAW_COLLECTION_KEYS:
                count_name = {
                    "equivalence_classes": "equivalence_class_count",
                    "members": "member_count",
                    "per_record": "per_record_count",
                    "records": "record_count",
                    "suppressed_records": "suppressed_record_count",
                }[normalized_name]
                sanitized[count_name] = _safe_count(item)
                continue
            sanitized[name] = _sanitize_audit_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_audit_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_audit_value(item) for item in value]
    if isinstance(value, str) and _contains_sensitive_india_identifier(value):
        return f"redacted:{hash_text(value)}"
    return copy.deepcopy(value)


def _is_sha256_hash(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(f"{_HASH_ALGORITHM}:"):
        return False
    digest = value.removeprefix(f"{_HASH_ALGORITHM}:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _raw_context_descriptor(
    value: str,
    *,
    side: str,
    span_start: int,
    span_end: int,
) -> dict[str, int | str]:
    """Convert a legacy plaintext context value into a PHI-safe descriptor."""
    if side == "before":
        segment_end = max(0, span_start)
        segment_length = min(len(value), segment_end)
        safe_value = value[-segment_length:] if segment_length else ""
        segment_start = segment_end - segment_length
    else:
        segment_start = max(0, span_end)
        safe_value = value
        segment_length = len(safe_value)
        segment_end = segment_start + segment_length
    return {
        "start": segment_start,
        "end": segment_end,
        "length": segment_length,
        "text_hash": hash_text(safe_value),
    }


def _safe_context_descriptor(
    value: Any,
    *,
    side: str,
    span_start: int,
    span_end: int,
    document_length: int | None = None,
) -> dict[str, int | str] | None:
    start: Any
    end: Any
    length: Any
    text_hash: Any
    if isinstance(value, str):
        descriptor = _raw_context_descriptor(
            value,
            side=side,
            span_start=span_start,
            span_end=span_end,
        )
        start = descriptor["start"]
        end = descriptor["end"]
        length = descriptor["length"]
        text_hash = descriptor["text_hash"]
    else:
        if not isinstance(value, Mapping):
            return None
        start = value.get("start")
        end = value.get("end")
        length = value.get("length")
        text_hash = value.get("text_hash")

    if not all(type(item) is int and item >= 0 for item in (start, end, length)):
        return None
    if end < start or length != end - start or not _is_sha256_hash(text_hash):
        return None
    if side == "before" and end != span_start:
        return None
    if side == "after" and start != span_end:
        return None
    if document_length is not None and end > document_length:
        return None
    return {
        "start": cast(int, start),
        "end": cast(int, end),
        "length": cast(int, length),
        "text_hash": cast(str, text_hash),
    }


def _sanitize_audit_context(
    context: Any,
    *,
    span_start: int,
    span_end: int,
    document_length: int | None = None,
) -> dict[str, dict[str, int | str]]:
    """Return only validated offsets, lengths, and hashes for audit context."""
    if not all(type(item) is int and item >= 0 for item in (span_start, span_end)):
        return {}
    if span_end < span_start:
        return {}
    if document_length is not None:
        if type(document_length) is not int or document_length < span_end:
            return {}
    if not isinstance(context, Mapping):
        return {}
    safe_context: dict[str, dict[str, int | str]] = {}
    for side in ("before", "after"):
        descriptor = _safe_context_descriptor(
            context.get(side),
            side=side,
            span_start=span_start,
            span_end=span_end,
            document_length=document_length,
        )
        if descriptor is not None:
            safe_context[side] = descriptor
    return safe_context


def stable_hash(data: Any) -> str:
    """Return a stable SHA-256 hash for canonical JSON data."""
    return _sha256_bytes(_canonical_json(data).encode("utf-8"))


def manifest_hash(path: Path | None = None) -> str:
    """Return the committed model manifest hash used by audit reports."""
    manifest_path = path or Path(__file__).resolve().parents[2] / "models.jsonl"
    try:
        return _sha256_bytes(manifest_path.read_bytes())
    except OSError:
        return _sha256_bytes(b"")


def _key_bytes(key: bytes | str | None) -> bytes:
    if key is None:
        raise ValueError(_MISSING_KEY_ERROR)
    if isinstance(key, bytes):
        if not key:
            raise ValueError(_MISSING_KEY_ERROR)
        return key
    if isinstance(key, str):
        if not key:
            raise ValueError(_MISSING_KEY_ERROR)
        return key.encode("utf-8")
    raise TypeError("Signing key must be str, bytes, or None")


@dataclass
class DetectorInfo:
    """Detector provenance recorded in an audit report."""

    source: str
    model_id: str
    model_format: str
    commit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model_id": self.model_id,
            "model_format": self.model_format,
            "commit": self.commit,
            "metadata": _sanitize_audit_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectorInfo":
        return cls(
            source=str(data.get("source", "")),
            model_id=str(data.get("model_id", "")),
            model_format=str(data.get("model_format", "")),
            commit=(str(data["commit"]) if data.get("commit") is not None else None),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class AuditSpan:
    """Per-span provenance and redaction action."""

    start: int
    end: int
    label: str
    canonical_label: str
    sources: list[str]
    confidence: float
    threshold: float
    action: str
    surrogate: str | None
    text_hash: str
    evidence: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_offsets()
        self.evidence = _sanitize_audit_value(self.evidence)
        self.context = self._serialized_context()

    def _validate_offsets(self, *, document_length: int | None = None) -> None:
        if type(self.start) is not int or self.start < 0:
            raise ValueError("audit span start must be a non-negative integer")
        if type(self.end) is not int or self.end < self.start:
            raise ValueError("audit span end must be an integer at or after start")
        if document_length is not None and self.end > document_length:
            raise ValueError("audit span offsets must be within document_length")

    def _serialized_context(
        self,
        *,
        document_length: int | None = None,
    ) -> dict[str, dict[str, int | str]]:
        self._validate_offsets(document_length=document_length)
        return _sanitize_audit_context(
            self.context,
            span_start=self.start,
            span_end=self.end,
            document_length=document_length,
        )

    def to_dict(self, *, document_length: int | None = None) -> dict[str, Any]:
        self._validate_offsets(document_length=document_length)
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "canonical_label": self.canonical_label,
            "sources": list(self.sources),
            "confidence": float(self.confidence),
            "threshold": float(self.threshold),
            "action": self.action,
            "surrogate": self.surrogate,
            "text_hash": self.text_hash,
            "evidence": _sanitize_audit_value(self.evidence),
            "context": self._serialized_context(document_length=document_length),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditSpan":
        context = data.get("context")
        return cls(
            start=data.get("start", 0),
            end=data.get("end", 0),
            label=str(data.get("label", "")),
            canonical_label=str(data.get("canonical_label", "")),
            sources=[str(source) for source in data.get("sources", [])],
            confidence=float(data.get("confidence", 0.0)),
            threshold=float(data.get("threshold", 0.0)),
            action=str(data.get("action", "")),
            surrogate=(
                str(data["surrogate"]) if data.get("surrogate") is not None else None
            ),
            text_hash=str(data.get("text_hash", "")),
            evidence=dict(data.get("evidence") or {}),
            context=dict(context) if isinstance(context, Mapping) else {},
        )


@dataclass
class AuditSignature:
    """Signature metadata for an audit report."""

    key_id: str
    algorithm: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditSignature":
        return cls(
            key_id=str(data.get("key_id", "")),
            algorithm=str(data.get("algorithm", "")),
            value=str(data.get("value", "")),
        )


@dataclass
class AuditReport:
    """Signed, reproducible de-identification audit report."""

    policy: str
    resolved_profile: dict[str, Any]
    detectors: list[DetectorInfo]
    safety_sweep: dict[str, Any]
    spans: list[AuditSpan]
    thresholds: dict[str, float]
    residual_risk: dict[str, Any]
    openmed_version: str
    manifest_hash: str
    document_length: int
    input_hash: str
    deidentified_text_hash: str
    grounding: list[dict[str, Any]] = field(default_factory=list)
    repro_hash: str = ""
    signature: AuditSignature | None = None

    def __post_init__(self) -> None:
        self._validate_structure()
        for span in self.spans:
            span.context = span._serialized_context(
                document_length=self.document_length
            )
        self.grounding = self._normalized_grounding()
        if not self.repro_hash:
            self.repro_hash = self.recompute_repro_hash()

    def _normalized_grounding(self) -> list[dict[str, Any]]:
        """Return grounding provenance as sanitized, deterministically ordered dicts.

        Grounding provenance records (see
        :class:`openmed.clinical.grounding.GroundingProvenance`) are fed in as
        PHI-safe mappings alongside detector provenance -- the no-raw-text
        guarantee is a property of the record contract itself (it carries only
        offsets, hashes, and code metadata). They are passed through the standard
        audit sanitizer and sorted by canonical JSON so two logically identical
        runs hash identically regardless of the order codes were emitted.
        """
        normalized = [
            _sanitize_audit_value(dict(entry))
            for entry in self.grounding
            if isinstance(entry, Mapping)
        ]
        return sorted(normalized, key=_canonical_json)

    def _validate_structure(self) -> None:
        if type(self.document_length) is not int or self.document_length < 0:
            raise ValueError("document_length must be a non-negative integer")
        for span in self.spans:
            if not isinstance(span, AuditSpan):
                raise TypeError("audit report spans must contain AuditSpan values")
            span._validate_offsets(document_length=self.document_length)

    def _sorted_spans(self) -> list[AuditSpan]:
        """Return spans in a stable, content-derived order for hashing/export.

        Span order must not depend on detector arbitration or safety-sweep
        sequencing: two logically identical runs that build the same spans in a
        different order must produce identical canonical payloads (and therefore
        identical ``repro_hash`` values). Sorting primarily by ``(start, end,
        canonical_label, action)`` makes ordering a property of the report's
        content rather than of how it was assembled. The canonical JSON of the
        full span dict is appended as a total-order tie-breaker so spans that
        collide on the primary key but differ in any other field (``label``,
        ``sources``, ``confidence``, ``evidence``, ...) still sort
        deterministically rather than retaining input order.
        """
        self._validate_structure()
        return sorted(
            self.spans,
            key=lambda span: (
                span.start,
                span.end,
                span.canonical_label,
                span.action,
                _canonical_json(span.to_dict(document_length=self.document_length)),
            ),
        )

    def _payload(
        self,
        *,
        include_repro_hash: bool,
        include_signature: bool,
        spans: list[AuditSpan] | None = None,
    ) -> dict[str, Any]:
        self._validate_structure()
        span_source = self._sorted_spans() if spans is None else spans
        payload = {
            "policy": self.policy,
            "resolved_profile": _sanitize_audit_value(self.resolved_profile),
            "detectors": [detector.to_dict() for detector in self.detectors],
            "safety_sweep": _sanitize_audit_value(self.safety_sweep),
            "spans": [
                span.to_dict(document_length=self.document_length)
                for span in span_source
            ],
            "thresholds": {
                str(label): float(value)
                for label, value in sorted(self.thresholds.items())
            },
            "residual_risk": _sanitize_audit_value(self.residual_risk),
            "openmed_version": self.openmed_version,
            "manifest_hash": self.manifest_hash,
            "document_length": int(self.document_length),
            "input_hash": self.input_hash,
            "deidentified_text_hash": self.deidentified_text_hash,
        }
        if self.grounding:
            payload["grounding"] = self.grounding
        if include_repro_hash:
            payload["repro_hash"] = self.repro_hash
        if include_signature:
            payload["signature"] = (
                self.signature.to_dict() if self.signature is not None else None
            )
        return payload

    def _hash_for_spans(self, spans: list[AuditSpan]) -> str:
        return stable_hash(
            self._payload(
                include_repro_hash=False,
                include_signature=False,
                spans=spans,
            )
        )

    def recompute_repro_hash(self) -> str:
        """Recompute the report hash without trusting the stored value.

        Uses the deterministic, order-invariant span ordering so two logically
        identical runs hash identically regardless of how spans were assembled.
        """
        return self._hash_for_spans(self._sorted_spans())

    def _legacy_repro_hash(self) -> str:
        """Recompute the hash using the stored span order (pre-sort layout).

        Reports signed before deterministic span ordering was introduced hashed
        spans in their stored array order. This reproduces that legacy payload so
        such untampered reports can still be verified after upgrading.
        """
        return self._hash_for_spans(list(self.spans))

    def repro_hash_matches(self) -> bool:
        """Whether the stored hash matches the deterministic or legacy payload.

        Accepts the legacy stored-order hash so untampered reports produced
        before deterministic span ordering still validate.
        """
        return self.repro_hash in (
            self.recompute_repro_hash(),
            self._legacy_repro_hash(),
        )

    def _serialization_spans(self) -> list[AuditSpan]:
        """Return the span order that keeps persisted report integrity intact.

        Newly produced reports serialize spans in deterministic sorted order.
        Legacy reports that already carry a stored-order ``repro_hash`` must keep
        that stored order when serialized, otherwise a JSON round-trip rewrites
        the signed payload while retaining the old hash and HMAC.
        """
        sorted_spans = self._sorted_spans()
        if self.repro_hash != self._hash_for_spans(sorted_spans):
            stored_spans = list(self.spans)
            if self.repro_hash == self._hash_for_spans(stored_spans):
                return stored_spans
        return sorted_spans

    def to_dict(self) -> dict[str, Any]:
        return self._payload(
            include_repro_hash=True,
            include_signature=True,
            spans=self._serialization_spans(),
        )

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditReport":
        signature_data = data.get("signature")
        return cls(
            policy=str(data.get("policy", "")),
            resolved_profile=dict(data.get("resolved_profile") or {}),
            detectors=[
                DetectorInfo.from_dict(item)
                for item in data.get("detectors", [])
                if isinstance(item, Mapping)
            ],
            safety_sweep=dict(data.get("safety_sweep") or {}),
            spans=[
                AuditSpan.from_dict(item)
                for item in data.get("spans", [])
                if isinstance(item, Mapping)
            ],
            thresholds={
                str(label): float(value)
                for label, value in (data.get("thresholds") or {}).items()
            },
            residual_risk=dict(data.get("residual_risk") or {}),
            openmed_version=str(data.get("openmed_version", "")),
            manifest_hash=str(data.get("manifest_hash", "")),
            document_length=data.get("document_length", 0),
            input_hash=str(data.get("input_hash", "")),
            deidentified_text_hash=str(data.get("deidentified_text_hash", "")),
            grounding=[
                dict(item)
                for item in data.get("grounding", [])
                if isinstance(item, Mapping)
            ],
            repro_hash=str(data.get("repro_hash", "")),
            signature=(
                AuditSignature.from_dict(signature_data)
                if isinstance(signature_data, Mapping)
                else None
            ),
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "AuditReport":
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for AuditReport: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("AuditReport JSON must contain an object")
        return cls.from_dict(parsed)

    def sign(
        self,
        key: bytes | str | None,
        *,
        key_id: str = "release",
    ) -> "AuditReport":
        """Sign the report with a non-empty release HMAC key and return ``self``."""
        self.repro_hash = self.recompute_repro_hash()
        message = _canonical_json(
            self._payload(include_repro_hash=True, include_signature=False)
        ).encode("utf-8")
        signature = hmac.new(_key_bytes(key), message, hashlib.sha256).hexdigest()
        self.signature = AuditSignature(
            key_id=key_id,
            algorithm=_SIGNATURE_ALGORITHM,
            value=signature,
        )
        return self

    def verify(
        self,
        key: bytes | str | None,
        *,
        original_text: str | None = None,
        deidentified_text: str | None = None,
    ) -> bool:
        """Verify the signature and reproducibility hash with a non-empty key."""
        key_bytes = _key_bytes(key)
        if original_text is not None and hash_text(original_text) != self.input_hash:
            return False
        if (
            deidentified_text is not None
            and hash_text(deidentified_text) != self.deidentified_text_hash
        ):
            return False
        if self.signature is None or self.signature.algorithm != _SIGNATURE_ALGORITHM:
            return False

        # Accept either the deterministic (sorted) span ordering or the legacy
        # stored ordering used before deterministic ordering was introduced, so
        # untampered reports signed by earlier versions still verify. Both the
        # repro_hash check and the HMAC are evaluated against the same ordering.
        for spans in (self._sorted_spans(), list(self.spans)):
            if self._hash_for_spans(spans) != self.repro_hash:
                continue
            message = _canonical_json(
                self._payload(
                    include_repro_hash=True,
                    include_signature=False,
                    spans=spans,
                )
            ).encode("utf-8")
            expected = hmac.new(key_bytes, message, hashlib.sha256).hexdigest()
            if hmac.compare_digest(expected, self.signature.value):
                return True
        return False

    def export_review_bundle(self) -> dict[str, Any]:
        """Export reviewable spans and hashed context descriptors without text."""
        self._validate_structure()
        return {
            "policy": self.policy,
            "document_length": self.document_length,
            "input_hash": self.input_hash,
            "deidentified_text_hash": self.deidentified_text_hash,
            "repro_hash": self.repro_hash,
            "spans": [
                {
                    "start": span.start,
                    "end": span.end,
                    "label": span.label,
                    "canonical_label": span.canonical_label,
                    "sources": list(span.sources),
                    "confidence": float(span.confidence),
                    "threshold": float(span.threshold),
                    "action": span.action,
                    "surrogate": span.surrogate,
                    "text_hash": span.text_hash,
                    "context": span._serialized_context(
                        document_length=self.document_length
                    ),
                }
                for span in self._sorted_spans()
            ],
        }

    def export_review_bundle_json(self) -> str:
        return _canonical_json(self.export_review_bundle())

    def attest(
        self,
        profile: str,
        *,
        model_artifacts: Mapping[str, str | Path],
        generated_at: Any = None,
        template_path: str | Path | None = None,
    ) -> Any:
        """Create a data-residency attestation from this audit report.

        Args:
            profile: Bundled jurisdiction attestation profile identifier.
            model_artifacts: Mapping of model labels to local files or
                directories. Contents are hashed without being copied into the
                attestation.
            generated_at: Optional timezone-aware timestamp override.
            template_path: Optional JSON template path for controlled wording
                overrides.

        Returns:
            An ``AttestationReport`` with JSON and Markdown renderers.
        """

        from .attestation import generate_attestation

        return generate_attestation(
            self,
            profile,
            model_artifacts=model_artifacts,
            generated_at=generated_at,
            template_path=template_path,
        )


def recompute_repro_hash(report: AuditReport | Mapping[str, Any]) -> str:
    """Offline helper to recompute a report hash from an object or mapping."""
    if isinstance(report, AuditReport):
        return report.recompute_repro_hash()
    return AuditReport.from_dict(report).recompute_repro_hash()


def verify_repro_hash(report: AuditReport | Mapping[str, Any]) -> bool:
    """Return whether a report's stored hash matches its canonical payload.

    Accepts both the deterministic (sorted) span ordering and the legacy stored
    ordering so untampered reports signed before deterministic ordering still
    validate.
    """
    if isinstance(report, AuditReport):
        return report.repro_hash_matches()
    return AuditReport.from_dict(report).repro_hash_matches()


__all__ = [
    "AuditReport",
    "AuditSignature",
    "AuditSpan",
    "DetectorInfo",
    "hash_text",
    "manifest_hash",
    "recompute_repro_hash",
    "stable_hash",
    "verify_repro_hash",
]
