"""Policy profile linting for bundled and file-based profiles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from .labels import (
    CANONICAL_LABELS,
    DIRECT_IDENTIFIER,
    POLICY_LABELS,
    policy_label_for,
)
from .policy import _profile_from_mapping, canonical_policy_name
from .thresholds import load_thresholds

POLICY_SOURCE_NOT_FOUND = "POLICY_SOURCE_NOT_FOUND"
POLICY_JSON_INVALID = "POLICY_JSON_INVALID"
POLICY_SCHEMA_INVALID = "POLICY_SCHEMA_INVALID"
POLICY_UNKNOWN_ACTION_LABEL = "POLICY_UNKNOWN_ACTION_LABEL"
POLICY_DUPLICATE_ACTION_LABEL = "POLICY_DUPLICATE_ACTION_LABEL"
POLICY_ACTION_LABEL_SET_MISMATCH = "POLICY_ACTION_LABEL_SET_MISMATCH"
POLICY_UNKNOWN_POLICY_LABEL = "POLICY_UNKNOWN_POLICY_LABEL"
POLICY_DUPLICATE_POLICY_LABEL = "POLICY_DUPLICATE_POLICY_LABEL"
POLICY_UNKNOWN_THRESHOLD_PROFILE = "POLICY_UNKNOWN_THRESHOLD_PROFILE"
POLICY_HIPAA_DIRECT_IDENTIFIER_KEEP = "POLICY_HIPAA_DIRECT_IDENTIFIER_KEEP"
POLICY_STRICT_POSTURE_KEEP_BIAS = "POLICY_STRICT_POSTURE_KEEP_BIAS"
POLICY_STRICT_POSTURE_REQUIRES_SWEEP = "POLICY_STRICT_POSTURE_REQUIRES_SWEEP"
POLICY_UNREACHABLE_POLICY_LABEL_ACTION = "POLICY_UNREACHABLE_POLICY_LABEL_ACTION"

_SAFE_PATH_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STRICT_POSTURES = frozenset(
    {
        "hipaa_safe_harbor_deidentification",
        "maximum_recall_no_leak",
    }
)


@dataclass(frozen=True)
class PolicyLintFinding:
    """A stable, JSON-serializable policy lint finding."""

    code: str
    message: str
    path: str

    def to_dict(self) -> dict[str, str]:
        """Return the finding as plain JSON-compatible data."""

        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


class _DuplicateTrackingDict(dict[str, Any]):
    __slots__ = ("duplicate_keys",)

    duplicate_keys: tuple[str, ...]


def lint_policy(payload_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Lint a policy profile mapping, bundled policy name, or JSON file path.

    The returned report intentionally includes only structural paths, stable
    codes, and profile metadata needed to diagnose configuration issues. It
    does not echo raw input payloads or free-form metadata values.
    """

    load_result = _load_payload(payload_or_path)
    errors = list(load_result["errors"])
    warnings: list[PolicyLintFinding] = []
    payload = load_result["payload"]
    source = str(load_result["source"])

    if payload is not None and not isinstance(payload, Mapping):
        errors.append(
            PolicyLintFinding(
                code=POLICY_SCHEMA_INVALID,
                message="policy profile must be a JSON object",
                path="$",
            )
        )

    if isinstance(payload, Mapping):
        errors.extend(_duplicate_key_errors(payload))
        errors.extend(_action_label_errors(payload))
        errors.extend(_policy_label_action_errors(payload))

        schema_profile = None
        try:
            schema_profile = _profile_from_mapping(payload, source=source)
        except (TypeError, ValueError):
            if not errors:
                errors.append(
                    PolicyLintFinding(
                        code=POLICY_SCHEMA_INVALID,
                        message="policy profile failed built-in schema validation",
                        path="$",
                    )
                )

        errors.extend(_threshold_profile_errors(payload))

        if schema_profile is not None:
            warnings.extend(_posture_warnings(payload))
            warnings.extend(_unreachable_policy_label_action_warnings(payload))

    report_errors = [finding.to_dict() for finding in errors]
    report_warnings = [finding.to_dict() for finding in warnings]
    return {
        "source": source,
        "valid": not report_errors,
        "error_count": len(report_errors),
        "warning_count": len(report_warnings),
        "errors": report_errors,
        "warnings": report_warnings,
    }


def _load_payload(
    payload_or_path: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(payload_or_path, Mapping):
        return {
            "payload": payload_or_path,
            "source": "mapping",
            "errors": [],
        }

    if isinstance(payload_or_path, Path):
        return _load_path_payload(payload_or_path)

    raw_source = str(payload_or_path)
    candidate = Path(raw_source)
    if candidate.exists():
        return _load_path_payload(candidate)
    if candidate.suffix or "/" in raw_source or "\\" in raw_source:
        return {
            "payload": None,
            "source": "path",
            "errors": [
                PolicyLintFinding(
                    code=POLICY_SOURCE_NOT_FOUND,
                    message="policy profile path was not found",
                    path="$",
                )
            ],
        }

    try:
        canonical_name = canonical_policy_name(raw_source)
    except ValueError:
        return {
            "payload": None,
            "source": "policy",
            "errors": [
                PolicyLintFinding(
                    code=POLICY_SOURCE_NOT_FOUND,
                    message="policy profile name was not found",
                    path="$",
                )
            ],
        }

    resource = resources.files("openmed.core").joinpath(
        "policies",
        f"{canonical_name}.json",
    )
    try:
        with resource.open("r", encoding="utf-8") as handle:
            return {
                "payload": _loads_tracking_duplicates(handle.read()),
                "source": f"policy:{canonical_name}",
                "errors": [],
            }
    except json.JSONDecodeError:
        return {
            "payload": None,
            "source": f"policy:{canonical_name}",
            "errors": [
                PolicyLintFinding(
                    code=POLICY_JSON_INVALID,
                    message="policy profile JSON could not be parsed",
                    path="$",
                )
            ],
        }


def _load_path_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "payload": None,
            "source": "path",
            "errors": [
                PolicyLintFinding(
                    code=POLICY_SOURCE_NOT_FOUND,
                    message="policy profile path was not found",
                    path="$",
                )
            ],
        }
    try:
        payload = _loads_tracking_duplicates(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "payload": None,
            "source": "path",
            "errors": [
                PolicyLintFinding(
                    code=POLICY_JSON_INVALID,
                    message="policy profile JSON could not be parsed",
                    path="$",
                )
            ],
        }
    return {
        "payload": payload,
        "source": "path",
        "errors": [],
    }


def _loads_tracking_duplicates(raw_json: str) -> Mapping[str, Any]:
    try:
        return json.loads(raw_json, object_pairs_hook=_duplicate_tracking_hook)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc


def _duplicate_tracking_hook(
    pairs: list[tuple[str, Any]],
) -> _DuplicateTrackingDict:
    result = _DuplicateTrackingDict()
    duplicate_keys: list[str] = []
    for key, value in pairs:
        if key in result:
            duplicate_keys.append(key)
        result[key] = value
    result.duplicate_keys = tuple(duplicate_keys)
    return result


def _duplicate_key_errors(payload: Mapping[str, Any]) -> list[PolicyLintFinding]:
    findings: list[PolicyLintFinding] = []
    for path, value in _walk_mappings(payload, "$"):
        if not isinstance(value, _DuplicateTrackingDict):
            continue
        if path == "$.actions":
            for key in value.duplicate_keys:
                findings.append(
                    PolicyLintFinding(
                        code=POLICY_DUPLICATE_ACTION_LABEL,
                        message="actions must not define the same label twice",
                        path=_json_path(path, key),
                    )
                )
        elif path == "$.policy_label_actions":
            for key in value.duplicate_keys:
                findings.append(
                    PolicyLintFinding(
                        code=POLICY_DUPLICATE_POLICY_LABEL,
                        message="policy_label_actions must not define a label twice",
                        path=_json_path(path, key),
                    )
                )
    return findings


def _walk_mappings(
    value: Mapping[str, Any],
    path: str,
) -> list[tuple[str, Mapping[str, Any]]]:
    nodes: list[tuple[str, Mapping[str, Any]]] = [(path, value)]
    for key, child in value.items():
        if isinstance(child, Mapping):
            nodes.extend(_walk_mappings(child, _json_path(path, str(key))))
    return nodes


def _action_label_errors(payload: Mapping[str, Any]) -> list[PolicyLintFinding]:
    actions = payload.get("actions")
    if not isinstance(actions, Mapping):
        return []

    findings: list[PolicyLintFinding] = []
    action_labels = {str(label) for label in actions}
    for label in sorted(action_labels):
        if label not in CANONICAL_LABELS:
            findings.append(
                PolicyLintFinding(
                    code=POLICY_UNKNOWN_ACTION_LABEL,
                    message="actions must use canonical OpenMed labels",
                    path=_json_path("$.actions", label),
                )
            )

    missing = sorted(CANONICAL_LABELS - action_labels)
    if missing:
        findings.append(
            PolicyLintFinding(
                code=POLICY_ACTION_LABEL_SET_MISMATCH,
                message="actions must cover the canonical label set exactly",
                path="$.actions",
            )
        )
    return findings


def _policy_label_action_errors(
    payload: Mapping[str, Any],
) -> list[PolicyLintFinding]:
    policy_label_actions = payload.get("policy_label_actions") or {}
    if not isinstance(policy_label_actions, Mapping):
        return []

    findings: list[PolicyLintFinding] = []
    for label in sorted(str(label) for label in policy_label_actions):
        if label not in POLICY_LABELS:
            findings.append(
                PolicyLintFinding(
                    code=POLICY_UNKNOWN_POLICY_LABEL,
                    message="policy_label_actions must use canonical policy labels",
                    path=_json_path("$.policy_label_actions", label),
                )
            )
    return findings


def _threshold_profile_errors(
    payload: Mapping[str, Any],
) -> list[PolicyLintFinding]:
    threshold_profile = payload.get("threshold_profile")
    if not isinstance(threshold_profile, str):
        return []

    matrix = load_thresholds()
    profiles = matrix.get("profiles") or {}
    if threshold_profile in profiles:
        return []
    return [
        PolicyLintFinding(
            code=POLICY_UNKNOWN_THRESHOLD_PROFILE,
            message="threshold_profile must reference the thresholds matrix",
            path="$.threshold_profile",
        )
    ]


def _posture_warnings(payload: Mapping[str, Any]) -> list[PolicyLintFinding]:
    warnings: list[PolicyLintFinding] = []
    posture = str(payload.get("posture") or "").lower()
    hipaa_posture = posture.startswith("hipaa")
    strict_posture = _is_strict_posture(payload)

    policy_label_actions = payload.get("policy_label_actions") or {}
    if (
        hipaa_posture
        and isinstance(policy_label_actions, Mapping)
        and policy_label_actions.get(DIRECT_IDENTIFIER) == "keep"
    ):
        warnings.append(
            PolicyLintFinding(
                code=POLICY_HIPAA_DIRECT_IDENTIFIER_KEEP,
                message="HIPAA posture keeps direct identifiers at policy-label level",
                path=_json_path("$.policy_label_actions", DIRECT_IDENTIFIER),
            )
        )

    actions = payload.get("actions") or {}
    if hipaa_posture and isinstance(actions, Mapping):
        for label, action in sorted(actions.items()):
            canonical_label = str(label)
            if (
                canonical_label in CANONICAL_LABELS
                and action == "keep"
                and policy_label_for(canonical_label) == DIRECT_IDENTIFIER
            ):
                warnings.append(
                    PolicyLintFinding(
                        code=POLICY_HIPAA_DIRECT_IDENTIFIER_KEEP,
                        message="HIPAA posture keeps a direct identifier label",
                        path=_json_path("$.actions", canonical_label),
                    )
                )

    if strict_posture and payload.get("safety_sweep_mandatory") is not True:
        warnings.append(
            PolicyLintFinding(
                code=POLICY_STRICT_POSTURE_REQUIRES_SWEEP,
                message="strict postures should require the deterministic safety sweep",
                path="$.safety_sweep_mandatory",
            )
        )

    default_action = payload.get("default_action")
    default_action_bias = str(payload.get("default_action_bias") or "")
    if strict_posture and (
        default_action == "keep" or default_action_bias.startswith("keep")
    ):
        warnings.append(
            PolicyLintFinding(
                code=POLICY_STRICT_POSTURE_KEEP_BIAS,
                message="strict postures should not default toward keeping labels",
                path="$.default_action_bias",
            )
        )

    return warnings


def _unreachable_policy_label_action_warnings(
    payload: Mapping[str, Any],
) -> list[PolicyLintFinding]:
    actions = payload.get("actions") or {}
    policy_label_actions = payload.get("policy_label_actions") or {}
    if not isinstance(actions, Mapping) or not isinstance(
        policy_label_actions, Mapping
    ):
        return []

    warnings: list[PolicyLintFinding] = []
    for policy_label, fallback_action in sorted(policy_label_actions.items()):
        if policy_label not in POLICY_LABELS:
            continue
        labels_for_policy = [
            label
            for label in CANONICAL_LABELS
            if policy_label_for(label) == str(policy_label)
        ]
        if any(actions.get(label) == fallback_action for label in labels_for_policy):
            continue
        warnings.append(
            PolicyLintFinding(
                code=POLICY_UNREACHABLE_POLICY_LABEL_ACTION,
                message="policy_label_actions entry is shadowed by explicit actions",
                path=_json_path("$.policy_label_actions", str(policy_label)),
            )
        )
    return warnings


def _is_strict_posture(payload: Mapping[str, Any]) -> bool:
    posture = str(payload.get("posture") or "").lower()
    return (
        bool(payload.get("strict_no_leak"))
        or posture in _STRICT_POSTURES
        or ("hipaa" in posture and "safe_harbor" in posture)
        or "no_leak" in posture
    )


def _json_path(parent: str, key: str) -> str:
    if _SAFE_PATH_KEY_RE.fullmatch(key):
        return f"{parent}.{key}"
    return f"{parent}[*]"


__all__ = [
    "PolicyLintFinding",
    "lint_policy",
    "POLICY_ACTION_LABEL_SET_MISMATCH",
    "POLICY_DUPLICATE_ACTION_LABEL",
    "POLICY_DUPLICATE_POLICY_LABEL",
    "POLICY_HIPAA_DIRECT_IDENTIFIER_KEEP",
    "POLICY_JSON_INVALID",
    "POLICY_SCHEMA_INVALID",
    "POLICY_SOURCE_NOT_FOUND",
    "POLICY_STRICT_POSTURE_KEEP_BIAS",
    "POLICY_STRICT_POSTURE_REQUIRES_SWEEP",
    "POLICY_UNKNOWN_ACTION_LABEL",
    "POLICY_UNKNOWN_POLICY_LABEL",
    "POLICY_UNKNOWN_THRESHOLD_PROFILE",
    "POLICY_UNREACHABLE_POLICY_LABEL_ACTION",
]
