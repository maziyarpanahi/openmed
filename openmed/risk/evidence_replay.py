"""Deterministic, counts-only replay of privacy policy evidence.

Evidence replay answers a narrower question than a clinical or compliance
assessment: given the same synthetic aggregate inputs and policy, do the
recorded policy decisions still produce the same aggregate result?  The
manifest contract deliberately contains category counts rather than source
documents, and replay reports contain only counts, identifiers, and digests.
No network or model download is performed by this module.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.audit import stable_hash

EVIDENCE_REPLAY_SCHEMA_VERSION = 1
"""Schema version for counts-only evidence replay manifests."""

EVIDENCE_REPLAY_MANIFEST_KIND = "openmed.risk.evidence_replay"
"""Stable domain label used when hashing replay artifacts."""

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+@-]{0,127}$")
_MISMATCH_ORDER = ("schema", "environment", "policy", "result")
_MISMATCH_CATEGORIES = frozenset(_MISMATCH_ORDER)
_ALLOWED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "policy",
        "environment",
        "synthetic_inputs",
        "expected",
    }
)
_ALLOWED_POLICY_FIELDS = frozenset(
    {"id", "version", "rules", "default_action", "fingerprint"}
)
_ALLOWED_ENVIRONMENT_FIELDS = frozenset(
    {
        "runtime",
        "runtime_version",
        "python",
        "python_version",
        "package",
        "package_version",
        "platform",
        "os",
        "architecture",
        "engine",
        "engine_version",
        "backend",
        "device",
        "offline",
        "dependency_digest",
        "lock_digest",
        "fingerprint",
    }
)
_ALLOWED_EXPECTED_FIELDS = frozenset(
    {"decision_counts", "result_fingerprint", "synthetic_input_count"}
)


class EvidenceReplayError(ValueError):
    """Base error for malformed or unsafe evidence replay inputs."""


class EvidenceReplaySchemaError(EvidenceReplayError):
    """Raised when a replay manifest violates the supported schema."""


class UnsafeReplayInputError(EvidenceReplayError):
    """Raised when input data is not a counts-only synthetic representation."""


@dataclass(frozen=True)
class ReplayMismatch:
    """One privacy-safe replay mismatch.

    ``expected`` and ``actual`` are restricted to validated identifiers,
    digests, counts, booleans, and numbers.  Raw input values are never
    retained by a mismatch or interpolated into its error text.
    """

    category: str
    field: str
    expected: Any = None
    actual: Any = None

    def __post_init__(self) -> None:
        if self.category not in _MISMATCH_CATEGORIES:
            raise ValueError("unsupported replay mismatch category")
        if not isinstance(self.field, str) or not _IDENTIFIER_RE.fullmatch(self.field):
            raise ValueError("replay mismatch field must be a safe identifier")
        object.__setattr__(self, "expected", _normalise_report_value(self.expected))
        object.__setattr__(self, "actual", _normalise_report_value(self.actual))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-safe mismatch payload."""

        return {
            "actual": self.actual,
            "category": self.category,
            "expected": self.expected,
            "field": self.field,
        }


@dataclass(frozen=True)
class EvidenceReplayReport:
    """Aggregate result of one offline evidence replay.

    The report intentionally omits manifest inputs, policy rules, case IDs,
    and any payload-like values.  It is suitable for storage with review
    evidence after callers have separately handled the source manifest.
    """

    manifest_fingerprint: str
    manifest_schema_version: int | str
    verifier_schema_version: int
    expected_policy_fingerprint: str
    actual_policy_fingerprint: str
    expected_environment_fingerprint: str
    actual_environment_fingerprint: str
    expected_decision_counts: Mapping[str, int]
    actual_decision_counts: Mapping[str, int]
    expected_result_fingerprint: str
    actual_result_fingerprint: str
    expected_synthetic_input_count: int
    actual_synthetic_input_count: int
    mismatches: tuple[ReplayMismatch, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest_fingerprint",
            _digest(self.manifest_fingerprint, "manifest_fingerprint"),
        )
        object.__setattr__(
            self,
            "expected_policy_fingerprint",
            _digest(self.expected_policy_fingerprint, "expected_policy_fingerprint"),
        )
        object.__setattr__(
            self,
            "actual_policy_fingerprint",
            _digest(self.actual_policy_fingerprint, "actual_policy_fingerprint"),
        )
        object.__setattr__(
            self,
            "expected_environment_fingerprint",
            _digest(
                self.expected_environment_fingerprint,
                "expected_environment_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "actual_environment_fingerprint",
            _digest(
                self.actual_environment_fingerprint,
                "actual_environment_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "expected_result_fingerprint",
            _digest(self.expected_result_fingerprint, "expected_result_fingerprint"),
        )
        object.__setattr__(
            self,
            "actual_result_fingerprint",
            _digest(self.actual_result_fingerprint, "actual_result_fingerprint"),
        )
        if not isinstance(self.verifier_schema_version, int) or isinstance(
            self.verifier_schema_version, bool
        ):
            raise ValueError("verifier_schema_version must be an integer")
        object.__setattr__(
            self,
            "expected_decision_counts",
            _normalise_count_mapping(self.expected_decision_counts, "expected counts"),
        )
        object.__setattr__(
            self,
            "actual_decision_counts",
            _normalise_count_mapping(self.actual_decision_counts, "actual counts"),
        )
        object.__setattr__(
            self,
            "expected_synthetic_input_count",
            _non_negative_int(
                self.expected_synthetic_input_count,
                "expected_synthetic_input_count",
            ),
        )
        object.__setattr__(
            self,
            "actual_synthetic_input_count",
            _non_negative_int(
                self.actual_synthetic_input_count,
                "actual_synthetic_input_count",
            ),
        )
        object.__setattr__(self, "mismatches", tuple(self.mismatches))

    @property
    def matched(self) -> bool:
        """Whether schema, environment, policy, and result all matched."""

        return not self.mismatches

    @property
    def is_match(self) -> bool:
        """Alias for :attr:`matched` for callers using verifier terminology."""

        return self.matched

    @property
    def mismatch_categories(self) -> tuple[str, ...]:
        """Return mismatch categories in a stable presentation order."""

        present = {mismatch.category for mismatch in self.mismatches}
        return tuple(category for category in _MISMATCH_ORDER if category in present)

    @property
    def result_fingerprint(self) -> str:
        """Return the replayed result fingerprint."""

        return self.actual_result_fingerprint

    def to_dict(self) -> dict[str, Any]:
        """Return the aggregate-only report representation."""

        return {
            "actual_decision_counts": dict(self.actual_decision_counts),
            "actual_environment_fingerprint": self.actual_environment_fingerprint,
            "actual_policy_fingerprint": self.actual_policy_fingerprint,
            "actual_result_fingerprint": self.actual_result_fingerprint,
            "actual_synthetic_input_count": self.actual_synthetic_input_count,
            "expected_decision_counts": dict(self.expected_decision_counts),
            "expected_environment_fingerprint": self.expected_environment_fingerprint,
            "expected_policy_fingerprint": self.expected_policy_fingerprint,
            "expected_result_fingerprint": self.expected_result_fingerprint,
            "expected_synthetic_input_count": self.expected_synthetic_input_count,
            "manifest_fingerprint": self.manifest_fingerprint,
            "manifest_schema_version": self.manifest_schema_version,
            "matched": self.matched,
            "mismatch_categories": list(self.mismatch_categories),
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
            "verifier_schema_version": self.verifier_schema_version,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the report with deterministic JSON settings."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a compact aggregate-only review summary."""

        status = "MATCH" if self.matched else "MISMATCH"
        lines = [
            "# Evidence Replay",
            "",
            f"- Status: `{status}`",
            f"- Manifest fingerprint: `{self.manifest_fingerprint}`",
            f"- Synthetic input count: `{self.actual_synthetic_input_count}`",
            f"- Result fingerprint: `{self.actual_result_fingerprint}`",
            "",
            "## Decision counts",
            "",
            "| Action | Expected | Actual |",
            "|---|---:|---:|",
        ]
        actions = sorted(
            set(self.expected_decision_counts) | set(self.actual_decision_counts)
        )
        if actions:
            for action in actions:
                lines.append(
                    f"| `{action}` | {self.expected_decision_counts.get(action, 0)} "
                    f"| {self.actual_decision_counts.get(action, 0)} |"
                )
        else:
            lines.append("| _none_ | 0 | 0 |")
        lines.extend(["", "## Mismatch categories", ""])
        if self.mismatch_categories:
            lines.extend(f"- `{category}`" for category in self.mismatch_categories)
        else:
            lines.append("None.")
        return "\n".join(lines)


def compute_policy_fingerprint(policy: Mapping[str, Any]) -> str:
    """Return a stable digest for a validated count-based policy."""

    normalized = _normalise_policy(policy)
    return normalized["fingerprint"]


def compute_environment_fingerprint(
    environment: Mapping[str, Any] | str,
) -> str:
    """Return a stable digest for safe local environment metadata.

    A digest string may be supplied when the caller already has a trusted
    environment lock digest.  Mapping values are limited to allow-listed,
    non-payload metadata and are never copied into a replay report.
    """

    normalized = _normalise_environment(environment)
    return normalized["fingerprint"]


def compute_result_fingerprint(
    decision_counts: Mapping[str, int],
    *,
    synthetic_input_count: int | None = None,
) -> str:
    """Return a stable digest for aggregate policy decisions.

    The optional synthetic-input count binds the number of aggregate input
    records without retaining their identifiers or payloads.
    """

    counts = _normalise_count_mapping(decision_counts, "decision counts")
    payload: dict[str, Any] = {
        "decision_counts": counts,
        "kind": f"{EVIDENCE_REPLAY_MANIFEST_KIND}.result",
    }
    if synthetic_input_count is not None:
        payload["synthetic_input_count"] = _non_negative_int(
            synthetic_input_count,
            "synthetic_input_count",
        )
    return stable_hash(payload)


def build_evidence_manifest(
    *,
    policy: Mapping[str, Any],
    synthetic_inputs: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any] | str,
    manifest_id: str = "synthetic-policy-replay",
) -> dict[str, Any]:
    """Build a deterministic counts-only replay manifest.

    Each synthetic input must contain only a ``category_counts`` object whose
    values are non-negative integers.  The helper is useful for producing a
    baseline fixture; callers can later pass the resulting mapping to
    :func:`replay_evidence` with a different policy or environment to classify
    drift.
    """

    manifest_name = _identifier(manifest_id, "manifest_id")
    normalized_policy = _normalise_policy(policy)
    normalized_environment = _normalise_environment(environment)
    normalized_inputs = _normalise_synthetic_inputs(synthetic_inputs)
    decision_counts = _replay_decision_counts(normalized_policy, normalized_inputs)
    input_count = len(normalized_inputs)
    return {
        "environment": normalized_environment,
        "expected": {
            "decision_counts": decision_counts,
            "result_fingerprint": compute_result_fingerprint(
                decision_counts,
                synthetic_input_count=input_count,
            ),
            "synthetic_input_count": input_count,
        },
        "manifest_id": manifest_name,
        "policy": normalized_policy,
        "schema_version": EVIDENCE_REPLAY_SCHEMA_VERSION,
        "synthetic_inputs": normalized_inputs,
    }


def load_evidence_manifest(
    manifest: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Load a manifest mapping or a local JSON manifest file.

    Read failures intentionally use a generic error message so a path or file
    contents cannot leak through an exception into logs or review artifacts.
    """

    if isinstance(manifest, Mapping):
        return dict(manifest)
    if not isinstance(manifest, (str, Path)):
        raise EvidenceReplaySchemaError("evidence replay manifest must be an object")
    try:
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceReplaySchemaError(
            "could not read evidence replay manifest"
        ) from exc
    if not isinstance(payload, Mapping):
        raise EvidenceReplaySchemaError("evidence replay manifest must be an object")
    return dict(payload)


def replay_evidence(
    manifest: Mapping[str, Any] | str | Path,
    *,
    policy: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | str | None = None,
    synthetic_inputs: Sequence[Mapping[str, Any]] | None = None,
) -> EvidenceReplayReport:
    """Replay a counts-only evidence manifest without reading protected data.

    ``policy``, ``environment``, and ``synthetic_inputs`` are optional current
    run values.  When omitted, the corresponding manifest values are replayed,
    which makes a freshly built manifest verify as a match.  Differences are
    returned in the report instead of being raised, while malformed or
    payload-bearing inputs fail closed with a generic, privacy-safe exception.
    """

    payload = load_evidence_manifest(manifest)
    normalized = _normalise_manifest(payload)
    expected_policy = normalized["policy"]
    expected_environment = normalized["environment"]
    manifest_inputs = normalized["synthetic_inputs"]
    expected = normalized["expected"]

    current_policy = expected_policy if policy is None else _normalise_policy(policy)
    current_environment = (
        {"fingerprint": expected_environment["fingerprint"]}
        if environment is None
        else _normalise_environment(environment)
    )
    current_inputs = (
        manifest_inputs
        if synthetic_inputs is None
        else _normalise_synthetic_inputs(synthetic_inputs)
    )
    actual_counts = _replay_decision_counts(current_policy, current_inputs)
    actual_input_count = len(current_inputs)
    actual_result = compute_result_fingerprint(
        actual_counts,
        synthetic_input_count=actual_input_count,
    )

    mismatches: list[ReplayMismatch] = []
    manifest_schema_version = normalized["schema_version"]
    if manifest_schema_version != EVIDENCE_REPLAY_SCHEMA_VERSION:
        mismatches.append(
            ReplayMismatch(
                category="schema",
                field="schema_version",
                expected=manifest_schema_version,
                actual=EVIDENCE_REPLAY_SCHEMA_VERSION,
            )
        )
    if expected_environment["fingerprint"] != current_environment["fingerprint"]:
        mismatches.append(
            ReplayMismatch(
                category="environment",
                field="environment_fingerprint",
                expected=expected_environment["fingerprint"],
                actual=current_environment["fingerprint"],
            )
        )
    if expected_policy["fingerprint"] != current_policy["fingerprint"]:
        mismatches.append(
            ReplayMismatch(
                category="policy",
                field="policy_fingerprint",
                expected=expected_policy["fingerprint"],
                actual=current_policy["fingerprint"],
            )
        )
    if expected["decision_counts"] != actual_counts:
        mismatches.append(
            ReplayMismatch(
                category="result",
                field="decision_counts",
                expected=expected["decision_counts"],
                actual=actual_counts,
            )
        )
    if expected["result_fingerprint"] != actual_result:
        mismatches.append(
            ReplayMismatch(
                category="result",
                field="result_fingerprint",
                expected=expected["result_fingerprint"],
                actual=actual_result,
            )
        )
    if expected["synthetic_input_count"] != actual_input_count:
        mismatches.append(
            ReplayMismatch(
                category="result",
                field="synthetic_input_count",
                expected=expected["synthetic_input_count"],
                actual=actual_input_count,
            )
        )

    return EvidenceReplayReport(
        manifest_fingerprint=stable_hash(_manifest_hash_payload(normalized)),
        manifest_schema_version=manifest_schema_version,
        verifier_schema_version=EVIDENCE_REPLAY_SCHEMA_VERSION,
        expected_policy_fingerprint=expected_policy["fingerprint"],
        actual_policy_fingerprint=current_policy["fingerprint"],
        expected_environment_fingerprint=expected_environment["fingerprint"],
        actual_environment_fingerprint=current_environment["fingerprint"],
        expected_decision_counts=expected["decision_counts"],
        actual_decision_counts=actual_counts,
        expected_result_fingerprint=expected["result_fingerprint"],
        actual_result_fingerprint=actual_result,
        expected_synthetic_input_count=expected["synthetic_input_count"],
        actual_synthetic_input_count=actual_input_count,
        mismatches=tuple(mismatches),
    )


def verify_evidence_replay(
    manifest: Mapping[str, Any] | str | Path,
    *,
    policy: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | str | None = None,
    synthetic_inputs: Sequence[Mapping[str, Any]] | None = None,
) -> EvidenceReplayReport:
    """Verify a replay manifest; equivalent to :func:`replay_evidence`."""

    return replay_evidence(
        manifest,
        policy=policy,
        environment=environment,
        synthetic_inputs=synthetic_inputs,
    )


def _normalise_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_fields(payload, _ALLOWED_MANIFEST_FIELDS, "manifest")
    if "schema_version" not in payload:
        raise EvidenceReplaySchemaError("manifest is missing schema_version")
    schema_version = payload["schema_version"]
    if isinstance(schema_version, str):
        schema_version = _identifier(schema_version, "manifest schema version")
    elif not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise EvidenceReplaySchemaError("manifest schema_version is invalid")
    manifest_id = _identifier(
        payload.get("manifest_id", "evidence-replay"), "manifest_id"
    )
    policy = _normalise_policy(payload.get("policy"))
    environment = _normalise_environment(payload.get("environment"))
    synthetic_inputs = _normalise_synthetic_inputs(payload.get("synthetic_inputs"))
    expected = _normalise_expected(payload.get("expected"))
    return {
        "environment": environment,
        "expected": expected,
        "manifest_id": manifest_id,
        "policy": policy,
        "schema_version": schema_version,
        "synthetic_inputs": synthetic_inputs,
    }


def _normalise_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceReplaySchemaError("policy must be an object")
    _require_fields(value, _ALLOWED_POLICY_FIELDS, "policy")
    policy_id = _identifier(value.get("id"), "policy id")
    policy_version = _identifier(value.get("version"), "policy version")
    default_action = _identifier(value.get("default_action"), "default action")
    raw_rules = value.get("rules")
    if not isinstance(raw_rules, Mapping):
        raise EvidenceReplaySchemaError("policy rules must be an object")
    rules: dict[str, str] = {}
    for category, action in raw_rules.items():
        category_name = _identifier(category, "policy category")
        rules[category_name] = _identifier(action, "policy action")
    normalized: dict[str, Any] = {
        "default_action": default_action,
        "id": policy_id,
        "rules": dict(sorted(rules.items())),
        "version": policy_version,
    }
    fingerprint = stable_hash(
        {
            "kind": f"{EVIDENCE_REPLAY_MANIFEST_KIND}.policy",
            **normalized,
        }
    )
    stored = value.get("fingerprint")
    if stored is not None and stored != fingerprint:
        raise EvidenceReplaySchemaError("policy fingerprint does not match policy")
    normalized["fingerprint"] = fingerprint
    return normalized


def _normalise_environment(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"fingerprint": _digest(value, "environment fingerprint")}
    if not isinstance(value, Mapping):
        raise EvidenceReplaySchemaError("environment must be an object or digest")
    _require_fields(value, _ALLOWED_ENVIRONMENT_FIELDS, "environment")
    metadata: dict[str, Any] = {}
    for key, item in value.items():
        if key == "fingerprint":
            continue
        if key not in _ALLOWED_ENVIRONMENT_FIELDS:
            raise EvidenceReplaySchemaError("environment contains unsupported metadata")
        if isinstance(item, bool):
            metadata[key] = item
        elif isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            metadata[key] = item
        elif isinstance(item, str):
            if key.endswith("digest") or key == "lock_digest":
                metadata[key] = _digest(item, f"environment {key}")
            else:
                metadata[key] = _identifier(item, f"environment {key}")
        else:
            raise EvidenceReplaySchemaError("environment metadata must be safe scalars")
    if not metadata and value.get("fingerprint") is None:
        raise EvidenceReplaySchemaError("environment metadata must not be empty")
    if not metadata and value.get("fingerprint") is not None:
        return {"fingerprint": _digest(value["fingerprint"], "environment fingerprint")}
    fingerprint = stable_hash(
        {
            "kind": f"{EVIDENCE_REPLAY_MANIFEST_KIND}.environment",
            "metadata": dict(sorted(metadata.items())),
        }
    )
    stored = value.get("fingerprint")
    if stored is not None:
        stored_digest = _digest(stored, "environment fingerprint")
        if stored_digest != fingerprint:
            raise EvidenceReplaySchemaError(
                "environment fingerprint does not match metadata"
            )
    return {**dict(sorted(metadata.items())), "fingerprint": fingerprint}


def _normalise_synthetic_inputs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise UnsafeReplayInputError("synthetic_inputs must be a list of count objects")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise UnsafeReplayInputError("synthetic input must contain category counts")
        if set(item) != {"category_counts"}:
            raise UnsafeReplayInputError(
                "synthetic inputs may contain only category_counts"
            )
        normalized.append(
            {
                "category_counts": _normalise_count_mapping(
                    item["category_counts"], "category counts"
                )
            }
        )
    return normalized


def _normalise_expected(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceReplaySchemaError("expected must be an object")
    _require_fields(value, _ALLOWED_EXPECTED_FIELDS, "expected")
    counts = _normalise_count_mapping(value.get("decision_counts"), "expected counts")
    result_fingerprint = _digest(
        value.get("result_fingerprint"),
        "expected result fingerprint",
    )
    input_count = _non_negative_int(
        value.get("synthetic_input_count"),
        "expected synthetic input count",
    )
    return {
        "decision_counts": counts,
        "result_fingerprint": result_fingerprint,
        "synthetic_input_count": input_count,
    }


def _normalise_count_mapping(value: Any, field_name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise UnsafeReplayInputError(f"{field_name} must be an object")
    counts: dict[str, int] = {}
    for key, count in value.items():
        name = _identifier(key, f"{field_name} key")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise UnsafeReplayInputError(
                f"{field_name} values must be non-negative integers"
            )
        counts[name] = count
    return dict(sorted(counts.items()))


def _replay_decision_counts(
    policy: Mapping[str, Any],
    synthetic_inputs: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    rules = policy["rules"]
    default_action = policy["default_action"]
    counts: Counter[str] = Counter()
    for item in synthetic_inputs:
        category_counts = item["category_counts"]
        for category, count in category_counts.items():
            action = rules.get(category, default_action)
            counts[action] += count
    return dict(sorted((action, count) for action, count in counts.items() if count))


def _manifest_hash_payload(normalized: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validated manifest fields used for a safe manifest digest."""

    return {
        "environment": normalized["environment"],
        "expected": normalized["expected"],
        "kind": f"{EVIDENCE_REPLAY_MANIFEST_KIND}.manifest",
        "manifest_id": normalized["manifest_id"],
        "policy": normalized["policy"],
        "schema_version": normalized["schema_version"],
        "synthetic_inputs": normalized["synthetic_inputs"],
    }


def _normalise_report_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("replay report values must be finite")
        return value
    if isinstance(value, str):
        if _DIGEST_RE.fullmatch(value) or _IDENTIFIER_RE.fullmatch(value):
            return value
        raise ValueError("replay report values must be safe identifiers or digests")
    if isinstance(value, Mapping):
        return {
            _identifier(key, "replay report key"): _normalise_report_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_report_value(item) for item in value]
    raise ValueError("replay report values are not JSON-safe")


def _require_fields(
    value: Mapping[Any, Any], allowed: set[str] | frozenset[str], context: str
) -> None:
    if not set(value).issubset(allowed):
        raise EvidenceReplaySchemaError(f"{context} contains unsupported fields")


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise EvidenceReplaySchemaError(f"{field_name} must be a safe identifier")
    return value


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise EvidenceReplaySchemaError(f"{field_name} must be a sha256 digest")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EvidenceReplaySchemaError(f"{field_name} must be a non-negative integer")
    return value


__all__ = [
    "EVIDENCE_REPLAY_MANIFEST_KIND",
    "EVIDENCE_REPLAY_SCHEMA_VERSION",
    "EvidenceReplayError",
    "EvidenceReplayReport",
    "EvidenceReplaySchemaError",
    "ReplayMismatch",
    "UnsafeReplayInputError",
    "build_evidence_manifest",
    "compute_environment_fingerprint",
    "compute_policy_fingerprint",
    "compute_result_fingerprint",
    "load_evidence_manifest",
    "replay_evidence",
    "verify_evidence_replay",
]
