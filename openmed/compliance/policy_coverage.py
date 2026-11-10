"""Deterministic, privacy-safe coverage matrices for policy profiles.

The matrix joins bundled policy actions with the canonical label taxonomy and
the structured-column semantic map.  Fixture and focused-test references are
identifiers only: the generator never loads fixture payloads or copies source
values into the report.  It is therefore suitable for local release evidence
without a network call or a user-data dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Final

from openmed.core.audit import stable_hash
from openmed.core.labels import (
    CANONICAL_LABELS,
    COLUMN_SEMANTIC_LABELS,
    policy_label_for,
)
from openmed.core.policy import (
    PolicyProfile,
    list_policies,
    load_policy,
    policy_requirements,
)
from openmed.core.schemas.span import ACTION_VALUES

SCHEMA_VERSION: Final = "openmed.policy_coverage.v1"
MATRIX_ID: Final = "openmed-policy-coverage"
MANIFEST_FILENAME: Final = "policy-coverage.json"
MARKDOWN_FILENAME: Final = "policy-coverage.md"
POLICY_RESOURCE_ROOT: Final = "openmed/core/policies"
STRUCTURED_RESOURCE_ROOT: Final = "openmed/core/labels.py::COLUMN_SEMANTIC_LABELS"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_TEST_REFERENCE_RE = re.compile(r"^[A-Za-z0-9_./-]+(?:::[A-Za-z_][A-Za-z0-9_]*)+$")
_DEFAULT_TEST_REFERENCE: Final = (
    "tests/unit/compliance/test_policy_coverage.py::"
    "test_required_rules_have_fixture_and_focused_test_coverage"
)
_FIXTURE_BY_POLICY_LABEL: Final = {
    "DIRECT_IDENTIFIER": "synthetic-policy-direct-identifiers",
    "QUASI_IDENTIFIER": "synthetic-policy-quasi-identifiers",
    "SENSITIVE_ATTRIBUTE": "synthetic-policy-sensitive-attributes",
    "CLINICAL_CONCEPT": "synthetic-policy-clinical-concepts",
}


class PolicyCoverageError(ValueError):
    """Base error for invalid or incomplete policy coverage evidence."""


class UncoveredPolicyRuleError(PolicyCoverageError):
    """Raised when a required policy rule has no complete evidence binding."""

    def __init__(self, uncovered_rule_ids: Sequence[str]) -> None:
        self.uncovered_rule_ids = tuple(
            sorted(str(item) for item in uncovered_rule_ids)
        )
        super().__init__(
            "policy coverage matrix has "
            f"{len(self.uncovered_rule_ids)} uncovered required rule(s)"
        )


@dataclass(frozen=True)
class PolicyCoverageBinding:
    """Identifier-only fixture and focused-test evidence for one policy rule."""

    policy_name: str
    label: str
    fixture_id: str
    focused_test: str

    def __post_init__(self) -> None:
        _validate_identifier(self.policy_name, "policy_name")
        _validate_canonical_label(self.label)
        _validate_identifier(self.fixture_id, "fixture_id")
        _validate_test_reference(self.focused_test)

    @property
    def rule_id(self) -> str:
        """Return the stable identifier for the policy action rule."""

        return f"{self.policy_name}.actions.{self.label}"

    def to_dict(self) -> dict[str, str]:
        """Return the binding without fixture contents or test output."""

        return {
            "focused_test": self.focused_test,
            "fixture_id": self.fixture_id,
            "label": self.label,
            "policy_name": self.policy_name,
            "rule_id": self.rule_id,
        }


@dataclass(frozen=True)
class PolicyCoverageRow:
    """One privacy-safe policy-rule row in the coverage matrix."""

    policy_name: str
    label: str
    policy_label: str
    resource_path: str
    action: str
    required: bool
    structured_fields: tuple[str, ...] = ()
    fixture_id: str | None = None
    focused_test: str | None = None
    resource_hash: str = ""
    fixture_hash: str | None = None
    focused_test_hash: str | None = None
    row_hash: str = ""

    def __post_init__(self) -> None:
        _validate_identifier(self.policy_name, "policy_name")
        _validate_canonical_label(self.label)
        _validate_identifier(self.policy_label, "policy_label")
        if self.policy_label != policy_label_for(self.label):
            raise ValueError("policy label does not match canonical label metadata")
        if self.resource_path != _policy_resource_path(self.policy_name, self.label):
            raise ValueError("resource_path must identify the policy action")
        if self.action not in ACTION_VALUES:
            raise ValueError("action is not a supported policy action")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")
        fields = tuple(sorted(self.structured_fields))
        for field_name in fields:
            _validate_identifier(field_name, "structured_field")
        object.__setattr__(self, "structured_fields", fields)
        if self.fixture_id is not None:
            _validate_identifier(self.fixture_id, "fixture_id")
        if self.focused_test is not None:
            _validate_test_reference(self.focused_test)
        for field_name in ("resource_hash", "fixture_hash", "focused_test_hash"):
            value = getattr(self, field_name)
            if value is not None and value and not _is_sha256_hash(value):
                raise ValueError(f"{field_name} must be a sha256 digest")
        if self.row_hash and not _is_sha256_hash(self.row_hash):
            raise ValueError("row_hash must be a sha256 digest")

    @property
    def rule_id(self) -> str:
        """Return the stable identifier for this row's policy rule."""

        return f"{self.policy_name}.actions.{self.label}"

    @property
    def covered(self) -> bool:
        """Return whether both required evidence references are present."""

        return bool(self.fixture_id and self.focused_test)

    @property
    def status(self) -> str:
        """Return a stable, non-sensitive coverage status code."""

        if self.covered:
            return "covered"
        if self.required:
            return "uncovered"
        return "not-required"

    def to_dict(self) -> dict[str, Any]:
        """Return only identifiers, counts, status, and hashes."""

        return {
            "action": self.action,
            "covered": self.covered,
            "fixture_id": self.fixture_id,
            "focused_test": self.focused_test,
            "hashes": {
                "fixture": self.fixture_hash,
                "focused_test": self.focused_test_hash,
                "resource": self.resource_hash,
                "row": self.row_hash,
            },
            "label": self.label,
            "policy_label": self.policy_label,
            "policy_name": self.policy_name,
            "required": self.required,
            "resource_path": self.resource_path,
            "rule_id": self.rule_id,
            "status": self.status,
            "structured_field_count": len(self.structured_fields),
            "structured_fields": list(self.structured_fields),
        }


@dataclass(frozen=True)
class PolicyCoverageMatrix:
    """Deterministic matrix of policy rules and local coverage evidence."""

    rows: tuple[PolicyCoverageRow, ...]
    policy_resource_hashes: Mapping[str, str]
    structured_field_ids: tuple[str, ...]
    structured_field_hash: str
    schema_version: str = SCHEMA_VERSION
    matrix_id: str = MATRIX_ID

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported policy coverage schema version")
        if self.matrix_id != MATRIX_ID:
            raise ValueError("unsupported policy coverage matrix id")
        rows = tuple(sorted(self.rows, key=lambda row: row.rule_id))
        rule_ids = [row.rule_id for row in rows]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("policy coverage rule identifiers must be unique")
        if not rows:
            raise ValueError("policy coverage matrix must contain rows")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(
            self, "structured_field_ids", tuple(sorted(self.structured_field_ids))
        )
        if not _is_sha256_hash(self.structured_field_hash):
            raise ValueError("structured_field_hash must be a sha256 digest")

    @property
    def policy_count(self) -> int:
        """Return the number of policy profiles in the matrix."""

        return len({row.policy_name for row in self.rows})

    @property
    def required_rule_count(self) -> int:
        """Return the number of non-``keep`` policy rules."""

        return sum(row.required for row in self.rows)

    @property
    def covered_required_rule_count(self) -> int:
        """Return the number of covered required rules."""

        return sum(row.required and row.covered for row in self.rows)

    @property
    def uncovered_required_rules(self) -> tuple[str, ...]:
        """Return stable identifiers for required rows lacking evidence."""

        return tuple(
            row.rule_id for row in self.rows if row.required and not row.covered
        )

    @property
    def coverage_percent(self) -> float:
        """Return required-rule coverage as a percentage."""

        if not self.required_rule_count:
            return 100.0
        return round(
            100.0 * self.covered_required_rule_count / self.required_rule_count,
            6,
        )

    @property
    def fixture_ids(self) -> tuple[str, ...]:
        """Return the distinct fixture identifiers referenced by the matrix."""

        return tuple(sorted({row.fixture_id for row in self.rows if row.fixture_id}))

    @property
    def focused_tests(self) -> tuple[str, ...]:
        """Return the distinct focused-test references in the matrix."""

        return tuple(
            sorted({row.focused_test for row in self.rows if row.focused_test})
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable hash of the matrix's privacy-safe contents."""

        return stable_hash(self._identity_payload())

    @property
    def verified(self) -> bool:
        """Return whether every required rule has complete evidence."""

        return not self.uncovered_required_rules

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "matrix_id": self.matrix_id,
            "policy_resource_hashes": dict(sorted(self.policy_resource_hashes.items())),
            "rows": [row.to_dict() for row in self.rows],
            "schema_version": self.schema_version,
            "structured_field_hash": self.structured_field_hash,
            "structured_field_ids": list(self.structured_field_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a counts-and-hashes manifest without source payloads."""

        return {
            "hashes": {
                "matrix": self.fingerprint,
                "policy_resources": dict(sorted(self.policy_resource_hashes.items())),
                "structured_fields": self.structured_field_hash,
            },
            "matrix_id": self.matrix_id,
            "rows": [row.to_dict() for row in self.rows],
            "schema_version": self.schema_version,
            "summary": {
                "covered_required_rule_count": self.covered_required_rule_count,
                "coverage_percent": self.coverage_percent,
                "fixture_count": len(self.fixture_ids),
                "focused_test_count": len(self.focused_tests),
                "policy_count": self.policy_count,
                "required_rule_count": self.required_rule_count,
                "row_count": len(self.rows),
                "structured_field_count": len(self.structured_field_ids),
                "uncovered_required_rule_count": len(self.uncovered_required_rules),
            },
            "uncovered_required_rules": list(self.uncovered_required_rules),
        }


@dataclass(frozen=True)
class PolicyCoverageResult:
    """In-memory and on-disk output produced by the coverage generator."""

    output_dir: Path
    manifest_path: Path
    markdown_path: Path
    matrix: PolicyCoverageMatrix
    manifest: Mapping[str, Any]
    markdown: str


def build_policy_coverage_matrix(
    policies: Sequence[str | PolicyProfile] | None = None,
    *,
    bindings: Sequence[PolicyCoverageBinding]
    | Mapping[str, PolicyCoverageBinding]
    | None = None,
    raise_on_uncovered: bool = True,
) -> PolicyCoverageMatrix:
    """Build a local policy-coverage matrix.

    Args:
        policies: Canonical policy names or already-loaded profiles.  The
            bundled canonical set is used by default.
        bindings: Optional identifier-only evidence bindings keyed by rule.
            Omitting this argument uses the bundled synthetic coverage
            catalog.  The catalog contains no fixture payloads.
        raise_on_uncovered: Raise :class:`UncoveredPolicyRuleError` when a
            required rule lacks either evidence reference.

    Returns:
        A deterministic matrix covering every canonical label for every
        selected profile, including ``keep`` rules for drift visibility.
    """

    profiles = _resolve_profiles(policies)
    names = tuple(profile.name for profile in profiles)
    binding_map = (
        _default_bindings(names) if bindings is None else _normalise_bindings(bindings)
    )
    structured_fields = _structured_fields_by_label()
    structured_field_ids = tuple(
        f"{field}:{label}" for field, label in sorted(COLUMN_SEMANTIC_LABELS.items())
    )
    structured_field_hash = stable_hash(
        [
            {"field": field, "label": label}
            for field, label in sorted(COLUMN_SEMANTIC_LABELS.items())
        ]
    )
    resource_hashes = {
        profile.name: _policy_resource_hash(profile.name) for profile in profiles
    }
    rows: list[PolicyCoverageRow] = []

    for profile in profiles:
        required_labels = {
            requirement.label for requirement in policy_requirements(profile)
        }
        for label in sorted(CANONICAL_LABELS):
            action = profile.action_for(label)
            policy_label = policy_label_for(label)
            resource_path = _policy_resource_path(profile.name, label)
            binding = binding_map.get(f"{profile.name}.actions.{label}")
            fixture_id = binding.fixture_id if binding else None
            focused_test = binding.focused_test if binding else None
            fixture_hash = (
                stable_hash(
                    {
                        "fixture_id": fixture_id,
                        "policy_label": policy_label,
                    }
                )
                if fixture_id
                else None
            )
            focused_test_hash = (
                stable_hash({"focused_test": focused_test}) if focused_test else None
            )
            row_payload = {
                "action": action,
                "fixture_hash": fixture_hash,
                "fixture_id": fixture_id,
                "focused_test": focused_test,
                "focused_test_hash": focused_test_hash,
                "label": label,
                "policy_label": policy_label,
                "policy_name": profile.name,
                "required": label in required_labels,
                "resource_hash": resource_hashes[profile.name],
                "resource_path": resource_path,
                "structured_fields": structured_fields.get(label, ()),
            }
            rows.append(
                PolicyCoverageRow(
                    **row_payload,
                    row_hash=stable_hash(row_payload),
                )
            )

    matrix = PolicyCoverageMatrix(
        rows=tuple(rows),
        policy_resource_hashes=resource_hashes,
        structured_field_ids=structured_field_ids,
        structured_field_hash=structured_field_hash,
    )
    if raise_on_uncovered and matrix.uncovered_required_rules:
        raise UncoveredPolicyRuleError(matrix.uncovered_required_rules)
    return matrix


def generate_policy_coverage(
    output_dir: str | Path,
    *,
    policies: Sequence[str | PolicyProfile] | None = None,
    bindings: Sequence[PolicyCoverageBinding]
    | Mapping[str, PolicyCoverageBinding]
    | None = None,
) -> PolicyCoverageResult:
    """Write deterministic JSON and Markdown policy-coverage artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matrix = build_policy_coverage_matrix(policies, bindings=bindings)
    manifest = matrix.to_dict()
    markdown = render_policy_coverage_markdown(matrix)
    manifest_path = destination / MANIFEST_FILENAME
    markdown_path = destination / MARKDOWN_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return PolicyCoverageResult(
        output_dir=destination,
        manifest_path=manifest_path,
        markdown_path=markdown_path,
        matrix=matrix,
        manifest=manifest,
        markdown=markdown,
    )


def render_policy_coverage_markdown(matrix: PolicyCoverageMatrix) -> str:
    """Render a deterministic matrix using identifiers, counts, and hashes."""

    if not isinstance(matrix, PolicyCoverageMatrix):
        raise TypeError("matrix must be a PolicyCoverageMatrix")
    lines = [
        "# Privacy-policy coverage matrix",
        "",
        "This is deterministic local evidence, not a compliance certification.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Policies | {matrix.policy_count} |",
        f"| Rows | {len(matrix.rows)} |",
        f"| Required rules | {matrix.required_rule_count} |",
        f"| Covered required rules | {matrix.covered_required_rule_count} |",
        f"| Coverage | {matrix.coverage_percent:.6f}% |",
        f"| Structured fields | {len(matrix.structured_field_ids)} |",
        f"| Fixtures | {len(matrix.fixture_ids)} |",
        f"| Focused tests | {len(matrix.focused_tests)} |",
        f"| Matrix hash | `{matrix.fingerprint}` |",
        f"| Structured-field hash | `{matrix.structured_field_hash}` |",
        "",
        "## Rule matrix",
        "",
        "| Policy rule | Resource path | Action | Fixture | Focused test | "
        "Structured-field count | Status | Row hash |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in matrix.rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(row.rule_id),
                    _markdown_cell(row.resource_path),
                    _markdown_cell(row.action),
                    _markdown_cell(row.fixture_id or ""),
                    _markdown_cell(row.focused_test or ""),
                    str(len(row.structured_fields)),
                    _markdown_cell(row.status),
                    _markdown_cell(row.row_hash),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def validate_policy_coverage(matrix: PolicyCoverageMatrix) -> None:
    """Fail closed if a matrix contains an uncovered required rule."""

    if not isinstance(matrix, PolicyCoverageMatrix):
        raise TypeError("matrix must be a PolicyCoverageMatrix")
    if matrix.uncovered_required_rules:
        raise UncoveredPolicyRuleError(matrix.uncovered_required_rules)


def _resolve_profiles(
    policies: Sequence[str | PolicyProfile] | None,
) -> tuple[PolicyProfile, ...]:
    selected = list_policies() if policies is None else tuple(policies)
    if not selected:
        raise ValueError("at least one policy profile is required")
    profiles: list[PolicyProfile] = []
    seen: set[str] = set()
    for item in selected:
        if isinstance(item, PolicyProfile):
            profile = item
        elif isinstance(item, str):
            _validate_identifier(item, "policy_name")
            try:
                profile = load_policy(item)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise PolicyCoverageError("policy profile could not be loaded") from exc
        else:
            raise TypeError("policies must contain policy names or profiles")
        if profile.name in seen:
            continue
        seen.add(profile.name)
        profiles.append(profile)
    return tuple(sorted(profiles, key=lambda profile: profile.name))


def _default_bindings(
    policy_names: Sequence[str],
) -> dict[str, PolicyCoverageBinding]:
    bindings: dict[str, PolicyCoverageBinding] = {}
    for policy_name in policy_names:
        for label in sorted(CANONICAL_LABELS):
            policy_label = policy_label_for(label)
            fixture_id = _FIXTURE_BY_POLICY_LABEL.get(policy_label)
            if fixture_id is None:
                raise PolicyCoverageError(
                    "canonical label metadata contains an unsupported policy class"
                )
            binding = PolicyCoverageBinding(
                policy_name=policy_name,
                label=label,
                fixture_id=fixture_id,
                focused_test=_DEFAULT_TEST_REFERENCE,
            )
            bindings[binding.rule_id] = binding
    return bindings


def _normalise_bindings(
    bindings: Sequence[PolicyCoverageBinding] | Mapping[str, PolicyCoverageBinding],
) -> dict[str, PolicyCoverageBinding]:
    values = (
        tuple(bindings.values()) if isinstance(bindings, Mapping) else tuple(bindings)
    )
    result: dict[str, PolicyCoverageBinding] = {}
    for binding in values:
        if not isinstance(binding, PolicyCoverageBinding):
            raise TypeError("bindings must contain PolicyCoverageBinding objects")
        if binding.rule_id in result:
            raise ValueError("policy coverage bindings must have unique rule ids")
        result[binding.rule_id] = binding
    return result


def _structured_fields_by_label() -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {}
    for field_name, label in COLUMN_SEMANTIC_LABELS.items():
        if not isinstance(field_name, str) or not _IDENTIFIER_RE.fullmatch(field_name):
            raise PolicyCoverageError(
                "structured semantic map contains an invalid field id"
            )
        if label not in CANONICAL_LABELS:
            raise PolicyCoverageError(
                "structured semantic map contains an unknown canonical label"
            )
        result.setdefault(label, []).append(field_name)
    return {label: tuple(sorted(fields)) for label, fields in result.items()}


def _policy_resource_path(policy_name: str, label: str) -> str:
    return f"{POLICY_RESOURCE_ROOT}/{policy_name}.json#/actions/{label}"


def _policy_resource_hash(policy_name: str) -> str:
    try:
        resource = resources.files("openmed.core").joinpath(
            "policies", f"{policy_name}.json"
        )
        return _sha256(resource.read_bytes())
    except OSError as exc:
        raise PolicyCoverageError(
            "bundled policy resource could not be hashed"
        ) from exc


def _validate_identifier(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a stable identifier")


def _validate_canonical_label(value: Any) -> None:
    if not isinstance(value, str) or value not in CANONICAL_LABELS:
        raise ValueError("label must be a canonical policy label")


def _validate_test_reference(value: Any) -> None:
    if not isinstance(value, str) or not _TEST_REFERENCE_RE.fullmatch(value):
        raise ValueError("focused_test must be a stable test reference")


def _is_sha256_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", value))


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "MANIFEST_FILENAME",
    "MARKDOWN_FILENAME",
    "MATRIX_ID",
    "POLICY_RESOURCE_ROOT",
    "SCHEMA_VERSION",
    "STRUCTURED_RESOURCE_ROOT",
    "PolicyCoverageBinding",
    "PolicyCoverageError",
    "PolicyCoverageMatrix",
    "PolicyCoverageResult",
    "PolicyCoverageRow",
    "UncoveredPolicyRuleError",
    "build_policy_coverage_matrix",
    "generate_policy_coverage",
    "render_policy_coverage_markdown",
    "validate_policy_coverage",
]
