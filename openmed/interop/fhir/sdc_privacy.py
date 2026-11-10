"""Privacy projections for FHIR SDC ``QuestionnaireResponse`` resources.

The projection is deliberately narrower than de-identification.  It applies
an explicit allow/drop policy to answer ``value[x]`` fields while retaining
questionnaire item links, nested evidence paths, and the order of items and
answers.  Policy paths use fully indexed FHIRPath-like locations, for example
``QuestionnaireResponse.item[0].answer[0].valueString``.  Repeated elements
must be indexed so that a policy never has to guess which answer it addresses.

Only the response structure and answer values supplied by the caller are
processed.  No network, model, or optional dependency is required.  Returned
summaries contain counts and removed paths only; they never copy answer values.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

__all__ = [
    "AmbiguousPolicyPathError",
    "FieldPolicyInput",
    "InvalidPolicyError",
    "PrivacyProjectionSummary",
    "PolicyDecision",
    "QuestionnaireResponseChangeSummary",
    "QuestionnaireResponsePrivacyError",
    "QuestionnaireResponsePrivacyPolicy",
    "QuestionnaireResponseProjection",
    "UnknownPolicyPathError",
    "project_questionnaire_response",
    "project_questionnaire_response_result",
    "project_questionnaire_response_with_manifest",
    "project_questionnaire_response_with_summary",
]

PolicyDecision: TypeAlias = Literal["allow", "drop"]
FieldPolicyInput: TypeAlias = (
    "Mapping[str, Any] | Sequence[str] | QuestionnaireResponsePrivacyPolicy"
)

_RESOURCE_ROOT = "QuestionnaireResponse"
_DEFAULT_DECISION: PolicyDecision = "drop"
_ITEM_SEGMENT = re.compile(r"item\[(\d+)\]")
_ANSWER_SEGMENT = re.compile(r"answer\[(\d+)\]")
_VALUE_SEGMENT = re.compile(r"value(?:[A-Z][A-Za-z0-9]*|\[x\])?\Z")
_POLICY_RESERVED_KEYS = frozenset(
    {
        "allow",
        "allowed",
        "default",
        "deny",
        "denied",
        "fields",
        "paths",
    }
)


class QuestionnaireResponsePrivacyError(ValueError):
    """Base error for invalid QuestionnaireResponse privacy operations."""


class InvalidPolicyError(QuestionnaireResponsePrivacyError):
    """Raised when a field policy has an invalid shape or decision."""


class AmbiguousPolicyPathError(InvalidPolicyError):
    """Raised when a policy path does not identify one answer value."""


class UnknownPolicyPathError(InvalidPolicyError):
    """Raised when a policy names an answer value absent from the response."""


@dataclass(frozen=True)
class QuestionnaireResponseChangeSummary:
    """Value-free evidence of a QuestionnaireResponse projection.

    ``changed_paths`` contains only canonical structural paths.  It never
    contains a removed value, a replacement value, or a serialized answer.
    """

    items_seen: int = 0
    answers_seen: int = 0
    answers_removed: int = 0
    values_removed: int = 0
    changed_paths: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        """Return whether at least one answer value was removed."""

        return bool(self.changed_paths)

    @property
    def removed_paths(self) -> tuple[str, ...]:
        """Return canonical paths for removed answer values."""

        return self.changed_paths

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free summary mapping."""

        return {
            "items_seen": self.items_seen,
            "answers_seen": self.answers_seen,
            "answers_removed": self.answers_removed,
            "values_removed": self.values_removed,
            "changed": self.changed,
            "changed_paths": list(self.changed_paths),
        }

    as_dict = to_dict

    def to_json(self) -> str:
        """Return deterministic JSON for the value-free summary."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


PrivacyProjectionSummary = QuestionnaireResponseChangeSummary


@dataclass(frozen=True)
class QuestionnaireResponsePrivacyPolicy:
    """Normalized allow/drop rules for QuestionnaireResponse answer values.

    ``rules`` stores canonical paths and decisions in path order.  The
    default is fail-closed: answer values not explicitly allowed are dropped.
    Set ``default`` to ``"allow"`` when using a deny-list policy.
    """

    rules: tuple[tuple[str, PolicyDecision], ...] = field(default_factory=tuple)
    default: PolicyDecision = _DEFAULT_DECISION

    @classmethod
    def from_input(cls, policy: FieldPolicyInput) -> QuestionnaireResponsePrivacyPolicy:
        """Normalize a mapping or allow-list into a strict policy object."""

        return _normalize_policy(policy)

    @property
    def fields(self) -> dict[str, PolicyDecision]:
        """Return a copy of the canonical path decisions."""

        return dict(self.rules)

    def decision_for(self, path: str) -> PolicyDecision:
        """Return the explicit or default decision for a canonical path."""

        for rule_path, decision in self.rules:
            if rule_path == path:
                return decision
        return self.default


@dataclass(frozen=True)
class QuestionnaireResponseProjection:
    """Projected resource paired with its value-free change summary."""

    resource: dict[str, Any]
    summary: QuestionnaireResponseChangeSummary

    def __iter__(self):
        """Allow convenient ``resource, summary = result`` unpacking."""

        yield self.resource
        yield self.summary


@dataclass
class _ProjectionState:
    value_paths: set[str] = field(default_factory=set)
    items_seen: int = 0
    answers_seen: int = 0
    removed_paths: set[str] = field(default_factory=set)
    answers_removed: int = 0


def project_questionnaire_response(
    response: Mapping[str, Any],
    policy: FieldPolicyInput | None = None,
    *,
    field_policy: FieldPolicyInput | None = None,
) -> dict[str, Any]:
    """Return a policy-projected copy of a FHIR QuestionnaireResponse.

    The policy is an allow-list by default.  It may be a direct mapping from
    canonical answer paths to ``"allow"``/``"drop"`` (or booleans), a mapping
    with ``fields``/``paths`` plus an optional ``default``, or a sequence of
    paths to allow.  See :func:`project_questionnaire_response_with_summary`
    for the value-free change summary.

    Args:
        response: FHIR ``QuestionnaireResponse`` mapping.  It is never mutated.
        policy: Explicit field policy.
        field_policy: Keyword alias for ``policy``.

    Returns:
        A deep-copied projected resource with the original item and answer
        ordering preserved.

    Raises:
        AmbiguousPolicyPathError: If a path omits required repeated-element
            indexes, uses a wildcard, or targets a generic value selector.
        UnknownPolicyPathError: If a policy path is not present in the input.
        InvalidPolicyError: If the policy shape or decision is invalid.
        TypeError, ValueError: If the response is not a valid resource shape.
    """

    projected, _ = _project(response, policy, field_policy=field_policy)
    return projected


def project_questionnaire_response_with_summary(
    response: Mapping[str, Any],
    policy: FieldPolicyInput | None = None,
    *,
    field_policy: FieldPolicyInput | None = None,
) -> tuple[dict[str, Any], QuestionnaireResponseChangeSummary]:
    """Project a QuestionnaireResponse and emit a value-free change summary.

    Each removed value is represented by its canonical FHIRPath-like path.
    Removed answer objects are counted, while their retained ``linkId`` item
    containers and nested evidence items remain in their original order.
    """

    return _project(response, policy, field_policy=field_policy)


def project_questionnaire_response_result(
    response: Mapping[str, Any],
    policy: FieldPolicyInput | None = None,
    *,
    field_policy: FieldPolicyInput | None = None,
) -> QuestionnaireResponseProjection:
    """Return a named resource/summary result for callers preferring objects."""

    projected, summary = _project(response, policy, field_policy=field_policy)
    return QuestionnaireResponseProjection(projected, summary)


project_questionnaire_response_with_manifest = (
    project_questionnaire_response_with_summary
)


def _project(
    response: Mapping[str, Any],
    policy: FieldPolicyInput | None,
    *,
    field_policy: FieldPolicyInput | None,
) -> tuple[dict[str, Any], QuestionnaireResponseChangeSummary]:
    """Validate, normalize, and apply one projection without side effects."""

    normalized_policy = _resolve_policy(policy, field_policy)
    _validate_response(response)
    projected = copy.deepcopy(dict(response))

    state = _ProjectionState()
    _collect_paths(projected, state)
    policy_paths = {path for path, _ in normalized_policy.rules}
    missing_paths = policy_paths - state.value_paths
    if missing_paths:
        raise UnknownPolicyPathError(
            "field policy references an answer path absent from the response"
        )

    items = projected.get("item")
    if items is not None:
        _apply_items(items, _RESOURCE_ROOT, normalized_policy, state)

    summary = QuestionnaireResponseChangeSummary(
        items_seen=state.items_seen,
        answers_seen=state.answers_seen,
        answers_removed=state.answers_removed,
        values_removed=len(state.removed_paths),
        changed_paths=tuple(sorted(state.removed_paths)),
    )
    return projected, summary


def _resolve_policy(
    policy: FieldPolicyInput | None,
    field_policy: FieldPolicyInput | None,
) -> QuestionnaireResponsePrivacyPolicy:
    if policy is not None and field_policy is not None:
        raise InvalidPolicyError("provide one field policy, not two")
    selected = policy if policy is not None else field_policy
    if selected is None:
        raise InvalidPolicyError("an explicit field policy is required")
    if isinstance(selected, QuestionnaireResponsePrivacyPolicy):
        return selected
    return _normalize_policy(selected)


def _normalize_policy(policy: FieldPolicyInput) -> QuestionnaireResponsePrivacyPolicy:
    if isinstance(policy, QuestionnaireResponsePrivacyPolicy):
        return policy

    if isinstance(policy, Mapping):
        return _normalize_mapping_policy(policy)

    if isinstance(policy, Sequence) and not isinstance(policy, (str, bytes, bytearray)):
        rules = _rules_from_paths(policy, "allow")
        return QuestionnaireResponsePrivacyPolicy(
            rules=tuple(sorted(rules)),
            default=_DEFAULT_DECISION,
        )

    raise InvalidPolicyError("field policy must be a mapping or path sequence")


def _normalize_mapping_policy(
    policy: Mapping[str, Any],
) -> QuestionnaireResponsePrivacyPolicy:
    keys = set(policy)
    default = _DEFAULT_DECISION
    if "default" in policy:
        default = _coerce_decision(policy["default"])

    has_groups = bool((keys & _POLICY_RESERVED_KEYS) - {"default"})
    if "default" in policy and keys == {"default"}:
        return QuestionnaireResponsePrivacyPolicy(default=default)
    if "default" in policy and not has_groups:
        rules = _rules_from_mapping(
            {path: decision for path, decision in policy.items() if path != "default"}
        )
        return QuestionnaireResponsePrivacyPolicy(
            rules=tuple(sorted(rules)),
            default=default,
        )
    if not has_groups:
        rules = _rules_from_mapping(policy)
        return QuestionnaireResponsePrivacyPolicy(
            rules=tuple(sorted(rules)),
            default=default,
        )

    unknown = keys - _POLICY_RESERVED_KEYS
    if unknown:
        raise InvalidPolicyError("field policy mixes reserved and path keys")

    containers = [name for name in ("fields", "paths") if name in policy]
    groups = [name for name in ("allow", "allowed", "deny", "denied") if name in policy]
    if containers and groups:
        raise InvalidPolicyError("field policy has multiple rule formats")
    if len(containers) > 1:
        raise InvalidPolicyError("field policy repeats its path container")

    if containers:
        rules = _rules_from_mapping(policy[containers[0]])
    else:
        rules = []
        for name in ("allow", "allowed"):
            if name in policy:
                rules.extend(_rules_from_paths(policy[name], "allow"))
        for name in ("deny", "denied"):
            if name in policy:
                rules.extend(_rules_from_paths(policy[name], "drop"))

        if "default" not in policy and {name for name in groups} <= {
            "deny",
            "denied",
        }:
            default = "allow"

    _ensure_unique_rules(rules)
    return QuestionnaireResponsePrivacyPolicy(
        rules=tuple(sorted(rules)),
        default=default,
    )


def _rules_from_mapping(value: Any) -> list[tuple[str, PolicyDecision]]:
    if not isinstance(value, Mapping):
        raise InvalidPolicyError("field policy paths must be a mapping")
    rules: list[tuple[str, PolicyDecision]] = []
    for path, decision in value.items():
        if not isinstance(path, str):
            raise InvalidPolicyError("field policy paths must be strings")
        rules.append((_normalize_policy_path(path), _coerce_decision(decision)))
    _ensure_unique_rules(rules)
    return rules


def _rules_from_paths(
    value: Any, decision: PolicyDecision
) -> list[tuple[str, PolicyDecision]]:
    if isinstance(value, str):
        paths = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        paths = list(value)
    else:
        raise InvalidPolicyError("allow and deny policies must contain paths")

    rules: list[tuple[str, PolicyDecision]] = []
    for path in paths:
        if not isinstance(path, str):
            raise InvalidPolicyError("field policy paths must be strings")
        rules.append((_normalize_policy_path(path), decision))
    _ensure_unique_rules(rules)
    return rules


def _ensure_unique_rules(rules: Sequence[tuple[str, PolicyDecision]]) -> None:
    seen: dict[str, PolicyDecision] = {}
    for path, decision in rules:
        previous = seen.get(path)
        if previous is not None:
            raise AmbiguousPolicyPathError(
                "field policy repeats a canonical answer path"
            )
        seen[path] = decision


def _coerce_decision(value: Any) -> PolicyDecision:
    if value is True:
        return "allow"
    if value is False:
        return "drop"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"allow", "keep", "retain"}:
            return "allow"
        if normalized in {"drop", "deny", "remove"}:
            return "drop"
    raise InvalidPolicyError("field policy decisions must be allow or drop")


def _normalize_policy_path(path: str) -> str:
    candidate = path.strip()
    if not candidate:
        raise AmbiguousPolicyPathError("field policy contains an empty path")

    parts = candidate.split(".")
    if parts and parts[0] == _RESOURCE_ROOT:
        parts = parts[1:]
    if not parts:
        raise AmbiguousPolicyPathError("field policy path does not select an answer")

    normalized: list[str] = []
    previous = "root"
    for index, part in enumerate(parts):
        item_match = _ITEM_SEGMENT.fullmatch(part)
        answer_match = _ANSWER_SEGMENT.fullmatch(part)
        value_match = _VALUE_SEGMENT.fullmatch(part)

        if item_match:
            if previous not in {"root", "item", "answer"}:
                raise InvalidPolicyError("field policy path has invalid item nesting")
            normalized.append(f"item[{_canonical_index(item_match.group(1))}]")
            previous = "item"
            continue

        if answer_match:
            if previous != "item":
                raise InvalidPolicyError(
                    "field policy answer path is not under an item"
                )
            normalized.append(f"answer[{_canonical_index(answer_match.group(1))}]")
            previous = "answer"
            continue

        if value_match:
            if previous != "answer" or index != len(parts) - 1:
                raise InvalidPolicyError(
                    "field policy value path must end at an answer"
                )
            if part == "value[x]":
                raise AmbiguousPolicyPathError(
                    "field policy must name the concrete answer value type"
                )
            normalized.append(part)
            previous = "value"
            continue

        if part in {"item", "answer"} or "*" in part or "[x]" in part:
            raise AmbiguousPolicyPathError(
                "field policy paths must index every repeated element"
            )
        raise InvalidPolicyError(
            "field policy path is not a QuestionnaireResponse answer path"
        )

    if previous != "value":
        raise AmbiguousPolicyPathError("field policy path must select one answer value")
    return f"{_RESOURCE_ROOT}.{'.'.join(normalized)}"


def _canonical_index(value: str) -> int:
    if len(value) > 1 and value.startswith("0"):
        raise AmbiguousPolicyPathError(
            "field policy indexes must be canonical integers"
        )
    return int(value)


def _validate_response(response: Mapping[str, Any]) -> None:
    if not isinstance(response, Mapping):
        raise TypeError("questionnaire response must be a mapping")
    if response.get("resourceType") != _RESOURCE_ROOT:
        raise ValueError("resourceType must be QuestionnaireResponse")
    if "item" in response:
        _validate_items(response["item"])


def _validate_items(items: Any) -> None:
    if not isinstance(items, list):
        raise ValueError("QuestionnaireResponse item fields must be arrays")
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("QuestionnaireResponse items must be objects")
        if "answer" in item:
            answers = item["answer"]
            if not isinstance(answers, list):
                raise ValueError("QuestionnaireResponse answer fields must be arrays")
            for answer in answers:
                if not isinstance(answer, Mapping):
                    raise TypeError("QuestionnaireResponse answers must be objects")
                if "item" in answer:
                    _validate_items(answer["item"])
        if "item" in item:
            _validate_items(item["item"])


def _collect_paths(node: Mapping[str, Any], state: _ProjectionState) -> None:
    items = node.get("item")
    if items is not None:
        _collect_items(items, _RESOURCE_ROOT, state)


def _collect_items(
    items: list[Any],
    parent_path: str,
    state: _ProjectionState,
) -> None:
    for item_index, item in enumerate(items):
        item_path = f"{parent_path}.item[{item_index}]"
        state.items_seen += 1
        answers = item.get("answer")
        if answers is not None:
            for answer_index, answer in enumerate(answers):
                answer_path = f"{item_path}.answer[{answer_index}]"
                state.answers_seen += 1
                for key in answer:
                    if _is_answer_value_key(key):
                        state.value_paths.add(f"{answer_path}.{key}")
                nested_items = answer.get("item")
                if nested_items is not None:
                    _collect_items(nested_items, answer_path, state)
        nested_items = item.get("item")
        if nested_items is not None:
            _collect_items(nested_items, item_path, state)


def _apply_items(
    items: list[Any],
    parent_path: str,
    policy: QuestionnaireResponsePrivacyPolicy,
    state: _ProjectionState,
) -> None:
    for item_index, item in enumerate(items):
        item_path = f"{parent_path}.item[{item_index}]"
        answers = item.get("answer")
        if answers is not None:
            retained_answers: list[Any] = []
            for answer_index, answer in enumerate(answers):
                answer_path = f"{item_path}.answer[{answer_index}]"
                original_value_keys = [
                    key for key in answer if _is_answer_value_key(key)
                ]
                for key in original_value_keys:
                    value_path = f"{answer_path}.{key}"
                    if policy.decision_for(value_path) == "drop":
                        del answer[key]
                        state.removed_paths.add(value_path)

                nested_items = answer.get("item")
                if nested_items is not None:
                    _apply_items(nested_items, answer_path, policy, state)

                if (
                    original_value_keys
                    and not any(_is_answer_value_key(key) for key in answer)
                    and not nested_items
                ):
                    state.answers_removed += 1
                    continue
                retained_answers.append(answer)
            item["answer"] = retained_answers

        nested_items = item.get("item")
        if nested_items is not None:
            _apply_items(nested_items, item_path, policy, state)


def _is_answer_value_key(key: Any) -> bool:
    """Return whether an answer key carries a value[x] payload."""

    return isinstance(key, str) and (
        key == "value" or (key.startswith("value") and len(key) > len("value"))
    )
