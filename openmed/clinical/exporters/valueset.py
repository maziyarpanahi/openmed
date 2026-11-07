"""Offline FHIR R4 ValueSet membership validation.

The helpers in this module implement the small, deterministic part of FHIR
``$validate-code`` that can be evaluated from a caller-provided ``ValueSet``
JSON document.  They never contact a terminology server and never bundle a
terminology vocabulary.  Explicit ``compose.include.concept`` entries,
code-oriented filters, exclusions, and complete local expansions are
supported; whole-code-system and hierarchy filters remain non-membership
without a local ``CodeSystem`` expansion.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

__all__ = [
    "VALUESET_VALIDATION_EXTENSION_URL",
    "CodeableConceptValidationResult",
    "ValueSetSource",
    "ValueSetValidationResult",
    "ValidatedCodeableConcept",
    "ValidationPolicy",
    "load_valueset",
    "validate_code",
    "validate_codeable_concept",
]

ValueSetSource: TypeAlias = Mapping[str, Any] | str | PathLike[str]
ValidationPolicy: TypeAlias = Literal["annotate", "drop", "downgrade"]

VALUESET_VALIDATION_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/valueset-validation"
)

_POLICIES = frozenset({"annotate", "drop", "downgrade"})
_MISSING = object()


@dataclass(frozen=True)
class ValueSetValidationResult:
    """Result of validating one ``(system, code)`` pair.

    ``ValueSetValidationResult`` exposes named ``valid`` and ``message``
    fields.  It is also iterable and indexable as a two-item
    ``(valid, message)`` result for callers that prefer tuple unpacking.
    """

    valid: bool
    message: str

    @property
    def ok(self) -> bool:
        """Return whether the code is a member of the supplied ValueSet."""

        return self.valid

    @property
    def is_member(self) -> bool:
        """Return whether the code is a member of the supplied ValueSet."""

        return self.valid

    def __bool__(self) -> bool:
        """Use membership validity for truth-value checks."""

        return self.valid

    def __iter__(self) -> Iterator[bool | str]:
        """Yield ``valid`` and ``message`` for tuple-style unpacking."""

        yield self.valid
        yield self.message

    def __getitem__(self, index: int | str) -> bool | str:
        """Return the tuple-style item at *index*."""

        if index == "valid":
            return self.valid
        if index == "message":
            return self.message
        if index == 0 or index == -2:
            return self.valid
        if index == 1 or index == -1:
            return self.message
        raise IndexError("ValueSetValidationResult has two items")

    def __len__(self) -> int:
        """Return the tuple-style result length."""

        return 2

    def to_dict(self) -> dict[str, bool | str]:
        """Return the result in JSON-friendly mapping form."""

        return {"valid": self.valid, "message": self.message}


class CodeableConceptValidationResult(dict[str, Any]):
    """Validated CodeableConcept mapping with non-FHIR result metadata.

    The object is a ``dict`` containing the validated CodeableConcept, so it
    can be passed directly to an exporter or serialized as FHIR JSON.  The
    ``valid``, ``message``, ``results``, and ``issues`` attributes expose the
    validation metadata without adding non-FHIR keys to the returned concept.
    """

    def __init__(
        self,
        concept: Mapping[str, Any],
        *,
        results: Sequence[ValueSetValidationResult],
        issues: Sequence[Mapping[str, Any]],
    ) -> None:
        super().__init__(concept)
        self.results = tuple(results)
        self.issues = tuple(deepcopy(dict(issue)) for issue in issues)

    @property
    def valid(self) -> bool:
        """Return whether every checked coding is a ValueSet member."""

        return all(result.valid for result in self.results)

    @property
    def ok(self) -> bool:
        """Return whether every checked coding is a ValueSet member."""

        return self.valid

    def __bool__(self) -> bool:
        """Use the aggregate membership result for truth-value checks."""

        return self.valid

    @property
    def message(self) -> str:
        """Return a deterministic summary of the validation results."""

        invalid = [result.message for result in self.results if not result.valid]
        if not invalid:
            return "All CodeableConcept codings are members of the supplied ValueSet."
        return "; ".join(invalid)

    @property
    def codeable_concept(self) -> dict[str, Any]:
        """Return a detached plain mapping of the validated concept."""

        return deepcopy(dict(self))

    @property
    def findings(self) -> tuple[dict[str, Any], ...]:
        """Return validation findings in OperationOutcome-compatible shape."""

        return self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a detached plain mapping of the validated concept."""

        return self.codeable_concept


ValidatedCodeableConcept = CodeableConceptValidationResult


def load_valueset(source: ValueSetSource) -> dict[str, Any]:
    """Load a caller-provided FHIR ``ValueSet`` definition from local JSON.

    Args:
        source: A mapping, an inline JSON object, or a local JSON file path.
            URL sources are deliberately rejected; terminology resolution is
            an explicit out-of-scope network boundary for this helper.

    Returns:
        A detached ValueSet mapping. The source mapping is never mutated.

    Raises:
        TypeError: If *source* is not a mapping, JSON string, or path-like
            source.
        ValueError: If the JSON is malformed, is not an object, or declares a
            resource type other than ``ValueSet``.
        OSError: If a local path cannot be read.
    """

    if isinstance(source, Mapping):
        payload: Any = deepcopy(dict(source))
    elif isinstance(source, str):
        payload = _load_string_source(source)
    elif isinstance(source, PathLike):
        payload = _load_json_path(Path(source))
    else:
        raise TypeError(
            "ValueSet source must be a mapping, inline JSON object, or local path"
        )

    if not isinstance(payload, Mapping):
        raise ValueError("ValueSet JSON must be a JSON object")
    result = deepcopy(dict(payload))
    resource_type = result.get("resourceType")
    if resource_type is not None and resource_type != "ValueSet":
        raise ValueError("ValueSet resourceType must be 'ValueSet'")
    return result


def validate_code(
    system: str,
    code: str,
    valueset: ValueSetSource,
) -> ValueSetValidationResult:
    """Validate one FHIR ``Coding`` against a local ValueSet definition.

    Membership is exact for ``compose.include.concept`` entries and complete
    ``expansion.contains`` entries.  A filter can be evaluated locally when it
    addresses ``code``, ``system``, or a known concept ``display`` using ``=``
    or ``regex``.  A whole-code-system include or a hierarchy filter cannot be
    proven without a CodeSystem and therefore returns ``valid=False`` with a
    clear local-only diagnostic.

    Args:
        system: Canonical FHIR code-system URI.
        code: Code within *system*.
        valueset: A mapping, inline JSON object, or local JSON file path.

    Returns:
        A :class:`ValueSetValidationResult` with ``valid`` and ``message``
        fields. Invalid or locally incomplete membership is never treated as a
        successful validation.
    """

    if not isinstance(system, str) or not system.strip():
        return ValueSetValidationResult(
            False,
            "Code-system URI is required for ValueSet membership validation.",
        )
    if not isinstance(code, str) or not code:
        return ValueSetValidationResult(
            False,
            "Code is required for ValueSet membership validation.",
        )

    payload = load_valueset(valueset)
    return _validate_code_payload(system.strip(), code, payload)


def validate_codeable_concept(
    concept: Mapping[str, Any],
    binding: ValueSetSource,
    *,
    policy: ValidationPolicy = "annotate",
    expression: str = "CodeableConcept",
) -> CodeableConceptValidationResult:
    """Validate and optionally transform every coding in a CodeableConcept.

    The input is copied before validation.  ``annotate`` is the safe default:
    offending codings remain present and receive a FHIR extension containing
    the deterministic membership result.  ``drop`` removes offending codings,
    while ``downgrade`` preserves them and marks the finding as a warning in
    the returned ``issues`` metadata and extension.

    Args:
        concept: FHIR R4 ``CodeableConcept`` mapping.
        binding: A caller-provided ValueSet mapping, inline JSON object, or
            local JSON file path.
        policy: One of ``"annotate"``, ``"drop"``, or ``"downgrade"``.
        expression: FHIRPath base used for returned finding locations.

    Returns:
        A dict-compatible :class:`CodeableConceptValidationResult`. Its FHIR
        mapping contains no validation metadata keys; inspect ``valid``,
        ``message``, or ``issues`` for the local result.

    Raises:
        TypeError: If *concept* is not mapping-shaped.
        ValueError: If *policy* is unknown or the binding is malformed.
    """

    if not isinstance(concept, Mapping):
        raise TypeError("CodeableConcept must be a mapping")
    normalized_policy = _normalize_policy(policy)
    payload = load_valueset(binding)
    result_concept = deepcopy(dict(concept))
    coding_value = result_concept.get("coding", _MISSING)

    if coding_value is _MISSING or coding_value is None:
        return CodeableConceptValidationResult(
            result_concept,
            results=(),
            issues=(),
        )

    single_coding = isinstance(coding_value, Mapping)
    if single_coding:
        raw_codings: list[Any] = [coding_value]
    elif isinstance(coding_value, Sequence) and not isinstance(
        coding_value, (str, bytes)
    ):
        raw_codings = list(coding_value)
    else:
        return CodeableConceptValidationResult(
            result_concept,
            results=(),
            issues=(),
        )

    checked_results: list[ValueSetValidationResult] = []
    issues: list[dict[str, Any]] = []
    transformed: list[Any] = []
    for index, raw_coding in enumerate(raw_codings):
        if not isinstance(raw_coding, Mapping):
            transformed.append(raw_coding)
            continue

        system = raw_coding.get("system")
        code = raw_coding.get("code")
        if isinstance(system, str) and isinstance(code, str):
            validation = _validate_code_payload(
                system.strip(),
                code,
                payload,
            )
        else:
            validation = ValueSetValidationResult(
                False,
                "Coding must include a system and code for ValueSet validation.",
            )
        checked_results.append(validation)

        coding = deepcopy(dict(raw_coding))
        if validation.valid:
            transformed.append(coding)
            continue

        severity = "warning" if normalized_policy == "downgrade" else "error"
        issues.append(
            {
                "finding_code": "valueset-membership",
                "severity": severity,
                "code": "code-invalid",
                "diagnostics": validation.message,
                "expression": [f"{_base_expression(expression)}.coding[{index}]"],
            }
        )
        if normalized_policy == "drop":
            continue
        transformed.append(
            _annotate_coding(
                coding,
                validation,
                policy=normalized_policy,
            )
        )

    if single_coding:
        if transformed:
            result_concept["coding"] = transformed[0]
        else:
            result_concept.pop("coding", None)
    else:
        result_concept["coding"] = transformed

    return CodeableConceptValidationResult(
        result_concept,
        results=checked_results,
        issues=issues,
    )


@dataclass(frozen=True)
class _ClauseEvaluation:
    matched: bool = False
    comparable: bool = False
    unsupported: bool = False


def _validate_code_payload(
    system: str,
    code: str,
    valueset: Mapping[str, Any],
) -> ValueSetValidationResult:
    expansion = valueset.get("expansion")
    if isinstance(expansion, Mapping) and "contains" in expansion:
        matched, complete = _match_expansion(system, code, expansion)
        if matched:
            return _member_result()
        if not complete:
            return ValueSetValidationResult(
                False,
                "The local ValueSet expansion is incomplete; membership "
                "could not be established.",
            )
        return _non_member_result()

    compose = valueset.get("compose")
    if not isinstance(compose, Mapping):
        return ValueSetValidationResult(
            False,
            "The ValueSet has no locally evaluable compose or expansion.",
        )

    includes, include_malformed = _mapping_items(compose.get("include"))
    if include_malformed or not includes:
        return ValueSetValidationResult(
            False,
            "The ValueSet has no locally evaluable compose.include entries.",
        )
    excludes, exclude_malformed = _mapping_items(compose.get("exclude"))
    if exclude_malformed:
        return ValueSetValidationResult(
            False,
            "The ValueSet compose.exclude definition is malformed.",
        )

    included = _evaluate_clauses(includes, system, code, exclusion=False)
    if not included.matched:
        if included.unsupported:
            return ValueSetValidationResult(
                False,
                "The ValueSet composition requires a local CodeSystem "
                "expansion that was not supplied.",
            )
        return _non_member_result()

    excluded = _evaluate_clauses(excludes, system, code, exclusion=True)
    if excluded.matched:
        return ValueSetValidationResult(
            False,
            "The code is explicitly excluded from the supplied ValueSet.",
        )
    if excluded.unsupported:
        return ValueSetValidationResult(
            False,
            "The ValueSet exclusion requires a local CodeSystem expansion "
            "that was not supplied.",
        )
    return _member_result()


def _member_result() -> ValueSetValidationResult:
    return ValueSetValidationResult(
        True,
        "Code is a member of the supplied ValueSet.",
    )


def _non_member_result() -> ValueSetValidationResult:
    return ValueSetValidationResult(
        False,
        "Code is not a member of the supplied ValueSet.",
    )


def _evaluate_clauses(
    clauses: Sequence[Mapping[str, Any]],
    system: str,
    code: str,
    *,
    exclusion: bool,
) -> _ClauseEvaluation:
    comparable = False
    unsupported = False
    for clause in clauses:
        clause_system = clause.get("system")
        if not isinstance(clause_system, str) or not clause_system.strip():
            unsupported = True
            continue
        if clause_system.strip() != system:
            continue
        comparable = True
        concepts, concepts_malformed = _mapping_items(clause.get("concept"))
        filters, filters_malformed = _mapping_items(clause.get("filter"))
        if concepts_malformed or filters_malformed:
            unsupported = True
            continue

        if not concepts and not filters:
            if exclusion:
                return _ClauseEvaluation(
                    matched=True,
                    comparable=True,
                    unsupported=unsupported,
                )
            unsupported = True
            continue

        for concept in concepts:
            concept_code = concept.get("code")
            if not isinstance(concept_code, str):
                unsupported = True
                continue
            if concept_code == code and concept.get("abstract") is not True:
                return _ClauseEvaluation(
                    matched=True,
                    comparable=True,
                    unsupported=unsupported,
                )

        if filters:
            filter_result = _filters_match(
                filters,
                system=system,
                code=code,
                concepts=concepts,
            )
            if filter_result is True:
                return _ClauseEvaluation(
                    matched=True,
                    comparable=True,
                    unsupported=unsupported,
                )
            if filter_result is None:
                unsupported = True

    return _ClauseEvaluation(
        matched=False,
        comparable=comparable,
        unsupported=unsupported,
    )


def _filters_match(
    filters: Sequence[Mapping[str, Any]],
    *,
    system: str,
    code: str,
    concepts: Sequence[Mapping[str, Any]],
) -> bool | None:
    display = next(
        (
            item.get("display")
            for item in concepts
            if item.get("code") == code and isinstance(item.get("display"), str)
        ),
        None,
    )
    for filter_definition in filters:
        property_name = filter_definition.get("property")
        operator = filter_definition.get("op")
        value = filter_definition.get("value")
        if not isinstance(property_name, str) or not isinstance(operator, str):
            return None
        actual: str | None
        if property_name in {"code", "concept"}:
            actual = code
        elif property_name == "system":
            actual = system
        elif property_name == "display":
            actual = display
            if actual is None:
                return None
        else:
            return None

        if not isinstance(value, (str, bool, int, float)):
            return None
        if operator in {"=", "=="}:
            if actual != str(value):
                return False
            continue
        if operator == "regex":
            try:
                matches = re.search(str(value), actual) is not None
            except re.error:
                return None
            if not matches:
                return False
            continue
        if operator == "exists":
            expected = str(value).casefold() in {"true", "1", "yes"}
            if (actual is not None) != expected:
                return False
            continue
        return None
    return True


def _match_expansion(
    system: str,
    code: str,
    expansion: Mapping[str, Any],
) -> tuple[bool, bool]:
    contains, malformed = _mapping_items(expansion.get("contains"))
    flattened = list(_walk_contains(contains))
    for item, inherited_system in flattened:
        item_system = item.get("system") or inherited_system
        if item_system != system or item.get("code") != code:
            continue
        if item.get("abstract") is not True:
            return True, not malformed

    declared_total = expansion.get("total")
    complete = not malformed
    if isinstance(declared_total, int) and declared_total > len(flattened):
        complete = False
    if expansion.get("offset") not in (None, 0):
        complete = False
    return False, complete


def _walk_contains(
    contains: Sequence[Mapping[str, Any]],
    inherited_system: str | None = None,
) -> Iterator[tuple[Mapping[str, Any], str | None]]:
    for item in contains:
        system = item.get("system")
        effective_system = system if isinstance(system, str) else inherited_system
        yield item, effective_system
        nested, _ = _mapping_items(item.get("contains"))
        yield from _walk_contains(nested, effective_system)


def _annotate_coding(
    coding: Mapping[str, Any],
    validation: ValueSetValidationResult,
    *,
    policy: ValidationPolicy,
) -> dict[str, Any]:
    result = deepcopy(dict(coding))
    extensions = result.get("extension")
    if isinstance(extensions, Mapping):
        retained_extensions: list[Any] = [deepcopy(dict(extensions))]
    elif isinstance(extensions, Sequence) and not isinstance(extensions, (str, bytes)):
        retained_extensions = [
            deepcopy(item)
            for item in extensions
            if not (
                isinstance(item, Mapping)
                and item.get("url") == VALUESET_VALIDATION_EXTENSION_URL
            )
        ]
    else:
        retained_extensions = []

    retained_extensions.append(
        {
            "url": VALUESET_VALIDATION_EXTENSION_URL,
            "extension": [
                {"url": "status", "valueCode": "not-member"},
                {"url": "policy", "valueCode": policy},
                {"url": "message", "valueString": validation.message},
            ],
        }
    )
    result["extension"] = retained_extensions
    return result


def _normalize_policy(policy: str) -> ValidationPolicy:
    if not isinstance(policy, str) or policy.casefold() not in _POLICIES:
        raise ValueError("policy must be one of 'annotate', 'drop', or 'downgrade'")
    return cast(ValidationPolicy, policy.casefold())


def _base_expression(expression: Any) -> str:
    if isinstance(expression, str) and expression.strip():
        return expression.strip()
    return "CodeableConcept"


def _mapping_items(value: Any) -> tuple[list[Mapping[str, Any]], bool]:
    if value is None:
        return [], False
    if isinstance(value, Mapping):
        return [value], False
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return [], True
    items = list(value)
    return [item for item in items if isinstance(item, Mapping)], any(
        not isinstance(item, Mapping) for item in items
    )


def _load_string_source(source: str) -> Any:
    stripped = source.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(source)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid ValueSet JSON: {exc}") from exc
    if "://" in source:
        raise ValueError(
            "ValueSet sources must be local JSON or a local path; URLs are not resolved"
        )
    return _load_json_path(Path(source))


def _load_json_path(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid ValueSet JSON in {path}: {exc}") from exc
