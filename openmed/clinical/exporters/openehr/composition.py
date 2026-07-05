"""openEHR flat-JSON COMPOSITION export for grounded clinical spans."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openmed.clinical.exporters.flat_table import flatten_entities

from .binding import DEFAULT_OPENEHR_BINDINGS, OpenEHRBinding, binding_for_entity_kind

__all__ = [
    "OpenEHRCoding",
    "OpenEHRTemplate",
    "OpenEHRValidationResult",
    "extract_round_trip_coded_values",
    "parse_operational_template",
    "to_openehr_composition",
    "validate_openehr_composition",
]

_MISSING = object()
_INDEX_RE = re.compile(r":\d+(?=/|\|)")
_SPAN_POINTER_RE = re.compile(r"^(?P<doc_id>.*):(?P<start>\d+)-(?P<end>\d+)$")

_CONTEXT_PATHS = frozenset(
    {
        "ctx/language",
        "ctx/territory",
        "ctx/time",
        "ctx/composer_name",
        "ctx/composer_self",
    }
)
_FEEDER_SUFFIXES = frozenset(
    {
        "originating_system_audit|system_id",
        "originating_system_audit|version_id",
        "originating_system_item_id:0|id",
        "originating_system_item_id:0|issuer",
        "originating_system_item_id:0|assigner",
        "originating_system_item_id:0|type",
    }
)
_TEXT_FIELDS = ("normalized_text", "text", "entity_text", "word", "surface")
_VALUE_FIELDS = (
    "value",
    "magnitude",
    "quantity_value",
    "measurement_value",
    "result_value",
)
_UNIT_FIELDS = ("unit", "units", "value_unit", "result_unit")
_CANDIDATE_FIELDS = ("candidates", "grounding_candidates", "ranked_candidates")
_SYSTEM_TERMINOLOGY = {
    "HTTP://SNOMED.INFO/SCT": "SNOMED-CT",
    "SNOMED": "SNOMED-CT",
    "SNOMED-CT": "SNOMED-CT",
    "HTTP://LOINC.ORG": "LOINC",
    "LOINC": "LOINC",
    "HTTP://WWW.NLM.NIH.GOV/RESEARCH/UMLS/RXNORM": "RxNorm",
    "RXNORM": "RxNorm",
    "HTTP://HL7.ORG/FHIR/SID/ICD-10-CM": "ICD-10-CM",
    "ICD10CM": "ICD-10-CM",
    "ICD-10-CM": "ICD-10-CM",
}


@dataclass(frozen=True)
class OpenEHRCoding:
    """Terminology coding emitted into an openEHR coded-text flat element."""

    system: str
    code: str
    display: str
    terminology: str


@dataclass(frozen=True)
class _EntitySpan:
    kind: str
    text: str
    start: int
    end: int
    coding: OpenEHRCoding | None
    value: int | float | None
    unit: str


@dataclass(frozen=True)
class OpenEHRTemplate:
    """Allowed flat paths parsed from an EHRbase WebTemplate."""

    template_id: str
    allowed_paths: frozenset[str]
    element_paths: frozenset[str]
    source: str = "webtemplate"

    def allows_path(self, path: str) -> bool:
        """Return whether ``path`` is valid for this flat COMPOSITION."""

        if path in _CONTEXT_PATHS:
            return True
        feeder = _split_feeder_audit_path(path)
        if feeder is not None:
            base_path, feeder_suffix = feeder
            return (
                _normalize_flat_path(base_path) in self.element_paths
                and feeder_suffix in _FEEDER_SUFFIXES
            )
        return _normalize_flat_path(path) in self.allowed_paths


@dataclass(frozen=True)
class OpenEHRValidationResult:
    """Conformance and provenance findings for a flat COMPOSITION."""

    out_of_template_paths: tuple[str, ...] = ()
    missing_feeder_audit_paths: tuple[str, ...] = ()
    unresolved_feeder_audit_paths: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Return ``True`` when the COMPOSITION satisfies all local gates."""

        return not (
            self.out_of_template_paths
            or self.missing_feeder_audit_paths
            or self.unresolved_feeder_audit_paths
        )

    def require_ok(self) -> None:
        """Raise ``ValueError`` if any validation finding is present."""

        if self.ok:
            return
        raise ValueError(self.message())

    def message(self) -> str:
        """Return a concise validation error summary."""

        parts: list[str] = []
        if self.out_of_template_paths:
            parts.append(
                "out-of-template paths: "
                + ", ".join(sorted(self.out_of_template_paths))
            )
        if self.missing_feeder_audit_paths:
            parts.append(
                "missing FEEDER_AUDIT pointers: "
                + ", ".join(sorted(self.missing_feeder_audit_paths))
            )
        if self.unresolved_feeder_audit_paths:
            parts.append(
                "unresolved FEEDER_AUDIT pointers: "
                + ", ".join(sorted(self.unresolved_feeder_audit_paths))
            )
        return "; ".join(parts)


def parse_operational_template(
    template: OpenEHRTemplate | Mapping[str, Any] | str | os.PathLike[str],
) -> OpenEHRTemplate:
    """Parse a caller-supplied OPT/WebTemplate into allowed flat paths.

    EHRbase flat JSON is driven by the WebTemplate representation generated
    from an Operational Template. For tests and embedded deployments, callers
    may also provide a mapping with ``templateId`` and explicit
    ``allowed_paths``.
    """

    if isinstance(template, OpenEHRTemplate):
        return template
    if isinstance(template, Mapping):
        return _template_from_mapping(template)

    text, source = _read_template_text(template)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        if text.lstrip().startswith("<"):
            raise ValueError(
                "openEHR flat export requires an EHRbase WebTemplate JSON "
                "or a mapping with allowed_paths; XML OPT files do not carry "
                "the simplified flat paths needed for local validation."
            ) from exc
        raise ValueError(f"Invalid openEHR template JSON in {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"openEHR template in {source} must be a JSON object")
    return _template_from_mapping(payload, source=source)


def to_openehr_composition(
    entities: Iterable[Any],
    *,
    operational_template: OpenEHRTemplate | Mapping[str, Any] | str | os.PathLike[str],
    doc_id: str = "openmed-document",
    source_text: str | None = None,
    composer_name: str = "OpenMed",
    language: str = "en",
    territory: str = "US",
    time: str | datetime | None = None,
    vocabulary_key: str | None = None,
    bindings: Sequence[OpenEHRBinding] = DEFAULT_OPENEHR_BINDINGS,
    validate: bool = True,
) -> dict[str, Any]:
    """Serialize grounded entities into an EHRbase-compatible flat COMPOSITION.

    ``vocabulary_key`` is an explicit caller opt-in gate. When omitted, the
    exporter emits text and quantity values only, even if entity objects already
    carry terminology candidates.
    """

    template = parse_operational_template(operational_template)
    composition: dict[str, Any] = {
        "ctx/language": language,
        "ctx/territory": territory,
        "ctx/time": _format_time(time),
        "ctx/composer_name": composer_name,
    }
    counters: dict[str, int] = {}

    for entity in entities:
        span = _normalize_entity(
            entity, vocabulary_key=vocabulary_key, bindings=bindings
        )
        binding = binding_for_entity_kind(span.kind, bindings)
        index = counters.get(binding.kind, 0)
        counters[binding.kind] = index + 1

        text_path = binding.flat_text_path(template.template_id, index)
        composition[text_path] = span.text
        _add_feeder_audit(composition, text_path, doc_id=doc_id, span=span)

        if span.coding is not None:
            code_path = binding.flat_code_path(template.template_id, index)
            if code_path is not None:
                composition[f"{code_path}|code"] = span.coding.code
                composition[f"{code_path}|value"] = span.coding.display
                composition[f"{code_path}|terminology"] = span.coding.terminology
                _add_feeder_audit(composition, code_path, doc_id=doc_id, span=span)

        quantity_path = binding.flat_quantity_path(template.template_id, index)
        if span.value is not None and quantity_path is not None:
            if not span.unit:
                raise ValueError(
                    f"Entity {span.text!r} has a quantity value but no unit"
                )
            composition[f"{quantity_path}|magnitude"] = span.value
            composition[f"{quantity_path}|unit"] = span.unit
            _add_feeder_audit(composition, quantity_path, doc_id=doc_id, span=span)

    if validate:
        validate_openehr_composition(
            composition,
            template,
            source_text=source_text,
        ).require_ok()
    return composition


def validate_openehr_composition(
    composition: Mapping[str, Any],
    operational_template: OpenEHRTemplate | Mapping[str, Any] | str | os.PathLike[str],
    *,
    source_text: str | None = None,
) -> OpenEHRValidationResult:
    """Validate template conformance and source-offset FEEDER_AUDIT coverage."""

    template = parse_operational_template(operational_template)
    out_of_template = tuple(
        path for path in composition if not template.allows_path(path)
    )

    element_bases = _non_empty_element_bases(composition)
    missing: list[str] = []
    unresolved: list[str] = []
    for base_path in element_bases:
        pointer_path = f"{base_path}/_feeder_audit/originating_system_item_id:0|id"
        pointer = composition.get(pointer_path)
        if not isinstance(pointer, str) or not pointer:
            missing.append(base_path)
            continue
        if not _pointer_resolves(pointer, source_text):
            unresolved.append(pointer_path)

    return OpenEHRValidationResult(
        out_of_template_paths=tuple(sorted(out_of_template)),
        missing_feeder_audit_paths=tuple(sorted(missing)),
        unresolved_feeder_audit_paths=tuple(sorted(unresolved)),
    )


def extract_round_trip_coded_values(
    composition: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract coded-text and quantity values in deterministic flat-path order."""

    bases = sorted(
        {
            path[: -len("|code")]
            for path in composition
            if path.endswith("|code")
            and f"{path[: -len('|code')]}|value" in composition
        }
    )
    values: list[dict[str, Any]] = []
    for base in bases:
        values.append(
            {
                "path": base,
                "code": composition[f"{base}|code"],
                "value": composition[f"{base}|value"],
                "terminology": composition.get(f"{base}|terminology", ""),
            }
        )

    quantity_bases = sorted(
        {
            path[: -len("|magnitude")]
            for path in composition
            if path.endswith("|magnitude")
            and f"{path[: -len('|magnitude')]}|unit" in composition
        }
    )
    for base in quantity_bases:
        values.append(
            {
                "path": base,
                "magnitude": composition[f"{base}|magnitude"],
                "unit": composition[f"{base}|unit"],
            }
        )
    return values


def _template_from_mapping(
    payload: Mapping[str, Any],
    *,
    source: str = "mapping",
) -> OpenEHRTemplate:
    template_id = _template_id(payload)
    explicit_paths = payload.get("allowed_paths")
    if explicit_paths is not None:
        if not isinstance(explicit_paths, Sequence) or isinstance(
            explicit_paths, (str, bytes)
        ):
            raise ValueError("openEHR allowed_paths must be a sequence")
        allowed = {_normalize_flat_path(str(path)) for path in explicit_paths}
        element_paths = {
            _normalize_flat_path(path.split("|", 1)[0])
            for path in allowed
            if not path.startswith("ctx/")
        }
        return OpenEHRTemplate(
            template_id=template_id,
            allowed_paths=frozenset(allowed | _CONTEXT_PATHS),
            element_paths=frozenset(element_paths),
            source=source,
        )

    tree = payload.get("tree")
    if not isinstance(tree, Mapping):
        raise ValueError(
            "openEHR template must contain a WebTemplate 'tree' or allowed_paths"
        )

    allowed, element_paths = _allowed_paths_from_webtemplate(tree)
    return OpenEHRTemplate(
        template_id=template_id,
        allowed_paths=frozenset(allowed | _CONTEXT_PATHS),
        element_paths=frozenset(element_paths),
        source=source,
    )


def _template_id(payload: Mapping[str, Any]) -> str:
    for key in ("templateId", "template_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    tree = payload.get("tree")
    if isinstance(tree, Mapping):
        value = tree.get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("openEHR template is missing templateId")


def _allowed_paths_from_webtemplate(
    root: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    allowed: set[str] = set()
    element_paths: set[str] = set()

    def walk(node: Mapping[str, Any], prefix: str | None) -> None:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            return
        path = node_id if prefix is None else f"{prefix}/{node_id}"
        children = node.get("children")
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
            children = ()
        inputs = node.get("inputs")
        if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
            for item in inputs:
                if not isinstance(item, Mapping):
                    continue
                suffix = item.get("suffix")
                if isinstance(suffix, str) and suffix:
                    allowed.add(_normalize_flat_path(f"{path}|{suffix}"))
                else:
                    allowed.add(_normalize_flat_path(path))
                element_paths.add(_normalize_flat_path(path))
        elif not children and prefix is not None:
            allowed.add(_normalize_flat_path(path))
            element_paths.add(_normalize_flat_path(path))

        for child in children:
            if isinstance(child, Mapping):
                walk(child, path)

    walk(root, None)
    return allowed, element_paths


def _read_template_text(template: str | os.PathLike[str]) -> tuple[str, str]:
    if isinstance(template, os.PathLike):
        path = Path(template)
        return path.read_text(encoding="utf-8"), str(path)

    if template.lstrip().startswith(("{", "[", "<")):
        return template, "inline template"

    candidate = Path(template)
    if "\n" not in template and candidate.exists():
        return candidate.read_text(encoding="utf-8"), str(candidate)
    return template, "inline template"


def _normalize_entity(
    entity: Any,
    *,
    vocabulary_key: str | None,
    bindings: Sequence[OpenEHRBinding],
) -> _EntitySpan:
    row = flatten_entities([entity])[0]
    kind = str(row["entity_label"] or _first_scalar(_sources(entity), ("kind",)))
    if not kind:
        raise ValueError("openEHR export requires each entity to carry a label")
    binding = binding_for_entity_kind(kind, bindings)
    text = str(row["normalized_text"] or row["display"] or _first_text(entity))
    if not text:
        raise ValueError(f"Entity {kind!r} is missing text")
    start = _required_offset(row.get("start"), "start", text)
    end = _required_offset(row.get("end"), "end", text)
    if start > end:
        raise ValueError(f"Entity {text!r} has start offset after end offset")

    return _EntitySpan(
        kind=kind,
        text=text,
        start=start,
        end=end,
        coding=_coding_for_entity(entity, row, binding, vocabulary_key=vocabulary_key),
        value=_number_value(entity),
        unit=_first_scalar(_sources(entity), _UNIT_FIELDS),
    )


def _coding_for_entity(
    entity: Any,
    row: Mapping[str, Any],
    binding: OpenEHRBinding,
    *,
    vocabulary_key: str | None,
) -> OpenEHRCoding | None:
    if not vocabulary_key:
        return None

    system = str(row.get("system") or "")
    code = str(row.get("code") or "")
    display = str(row.get("display") or row.get("normalized_text") or "")
    if system and code:
        return OpenEHRCoding(
            system=system,
            code=code,
            display=display or code,
            terminology=_terminology_for(system),
        )

    candidates = _candidate_codings(entity)
    if not candidates:
        return None
    return _select_preferred_coding(candidates, binding.preferred_code_systems)


def _candidate_codings(entity: Any) -> list[OpenEHRCoding]:
    candidates: list[OpenEHRCoding] = []
    for source in _sources(entity):
        for field in _CANDIDATE_FIELDS:
            value = _value(source, field)
            if value is _MISSING or value is None or isinstance(value, (str, bytes)):
                continue
            for item in value:
                system = _first_scalar((item,), ("system", "code_system"))
                code = _first_scalar((item,), ("code", "code_value"))
                display = _first_scalar((item,), ("display", "label", "text"))
                if system and code:
                    candidates.append(
                        OpenEHRCoding(
                            system=system,
                            code=code,
                            display=display or code,
                            terminology=_terminology_for(system),
                        )
                    )
    return candidates


def _select_preferred_coding(
    candidates: Sequence[OpenEHRCoding],
    preferred_systems: Sequence[str],
) -> OpenEHRCoding:
    ranks = {
        _terminology_for(system).upper(): index
        for index, system in enumerate(preferred_systems)
    }
    return sorted(
        candidates,
        key=lambda coding: (
            ranks.get(coding.terminology.upper(), len(ranks)),
            coding.terminology,
            coding.code,
        ),
    )[0]


def _terminology_for(system: str) -> str:
    normalized = system.strip().upper()
    return _SYSTEM_TERMINOLOGY.get(normalized, system.strip() or "local")


def _first_text(entity: Any) -> str:
    return _first_scalar(_sources(entity), _TEXT_FIELDS)


def _number_value(entity: Any) -> int | float | None:
    value = _first_value(_sources(entity), _VALUE_FIELDS)
    if value is _MISSING or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
        return parsed
    return None


def _required_offset(value: Any, name: str, text: str) -> int:
    if value == "" or value is None:
        raise ValueError(
            f"Entity {text!r} is missing {name} offset required for FEEDER_AUDIT"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Entity {text!r} has invalid {name} offset {value!r}"
        ) from exc
    if parsed < 0:
        raise ValueError(f"Entity {text!r} has negative {name} offset")
    return parsed


def _add_feeder_audit(
    composition: dict[str, Any],
    base_path: str,
    *,
    doc_id: str,
    span: _EntitySpan,
) -> None:
    pointer = f"{doc_id}:{span.start}-{span.end}"
    audit_base = f"{base_path}/_feeder_audit"
    composition[f"{audit_base}/originating_system_audit|system_id"] = "openmed"
    composition[f"{audit_base}/originating_system_audit|version_id"] = (
        "source-offsets-v1"
    )
    composition[f"{audit_base}/originating_system_item_id:0|id"] = pointer
    composition[f"{audit_base}/originating_system_item_id:0|issuer"] = "openmed"
    composition[f"{audit_base}/originating_system_item_id:0|assigner"] = doc_id
    composition[f"{audit_base}/originating_system_item_id:0|type"] = "SOURCE_SPAN"


def _non_empty_element_bases(composition: Mapping[str, Any]) -> tuple[str, ...]:
    bases: set[str] = set()
    for path, value in composition.items():
        if value in (None, "") or path.startswith("ctx/") or "/_feeder_audit/" in path:
            continue
        bases.add(_element_base(path))
    return tuple(sorted(bases))


def _pointer_resolves(pointer: str, source_text: str | None) -> bool:
    match = _SPAN_POINTER_RE.match(pointer)
    if match is None:
        return False
    start = int(match.group("start"))
    end = int(match.group("end"))
    if start > end:
        return False
    return source_text is None or end <= len(source_text)


def _split_feeder_audit_path(path: str) -> tuple[str, str] | None:
    marker = "/_feeder_audit/"
    if marker not in path:
        return None
    base_path, feeder_suffix = path.split(marker, 1)
    return base_path, feeder_suffix


def _element_base(path: str) -> str:
    return path.split("|", 1)[0]


def _normalize_flat_path(path: str) -> str:
    return _INDEX_RE.sub("", path)


def _format_time(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _sources(entity: Any) -> tuple[Any, ...]:
    sources: list[Any] = [entity]
    for name in ("metadata", "meta", "context", "clinical_context"):
        nested = _value(entity, name)
        if nested is not _MISSING and nested is not None:
            sources.append(nested)
    return tuple(sources)


def _first_scalar(sources: Iterable[Any], fields: Sequence[str]) -> str:
    value = _first_value(sources, fields)
    if value is _MISSING or value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    return ""


def _first_value(sources: Iterable[Any], fields: Sequence[str]) -> Any:
    for source in sources:
        for field in fields:
            value = _value(source, field)
            if value is not _MISSING:
                return value
    return _MISSING


def _value(source: Any, field: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(field, _MISSING)
    return getattr(source, field, _MISSING)
