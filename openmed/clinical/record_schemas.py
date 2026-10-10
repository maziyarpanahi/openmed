"""Bundled JSON Schemas for the guarded clinical record families.

The runtime contracts in :mod:`openmed.clinical.brief`,
:mod:`openmed.clinical.evidence_packet`, :mod:`openmed.clinical.nli`, and
:mod:`openmed.clinical.sdoh_evidence` serialize PHI-free records: controlled
codes, counts, offsets, opaque references, and digests. Non-Python consumers
cannot import those validators, so this module publishes the same contracts as
Draft 2020-12 JSON Schema files under ``openmed/core/schemas/json/`` and pins
each one in a committed fingerprint snapshot.

The schemas describe *serialized records*, not the dataclasses that produce
them. This module never changes record contents, record versions, or producer
behavior; it only loads, fingerprints, exports, compares, and (optionally)
validates them. The clinical registry is deliberately separate from
:data:`openmed.core.schemas.span.SCHEMA_NAMES`: those core schemas share one
``schema_version`` constant, while the clinical records are independently
versioned (for example an evidence packet is version 2 and an NLI verification
record carries no version field at all).

Validating a record needs the optional ``jsonschema`` dependency. Loading,
exporting, and fingerprinting never do.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from openmed.core.capabilities import MissingOptionalDependencyError
from openmed.core.schemas.span import (
    SCHEMA_PACKAGE,
    SchemaDriftResult,
    load_schema_snapshot,
    schema_fingerprint,
)

CLINICAL_RECORD_SCHEMA_NAMES: Final[tuple[str, ...]] = (
    "brief_audit",
    "brief_response",
    "evidence_packet",
    "nli_verification",
    "sdoh_evidence_report",
)

CLINICAL_RECORD_SCHEMA_BUNDLE_VERSION: Final = 1

CLINICAL_RECORD_SCHEMA_PACKAGE: Final = SCHEMA_PACKAGE

CLINICAL_RECORD_SCHEMA_SNAPSHOT_NAME: Final = "clinical-record-fingerprints.json"

CLINICAL_RECORD_SCHEMA_FILENAMES: Final[Mapping[str, str]] = MappingProxyType(
    {name: f"clinical_{name}.schema.json" for name in CLINICAL_RECORD_SCHEMA_NAMES}
)

CLINICAL_RECORD_SCHEMA_IDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        name: f"https://openmed.life/schemas/clinical_{name}.schema.json"
        for name in CLINICAL_RECORD_SCHEMA_NAMES
    }
)

_OFFSET_COLLECTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "brief_audit": "citations",
        "brief_response": "citations",
        "evidence_packet": "references",
        "sdoh_evidence_report": "findings",
    }
)


class ClinicalRecordSchemaError(ValueError):
    """Raised when a guarded clinical record violates its JSON Schema."""


def _normalise_name(name: str) -> str:
    """Return a known clinical record schema name or raise ``KeyError``."""

    if type(name) is not str or name not in CLINICAL_RECORD_SCHEMA_NAMES:
        raise KeyError("unknown clinical record schema")
    return name


def clinical_record_schema_filename(name: str) -> str:
    """Return the bundled file name for one clinical record schema."""

    return CLINICAL_RECORD_SCHEMA_FILENAMES[_normalise_name(name)]


def clinical_record_schema_id(name: str) -> str:
    """Return the canonical ``$id`` for one clinical record schema."""

    return CLINICAL_RECORD_SCHEMA_IDS[_normalise_name(name)]


def load_clinical_record_schema(name: str) -> dict[str, Any]:
    """Load one bundled clinical record schema by logical name."""

    schema_name = _normalise_name(name)
    resource = resources.files(CLINICAL_RECORD_SCHEMA_PACKAGE).joinpath(
        CLINICAL_RECORD_SCHEMA_FILENAMES[schema_name]
    )
    with resource.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if "schema_version" not in schema:
        raise ClinicalRecordSchemaError(
            f"{schema_name} schema is missing schema_version"
        )
    return schema


def load_all_clinical_record_schemas() -> dict[str, dict[str, Any]]:
    """Load every bundled clinical record schema."""

    return {
        name: load_clinical_record_schema(name) for name in CLINICAL_RECORD_SCHEMA_NAMES
    }


def load_clinical_record_schema_bundle() -> dict[str, Any]:
    """Return the bundle version plus every parsed clinical record schema."""

    return {
        "schema_version": CLINICAL_RECORD_SCHEMA_BUNDLE_VERSION,
        "schemas": load_all_clinical_record_schemas(),
    }


def clinical_record_schema_json(name: str) -> str:
    """Return one schema as canonical sorted JSON text.

    The bundled files keep the repository's schema layout (compact property
    definitions, alphabetical ``required`` entries). This helper is the
    canonical text form for callers that need to hand a schema to another
    process; it is not used to rewrite the shipped files.
    """

    payload = json.dumps(load_clinical_record_schema(name), indent=2, sort_keys=True)
    return payload + "\n"


def clinical_record_schema_fingerprint(name: str) -> str:
    """Return the deterministic fingerprint for one clinical record schema."""

    return schema_fingerprint(load_clinical_record_schema(name))


def clinical_record_schema_fingerprints() -> dict[str, str]:
    """Return the fingerprint of every bundled clinical record schema."""

    return {
        name: clinical_record_schema_fingerprint(name)
        for name in CLINICAL_RECORD_SCHEMA_NAMES
    }


def _resolve_schema_ref(
    schema: Mapping[str, Any], node: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Follow local ``$ref`` pointers so drift sees the record object."""

    seen: set[str] = set()
    while isinstance(node.get("$ref"), str):
        ref = node["$ref"]
        if not ref.startswith("#/") or ref in seen:
            break
        seen.add(ref)
        target: Any = schema
        for part in ref[2:].split("/"):
            if not isinstance(target, Mapping) or part not in target:
                target = None
                break
            target = target[part]
        if not isinstance(target, Mapping):
            break
        node = target
    return node


def _schema_state(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Describe one schema the way the committed drift snapshot does.

    Array-rooted record schemas (an NLI verification is a list) keep their
    required names and properties on the item schema, so the fallback mirrors
    the core snapshot shape instead of reporting an empty field set.
    """

    root: Mapping[str, Any] = schema
    if root.get("type") == "array" and isinstance(root.get("items"), Mapping):
        root = root["items"]
    items = _resolve_schema_ref(schema, root)
    properties = items.get("properties")
    required = items.get("required")
    return {
        "fingerprint": schema_fingerprint(schema),
        "properties": sorted(str(key) for key in properties or ()),
        "required": sorted(str(key) for key in required or ()),
        "schema_version": int(schema.get("schema_version", 0)),
    }


def clinical_record_schema_snapshot() -> dict[str, dict[str, Any]]:
    """Build the drift snapshot for every clinical record schema."""

    return {
        name: _schema_state(schema)
        for name, schema in sorted(
            load_all_clinical_record_schemas().items(), key=lambda item: item[0]
        )
    }


def load_clinical_record_snapshot(
    path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the committed clinical record fingerprint snapshot."""

    if path is not None:
        return load_schema_snapshot(path)
    resource = resources.files(CLINICAL_RECORD_SCHEMA_PACKAGE).joinpath(
        CLINICAL_RECORD_SCHEMA_SNAPSHOT_NAME
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_clinical_record_drift(
    name: str,
    current_schema: Mapping[str, Any] | None = None,
    *,
    snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> SchemaDriftResult:
    """Compare one clinical record schema against a saved snapshot.

    A field removal without a ``schema_version`` bump is reported as a breaking
    change, matching :func:`openmed.core.schemas.span.compare_schema_drift`
    while keeping the clinical registry independently versioned.
    """

    schema_name = _normalise_name(name)
    schema = (
        current_schema
        if current_schema is not None
        else load_clinical_record_schema(schema_name)
    )
    snapshot_payload = (
        snapshot if snapshot is not None else load_clinical_record_snapshot()
    )
    snapshot_state = snapshot_payload.get(schema_name) or {}
    state = _schema_state(schema)
    current_version = int(schema.get("schema_version", 0))
    snapshot_version = snapshot_state.get("schema_version")
    version_bumped = bool(snapshot_state) and current_version > int(
        snapshot_version or 0
    )
    removed_required = tuple(
        sorted(set(snapshot_state.get("required") or ()) - set(state["required"]))
    )
    removed_properties = tuple(
        sorted(set(snapshot_state.get("properties") or ()) - set(state["properties"]))
    )
    return SchemaDriftResult(
        schema_name=schema_name,
        schema_version=current_version,
        snapshot_version=int(snapshot_version)
        if snapshot_version is not None
        else None,
        fingerprint=schema_fingerprint(schema),
        snapshot_fingerprint=snapshot_state.get("fingerprint"),
        version_bumped=version_bumped,
        breaking_change=bool(snapshot_state)
        and not version_bumped
        and bool(removed_required or removed_properties),
        removed_required=removed_required,
        removed_properties=removed_properties,
    )


def compare_all_clinical_record_drift(
    *,
    schemas: Mapping[str, Mapping[str, Any]] | None = None,
    snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, SchemaDriftResult]:
    """Compare every clinical record schema against a saved snapshot."""

    bundle = schemas if schemas is not None else load_all_clinical_record_schemas()
    snapshot_payload = (
        snapshot if snapshot is not None else load_clinical_record_snapshot()
    )
    return {
        name: compare_clinical_record_drift(
            name, bundle.get(name), snapshot=snapshot_payload
        )
        for name in CLINICAL_RECORD_SCHEMA_NAMES
    }


def _mappings(value: Any) -> list[Mapping[str, Any]]:
    """Return only the mapping entries of a serialized record field."""

    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, Mapping)]


def _offset_pairs(name: str, record: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return ``(location, start, end)`` offsets for one clinical record."""

    collection = _OFFSET_COLLECTIONS.get(name)
    if collection is None:
        return []
    pairs: list[tuple[str, Any, Any]] = []
    for index, entry in enumerate(_mappings(record.get(collection))):
        location = f"{collection}[{index}]"
        if collection == "citations":
            pairs.append(
                (
                    f"{location}.source",
                    entry.get("source_start"),
                    entry.get("source_end"),
                )
            )
            pairs.append(
                (
                    f"{location}.output",
                    entry.get("output_start"),
                    entry.get("output_end"),
                )
            )
            continue
        if collection == "references":
            pairs.append((location, entry.get("start"), entry.get("end")))
            continue
        span = entry.get("source_span")
        if isinstance(span, Sequence) and len(span) == 2:
            pairs.append((f"{location}.source_span", span[0], span[1]))
    return pairs


def _check_offset_invariants(name: str, record: Mapping[str, Any]) -> None:
    """Reject inverted offsets that JSON Schema cannot express."""

    for location, start, end in _offset_pairs(name, record):
        if isinstance(start, int) and isinstance(end, int) and start > end:
            raise ClinicalRecordSchemaError(
                f"{name} record has an inverted offset at {location}: {start}:{end}"
            )
    metrics = record.get("metrics", {})
    if name in {"brief_audit", "brief_response"} and "reviewed_evidence" in metrics:
        from openmed.clinical.reviewed_local_evidence import ReviewedLocalEvidence

        invalid = False
        try:
            ReviewedLocalEvidence.from_dict(metrics["reviewed_evidence"])
        except (TypeError, ValueError):
            invalid = True
        if invalid:
            raise ClinicalRecordSchemaError(
                "reviewed-local metadata violates its contract"
            )
    if name in {"brief_audit", "brief_response"} and "generation_contract" in record:
        bindings = record["claim_bindings"]
        citations = record["citations"]
        if len(bindings) != len(citations) or any(
            binding["claim_index"] != index or citation["claim_index"] != index
            for index, (binding, citation) in enumerate(zip(bindings, citations))
        ):
            raise ClinicalRecordSchemaError(
                "claim bindings do not match the citation sequence"
            )


def validate_clinical_record(name: str, record: Any) -> None:
    """Validate one serialized clinical record against its bundled schema.

    Args:
        name: Logical record schema name, for example ``"evidence_packet"``.
        record: A serialized record, typically the ``to_dict()`` result of the
            matching runtime contract.

    Raises:
        KeyError: If ``name`` is not a bundled clinical record schema.
        MissingOptionalDependencyError: If ``jsonschema`` is not installed.
        ClinicalRecordSchemaError: If the record violates the schema or stores
            an inverted offset pair.
    """

    schema_name = _normalise_name(name)
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise MissingOptionalDependencyError(
            package="jsonschema",
            feature="clinical record schema validation",
            extra="dev",
        ) from exc
    schema = load_clinical_record_schema(schema_name)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    first = next(validator.iter_errors(record), None)
    if first is not None:
        known_fields: set[str] = set()
        pending: list[Any] = [schema]
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                known_fields.update(node.get("properties", {}))
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)
        location = (
            "/".join(
                str(part) if type(part) is int or part in known_fields else "*"
                for part in first.absolute_path
            )
            or "$"
        )
        raise ClinicalRecordSchemaError(
            f"{schema_name} record violates its schema at {location}"
        )
    if isinstance(record, Mapping):
        _check_offset_invariants(schema_name, record)


def write_clinical_record_snapshot(path: str | Path) -> Path:
    """Write the clinical record fingerprint snapshot to ``path``.

    The snapshot is rebuilt from the bundled schemas, so re-exporting an
    unchanged tree reproduces the committed file byte for byte.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(clinical_record_schema_snapshot(), indent=2, sort_keys=True)
    target.write_bytes((payload + "\n").encode("utf-8"))
    return target
