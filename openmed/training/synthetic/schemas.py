"""Versioned JSON Schemas for the synthetic training-data records.

The generators in this package emit plain dictionaries, which keeps the record
contracts implicit and easy to break silently. This module publishes one
deterministic `JSON Schema`_ per record family the package already emits, so a
consumer can validate a synthetic dataset offline before training on it.

Published schemas:

* ``locale_phi_example`` -- :meth:`LocalePhiExample.to_training_item
  <openmed.training.synthetic.locale_phi.LocalePhiExample.to_training_item>`.
* ``burned_in_annotation`` -- :meth:`BurnedInTextBox.to_dict
  <openmed.training.synthetic.burned_in.BurnedInTextBox.to_dict>`; the example
  wrapper also carries a live Pillow image, so the schema covers the JSON-safe
  annotation records.
* ``social_history_example`` -- :meth:`SyntheticSocialHistory.to_dict
  <openmed.training.synthetic.social_history.SyntheticSocialHistory.to_dict>`.
* ``section_label_record`` -- the JSONL records written by
  :func:`build_section_dataset
  <openmed.training.synthetic.section_labels.build_section_dataset>`.
* ``translation_augmented_example`` -- :meth:`TranslationAugmentedExample
  .to_training_item
  <openmed.training.synthetic.translation_augment.TranslationAugmentedExample.to_training_item>`.

Every schema targets JSON Schema draft 2020-12, carries a versioned ``$id``
(:data:`SYNTHETIC_RECORD_SCHEMA_VERSION`), is closed with
``additionalProperties: false`` for every object whose keys this package
controls, and only references local ``#/$defs`` entries, so it resolves without
network access. Each one also requires the generator's synthetic markers
(``is_synthetic``/``synthetic``), its ``contains_real_phi: false`` declaration
and its ``synthetic_source``, which makes a record that claims real PHI fail
validation instead of entering a training set unnoticed.

Schemas describe structure only: the enums and constants are the runtime
constants the generators use, and no generated example values are embedded.
Because JSON Schema cannot express the cross-field ``end > start`` invariant,
:func:`validate_record` pairs schema validation with the offset invariants of
:mod:`openmed.training.synthetic.offset_projection`.

Example:
    >>> from openmed.training.synthetic.schemas import (
    ...     SYNTHETIC_RECORD_SCHEMA_NAMES,
    ...     validate_record,
    ... )
    >>> "section_label_record" in SYNTHETIC_RECORD_SCHEMA_NAMES
    True
    >>> validate_record(
    ...     "burned_in_annotation",
    ...     {
    ...         "bbox": [0, 0, 10, 10],
    ...         "end": 6,
    ...         "font_name": "DejaVuSans.ttf",
    ...         "font_size": 18,
    ...         "label": "PERSON",
    ...         "metadata": {"synthetic": True},
    ...         "start": 0,
    ...         "text": "SYNTH",
    ...     },
    ... )

.. _JSON Schema: https://json-schema.org/draft/2020-12/schema
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from openmed.core.capabilities import MissingOptionalDependencyError

from .burned_in import BURNED_IN_LABELS
from .locale_phi import LOCALE_PHI_LABELS, SUPPORTED_LOCALE_PHI_LANGUAGES
from .section_labels import (
    DOCUMENT_TYPES,
    SECTION_LABELS,
    SECTION_RECORD_SCHEMA_VERSION,
    SYNTHETIC_SECTION_LICENSE,
    SYNTHETIC_SECTION_SOURCE,
)
from .social_history import SOCIAL_HISTORY_CATEGORIES, SYNTHETIC_SOCIAL_HISTORY_SOURCE
from .translation_augment import SYNTHETIC_SOURCE

_LOCALE_PHI_EXAMPLE: Final = "locale_phi_example"
_BURNED_IN_ANNOTATION: Final = "burned_in_annotation"
_SOCIAL_HISTORY_EXAMPLE: Final = "social_history_example"
_SECTION_LABEL_RECORD: Final = "section_label_record"
_TRANSLATION_AUGMENTED_EXAMPLE: Final = "translation_augmented_example"

_JSON_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_ID_BASE: Final = "https://openmed.ai/schemas/training/synthetic"
_SECTION_RECORD_ID_PATTERN: Final = r"^synthetic-section-[0-9]+-[0-9]{4}$"
_BIO_TAG_PATTERN: Final = r"^(?:[BI]-[a-z0-9_]+|O)$"

SYNTHETIC_RECORD_SCHEMA_VERSION: Final = 1
"""Version stamped into every published synthetic record ``$id``."""

SYNTHETIC_RECORD_SCHEMA_NAMES: Final[tuple[str, ...]] = (
    _LOCALE_PHI_EXAMPLE,
    _BURNED_IN_ANNOTATION,
    _SOCIAL_HISTORY_EXAMPLE,
    _SECTION_LABEL_RECORD,
    _TRANSLATION_AUGMENTED_EXAMPLE,
)
"""Record families that have a published schema, in export order."""


class SyntheticRecordSchemaError(ValueError):
    """Raised when a synthetic training record violates its published contract.

    The message names the failing JSON pointer and the violated invariant. It
    never repeats record values beyond the offsets that failed, so it is safe to
    surface in a training pipeline log.
    """


def record_schema_id(name: str) -> str:
    """Return the versioned ``$id`` published for the ``name`` record schema.

    Args:
        name: One of :data:`SYNTHETIC_RECORD_SCHEMA_NAMES`.

    Returns:
        An absolute ``https://openmed.ai/schemas/training/synthetic/...`` URL
        containing the :data:`SYNTHETIC_RECORD_SCHEMA_VERSION`.

    Raises:
        KeyError: If ``name`` is not a published record schema.
    """
    if name not in SYNTHETIC_RECORD_SCHEMA_NAMES:
        raise KeyError(f"unknown synthetic record schema: {name!r}")
    slug = name.replace("_", "-")
    version = SYNTHETIC_RECORD_SCHEMA_VERSION
    return f"{_SCHEMA_ID_BASE}/{slug}-v{version}.schema.json"


SYNTHETIC_RECORD_SCHEMA_IDS: Final[Mapping[str, str]] = {
    name: record_schema_id(name) for name in SYNTHETIC_RECORD_SCHEMA_NAMES
}
"""Published ``$id`` per record family."""


def _span_definition(labels: Sequence[str] | None = None) -> dict[str, Any]:
    label_schema: dict[str, Any] = {"type": "string", "minLength": 1}
    if labels is not None:
        label_schema = {"enum": list(labels)}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["end", "label", "metadata", "start", "text"],
        "properties": {
            "end": {"type": "integer", "minimum": 1},
            "label": label_schema,
            "metadata": {"type": "object"},
            "start": {"type": "integer", "minimum": 0},
            "text": {"type": "string", "minLength": 1},
        },
    }


def _locale_phi_example_schema() -> dict[str, Any]:
    return {
        "$schema": _JSON_SCHEMA_DIALECT,
        "$id": record_schema_id(_LOCALE_PHI_EXAMPLE),
        "title": "OpenMed synthetic locale PHI example",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "is_synthetic",
            "labels",
            "language",
            "locale",
            "metadata",
            "synthetic_source",
            "text",
        ],
        "properties": {
            "is_synthetic": {"const": True},
            "labels": {"type": "array", "items": {"$ref": "#/$defs/span"}},
            "language": {"enum": list(SUPPORTED_LOCALE_PHI_LANGUAGES)},
            "locale": {"type": "string", "minLength": 1},
            "metadata": {"$ref": "#/$defs/metadata"},
            "synthetic_source": {"const": "locale_phi"},
            "text": {"type": "string", "minLength": 1},
        },
        "$defs": {
            "metadata": {
                "type": "object",
                "required": ["augmentation_only", "contains_real_phi", "synthetic"],
                "properties": {
                    "augmentation_only": {"const": True},
                    "contains_real_phi": {"const": False},
                    "synthetic": {"const": True},
                },
            },
            "span": _span_definition(LOCALE_PHI_LABELS),
        },
    }


def _burned_in_annotation_schema() -> dict[str, Any]:
    return {
        "$schema": _JSON_SCHEMA_DIALECT,
        "$id": record_schema_id(_BURNED_IN_ANNOTATION),
        "title": "OpenMed synthetic burned-in annotation",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "bbox",
            "end",
            "font_name",
            "font_size",
            "label",
            "metadata",
            "start",
            "text",
        ],
        "properties": {
            "bbox": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "minItems": 4,
                "maxItems": 4,
            },
            "end": {"type": "integer", "minimum": 1},
            "font_name": {"type": "string", "minLength": 1},
            "font_size": {"type": "integer", "minimum": 1},
            "label": {"enum": list(BURNED_IN_LABELS)},
            "metadata": {"$ref": "#/$defs/metadata"},
            "start": {"type": "integer", "minimum": 0},
            "text": {"type": "string", "minLength": 1},
        },
        "$defs": {
            "metadata": {
                "type": "object",
                "required": ["synthetic"],
                "properties": {"synthetic": {"const": True}},
            },
        },
    }


def _social_history_example_schema() -> dict[str, Any]:
    return {
        "$schema": _JSON_SCHEMA_DIALECT,
        "$id": record_schema_id(_SOCIAL_HISTORY_EXAMPLE),
        "title": "OpenMed synthetic social-history example",
        "type": "object",
        "additionalProperties": False,
        "required": ["events", "metadata", "text"],
        "properties": {
            "events": {"type": "array", "items": {"$ref": "#/$defs/event"}},
            "metadata": {"$ref": "#/$defs/metadata"},
            "text": {"type": "string", "minLength": 1},
        },
        "$defs": {
            "event": {
                "type": "object",
                "additionalProperties": False,
                "required": ["attributes", "category", "span", "status", "value"],
                "properties": {
                    "attributes": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["extent", "temporality"],
                        "properties": {
                            "extent": {"type": ["string", "null"]},
                            "temporality": {"type": "string", "minLength": 1},
                        },
                    },
                    "category": {"enum": list(SOCIAL_HISTORY_CATEGORIES)},
                    "span": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "status": {"type": "string", "minLength": 1},
                    "value": {"type": "string", "minLength": 1},
                },
            },
            "metadata": {
                "type": "object",
                "required": ["restricted_data", "source", "synthetic"],
                "properties": {
                    "restricted_data": {"const": False},
                    "source": {"const": SYNTHETIC_SOCIAL_HISTORY_SOURCE},
                    "synthetic": {"const": True},
                },
            },
        },
    }


def _section_label_record_schema() -> dict[str, Any]:
    return {
        "$schema": _JSON_SCHEMA_DIALECT,
        "$id": record_schema_id(_SECTION_LABEL_RECORD),
        "title": "OpenMed synthetic section-label record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "contains_real_phi",
            "doc_type",
            "id",
            "labels",
            "license",
            "metadata",
            "record_id",
            "restricted_data",
            "schema_version",
            "section_tags",
            "sections",
            "source",
            "synthetic",
            "text",
            "token_offsets",
            "tokens",
        ],
        "properties": {
            "contains_real_phi": {"const": False},
            "doc_type": {"enum": list(DOCUMENT_TYPES)},
            "id": {"type": "string", "pattern": _SECTION_RECORD_ID_PATTERN},
            "labels": {"type": "array", "items": {"$ref": "#/$defs/bio_tag"}},
            "license": {"const": SYNTHETIC_SECTION_LICENSE},
            "metadata": {"$ref": "#/$defs/metadata"},
            "record_id": {"type": "string", "pattern": _SECTION_RECORD_ID_PATTERN},
            "restricted_data": {"const": False},
            "schema_version": {"const": SECTION_RECORD_SCHEMA_VERSION},
            "section_tags": {"type": "array", "items": {"$ref": "#/$defs/bio_tag"}},
            "sections": {"type": "array", "items": {"$ref": "#/$defs/section"}},
            "source": {"const": SYNTHETIC_SECTION_SOURCE},
            "synthetic": {"const": True},
            "text": {"type": "string", "minLength": 1},
            "token_offsets": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "tokens": {"type": "array", "items": {"type": "string", "minLength": 1}},
        },
        "$defs": {
            "bio_tag": {"type": "string", "pattern": _BIO_TAG_PATTERN},
            "metadata": {
                "type": "object",
                "required": [
                    "contains_real_phi",
                    "restricted_data",
                    "synthetic",
                    "synthetic_source",
                ],
                "properties": {
                    "contains_real_phi": {"const": False},
                    "restricted_data": {"const": False},
                    "synthetic": {"const": True},
                    "synthetic_source": {"const": SYNTHETIC_SECTION_SOURCE},
                },
            },
            "section": {
                "type": "object",
                "additionalProperties": False,
                "required": ["end", "header", "label", "start"],
                "properties": {
                    "end": {"type": "integer", "minimum": 1},
                    "header": {"type": "string"},
                    "label": {"enum": list(SECTION_LABELS)},
                    "start": {"type": "integer", "minimum": 0},
                },
            },
        },
    }


def _translation_augmented_example_schema() -> dict[str, Any]:
    return {
        "$schema": _JSON_SCHEMA_DIALECT,
        "$id": record_schema_id(_TRANSLATION_AUGMENTED_EXAMPLE),
        "title": "OpenMed synthetic translation-augmented example",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "gold_spans",
            "id",
            "is_synthetic",
            "labels",
            "language",
            "metadata",
            "source_id",
            "source_language",
            "synthetic_source",
            "text",
        ],
        "properties": {
            "gold_spans": {"type": "array", "items": {"$ref": "#/$defs/span"}},
            "id": {"type": "string", "minLength": 1},
            "is_synthetic": {"const": True},
            "labels": {"type": "array", "items": {"$ref": "#/$defs/span"}},
            "language": {"type": "string", "minLength": 1},
            "metadata": {"$ref": "#/$defs/metadata"},
            "source_id": {"type": "string", "minLength": 1},
            "source_language": {"type": "string", "minLength": 1},
            "synthetic_source": {"const": SYNTHETIC_SOURCE},
            "text": {"type": "string", "minLength": 1},
        },
        "$defs": {
            "metadata": {
                "type": "object",
                "required": [
                    "contains_real_phi",
                    "provenance",
                    "synthetic",
                    "synthetic_source",
                ],
                "properties": {
                    "contains_real_phi": {"const": False},
                    "provenance": {"$ref": "#/$defs/provenance"},
                    "synthetic": {"const": True},
                    "synthetic_source": {"const": SYNTHETIC_SOURCE},
                },
            },
            "provenance": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_id",
                    "source_language",
                    "source_text_hash",
                    "transform",
                ],
                "properties": {
                    "source_id": {"type": "string", "minLength": 1},
                    "source_language": {"type": "string", "minLength": 1},
                    "source_text_hash": {"type": "string", "minLength": 1},
                    "transform": {"type": "string", "minLength": 1},
                },
            },
            "span": _span_definition(),
        },
    }


_SCHEMA_BUILDERS: Final[Mapping[str, Callable[[], dict[str, Any]]]] = {
    _LOCALE_PHI_EXAMPLE: _locale_phi_example_schema,
    _BURNED_IN_ANNOTATION: _burned_in_annotation_schema,
    _SOCIAL_HISTORY_EXAMPLE: _social_history_example_schema,
    _SECTION_LABEL_RECORD: _section_label_record_schema,
    _TRANSLATION_AUGMENTED_EXAMPLE: _translation_augmented_example_schema,
}


def export_record_schema(name: str) -> dict[str, Any]:
    """Return a fresh schema mapping for the ``name`` record family.

    Args:
        name: One of :data:`SYNTHETIC_RECORD_SCHEMA_NAMES`.

    Returns:
        A new ``dict`` on every call, so callers may mutate the result without
        changing what the next caller sees.

    Raises:
        KeyError: If ``name`` is not a published record schema.
    """
    try:
        builder = _SCHEMA_BUILDERS[name]
    except KeyError:
        raise KeyError(f"unknown synthetic record schema: {name!r}") from None
    return builder()


def export_record_schemas() -> dict[str, dict[str, Any]]:
    """Return every published schema keyed by record family name."""
    return {name: export_record_schema(name) for name in SYNTHETIC_RECORD_SCHEMA_NAMES}


def export_record_schema_json(name: str) -> str:
    """Return the canonical JSON text of one published schema.

    The text is deterministic across runs and platforms (sorted keys, no
    insignificant whitespace, ``ensure_ascii=False``), which is what makes
    :func:`record_schema_fingerprint` a stable drift signal.
    """
    return _canonical_json(export_record_schema(name))


def export_record_schemas_json() -> str:
    """Return the canonical JSON bundle of every published schema.

    The bundle wraps the individual schemas as ``{"record_schema_version": N,
    "schemas": {...}}`` so a consumer can publish or diff all contracts at once.
    """
    bundle = {
        "record_schema_version": SYNTHETIC_RECORD_SCHEMA_VERSION,
        "schemas": export_record_schemas(),
    }
    return _canonical_json(bundle)


def record_schema_fingerprint(name: str) -> str:
    """Return the ``sha256:<hex>`` fingerprint of one canonical schema export."""
    digest = hashlib.sha256(export_record_schema_json(name).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def export_record_schema_fingerprints() -> dict[str, str]:
    """Return the canonical fingerprint of every published schema."""
    return {
        name: record_schema_fingerprint(name) for name in SYNTHETIC_RECORD_SCHEMA_NAMES
    }


def write_record_schemas(directory: str | Path) -> tuple[Path, ...]:
    """Write one ``<name>.schema.json`` file per published schema.

    Files are written as UTF-8 with LF newlines, two-space indentation and
    sorted keys, so the same package version always produces identical bytes.

    Args:
        directory: Destination directory, created when it does not exist.

    Returns:
        The written paths, ordered like :data:`SYNTHETIC_RECORD_SCHEMA_NAMES`.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in SYNTHETIC_RECORD_SCHEMA_NAMES:
        schema = export_record_schema(name)
        payload = json.dumps(
            schema,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        path = destination / f"{name.replace('_', '-')}.schema.json"
        path.write_bytes(f"{payload}\n".encode("utf-8"))
        written.append(path)
    return tuple(written)


def validate_record(name: str, record: Mapping[str, Any]) -> None:
    """Validate a synthetic training record against its published contract.

    The record is checked against its JSON Schema and then against the offset
    invariants JSON Schema cannot express: every span must satisfy
    ``start >= 0`` and ``end > start``, and every burned-in box must be
    non-negative and ordered.

    Args:
        name: One of :data:`SYNTHETIC_RECORD_SCHEMA_NAMES`.
        record: A record mapping produced by the matching generator.

    Raises:
        SyntheticRecordSchemaError: If the record violates the schema or an
            offset invariant.
        KeyError: If ``name`` is not a published record schema.
        MissingOptionalDependencyError: If ``jsonschema`` is not installed.
    """
    schema = export_record_schema(name)
    validator_class = _draft202012_validator()
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    errors = sorted(
        validator.iter_errors(dict(record)),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise SyntheticRecordSchemaError(
            f"{name} record violates its schema at {location}: {first.message}"
        )
    _check_offset_invariants(name, record)


def _check_offset_invariants(name: str, record: Mapping[str, Any]) -> None:
    for path, start, end in _iter_span_offsets(record, ""):
        location = path or "<root>"
        if start < 0 or end < 0:
            raise SyntheticRecordSchemaError(
                f"{name} record has a negative offset at {location}: {start}:{end}"
            )
        if end <= start:
            raise SyntheticRecordSchemaError(
                f"{name} record has an inverted offset at {location}: {start}:{end}"
            )
    for path, x_min, y_min, x_max, y_max in _iter_boxes(record, ""):
        location = path or "<root>"
        if x_max < x_min or y_max < y_min:
            raise SyntheticRecordSchemaError(
                f"{name} record has an inverted bounding box at {location}: "
                f"{x_min},{y_min},{x_max},{y_max}"
            )


def _iter_span_offsets(value: Any, path: str) -> Iterator[tuple[str, int, int]]:
    if isinstance(value, Mapping):
        start = value.get("start")
        end = value.get("end")
        if _is_int(start) and _is_int(end) and "start" in value and "end" in value:
            yield path, start, end
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if key == "span" and _is_pair(item):
                yield child, item[0], item[1]
                continue
            if key == "token_offsets" and _is_sequence(item):
                for index, pair in enumerate(item):
                    if _is_pair(pair):
                        yield f"{child}[{index}]", pair[0], pair[1]
                continue
            yield from _iter_span_offsets(item, child)
    elif _is_sequence(value) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from _iter_span_offsets(item, f"{path}[{index}]")


def _iter_boxes(value: Any, path: str) -> Iterator[tuple[str, int, int, int, int]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if key == "bbox" and _is_sequence(item) and len(item) == 4:
                if all(_is_int(part) for part in item):
                    yield child, item[0], item[1], item[2], item[3]
                continue
            yield from _iter_boxes(item, child)
    elif _is_sequence(value) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from _iter_boxes(item, f"{path}[{index}]")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_pair(value: Any) -> bool:
    return (
        _is_sequence(value) and len(value) == 2 and all(_is_int(part) for part in value)
    )


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _draft202012_validator() -> Any:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MissingOptionalDependencyError(
            package="jsonschema",
            feature="synthetic record validation",
            extra="dev",
        ) from exc
    return Draft202012Validator


__all__ = [
    "SYNTHETIC_RECORD_SCHEMA_IDS",
    "SYNTHETIC_RECORD_SCHEMA_NAMES",
    "SYNTHETIC_RECORD_SCHEMA_VERSION",
    "SyntheticRecordSchemaError",
    "export_record_schema",
    "export_record_schema_fingerprints",
    "export_record_schema_json",
    "export_record_schemas",
    "export_record_schemas_json",
    "record_schema_fingerprint",
    "record_schema_id",
    "validate_record",
    "write_record_schemas",
]
