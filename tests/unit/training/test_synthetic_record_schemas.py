"""Contracts for the published synthetic training-record JSON Schemas."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.training.synthetic import (
    BURNED_IN_LABELS,
    DOCUMENT_TYPES,
    LOCALE_PHI_LABELS,
    SECTION_LABELS,
    SECTION_RECORD_SCHEMA_VERSION,
    SOCIAL_HISTORY_CATEGORIES,
    SUPPORTED_LOCALE_PHI_LANGUAGES,
    SYNTHETIC_RECORD_SCHEMA_IDS,
    SYNTHETIC_RECORD_SCHEMA_NAMES,
    SYNTHETIC_RECORD_SCHEMA_VERSION,
    SYNTHETIC_SECTION_LICENSE,
    SYNTHETIC_SECTION_SOURCE,
    SYNTHETIC_SOCIAL_HISTORY_SOURCE,
    SYNTHETIC_SOURCE,
    DictionaryTranslator,
    SyntheticRecordSchemaError,
    augment_span_annotated_examples,
    build_section_dataset,
    export_record_schema,
    export_record_schema_fingerprints,
    export_record_schema_json,
    export_record_schemas_json,
    generate_burned_in_example,
    generate_locale_phi_examples,
    generate_social_history_examples,
    record_schema_fingerprint,
    record_schema_id,
    validate_record,
    write_record_schemas,
)

SCHEMA_FINGERPRINTS = {
    "burned_in_annotation": (
        "sha256:e452da60cb8456f1274cf8bdb280d883b29f7d7e53ea3f7e894466a365cf2e9a"
    ),
    "locale_phi_example": (
        "sha256:9b26c374261b95669935f7d09afd3fe8bff2cde980eb3fc7b5768dbc1890420c"
    ),
    "section_label_record": (
        "sha256:4c4906112eb073c984dfa48d0d4b4b7a0a5172f97fbf85d131b2a19b1eff92c8"
    ),
    "social_history_example": (
        "sha256:4402db738463152f3b3ed3e9683395d40bfc011485c0ca5524f621cfa31a93e2"
    ),
    "translation_augmented_example": (
        "sha256:3515babe286badf3d5fee909c2e0506891465d2435cc057df9d664417ea2e112"
    ),
}

OPEN_OBJECT_PATHS = {
    "burned_in_annotation": ["#/$defs/metadata"],
    "locale_phi_example": [
        "#/$defs/metadata",
        "#/$defs/span/properties/metadata",
    ],
    "section_label_record": ["#/$defs/metadata"],
    "social_history_example": ["#/$defs/metadata"],
    "translation_augmented_example": [
        "#/$defs/metadata",
        "#/$defs/span/properties/metadata",
    ],
}


def _span(text: str, value: str, label: str) -> dict[str, Any]:
    start = text.index(value)
    return {"end": start + len(value), "label": label, "start": start, "text": value}


def _translation_seed() -> dict[str, Any]:
    text = "Patient has diabetes and takes metformin."
    return {
        "gold_spans": [
            _span(text, "diabetes", "CONDITION"),
            _span(text, "metformin", "MEDICATION"),
        ],
        "id": "seed-en-001",
        "language": "en",
        "metadata": {"split": "train", "synthetic": True},
        "text": text,
    }


def _section_record(tmp_path: Path) -> dict[str, Any]:
    result = build_section_dataset(
        seed=7,
        n=1,
        output_path=tmp_path / "section_labels.jsonl",
    )
    return result.records[0]


def _records(tmp_path: Path) -> dict[str, dict[str, Any]]:
    burned_in = generate_burned_in_example(seed=7)
    return {
        "burned_in_annotation": burned_in.gold_boxes[0].to_dict(),
        "locale_phi_example": generate_locale_phi_examples(
            languages=("en",),
            seed=7,
        )[0].to_training_item(),
        "section_label_record": _section_record(tmp_path),
        "social_history_example": generate_social_history_examples(2, seed=7)[
            0
        ].to_dict(),
        "translation_augmented_example": augment_span_annotated_examples(
            [_translation_seed()],
            translator=DictionaryTranslator(),
        )[0].to_training_item(),
    }


@pytest.fixture(scope="module")
def records(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    return _records(tmp_path_factory.mktemp("synthetic-record-schemas"))


def _validator(name: str) -> Draft202012Validator:
    schema = export_record_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _refs(value: Any) -> list[str]:
    if isinstance(value, dict):
        refs = [value["$ref"]] if "$ref" in value else []
        for nested in value.values():
            refs.extend(_refs(nested))
        return refs
    if isinstance(value, list):
        return [ref for nested in value for ref in _refs(nested)]
    return []


def _keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        keys = list(value)
        for nested in value.values():
            keys.extend(_keys(nested))
        return keys
    if isinstance(value, list):
        return [key for nested in value for key in _keys(nested)]
    return []


def _open_object_paths(value: Any, path: str = "#") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        if (
            value.get("type") == "object"
            and value.get("additionalProperties") is not False
        ):
            paths.append(path)
        for key, nested in value.items():
            paths.extend(_open_object_paths(nested, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(_open_object_paths(nested, f"{path}/{index}"))
    return paths


@pytest.mark.parametrize("name", SYNTHETIC_RECORD_SCHEMA_NAMES)
def test_schemas_are_draft_2020_12_and_resolve_entirely_locally(name: str) -> None:
    schema = export_record_schema(name)

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == SYNTHETIC_RECORD_SCHEMA_IDS[name] == record_schema_id(name)
    assert f"-v{SYNTHETIC_RECORD_SCHEMA_VERSION}.schema.json" in schema["$id"]
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("name", SYNTHETIC_RECORD_SCHEMA_NAMES)
def test_schemas_reference_only_local_definitions(name: str) -> None:
    refs = _refs(export_record_schema(name))

    assert refs
    assert all(ref.startswith("#/$defs/") for ref in refs)


def test_schema_exports_are_byte_stable_and_drift_guarded() -> None:
    for name in SYNTHETIC_RECORD_SCHEMA_NAMES:
        first = export_record_schema_json(name)
        second = export_record_schema_json(name)

        assert first == second
        assert json.loads(first) == export_record_schema(name)
        assert record_schema_fingerprint(name) == SCHEMA_FINGERPRINTS[name]
        assert hashlib.sha256(first.encode("utf-8")).hexdigest() == (
            SCHEMA_FINGERPRINTS[name].removeprefix("sha256:")
        )

    bundle = export_record_schemas_json()
    assert bundle == export_record_schemas_json()
    assert json.loads(bundle)["record_schema_version"] == (
        SYNTHETIC_RECORD_SCHEMA_VERSION
    )
    assert sorted(json.loads(bundle)["schemas"]) == sorted(
        SYNTHETIC_RECORD_SCHEMA_NAMES
    )
    assert export_record_schema_fingerprints() == SCHEMA_FINGERPRINTS


@pytest.mark.parametrize("name", SYNTHETIC_RECORD_SCHEMA_NAMES)
def test_generator_records_validate_against_their_schema(
    name: str,
    records: dict[str, dict[str, Any]],
) -> None:
    record = records[name]

    validate_record(name, record)
    assert _validator(name).is_valid(record)


def test_exported_schemas_are_fresh_mappings() -> None:
    schema = export_record_schema("locale_phi_example")
    schema["title"] = "mutated"
    schema["properties"]["text"]["minLength"] = 99

    assert export_record_schema("locale_phi_example")["title"] != "mutated"
    assert export_record_schema("locale_phi_example")["properties"]["text"] == {
        "type": "string",
        "minLength": 1,
    }


def test_schemas_track_the_runtime_generator_constants() -> None:
    locale = export_record_schema("locale_phi_example")
    burned_in = export_record_schema("burned_in_annotation")
    social = export_record_schema("social_history_example")
    section = export_record_schema("section_label_record")
    translation = export_record_schema("translation_augmented_example")

    assert locale["properties"]["language"]["enum"] == list(
        SUPPORTED_LOCALE_PHI_LANGUAGES
    )
    assert locale["$defs"]["span"]["properties"]["label"]["enum"] == list(
        LOCALE_PHI_LABELS
    )
    assert locale["properties"]["synthetic_source"] == {"const": "locale_phi"}
    assert burned_in["properties"]["label"]["enum"] == list(BURNED_IN_LABELS)
    assert social["$defs"]["event"]["properties"]["category"]["enum"] == list(
        SOCIAL_HISTORY_CATEGORIES
    )
    assert social["$defs"]["metadata"]["properties"]["source"] == {
        "const": SYNTHETIC_SOCIAL_HISTORY_SOURCE
    }
    assert section["properties"]["doc_type"]["enum"] == list(DOCUMENT_TYPES)
    assert section["$defs"]["section"]["properties"]["label"]["enum"] == list(
        SECTION_LABELS
    )
    assert section["properties"]["schema_version"] == {
        "const": SECTION_RECORD_SCHEMA_VERSION
    }
    assert section["properties"]["license"] == {"const": SYNTHETIC_SECTION_LICENSE}
    assert section["properties"]["source"] == {"const": SYNTHETIC_SECTION_SOURCE}
    assert section["$defs"]["metadata"]["properties"]["synthetic_source"] == {
        "const": SYNTHETIC_SECTION_SOURCE
    }
    assert translation["properties"]["synthetic_source"] == {"const": SYNTHETIC_SOURCE}
    assert translation["$defs"]["metadata"]["properties"]["synthetic_source"] == {
        "const": SYNTHETIC_SOURCE
    }


@pytest.mark.parametrize("name", SYNTHETIC_RECORD_SCHEMA_NAMES)
def test_only_free_form_metadata_bags_allow_extra_properties(name: str) -> None:
    schema = export_record_schema(name)

    assert schema["additionalProperties"] is False
    assert sorted(_open_object_paths(schema)) == sorted(OPEN_OBJECT_PATHS[name])


@pytest.mark.parametrize(
    "name",
    [
        "burned_in_annotation",
        "locale_phi_example",
        "section_label_record",
        "social_history_example",
        "translation_augmented_example",
    ],
)
def test_every_schema_rejects_an_undeclared_field(
    name: str,
    records: dict[str, dict[str, Any]],
) -> None:
    record = dict(records[name])
    record["synthetic_extra"] = 1

    assert not _validator(name).is_valid(record)
    with pytest.raises(SyntheticRecordSchemaError, match="Additional properties"):
        validate_record(name, record)


def test_schemas_require_the_synthetic_markers_and_source(
    records: dict[str, dict[str, Any]],
) -> None:
    locale = dict(records["locale_phi_example"])
    locale["metadata"] = {**locale["metadata"], "contains_real_phi": True}
    assert not _validator("locale_phi_example").is_valid(locale)
    with pytest.raises(SyntheticRecordSchemaError, match="contains_real_phi"):
        validate_record("locale_phi_example", locale)

    locale = dict(records["locale_phi_example"])
    locale.pop("is_synthetic")
    assert not _validator("locale_phi_example").is_valid(locale)

    burned_in = json.loads(json.dumps(records["burned_in_annotation"]))
    burned_in["metadata"].pop("synthetic")
    assert not _validator("burned_in_annotation").is_valid(burned_in)

    social = json.loads(json.dumps(records["social_history_example"]))
    social["metadata"] = {"source": SYNTHETIC_SOCIAL_HISTORY_SOURCE}
    assert not _validator("social_history_example").is_valid(social)

    section = dict(records["section_label_record"])
    section.pop("synthetic")
    assert not _validator("section_label_record").is_valid(section)
    section = dict(records["section_label_record"])
    section["restricted_data"] = True
    assert not _validator("section_label_record").is_valid(section)

    translation = json.loads(json.dumps(records["translation_augmented_example"]))
    translation["metadata"].pop("provenance")
    assert not _validator("translation_augmented_example").is_valid(translation)


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "locale_phi_example",
            lambda record: record["labels"][0].update({"start": 40, "end": 30}),
            "inverted offset",
        ),
        (
            "locale_phi_example",
            lambda record: record["labels"][0].update({"start": -5}),
            "less than the minimum",
        ),
        (
            "social_history_example",
            lambda record: record["events"][0].update({"span": [40, 10]}),
            "inverted offset",
        ),
        (
            "section_label_record",
            lambda record: record["sections"][0].update({"start": 90, "end": 20}),
            "inverted offset",
        ),
        (
            "burned_in_annotation",
            lambda record: record.update({"bbox": [527, 65, 386, 83]}),
            "inverted bounding box",
        ),
        (
            "burned_in_annotation",
            lambda record: record.update({"bbox": [0, 0, 10, -3]}),
            "less than the minimum",
        ),
    ],
)
def test_rejects_inverted_and_negative_offsets(
    name: str,
    mutate: Any,
    message: str,
    records: dict[str, dict[str, Any]],
) -> None:
    record = json.loads(json.dumps(records[name]))
    mutate(record)

    with pytest.raises(SyntheticRecordSchemaError, match=message):
        validate_record(name, record)


def test_validate_record_rejects_unknown_names(
    records: dict[str, dict[str, Any]],
) -> None:
    with pytest.raises(KeyError, match="unknown synthetic record schema"):
        validate_record("synthetic_unknown", records["locale_phi_example"])

    with pytest.raises(KeyError, match="unknown synthetic record schema"):
        export_record_schema("synthetic_unknown")

    with pytest.raises(KeyError, match="unknown synthetic record schema"):
        record_schema_id("synthetic_unknown")


def test_write_record_schemas_is_deterministic_and_offline_parseable(
    tmp_path: Path,
) -> None:
    first = write_record_schemas(tmp_path / "first")
    second = write_record_schemas(tmp_path / "second")

    assert [path.name for path in first] == [
        "locale-phi-example.schema.json",
        "burned-in-annotation.schema.json",
        "social-history-example.schema.json",
        "section-label-record.schema.json",
        "translation-augmented-example.schema.json",
    ]
    for written, repeated in zip(first, second, strict=True):
        payload = written.read_bytes()

        assert payload == repeated.read_bytes()
        assert b"\r\n" not in payload
        assert payload.endswith(b"\n")
        assert json.loads(payload.decode("utf-8")) == export_record_schema(
            written.name.removesuffix(".schema.json").replace("-", "_")
        )


@pytest.mark.parametrize("name", SYNTHETIC_RECORD_SCHEMA_NAMES)
def test_schemas_describe_structure_without_example_values(
    name: str,
    records: dict[str, dict[str, Any]],
) -> None:
    schema = export_record_schema(name)
    text = export_record_schema_json(name)

    assert "examples" not in _keys(schema)
    assert "default" not in _keys(schema)
    assert str(records[name]["text"]) not in text


def test_schema_exports_do_not_depend_on_generated_records(tmp_path: Path) -> None:
    before = export_record_schemas_json()

    generate_locale_phi_examples(languages=("fr",), seed=99)
    generate_burned_in_example(seed=99)
    generate_social_history_examples(3, seed=99)
    build_section_dataset(seed=99, n=2, output_path=tmp_path / "other.jsonl")
    augment_span_annotated_examples(
        [_translation_seed()],
        translator=DictionaryTranslator(),
        include_backtranslations=False,
    )

    assert export_record_schemas_json() == before


def test_validate_record_reports_the_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
    records: dict[str, dict[str, Any]],
) -> None:
    monkeypatch.setitem(sys.modules, "jsonschema", None)

    with pytest.raises(ImportError) as excinfo:
        validate_record("locale_phi_example", records["locale_phi_example"])

    assert "requires optional dependency 'jsonschema'" in str(excinfo.value)
    assert getattr(excinfo.value, "extra", None) == "dev"
