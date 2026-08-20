"""Focused tests for the offline training-conversation schema registry."""

from __future__ import annotations

import pytest

from openmed.traces.schemas.registry import (
    AmbiguousSchemaError,
    InvalidSchemaError,
    MessagesSchema,
    SchemaMismatchError,
    SchemaRegistryError,
    ShareGPTSchema,
    TrainingSchemaRegistry,
    UnknownSchemaError,
)


def test_messages_schema_walks_and_reconstructs_without_flattening() -> None:
    record = {
        "id": "synthetic-row-1",
        "messages": [
            {"role": "system", "content": "Synthetic system instruction"},
            {"role": "user", "content": "Synthetic user value"},
        ],
        "metadata": {"split": "train"},
    }
    registry = TrainingSchemaRegistry()

    walked = registry.walk(record, schema="messages")
    rebuilt = registry.reconstruct(
        record,
        {path: f"[REDACTED-{index}]" for index, (path, _) in enumerate(walked)},
        schema="messages",
    )

    assert walked == (
        (("messages", 0, "content"), "Synthetic system instruction"),
        (("messages", 1, "content"), "Synthetic user value"),
    )
    assert rebuilt == {
        "id": "synthetic-row-1",
        "messages": [
            {"role": "system", "content": "[REDACTED-0]"},
            {"role": "user", "content": "[REDACTED-1]"},
        ],
        "metadata": {"split": "train"},
    }
    assert record["messages"][1]["content"] == "Synthetic user value"


def test_preference_schema_supports_nested_message_and_response_layouts() -> None:
    record = {
        "preference": {
            "prompt": [{"role": "user", "content": "Synthetic prompt"}],
            "chosen": {
                "messages": [{"role": "assistant", "content": "Synthetic chosen"}]
            },
            "rejected": {
                "messages": [{"role": "assistant", "content": "Synthetic rejected"}]
            },
        },
        "annotations": {"source": "synthetic"},
    }
    registry = TrainingSchemaRegistry()

    walked = registry.walk(record, schema="preference")
    rebuilt = registry.transform(
        record,
        lambda text: text.replace("Synthetic", "Redacted"),
        schema="preference",
    )

    assert [path for path, _ in walked] == [
        ("preference", "prompt", 0, "content"),
        ("preference", "chosen", "messages", 0, "content"),
        ("preference", "rejected", "messages", 0, "content"),
    ]
    assert rebuilt["preference"]["chosen"]["messages"][0]["content"] == (
        "Redacted chosen"
    )
    assert rebuilt["preference"]["rejected"]["messages"][0]["content"] == (
        "Redacted rejected"
    )
    assert record["preference"]["prompt"][0]["content"] == "Synthetic prompt"


def test_sharegpt_schema_is_available_by_explicit_alias() -> None:
    record = {
        "conversations": [
            {"from": "human", "value": "Synthetic question"},
            {"from": "gpt", "value": "Synthetic answer"},
        ]
    }
    registry = TrainingSchemaRegistry()

    assert isinstance(registry.resolve(record, schema="conversations"), ShareGPTSchema)
    assert registry.walk(record, schema="sharegpt")[1][1] == "Synthetic answer"


def test_auto_detection_rejects_ambiguous_records_before_reconstruction() -> None:
    record = {
        "messages": [{"role": "user", "content": "Synthetic private value"}],
        "prompt": "Synthetic prompt",
        "chosen": "Synthetic preferred response",
        "rejected": "Synthetic rejected response",
    }
    registry = TrainingSchemaRegistry()

    with pytest.raises(AmbiguousSchemaError) as error:
        registry.reconstruct(record, {})

    assert "Synthetic private value" not in str(error.value)
    assert record["messages"][0]["content"] == "Synthetic private value"

    explicit = registry.transform(record, str.upper, schema="preference")
    assert explicit["chosen"] == "SYNTHETIC PREFERRED RESPONSE"
    assert record["chosen"] == "Synthetic preferred response"


def test_registry_is_deterministic_and_rejects_unknown_or_unmatched_records() -> None:
    registry = TrainingSchemaRegistry()
    record = {"messages": [{"role": "user", "content": "Synthetic value"}]}

    assert registry.available() == ("messages", "preference", "sharegpt")
    assert registry.matching_schemas(record) == ("messages",)
    with pytest.raises(UnknownSchemaError, match="no training schema"):
        registry.resolve({"metadata": {"split": "train"}})
    with pytest.raises(UnknownSchemaError, match="not registered"):
        registry.get("missing")
    with pytest.raises(SchemaMismatchError, match="does not match"):
        registry.resolve(record, schema="preference")


def test_registry_validates_custom_protocol_and_duplicate_names() -> None:
    class SyntheticSchema:
        name = "synthetic"

        def detect(self, record: object) -> bool:
            return isinstance(record, dict) and "text" in record

        def walk(self, record: dict[str, str]):
            return ((("text",), record["text"]),)

        def reconstruct(self, record: dict[str, str], replacements):
            rebuilt = dict(record)
            rebuilt["text"] = replacements[("text",)]
            return rebuilt

    registry = TrainingSchemaRegistry(include_defaults=False)
    registry.register(SyntheticSchema())
    with pytest.raises(SchemaRegistryError, match="already registered"):
        registry.register(SyntheticSchema())
    with pytest.raises(InvalidSchemaError, match="must define detect"):
        registry.register(type("IncompleteSchema", (), {"name": "incomplete"})())

    assert registry.transform({"text": "Synthetic value"}, str.upper) == {
        "text": "SYNTHETIC VALUE"
    }


def test_schema_names_are_hashed_in_diagnostics() -> None:
    sensitive = "PatientJaneDoe"
    registry = TrainingSchemaRegistry(include_defaults=False)

    with pytest.raises(UnknownSchemaError) as unknown_error:
        registry.get(sensitive)
    assert sensitive not in str(unknown_error.value)
    assert "schema_sha256_" in str(unknown_error.value)

    incomplete = type("IncompleteSchema", (), {"name": sensitive})()
    with pytest.raises(InvalidSchemaError) as invalid_error:
        registry.register(incomplete)
    assert sensitive not in str(invalid_error.value)
    assert "schema_sha256_" in str(invalid_error.value)


def test_registry_rejects_a_canonical_name_that_collides_with_an_alias() -> None:
    class AliasCollisionSchema:
        name = "chat"

        def detect(self, record: object) -> bool:
            return isinstance(record, dict)

        def walk(self, record: object):
            del record
            return ()

        def reconstruct(self, record: object, replacements: object) -> object:
            del replacements
            return record

    registry = TrainingSchemaRegistry()

    with pytest.raises(SchemaRegistryError, match="already registered"):
        registry.register(AliasCollisionSchema())

    assert registry.get("chat").name == "messages"
    assert "chat" not in registry.available()


def test_custom_detection_and_walk_cannot_mutate_the_source_record() -> None:
    class MutatingSchema:
        name = "mutating"

        def detect(self, record: dict[str, object]) -> bool:
            record["detect_mutation"] = True
            return "text" in record

        def walk(self, record: dict[str, object]):
            record["walk_mutation"] = True
            return ((("text",), record["text"]),)

        def reconstruct(self, record: dict[str, object], replacements):
            record["text"] = replacements[("text",)]
            return record

    record = {"text": "Synthetic private value"}
    registry = TrainingSchemaRegistry([MutatingSchema()], include_defaults=False)

    walked = registry.walk(record)
    rebuilt = registry.reconstruct(
        record,
        {("text",): "[REDACTED]"},
    )

    assert walked == ((("text",), "Synthetic private value"),)
    assert rebuilt == {"text": "[REDACTED]"}
    assert record == {"text": "Synthetic private value"}


def test_messages_schema_rejects_partially_unrecognized_content() -> None:
    registry = TrainingSchemaRegistry()
    record = {
        "messages": [
            {"role": "user", "content": "Synthetic private value"},
            {"role": "assistant", "unexpected": "Synthetic hidden value"},
        ]
    }

    with pytest.raises(SchemaMismatchError):
        registry.resolve(record, schema="messages")

    assert record["messages"][0]["content"] == "Synthetic private value"
