"""Tests for the bundled OpenMedConfig JSON Schema."""

from __future__ import annotations

import json
from dataclasses import fields

import pytest
from jsonschema import Draft202012Validator

from openmed.core.config import (
    ConfigValidationError,
    OpenMedConfig,
    config_schema_path,
    load_config_from_file,
)


def _schema() -> dict[str, object]:
    return json.loads(config_schema_path().read_text(encoding="utf-8"))


def test_schema_is_valid_and_covers_every_config_field() -> None:
    schema = _schema()
    Draft202012Validator.check_schema(schema)

    field_names = {field.name for field in fields(OpenMedConfig)}
    assert set(schema["properties"]) == field_names
    assert set(schema["x-profile-keys"]) == field_names - {"profile"}


def test_default_and_remote_configs_validate_against_published_schema() -> None:
    schema = _schema()
    validator = Draft202012Validator(schema)
    default = OpenMedConfig()
    remote = OpenMedConfig(
        backend="remote",
        remote_inference_endpoint="https://inference.example.invalid",
        remote_inference_protocol="grpc",
        remote_inference_model_name="synthetic-model",
        remote_inference_model_version="1",
        remote_inference_tokenizer="synthetic-tokenizer",
    )

    assert default.validate() is None
    assert remote.validate() is None
    assert list(validator.iter_errors(default.to_dict())) == []
    assert list(validator.iter_errors(remote.to_dict())) == []


def test_file_validation_aggregates_unknown_and_wrong_types_without_values(
    tmp_path,
) -> None:
    private_value = "private-canary-value"
    path = tmp_path / "config.toml"
    path.write_text(
        f'timeout = "slow"\nunknown_private_key = "{private_value}"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        load_config_from_file(path)

    message = str(exc_info.value)
    assert "timeout" in message
    assert "expected integer" in message
    assert "unknown_private_key" in message
    assert private_value not in message
    assert len(exc_info.value.errors) == 2


def test_from_dict_rejects_unknown_keys_instead_of_filtering_them() -> None:
    with pytest.raises(ConfigValidationError, match="typo_timeout"):
        OpenMedConfig.from_dict({"typo_timeout": 30})


def test_from_dict_aggregates_invalid_values_before_post_init() -> None:
    private_value = "private-canary-value"

    with pytest.raises(ConfigValidationError) as exc_info:
        OpenMedConfig.from_dict(
            {
                "batch_size": "many",
                "chinese_pkuseg_domain": 7,
                "cjk_width_convention": private_value,
            }
        )

    message = str(exc_info.value)
    assert "batch_size" in message
    assert "chinese_pkuseg_domain" in message
    assert "cjk_width_convention" in message
    assert private_value not in message
    assert len(exc_info.value.errors) == 3


def test_from_dict_rejects_numeric_overflow_as_a_value_free_schema_error() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        OpenMedConfig.from_dict({"remote_inference_timeout_seconds": 10**400})

    assert exc_info.value.errors == (
        "remote_inference_timeout_seconds: number must be finite",
    )


@pytest.mark.parametrize(
    ("contents", "bad_key"),
    [
        ('timeout = "slow"\n', "timeout"),
        ("typo_timeout = 30\n", "typo_timeout"),
    ],
)
def test_load_config_rejects_invalid_file_with_offending_key(
    tmp_path,
    contents: str,
    bad_key: str,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigValidationError, match=bad_key):
        load_config_from_file(path)


def test_schema_path_is_an_installed_json_resource() -> None:
    path = config_schema_path()

    assert path.name == "config.schema.json"
    assert path.is_file()
    assert _schema()["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_list_items_and_numeric_bounds_are_validated() -> None:
    config = OpenMedConfig()
    config.medical_tokenizer_exceptions = ["safe", 7]  # type: ignore[list-item]
    config.indic_name_similarity_threshold = 1.5
    config.remote_inference_timeout_seconds = 0

    with pytest.raises(ConfigValidationError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "medical_tokenizer_exceptions[1]" in message
    assert "indic_name_similarity_threshold" in message
    assert "remote_inference_timeout_seconds" in message
