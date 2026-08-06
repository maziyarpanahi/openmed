"""Tests for OpenMedConfig JSON Schema validation."""

import json

import pytest
from jsonschema import Draft202012Validator

from openmed.core.config import (
    PROFILE_PRESETS,
    OpenMedConfig,
    config_schema_path,
    load_config_from_file,
)


def test_bundled_schema_validates_default_config() -> None:
    schema_path = config_schema_path()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema_path.exists()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(OpenMedConfig().to_dict())
    OpenMedConfig().validate()


def test_validate_reports_the_invalid_field() -> None:
    config = OpenMedConfig()
    config.timeout = "not-an-integer"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="timeout"):
        config.validate()


def test_schema_lists_the_keys_available_to_profiles() -> None:
    schema = json.loads(config_schema_path().read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    assert set(schema["x-profile-keys"]) == set(schema["properties"]) - {"profile"}
    for profile_settings in PROFILE_PRESETS.values():
        validator.validate(profile_settings)


def test_from_dict_aggregates_unknown_and_wrong_typed_keys() -> None:
    with pytest.raises(ValueError) as exc_info:
        OpenMedConfig.from_dict(
            {
                "timeout": "not-an-integer",
                "unknown_option": True,
            }
        )

    message = str(exc_info.value)
    assert "timeout" in message
    assert "integer" in message
    assert "unknown_option" in message


def test_load_config_from_file_validates_unknown_and_wrong_typed_keys(tmp_path) -> None:
    config_path = tmp_path / "invalid-config.toml"
    config_path.write_text(
        'timeout = "not-an-integer"\nunknown_option = true\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_config_from_file(config_path)

    message = str(exc_info.value)
    assert "timeout" in message
    assert "unknown_option" in message
