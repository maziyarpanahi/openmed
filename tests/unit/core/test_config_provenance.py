"""Focused tests for deterministic configuration provenance."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from openmed.core.config_provenance import (
    CONFLICT_OVERRIDDEN,
    CONFLICT_SAME_VALUE,
    ConfigurationResolutionError,
    audit_config_precedence,
    resolve_configuration,
)


def test_file_environment_and_cli_precedence_has_value_free_report():
    result = resolve_configuration(
        defaults={"timeout": 300, "api_token": "synthetic-default"},
        file_config={"timeout": 120, "api_token": "synthetic-file"},
        environment={
            "OPENMED_TIMEOUT": "60",
            "OPENMED_API_TOKEN": "synthetic-environment",
        },
        cli={"timeout": 30},
    )

    assert result.values["timeout"] == 30
    assert result.values["api_token"] == "synthetic-environment"
    report = result.provenance_report
    assert report["precedence"] == ["default", "file", "environment", "cli"]
    assert report["keys"]["timeout"] == {
        "source_class": "cli",
        "conflict_category": CONFLICT_OVERRIDDEN,
        "sources": ["default", "file", "environment", "cli"],
        "overridden_sources": ["default", "file", "environment"],
    }
    assert report["keys"]["api_token"]["source_class"] == "environment"
    serialized = json.dumps(report, sort_keys=True)
    assert "synthetic-default" not in serialized
    assert "synthetic-file" not in serialized
    assert "synthetic-environment" not in serialized
    assert "30" not in serialized

    serialized_resolution = json.dumps(result.to_dict(), sort_keys=True)
    assert serialized_resolution == serialized
    assert "synthetic-default" not in serialized_resolution
    assert "synthetic-file" not in serialized_resolution
    assert "synthetic-environment" not in serialized_resolution


def test_same_values_are_not_reported_as_a_conflict_and_order_is_stable():
    first = resolve_configuration(
        defaults={"timeout": 120, "device": "cpu"},
        file_config={"device": "cpu", "timeout": 120},
        environment={"OPENMED_TIMEOUT": "120"},
    )
    second = resolve_configuration(
        defaults={"device": "cpu", "timeout": 120},
        file_config={"timeout": 120, "device": "cpu"},
        environment={"OPENMED_TIMEOUT": "120"},
    )

    assert first.values == second.values
    assert first.provenance_report == second.provenance_report
    assert first.provenance_report["keys"]["timeout"]["conflict_category"] == (
        CONFLICT_SAME_VALUE
    )


def test_local_toml_and_namespace_inputs_are_supported(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("timeout = 90\nlocal_only = true\n", encoding="utf-8")

    result = resolve_configuration(
        defaults={"timeout": 300, "local_only": False},
        file_config=config_path,
        environment={},
        cli=Namespace(timeout=45, local_only=None),
    )

    assert result.values == {"local_only": True, "timeout": 45}
    assert result.report["keys"]["local_only"]["source_class"] == "file"


def test_environment_aliases_are_deterministic_and_typed():
    result = resolve_configuration(
        defaults={"device": "auto", "local_only": False, "timeout": 300},
        environment={
            "OPENMED_DEVICE": "cpu",
            "OPENMED_TORCH_DEVICE": "cuda",
            "OPENMED_OFFLINE": "1",
            "OPENMED_TIMEOUT": "15",
        },
    )

    assert result.values == {"device": "cuda", "local_only": True, "timeout": 15}
    assert result.report["keys"]["device"]["source_class"] == "environment"


def test_unprefixed_ambient_variables_do_not_override_openmed_settings():
    result = resolve_configuration(
        defaults={"device": "cpu", "profile": None, "timeout": 300},
        environment={
            "DEVICE": "cuda",
            "PROFILE": "production",
            "timeout": "1",
            "OPENMED_TIMEOUT": "15",
        },
    )

    assert result.values == {"device": "cpu", "profile": None, "timeout": 15}
    assert result.report["keys"]["device"]["source_class"] == "default"
    assert result.report["keys"]["profile"]["source_class"] == "default"


def test_invalid_environment_value_does_not_echo_raw_value():
    with pytest.raises(ConfigurationResolutionError) as error:
        resolve_configuration(
            defaults={"timeout": 300},
            environment={"OPENMED_TIMEOUT": "synthetic-invalid-input"},
        )

    message = str(error.value)
    assert "timeout" in message
    assert "synthetic-invalid-input" not in message


def test_audit_helper_returns_only_provenance():
    report = audit_config_precedence(
        defaults={"mode": "safe"},
        file_config={"mode": "safe"},
        environment={},
        cli={},
    )

    assert "values" not in report
    assert report["keys"]["mode"]["conflict_category"] == CONFLICT_SAME_VALUE
