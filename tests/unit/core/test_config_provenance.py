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


def test_config_credentials_never_serialize_or_persist(monkeypatch, tmp_path):
    import os

    from openmed.core import config as config_module

    sentinel = "hf_SYNTHETIC_CONFIG_TOKEN_NEVER_SERIALIZE"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    monkeypatch.setattr(config_module, "PROFILES_DIR", tmp_path / "profiles")
    config = config_module.OpenMedConfig(timeout=77)
    assert config.hf_token == sentinel
    assert sentinel not in repr(config)
    assert "hf_token" not in config.to_dict()
    assert config.with_profile("test").hf_token == sentinel
    saved = config_module.save_config_to_file(config, tmp_path / "config.toml")
    profile = config_module.save_profile(
        "synthetic", {"timeout": 77, "hf_token": sentinel}
    )
    for path in (saved, profile):
        assert sentinel not in path.read_text()
        if os.name != "nt":
            assert path.stat().st_mode & 0o777 == 0o600
    assert config_module.load_config_from_file(saved).timeout == 77


@pytest.mark.parametrize("as_json", [True, False])
def test_cli_config_show_and_set_do_not_expose_credentials(
    monkeypatch, tmp_path, capsys, as_json
):
    from openmed.cli.main import main
    from openmed.core import config as config_module

    sentinel = "hf_SYNTHETIC_CLI_CONFIG_TOKEN"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    monkeypatch.setenv("OPENMED_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(config_module, "_config", config_module.OpenMedConfig())
    options = ["--json"] if as_json else []
    assert main(["config", "show", *options]) == 0
    captured = capsys.readouterr()
    assert sentinel not in captured.out + captured.err
    assert "hf_token_present" in captured.out
    assert main(["config", "set", "timeout", "77", *options]) == 0
    captured = capsys.readouterr()
    assert sentinel not in captured.out + captured.err
    assert sentinel not in (tmp_path / "config.toml").read_text()


def test_legacy_stored_credential_warns_and_doctor_never_prints_it(
    monkeypatch, tmp_path
):
    from openmed.core import config as config_module
    from openmed.core.doctor import _check_persisted_credentials

    sentinel = "hf_SYNTHETIC_PERSISTED_TOKEN"
    path = tmp_path / "config.toml"
    path.write_text(f'hf_token = "{sentinel}"\ntimeout = 77\n')
    monkeypatch.setenv("OPENMED_CONFIG", str(path))
    monkeypatch.setattr(config_module, "PROFILES_DIR", tmp_path / "profiles")
    with pytest.warns(UserWarning, match="persisted_credential") as warnings:
        config = config_module.load_config_from_file(path)
    assert config.hf_token == sentinel
    assert sentinel not in str(warnings[0].message)
    checks = []
    _check_persisted_credentials(checks)
    assert checks[0]["details"] == "persisted_credential"
    assert sentinel not in json.dumps(checks)
    config_module.save_config_to_file(config, path)
    assert sentinel not in path.read_text()


def test_saved_configuration_has_owner_only_posix_mode(tmp_path):
    import os

    from openmed.core.config import OpenMedConfig, save_config_to_file

    if os.name == "nt":
        pytest.skip("Windows confidentiality uses directory ACLs, not POSIX mode bits")
    path = tmp_path / "config.toml"
    path.write_text("timeout = 12\n")
    path.chmod(0o644)
    save_config_to_file(OpenMedConfig(timeout=77), path)
    assert path.stat().st_mode & 0o777 == 0o600
