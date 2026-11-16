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


def test_cli_config_show_and_set_do_not_expose_credentials(subtests, tmp_path, capsys):
    case_root = tmp_path
    for as_json in [True, False]:
        with subtests.test(as_json=as_json):
            tmp_path = case_root / str(len(list(case_root.iterdir())))
            tmp_path.mkdir()
            capsys.readouterr()
            with pytest.MonkeyPatch.context() as monkeypatch:
                from openmed.cli.main import main
                from openmed.core import config as config_module

                sentinel = "hf_SYNTHETIC_CLI_CONFIG_TOKEN"
                monkeypatch.setenv("HF_TOKEN", sentinel)
                monkeypatch.setenv("OPENMED_CONFIG", str(tmp_path / "config.toml"))
                monkeypatch.setattr(
                    config_module, "_config", config_module.OpenMedConfig()
                )
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


# -- service settings -------------------------------------------------------------

SERVICE_SENTINEL = "synthetic-service-sentinel-5341"


def _service_snapshot() -> dict[str, str]:
    return {
        # Non-secret settings: one changed, one equal to its default, one blank.
        "OPENMED_SERVICE_AUTH_ENABLED": "true",
        "OPENMED_SERVICE_RATE_LIMIT_RPS": "0",
        "OPENMED_SERVICE_TRUSTED_HOSTS": "localhost, 127.0.0.1,[::1]",
        "OPENMED_SERVICE_LOG_LEVEL": "   ",
        "OPENMED_SERVICE_OTLP_ENDPOINT": f"https://{SERVICE_SENTINEL}.invalid",
        # Secrets: inline, mounted file, and both at once.
        "OPENMED_SERVICE_AUTH_API_KEYS": f'[{{"key_hash": "{SERVICE_SENTINEL}"}}]',
        "OPENMED_SERVICE_DIGEST_KEY_FILE": f"/run/secrets/{SERVICE_SENTINEL}",
        "OPENMED_SERVICE_OTLP_HEADERS": f"authorization={SERVICE_SENTINEL}",
        "OPENMED_SERVICE_OTLP_HEADERS_FILE": f"/run/secrets/{SERVICE_SENTINEL}-h",
        # Unrelated variables are ignored.
        "OPENMED_PROFILE": "prod",
        "PATH": SERVICE_SENTINEL,
    }


def test_service_report_lists_every_declared_setting_without_values():
    from openmed.core.config_provenance import (
        SERVICE_CONFLICT_CATEGORIES,
        SERVICE_PRECEDENCE,
        resolve_service_configuration,
    )
    from openmed.core.settings_registry import declared_setting_names

    report = resolve_service_configuration(_service_snapshot()).provenance_report

    assert report["section"] == "service"
    assert report["precedence"] == ["default", "environment", "secret_file"]
    assert set(report["keys"]) == declared_setting_names("service")
    for name, entry in report["keys"].items():
        assert entry["source_class"] in SERVICE_PRECEDENCE, name
        assert entry["conflict_category"] in SERVICE_CONFLICT_CATEGORIES, name
        assert set(entry) == {
            "source_class",
            "conflict_category",
            "sources",
            "overridden_sources",
            "secret",
            "present",
        }


def test_service_report_classifies_sources_and_conflicts():
    from openmed.core.config_provenance import (
        CONFLICT_SECRET_SOURCES,
        resolve_service_configuration,
    )

    keys = resolve_service_configuration(_service_snapshot()).provenance_report["keys"]

    assert keys["OPENMED_SERVICE_AUTH_ENABLED"] == {
        "source_class": "environment",
        "conflict_category": CONFLICT_OVERRIDDEN,
        "sources": ["default", "environment"],
        "overridden_sources": ["default"],
        "secret": False,
        "present": True,
    }
    assert keys["OPENMED_SERVICE_RATE_LIMIT_RPS"]["conflict_category"] == (
        CONFLICT_SAME_VALUE
    )
    assert keys["OPENMED_SERVICE_TRUSTED_HOSTS"]["conflict_category"] == (
        CONFLICT_SAME_VALUE
    )
    # A blank value counts as unset, as the service parsers treat it.
    assert keys["OPENMED_SERVICE_LOG_LEVEL"]["source_class"] == "default"
    assert keys["OPENMED_SERVICE_METRICS_ENABLED"] == {
        "source_class": "default",
        "conflict_category": "none",
        "sources": ["default"],
        "overridden_sources": [],
        "secret": False,
        "present": False,
    }

    assert keys["OPENMED_SERVICE_AUTH_API_KEYS"] == {
        "source_class": "environment",
        "conflict_category": "none",
        "sources": ["environment"],
        "overridden_sources": [],
        "secret": True,
        "present": True,
    }
    assert keys["OPENMED_SERVICE_DIGEST_KEY"]["source_class"] == "secret_file"
    assert keys["OPENMED_SERVICE_DIGEST_KEY"]["present"] is True
    assert keys["OPENMED_SERVICE_OTLP_HEADERS"]["conflict_category"] == (
        CONFLICT_SECRET_SOURCES
    )
    assert keys["OPENMED_SERVICE_OTLP_HEADERS"]["sources"] == [
        "environment",
        "secret_file",
    ]
    assert keys["OPENMED_SERVICE_AUTH_JWKS"] == {
        "source_class": "default",
        "conflict_category": "none",
        "sources": ["default"],
        "overridden_sources": [],
        "secret": True,
        "present": False,
    }


def test_service_report_never_contains_planted_values():
    from openmed.core.config_provenance import resolve_service_configuration

    provenance = resolve_service_configuration(_service_snapshot())

    for rendered in (
        json.dumps(provenance.provenance_report, sort_keys=True),
        json.dumps(provenance.to_dict()),
        provenance.to_json(),
        repr(provenance),
        str(provenance),
        repr(provenance.entries),
    ):
        assert SERVICE_SENTINEL not in rendered
        assert "/run/secrets" not in rendered


def test_service_report_is_identical_for_reordered_snapshots():
    from openmed.core.config_provenance import resolve_service_configuration

    forward = _service_snapshot()
    backward = dict(reversed(list(forward.items())))
    assert list(forward) != list(backward)

    first = resolve_service_configuration(forward)
    second = resolve_service_configuration(environment=backward)

    assert first.to_json() == second.to_json()
    assert first.provenance_report == second.provenance_report
    assert list(first.provenance_report["keys"]) == sorted(
        first.provenance_report["keys"]
    )


def test_service_report_reads_the_process_environment_by_default(monkeypatch):
    from openmed.core.config_provenance import resolve_service_configuration

    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "yes")
    monkeypatch.delenv("OPENMED_SERVICE_AUTH_API_KEYS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_AUTH_API_KEYS_FILE", raising=False)

    provenance = resolve_service_configuration()

    assert provenance.entry("OPENMED_SERVICE_METRICS_ENABLED").source_class == (
        "environment"
    )
    assert provenance.entry("OPENMED_SERVICE_AUTH_API_KEYS").source_class == ("default")
    with pytest.raises(KeyError):
        provenance.entry("OPENMED_SERVICE_NOT_DECLARED")


def test_service_report_rejects_non_string_values_without_echoing_them():
    from openmed.core.config_provenance import resolve_service_configuration

    with pytest.raises(ConfigurationResolutionError) as exc_info:
        resolve_service_configuration({"OPENMED_SERVICE_JOBS_WORKERS": 4096})
    assert "4096" not in str(exc_info.value)
    with pytest.raises(ConfigurationResolutionError):
        resolve_service_configuration(environment={}, env={})


def test_unparseable_service_values_report_as_overridden():
    from openmed.core.config_provenance import resolve_service_configuration

    keys = resolve_service_configuration(
        {
            "OPENMED_SERVICE_JOBS_WORKERS": "two",
            "OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES": "4GiB",
            "OPENMED_SERVICE_THROTTLE_KEY": "GLOBAL",
        }
    ).provenance_report["keys"]

    assert keys["OPENMED_SERVICE_JOBS_WORKERS"]["conflict_category"] == (
        CONFLICT_OVERRIDDEN
    )
    budget = keys["OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES"]
    assert budget["conflict_category"] == CONFLICT_OVERRIDDEN
    assert keys["OPENMED_SERVICE_THROTTLE_KEY"]["conflict_category"] == (
        CONFLICT_SAME_VALUE
    )
