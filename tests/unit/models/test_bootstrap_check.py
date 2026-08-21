"""Focused tests for deterministic offline bootstrap diagnostics."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from openmed.core.model_integrity import sha256_file
from openmed.models import bootstrap_check


def _write_snapshot(cache_dir: Path) -> Path:
    artifact = (
        cache_dir
        / "models--OpenMed--synthetic-model"
        / "snapshots"
        / "synthetic-revision"
        / "config.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"synthetic": true}\n', encoding="utf-8")
    return artifact


def _write_integrity_manifest(cache_dir: Path, artifact: Path) -> Path:
    manifest = cache_dir / "integrity" / "synthetic" / "synthetic.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "openmed.model_integrity.v1",
                "model_id": "OpenMed/synthetic-model",
                "reproducibility_hash": "sha256:" + "0" * 64,
                "artifact_root": str(artifact.parent),
                "artifacts": [
                    {
                        "path": artifact.name,
                        "sha256": sha256_file(artifact),
                        "size": artifact.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_report_is_deterministic_and_keeps_inputs_out_of_output(tmp_path: Path) -> None:
    cache_dir = tmp_path / "synthetic-cache"
    _write_snapshot(cache_dir)

    first = bootstrap_check.run_bootstrap_check(
        cache_dir=cache_dir,
        model_id="OpenMed/synthetic-model",
    )
    second = bootstrap_check.run_bootstrap_check(
        cache_dir=cache_dir,
        model_id="OpenMed/synthetic-model",
    )

    assert first.to_dict() == second.to_dict()
    assert first.ready is True
    assert first.exit_code == bootstrap_check.EXIT_READY
    assert set(first.categories) == {
        "cache",
        "checksum",
        "optional_extras",
        "offline_policy",
    }
    assert first.categories["checksum"].status == bootstrap_check.STATUS_WARN

    serialized = bootstrap_check.render_json(first)
    assert str(cache_dir) not in serialized
    assert "OpenMed/synthetic-model" not in serialized
    assert '"synthetic": true' not in serialized


def test_verified_checksum_is_pass_and_tampering_is_not_ready(tmp_path: Path) -> None:
    cache_dir = tmp_path / "synthetic-cache"
    artifact = _write_snapshot(cache_dir)
    _write_integrity_manifest(cache_dir, artifact)

    clean = bootstrap_check.run_bootstrap_check(
        cache_dir=cache_dir,
        model_id="OpenMed/synthetic-model",
        require_checksum=True,
    )
    assert clean.ready is True
    assert clean.categories["checksum"].to_dict() == {
        "status": "pass",
        "reason": "checksums_verified",
        "manifests_checked": 1,
        "verified": 1,
    }

    artifact.write_text('{"synthetic": false}\n', encoding="utf-8")
    tampered = bootstrap_check.run_bootstrap_check(
        cache_dir=cache_dir,
        model_id="OpenMed/synthetic-model",
        require_checksum=True,
    )
    assert tampered.ready is False
    assert tampered.exit_code == bootstrap_check.EXIT_NOT_READY
    assert tampered.categories["checksum"].reason == "checksum_mismatch"
    assert str(cache_dir) not in bootstrap_check.render_json(tampered)


def test_deeply_nested_manifest_fails_closed_without_leaking_values(
    tmp_path: Path,
) -> None:
    """A manifest decoder recursion failure becomes a value-free diagnostic."""

    cache_dir = tmp_path / "synthetic-cache"
    _write_snapshot(cache_dir)
    manifest = cache_dir / "integrity" / "synthetic" / "hostile.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"model_id":"OpenMed/synthetic-model","nested":'
        + "[" * 2_000
        + "0"
        + "]" * 2_000
        + "}",
        encoding="utf-8",
    )

    report = bootstrap_check.run_bootstrap_check(
        cache_dir=cache_dir,
        model_id="OpenMed/synthetic-model",
        require_checksum=True,
    )

    assert report.ready is False
    assert report.categories["checksum"].reason == "checksum_mismatch"
    assert report.categories["checksum"].to_dict() == {
        "status": "fail",
        "reason": "checksum_mismatch",
        "manifests_checked": 1,
        "verified": 0,
        "failed": 1,
    }
    assert str(cache_dir) not in bootstrap_check.render_json(report)


def test_required_extra_and_offline_policy_fail_without_leaking_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "synthetic-cache"
    _write_snapshot(cache_dir)

    def fake_find_spec(module_name: str):
        return None if module_name == "mlx" else object()

    monkeypatch.setattr(bootstrap_check.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    for name in bootstrap_check.HF_OFFLINE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    report = bootstrap_check.run_bootstrap_check(
        cache_dir=cache_dir,
        required_extras=["mlx"],
        require_offline=True,
    )

    assert report.ready is False
    assert report.categories["optional_extras"].reason == "required_extras_missing"
    assert (
        report.categories["offline_policy"].reason == "offline_configuration_incomplete"
    )
    serialized = bootstrap_check.render_json(report)
    assert "OPENMED_OFFLINE" not in serialized


def test_configured_offline_policy_is_reported_without_network_access(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "synthetic-cache"
    _write_snapshot(cache_dir)
    monkeypatch.setenv("OPENMED_OFFLINE", "true")
    for name in bootstrap_check.HF_OFFLINE_ENV_VARS:
        monkeypatch.setenv(name, "1")

    def fail_network(*_args, **_kwargs):
        raise AssertionError("bootstrap diagnostics attempted network access")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    report = bootstrap_check.run_bootstrap_check(cache_dir=cache_dir)

    assert report.ready is True
    assert report.categories["offline_policy"].to_dict() == {
        "status": "pass",
        "reason": "offline_configured",
        "requested": True,
        "configured": True,
        "network_guard": "requested",
        "dependency_flags": "enabled",
        "source": "environment",
    }


@pytest.mark.parametrize(
    "model_id",
    [
        r"OpenMed\model",
        r"OpenMed\branch\..\outside",
        "C:model",
        "OpenMed/../outside",
        "OpenMed/model/extra",
    ],
)
def test_model_id_rejects_cross_platform_path_syntax(model_id: str) -> None:
    """A model id cannot escape the cache through Windows path semantics."""

    with pytest.raises(ValueError, match="safe string") as raised:
        bootstrap_check.run_bootstrap_check(model_id=model_id)

    assert model_id not in str(raised.value)
    assert raised.value.__cause__ is None


def test_hostile_required_extra_iterable_has_a_value_free_error() -> None:
    """Iterator failures cannot copy their source value into an exception."""

    marker = "synthetic-extra-iterator-value-884"

    class CallerHookFailure(BaseException):
        pass

    def failing_extras():
        yield "hf"
        raise CallerHookFailure(marker)

    with pytest.raises(ValueError, match="could not be read") as raised:
        bootstrap_check.run_bootstrap_check(required_extras=failing_extras())

    assert marker not in str(raised.value)
    assert raised.value.__cause__ is None


def test_hostile_config_property_becomes_a_value_free_failed_category(
    tmp_path: Path,
) -> None:
    """Configuration access failures do not escape or claim offline safety."""

    marker = "synthetic-config-property-value-117"
    cache_dir = tmp_path / "synthetic-cache"
    _write_snapshot(cache_dir)

    class FailingConfig:
        @property
        def local_only(self) -> bool:
            raise RuntimeError(marker)

    report = bootstrap_check.run_bootstrap_check(
        cache_dir=cache_dir,
        config=FailingConfig(),
    )

    assert report.ready is False
    assert report.categories["offline_policy"].reason == "offline_policy_invalid"
    assert marker not in bootstrap_check.render_json(report)


def test_optional_import_finder_failures_are_value_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A hostile import finder is treated as a missing required extra."""

    marker = "synthetic-import-finder-value-492"
    cache_dir = tmp_path / "synthetic-cache"
    _write_snapshot(cache_dir)

    def failing_find_spec(_module_name: str):
        raise RuntimeError(marker)

    monkeypatch.setattr(bootstrap_check.importlib.util, "find_spec", failing_find_spec)
    report = bootstrap_check.run_bootstrap_check(
        cache_dir=cache_dir,
        required_extras=["hf"],
    )

    assert report.ready is False
    assert report.categories["optional_extras"].reason == "required_extras_missing"
    assert marker not in bootstrap_check.render_json(report)


def test_hostile_pathlike_failure_is_value_free() -> None:
    """Path conversion hooks cannot leak values through the public check."""

    marker = "synthetic-pathlike-value-714"

    class FailingPath:
        def __fspath__(self) -> str:
            raise RuntimeError(marker)

    report = bootstrap_check.run_bootstrap_check(  # type: ignore[arg-type]
        cache_dir=FailingPath()
    )

    assert report.ready is False
    assert report.categories["cache"].reason == "cache_missing"
    assert marker not in bootstrap_check.render_json(report)


def test_json_cli_uses_stable_not_ready_exit_and_value_free_output(
    capsys,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "synthetic-cache"
    _write_snapshot(cache_dir)

    exit_code = bootstrap_check.main(
        [
            "--cache-dir",
            str(cache_dir),
            "--model-id",
            "OpenMed/synthetic-model",
            "--require-checksum",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == bootstrap_check.EXIT_NOT_READY
    assert payload["schema_version"] == bootstrap_check.SCHEMA_VERSION
    assert payload["ready"] is False
    assert payload["exit_code"] == bootstrap_check.EXIT_NOT_READY
    assert captured.err == ""
    assert str(cache_dir) not in captured.out
    assert "OpenMed/synthetic-model" not in captured.out


def test_human_cli_reports_categories_without_values(capsys, tmp_path: Path) -> None:
    cache_dir = tmp_path / "synthetic-cache"
    _write_snapshot(cache_dir)

    exit_code = bootstrap_check.main(["--cache-dir", str(cache_dir)])
    captured = capsys.readouterr()

    assert exit_code == bootstrap_check.EXIT_READY
    assert "Bootstrap readiness: READY (exit code 0)" in captured.out
    assert "cache: PASS" in captured.out
    assert "checksum: WARN" in captured.out
    assert "optional_extras: PASS" in captured.out
    assert "offline_policy: PASS" in captured.out
    assert str(cache_dir) not in captured.out


def test_invalid_input_has_usage_exit_without_echoing_input(capsys) -> None:
    exit_code = bootstrap_check.main(["--extra", "not-a-real-extra", "--json"])
    captured = capsys.readouterr()

    assert exit_code == bootstrap_check.EXIT_USAGE
    assert json.loads(captured.out)["error"] == {
        "code": "invalid_input",
        "message": "Invalid bootstrap diagnostic input.",
    }
    assert "not-a-real-extra" not in captured.out


def test_unknown_cli_arguments_have_value_free_usage_output(capsys) -> None:
    """Argument-parser errors cannot echo an unknown option or its value."""

    marker = "synthetic-unknown-cli-value-227"
    exit_code = bootstrap_check.main(["--unknown-option", marker, "--json"])
    captured = capsys.readouterr()

    assert exit_code == bootstrap_check.EXIT_USAGE
    assert json.loads(captured.out)["error"]["code"] == "invalid_input"
    assert captured.err == ""
    assert marker not in captured.out


def test_report_state_is_immutable_after_the_check(tmp_path: Path) -> None:
    """Callers cannot inject source values into a completed safe report."""

    cache_dir = tmp_path / "synthetic-cache"
    _write_snapshot(cache_dir)
    report = bootstrap_check.run_bootstrap_check(cache_dir=cache_dir)
    rendered = bootstrap_check.render_json(report)

    with pytest.raises(TypeError):
        report.categories["unsafe"] = report.categories["cache"]  # type: ignore[index]
    with pytest.raises(TypeError):
        report.categories["cache"].facts["unsafe"] = "synthetic-value"  # type: ignore[index]

    assert bootstrap_check.render_json(report) == rendered


def test_public_report_types_reject_unsafe_or_inconsistent_state() -> None:
    """Exported constructors accept only enumerated, internally consistent data."""

    marker = "synthetic-report-value-663"
    with pytest.raises(ValueError, match="safe metadata") as raised:
        bootstrap_check.DiagnosticCategory(
            status="pass",
            reason="no_required_extras",
            facts={
                "required": [marker],
                "missing_required": [],
                "available_optional": [],
            },
        )
    assert marker not in str(raised.value)

    valid = bootstrap_check.DiagnosticCategory(
        status="pass",
        reason="no_required_extras",
        facts={
            "required": [],
            "missing_required": [],
            "available_optional": [],
        },
    )
    categories = {name: valid for name in bootstrap_check.CATEGORY_ORDER}
    with pytest.raises(ValueError, match="readiness"):
        bootstrap_check.BootstrapReport(ready=False, categories=categories)
