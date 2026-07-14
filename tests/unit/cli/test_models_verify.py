"""Tests for the offline ``openmed models verify`` command."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from openmed.cli import main_module, typer_app
from openmed.core.model_integrity import (
    ARTIFACT_MANIFEST_SCHEMA,
    sha256_file,
)


def _write_cached_model(cache_dir: Path) -> Path:
    artifact_root = cache_dir / "snapshots" / "fixture-revision"
    artifact_root.mkdir(parents=True)
    artifact = artifact_root / "model.safetensors"
    artifact.write_bytes(b"untampered cached artifact")

    sidecar = cache_dir / "integrity" / "fixture" / "fixture-revision.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_MANIFEST_SCHEMA,
                "model_id": "OpenMed/fixture-model",
                "reproducibility_hash": "sha256:" + "2" * 64,
                "artifact_root": str(artifact_root),
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
    return artifact


def test_models_verify_all_is_offline_and_returns_failure_on_tamper(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact = _write_cached_model(tmp_path)
    monkeypatch.setattr(
        typer_app,
        "_load_config",
        lambda _path: SimpleNamespace(cache_dir=str(tmp_path)),
    )

    def fail_network(*_args, **_kwargs):
        raise AssertionError("models verify attempted network access")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    runner = CliRunner()
    app = typer_app.build_app()

    clean = runner.invoke(app, ["models", "verify", "--all"])
    assert clean.exit_code == 0, clean.output
    assert "PASS" in clean.output
    assert "OpenMed/fixture" in clean.output

    artifact.write_bytes(b"tampered cached artifact")
    tampered = runner.invoke(app, ["models", "verify", "--all"])
    assert tampered.exit_code == 1
    assert "FAIL" in tampered.output
    assert "expected" in tampered.output.lower()
    assert "actual" in tampered.output.lower()


def test_models_verify_requires_model_id_or_all() -> None:
    result = CliRunner().invoke(typer_app.build_app(), ["models", "verify"])
    assert result.exit_code != 0
    assert "Provide MODEL_ID or --all" in result.output


def test_installed_cli_models_verify_uses_same_offline_verdict(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    artifact = _write_cached_model(tmp_path)
    monkeypatch.setattr(
        main_module,
        "_load_and_apply_config",
        lambda _args: SimpleNamespace(cache_dir=str(tmp_path)),
    )
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("models verify attempted network access")
        ),
    )

    assert main_module.main(["models", "verify", "--all"]) == 0
    clean = capsys.readouterr()
    assert "PASS" in clean.out
    assert clean.err == ""

    artifact.write_bytes(b"tampered cached artifact")
    assert main_module.main(["models", "verify", "--all"]) == 1
    tampered = capsys.readouterr()
    assert "FAIL" in tampered.out
    assert "expected sha256:" in tampered.err
    assert "actual sha256:" in tampered.err
