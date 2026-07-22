"""Tests for the offline wheelhouse and air-gap bundle commands."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from openmed.cli import airgap, main_module


def _fake_download_wheels(wheels_dir: Path, **_kwargs: object) -> None:
    (wheels_dir / "openmed-1-py3-none-any.whl").write_bytes(b"wheel-data")
    (wheels_dir / "dependency-1-py3-none-any.whl").write_bytes(b"dependency")


def _fake_prefetch_model(model: str, *, cache_dir: str) -> str:
    repo_id = airgap.resolve_repo_id(model)
    repo_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"deberta-v2"}')
    blobs = repo_dir / "blobs"
    blobs.mkdir()
    (blobs / "model-data").write_bytes(b"model-data")
    (snapshot / "model.safetensors").symlink_to(
        Path("..") / ".." / "blobs" / "model-data"
    )
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    return str(snapshot)


@pytest.fixture
def fake_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(airgap, "_download_wheels", _fake_download_wheels)
    monkeypatch.setattr(airgap, "prefetch_model", _fake_prefetch_model)


def _manifest(bundle: Path) -> dict[str, object]:
    return json.loads((bundle / airgap.MANIFEST_NAME).read_text(encoding="utf-8"))


def test_airgap_commands_are_registered_with_current_defaults() -> None:
    parser = main_module.build_parser()

    args = parser.parse_args(["airgap", "bundle", "kit"])

    assert args.command == "airgap"
    assert args.airgap_command == "bundle"
    assert args.output == Path("kit")
    assert args.extras is None
    assert args.models is None
    assert args.target_platform is None
    assert args.python_version is None


def test_target_python_accepts_common_cp311_spellings() -> None:
    parser = main_module.build_parser()

    dotted = parser.parse_args(["airgap", "bundle", "kit", "--python-version", "3.11"])
    compact = parser.parse_args(["airgap", "bundle", "kit", "--target-python", "cp311"])

    assert dotted.python_version == "3.11"
    assert compact.python_version == "3.11"


def test_linux_aarch64_cp311_build_uses_binary_only_target_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)
        destination = Path(command[command.index("--dest") + 1])
        (destination / "openmed-1-py3-none-any.whl").write_bytes(b"wheel")

    monkeypatch.setattr(airgap.subprocess, "run", fake_run)
    monkeypatch.setattr(airgap, "prefetch_model", _fake_prefetch_model)
    output = tmp_path / "kit"

    airgap.build_bundle(
        output,
        target_platform="linux/aarch64",
        python_version="3.11",
    )

    command = calls[0]
    assert "--only-binary=:all:" in command
    assert command[command.index("--platform") + 1] == "manylinux2014_aarch64"
    assert command[command.index("--python-version") + 1] == "3.11"
    assert command[command.index("--implementation") + 1] == "cp"
    assert command[command.index("--abi") + 1] == "cp311"
    assert command[-1].startswith("openmed[cli,hf]==")
    manifest = _manifest(output)
    assert manifest["target"] == {
        "abi": "cp311",
        "implementation": "cp",
        "platform": "manylinux2014_aarch64",
        "python_version": "3.11",
    }


def test_bundle_contains_wheels_models_docs_installer_and_exact_manifest(
    fake_downloads: None,
    tmp_path: Path,
) -> None:
    output = tmp_path / "kit"

    result = airgap.build_bundle(output)

    manifest = _manifest(output)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    paths = {item["path"] for item in artifacts}
    assert "INSTALL.md" in paths
    assert "install.sh" in paths
    assert "docs/offline-install.md" in paths
    assert any(path.startswith("wheels/") for path in paths)
    assert any(path.endswith("model.safetensors") for path in paths)
    assert manifest["extras"] == ["cli"]
    assert manifest["models"] == [
        {
            "repo_id": airgap.DEFAULT_MODEL,
            "snapshot_path": (
                "models/hub/models--OpenMed--OpenMed-PII-"
                "SuperClinical-Small-44M-v1/snapshots/abc123"
            ),
        }
    ]
    exact_size = sum(item["size_bytes"] for item in artifacts)
    assert manifest["artifact_count"] == len(artifacts)
    assert manifest["total_size_bytes"] == exact_size
    assert manifest["bundle_size_bytes"] == exact_size
    assert exact_size == result.total_size_bytes
    assert exact_size <= airgap.DEFAULT_BUNDLE_LIMIT_BYTES
    assert os.access(output / "install.sh", os.X_OK)
    assert airgap.verify_bundle(output) == []


@pytest.mark.parametrize(
    "artifact_glob",
    [
        "wheels/*.whl",
        "models/hub/*/snapshots/*/model.safetensors",
    ],
)
def test_verify_rejects_one_byte_corruption_in_wheel_or_model(
    fake_downloads: None,
    tmp_path: Path,
    artifact_glob: str,
) -> None:
    output = tmp_path / artifact_glob.split("/", 1)[0]
    airgap.build_bundle(output)
    artifact = next(output.glob(artifact_glob))
    original = artifact.read_bytes()

    artifact.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    errors = airgap.verify_bundle(output)
    relative = artifact.relative_to(output).as_posix()
    if artifact.is_symlink():
        assert any(error.startswith("sha256 mismatch for models/") for error in errors)
    else:
        assert f"sha256 mismatch for {relative}" in errors


def test_manifest_counts_symlink_itself_without_double_counting_target(
    fake_downloads: None,
    tmp_path: Path,
) -> None:
    output = tmp_path / "kit"
    airgap.build_bundle(output)
    manifest = _manifest(output)
    artifacts = manifest["artifacts"]
    symlink = next(item for item in artifacts if item["type"] == "symlink")
    link_path = output / symlink["path"]

    assert symlink["link_target"] == "../../blobs/model-data"
    assert os.readlink(link_path).replace(os.sep, "/") == symlink["link_target"]
    assert symlink["size_bytes"] == len(symlink["link_target"].encode("utf-8"))
    assert symlink["size_bytes"] != link_path.stat().st_size


def test_bundle_rejects_model_symlink_that_escapes_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_prefetch_with_escape(model: str, *, cache_dir: str) -> str:
        repo_id = airgap.resolve_repo_id(model)
        repo_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
        snapshot = repo_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        outside = Path(cache_dir).parents[2] / "outside-model-data"
        outside.write_bytes(b"outside")
        (snapshot / "model.safetensors").symlink_to(os.path.relpath(outside, snapshot))
        return str(snapshot)

    monkeypatch.setattr(airgap, "_download_wheels", _fake_download_wheels)
    monkeypatch.setattr(airgap, "prefetch_model", fake_prefetch_with_escape)

    with pytest.raises(ValueError, match="path escapes bundle root"):
        airgap.build_bundle(tmp_path / "kit")


def test_verify_rejects_missing_and_unexpected_artifacts(
    fake_downloads: None,
    tmp_path: Path,
) -> None:
    output = tmp_path / "kit"
    airgap.build_bundle(output)
    wheel = next((output / "wheels").glob("*.whl"))
    wheel.unlink()
    (output / "untracked.txt").write_text("not declared", encoding="utf-8")

    errors = airgap.verify_bundle(output)

    assert any(error.startswith("missing artifact: wheels/") for error in errors)
    assert "unexpected artifact: untracked.txt" in errors


def test_archive_bundle_verifies_after_safe_extraction(
    fake_downloads: None,
    tmp_path: Path,
) -> None:
    output = tmp_path / "kit.tar.gz"

    airgap.build_bundle(output, archive=True)

    assert output.is_file()
    assert airgap.verify_bundle(output) == []


def test_archive_verifier_rejects_path_traversal(tmp_path: Path) -> None:
    source = tmp_path / "payload"
    source.write_text("bad", encoding="utf-8")
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="../outside")

    with pytest.raises(ValueError, match="unsafe archive member"):
        airgap.verify_bundle(archive)


@pytest.mark.skipif(os.name == "nt", reason="install.sh is a POSIX installer")
def test_installer_uses_local_wheels_and_copies_standard_model_cache(
    fake_downloads: None,
    tmp_path: Path,
) -> None:
    output = tmp_path / "kit"
    airgap.build_bundle(output)
    calls = tmp_path / "python-calls.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    target_cache = tmp_path / "offline-cache"
    offline_home = tmp_path / "offline-home"
    offline_home.mkdir()

    completed = subprocess.run(
        [str(output / "install.sh")],
        env={
            **os.environ,
            "PYTHON": str(fake_python),
            "HF_HUB_CACHE": str(target_cache),
            "HOME": str(offline_home),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    pip_args = calls.read_text(encoding="utf-8")
    assert "-m pip install --no-index --find-links" in pip_args
    assert str(output / "wheels") in pip_args
    assert "openmed[cli,hf]==" in pip_args
    assert next(target_cache.glob("models--*/snapshots/*/model.safetensors"))
    openmed_cache = offline_home / ".cache" / "openmed"
    assert next(openmed_cache.glob("models--*")).is_symlink()
    assert "export OPENMED_OFFLINE=1" in completed.stdout
    assert "openmed doctor" in completed.stdout


def test_default_bundle_enforces_900_mb_limit(
    fake_downloads: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(airgap, "DEFAULT_BUNDLE_LIMIT_BYTES", 1)

    with pytest.raises(RuntimeError, match="larger than the 900 MB limit"):
        airgap.build_bundle(tmp_path / "too-large")

    assert not (tmp_path / "too-large").exists()


def test_bundle_refuses_to_overwrite_existing_output(
    fake_downloads: None,
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        airgap.build_bundle(output)


def test_verify_cli_exits_nonzero_for_corruption(
    fake_downloads: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "kit"
    airgap.build_bundle(output)
    model = next(output.glob("models/hub/*/snapshots/*/model.safetensors"))
    model.write_bytes(b"tampered!")

    exit_code = main_module.main(["airgap", "verify", str(output)])

    assert exit_code == 1
    assert "sha256 mismatch" in capsys.readouterr().err
