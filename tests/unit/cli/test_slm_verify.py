"""Offline tests for the ``openmed models slm-verify`` command."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import socket
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import openmed.models.clinical_slm_manifest as manifest_module
from openmed.cli import main_module, slm_verify
from openmed.models.clinical_slm_manifest import (
    CLINICAL_SLM_MANIFEST_FILENAME,
    ClinicalSLMArtifact,
    ClinicalSLMArtifactManifest,
    ClinicalSLMQuantization,
)

requires_secure_reads = pytest.mark.skipif(
    not manifest_module._HAS_SECURE_LOCAL_READ,
    reason="package verification requires POSIX no-follow directory descriptors",
)

_MODEL_SENTINEL = "Site7f3a91c2d4e5"
_MEMORY_BUDGET = 10**9
_COMPONENT_FILES = {
    "weights/model.safetensors": b"synthetic-weights",
    "tokenizer/tokenizer.json": b"synthetic-tokenizer",
    "templates/templates.json": b"synthetic-templates",
    "quantization/quantization.json": b"synthetic-quantization",
}
_LICENSES = {
    "weights": "Apache-2.0",
    "tokenizer": "MIT",
    "templates": "BSD-3-Clause",
    "quantization": "Apache-2.0",
}
_CAPABILITY_MANIFEST: dict[str, Any] = {
    "model_id": _MODEL_SENTINEL,
    "supported_tasks": ["bounded_summarization", "nli"],
    "context_limits": {
        "max_context_tokens": 4608,
        "max_input_tokens": 4096,
        "max_output_tokens": 512,
    },
    "quantization": {"scheme": "int4", "bits": 4},
    "required_runtime_features": [],
    "offline": True,
    "human_review_required": True,
    "cloud_fallback": False,
}


def _write_package(
    root: Path,
    *,
    licenses: dict[str, str] | None = None,
    mutate: Sequence[str] | None = None,
    remove: Sequence[str] | None = None,
    capabilities: dict[str, Any] | None = None,
) -> Path:
    mutated = set(mutate or ())
    artifacts: list[ClinicalSLMArtifact] = []
    for relative_path, content in _COMPONENT_FILES.items():
        local_path = root / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(
            b"synthetic-mutated" if relative_path in mutated else content
        )
        artifacts.append(
            ClinicalSLMArtifact(
                component=relative_path.split("/", 1)[0],
                path=relative_path,
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
            )
        )
    for relative_path in remove or ():
        (root / relative_path).unlink()

    manifest = ClinicalSLMArtifactManifest(
        model_id=f"OpenMed/{_MODEL_SENTINEL}",
        revision="a" * 40,
        components=artifacts,
        quantization=ClinicalSLMQuantization(scheme="int4", bits=4),
        licenses=_LICENSES if licenses is None else licenses,
        supported_tasks=["clinical-summarization", "nli"],
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / CLINICAL_SLM_MANIFEST_FILENAME).write_text(
        manifest.to_json() + "\n",
        encoding="utf-8",
    )
    capability_manifest = _CAPABILITY_MANIFEST if capabilities is None else capabilities
    (root / "clinical-slm-capabilities.json").write_text(
        json.dumps(capability_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def _run(*arguments: str) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            exit_code = main_module.main(list(arguments))
        except SystemExit as exc:  # argparse exits for usage errors and --help
            exit_code = 0 if exc.code is None else int(exc.code)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def _verify_args(package: Path, *extra: str) -> list[str]:
    return [
        "models",
        "slm-verify",
        str(package),
        "--task",
        "bounded_summarization",
        "--memory-budget",
        str(_MEMORY_BUDGET),
        *extra,
    ]


def _stub_verification(**overrides: Any) -> SimpleNamespace:
    payload: dict[str, Any] = {
        "verified": True,
        "component_count": len(_COMPONENT_FILES),
        "executable_component_count": 0,
        "bytes_checked": sum(len(item) for item in _COMPONENT_FILES.values()),
        "manifest_digest": "sha256:" + "0" * 64,
        "reason_codes": [],
    }
    payload.update(overrides)
    return SimpleNamespace(to_dict=lambda: dict(payload))


@pytest.fixture()
def supported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(slm_verify, "_secure_local_reads_available", lambda: True)


@pytest.fixture()
def stubbed_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        slm_verify,
        "verify_clinical_slm_package",
        lambda *_args, **_kwargs: _stub_verification(),
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--memory-budget", "1024"],
        ["--task", "nli"],
        ["--task", "nli", "--memory-budget", "0"],
        ["--task", "nli", "--memory-budget", "-1024"],
        ["--task", "nli", "--memory-budget", "plenty"],
        ["--task", "nli", "--memory-budget", "1024", "--headroom-bytes", "-1"],
    ],
)
def test_usage_errors_exit_two(tmp_path: Path, arguments: list[str]) -> None:
    package = _write_package(tmp_path / "package")

    exit_code, stdout, stderr = _run("models", "slm-verify", str(package), *arguments)

    assert exit_code == 2
    assert "usage: openmed models slm-verify" in stderr
    assert stdout == ""
    assert _MODEL_SENTINEL not in stderr


def test_help_lists_the_command_with_its_own_usage_line() -> None:
    exit_code, stdout, _ = _run("models", "--help")

    assert exit_code == 0
    assert "slm-verify" in stdout

    exit_code, stdout, stderr = _run("models", "slm-verify", "--help")

    assert exit_code == 0
    assert "usage: openmed models slm-verify" in stdout
    assert "--memory-budget" in stdout and "--task" in stdout
    assert stderr == ""


def test_unsupported_platform_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(manifest_module, "_HAS_SECURE_LOCAL_READ", False)
    package = _write_package(tmp_path / "package")

    exit_code, stdout, stderr = _run(*_verify_args(package, "--json"))

    payload = json.loads(stdout)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_platform"
    assert payload["error"]["message"] == (
        "Clinical SLM package verification needs secure local reads, which this "
        "platform does not provide"
    )
    assert _MODEL_SENTINEL not in stdout + stderr
    assert str(package) not in stdout + stderr


def test_report_contains_metadata_only_and_passes(
    supported_platform: None,
    stubbed_verification: None,
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path / "package")

    exit_code, stdout, stderr = _run(*_verify_args(package, "--json"))

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["command"] == "models slm-verify"
    report = payload["data"]
    assert set(report) == {
        "verdict",
        "manifest",
        "verification",
        "capabilities",
        "memory",
        "reason_codes",
        "network",
    }
    assert report["verdict"] == "pass"
    assert report["reason_codes"] == []
    assert report["network"] == {"mandatory": False}
    assert report["manifest"]["component_count"] == len(_COMPONENT_FILES)
    assert report["manifest"]["manifest_digest"]
    assert report["capabilities"]["decision"] == "supported"
    assert report["capabilities"]["requested_capabilities"] == ["bounded_summarization"]
    assert report["memory"]["decision"] == "accepted"
    assert report["memory"]["status"] == "accept"
    assert _MODEL_SENTINEL not in stdout + stderr
    assert str(package) not in stdout + stderr
    for relative_path in _COMPONENT_FILES:
        assert relative_path not in stdout + stderr


def test_human_output_is_rendered_without_json(
    supported_platform: None,
    stubbed_verification: None,
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path / "package")

    exit_code, stdout, stderr = _run(*_verify_args(package))

    assert exit_code == 0
    assert stdout.startswith("slm-verify: pass;")
    assert "reason_codes=none" in stdout
    assert stderr == ""
    assert _MODEL_SENTINEL not in stdout


@pytest.mark.parametrize(
    "reason_code",
    ["component_mutated", "component_unreadable", "package_invalid"],
)
def test_verification_failures_are_reported_as_reason_codes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reason_code: str,
) -> None:
    monkeypatch.setattr(slm_verify, "_secure_local_reads_available", lambda: True)
    monkeypatch.setattr(
        slm_verify,
        "verify_clinical_slm_package",
        lambda *_args, **_kwargs: _stub_verification(
            verified=False,
            reason_codes=[reason_code],
        ),
    )
    package = _write_package(tmp_path / "package")

    exit_code, stdout, _ = _run(*_verify_args(package, "--json"))

    report = json.loads(stdout)["data"]
    assert exit_code == 1
    assert report["verdict"] == "fail"
    assert report["verification"]["verified"] is False
    assert reason_code in report["reason_codes"]


def test_verification_errors_become_closed_reason_codes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(slm_verify, "_secure_local_reads_available", lambda: True)

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("synthetic read failure")

    monkeypatch.setattr(slm_verify, "verify_clinical_slm_package", explode)
    package = _write_package(tmp_path / "package")

    exit_code, stdout, _ = _run(*_verify_args(package, "--json"))

    report = json.loads(stdout)["data"]
    assert exit_code == 1
    assert report["reason_codes"] == ["package_invalid"]
    assert "synthetic read failure" not in stdout


def test_invalid_manifest_uses_a_documented_code(
    supported_platform: None,
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path / "package")
    manifest_path = package / CLINICAL_SLM_MANIFEST_FILENAME
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest_payload["licenses"]:
        if entry["component"] == "weights":
            entry["spdx_id"] = "synthetic-proprietary-license"
    manifest_path.write_text(json.dumps(manifest_payload) + "\n", encoding="utf-8")

    exit_code, stdout, stderr = _run(*_verify_args(package, "--json"))

    payload = json.loads(stdout)
    assert exit_code == 1
    assert payload["error"]["code"] == "slm_manifest_invalid"
    assert "unknown_license" in payload["error"]["message"]
    assert "synthetic-proprietary-license" not in stdout + stderr


def test_unknown_capability_is_a_usage_error(
    supported_platform: None,
    stubbed_verification: None,
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path / "package")

    exit_code, stdout, _ = _run(
        "models",
        "slm-verify",
        str(package),
        "--task",
        "clinical-ner",
        "--memory-budget",
        str(_MEMORY_BUDGET),
        "--json",
    )

    payload = json.loads(stdout)
    assert exit_code == 2
    assert payload["error"]["code"] == "slm_unknown_task"
    assert "clinical-ner" not in stdout


def test_undeclared_task_fails_closed(
    supported_platform: None,
    stubbed_verification: None,
    tmp_path: Path,
) -> None:
    capabilities = dict(_CAPABILITY_MANIFEST)
    capabilities["supported_tasks"] = ["bounded_summarization"]
    package = _write_package(tmp_path / "package", capabilities=capabilities)

    exit_code, stdout, _ = _run(
        "models",
        "slm-verify",
        str(package),
        "--task",
        "nli",
        "--memory-budget",
        str(_MEMORY_BUDGET),
        "--json",
    )

    report = json.loads(stdout)["data"]
    assert exit_code == 1
    assert report["verdict"] == "fail"
    assert report["capabilities"]["decision"] == "unsupported"
    assert "task_not_declared" in report["reason_codes"]


def test_packages_over_budget_fail_closed(
    supported_platform: None,
    stubbed_verification: None,
    tmp_path: Path,
) -> None:
    package = _write_package(tmp_path / "package")

    exit_code, stdout, _ = _run(
        "models",
        "slm-verify",
        str(package),
        "--task",
        "nli",
        "--memory-budget",
        "1",
        "--json",
    )

    report = json.loads(stdout)["data"]
    assert exit_code == 1
    assert report["verdict"] == "fail"
    assert report["memory"]["decision"] == "rejected"
    assert sorted(report["memory"]["reason_codes"]) == [
        "headroom_insufficient",
        "memory_budget_exceeded",
    ]


def test_command_opens_no_network_sockets(
    monkeypatch: pytest.MonkeyPatch,
    supported_platform: None,
    stubbed_verification: None,
    tmp_path: Path,
) -> None:
    def deny_socket(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the verification command attempted network access")

    monkeypatch.setattr(socket, "socket", deny_socket)
    monkeypatch.setattr(socket, "create_connection", deny_socket)
    package = _write_package(tmp_path / "package")

    exit_code, stdout, _ = _run(*_verify_args(package, "--json"))

    assert exit_code == 0
    assert json.loads(stdout)["data"]["verdict"] == "pass"


@requires_secure_reads
def test_synthetic_package_verifies_offline(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "package")

    exit_code, stdout, _ = _run(
        "models",
        "slm-verify",
        str(package),
        "--task",
        "bounded_summarization",
        "--task",
        "nli",
        "--memory-budget",
        str(_MEMORY_BUDGET),
        "--json",
    )

    report = json.loads(stdout)["data"]
    assert exit_code == 0
    assert report["verdict"] == "pass"
    assert report["verification"]["verified"] is True
    assert report["verification"]["bytes_checked"] == sum(
        len(item) for item in _COMPONENT_FILES.values()
    )
    assert report["capabilities"]["requested_capabilities"] == [
        "bounded_summarization",
        "nli",
    ]


@requires_secure_reads
def test_changed_component_fails_closed(tmp_path: Path) -> None:
    package = _write_package(
        tmp_path / "package",
        mutate=["weights/model.safetensors"],
    )

    exit_code, stdout, _ = _run(*_verify_args(package, "--json"))

    report = json.loads(stdout)["data"]
    assert exit_code == 1
    assert report["verdict"] == "fail"
    assert "component_digest_mismatch" in report["reason_codes"]
    assert "weights/model.safetensors" not in stdout


@requires_secure_reads
def test_missing_component_fails_closed(tmp_path: Path) -> None:
    package = _write_package(
        tmp_path / "package",
        remove=["tokenizer/tokenizer.json"],
    )

    exit_code, stdout, _ = _run(*_verify_args(package, "--json"))

    report = json.loads(stdout)["data"]
    assert exit_code == 1
    assert report["verdict"] == "fail"
    assert "component_unreadable" in report["reason_codes"]


@requires_secure_reads
def test_manifest_digest_is_required_for_persisted_packages(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "package")
    manifest_path = package / CLINICAL_SLM_MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("manifest_digest", None)
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    exit_code, stdout, _ = _run(*_verify_args(package, "--json"))

    payload = json.loads(stdout)
    assert exit_code == 1
    assert payload["error"]["code"] == "slm_manifest_invalid"
    assert "manifest_digest_required" in payload["error"]["message"]
