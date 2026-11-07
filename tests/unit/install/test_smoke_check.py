"""Focused tests for the offline install smoke check."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.install import smoke_check


def _fake_install_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_name = "python.exe" if os.name == "nt" else "python"
    entry_point_name = "openmed.exe" if os.name == "nt" else "openmed"
    python_executable = bin_dir / python_name
    python_executable.touch(mode=0o755)
    (bin_dir / entry_point_name).touch(mode=0o755)
    return python_executable


def _completed(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, "")


def test_report_serialization_is_compact_and_stable() -> None:
    report = smoke_check.SmokeReport(
        status="passed",
        checks=(
            smoke_check.CheckResult(
                name="synthetic_offline_command",
                status="passed",
                details={
                    "change_count": 1,
                    "document_hash": "sha256:" + "a" * 64,
                },
            ),
        ),
    )

    rendered = report.to_json()
    assert "\n" not in rendered
    assert json.loads(rendered) == {
        "checks": [
            {
                "change_count": 1,
                "document_hash": "sha256:" + "a" * 64,
                "name": "synthetic_offline_command",
                "status": "passed",
            }
        ],
        "offline": True,
        "schema_version": 1,
        "status": "passed",
    }


def test_run_smoke_check_uses_offline_clean_environment(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    python_executable = _fake_install_bin(tmp_path)

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        if command[-1] == "--version":
            return _completed(command, stdout="openmed 2.0.0\n")
        if command[-3:] == ["models", "validate", "--json"]:
            return _completed(
                command,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "ok": True,
                            "violation_count": 0,
                            "messages": ["manifest: OK (3 rows checked)"],
                        },
                    }
                ),
            )
        return _completed(
            command,
            stdout=json.dumps(
                {
                    "change_count": 1,
                    "document_hash": "sha256:" + "b" * 64,
                    "entry_point_declared": True,
                    "package_version": "2.0.0",
                    "surface_hash": "sha256:" + "c" * 64,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    report = smoke_check.run_smoke_check(
        python_executable=str(python_executable),
        runner=runner,
    )

    assert report.status == "passed"
    assert [check.name for check in report.checks] == [
        "entry_point",
        "bundled_manifest",
        "synthetic_offline_command",
    ]
    assert len(calls) == 4
    for call in calls:
        assert call["env"]["OPENMED_OFFLINE"] == "1"
        assert call["env"]["HF_HUB_OFFLINE"] == "1"
        assert call["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert call["env"]["HF_DATASETS_OFFLINE"] == "1"
        assert call["env"]["PYTHONNOUSERSITE"] == "1"
        assert call["env"]["PATH"] == str(python_executable.parent)
        assert call["cwd"] != Path.cwd()


def test_only_the_brief_probe_gets_the_longer_child_timeout(tmp_path: Path) -> None:
    timeouts: dict[str, float] = {}
    python_executable = _fake_install_bin(tmp_path)

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "--_brief-probe" in command:
            timeouts["brief_probe"] = kwargs["timeout"]
            return _completed(command, stdout="{}")
        timeouts.setdefault("other", kwargs["timeout"])
        assert kwargs["timeout"] == timeouts["other"]
        if command[-1] == "--version":
            return _completed(command, stdout="openmed 2.0.0\n")
        if command[-3:] == ["models", "validate", "--json"]:
            return _completed(
                command,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "ok": True,
                            "violation_count": 0,
                            "messages": ["manifest: OK (3 rows checked)"],
                        },
                    }
                ),
            )
        return _completed(
            command,
            stdout=json.dumps(
                {
                    "change_count": 1,
                    "document_hash": "sha256:" + "b" * 64,
                    "entry_point_declared": True,
                    "package_version": "2.0.0",
                    "surface_hash": "sha256:" + "c" * 64,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    smoke_check.run_smoke_check(
        python_executable=str(python_executable),
        runner=runner,
        brief_profile="core",
    )

    assert timeouts == {
        "brief_probe": smoke_check.BRIEF_PROBE_TIMEOUT_SECONDS,
        "other": smoke_check.COMMAND_TIMEOUT_SECONDS,
    }


def test_manifest_check_rejects_failure_without_echoing_child_output() -> None:
    raw_sensitive_value = "synthetic-only-value-that-must-not-be-reported"
    result = _completed(
        ["openmed", "models", "validate", "--json"],
        returncode=1,
        stdout=json.dumps(
            {
                "ok": False,
                "error": {"message": raw_sensitive_value},
            }
        ),
    )

    check = smoke_check._manifest_check(result)

    assert check.status == "failed"
    assert raw_sensitive_value not in json.dumps(check.to_dict())


def test_manifest_check_requires_positive_validated_row_count() -> None:
    result = _completed(
        ["openmed", "models", "validate", "--json"],
        stdout=json.dumps(
            {
                "ok": True,
                "data": {
                    "ok": True,
                    "violation_count": 0,
                    "messages": ["manifest: OK"],
                },
            }
        ),
    )

    assert smoke_check._manifest_check(result).to_dict() == {
        "name": "bundled_manifest",
        "reason": "invalid_result",
        "status": "failed",
    }


def test_synthetic_check_rejects_nondeterministic_output(tmp_path: Path) -> None:
    outputs = iter(
        [
            json.dumps(
                {
                    "change_count": 1,
                    "document_hash": "sha256:" + "a" * 64,
                    "entry_point_declared": True,
                    "package_version": "2.0.0",
                    "surface_hash": "sha256:" + "b" * 64,
                }
            ),
            json.dumps(
                {
                    "change_count": 1,
                    "document_hash": "sha256:" + "c" * 64,
                    "entry_point_declared": True,
                    "package_version": "2.0.0",
                    "surface_hash": "sha256:" + "d" * 64,
                }
            ),
        ]
    )

    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return _completed(command, stdout=next(outputs))

    check = smoke_check._synthetic_check(
        "/tmp/fake-env/bin/python",
        expected_version="2.0.0",
        cwd=tmp_path,
        environment={"OPENMED_OFFLINE": "1"},
        runner=runner,
    )

    assert check.to_dict() == {
        "name": "synthetic_offline_command",
        "reason": "non_deterministic_output",
        "status": "failed",
    }


def test_main_redacts_unexpected_exception(monkeypatch: Any, capsys: Any) -> None:
    raw_sensitive_value = "synthetic-only-value-that-must-not-be-reported"

    def fail(**_: Any) -> smoke_check.SmokeReport:
        raise RuntimeError(raw_sensitive_value)

    monkeypatch.setattr(smoke_check, "run_smoke_check", fail)

    assert smoke_check.main([]) == 1
    output = capsys.readouterr().out
    assert raw_sensitive_value not in output
    assert json.loads(output)["checks"] == [
        {
            "name": "smoke_check",
            "reason": "internal_error",
            "status": "failed",
        }
    ]


def test_run_smoke_check_rejects_console_package_version_mismatch(
    tmp_path: Path,
) -> None:
    python_executable = _fake_install_bin(tmp_path)

    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if command[-1] == "--version":
            return _completed(command, stdout="openmed 2.0.0\n")
        if command[-3:] == ["models", "validate", "--json"]:
            return _completed(
                command,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "ok": True,
                            "violation_count": 0,
                            "messages": ["manifest: OK (3 rows checked)"],
                        },
                    }
                ),
            )
        return _completed(
            command,
            stdout=json.dumps(
                {
                    "change_count": 1,
                    "document_hash": "sha256:" + "b" * 64,
                    "entry_point_declared": True,
                    "package_version": "2.0.1",
                    "surface_hash": "sha256:" + "c" * 64,
                }
            ),
        )

    report = smoke_check.run_smoke_check(
        python_executable=str(python_executable),
        runner=runner,
    )

    assert report.status == "failed"
    assert report.checks[-1].to_dict() == {
        "name": "synthetic_offline_command",
        "reason": "version_mismatch",
        "status": "failed",
    }
    assert "2.0.0" not in report.to_json()
    assert "2.0.1" not in report.to_json()


def test_run_smoke_check_does_not_fall_back_to_unrelated_path_entry_point(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    python_dir = tmp_path / "python-bin"
    python_dir.mkdir()
    python_name = "python.exe" if os.name == "nt" else "python"
    entry_point_name = "openmed.exe" if os.name == "nt" else "openmed"
    python_executable = python_dir / python_name
    python_executable.touch(mode=0o755)
    unrelated_dir = tmp_path / "unrelated-bin"
    unrelated_dir.mkdir()
    (unrelated_dir / entry_point_name).touch(mode=0o755)
    monkeypatch.setenv("PATH", f"{unrelated_dir}{os.pathsep}{python_dir}")

    report = smoke_check.run_smoke_check(
        python_executable=str(python_executable),
        runner=lambda *args, **kwargs: _completed(list(args[0])),
    )

    assert report.to_dict()["checks"] == [
        {"name": "entry_point", "reason": "not_installed", "status": "failed"}
    ]


def test_unsafe_version_output_is_not_relayed(tmp_path: Path) -> None:
    raw_sensitive_value = "synthetic-only-value-that-must-not-be-reported"
    python_executable = _fake_install_bin(tmp_path)

    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return _completed(command, stdout=f"openmed {raw_sensitive_value}\n")

    report = smoke_check.run_smoke_check(
        python_executable=str(python_executable),
        runner=runner,
    )

    rendered = report.to_json()
    assert raw_sensitive_value not in rendered
    assert report.checks[0].to_dict() == {
        "name": "entry_point",
        "reason": "invalid_version_output",
        "status": "failed",
    }


def test_probe_profiles_order_service_transports_before_network_check():
    core = smoke_check._expected_checks("core")
    adapters = smoke_check._expected_checks("adapters")
    assert not set(core) & set(smoke_check.ADAPTER_CHECKS)
    assert adapters[-1] == core[-1] == "no_network"
    assert adapters[adapters.index("brief_mcp") + 1 : -1] == (
        "rest_pii_extract",
        "graphql_query",
        "grpc_stubs",
        "grpc_loopback",
    )


def test_brief_probe_report_requires_the_complete_ordered_contract(subtests):
    for profile in smoke_check.BRIEF_PROFILES:
        with subtests.test(profile=profile):
            names = list(smoke_check._expected_checks(profile))
            rows = [{"name": name, "status": "passed"} for name in names]
            result = _completed(
                [], stdout=json.dumps({"status": "passed", "checks": rows})
            )
            assert [
                c.name for c in smoke_check._brief_probe_results(result, profile)
            ] == names
            rows.pop()
            result = _completed(
                [], stdout=json.dumps({"status": "passed", "checks": rows})
            )
            assert (
                smoke_check._brief_probe_results(result, profile)[-1].status == "failed"
            )


def test_brief_probe_rejects_unsafe_child_output(subtests):
    for mutation in ["extra", "name", "reason", "status", "exit"]:
        with subtests.test(mutation=mutation):
            secret = "synthetic-install-sensitive-value"
            rows = [
                dict(name="installed_origin", status="failed", reason="contract_failed")
            ]
            payload = dict(status="failed", checks=rows)
            code = 1
            if mutation == "extra":
                rows[0]["text"] = secret
            elif mutation == "name":
                rows[0]["name"] = secret
            elif mutation == "reason":
                rows[0]["reason"] = secret
            elif mutation == "status":
                payload["status"] = secret
            else:
                code = 0
            checks = smoke_check._brief_probe_results(
                _completed([], returncode=code, stdout=json.dumps(payload)), "core"
            )
            assert checks == (smoke_check._failed("brief_probe", "invalid_result"),)
            assert secret not in json.dumps([check.to_dict() for check in checks])


def test_brief_probe_preserves_a_specific_controlled_failure():
    rows = [
        smoke_check._passed("installed_origin"),
        smoke_check._failed("packaged_resources", "contract_failed"),
    ]
    report = smoke_check.SmokeReport("failed", tuple(rows))
    result = _completed([], returncode=1, stdout=report.to_json())
    assert smoke_check._brief_probe_results(result, "core") == tuple(rows)


def _synthetic_artifact(directory: Path, kind: str, resources: tuple[str, ...]) -> Path:
    import io
    import tarfile
    import zipfile

    names = ["openmed/" + resource for resource in resources]
    names.append("openmed/__init__.py")
    if kind == "wheel":
        artifact = directory / "openmed-0.0.0-py3-none-any.whl"
        with zipfile.ZipFile(artifact, "w") as archive:
            for name in names:
                archive.writestr(name, b'{"synthetic": true}')
        return artifact
    artifact = directory / "openmed-0.0.0.tar.gz"
    with tarfile.open(artifact, "w:gz") as archive:
        for name in names:
            content = b'{"synthetic": true}'
            member = tarfile.TarInfo("openmed-0.0.0/" + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return artifact


def test_negative_controls_change_only_disposable_artifacts(subtests, tmp_path):
    import hashlib
    import tarfile
    import zipfile

    for control, (resources, _, _) in smoke_check.NEGATIVE_CONTROLS.items():
        for kind in ["wheel", "sdist"]:
            with subtests.test(control=control, kind=kind):
                case = tmp_path / f"{control}-{kind}"
                output = case / "control"
                output.mkdir(parents=True)
                artifact = _synthetic_artifact(case, kind, resources)
                before = hashlib.sha256(artifact.read_bytes()).hexdigest()
                damaged = smoke_check._omit_packaged_resources(
                    artifact, output, resources
                )
                assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before
                assert damaged.parent == output
                if kind == "wheel":
                    with zipfile.ZipFile(damaged) as archive:
                        assert archive.namelist() == ["openmed/__init__.py"]
                else:
                    with tarfile.open(damaged, "r:gz") as archive:
                        assert archive.getnames() == [
                            "openmed-0.0.0/openmed/__init__.py"
                        ]


def test_proto_control_omits_every_generated_stub():
    resources, profile, check = smoke_check.NEGATIVE_CONTROLS["missing_proto_stubs"]
    generated = Path(smoke_check.__file__).resolve().parents[2] / "openmed"
    assert resources == smoke_check.PROTO_STUBS
    assert (profile, check) == ("adapters", "grpc_stubs")
    assert {
        path.relative_to(generated).as_posix()
        for path in (generated / "service/proto/generated").glob("*_pb2*.py")
    } == set(resources)


def test_resource_control_cannot_succeed_when_resource_was_already_missing(
    subtests, tmp_path
):
    for control, (resources, _, _) in smoke_check.NEGATIVE_CONTROLS.items():
        for kind in ["wheel", "sdist"]:
            with subtests.test(control=control, kind=kind):
                case = tmp_path / f"{control}-{kind}"
                output = case / "control"
                output.mkdir(parents=True)
                # Drop one required resource: the control must refuse, not pass.
                artifact = _synthetic_artifact(case, kind, resources[1:])
                with pytest.raises(ValueError, match="invalid resource inventory"):
                    smoke_check._omit_packaged_resources(artifact, output, resources)


def test_brief_fixture_is_synthetic_reviewed_and_not_diagnostic():
    from openmed.clinical.brief import build_clinical_brief

    value, context = smoke_check._brief_fixture()
    result = build_clinical_brief(value, model="extractive", context=context)
    assert result.refusal_reason is None
    assert result.summary == value.deidentified_text
    assert len(result.citations) == 3
    assert result.envelope["requires_human_review"]
    assert not result.envelope["is_diagnostic"]
    assert "dehydration" not in json.dumps(result.to_dict())


def test_artifact_checks_reject_missing_or_ambiguous_inputs(tmp_path):
    with pytest.raises(ValueError, match="expected one wheel and one sdist"):
        smoke_check.run_artifact_checks(tmp_path, python_executable="python")


def _fake_artifacts(tmp_path: Path, monkeypatch: Any) -> list[tuple[str, ...]]:
    """Create two empty artifacts and record each damaged-copy request."""
    for name in ("openmed-0.0.0-py3-none-any.whl", "openmed-0.0.0.tar.gz"):
        (tmp_path / name).touch()
    omitted: list[tuple[str, ...]] = []

    def omit(artifact, destination, resources):
        omitted.append(tuple(resources))
        return artifact

    monkeypatch.setattr(smoke_check.shutil, "which", lambda _: "/synthetic/uv")
    monkeypatch.setattr(smoke_check, "_omit_packaged_resources", omit)
    return omitted


def test_artifact_install_failures_are_redacted_and_all_eight_cases_run(
    tmp_path, monkeypatch, capsys
):
    omitted = _fake_artifacts(tmp_path, monkeypatch)
    secret = "synthetic-sensitive-installer-error"
    monkeypatch.setattr(
        smoke_check.subprocess,
        "run",
        lambda *args, **kwargs: _completed([], returncode=1, stdout=secret),
    )
    payload = smoke_check.run_artifact_checks(tmp_path, python_executable="python")
    rendered = json.dumps(payload) + capsys.readouterr().out
    assert payload["status"] == "failed" and len(payload["checks"]) == 8
    assert all(row["reason"] == "installation_failed" for row in payload["checks"])
    assert [row["profile"] for row in payload["checks"][:4]] == [
        "core",
        "adapters",
        "missing_resource",
        "missing_proto_stubs",
    ]
    assert (
        omitted
        == [
            (smoke_check.BRIEF_RESOURCE,),
            smoke_check.PROTO_STUBS,
        ]
        * 2
    )
    # Rows hold only controlled fields; the digest is of the original artifact.
    empty = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    for row in payload["checks"]:
        assert set(row) == {"artifact", "digest", "profile", "reason", "status"}
        assert row["digest"] == empty
    assert secret not in rendered and str(tmp_path) not in rendered


def test_negative_controls_must_fail_at_their_own_check(
    subtests, tmp_path, monkeypatch
):
    _fake_artifacts(tmp_path, monkeypatch)
    requirements: list[str] = []

    def run(command, **kwargs):
        if command[1:3] == ["pip", "install"]:
            requirements.append(command[5])
        return _completed([])

    monkeypatch.setattr(smoke_check.subprocess, "run", run)
    expected = {"core": "packaged_resources", "adapters": "grpc_stubs"}
    for swapped in (False, True):
        with subtests.test(swapped=swapped):
            requirements.clear()

            def report(*, python_executable, brief_profile):
                if swapped:
                    brief_profile = {"core": "adapters", "adapters": "core"}[
                        brief_profile
                    ]
                failed = expected[brief_profile]
                return smoke_check.SmokeReport(
                    "failed",
                    (
                        smoke_check._passed("entry_point"),
                        smoke_check._failed(failed, "contract_failed"),
                    ),
                )

            monkeypatch.setattr(smoke_check, "run_smoke_check", report)
            payload = smoke_check.run_artifact_checks(
                tmp_path, python_executable="python"
            )
            controls = {
                (row["artifact"], row["profile"]): row["reason"]
                for row in payload["checks"]
                if row["profile"] in smoke_check.NEGATIVE_CONTROLS
            }
            assert len(controls) == 4
            if swapped:
                assert set(controls.values()) == {"negative_control_failed"}
            else:
                assert controls == {
                    (kind, control): f"{control}_rejected"
                    for kind in ("wheel", "sdist")
                    for control in smoke_check.NEGATIVE_CONTROLS
                }
            # Only adapter installs, including the proto control, add extras.
            assert [r.endswith(smoke_check.ADAPTER_EXTRAS) for r in requirements] == [
                False,
                True,
                False,
                True,
            ] * 2


@pytest.fixture
def service_probe_env(monkeypatch: Any):
    """Mirror the child's clean, default-profile, auth-free service environment."""
    for module in ("fastapi", "grpc", "httpx", "strawberry"):
        pytest.importorskip(module)
    for name in list(os.environ):
        if name == "OPENMED_PROFILE" or name.startswith("OPENMED_SERVICE_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(
        "openmed.service.runtime.ModelLoader", smoke_check._SyntheticPIILoader
    )
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_service_transport_probes_share_a_fake_runtime_offline(
    service_probe_env, caplog
):
    import logging

    import openmed
    from openmed.service.app import create_app

    caplog.set_level(logging.DEBUG)
    app = create_app()
    spans = smoke_check._probe_rest_pii(app, service_probe_env)
    assert spans == smoke_check._SERVICE_SPANS
    smoke_check._probe_graphql(app, service_probe_env)
    smoke_check._check_proto_stubs(Path(openmed.__file__).resolve().parent)
    runtime = app.state.runtime
    assert smoke_check._probe_grpc(runtime) == spans
    loader = runtime.get_model_loader()
    assert isinstance(loader, smoke_check._SyntheticPIILoader)
    assert loader.loaded and set(loader.loaded) <= set(runtime.served_model_ids())
    assert smoke_check._SERVICE_MARKER not in caplog.text


def test_service_probes_fail_without_the_synthetic_prediction(
    service_probe_env, monkeypatch
):
    from openmed.service.app import create_app

    monkeypatch.setattr(
        smoke_check._SyntheticPIIPipeline, "__call__", lambda self, inputs, **_: []
    )
    app = create_app()
    with pytest.raises(AssertionError):
        smoke_check._probe_rest_pii(app, service_probe_env)
    with pytest.raises(AssertionError):
        smoke_check._probe_graphql(app, service_probe_env)
    assert smoke_check._probe_grpc(app.state.runtime) == ()


def test_grpc_probe_binds_only_an_ephemeral_loopback_port(
    service_probe_env, monkeypatch
):
    from openmed.service import grpc_server
    from openmed.service.runtime import ServiceRuntime

    original = grpc_server.create_grpc_server
    addresses: list[str] = []

    def create(**kwargs):
        server = original(**kwargs)
        bind = server.add_insecure_port

        def record(address):
            addresses.append(address)
            return bind(address)

        server.add_insecure_port = record
        return server

    monkeypatch.setattr(grpc_server, "create_grpc_server", create)
    assert smoke_check._probe_grpc(ServiceRuntime.from_env()) == (
        smoke_check._SERVICE_SPANS
    )
    assert addresses == ["127.0.0.1:0"]


def test_proto_stub_check_rejects_missing_or_foreign_stubs(
    service_probe_env, monkeypatch, tmp_path
):
    import openmed

    root = Path(openmed.__file__).resolve().parent
    with pytest.raises(AssertionError):
        smoke_check._check_proto_stubs(tmp_path)
    monkeypatch.setattr(
        smoke_check,
        "PROTO_STUBS",
        (*smoke_check.PROTO_STUBS, "service/proto/generated/absent_pb2.py"),
    )
    with pytest.raises(AssertionError):
        smoke_check._check_proto_stubs(root)
