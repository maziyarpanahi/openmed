"""Focused offline tests for FHIR Bulk Data service jobs."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from openmed.service.app import create_app
from openmed.service.bulk_data import FHIRBulkJobConfig, FHIRBulkJobManager


@dataclass
class _FakeResult:
    deidentified_text: str


def _fake_deidentify(text: str, **_: Any) -> _FakeResult:
    return _FakeResult(deidentified_text=text.replace("Jane Roe", "[NAME]"))


def _write_export(path: Path, count: int = 1) -> None:
    path.mkdir(parents=True, exist_ok=True)
    resources = [
        {
            "resourceType": "Patient",
            "id": f"patient-{index}",
            "name": [{"text": "Jane Roe"}],
        }
        for index in range(count)
    ]
    (path / "Patient.ndjson").write_text(
        "".join(
            json.dumps(resource, separators=(",", ":")) + "\n" for resource in resources
        ),
        encoding="utf-8",
    )


def _wait_for_status(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(100):
        payload = client.get(f"/fhir/bulk/exports/{job_id}").json()
        if payload["status"] != "running":
            return payload
        time.sleep(0.01)
    raise AssertionError("bulk job did not finish")


def test_bulk_routes_poll_manifest_report_and_preserve_privacy_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_BULK_INPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("OPENMED_SERVICE_BULK_OUTPUT_ROOT", str(tmp_path))
    app = create_app()
    app.state.fhir_bulk_deidentifier = _fake_deidentify
    source = tmp_path / "synthetic-source"
    output = tmp_path / "deidentified-output"
    _write_export(source)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        started = client.post(
            "/fhir/bulk/exports",
            json={"input_dir": source.name, "output_dir": output.name},
        )
        assert started.status_code == 202
        job_id = started.json()["job_id"]
        assert started.headers["content-location"].endswith(job_id)

        status = _wait_for_status(client, job_id)
        assert status["status"] == "succeeded"
        manifest = client.get(f"/fhir/bulk/exports/{job_id}/manifest")
        report = client.get(f"/fhir/bulk/exports/{job_id}/report")

    assert manifest.status_code == 200
    assert report.status_code == 200
    rendered = json.dumps(
        {"status": status, "manifest": manifest.json(), "report": report.json()},
        sort_keys=True,
    )
    assert "Jane Roe" not in rendered
    assert "patient-0" not in rendered
    assert report.json()["bulk_data_version"] == "3.0.0"
    assert report.json()["peak_buffered_resources"] == 1
    assert (output / "Patient.ndjson").read_text(encoding="utf-8").count("[NAME]") == 1


def test_bulk_route_cancellation_is_reported_without_resource_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_BULK_INPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("OPENMED_SERVICE_BULK_OUTPUT_ROOT", str(tmp_path))
    app = create_app()

    def slow_deidentify(text: str, **_: Any) -> _FakeResult:
        time.sleep(0.02)
        return _fake_deidentify(text)

    app.state.fhir_bulk_deidentifier = slow_deidentify
    source = tmp_path / "synthetic-source"
    output = tmp_path / "deidentified-output"
    _write_export(source, count=20)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        started = client.post(
            "/fhir/bulk/exports",
            json={"input_dir": source.name, "output_dir": output.name},
        )
        job_id = started.json()["job_id"]
        cancelled = client.delete(f"/fhir/bulk/exports/{job_id}")
        status = client.get(f"/fhir/bulk/exports/{job_id}")

    assert cancelled.status_code == 202
    assert cancelled.json()["status"] == "cancelled"
    assert status.json()["status"] == "cancelled"
    assert "Jane Roe" not in json.dumps(status.json())


def test_socket_blocked_smart_job_does_not_expose_credentials_or_urls() -> None:
    token_secret = "access-token-synthetic-secret"
    assertion_secret = "client-assertion-synthetic-secret"
    phi_url = "https://fhir.example.test/export/patient-jane-roe"

    def blocked(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"blocked {token_secret} {assertion_secret} {phi_url}",
            request=request,
        )

    manager = FHIRBulkJobManager(
        transport=httpx.MockTransport(blocked),
        client_assertion_builder=lambda _: assertion_secret,
    )
    config = FHIRBulkJobConfig(
        output_dir="/tmp/openmed-synthetic-bulk-output",
        fhir_base_url="https://fhir.example.test",
        token_url="https://auth.example.test/token",
        client_id="synthetic-client",
        private_key_pem="synthetic-private-key",
    )

    async def run_job() -> dict[str, Any]:
        manager.start(config, job_id="socket-blocked")
        for _ in range(100):
            status = manager.get("socket-blocked")
            if status.status != "running":
                return status.to_dict()
            await asyncio.sleep(0)
        raise AssertionError("SMART job did not finish")

    payload = asyncio.run(run_job())
    rendered = json.dumps(payload, sort_keys=True)
    assert token_secret not in rendered
    assert assertion_secret not in rendered
    assert phi_url not in rendered
    assert payload["status"] == "failed"


@pytest.mark.parametrize(
    "submitted",
    [
        "/etc/SYNTHETIC_PATH",
        "../SYNTHETIC_PATH",
        "~/.ssh",
        "C:\\SYNTHETIC_PATH",
        "one/../SYNTHETIC_PATH",
        "one//SYNTHETIC_PATH",
        "./SYNTHETIC_PATH",
    ],
)
@pytest.mark.parametrize("field", ["input_dir", "output_dir", "checkpoint_path"])
def test_bulk_paths_reject_lexical_escapes_without_side_effects(
    tmp_path, monkeypatch, submitted, field, caplog
):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_BULK_INPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("OPENMED_SERVICE_BULK_OUTPUT_ROOT", str(tmp_path))
    source = tmp_path / "source"
    _write_export(source)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    payload = {"input_dir": "source", "output_dir": "fresh"}
    payload[field] = submitted
    app = create_app()
    app.state.fhir_bulk_deidentifier = _fake_deidentify
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/fhir/bulk/exports", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "path_invalid"
    assert (
        sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
        == before
    )
    assert not app.state.fhir_bulk_jobs._jobs
    assert "SYNTHETIC_PATH" not in response.text + caplog.text


def test_bulk_roots_disabled_and_symlink_or_existing_outputs_are_rejected(
    tmp_path, monkeypatch
):
    from openmed.service.bulk_data import BulkPathError, BulkStoragePolicy

    with pytest.raises(BulkPathError, match="bulk_paths_disabled"):
        BulkStoragePolicy().prepare(
            input_dir="/arbitrary",
            output_dir="/arbitrary",
            checkpoint_path=None,
            job_id="synthetic",
        )
    source = tmp_path / "source"
    _write_export(source)
    policy = BulkStoragePolicy((tmp_path,), tmp_path)
    with pytest.raises(BulkPathError, match="path_exists"):
        policy.prepare(
            input_dir="source",
            output_dir="source",
            checkpoint_path=None,
            job_id="synthetic",
        )
    try:
        (source / "escape.ndjson").symlink_to(tmp_path / "outside.ndjson")
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")
    with pytest.raises(BulkPathError, match="path_outside_root"):
        policy.prepare(
            input_dir="source",
            output_dir="fresh",
            checkpoint_path=None,
            job_id="synthetic",
        )
    assert not (tmp_path / "fresh").exists()


def test_bulk_safe_relative_paths_reserve_exclusive_output_and_checkpoint(tmp_path):
    from openmed.service.bulk_data import BulkPathError, BulkStoragePolicy

    _write_export(tmp_path / "source")
    policy = BulkStoragePolicy((tmp_path,), tmp_path)
    source, output, checkpoint = policy.prepare(
        input_dir="source", output_dir="fresh", checkpoint_path=None, job_id="synthetic"
    )
    assert source == tmp_path / "source"
    assert checkpoint == output / ".checkpoint-synthetic.json"
    with pytest.raises(BulkPathError, match="path_exists"):
        policy.prepare(
            input_dir="source",
            output_dir="fresh",
            checkpoint_path=None,
            job_id="synthetic",
        )
