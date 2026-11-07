"""Synthetic, frozen service state for diagnostic-bundle tests.

Every value is synthetic.  Sentinel strings are planted in configuration
values, file and directory names, store payloads, queue records, environment
variables and exception messages so tests can prove none of them reaches a
diagnostic bundle, its manifest, the CLI output or logs.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from openmed.core.audit import AuditReport, AuditSpan, DetectorInfo, hash_text
from openmed.core.audit_chain import AuditChain
from openmed.structured.store.local import SQLiteJourneyStore

# Every sentinel contains "sentinel" (case-insensitively), so one scan finds all.
SENTINEL_NAME = "Jane Q. Sentinel"
SENTINEL_MRN = "MRN-SENTINEL-4417"
SENTINEL_KEY = "sk-live-SENTINEL-KEY-9000"
SENTINEL_TOKEN = "sentinel_phi_marker"
SENTINEL_DIR = "SENTINEL-Jane-Doe-root"
SENTINELS = (SENTINEL_NAME, SENTINEL_MRN, SENTINEL_KEY, SENTINEL_TOKEN, SENTINEL_DIR)


@dataclass(frozen=True)
class FrozenState:
    """Paths of one synthetic service state."""

    root: Path
    config_path: Path
    profiles_dir: Path
    jobs_store_path: Path
    journey_store_root: Path
    registry_state_path: Path
    audit_chain_path: Path


def _report(name: str, identifier: str) -> AuditReport:
    original = f"Patient {name} has record {identifier}."
    start = original.index(name)
    return AuditReport(
        policy="hipaa_safe_harbor",
        resolved_profile={"method": "mask", "language": "en"},
        detectors=[
            DetectorInfo(
                source="ml", model_id="local-unit-model", model_format="transformers"
            )
        ],
        spans=[
            AuditSpan(
                start=start,
                end=start + len(name),
                label="NAME",
                canonical_label="PERSON",
                sources=["ml"],
                confidence=0.98,
                threshold=0.7,
                action="mask",
                surrogate=f"Replacement for {name}",
                text_hash=hash_text(name),
            )
        ],
        safety_sweep={"spans_added": 0},
        thresholds={"PERSON": 0.7},
        residual_risk={"review_note": f"No remaining reference to {name}"},
        openmed_version="1.9.1",
        manifest_hash=hash_text("unit manifest"),
        document_length=len(original),
        input_hash=hash_text(original),
        deidentified_text_hash=hash_text("Patient [PERSON] has record [ID]."),
    )


def write_audit_chain(path: Path, *, entries: int = 2) -> None:
    """Write a valid synthetic audit chain."""

    chain = AuditChain()
    for index in range(entries):
        chain.append(_report(f"{SENTINEL_NAME} {index}", SENTINEL_MRN))
    chain.write(path)


def jobs_document() -> dict[str, Any]:
    """Return a job metadata document with sentinels in every free field."""

    def record(job_id: str, status: str, **extra: Any) -> dict[str, Any]:
        base = {
            "id": job_id,
            "status": status,
            "document_count": 1,
            "documents": [
                {
                    "id": SENTINEL_MRN,
                    "length": 12,
                    "text_hash": hash_text(SENTINEL_NAME),
                }
            ],
            "spans": [{"document_id": SENTINEL_MRN, "label": "PERSON", "start": 0}],
            "error": None,
            "webhook": None,
            "webhook_delivery": None,
        }
        base.update(extra)
        return base

    return {
        "jobs": {
            f"{SENTINEL_MRN}-1": record(f"{SENTINEL_MRN}-1", "done"),
            f"{SENTINEL_MRN}-2": record(
                f"{SENTINEL_MRN}-2",
                "failed",
                error={"type": "RuntimeError", "message": f"{SENTINEL_NAME} failed"},
                webhook_delivery={
                    "success": False,
                    "error": f"https://{SENTINEL_KEY}@hooks.example.test refused",
                },
            ),
            f"{SENTINEL_MRN}-3": record(f"{SENTINEL_MRN}-3", "running"),
            f"{SENTINEL_MRN}-4": record(f"{SENTINEL_MRN}-4", SENTINEL_TOKEN),
        }
    }


def registry_document(*, extra_slots: int = 0) -> dict[str, Any]:
    """Return a valid registry state whose repo ids carry sentinels."""

    target = f"OpenMed/{SENTINEL_DIR}-model"
    slots: dict[str, Any] = {
        "pii::small::mlx-fp": {
            "checkpoints": {target: "1.0.0"},
            "lineage": [],
            "pointers": {"canary": None, "last_green": target, "latest": target},
        }
    }
    for index in range(extra_slots):
        slots[f"pii::tier{index}::onnx"] = {
            "checkpoints": {f"{target}-{index}": "1.0.0"},
            "lineage": [],
            "pointers": {"canary": None, "last_green": None, "latest": None},
        }
    return {"schema_version": 2, "slots": slots}


def build_journey_store(root: Path) -> None:
    """Create a migrated Journey store with sentinel payloads and blobs."""

    root.mkdir(parents=True, exist_ok=True)
    database = root / "journey.sqlite3"
    SQLiteJourneyStore(database).close()
    connection = sqlite3.connect(database)
    try:
        payload = json.dumps({"name": SENTINEL_NAME, "mrn": SENTINEL_MRN})
        connection.execute(
            "INSERT INTO store_revisions(revision, committed_at) VALUES (1, ?)",
            ("2026-01-01T00:00:00Z",),
        )
        connection.execute(
            "INSERT INTO artifacts(artifact_id, content_hash, payload_hash, "
            "payload_json, created_revision) VALUES (?, ?, ?, ?, 1)",
            (SENTINEL_MRN, "sha256:" + "a" * 64, "sha256:" + "b" * 64, payload),
        )
        connection.execute(
            "INSERT INTO ingestion_quarantine_results(quarantine_id, job_id, "
            "classification, created_at, payload_hash, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                SENTINEL_MRN,
                SENTINEL_TOKEN,
                SENTINEL_NAME,
                "2026-01-01T00:00:00Z",
                "sha256:" + "c" * 64,
                payload,
            ),
        )
        connection.execute(f"CREATE TABLE {SENTINEL_TOKEN} (note TEXT)")
        connection.execute(f"INSERT INTO {SENTINEL_TOKEN} VALUES (?)", (SENTINEL_NAME,))
        connection.commit()
    finally:
        connection.close()
    shard = root / "artifacts" / "blobs" / "sha256" / "ab"
    shard.mkdir(parents=True, exist_ok=True)
    (shard / ("ab" + "0" * 62)).write_bytes(SENTINEL_NAME.encode("utf-8"))
    (shard / ".journey-blob-staged").write_bytes(SENTINEL_MRN.encode("utf-8"))
    (root / "artifacts" / f"{SENTINEL_DIR}.txt").write_text(SENTINEL_NAME)


def build_frozen_state(base: Path) -> FrozenState:
    """Build a complete synthetic service state below ``base``."""

    root = base / SENTINEL_DIR
    root.mkdir(parents=True)
    config_path = root / "config.toml"
    config_path.write_text(
        "\n".join(
            (
                f'hf_token = "hf_{SENTINEL_KEY}"',
                f'cache_dir = "/Users/{SENTINEL_DIR}/cache"',
                'device = "cpu"',
                f'{SENTINEL_TOKEN} = "{SENTINEL_NAME}"',
                f"{SENTINEL_MRN} = 1",
                "",
            )
        ),
        encoding="utf-8",
    )
    profiles_dir = root / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / f"{SENTINEL_TOKEN}.toml").write_text(
        f'log_level = "{SENTINEL_NAME}"\n', encoding="utf-8"
    )
    jobs_store_path = root / "service-jobs.json"
    jobs_store_path.write_text(json.dumps(jobs_document()), encoding="utf-8")
    journey_store_root = root / "journey"
    build_journey_store(journey_store_root)
    registry_state_path = root / "registry_state.json"
    registry_state_path.write_text(json.dumps(registry_document()), encoding="utf-8")
    audit_chain_path = root / "audit-chain.json"
    write_audit_chain(audit_chain_path)
    return FrozenState(
        root=root,
        config_path=config_path,
        profiles_dir=profiles_dir,
        jobs_store_path=jobs_store_path,
        journey_store_root=journey_store_root,
        registry_state_path=registry_state_path,
        audit_chain_path=audit_chain_path,
    )


def plant_sentinel_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Place sentinels in service, credential, proxy and unknown variables."""

    for name in list(os.environ):
        if name.startswith("OPENMED_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_API_KEYS", SENTINEL_KEY)
    monkeypatch.setenv("OPENMED_SERVICE_OTLP_HEADERS", f"authorization={SENTINEL_KEY}")
    monkeypatch.setenv("OPENMED_SERVICE_SERVED_MODELS", f"OpenMed/{SENTINEL_DIR}")
    monkeypatch.setenv("OPENMED_SERVICE_CORS_ORIGINS", f"https://{SENTINEL_DIR}.test")
    monkeypatch.setenv("OPENMED_SENTINEL_PHI_MARKER", SENTINEL_NAME)
    monkeypatch.setenv("OPENMED_PATIENT_SENTINEL_MRN", SENTINEL_MRN)
    monkeypatch.setenv("OPENMED_PROFILE", SENTINEL_TOKEN)
    monkeypatch.setenv("OPENMED_DEVICE", "cpu")
    monkeypatch.setenv("HF_TOKEN", f"hf_{SENTINEL_KEY}")
    monkeypatch.setenv("HF_ENDPOINT", f"https://{SENTINEL_DIR}.example.test")
    monkeypatch.setenv("HTTPS_PROXY", f"http://user:{SENTINEL_KEY}@proxy.example.test")


def snapshot_tree(root: Path) -> dict[str, tuple[int, bytes, int]]:
    """Return size, content and mtime for every file below ``root``."""

    return {
        str(path.relative_to(root)): (
            path.stat().st_size,
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def assert_no_sentinels(text: str, *extra: str) -> None:
    """Assert that no sentinel (or extra forbidden string) appears in *text*."""

    lowered = text.lower()
    assert "sentinel" not in lowered
    assert "jane" not in lowered
    for value in extra:
        assert value not in text
