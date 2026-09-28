"""Synthetic builders for the v3 Journey release-gate tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def make_release_repository(
    tmp_path: Path,
    *,
    source_files: Mapping[str, Path] | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    """Create a tagged local repository and a passing aggregate manifest."""

    root = tmp_path / "release-repo"
    root.mkdir()
    inputs = root / "inputs"
    inputs.mkdir()
    (inputs / "synthetic.json").write_text(
        json.dumps(
            {
                "provenance": "synthetic",
                "scenario": "journey-release-contract",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for relative, source in sorted((source_files or {}).items()):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    _git(root, "init", "-q")
    _git(root, "config", "user.name", "OpenMed Test")
    _git(root, "config", "user.email", "test@openmed.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "synthetic release evidence")
    commit = _git(root, "rev-parse", "HEAD")
    _git(root, "tag", "v3.0.0")

    manifest = make_passing_manifest(root, commit)
    for index, (relative, _source) in enumerate(
        sorted((source_files or {}).items()), start=1
    ):
        manifest["frozen_inputs"].append(
            {
                "id": f"synthetic_source_{index}",
                "origin": "synthetic",
                "path": relative,
                "sha256": file_digest(root / relative),
            }
        )
    return root, commit, manifest


def make_passing_manifest(root: Path, commit: str) -> dict[str, Any]:
    """Return a complete passing manifest for a tagged synthetic checkout."""

    evaluated_at = "2026-01-02T00:00:00Z"
    generated_at = "2026-01-01T23:00:00Z"
    return {
        "schema_version": "1.0.0",
        "compatibility_policy": "same_major",
        "release": {
            "version": "3.0.0",
            "git_commit": commit,
            "git_tag": "v3.0.0",
            "evaluated_at": evaluated_at,
            "source_date_epoch": int(
                datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp()
            ),
        },
        "environment": {
            "os": "linux",
            "architecture": "x86_64",
            "python_implementation": "cpython",
            "python_version": "3.11.9",
            "hardware_id": "synthetic_cpu_runner",
        },
        "policy": {"max_artifact_age_seconds": 86_400},
        "frozen_inputs": [
            {
                "id": "golden_scenario",
                "origin": "synthetic",
                "path": "inputs/synthetic.json",
                "sha256": file_digest(root / "inputs" / "synthetic.json"),
            }
        ],
        "gate_reports": [
            report(
                "schema",
                generated_at,
                public_schema_count=42,
                persisted_schema_count=15,
                invalid_schema_count=0,
                invalid_span_count=0,
                invalid_evidence_count=0,
            ),
            report(
                "provenance",
                generated_at,
                artifact_count=11,
                broken_link_count=0,
                unhashed_artifact_count=0,
            ),
            report(
                "clinical_nlp",
                generated_at,
                evaluated_case_count=250,
                unreviewed_high_risk_count=0,
                abstention_required_count=18,
                abstention_failure_count=0,
            ),
            report(
                "privacy",
                generated_at,
                evaluated_case_count=400,
                critical_leakage_count=0,
                raw_value_finding_count=0,
            ),
            report(
                "interoperability",
                generated_at,
                evaluated_exchange_count=120,
                conformance_failure_count=0,
                roundtrip_loss_without_disclosure_count=0,
            ),
            report(
                "application",
                generated_at,
                test_count=1000,
                failed_test_count=0,
                untyped_terminal_state_count=0,
                prohibited_public_claim_count=0,
            ),
            report(
                "performance",
                generated_at,
                dataset_digest="sha256:" + ("1" * 64),
                quantization="int8",
                concurrency=4,
                sample_count=1000,
                warmup_count=20,
                latency_p50_ms=12.5,
                latency_p95_ms=24.0,
                latency_p99_ms=31.0,
                throughput_per_second=175.0,
                peak_memory_mib=512.0,
                slo_breach_count=0,
            ),
            report(
                "recovery",
                generated_at,
                scenario_count=20,
                non_idempotent_replay_count=0,
                migration_failure_count=0,
                recovery_failure_count=0,
            ),
            report(
                "security",
                generated_at,
                finding_count=7,
                unresolved_critical_finding_count=0,
                unresolved_high_finding_count=0,
                threat_case_count=36,
                unmitigated_threat_count=0,
            ),
            report(
                "licensing",
                generated_at,
                model_count=1,
                data_asset_count=1,
                unchecked_asset_count=0,
                prohibited_distribution_count=0,
            ),
        ],
        "licenses": [
            {
                "asset_id": "synthetic_model",
                "asset_type": "model",
                "license_id": "Apache-2.0",
                "use": "runtime",
                "redistributable": True,
                "distribution": "bundled",
            },
            {
                "asset_id": "synthetic_dataset",
                "asset_type": "dataset",
                "license_id": "CC0-1.0",
                "use": "eval",
                "redistributable": True,
                "distribution": "bundled",
            },
        ],
        "claims": [
            "human_review_for_high_risk",
            "local_first",
            "offline_after_assets_downloaded",
            "synthetic_release_evidence",
            "typed_failure_states",
        ],
        "limitations": [
            "no_autonomous_clinical_action",
            "not_a_medical_device",
            "not_clinically_validated",
            "requires_local_validation",
            "restricted_assets_user_supplied",
        ],
        "exceptions": [],
    }


def report(gate: str, generated_at: str, **metrics: Any) -> dict[str, Any]:
    """Build one successful aggregate gate report."""

    return {
        "schema_version": "1.0.0",
        "compatibility_policy": "same_major",
        "gate": gate,
        "state": "success",
        "generated_at": generated_at,
        "metrics": metrics,
    }


def gate_report(manifest: Mapping[str, Any], gate: str) -> dict[str, Any]:
    """Return the mutable report for *gate* from a synthetic manifest."""

    return next(item for item in manifest["gate_reports"] if item["gate"] == gate)


def file_digest(path: Path) -> str:
    """Return a prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.strip()


__all__ = [
    "file_digest",
    "gate_report",
    "make_passing_manifest",
    "make_release_repository",
    "report",
]
