"""Tests for manifest rollback pointer flips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.core.baseline import BASELINE_SCHEMA_VERSION, baseline_key, write_baseline_store
from openmed.core.manifest import (
    ManifestRollbackError,
    load_manifest_rows,
    rollback_manifest_pointer,
    write_manifest_rows,
)


def _row(
    repo_id: str,
    *,
    released: str,
    digest_char: str,
    micro_f1: float,
) -> dict[str, object]:
    return {
        "repo_id": repo_id,
        "family": "PII",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Small",
        "param_count": 44_000_000,
        "architecture": "deberta-v2",
        "base_model": repo_id.replace("-mlx", ""),
        "formats": ["mlx-fp", "pytorch"],
        "canonical_labels": ["PERSON", "DATE"],
        "benchmark": {"dataset": "fixture", "micro_f1": micro_f1, "recall": 0.97},
        "arxiv": None,
        "license": "apache-2.0",
        "reproducibility_hash": f"sha256:{digest_char * 64}",
        "released": released,
    }


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        tmp_path / "models.jsonl",
        tmp_path / "baseline.json",
        tmp_path / "cards",
        tmp_path / "release-status.json",
        tmp_path / "rollback-log.jsonl",
    )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    manifest, baseline, card_dir, status_path, tracking_log = _paths(tmp_path)
    write_manifest_rows(
        [
            _row(
                "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v2-mlx",
                released="2026-06-14",
                digest_char="b",
                micro_f1=0.91,
            ),
            _row(
                "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx",
                released="2026-06-01",
                digest_char="a",
                micro_f1=0.98,
            ),
        ],
        manifest,
    )
    key = baseline_key("PII", "Small", "mlx-fp")
    write_baseline_store(
        {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "entries": {
                key: {
                    "key": key,
                    "family": "PII",
                    "tier": "Small",
                    "format": "mlx-fp",
                    "metrics": {"micro_f1": 0.98, "recall": 0.97},
                    "reproducibility_hash": "sha256:" + "a" * 64,
                    "repo_id": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx",
                    "source_model_id": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
                    "released": "2026-06-01",
                }
            },
        },
        baseline,
    )
    return manifest, baseline, card_dir, status_path, tracking_log


def test_rollback_flips_manifest_pointer_and_retains_prior_version(tmp_path: Path) -> None:
    manifest, baseline, card_dir, status_path, tracking_log = _write_fixture(tmp_path)

    result = rollback_manifest_pointer(
        family="PII",
        tier="Small",
        format_name="mlx-fp",
        manifest_path=manifest,
        baseline_path=baseline,
        card_dir=card_dir,
        status_path=status_path,
        tracking_log_path=tracking_log,
        reason="fixture regression",
    )

    rows = load_manifest_rows(manifest)
    active = rows[0]
    repo_ids = [row["repo_id"] for row in rows]
    card_text = result.card_path.read_text(encoding="utf-8")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in tracking_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result.changed is True
    assert result.previous_repo_id == "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v2-mlx"
    assert result.active_repo_id == "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx"
    assert active["repo_id"] == result.active_repo_id
    assert active["reproducibility_hash"] == "sha256:" + "a" * 64
    assert "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v2-mlx" in repo_ids
    assert len(repo_ids) == len(set(repo_ids))
    assert result.active_repo_id in card_text
    assert active["reproducibility_hash"] in card_text
    assert status["active"]["repo_id"] == result.active_repo_id
    assert records[-1]["from_repo_id"] == result.previous_repo_id
    assert records[-1]["to_repo_id"] == result.active_repo_id
    assert records[-1]["operation"] == "manifest-pointer-flip"


def test_release_rollback_cli_uses_fixture_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, baseline, card_dir, status_path, tracking_log = _write_fixture(tmp_path)

    code = main_module.main(
        [
            "release",
            "rollback",
            "PII",
            "--tier",
            "Small",
            "--format",
            "mlx-fp",
            "--manifest",
            str(manifest),
            "--baseline",
            str(baseline),
            "--card-dir",
            str(card_dir),
            "--status-path",
            str(status_path),
            "--tracking-log",
            str(tracking_log),
        ]
    )

    out = capsys.readouterr().out
    rows = load_manifest_rows(manifest)

    assert code == 0
    assert "Rolled back PII/Small/mlx-fp" in out
    assert rows[0]["repo_id"] == "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx"
    assert tracking_log.exists()


def test_rollback_requires_disambiguation_for_multiple_family_baselines(
    tmp_path: Path,
) -> None:
    manifest, baseline, card_dir, status_path, tracking_log = _write_fixture(tmp_path)
    key = baseline_key("PII", "Large", "pytorch")
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["entries"][key] = {
        "key": key,
        "family": "PII",
        "tier": "Large",
        "format": "pytorch",
        "metrics": {"micro_f1": 0.95},
        "reproducibility_hash": "sha256:" + "c" * 64,
        "repo_id": "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1",
    }
    baseline.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ManifestRollbackError, match="multiple last-green baselines"):
        rollback_manifest_pointer(
            family="PII",
            manifest_path=manifest,
            baseline_path=baseline,
            card_dir=card_dir,
            status_path=status_path,
            tracking_log_path=tracking_log,
        )
