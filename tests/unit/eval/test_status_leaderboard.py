"""Tests for generated benchmark, leaderboard, and status artifacts."""

from __future__ import annotations

from pathlib import Path

from openmed.core.baseline import load_baseline_store
from openmed.core.hf_publish import publish_artifact
from openmed.core.model_registry import load_manifest_rows
from openmed.eval.report import (
    BenchmarkReport,
    read_reports,
    render_benchmark_card,
    render_leaderboard,
)
from scripts.status.generate_status import render_status_page

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "models.jsonl"
BASELINE = ROOT / "gates" / "baseline.json"
GOLDEN_REPORT = ROOT / "docs" / "benchmarks" / "golden.report.json"
BENCHMARK_CARD = ROOT / "docs" / "benchmarks" / "golden.md"
LEADERBOARD = ROOT / "docs" / "leaderboard" / "index.md"
STATUS = ROOT / "docs" / "status" / "index.md"


class RepositoryNotFoundError(Exception):
    pass


class FakeApi:
    def repo_info(self, **kwargs: object) -> object:
        raise RepositoryNotFoundError("not found")

    def create_repo(self, **kwargs: object) -> None:
        return None

    def upload_folder(self, **kwargs: object) -> None:
        return None


def _fixture_manifest() -> list[dict[str, object]]:
    return [
        {
            "repo_id": "OpenMed/pii-small-mlx",
            "family": "PII",
            "tier": "Small",
            "formats": ["mlx-fp", "pytorch"],
            "released": "2026-06-01",
            "reproducibility_hash": "sha256:" + "a" * 64,
        },
        {
            "repo_id": "OpenMed/ner-large",
            "family": "NER",
            "tier": "Large",
            "formats": ["pytorch"],
            "released": "2026-05-20",
            "reproducibility_hash": "sha256:" + "b" * 64,
        },
    ]


def _fixture_report() -> BenchmarkReport:
    return BenchmarkReport(
        suite="golden",
        model_name="OpenMed/pii-small-mlx",
        device="mlx-fp",
        fixture_count=2,
        generated_at="2026-06-14T00:00:00Z",
        metrics={
            "leakage": {"overall": 0.125, "leaked_chars": 1, "total_chars": 8},
            "exact_span_f1": {"f1": 0.9},
        },
    )


def _fixture_baseline() -> dict[str, object]:
    return {
        "schema_version": 1,
        "entries": {
            "pii::small::mlx-fp": {
                "key": "pii::small::mlx-fp",
                "family": "PII",
                "tier": "Small",
                "format": "mlx-fp",
                "metrics": {"leakage": {"overall": 0.125}},
                "reproducibility_hash": "sha256:" + "a" * 64,
                "released": "2026-06-01",
                "metadata": {
                    "last_regression": "2026-05-30",
                    "last_rollback": "2026-05-31",
                },
            }
        },
    }


def test_benchmark_card_and_leaderboard_render_fixture_inputs() -> None:
    manifest = _fixture_manifest()
    report = _fixture_report()

    card = render_benchmark_card(report, manifest_rows=manifest)
    leaderboard = render_leaderboard(
        manifest_rows=manifest,
        reports=[report],
        baseline_store=_fixture_baseline(),
    )

    assert "| Model | `OpenMed/pii-small-mlx` |" in card
    assert "| `leakage.overall` | 0.125 |" in card
    assert "| OpenMed | `PII` | `Small` | `mlx-fp` | 1 | 12.50% |" in leaderboard
    assert "| SHIELD | `declared competitor`" in leaderboard


def test_status_page_renders_green_and_red_fixture_entries() -> None:
    manifest = _fixture_manifest()
    report = _fixture_report()
    baseline = _fixture_baseline()

    green = render_status_page(
        manifest_rows=manifest,
        baseline_store=baseline,
        reports=[report],
        smoke_status="green",
    )
    red = render_status_page(
        manifest_rows=manifest,
        baseline_store=baseline,
        reports=[report],
        smoke_status="red",
        smoke_failure_reason="smoke fixture failed",
    )

    assert "| `PII` | `Small` | `mlx-fp` | `mlx-fp` | 1 | 12.50%" in green
    assert "`2026-05-30 / 2026-05-31`" in green
    assert "| `green` |" in green
    assert "| `red` |" in red
    assert "smoke fixture failed" in red


def test_committed_generated_artifacts_do_not_drift() -> None:
    manifest = load_manifest_rows(MANIFEST)
    baseline = load_baseline_store(BASELINE)
    reports = read_reports([GOLDEN_REPORT])

    assert BENCHMARK_CARD.read_text(encoding="utf-8") == render_benchmark_card(
        reports[0],
        manifest_rows=manifest,
    )
    assert LEADERBOARD.read_text(encoding="utf-8") == render_leaderboard(
        manifest_rows=manifest,
        reports=reports,
        baseline_store=baseline,
    )
    assert STATUS.read_text(encoding="utf-8") == render_status_page(
        manifest_rows=manifest,
        baseline_store=baseline,
        reports=reports,
        smoke_status="green",
    )


def test_publish_path_refreshes_status_surfaces(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}", encoding="utf-8")
    (artifact / "weights.safetensors").write_bytes(b"weights")

    report_path = tmp_path / "golden.report.json"
    BenchmarkReport(
        suite="golden",
        model_name="OpenMed/status-model-v1-mlx",
        device="mlx-fp",
        fixture_count=1,
        generated_at="2026-06-14T00:00:00Z",
        metrics={"leakage": {"overall": 0.5}},
    ).write_json(report_path)
    monkeypatch.setenv("HF_WRITE_TOKEN", "secret-token")

    publish_artifact(
        artifact_dir=artifact,
        source_model_id="OpenMed/status-model",
        format_name="mlx-fp",
        manifest_path=tmp_path / "models.jsonl",
        baseline_path=tmp_path / "baseline.json",
        benchmark_report_paths=[report_path],
        benchmarks_dir=tmp_path / "benchmarks",
        leaderboard_dir=tmp_path / "leaderboard",
        status_output_path=tmp_path / "status.md",
        api=FakeApi(),
        released="2026-06-14",
        git_sha="abc123",
    )

    assert (tmp_path / "benchmarks" / "golden.md").exists()
    assert (tmp_path / "leaderboard" / "index.md").exists()
    status = (tmp_path / "status.md").read_text(encoding="utf-8")
    assert "50.00%" in status
    assert "`green`" in status
