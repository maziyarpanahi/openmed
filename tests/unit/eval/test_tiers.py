"""Tests for canonical device-tier budgets and compliance documentation."""

from __future__ import annotations

from pathlib import Path

from openmed.cli import COMPLIANCE_CAVEAT, main_module
from openmed.eval import TIERS
from openmed.eval.tiers import NANO_SUB_TIER

ROOT = Path(__file__).resolve().parents[3]
REQUIRED_FIELDS = {"ram_mb_max", "p50_ms_max", "p95_ms_max", "default_format"}


def test_all_device_tiers_expose_gate_budget_fields() -> None:
    assert tuple(TIERS) == ("Tiny", "Base", "Large", "Accurate-XLarge")

    for tier in TIERS.values():
        assert REQUIRED_FIELDS <= set(tier)
        assert isinstance(tier["ram_mb_max"], int)
        assert isinstance(tier["p50_ms_max"], int)
        assert isinstance(tier["p95_ms_max"], int)
        assert isinstance(tier["default_format"], str)


def test_tier_budget_values_match_section_6_2() -> None:
    assert TIERS["Tiny"]["ram_mb_max"] == 350
    assert TIERS["Tiny"]["p50_ms_max"] == 60
    assert TIERS["Tiny"]["p95_ms_max"] == 150
    assert TIERS["Base"]["ram_mb_max"] == 900
    assert TIERS["Base"]["p50_ms_max"] == 150
    assert TIERS["Base"]["p95_ms_max"] == 400
    assert TIERS["Large"]["ram_mb_max"] == 4096
    assert TIERS["Large"]["p50_ms_max"] == 250
    assert TIERS["Large"]["p95_ms_max"] == 800
    assert TIERS["Accurate-XLarge"]["ram_mb_max"] == 8192
    assert TIERS["Accurate-XLarge"]["p50_ms_max"] == 400
    assert TIERS["Accurate-XLarge"]["p95_ms_max"] == 1200


def test_nano_is_tiny_sub_tier_with_tighter_slo_budget() -> None:
    nano = TIERS["Tiny"]["sub_tiers"]["Nano"]

    assert nano == NANO_SUB_TIER
    assert nano["parent_tier"] == "Tiny"
    assert nano["param_count_min"] == 10_000_000
    assert nano["param_count_max"] == 30_000_000
    assert nano["ram_mb_max"] == 150
    assert nano["p50_ms_max"] == 25
    assert nano["p95_ms_max"] == 60
    assert nano["default_format"] == "INT8"


def test_tier_docs_capture_four_rows_and_budgets() -> None:
    text = (ROOT / "docs" / "tiers.md").read_text(encoding="utf-8")

    for expected in (
        "Tiny",
        "Base",
        "Large",
        "Accurate / XLarge",
        "≤ 350 MB",
        "≤ 900 MB",
        "≤ 4 GB",
        "≤ 8 GB",
        "≤ 60 ms / ≤ 150 ms",
        "≤ 150 ms / ≤ 400 ms",
        "≤ 250 ms / ≤ 800 ms",
        "≤ 400 ms / ≤ 1200 ms",
    ):
        assert expected in text


def test_compliance_caveat_appears_in_docs_and_cli_help() -> None:
    compliance = (ROOT / "docs" / "compliance.md").read_text(encoding="utf-8")
    help_text = main_module.build_parser().format_help()

    assert COMPLIANCE_CAVEAT in compliance
    assert COMPLIANCE_CAVEAT in help_text


def test_compliance_doc_covers_frameworks_and_evidence_links() -> None:
    text = (ROOT / "docs" / "compliance.md").read_text(encoding="utf-8")

    for expected in (
        "HIPAA Safe Harbor",
        'policy="hipaa_safe_harbor"',
        "HIPAA Expert Determination",
        'policy="hipaa_expert_review_assist"',
        "GDPR pseudonymization",
        'policy="gdpr_pseudonymization"',
        "EU AI Act high-risk",
        "OpenMed does not self-certify compliance",
        "Audit reports",
        "Leakage metrics",
        "Residual re-identification risk",
    ):
        assert expected in text


def test_compliance_templates_cover_required_legal_review_fields() -> None:
    compliance_dir = ROOT / "docs" / "compliance"
    dpia = (compliance_dir / "dpia-template.md").read_text(encoding="utf-8")
    dpa = (compliance_dir / "dpa-template.md").read_text(encoding="utf-8")
    model_card = (compliance_dir / "model-card-eu-ai-act-fields.md").read_text(
        encoding="utf-8"
    )

    for artifact in (dpia, dpa, model_card):
        assert "Template — requires legal review. Not legal advice." in artifact

    for expected in (
        "`hipaa_safe_harbor`",
        "`gdpr_pseudonymization`",
        "AuditReport.residual_risk.risk_report_record_score",
        "BenchmarkReport.metrics.exact_span_f1.f1",
        "BenchmarkReport.metrics.leakage.overall",
        "Risk-assessment method and baseline",
        "planned / partially implemented / implemented",
        "Processors and sub-processors",
    ):
        assert expected in dpia

    for expected in (
        "on-device",
        "no telemetry by default",
        "Sub-processors: none",
        "Reversible Pseudonymization and Key Custody",
        "`gdpr_pseudonymization`",
        "immediately inform the Controller",
    ):
        assert expected in dpa

    for expected in (
        "## EU AI Act / GDPR Compliance Fields",
        "### Intended purpose",
        "### Known limitations and reasonably foreseeable misuse",
        "### Accuracy and leakage metrics",
        "### Human oversight",
        "### Robustness, cybersecurity, and monitoring evidence",
        "`models.jsonl`",
        "`BenchmarkReport`",
        "benchmark.metrics.exact_span_f1.f1",
        "benchmark.metrics.relaxed_span_f1.f1",
        "benchmark.metrics.character_recall.rate",
        "benchmark.metrics.leakage.overall",
    ):
        assert expected in model_card


def test_compliance_posture_links_deployment_templates() -> None:
    text = (ROOT / "docs" / "compliance.md").read_text(encoding="utf-8")

    for link in (
        "compliance/dpia-template.md",
        "compliance/dpa-template.md",
        "compliance/model-card-eu-ai-act-fields.md",
    ):
        assert link in text
