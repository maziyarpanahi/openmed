"""Tests for the gold-corpus quality report and evidence-bundle attachment."""

from __future__ import annotations

import json

from openmed.eval.evidence_bundle import (
    GOLD_QUALITY_REPORT_ARTIFACT,
    bundle_gate_evidence,
)
from openmed.eval.golden.loader import load_consensus_corpus
from openmed.eval.report import GoldCorpusQualityReport, gold_corpus_quality_report


def _report() -> GoldCorpusQualityReport:
    return gold_corpus_quality_report(load_consensus_corpus())


# --------------------------------------------------------------------------
# Report content
# --------------------------------------------------------------------------


def test_report_covers_documents_and_agreement():
    report = _report()

    assert isinstance(report, GoldCorpusQualityReport)
    assert report.n_documents == 2
    assert 0.0 <= report.overall_agreement <= 1.0


def test_agreement_by_label():
    report = _report()

    # PERSON is agreed by every annotator across both documents.
    assert report.per_label["PERSON"] == 1.0
    # The medication/date/org labels are annotator disagreements.
    assert any(score < 1.0 for score in report.per_label.values())
    assert list(report.per_label) == sorted(
        report.per_label,
        key=lambda label: (report.per_label[label], label),
    )


def test_agreement_by_relation_type():
    report = _report()

    # One annotator linked the medication/date pair and one omitted it.
    assert report.relation_agreement["drug_to_date"] == 0.0
    assert report.to_dict()["relation_agreement"] == {"drug_to_date": 0.0}
    assert "Agreement by relation type" in report.to_markdown()


def test_relation_type_breakdown_from_consensus():
    report = _report()

    assert report.relation_types.get("drug_to_date") == 1


def test_adjudication_coverage_is_a_fraction():
    report = _report()
    assert 0.0 <= report.adjudication_coverage <= 1.0


def test_low_agreement_examples_have_ids_offsets_hashes_not_text():
    report = _report()

    assert report.low_agreement_examples
    example = report.low_agreement_examples[0]
    assert set(example) >= {"document_id", "offset", "labels", "hash"}
    assert "text" not in example
    # No raw clinical text anywhere in the serialized report.
    blob = json.dumps(report.to_dict())
    assert "Jane Roe" not in blob
    assert "Ridgeview" not in blob


# --------------------------------------------------------------------------
# Deterministic serialization
# --------------------------------------------------------------------------


def test_to_dict_is_deterministic():
    assert _report().to_dict() == _report().to_dict()


def test_to_markdown_is_deterministic_and_readable():
    markdown = _report().to_markdown()
    assert markdown == _report().to_markdown()
    assert "Gold Corpus Quality" in markdown
    assert "PERSON" in markdown


# --------------------------------------------------------------------------
# Evidence-bundle attachment
# --------------------------------------------------------------------------


def test_report_attaches_to_evidence_bundle_manifest(tmp_path):
    report_path = tmp_path / "gold_quality_report.json"
    report_path.write_text(json.dumps(_report().to_dict()), encoding="utf-8")

    destination = tmp_path / "bundle"
    result = bundle_gate_evidence(
        {},
        destination,
        extra_artifacts={GOLD_QUALITY_REPORT_ARTIFACT: report_path},
    )

    artifact_ids = {entry["artifact_id"] for entry in result.manifest["artifacts"]}
    assert GOLD_QUALITY_REPORT_ARTIFACT in artifact_ids
