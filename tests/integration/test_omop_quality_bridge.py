"""End-to-end synthetic journey through OMOP quality reconciliation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from openmed.clinical.journey_contracts import ClinicalFact
from openmed.eval.suites.omop_quality import load_frozen_omop_quality_fixture
from openmed.interop.omop import (
    OmopConceptMapping,
    OmopFactProjectionInput,
    OmopProjectionAggregate,
    OmopVocabularySnapshot,
    build_omop_quality_tool_output,
    project_clinical_facts_to_omop,
    projection_quality_checks,
    run_omop_quality_remote,
    verify_omop_quality_report,
)

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "fixtures" / "interop" / "omop"
)
PROJECTION_FIXTURE = FIXTURE_DIRECTORY / "fact_projection.json"
QUALITY_FIXTURE = FIXTURE_DIRECTORY / "quality_reconciliation.json"


def _bundled_fixture_digest(path: Path) -> str:
    """Hash the committed text fixture independent of checkout line endings."""

    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def test_projection_to_signed_quality_report_journey() -> None:
    payload = json.loads(PROJECTION_FIXTURE.read_text(encoding="utf-8"))
    inputs = []
    for raw in payload["facts"]:
        record = dict(raw)
        mapping = OmopConceptMapping.from_dict(record.pop("mapping"))
        inputs.append(
            OmopFactProjectionInput(
                fact=ClinicalFact.from_dict(record),
                source_key=payload["source_key"],
                source_revision=payload["source_revision"],
                mapping=mapping,
                dataset_split=payload["dataset_split"],
            )
        )
    projection_result = project_clinical_facts_to_omop(
        inputs,
        vocabulary_snapshot=OmopVocabularySnapshot.from_dict(
            payload["vocabulary_snapshot"]
        ),
        etl_version=payload["etl_version"],
        occurred_at=payload["occurred_at"],
    )
    assert projection_result.ok and projection_result.value is not None
    projection = projection_result.value
    cohort_digest = _bundled_fixture_digest(PROJECTION_FIXTURE)
    aggregate = OmopProjectionAggregate.from_projection(
        projection,
        cohort_digest=cohort_digest,
    )
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)
    assert aggregate == fixture.openmed
    reconciliation = fixture.reconcile()
    checks = projection_quality_checks(projection)

    def remote_runner(_request: object) -> dict[str, object]:
        return build_omop_quality_tool_output(
            fixture.quality_input,
            checks,
            tool_name="synthetic.quality_adapter",
            tool_version="1.0.0",
            execution_mode="remote",
        )

    signing_key = b"synthetic-integration-signing-key"
    quality_result = run_omop_quality_remote(
        fixture.quality_input,
        runner=remote_runner,
        reconciliation=reconciliation,
        signing_key=signing_key,
    )

    assert quality_result.ok and quality_result.value is not None
    report = quality_result.value
    assert report.quality_verdict == "pass"
    assert report.input_digest == fixture.quality_input.digest
    assert report.reconciliation.expectations_met
    assert verify_omop_quality_report(report, signing_key)
    serialized = report.to_json()
    for forbidden in ("patient", "raw_text", "credential", "secret"):
        assert forbidden not in serialized.lower()
