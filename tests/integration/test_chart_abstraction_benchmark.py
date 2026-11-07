"""Offline gold-set scoring through the real chart-abstraction evidence contract."""

from dataclasses import replace
from hashlib import sha256

import pytest

from openmed.agent.workflows.abstraction_evidence import (
    AbstractionEvidenceChain,
    ChartAbstractionEvidence,
    ReviewerState,
    SourceLocation,
    TransformationKind,
)
from openmed.eval.suites.chart_abstraction import (
    AbstractionMetric,
    ChartAbstractionPrediction,
    run_chart_abstraction_benchmark,
    synthetic_chart_abstraction_gold,
)


@pytest.mark.integration
def test_synthetic_gold_scoring_uses_real_chains_and_preserves_source_spans():
    rows = synthetic_chart_abstraction_gold()
    outputs = []
    source = SourceLocation("sha256:" + "a" * 64, 10, 20)
    for row in rows:
        if row.value is None:
            outputs.append(ChartAbstractionPrediction(row.case_id, row.field_id, None))
            continue
        # This test producer uses a synthetic scalar encoding; the scorer does
        # not impose an encoding on real normalized facts from other producers.
        fact_digest = "sha256:" + sha256(str(row.value).encode()).hexdigest()
        chain = AbstractionEvidenceChain(
            row.field_id,
            (source,),
            fact_digest,
            TransformationKind.RULE,
            "sha256:" + "b" * 64,
            0.0,
            ReviewerState.APPROVED,
        )
        evidence = ChartAbstractionEvidence((chain,))
        assert evidence.finalize((row.field_id,)).chain_count == 1
        outputs.append(
            ChartAbstractionPrediction(
                row.case_id, row.field_id, row.value, fact_digest, evidence.chains[0]
            )
        )
    report = run_chart_abstraction_benchmark(rows, outputs)
    assert report.overall.normalized_agreement == AbstractionMetric(8, 8)
    assert report.overall.evidence_support == AbstractionMetric(8, 8)
    assert source.start_offset == 10 and source.end_offset == 20
    damaged = [replace(output, evidence_chain=None) for output in outputs]
    unsupported = run_chart_abstraction_benchmark(rows, damaged)
    assert unsupported.overall.exact_agreement == report.overall.exact_agreement
    assert unsupported.overall.evidence_support == AbstractionMetric(0, 8)
    assert (
        report.to_json()
        == run_chart_abstraction_benchmark(
            tuple(reversed(rows)), tuple(reversed(outputs))
        ).to_json()
    )
