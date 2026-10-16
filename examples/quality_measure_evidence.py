"""Run synthetic signed quality-measure evidence checks offline.

Run from the repository root: python -m examples.quality_measure_evidence
The fixed demonstration key is public synthetic data, never a deployment key.
"""

import json

from openmed.agent.workflows import (
    CalculationTraceStep,
    InputProjectionEvidence,
    MeasureDefinitionEvidence,
    MeasureImplementationEvidence,
    MeasureResultEvidence,
    MeasureTimeBoundaries,
    QualityMeasureEvidenceError,
    QualityMeasureEvidencePacket,
    ValueSetEvidence,
    build_quality_measure_evidence_packet,
)

SYNTHETIC_KEY = b"synthetic-offline-example-key"


def run_example() -> dict:
    """Sign and round-trip synthetic evidence, then reject result tampering."""
    packet = build_quality_measure_evidence_packet(
        MeasureDefinitionEvidence("measure.synthetic", "1.0", "sha256:" + "a" * 64),
        (ValueSetEvidence("valueset.synthetic", "1.0", "sha256:" + "b" * 64),),
        InputProjectionEvidence("sha256:" + "c" * 64, "sha256:" + "d" * 64, 3),
        MeasureTimeBoundaries("2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z"),
        (),
        MeasureImplementationEvidence(
            "implementation.synthetic", "1.0", "sha256:" + "e" * 64
        ),
        (
            CalculationTraceStep(
                "numerator",
                "aggregate",
                "sha256:" + "f" * 64,
                "sha256:" + "d" * 64,
                "sha256:" + "1" * 64,
                2,
            ),
        ),
        MeasureResultEvidence(3, 2, 0, "sha256:" + "1" * 64),
        key_id="key.synthetic",
        signing_key=SYNTHETIC_KEY,
    )
    restored = QualityMeasureEvidencePacket.from_json(packet.to_json())
    if not restored.verify(SYNTHETIC_KEY):
        raise AssertionError("synthetic_signature_failed")
    tampered = packet.to_dict()
    tampered["result"]["numerator_count"] = 1
    try:
        QualityMeasureEvidencePacket.from_dict(tampered).verify(SYNTHETIC_KEY)
    except QualityMeasureEvidenceError as error:
        if error.code != "signature_mismatch":
            raise
        failure_code = error.code
    else:
        raise AssertionError("tampered_result_not_rejected")
    return {
        "workflow_id": "quality_measure_evidence",
        "passed": {"code": "verified", "packet_digest": restored.packet_digest},
        "fail_closed": {"code": failure_code},
    }


def main() -> dict:
    """Print deterministic JSON containing only digests, IDs and codes."""
    report = run_example()
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return report


if __name__ == "__main__":
    main()
