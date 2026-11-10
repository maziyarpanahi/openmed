#!/usr/bin/env python3
"""Synthetic structured-data release-risk workflow.

This offline example demonstrates:

1. advisory quasi-identifier discovery;
2. explicit expert role review and policy selection;
3. complete patient-level k/l/t assessment;
4. generalization and whole-patient suppression;
5. materialized-output reread and validation; and
6. aggregate-only evidence for qualified expert review.

All input records are synthetic. The generated release remains a data artifact,
while the discovery, assessment, validation, and expert-review files contain
only allow-listed aggregate evidence.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from openmed.compliance import (
    ExpertReviewEvidenceReport,
    ReleaseAssumptions,
    build_release_expert_review_evidence,
)
from openmed.core.audit import stable_hash
from openmed.risk import (
    AnonymityPolicy,
    anonymize_release,
    assess_release,
    validate_released_output,
)
from openmed.structured import read_table, scan_table, write_table

SYNTHETIC_ROWS: tuple[dict[str, Any], ...] = (
    {
        "patient_id": "synthetic-patient-001",
        "encounter_id": "synthetic-encounter-001",
        "full_name": "Avery Example",
        "age": 31,
        "zip": "10001",
        "visit_date": "2024-01-01",
        "disease": "influenza",
    },
    {
        "patient_id": "synthetic-patient-002",
        "encounter_id": "synthetic-encounter-002",
        "full_name": "Blair Example",
        "age": 32,
        "zip": "10002",
        "visit_date": "2024-01-02",
        "disease": "common-cold",
    },
    {
        "patient_id": "synthetic-patient-003",
        "encounter_id": "synthetic-encounter-003",
        "full_name": "Casey Example",
        "age": 41,
        "zip": "20001",
        "visit_date": "2024-01-03",
        "disease": "influenza",
    },
    {
        "patient_id": "synthetic-patient-004",
        "encounter_id": "synthetic-encounter-004",
        "full_name": "Devon Example",
        "age": 42,
        "zip": "20002",
        "visit_date": "2024-01-04",
        "disease": "common-cold",
    },
)


def run_workflow(output_dir: Path) -> dict[str, Any]:
    """Run the synthetic workflow and return an aggregate-only summary."""

    output_dir.mkdir(parents=True, exist_ok=False)
    source_path = output_dir / "synthetic-cohort.jsonl"
    discovery_path = output_dir / "qi-discovery.json"
    assessment_path = output_dir / "pre-release-assessment.json"
    release_path = output_dir / "validated-release.jsonl"
    validation_path = output_dir / "release-validation.json"
    evidence_path = output_dir / "expert-review-evidence.json"
    evidence_markdown_path = output_dir / "expert-review-evidence.md"

    write_table(source_path, SYNTHETIC_ROWS)

    # Discovery is deliberately advisory. A qualified reviewer must confirm
    # roles in the context of the population, recipient, and intended release.
    discovery = scan_table(
        source_path,
        privacy_unit="patient_id",
        quasi_identifier_columns=("age", "zip", "visit_date"),
        sensitive_columns=("disease",),
        role_overrides={
            "encounter_id": ("direct-id", "internal-linkage"),
            "full_name": "direct-id",
        },
    )
    discovery_path.write_text(
        json.dumps(discovery, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # This policy represents the explicit role-review decision. Thresholds are
    # examples for synthetic data, not regulatory defaults.
    policy = AnonymityPolicy(
        quasi_identifiers=("age", "zip", "visit_date"),
        sensitive_attributes=("disease",),
        direct_identifiers=("encounter_id", "full_name"),
        privacy_unit="patient_id",
        target_k=2,
        target_l=2,
        l_metric="distinct",
        target_t=0.0,
        suppression_rate=0.0,
        max_lattice_nodes=100_000,
        max_suppression_subsets=100_000,
    )

    source_rows = read_table(source_path)
    assessment = assess_release(source_rows, policy)
    assessment_path.write_text(assessment.to_json() + "\n", encoding="utf-8")

    result = anonymize_release(source_rows, policy)
    write_table(release_path, result.records)

    # Final validation is performed on the bytes materialized for release, not
    # only on the in-memory transformation result.
    reread_rows = read_table(release_path)
    validation = validate_released_output(reread_rows, result)
    if not validation.passed:
        raise RuntimeError("Synthetic materialized release failed validation")
    validation_path.write_text(
        json.dumps(validation.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assumptions = ReleaseAssumptions(
        privacy_unit="patient",
        population_scope="release_cohort",
        release_model="restricted",
        recipient_model="named_researchers",
        auxiliary_data_model="reasonably_available",
        notes_digest=stable_hash(
            {
                "kind": "synthetic-release-assumptions",
                "purpose": "offline documentation example",
            }
        ),
    )
    evidence = build_release_expert_review_evidence(
        result,
        validation=validation,
        assumptions=assumptions,
    )
    evidence_path.write_text(evidence.to_json() + "\n", encoding="utf-8")
    evidence_markdown_path.write_text(evidence.to_markdown(), encoding="utf-8")

    reparsed = ExpertReviewEvidenceReport.from_json(
        evidence_path.read_text(encoding="utf-8")
    )
    if not reparsed.verify():
        raise RuntimeError("Synthetic expert-review evidence failed verification")

    return {
        "output_directory": str(output_dir),
        "discovery_status": discovery["discovery"]["status"],
        "discovery_advisory": discovery["discovery"]["advisory"],
        "pre_release_meets_policy": assessment.meets_policy,
        "post_release_meets_policy": result.after.meets_policy,
        "achieved_k": result.after.achieved_k,
        "released_rows": result.utility.released_rows,
        "materialized_release_valid": validation.passed,
        "expert_review_evidence_verified": reparsed.verify(),
        "qualified_expert_review_required": True,
        "not_an_expert_determination": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "New directory for synthetic artifacts. When omitted, a new "
            "temporary directory is created and retained for inspection."
        ),
    )
    return parser


def main() -> None:
    """Run the example and print its aggregate-only summary."""

    args = _parser().parse_args()
    output_dir = args.output_dir or (
        Path(tempfile.mkdtemp(prefix="openmed-release-risk-")) / "artifacts"
    )
    summary = run_workflow(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
