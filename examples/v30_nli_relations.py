"""Offline synthetic NLI and guarded-relation contracts; no clinical validation.

Run ``python -m examples.v30_nli_relations`` from the repository root. This
example accepts no external input or arguments and loads no model. JSONL output
contains controlled labels, scores, flags, offsets, digests and review priorities.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from typing import Any

from openmed.clinical.nli import verify
from openmed.clinical.nli_gate import EvidenceLink, NLIThresholds, evaluate_nli
from openmed.clinical.relations.diagnosis_treatments import (
    generate_diagnosis_treatment_candidates,
)
from openmed.clinical.relations.evidence_binding import bind_relation_evidence
from openmed.clinical.relations.procedure_indications import (
    generate_procedure_indication_candidates,
)
from openmed.clinical.relations.review_priority import assign_review_priority
from openmed.clinical.sections import detect_sections
from openmed.core.offline import network_blocked_if_offline

SYNTHETIC_PREMISE = "Symptoms improved after fluids."
SYNTHETIC_CLAIMS = (
    SYNTHETIC_PREMISE,
    "Symptoms did not improve after fluids.",
    "A procedure is indicated.",
)
SYNTHETIC_NOTE = (
    "Assessment: possible pneumonia treated with ceftriaxone.\n"
    "Plan: biopsy performed for possible lung mass."
)


class SyntheticCalibratedNLI:
    """Calibration-contract double with authored, non-empirical probabilities.

    This callable is synthetic application code, not a trained or qualified
    checkpoint. Its calibration metadata identifies the fixture contract only.
    """

    backend_id = "synthetic-calibrated-double"

    def __init__(self) -> None:
        self.thresholds = NLIThresholds(
            entailment=0.9,
            contradiction=0.9,
            margin=0.05,
            calibration_id="synthetic-fixture-only",
            calibration_method="synthetic-fixture",
        )

    def __call__(self, premise: str, hypothesis: str) -> dict[str, str | float]:
        """Apply the public selective gate to one fixed synthetic score vector.

        Args:
            premise: Exact embedded synthetic source.
            hypothesis: One embedded synthetic claim.

        Returns:
            A selected label and score without input text.

        Raises:
            RuntimeError: If a different fixture is supplied.
        """
        if premise != SYNTHETIC_PREMISE or hypothesis not in SYNTHETIC_CLAIMS:
            raise RuntimeError("synthetic_nli_fixture_required")
        index = SYNTHETIC_CLAIMS.index(hypothesis)
        probabilities = {
            "entailment": float(index == 0),
            "contradiction": float(index == 1),
            "neutral": float(index == 2),
        }
        gate = evaluate_nli(
            probabilities,
            EvidenceLink.from_text(
                source_id="synthetic:source",
                claim_id=f"synthetic:claim-{index}",
                source_text=premise,
                claim_text=hypothesis,
            ),
            thresholds=self.thresholds,
        )
        return {
            "label": "abstention" if gate.outcome == "abstain" else gate.outcome,
            "score": gate.selected_probability,
        }


def _span(surface: str, label: str) -> dict[str, Any]:
    start = SYNTHETIC_NOTE.index(surface)
    return {
        "label": label,
        "start": start,
        "end": start + len(surface),
        "certainty": "uncertain",  # Authored synthetic annotation, not model output.
    }


def build_demo_records() -> list[dict[str, Any]]:
    """Run actual local NLI, candidate, binding and review-priority APIs.

    Returns:
        Value-free projections of the fixed synthetic demonstration.

    Raises:
        RuntimeError: If a fixed public-API contract drifts.
    """
    records: list[dict[str, Any]] = [
        {
            "record": "demonstration",
            "synthetic": True,
            "clinical_validation": False,
            "requires_human_review": True,
            "autonomous_decision": False,
        }
    ]
    with network_blocked_if_offline(local_only=True):
        for usage, backend, expected in (
            (
                "heuristic-development-only",
                "heuristic",
                ("entailment", "contradiction", "neutral"),
            ),
            (
                "synthetic-calibration-double",
                SyntheticCalibratedNLI(),
                ("entailment", "contradiction", "abstention"),
            ),
        ):
            verdicts = verify(SYNTHETIC_CLAIMS, SYNTHETIC_PREMISE, backend=backend)
            if tuple(item["label"] for item in verdicts) != expected:
                raise RuntimeError("synthetic_nli_contract_drift")
            records.extend(
                {
                    "record": "nli",
                    "usage": usage,
                    "claim_index": verdict["claim_index"],
                    "label": verdict["label"],
                    "score": verdict["score"],
                    "backend_id": verdict["backend_id"],
                    "contradicted": verdict["contradicted"],
                    "review_required": verdict["review_required"],
                }
                for verdict in verdicts
            )

        sections = detect_sections(SYNTHETIC_NOTE)
        diagnosis = generate_diagnosis_treatment_candidates(
            SYNTHETIC_NOTE,
            [_span("pneumonia", "DIAGNOSIS"), _span("ceftriaxone", "MEDICATION")],
            sections=sections,
        )
        procedure = generate_procedure_indication_candidates(
            SYNTHETIC_NOTE,
            [_span("biopsy", "PROCEDURE"), _span("lung mass", "CONDITION")],
            sections=sections,
        )
        if len(diagnosis) != 1 or len(procedure) != 1:
            raise RuntimeError("synthetic_relation_candidate_drift")

        for candidate, head, tail in (
            (diagnosis[0], diagnosis[0].diagnosis, diagnosis[0].treatment),
            (procedure[0], procedure[0].procedure, procedure[0].indication),
        ):
            # Candidate serialization provides relation_type; endpoint names
            # differ by candidate family, so bind them explicitly.
            guarded = bind_relation_evidence(
                candidate.to_dict(),
                document_id="synthetic:relations",
                head=head,
                tail=tail,
                evidence_spans=[candidate.linking_cue],
                assertion_state="uncertain",
                document_text=SYNTHETIC_NOTE,
            )
            priority = assign_review_priority(
                guarded.relation_type,
                conflict_state="none",
                evidence_completeness="complete",
            )
            if not guarded.requires_clinician_review or guarded.autonomous_decision:
                raise RuntimeError("synthetic_relation_review_contract_drift")
            records.append(
                {
                    "record": "relation",
                    "relation_type": guarded.relation_type,
                    "assertion_state": guarded.assertion_status,
                    "confidence": guarded.confidence,
                    "head_span": list(guarded.head.offset),
                    "tail_span": list(guarded.tail.offset),
                    "evidence_spans": [list(item.offset) for item in guarded.evidence],
                    "evidence_digests": [item.text_hash for item in guarded.evidence],
                    "binding_digest": "sha256:"
                    + hashlib.sha256(guarded.to_json().encode()).hexdigest(),
                    "requires_clinician_review": guarded.requires_clinician_review,
                    "autonomous_decision": guarded.autonomous_decision,
                    "review_band": priority.band,
                    "policy_score": priority.policy_score,
                    "clinical_urgency_inferred": priority.clinical_urgency_inferred,
                }
            )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    """Print the fixed demo or a content-free refusal, with no partial output.

    Args:
        argv: CLI arguments, used only to refuse any supplied arguments.

    Returns:
        Zero on success, two for arguments or one for a contract failure.
    """
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        print("synthetic_relation_example_accepts_no_arguments", file=sys.stderr)
        return 2
    failed = False
    try:
        records = build_demo_records()
    except Exception:
        failed = True
    if failed:
        print("synthetic_relation_example_contract_failed", file=sys.stderr)
        return 1
    for record in records:
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
