"""Offline synthetic contract demonstration, NOT clinical model validation.

The fixture NER and NLI providers below are test doubles restricted to this
embedded note. Review transitions describe synthetic golden evidence only.
They are not trained/calibrated models or real clinician approvals. No external
note input is accepted. Replace these integrations with reviewed, calibrated
local providers before building an application; do not deploy these doubles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openmed import analyze_text, deidentify
from openmed.clinical.brief import (
    BriefContext,
    BriefFact,
    _digest,
    brief_policy_fingerprint,
    build_clinical_brief,
)
from openmed.clinical.evidence_packet import (
    build_evidence_packet,
    fingerprint_evidence_review,
)
from openmed.clinical.grounding import VocabLoader, VocabSource, get_linker
from openmed.clinical.nli_gate import NLIThresholds
from openmed.clinical.review_state_machine import (
    ReviewState,
    ReviewStateMachine,
    make_opaque_event_id,
)
from openmed.core.offline import network_blocked_if_offline

SENTENCES = (
    "The admission problem was pneumonia.",
    "The discharge diagnosis was pneumonia.",
    "Symptoms improved after supportive care.",
)
NOTE = (
    "Synthetic note. Contact: demo.patient@example.test. Phone: 212-555-0198. "
    + " ".join(SENTENCES)
)
CALIBRATION = "synthetic-fixture-only-not-model-calibration"


class FixtureLoader:
    """Explicit offline detector double for one fixed synthetic note."""

    config = None

    def __init__(self, *, ner=False):
        self.ner = ner

    def get_max_sequence_length(self, *args, **kwargs):
        return 512

    def create_pipeline(self, *args, **kwargs):
        def predict(inputs, **kwargs):
            texts = inputs if isinstance(inputs, list) else [inputs]
            output = []
            for text in texts:
                spans = []
                if self.ner:
                    start = text.find("pneumonia")
                    if start >= 0:
                        spans.append(
                            {
                                "entity_group": "DISEASE",
                                "score": 1.0,
                                "start": start,
                                "end": start + 9,
                                "word": "pneumonia",
                            }
                        )
                output.append(spans)
            return output if isinstance(inputs, list) else output[0]

        predict.tokenizer = None
        return predict


def run_example(model="extractive"):
    """Run public APIs with explicit synthetic doubles; accept no external text."""
    with network_blocked_if_offline(local_only=True):
        value = deidentify(
            NOTE, method="mask", loader=FixtureLoader(), use_safety_sweep=True
        )
        assert "demo.patient@example.test" not in value.deidentified_text
        assert "212-555-0198" not in value.deidentified_text
        analysis = analyze_text(
            value.deidentified_text,
            model_name="synthetic-clinical-ner",
            loader=FixtureLoader(ner=True),
            confidence_threshold=0.5,
        )
        fixture = (
            Path(__file__).resolve().parents[1]
            / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"
        )
        loader = VocabLoader(
            registry={"icd10cm": VocabSource(system="icd10cm", path=fixture)}
        )
        linker = get_linker("icd10cm")(loader.get_index("icd10cm"))
        grounded = []
        for entity in analysis.entities:
            candidates = linker.link(
                entity.text, canonical_label=entity.label, fuzzy=False
            )
            grounded.append(
                {
                    "start": entity.start,
                    "end": entity.end,
                    "candidate_count": len(candidates),
                }
            )
        facts = tuple(
            BriefFact(
                "synthetic:brief-" + str(i),
                field,
                "affirmed",
                "certain",
                "recent",
                "patient",
            )
            for i, field in enumerate(
                ("admission_reason", "discharge_diagnoses", "hospital_course")
            )
        )
        policy = brief_policy_fingerprint(value.deidentified_text, facts)
        rows = []
        for fact, sentence in zip(facts, SENTENCES):
            start = value.deidentified_text.index(sentence)
            row = {
                "reference_id": fact.reference_id,
                "source_id": "synthetic:brief-demo",
                "start": start,
                "end": start + len(sentence),
                "policy_fingerprint": policy,
                "review_state": "approved",
                "synthetic": True,
                "verified": True,
            }
            binding = fingerprint_evidence_review(
                **{
                    k: row[k]
                    for k in (
                        "reference_id",
                        "source_id",
                        "start",
                        "end",
                        "policy_fingerprint",
                    )
                }
            )
            machine = ReviewStateMachine()
            for state in (ReviewState.IN_REVIEW, ReviewState.APPROVED):
                machine.transition(
                    state,
                    make_opaque_event_id((fact.reference_id, state.value)),
                    binding,
                )
            row["review_transitions"] = machine.transitions
            rows.append(row)

        def fixture_nli(premise, hypothesis):
            exact = premise == hypothesis and premise in SENTENCES
            return {
                "entailment": float(exact),
                "contradiction": 0.0,
                "neutral": float(not exact),
                "calibration_id": CALIBRATION,
            }

        def detector(text):
            # Fixed-fixture source-identifier scan, not a general PHI detector.
            # Inspect the complete rendered packet without misclassifying its
            # SHA-256 metadata as phone/account numbers.
            return [
                {
                    "label": entity.label,
                    "start": text.index(entity.text),
                    "end": text.index(entity.text) + len(entity.text),
                }
                for entity in value.pii_entities
                if entity.text in text
            ]

        context = BriefContext(
            build_evidence_packet(rows, policy_fingerprint=policy),
            _digest(value.deidentified_text),
            facts,
            fixture_nli,
            NLIThresholds(
                calibration_id=CALIBRATION, calibration_method="synthetic-fixture"
            ),
            detector,
        )
        brief = build_clinical_brief(value, context=context, model=model)
        return {
            "synthetic_only": True,
            "model_validation": False,
            "providers": "fixture-ner-and-nli-not-deployable",
            "deidentified_digest": _digest(value.deidentified_text),
            "extraction_count": len(analysis.entities),
            "grounding": grounded,
            "brief": brief.to_response(),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("extractive", "mlx"), default="extractive")
    args = parser.parse_args(argv)
    result = run_example(args.model)
    print(
        "SYNTHETIC CONTRACT DEMO — fixture NER/NLI; not clinical or model validation."
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["brief"]["status"] == "refused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
