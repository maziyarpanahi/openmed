"""Fixed synthetic providers for full brief regression, never model evidence."""

import json
from datetime import datetime
from pathlib import Path

from openmed.clinical.brief import (
    BriefContext,
    BriefFact,
    _digest,
    brief_policy_fingerprint,
)
from openmed.clinical.evidence_packet import (
    build_evidence_packet,
    fingerprint_evidence_review,
)
from openmed.clinical.nli_gate import NLIThresholds
from openmed.clinical.review_state_machine import (
    ReviewState,
    ReviewStateMachine,
    make_opaque_event_id,
)
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.core.text_normalize import normalize_for_detection

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "tests/fixtures/eval/summaries/multilingual_briefs.json"
WIRE = ROOT / "tests/fixtures/eval/summaries/multilingual_brief_packets.json"
REVIEW_ID = "a" * 64
CALIBRATION = "synthetic-multilingual-test-only"
SCENARIOS = (
    "preserved",
    "negation_conflict",
    "family_conflict",
    "temporal_conflict",
    "nli_contradiction",
    "unsupported_provider",
)


def corpus():
    """Load the versioned, explicitly synthetic test corpus."""
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def fixture_context(case, scenario="preserved"):
    """Normalize and gold-mask existing ID traps, then review synthetic spans.

    Gold masking isolates composition; it measures neither ID recall nor NLI
    quality. Providers declare support for these exact fixture pairs only.
    """
    traps = json.loads(
        (ROOT / "openmed/eval/golden/fixtures/per_language_id_traps.json").read_text()
    )["fixtures"]
    trap = next(row for row in traps if row["id"] == case["identifier_fixture"])
    identifier = trap["gold_spans"][0]["text"]
    raw = case["prefix"] + identifier + "\n" + " ".join(case["sentences"])
    source, normalization = normalize_for_detection(raw)
    identifier, _ = normalize_for_detection(identifier)
    start = source.index(identifier)
    entity = PIIEntity(
        text=identifier,
        label="ID_NUM",
        start=start,
        end=start + len(identifier),
        confidence=1.0,
        redacted_text="[ID_NUM]",
    )
    text = source[:start] + entity.redacted_text + source[entity.end :]
    value = DeidentificationResult(
        source,
        text,
        [entity],
        "mask",
        datetime(2026, 1, 1),
        mapping={"[ID_NUM]": identifier},
    )
    sentences = tuple(normalize_for_detection(s)[0] for s in case["sentences"])
    fields = ("admission_reason", "discharge_diagnoses", "hospital_course")
    facts = tuple(
        BriefFact(
            "synthetic:ref-" + str(i),
            field,
            "negated" if i == 0 else "affirmed",
            "certain",
            "historical" if i == 2 else "recent",
            "family" if i == 1 else "patient",
        )
        for i, field in enumerate(fields)
    )
    policy = brief_policy_fingerprint(text, facts)
    rows = []
    for sentence, fact in zip(sentences, facts):
        start = text.index(sentence)
        row = dict(
            reference_id=fact.reference_id,
            source_id="synthetic:document",
            start=start,
            end=start + len(sentence),
            policy_fingerprint=policy,
            review_state="approved",
            synthetic=True,
            verified=True,
        )
        fingerprint = fingerprint_evidence_review(
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
                fingerprint,
            )
        row["review_transitions"] = machine.transitions
        rows.append(row)

    def predict(premise, hypothesis):
        # Missing declared support is an explicit unavailable outcome, not a
        # locale lookup, guessed score, or invocation of a real model.
        if scenario == "unsupported_provider":
            return {}
        assert premise in sentences and hypothesis == premise
        label = "contradiction" if scenario == "nli_contradiction" else "entailment"
        return {"label": label, "score": 1.0, "calibration_id": CALIBRATION}

    def privacy_detector(candidate):
        return (
            [{"label": "ID_NUM", "start": 0, "end": 1, "critical": True}]
            if identifier in candidate
            else []
        )

    context = BriefContext(
        build_evidence_packet(rows, policy_fingerprint=policy),
        _digest(text),
        facts,
        predict,
        NLIThresholds(
            calibration_id=CALIBRATION, calibration_method="synthetic-fixture"
        ),
        privacy_detector,
    )
    generated = list(sentences)
    if scenario in SCENARIOS[1:4]:
        index = SCENARIOS.index(scenario) - 1
        generated[index] = normalize_for_detection(case["conflicts"][index])[0]
    return raw, normalization, value, context, " ".join(generated)


def parity_report(case, scenario, responses):
    """Return counts/digests/codes only, separate from model qualification."""
    from openmed.clinical.summary_claim_segments import segment_summary_claims

    # Import the existing catalog to populate the existing registry.
    from openmed.core import language_pack_catalog  # noqa: F401
    from openmed.core.language_pack import get_language_pack

    first = next(iter(responses.values()))
    atomicity_supported = all(
        not segment_summary_claims("Synthetic claim.", language=lang).review_required
        for lang in case["languages"]
    )
    return {
        "schema_version": 1,
        "case_digest": _digest(case),
        "scenario": scenario,
        "contract_parity": all(row == first for row in responses.values()),
        "surface_count": len(responses),
        "outcome": first["status"],
        "reason": first["refusal_reason"],
        "citation_count": len(first["citations"]),
        "packet_digest": first["digest"],
        "language_pack_count": sum(
            get_language_pack(lang) is not None for lang in case["languages"]
        ),
        "claim_atomicity": "supported" if atomicity_supported else "unsupported",
        "provider_support": (
            "unsupported"
            if scenario == "unsupported_provider"
            else "synthetic_fixture_pairs_only"
        ),
        "model_quality": "not_evaluated",
        "clinical_language_support": "not_established",
        "synthetic_only": True,
    }
