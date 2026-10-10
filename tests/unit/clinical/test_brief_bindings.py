"""Synthetic v1 bindings: deterministic scores are contract tests only."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.clinical.brief import STAGES, BriefRefusal, build_clinical_brief
from openmed.clinical.summarize_backends import (
    MAX_OUTPUT_BYTES,
    BriefGeneratedClaim,
    BriefGenerationResult,
    LocalSummarizerError,
)
from tests.unit.clinical.test_brief import SENTENCES, fixture_context

PARAPHRASES = (
    "Dehydration was the reason for admission.",
    "Dehydration was diagnosed at discharge.",
    "Symptoms improved following fluid treatment.",
)


def generation(texts=PARAPHRASES):
    return BriefGenerationResult(
        tuple(
            BriefGeneratedClaim(text, (f"synthetic:ref-{i}",))
            for i, text in enumerate(texts)
        )
    )


class SyntheticBoundGenerator:
    def __init__(self, output=None):
        self.output = generation() if output is None else output
        self.seen = None

    def generate_brief(self, evidence, *, mode):
        self.seen = evidence
        assert mode == "bhc"
        return self.output


def test_paraphrases_reach_all_guards_with_exact_traceable_offsets():
    result, context = fixture_context()
    seen = []

    def predict(premise, hypothesis):
        seen.append((premise, hypothesis))
        return context.nli_predict(premise, hypothesis)

    backend = SyntheticBoundGenerator()
    brief = build_clinical_brief(
        result, context=replace(context, nli_predict=predict), model=backend
    )
    assert brief.refusal_reason is None
    assert brief.summary == generation().render()
    assert seen == list(zip(SENTENCES, PARAPHRASES))
    assert brief.to_dict()["stages"] == list(STAGES)
    for i, citation in enumerate(brief.citations):
        assert (
            result.deidentified_text[citation["source_start"] : citation["source_end"]]
            == SENTENCES[i]
        )
        assert (
            brief.summary[citation["output_start"] : citation["output_end"]]
            == (PARAPHRASES[i])
        )
        assert backend.seen[i].text == SENTENCES[i]
        assert backend.seen[i].start == citation["source_start"]
        assert SENTENCES[i] not in repr(backend.seen[i])
    safe = json.dumps(brief.to_dict())
    assert all(text not in safe for text in (*SENTENCES, *PARAPHRASES))
    assert "synthetic:ref-" not in safe
    assert brief.metrics["coverage"]["recall"] == 1.0
    assert brief.to_dict()["generation_contract"]["schema_version"] == 1
    assert brief.envelope["requires_human_review"]
    assert build_clinical_brief(result, context=context, model=backend).digest == (
        brief.digest
    )


@pytest.mark.parametrize(
    "output",
    [
        BriefGenerationResult((), 1),
        replace(generation(), schema_version=2),
        replace(generation(), schema_version=True),
        replace(generation(), claims=list(generation().claims)),
        BriefGenerationResult((BriefGeneratedClaim("Signal improved.", ()),)),
        BriefGenerationResult(
            (BriefGeneratedClaim("Signal improved.", ("invented:ref",)),)
        ),
        BriefGenerationResult(
            (BriefGeneratedClaim("Signal improved.", ("synthetic:ref-0",) * 2),)
        ),
        BriefGenerationResult(
            (
                BriefGeneratedClaim(
                    "Signal improved.", ("synthetic:ref-0", "synthetic:ref-1")
                ),
            )
        ),
        BriefGenerationResult((BriefGeneratedClaim("Signal improved.", ("x" * 257,)),)),
        BriefGenerationResult(
            (BriefGeneratedClaim(" Signal improved.", ("synthetic:ref-0",)),)
        ),
        BriefGenerationResult(
            (
                BriefGeneratedClaim(
                    "Signal improved. Marker persisted.", ("synthetic:ref-0",)
                ),
            )
        ),
        BriefGenerationResult(
            (
                BriefGeneratedClaim(
                    "Signal improved and marker persisted.", ("synthetic:ref-0",)
                ),
            )
        ),
        BriefGenerationResult(
            (BriefGeneratedClaim("x" * (MAX_OUTPUT_BYTES + 1), ("synthetic:ref-0",)),)
        ),
        BriefGenerationResult(generation().claims * 22),
        {"schema_version": 1, "claims": [], "confidence": 1.0},
        generation().render(),
    ],
)
def test_invalid_or_ambiguous_contracts_refuse_without_verification(output):
    result, context = fixture_context()
    brief = build_clinical_brief(
        result,
        context=replace(context, nli_predict=lambda *_: pytest.fail("NLI called")),
        model=SyntheticBoundGenerator(output),
    )
    assert brief.refusal_reason is not None
    assert brief.summary == ""
    assert brief.citations == ()
    assert "nli" not in brief.to_dict()["stages"]
    assert "Signal improved" not in json.dumps(brief.to_dict())


@pytest.mark.parametrize("label", ["contradiction", "neutral"])
def test_binding_is_never_proof_of_support(label):
    result, context = fixture_context()
    brief = build_clinical_brief(
        result,
        context=replace(
            context,
            nli_predict=lambda *_: {
                "label": label,
                "score": 1.0,
                "calibration_id": "synthetic-test-only",
            },
        ),
        model=SyntheticBoundGenerator(),
    )
    assert brief.refusal_reason is BriefRefusal.NLI_REJECTED
    assert brief.summary == ""
    assert brief.citations == ()


def test_free_form_paraphrases_remain_unbound():
    result, context = fixture_context()
    brief = build_clinical_brief(
        result, context=context, model=lambda _: generation().render()
    )
    assert brief.refusal_reason is BriefRefusal.UNSUPPORTED_CLAIM


def test_structured_exact_claims_are_supported_without_changing_legacy_packet():
    result, context = fixture_context()
    legacy = build_clinical_brief(result, context=context, model="extractive")
    bound = build_clinical_brief(
        result, context=context, model=SyntheticBoundGenerator(generation(SENTENCES))
    )
    assert bound.summary == legacy.summary
    assert bound.citations == legacy.citations
    assert "generation_contract" not in legacy.to_dict()


@pytest.mark.parametrize(
    "identifier",
    ["SYNTHETIC_SENTINEL", "person@example.test", "+33 6 12 34 56 78", "患者識別子"],
)
def test_bound_output_does_not_bypass_direct_identifier_leakage(identifier):
    result, context = fixture_context()
    result.mapping = {"[IDENTIFIER]": identifier}
    output = generation((f"{identifier} was admitted.", *PARAPHRASES[1:]))
    brief = build_clinical_brief(
        result, context=context, model=SyntheticBoundGenerator(output)
    )
    assert brief.refusal_reason is BriefRefusal.PRIVACY
    assert brief.summary == ""
    assert identifier not in json.dumps(brief.to_dict())


def test_bound_output_and_complete_packet_are_privacy_scanned():
    result, context = fixture_context()
    seen = []

    def detector(text):
        seen.append(text)
        if '"claim_bindings"' in text:
            return [{"label": "NAME", "start": 0, "end": 1, "critical": True}]
        return []

    brief = build_clinical_brief(
        result,
        context=replace(context, privacy_detector=detector),
        model=SyntheticBoundGenerator(),
    )
    assert brief.refusal_reason is BriefRefusal.PRIVACY
    assert len(seen) == 2
    assert brief.summary == ""


def test_contract_errors_and_representations_are_value_free():
    output = BriefGenerationResult(
        (BriefGeneratedClaim("PRIVATE_SENTINEL", ("PRIVATE_REFERENCE", "extra")),)
    )
    with pytest.raises(LocalSummarizerError) as error:
        output.render()
    assert "PRIVATE" not in str(error.value)
    assert "PRIVATE" not in repr(output)
    assert "PRIVATE" not in repr(output.claims[0])


def test_shared_native_paraphrase_packet_matches_composer():
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/clinical/brief_parity/bound.json"
        ).read_text()
    )
    value, context = fixture_context()
    brief = build_clinical_brief(
        value, context=context, model=SyntheticBoundGenerator()
    )
    assert fixture["source"] == value.deidentified_text
    assert fixture["generation"] == {
        "schema_version": 1,
        "claims": [
            {"text": c.text, "reference_ids": list(c.reference_ids)}
            for c in generation().claims
        ],
    }
    assert fixture["evaluation_json"] == json.dumps(
        brief.to_response(), sort_keys=True, separators=(",", ":")
    )


@pytest.mark.parametrize(
    "offsets", [(0, len(SENTENCES[0])), (1, len(SENTENCES[0]) - 1)]
)
def test_overlapping_reviewed_bindings_are_ambiguous(offsets):
    from openmed.clinical.evidence_packet import (
        build_evidence_packet,
        fingerprint_evidence_review,
    )
    from openmed.clinical.review_state_machine import (
        ReviewState,
        ReviewStateMachine,
        make_opaque_event_id,
    )

    value, context = fixture_context()
    rows = [r.to_dict() for r in context.packet.references]
    rows[1]["start"], rows[1]["end"] = offsets
    row = rows[1]
    fingerprint = fingerprint_evidence_review(
        **{
            k: row[k]
            for k in ("reference_id", "source_id", "start", "end", "policy_fingerprint")
        }
    )
    machine = ReviewStateMachine()
    for state in (ReviewState.IN_REVIEW, ReviewState.APPROVED):
        machine.transition(
            state, make_opaque_event_id((fingerprint, state.value)), fingerprint
        )
    row["review_transitions"] = machine.transitions
    packet = build_evidence_packet(
        rows, policy_fingerprint=context.packet.policy_fingerprint
    )
    assert len(packet.references) == 3
    brief = build_clinical_brief(
        value,
        context=replace(
            context, packet=packet, nli_predict=lambda *_: pytest.fail("NLI called")
        ),
        model=SyntheticBoundGenerator(),
    )
    assert brief.refusal_reason is BriefRefusal.UNSUPPORTED_CLAIM
    assert brief.summary == ""


def test_unicode_output_uses_scalar_offsets_not_utf8_byte_offsets():
    value, context = fixture_context()
    texts = (*PARAPHRASES[:2], "Symptoms improved following fluids 🧪.")
    brief = build_clinical_brief(
        value, context=context, model=SyntheticBoundGenerator(generation(texts))
    )
    assert brief.refusal_reason is None
    citation = brief.citations[-1]
    assert brief.summary[citation["output_start"] : citation["output_end"]] == texts[-1]
    assert citation["output_end"] == len(brief.summary)
    assert citation["output_end"] < len(brief.summary.encode())


@pytest.mark.parametrize("field", ["text", "reference"])
def test_direct_generation_render_does_not_retain_malformed_private_values(field):
    private = "SYNTHETIC_PRIVATE_IDENTIFIER\ud800"
    claim = BriefGeneratedClaim(
        private if field == "text" else "Symptoms improved.",
        (private if field == "reference" else "synthetic:ref-0",),
    )
    with pytest.raises(LocalSummarizerError) as caught:
        BriefGenerationResult((claim,)).render()
    assert caught.value.__context__ is None
    assert "SYNTHETIC_PRIVATE_IDENTIFIER" not in str(caught.value)


def test_actual_bound_brief_matches_bundled_and_service_response_contracts():
    from openmed.clinical.record_schemas import validate_clinical_record
    from openmed.service.schemas import BriefResponse

    value, context = fixture_context()
    brief = build_clinical_brief(
        value, context=context, model=SyntheticBoundGenerator()
    )
    assert brief.refusal_reason is None
    validate_clinical_record("brief_audit", brief.to_dict())
    validate_clinical_record("brief_response", brief.to_response())
    response = BriefResponse.model_validate(brief.to_response())
    assert response.model_dump(exclude_unset=True) == brief.to_response()


@pytest.mark.parametrize("schema", ["brief_audit", "brief_response"])
@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.pop("generation_contract"),
        lambda x: x.pop("claim_bindings"),
        lambda x: x["generation_contract"].update(schema_version=True),
        lambda x: x["generation_contract"].update(text="Synthetic unknown value"),
        lambda x: x["claim_bindings"][0].update(reference_digest="invalid"),
        lambda x: x["claim_bindings"].pop(),
        lambda x: x["claim_bindings"][0].update(claim_index=1),
        lambda x: x["citations"][0].update(claim_index=1),
    ],
)
def test_bound_record_contract_rejects_inconsistent_metadata(schema, mutation):
    from openmed.clinical.record_schemas import (
        ClinicalRecordSchemaError,
        validate_clinical_record,
    )

    value, context = fixture_context()
    brief = build_clinical_brief(
        value, context=context, model=SyntheticBoundGenerator()
    )
    record = brief.to_dict() if schema == "brief_audit" else brief.to_response()
    mutation(record)
    with pytest.raises(ClinicalRecordSchemaError) as caught:
        validate_clinical_record(schema, record)
    assert "Synthetic unknown value" not in str(caught.value)
