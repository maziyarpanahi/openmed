"""Full-pipeline synthetic regressions independent of trained model accuracy."""

from __future__ import annotations

import json
import re

import pytest

from openmed.core.clinical_privacy import (
    ClinicalPrivacyDocument,
    ClinicalPrivacyOptions,
    ClinicalPrivacyProcessor,
)
from openmed.onnx.inference import OnnxEntity, OnnxPrediction


class Model:
    variant = "fp32"

    def __init__(self):
        self.calls = []

    def predict_batch_detailed(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        results = []
        for text in texts:
            entities = []
            for term, label in [
                ("Anna", "first_name"),
                ("Kurznarkose", "city"),
                ("Keine", "first_name"),
            ]:
                for match in re.finditer(term, text):
                    entities.append(
                        OnnxEntity(label, 0.99, match.start(), match.end(), match[0])
                    )
            results.append(
                OnnxPrediction(
                    tuple(entities), len(text.split()), len(text.split()), 1, True
                )
            )
        return results


def processor(**kwargs):
    return ClinicalPrivacyProcessor(
        Model(), model_id="synthetic-model", revision="f" * 40, **kwargs
    )


def doc(text, *, id="note-1", **kwargs):
    return ClinicalPrivacyDocument(
        id, text, ClinicalPrivacyOptions(language="de", **kwargs)
    )


TEXT = (
    "Patientin: Anna Beispiel, geboren am 14.03.1962.\r\n"
    "Sozialanamnese: Nichtraucherin. Keine Dyspnoe.\n"
    "Kardioversion in Kurznarkose. Metoprolol 47,5 mg; LVEF 55 %."
)


def test_batched_pipeline_retains_order_options_and_clinical_bytes():
    engine = processor()
    results = engine.process_batch([doc(TEXT), doc(TEXT, id="note-2", method="remove")])
    assert len(engine.model.calls) == 1
    assert len(engine.model.calls[0][0]) == 2
    assert [result.id for result in results] == ["note-1", "note-2"]
    for result in results:
        assert result.complete and result.status == "needs_review"
        assert result.warnings == ("model_language_not_qualified",)
        assert "Beispiel" not in result.deidentified_text
        assert "14.03.1962" not in result.deidentified_text
        assert result.deidentified_text.endswith(TEXT[TEXT.index(".\r\n") :])
        assert "original_text" not in result.to_dict()
        assert "Anna" not in json.dumps(result.spans)
    assert "[PERSON]" in results[0].deidentified_text
    assert "[PERSON]" not in results[1].deidentified_text


def test_subword_false_positives_preserve_whole_clinical_words_not_actual_names():
    class SubwordModel(Model):
        def predict_batch_detailed(self, texts, **kwargs):
            predictions = []
            for text in texts:
                entities = tuple(
                    OnnxEntity("first_name", 0.99, match.start(), match.end(), match[0])
                    for match in re.finditer(r" Ke| Dys| Park", text)
                )
                predictions.append(OnnxPrediction(entities, 20, 20, 1, True))
            return predictions

    engine = ClinicalPrivacyProcessor(
        SubwordModel(), model_id="fixture", revision="f" * 40
    )
    source = "Patient: Keine Parkinson. Keine Dyspnoe. Morbus Parkinson."
    result = engine.process_batch([doc(source)])[0]
    assert result.complete
    assert (
        result.deidentified_text
        == "Patient: [PERSON]. Keine Dyspnoe. Morbus Parkinson."
    )


def test_invalid_item_is_isolated_without_skipping_other_outputs():
    engine = processor()
    results = engine.process_batch(
        [doc(TEXT), doc("", id="empty"), doc(TEXT, id="last")]
    )
    assert [result.status for result in results] == [
        "needs_review",
        "failed",
        "needs_review",
    ]
    assert results[1].error == "invalid_document_or_controls"
    assert len(engine.model.calls[0][0]) == 2


def test_model_failure_returns_no_partial_success_or_error_text(monkeypatch):
    engine = processor()

    def fail(*args, **kwargs):
        raise RuntimeError("sensitive synthetic marker Anna Beispiel")

    monkeypatch.setattr(engine.model, "predict_batch_detailed", fail)
    results = engine.process_batch([doc(TEXT), doc(TEXT, id="second")])
    assert all(result.status == "failed" and not result.complete for result in results)
    assert "Anna" not in json.dumps([result.to_dict() for result in results])


def test_keyed_hash_is_scoped_stable_and_does_not_use_predictable_plain_hash():
    engine = processor(pseudonym_key=b"a" * 32)
    requests = [
        doc("Patient: Anna Beispiel.", id=str(i), method="hash", pseudonym_scope=scope)
        for i, scope in enumerate(
            ["tenant-1:study-1", "tenant-1:study-1", "tenant-2:study-1"]
        )
    ]
    a, b, c = engine.process_batch(requests)
    assert a.deidentified_text == b.deidentified_text != c.deidentified_text
    assert re.search(r"\[PERSON_[a-f0-9]{32}\]", a.deidentified_text)
    other = processor(pseudonym_key=b"b" * 32).process_batch([requests[0]])[0]
    assert other.deidentified_text != a.deidentified_text
    assert all(span["action"] == "hash" for span in a.spans)
    assert not any("key" in key or "scope" in key for key in a.provenance)


def test_hash_requires_managed_key_and_explicit_scope():
    with pytest.raises(ValueError, match="pseudonym_scope"):
        ClinicalPrivacyOptions(method="hash")
    with pytest.raises(ValueError, match="secret bytes"):
        processor(pseudonym_key=b"short")
    result = processor().process_batch(
        [doc(TEXT, method="hash", pseudonym_scope="study")]
    )[0]
    assert result.status == "failed" and result.deidentified_text is None


def test_localized_replacement_is_consistent_inside_document():
    text = "Patient: Anna Beispiel. Patient: Anna Beispiel. Keine Dyspnoe."
    result = processor().process_batch([doc(text, method="replace")])[0]
    assert result.complete
    assert "Anna Beispiel" not in result.deidentified_text
    assert "Keine Dyspnoe" in result.deidentified_text
    names = re.findall(r"Patient: (.+?)\.", result.deidentified_text)
    assert len(names) == 2 and names[0] == names[1]
    assert result.language["locale"] == "de_DE"
    assert all(span["action"] == "replace" for span in result.spans)


def test_date_shift_preserves_chronology_and_masks_non_dates():
    text = "Patient: Anna Beispiel, geboren am 14.03.1962. Aufnahme: 07.09.2026. Entlassung: 09.09.2026."
    result = processor().process_batch(
        [doc(text, method="shift_dates", date_shift_days=3)]
    )[0]
    assert result.complete
    assert "Anna Beispiel" not in result.deidentified_text
    assert "17.03.1962" in result.deidentified_text
    assert "10.09.2026" in result.deidentified_text
    assert "12.09.2026" in result.deidentified_text
    assert "date_shift_unresolved" not in result.warnings


def test_role_and_category_controls_survive_final_safety_sweep():
    text = "Patient: Anna Beispiel. Arzt: Dr. Parkinson. Kontakt: anna@example.com."
    result = processor().process_batch([doc(text, redact_roles=("patient",))])[0]
    assert "Anna Beispiel" not in result.deidentified_text
    assert "Dr. Parkinson" in result.deidentified_text
    assert "anna@example.com" not in result.deidentified_text
    assert "policy_narrowed" in result.warnings
    email_only = processor().process_batch([doc(text, redact_categories=("email",))])[0]
    assert "Anna Beispiel" in email_only.deidentified_text
    assert "anna@example.com" not in email_only.deidentified_text
    assert email_only.spans and all(
        span["label"] == "EMAIL" for span in email_only.spans
    )


def test_declared_qualified_route_still_requires_review_for_narrowed_policy():
    result = processor(qualified_languages=["de"]).process_batch([doc(TEXT)])[0]
    assert result.status == "complete"
    result = processor(qualified_languages=["de"]).process_batch(
        [doc(TEXT, redact_categories=("email",))]
    )[0]
    assert result.status == "needs_review"


def test_duplicate_ids_and_request_limits_reject_before_model_work():
    engine = processor(max_documents=1)
    with pytest.raises(ValueError, match="document limit"):
        engine.process_batch([doc(TEXT), doc(TEXT, id="second")])
    assert not engine.model.calls
    with pytest.raises(ValueError, match="IDs"):
        processor().process_batch([doc(TEXT), doc(TEXT)])


def test_explicit_keep_term_cannot_silently_exempt_a_required_person():
    result = processor(qualified_languages=["de"]).process_batch(
        [
            doc(
                "Patient: Anna Parkinson. Morbus Parkinson.", keep_terms=("Parkinson",)
            ),
        ]
    )[0]
    assert "Anna Parkinson" not in result.deidentified_text
    assert "Morbus Parkinson" in result.deidentified_text
    assert result.status == "needs_review"
    assert "keep_term_conflicts_with_identifier" in result.warnings


def test_unreviewed_custom_term_cannot_produce_a_qualified_result():
    result = processor(qualified_languages=["de"]).process_batch(
        [doc("Anna berichtet über Dyspnoe.", keep_terms=("Anna",))]
    )[0]
    assert "Anna" in result.deidentified_text
    assert result.status == "needs_review"
    assert "custom_protection_requires_review" in result.warnings
