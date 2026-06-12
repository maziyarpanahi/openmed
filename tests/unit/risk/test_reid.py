"""Tests for residual re-identification risk reports."""

from openmed.risk import risk_report


def _span(text, label, value, *, section="assessment"):
    start = text.index(value)
    return {
        "label": label,
        "start": start,
        "end": start + len(value),
        "metadata": {"section": section},
    }


def test_risk_report_exports_documented_shape_and_uses_span_offsets():
    text = (
        "Assessment: 94-year-old seen at North Clinic on 2024-02-03 "
        "with [RARE_CONDITION]."
    )
    report = risk_report(
        {
            "doc_id": "note-1",
            "text": text,
            "entities": [
                _span(text, "AGE", "94-year-old"),
                _span(text, "ORGANIZATION", "North Clinic"),
                _span(text, "DATE", "2024-02-03"),
                _span(text, "RARE_CONDITION", "[RARE_CONDITION]"),
            ],
        }
    )

    assert set(report) == {
        "leakage_rate",
        "reid_rate",
        "k_min",
        "singleton_records",
        "quasi_identifiers",
    }
    assert report["k_min"] == 1
    assert report["singleton_records"][0]["record_id"] == "note-1"

    categories = {qi["category"] for qi in report["quasi_identifiers"]}
    assert categories == {"age", "provider_institution", "date", "rare_condition"}
    age_qi = next(qi for qi in report["quasi_identifiers"] if qi["category"] == "age")
    assert age_qi["value"] == "94-year-old"
    assert age_qi["start"] == text.index("94-year-old")
    assert age_qi["section"] == "assessment"


def test_table_equivalence_classes_flag_singleton_and_compute_k_min():
    deidentified = [
        {
            "record_id": "a",
            "age": 73,
            "city": "Riverton",
            "visit_date": "2024-01-05",
        },
        {
            "record_id": "b",
            "age": 73,
            "city": "Riverton",
            "visit_date": "2024-01-05",
        },
        {
            "record_id": "unique",
            "age": 94,
            "city": "Smallville",
            "visit_date": "2024-01-05",
        },
    ]

    report = risk_report(deidentified)

    assert report["k_min"] == 1
    assert [record["record_id"] for record in report["singleton_records"]] == ["unique"]

    grouped_report = risk_report(deidentified[:2])
    assert grouped_report["k_min"] == 2
    assert grouped_report["singleton_records"] == []


def test_aux_linkage_attack_scores_nonzero_and_anonymized_records_score_zero():
    deidentified = [
        {
            "record_id": "target",
            "age": 94,
            "city": "Smallville",
            "visit_date": "2024-01-05",
            "diagnosis": "rare vasculitis",
        },
        {
            "record_id": "peer",
            "age": 73,
            "city": "Riverton",
            "visit_date": "2024-01-05",
            "diagnosis": "hypertension",
        },
    ]
    aux = [
        {
            "age": 94,
            "city": "Smallville",
            "visit_date": "2024-01-05",
            "diagnosis": "rare vasculitis",
        }
    ]

    report = risk_report(deidentified, aux=aux)

    assert report["reid_rate"] > 0

    anonymized_report = risk_report(
        [
            {"record_id": "target", "text": "Patient [AGE] from [LOCATION]."},
            {"record_id": "peer", "text": "Patient [AGE] from [LOCATION]."},
        ],
        aux=aux,
    )
    assert anonymized_report["reid_rate"] == 0


def test_leakage_rate_counts_residual_direct_identifier_from_original():
    original = {
        "record_id": "note-1",
        "text": "Patient Alice Morgan visited North Clinic.",
        "entities": [
            {
                "label": "PERSON",
                "text": "Alice Morgan",
                "start": 8,
                "end": 20,
            }
        ],
    }
    deidentified = {
        "record_id": "note-1",
        "text": "Patient Alice Morgan visited [CLINIC].",
    }

    report = risk_report(deidentified, original=original)

    assert report["leakage_rate"] == 1.0
