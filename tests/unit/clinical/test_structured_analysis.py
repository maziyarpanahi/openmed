"""Synthetic source/evidence regressions for bounded clinical composition."""

import json

import pytest

from openmed.clinical.analysis import ClinicalAnalysisError, analyze_clinical_context


def analyze(text, mentions, tasks, **kwargs):
    spans = [
        {
            "start": text.index(surface),
            "end": text.index(surface) + len(surface),
            "label": label,
            "score": score,
        }
        for surface, label, score in mentions
    ]
    return analyze_clinical_context(text, spans, tasks=tasks, language="de", **kwargs)


def test_german_cardiology_dose_frequency_lvef_and_blood_pressure():
    text = "Medikation: Metoprolol 47,5 mg zweimal täglich.\nBefund: LVEF 55 %.\nVitalwerte: RR 120/80 mmHg."
    result = analyze(
        text,
        [
            ("Metoprolol", "Drug", 0.9),
            ("47,5 mg", "Dose", 0.9),
            ("zweimal täglich", "Frequency", 0.9),
            ("LVEF", "Lab Test", 0.9),
            ("55 %", "Lab Value", 0.9),
            ("RR 120/80 mmHg", "Vital Sign", 0.9),
        ],
        ["medications", "labs", "vitals", "relations"],
    )
    assert result["complete"] and result["status"] == "needs_review"
    medication = result["tasks"]["medications"]["records"][0]
    attributes = {a["type"]: a for a in medication["attributes"]}
    assert attributes["dose"]["normalized"]["value"] == 47.5
    assert attributes["dose"]["normalized"]["unit"] == "mg"
    assert attributes["frequency"]["normalized"]["frequency_per_day"] == 2
    assert (
        text[
            attributes["dose"]["source"]["start"] : attributes["dose"]["source"]["end"]
        ]
        == "47,5 mg"
    )
    lab = result["tasks"]["labs"]["records"][0]
    assert lab["measurement"]["canonical_magnitude"] == 0.55
    assert lab["measurement"]["canonical_unit"] == "1"
    assert lab["abnormal_flag"] == "unknown"
    assert text[lab["evidence"][0]["start"] : lab["evidence"][0]["end"]] == "55 %"
    vital = result["tasks"]["vitals"]["records"][0]
    assert [c["value"] for c in vital["measurement"]["components"]] == [120, 80]
    relation = result["tasks"]["relations"]["records"][0]
    assert relation["type"] == "DRUG_DOSE"
    assert relation["head"]["id"] == medication["source"]["id"]
    assert not relation["coding_eligible"] and relation["score_kind"] == "heuristic"
    serialized = json.dumps(result, ensure_ascii=False)
    assert all(
        source not in serialized
        for source in ("Metoprolol", "47,5 mg", "zweimal täglich", "LVEF")
    )


@pytest.mark.parametrize("label", ["Dose", "Strength"])
def test_decimal_fragments_rejoin_source_quantity_without_changing_semantic_label(
    label,
):
    text = "Metoprolol 47,5 mg."
    result = analyze(
        text,
        [("Metoprolol", "Drug", 0.99), ("47", label, 0.82), ("5 mg", label, 0.9)],
        ["entities", "medications", "relations"],
    )
    original = result["tasks"]["entities"]["records"]
    assert text[original[1]["start"] : original[1]["end"]] == "47"
    attr = result["tasks"]["medications"]["records"][0]["attributes"][0]
    assert attr["type"] == label.lower()
    assert attr["normalized"]["value"] == 47.5 and attr["normalized"]["unit"] == "mg"
    source = attr["source"]
    assert (
        source["label"] == label
        and source["span_repair"] == "adjacent_decimal_fragments"
    )
    assert source["score"] == 0.82
    assert text[source["start"] : source["end"]] == "47,5 mg"
    assert [p["id"] for p in source["source_parts"]] == [
        original[1]["id"],
        original[2]["id"],
    ]
    relation = result["tasks"]["relations"]["records"][0]
    assert relation["tail"]["end"] == source["end"]
    assert relation["tail_context"]["entity_id"] == source["id"]
    assert relation["type"] == ("DRUG_STRENGTH" if label == "Strength" else "DRUG_DOSE")


@pytest.mark.parametrize(
    "text,left,right",
    [
        ("Metoprolol 47;5 mg.", "47", "5 mg"),
        ("Metoprolol 47, 5 mg.", "47", "5 mg"),
        ("Metoprolol 47,5 mg/kg.", "47", "5 mg"),
        ("Metoprolol 147,5 mg.", "47", "5 mg"),
        ("Metoprolol 47 mg,5 mg.", "47 mg", "5 mg"),
    ],
)
def test_numeric_repair_cannot_join_distinct_quantities_or_partial_units(
    text, left, right
):
    result = analyze(
        text,
        [
            ("Metoprolol", "Drug", 0.99),
            (left, "Strength", 0.82),
            (right, "Strength", 0.9),
        ],
        ["medications"],
    )
    assert all(
        "span_repair" not in attr["source"]
        for attr in result["tasks"]["medications"]["records"][0]["attributes"]
    )


def test_english_decimal_repair_uses_explicit_language():
    text = "metoprolol 47.5 mg"
    spans = [
        {"start": 0, "end": 10, "label": "Drug", "score": 0.99},
        {"start": 11, "end": 13, "label": "Strength", "score": 0.8},
        {"start": 14, "end": 18, "label": "Strength", "score": 0.9},
    ]
    result = analyze_clinical_context(text, spans, language="en", tasks=["medications"])
    attr = result["tasks"]["medications"]["records"][0]["attributes"][0]
    assert attr["normalized"]["value"] == 47.5
    assert attr["source"]["span_repair"] == "adjacent_decimal_fragments"


def test_different_model_labels_cannot_turn_decimal_tail_into_a_dose():
    text = "Metoprolol 47,5 mg"
    result = analyze(
        text,
        [
            ("Metoprolol", "Drug", 0.99),
            ("47", "Strength", 0.67),
            ("5 mg", "Dose", 0.96),
        ],
        ["entities", "medications", "relations"],
    )
    assert len(result["tasks"]["entities"]["records"]) == 3
    attrs = result["tasks"]["medications"]["records"][0]["attributes"]
    assert len(attrs) == 2
    assert all(
        a["normalized"] == {"recognized": False, "reason": "incomplete_quantity_span"}
        for a in attrs
    )
    assert result["tasks"]["relations"]["records"] == []
    written = result["tasks"]["medications"]["records"][0]["written_amounts"][0]
    assert written["normalized"]["value"] == 47.5
    assert written["normalized"]["unit"] == "mg"
    assert written["semantic_type"] == "unspecified" and written["requires_review"]
    assert {part["label"] for part in written["source_parts"]} == {"Dose", "Strength"}
    assert text[written["source"]["start"] : written["source"]["end"]] == "47,5 mg"


@pytest.mark.parametrize(
    "language,text,head,amount",
    [
        (
            "de",
            "Keine Dyspnoe. Patient nimmt Metoprolol 47,5 mg.",
            "Metoprolol 47",
            "47,5 mg",
        ),
        ("en", "Patient takes metoprolol 47.5 mg.", "metoprolol 47", "47.5 mg"),
        ("de", "Medikation: Vitamin B12 1,5 mg.", "Vitamin B12 1", "1,5 mg"),
    ],
)
def test_drug_partial_decimal_boundary_retains_model_evidence(
    language, text, head, amount
):
    start = text.index(head)
    original = {
        "start": start,
        "end": start + len(head),
        "label": "Drug",
        "score": 0.98,
        "text": head,
    }
    result = analyze_clinical_context(
        text,
        [original],
        language=language,
        tasks=["entities", "assertions", "medications"],
    )
    entity = result["tasks"]["entities"]["records"][0]
    assert text[entity["start"] : entity["end"]] == head.rsplit(" ", 1)[0]
    assert entity["source_parts"][0]["end"] == original["end"]
    assert entity["score_kind"] == "model_score_before_boundary_repair"
    assert result["tasks"]["assertions"]["records"][0]["end"] == entity["end"]
    written = result["tasks"]["medications"]["records"][0]["written_amounts"][0]
    assert text[written["source"]["start"] : written["source"]["end"]] == amount
    assert written["semantic_type"] == "unspecified"
    assert original["text"] == head and original["end"] == start + len(head)
    assert head not in json.dumps(result)


@pytest.mark.parametrize(
    "text,head",
    [
        ("Vitamin B12 1 mg.", "Vitamin B12"),
        ("Humalog Mix 25 mg.", "Humalog Mix 25"),
        ("Metoprolol 47,5.", "Metoprolol 47"),
        ("Metoprolol 47,5 mg/kg.", "Metoprolol 47"),
        ("Metoprolol 47,5 unknownunit.", "Metoprolol 47"),
        ("Metoprolol 47, 5 mg.", "Metoprolol 47"),
    ],
)
def test_drug_numeric_suffix_is_not_trimmed_without_full_decimal_quantity(text, head):
    result = analyze(text, [(head, "Drug", 0.99)], ["entities"])
    entity = result["tasks"]["entities"]["records"][0]
    assert text[entity["start"] : entity["end"]] == head and "span_repair" not in entity


@pytest.mark.parametrize(
    "text,head",
    [
        ("Metoprolol 47, 5 mg.", "Metoprolol 47"),
        ("Metoprolol 147 mg.", "Metoprolol 1"),
        ("Metoprolol 5 mg/kg.", "Metoprolol"),
        ("Metoprolol 5 mg / kg.", "Metoprolol"),
        ("Metoprolol.\n5 mg.", "Metoprolol"),
        ("Metoprolol und Ramipril 5 mg.", "Metoprolol"),
    ],
)
def test_written_amount_does_not_recover_partial_or_distant_quantities(text, head):
    result = analyze(text, [(head, "Drug", 0.99)], ["medications"])
    assert result["tasks"]["medications"]["records"][0]["written_amounts"] == []


def test_german_treatment_and_course_sections_end_family_attribution():
    text = "Familienanamnese:\nVater mit Diabetes.\nTherapie:\nMetoprolol begonnen.\nVerlauf:\nKeine Dyspnoe."
    result = analyze(
        text,
        [
            ("Diabetes", "Disease", 0.9),
            ("Metoprolol", "Drug", 0.9),
            ("Dyspnoe", "Symptom", 0.9),
        ],
        ["sections", "entities", "assertions"],
    )
    sections = result["tasks"]["sections"]["records"]
    assert [s["label"] for s in sections] == [
        "family_history",
        "treatment",
        "clinical_course",
    ]
    assert all(not s.get("coding") for s in sections[1:])
    assertions = result["tasks"]["assertions"]["records"]
    assert [a["experiencer"] for a in assertions] == ["family", "patient", "patient"]
    assert assertions[-1]["negation"] == "negated"


def test_quantity_scan_limit_does_not_apply_without_a_drug():
    text = "5 mg " * 2001
    result = analyze_clinical_context(text, [], language="de", tasks=["entities"])
    assert result["complete"] and result["tasks"]["entities"]["records"] == []


def test_quantity_scan_limit_is_explicit_for_medication_work():
    text = "Metoprolol " + "5 mg " * 2001
    with pytest.raises(ClinicalAnalysisError, match="clinical_quantity_limit"):
        analyze(text, [("Metoprolol", "Drug", 0.99)], ["medications"])


def test_unknown_medication_unit_cannot_echo_source_in_normalization_error():
    text = "Metformin 5 privateunitmarker"
    result = analyze(
        text,
        [("Metformin", "Drug", 0.99), ("5 privateunitmarker", "Dose", 0.9)],
        ["medications"],
    )
    attr = result["tasks"]["medications"]["records"][0]["attributes"][0]
    assert attr["normalized"] == {
        "recognized": False,
        "reason": "unrecognized_medication_attribute",
    }
    assert "privateunitmarker" not in json.dumps(result)


@pytest.mark.parametrize(
    "full,partial",
    [("55 %", "5 %"), ("47,5 mg", "5 mg"), ("-5 mg", "5 mg"), ("5 mg/kg", "5 mg")],
)
def test_measurement_numeric_or_unit_fragments_cannot_be_normalized(full, partial):
    text = "Measurement " + full
    start = text.rindex(partial)
    spans = [
        {"start": 0, "end": 11, "label": "Lab Test", "score": 0.9},
        {
            "start": start,
            "end": start + len(partial),
            "label": "Lab Value",
            "score": 0.9,
        },
    ]
    result = analyze_clinical_context(text, spans, language="de", tasks=["labs"])
    record = result["tasks"]["labs"]["records"][0]
    assert record["extraction_status"] == "unparsed"
    assert record["measurement"]["canonical_magnitude"] is None


def test_section_header_model_predictions_are_inspectable_but_not_structured_labs():
    text = "Befund: LVEF 55 %"
    result = analyze(
        text,
        [("Befund", "Lab Test", 0.6), ("LVEF 55 %", "Vital Sign", 0.9)],
        ["entities", "labs", "vitals"],
    )
    assert len(result["tasks"]["entities"]["records"]) == 2
    assert result["tasks"]["labs"]["records"] == []
    assert result["tasks"]["vitals"]["records"][0]["extraction_status"] == "unparsed"


@pytest.mark.parametrize("separator", ["\n", "; ", ". ", " aber "])
def test_medication_dose_cannot_cross_source_scope(separator):
    text = "Metformin" + separator + "500 mg"
    result = analyze(
        text,
        [("Metformin", "Drug", 0.9), ("500 mg", "Dose", 0.9)],
        ["medications", "relations"],
    )
    assert result["tasks"]["medications"]["records"][0]["attributes"] == []
    assert result["tasks"]["relations"]["records"] == []


def test_two_regimens_keep_their_own_dose_and_original_offsets():
    text = "Metformin 500 mg; Insulin 10 units."
    result = analyze(
        text,
        [
            ("Metformin", "Drug", 0.9),
            ("500 mg", "Dose", 0.9),
            ("Insulin", "Drug", 0.9),
            ("10 units", "Dose", 0.9),
        ],
        ["medications"],
    )
    records = result["tasks"]["medications"]["records"]
    assert len(records) == 2
    assert records[0]["attributes"][0]["normalized"]["value"] == 500
    # The SDK treats generic "units" as ambiguous. Keep the second regimen's
    # evidence without presenting its captured number as normalized dosage.
    second = records[1]["attributes"][0]
    assert second["normalized"]["recognized"] is False
    assert "value" not in second["normalized"]
    assert text[second["source"]["start"] : second["source"]["end"]] == "10 units"


def test_negated_and_family_medications_keep_context_without_becoming_current_facts():
    text = "Familienanamnese: Mutter nimmt Metformin 500 mg.\nMedikation: Patient nimmt kein Insulin 10 units."
    result = analyze(
        text,
        [
            ("Metformin", "Drug", 0.9),
            ("500 mg", "Dose", 0.9),
            ("Insulin", "Drug", 0.9),
            ("10 units", "Dose", 0.9),
        ],
        ["medications"],
    )
    family, patient = result["tasks"]["medications"]["records"]
    assert family["context"]["experiencer"] == "family"
    assert patient["context"]["experiencer"] == "patient"
    assert patient["context"]["negation"] == "negated"
    assert all(
        not record["coding_eligible"] and record["candidate_status"] == "unconfirmed"
        for record in (family, patient)
    )


def test_medication_filter_rejects_low_missing_confidence_and_observation_abbreviations():
    text = "Metformin; Insulin; K 4,2 mmol/L."
    result = analyze(
        text,
        [
            ("Metformin", "Drug", 0.7),
            ("Insulin", "Drug", None),
            ("K", "Chemical", 0.99),
        ],
        ["entities", "medications"],
    )
    assert len(result["tasks"]["entities"]["records"]) == 3
    assert result["tasks"]["medications"]["records"] == []
    assert (
        "medication_candidate_threshold_0.75_no_grounding"
        in result["tasks"]["medications"]["warnings"]
    )


@pytest.mark.parametrize(
    "flag,expected,origin",
    [(None, "high", "reference_comparison"), ("N", "normal", "explicit")],
)
def test_lab_decimal_comma_range_units_and_explicit_flag_precedence(
    flag, expected, origin
):
    text = "Glukose 6,2 mmol/L (3,9-5,5 mmol/L)" + (" " + flag if flag else "")
    mentions = [
        ("Glukose", "Lab Test", 0.9),
        ("6,2 mmol/L", "Lab Value", 0.9),
        ("3,9-5,5 mmol/L", "Reference Range", 0.9),
    ]
    if flag:
        mentions.append((flag, "Abnormal Flag", 0.9))
    result = analyze(text, mentions, ["labs"])
    record = result["tasks"]["labs"]["records"][0]
    assert record["extraction_status"] == "parsed"
    assert (
        record["abnormal_flag"] == expected and record["abnormal_flag_source"] == origin
    )
    value, reference = record["measurement"], record["reference_range"]
    assert value["canonical_unit"] == reference["unit"]
    assert reference["low"] < reference["high"] < value["canonical_magnitude"]
    assert len(record["evidence"]) == (3 if flag else 2)


def test_lab_with_missing_range_unit_uses_explicit_result_unit():
    text = "LVEF 55 % (50-70)"
    result = analyze(
        text,
        [
            ("LVEF", "Lab Test", 0.9),
            ("55 %", "Lab Value", 0.9),
            ("50-70", "Reference Range", 0.9),
        ],
        ["labs"],
    )
    record = result["tasks"]["labs"]["records"][0]
    assert record["reference_range"]["low"] == 0.5
    assert record["reference_range"]["high"] == pytest.approx(0.7)
    assert record["abnormal_flag"] == "normal"


def test_lab_unknown_units_and_orphan_values_are_inspectable_without_guessing():
    text = "Glukose 6,2 unknownunit.\n7 mmol/L."
    result = analyze(
        text,
        [
            ("Glukose", "Lab Test", 0.9),
            ("6,2 unknownunit", "Lab Value", 0.9),
            ("7 mmol/L", "Lab Value", 0.9),
        ],
        ["labs"],
    )
    records = result["tasks"]["labs"]["records"]
    assert records[0]["extraction_status"] == "unparsed"
    assert records[0]["measurement"]["canonical_magnitude"] is None
    assert records[1]["extraction_status"] == "unlinked_value"
    assert "unknownunit" not in json.dumps(result)


def test_lab_names_and_values_do_not_link_across_rows():
    text = "Glukose\n6,2 mmol/L"
    result = analyze(
        text, [("Glukose", "Lab Test", 0.9), ("6,2 mmol/L", "Lab Value", 0.9)], ["labs"]
    )
    assert [r["extraction_status"] for r in result["tasks"]["labs"]["records"]] == [
        "unlinked",
        "unlinked_value",
    ]


@pytest.mark.parametrize(
    "surface,parsed",
    [
        ("SpO2 98 %", True),
        ("RR 120/80 mmHg", True),
        ("RR 120/80 mmHg, HF 80/min", False),
        ("HF unbekannt", False),
        ("HF 80/min", True),
    ],
)
def test_vital_candidate_with_multiple_or_missing_values_cannot_silently_use_first(
    surface, parsed
):
    result = analyze(surface, [(surface, "Vital Sign", 0.9)], ["vitals"])
    record = result["tasks"]["vitals"]["records"][0]
    assert (record["extraction_status"] == "parsed") == parsed
    assert (record["measurement"] is not None) == parsed


def test_unsupported_language_and_incomplete_coverage_withhold_structured_records():
    spans = [{"start": 0, "end": 9, "label": "Drug", "score": 0.9}]
    for controls, status in [
        ({"language": "fr"}, "unsupported"),
        ({"language": "de", "entity_coverage_complete": False}, "failed"),
    ]:
        result = analyze_clinical_context(
            "Metformin 500 mg.",
            spans,
            tasks=["sections", "medications", "labs", "vitals", "relations"],
            **controls,
        )
        assert result["status"] == "partial"
        for task in ("medications", "labs", "vitals", "relations"):
            assert result["tasks"][task]["status"] == status
            assert result["tasks"][task]["records"] == []


def test_scope_and_output_limits_withhold_findings_and_preserve_independent_tasks(
    monkeypatch,
):
    import openmed.clinical.structured_analysis as structure

    text = "Metformin 500 mg"
    mentions = [("Metformin", "Drug", 0.9), ("500 mg", "Dose", 0.9)]
    monkeypatch.setattr(structure, "MAX_SCOPE_ENTITIES", 1)
    result = analyze(text, mentions, ["sections", "relations"])
    assert result["status"] == "partial"
    assert result["tasks"]["relations"]["error"] == "clinical_scope_entity_limit"
    monkeypatch.setattr(structure, "MAX_SCOPE_ENTITIES", 64)
    monkeypatch.setattr(structure, "MAX_STRUCTURED_RECORDS", 0)
    result = analyze(text, mentions, ["relations"])
    assert result["tasks"]["relations"]["error"] == "clinical_structured_output_limit"


def test_ambiguous_same_offset_labels_do_not_arbitrarily_choose_a_structured_head():
    result = analyze(
        "Metformin 500 mg",
        [
            ("Metformin", "Drug", 0.9),
            ("Metformin", "Chemical", 0.9),
            ("500 mg", "Dose", 0.9),
        ],
        ["entities", "medications"],
    )
    assert result["status"] == "partial" and result["tasks"]["entities"]["complete"]
    assert (
        result["tasks"]["medications"]["error"] == "clinical_ambiguous_entity_offsets"
    )


def test_structuring_failure_is_source_free_and_isolated_to_requested_task(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("private marker")

    monkeypatch.setattr(
        "openmed.clinical.structured_analysis.structure_vital_sign", broken
    )
    result = analyze(
        "RR 120/80 mmHg",
        [("RR 120/80 mmHg", "Vital Sign", 0.9)],
        ["sections", "vitals", "medications"],
    )
    assert result["status"] == "partial"
    assert result["tasks"]["medications"]["complete"]
    assert result["tasks"]["vitals"]["error"] == "clinical_structuring_failed"
    assert "private marker" not in json.dumps(result)


def test_cancel_during_structuring_cannot_return_earlier_success(monkeypatch):
    import openmed.clinical.structured_analysis as structure

    cancelled = [False]
    original = structure.structure_vital_sign

    def stopping(*args, **kwargs):
        cancelled[0] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(structure, "structure_vital_sign", stopping)
    with pytest.raises(ClinicalAnalysisError, match="clinical_cancelled"):
        analyze(
            "RR 120/80 mmHg",
            [("RR 120/80 mmHg", "Vital Sign", 0.9)],
            ["sections", "vitals"],
            cancel_check=lambda: cancelled[0],
        )
