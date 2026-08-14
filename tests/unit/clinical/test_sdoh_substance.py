from openmed.clinical.sdoh import (
    available_determinant_extractors,
    extract_sdoh,
)


def test_tobacco_pack_year_smoker_quit_is_past():
    findings = extract_sdoh(
        "30 pack-year smoker, quit 2010.",
        spans=[],
    )

    tobacco = [finding for finding in findings if finding.category == "tobacco"]

    assert len(tobacco) == 1
    assert tobacco[0].status == "past"
    assert tobacco[0].extent == "30 pack-years"
    assert tobacco[0].temporality == "historical"


def test_denies_alcohol_is_none():
    findings = extract_sdoh(
        "Patient denies alcohol.",
        spans=[],
    )

    alcohol = [finding for finding in findings if finding.category == "alcohol"]

    assert len(alcohol) == 1
    assert alcohol[0].status == "none"
    assert alcohol[0].extent is None


def test_occasional_ivdu_is_current():
    findings = extract_sdoh(
        "Occasional IVDU.",
        spans=[],
    )

    drug = [finding for finding in findings if finding.category == "drug"]

    assert len(drug) == 1
    assert drug[0].status == "current"
    assert drug[0].extent == "occasional"


def test_alcohol_drinks_per_week_extent():
    findings = extract_sdoh(
        "Patient reports 7 drinks/week.",
        spans=[],
    )

    alcohol = [finding for finding in findings if finding.category == "alcohol"]

    assert len(alcohol) == 1
    assert alcohol[0].status == "current"
    assert alcohol[0].extent == "7 drinks/week"


def test_negation_does_not_cross_sentence_boundary():
    findings = extract_sdoh(
        "Denies alcohol. Current smoker.",
        spans=[],
    )

    status_by_category = {finding.category: finding.status for finding in findings}

    assert status_by_category["alcohol"] == "none"
    assert status_by_category["tobacco"] == "current"


def test_multiple_tobacco_triggers_in_same_statement_are_deduplicated():
    findings = extract_sdoh(
        "30 pack-year smoker, quit 2010.",
        spans=[],
    )

    tobacco = [finding for finding in findings if finding.category == "tobacco"]

    assert len(tobacco) == 1


def test_separate_tobacco_statements_are_preserved():
    findings = extract_sdoh(
        "Former smoker. Currently vaping.",
        spans=[],
    )

    tobacco = [finding for finding in findings if finding.category == "tobacco"]

    assert len(tobacco) == 2

    statuses = [finding.status for finding in tobacco]

    assert statuses == ["past", "current"]


def test_substance_extractors_are_registered():
    available = available_determinant_extractors()

    assert "tobacco" in available
    assert "alcohol" in available
    assert "drug" in available
