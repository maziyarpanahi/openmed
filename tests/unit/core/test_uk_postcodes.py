"""Tests for the deterministic en_GB UK postcode rule.

Every postcode here is synthetic: the Z and ZZ areas are not allocated by Royal
Mail. ``GIR 0AA`` is the published non-geographic special code and identifies
no person or residence.
"""

from __future__ import annotations

import pytest

from openmed.core.labels import normalize_label
from openmed.core.pii import deidentify, extract_pii
from openmed.core.pii_entity_merger import PII_PATTERNS, find_semantic_units
from openmed.core.pii_i18n import get_patterns_for_language, validate_uk_postcode
from openmed.core.safety_sweep import safety_sweep
from openmed.processing.outputs import EntityPrediction, PredictionResult

# One synthetic postcode per Royal Mail outward-code shape.
OUTWARD_SHAPES = (
    ("A9", "Z1 1ZZ"),
    ("A99", "Z11 1ZZ"),
    ("A9A", "Z1A 1ZZ"),
    ("AA9", "ZZ1 1ZZ"),
    ("AA99", "ZZ11 1ZZ"),
    ("AA9A", "ZZ1A 1ZZ"),
)


def _uk_postcode_spans(text: str, lang: str = "en", locale: str | None = "en_GB"):
    patterns = get_patterns_for_language(lang, locale=locale)
    return [
        (text[start:end], start, end)
        for start, end, label, *_ in find_semantic_units(text, patterns)
        if label == "postcode"
    ]


def _install_fake_model(monkeypatch, entities=()):
    import openmed

    def fake_analyze_text(inference_text, **kwargs):
        return PredictionResult(
            text=inference_text,
            entities=list(entities),
            model_name=kwargs["model_name"],
            timestamp="2026-01-01T00:00:00",
        )

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze_text)


@pytest.mark.parametrize(("shape", "postcode"), OUTWARD_SHAPES)
def test_validate_uk_postcode_accepts_every_outward_shape(shape, postcode):
    no_space = postcode.replace(" ", "")

    assert validate_uk_postcode(postcode), shape
    assert validate_uk_postcode(no_space), shape
    assert validate_uk_postcode(postcode.lower()), shape
    assert validate_uk_postcode(postcode.replace(" ", "\u00a0")), shape


def test_validate_uk_postcode_accepts_gir_special_case():
    assert validate_uk_postcode("GIR 0AA")
    assert validate_uk_postcode("GIR0AA")
    assert validate_uk_postcode("gir 0aa")


@pytest.mark.parametrize("letter", list("CIKMOV"))
def test_validate_uk_postcode_rejects_excluded_inward_letters(letter):
    assert not validate_uk_postcode(f"ZZ1 1{letter}Z")
    assert not validate_uk_postcode(f"ZZ1 1Z{letter}")


@pytest.mark.parametrize(
    "value",
    (
        "ZZ1",  # outward code alone
        "1ZZ",  # inward code alone
        "QZ1 1ZZ",  # Q never opens an outward code
        "VZ1 1ZZ",  # V never opens an outward code
        "XZ1 1ZZ",  # X never opens an outward code
        "ZI1 1ZZ",  # I never sits second
        "Z1L 1ZZ",  # L cannot close an A9A district
        "ZZ1C 1ZZ",  # C cannot close an AA9A district
        "ZZ1 11Z",  # inward code needs two letters
        "ZZ1  1ZZ",  # one separator at most
        "ZZ1-1ZZ",  # space is the only separator
        "Zz1 1zZ",  # mixed case
        "ZZ1 1Z\u017f",  # non-ASCII letter that upper-cases to S
        "ZZ1 1ZZ ZZ1 1ZZ",  # two postcodes are not one
    ),
)
def test_validate_uk_postcode_rejects_malformed_values(value):
    assert not validate_uk_postcode(value)


def test_extract_pii_en_gb_returns_issue_example_postcode(monkeypatch):
    _install_fake_model(monkeypatch)
    text = (
        "Patient address: 1 Example Street, Exampletown ZZ1 1ZZ. "
        "NHS Number 943 476 5919."
    )

    result = extract_pii(
        text,
        model_name="fixture-pii-model",
        lang="en",
        locale="en_GB",
    )

    start = text.index("ZZ1 1ZZ")
    postcodes = [entity for entity in result.entities if entity.label == "postcode"]
    assert [(e.text, e.start, e.end) for e in postcodes] == [
        ("ZZ1 1ZZ", start, start + len("ZZ1 1ZZ"))
    ]
    assert normalize_label(postcodes[0].label) == "ZIPCODE"
    assert postcodes[0].metadata["detector_sources"] == ["locale_rule"]


def test_extract_pii_en_gb_detects_every_outward_shape_with_gold_offsets(
    monkeypatch,
):
    _install_fake_model(monkeypatch)
    lines = [
        f"Referral {index}: 2 Sample Road, Testville {postcode}."
        for index, (_, postcode) in enumerate(OUTWARD_SHAPES, start=1)
    ]
    lines.append("Correspondence to GIR 0AA.")
    text = "\n".join(lines)
    gold = []
    cursor = 0
    for postcode in [postcode for _, postcode in OUTWARD_SHAPES] + ["GIR 0AA"]:
        start = text.index(postcode, cursor)
        gold.append((postcode, "postcode", start, start + len(postcode)))
        cursor = start + len(postcode)

    result = extract_pii(
        text,
        model_name="fixture-pii-model",
        lang="en",
        locale="en_GB",
    )

    spans = [
        (entity.text, entity.label, entity.start, entity.end)
        for entity in result.entities
        if entity.label == "postcode"
    ]
    assert spans == gold
    assert all(text[start:end] == surface for surface, _, start, end in spans)
    assert {normalize_label(label) for _, label, _, _ in spans} == {"ZIPCODE"}


@pytest.mark.parametrize(
    "postcode",
    ("ZZ1 1ZZ", "ZZ11ZZ", "zz1 1zz", "zz11zz", "ZZ1\u00a01ZZ"),
)
def test_extract_pii_en_gb_keeps_exact_offsets_for_spacing_and_case(
    monkeypatch, postcode
):
    _install_fake_model(monkeypatch)
    text = f"Letter sent to Exampletown {postcode} on request."
    start = text.index(postcode)

    result = extract_pii(
        text,
        model_name="fixture-pii-model",
        lang="en",
        locale="en_GB",
    )

    assert [
        (entity.start, entity.end, text[entity.start : entity.end])
        for entity in result.entities
        if entity.label == "postcode"
    ] == [(start, start + len(postcode), postcode)]


def test_context_words_raise_score_but_are_not_required():
    patterns = get_patterns_for_language("en", locale="en_GB")
    bare_text = "Exampletown ZZ1 1ZZ"
    context_text = "Postcode: ZZ1 1ZZ"

    (bare,) = find_semantic_units(bare_text, patterns)
    (with_context,) = find_semantic_units(context_text, patterns)

    assert bare[2] == with_context[2] == "postcode"
    assert bare[3] == pytest.approx(0.6)
    assert with_context[3] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "text",
    (
        "Postcode ZZ1 1CZ was rejected by the form.",  # invalid inward letter
        "District ZZ1 only, inward code not recorded.",  # outward code alone
        "Phone upgraded to Galaxy S23 8GB today.",  # product storage size
        "Console PS5 1TB was left on the ward.",  # product storage size
        "Ward B4 2nd floor, bay 3.",  # ward label and ordinal
        "WARD B4 2ND FLOOR, BAY 3.",  # upper-case ward label and ordinal
        "Device SKU-ZZ1 1ZZ was replaced.",  # hyphen-joined product code
        "Firmware ZZ1 1ZZ.2 installed.",  # dotted version continuation
        "Pump firmware v2 1RC installed.",  # version string
        "Paracetamol 1 g q4 6hr as required.",  # dosing interval
        "Vitamin B12 5MG daily.",  # drug strength
        "HbA1c 48 mmol/mol; CYP2D6 poor metaboliser; staged T2N1M0.",
        "Grid reference TQ 3080 8060 and SU1234 for the site visit.",
        "Mixed case Zz1 1zz is not a postcode token.",
    ),
)
def test_postcode_shaped_non_postcodes_are_never_returned(text):
    swept = safety_sweep(text, [], lang="en", locale="en_GB")

    assert _uk_postcode_spans(text) == []
    assert [entity for entity in swept if entity.label == "postcode"] == []


def test_lookalike_inward_codes_need_address_context():
    bare = "Phone upgraded to Galaxy S23 8GB today."
    addressed = "Address: 1 Example Street, Exampletown ZZ1 2GB."
    start = addressed.index("ZZ1 2GB")

    assert _uk_postcode_spans(bare) == []
    assert _uk_postcode_spans(addressed) == [("ZZ1 2GB", start, start + 7)]


def test_postcode_next_to_nhs_number_and_nino_stays_separate(monkeypatch):
    _install_fake_model(monkeypatch)
    text = "Exampletown ZZ1 1ZZ 943 476 5919 AB 12 34 56 C"

    result = extract_pii(
        text,
        model_name="fixture-pii-model",
        lang="en",
        locale="en_GB",
    )

    spans = sorted(
        (entity.text, entity.label, entity.start, entity.end)
        for entity in result.entities
    )
    expected = []
    for surface, label in (
        ("ZZ1 1ZZ", "postcode"),
        ("943 476 5919", "national_id"),
        ("AB 12 34 56 C", "national_id"),
    ):
        start = text.index(surface)
        expected.append((surface, label, start, start + len(surface)))
    assert spans == sorted(expected)


@pytest.mark.parametrize(
    "model_surface",
    (
        "1 Example Street, Exampletown ZZ1",  # model stops inside the postcode
        "1 Example Street, Exampletown ZZ1 1ZZ",  # model covers the postcode
        "Exampletown ZZ1 1ZZ, United Kingdom",  # model runs past the postcode
    ),
)
def test_model_address_span_keeps_union_with_postcode(monkeypatch, model_surface):
    text = "Address: 1 Example Street, Exampletown ZZ1 1ZZ, United Kingdom."
    model_start = text.index(model_surface)
    model_end = model_start + len(model_surface)
    _install_fake_model(
        monkeypatch,
        [
            EntityPrediction(
                text=model_surface,
                label="street_address",
                confidence=0.9,
                start=model_start,
                end=model_end,
            )
        ],
    )
    postcode_start = text.index("ZZ1 1ZZ")
    postcode_end = postcode_start + len("ZZ1 1ZZ")

    result = extract_pii(
        text,
        model_name="fixture-pii-model",
        lang="en",
        locale="en_GB",
    )

    covering = [
        entity
        for entity in result.entities
        if entity.start <= postcode_start and entity.end >= postcode_end
    ]
    assert len(covering) == 1
    assert covering[0].start == min(model_start, postcode_start)
    assert covering[0].end == max(model_end, postcode_end)
    assert not any(
        entity.start < postcode_end and entity.end > postcode_start
        for entity in result.entities
        if entity is not covering[0]
    )


def test_deidentify_en_gb_masks_postcode(monkeypatch):
    _install_fake_model(monkeypatch)
    text = "Patient address: 1 Example Street, Exampletown ZZ1 1ZZ."

    result = deidentify(
        text,
        model_name="fixture-pii-model",
        lang="en",
        locale="en_GB",
        method="mask",
    )

    assert "ZZ1 1ZZ" not in result.deidentified_text
    assert "[postcode]" in result.deidentified_text


@pytest.mark.parametrize(
    ("lang", "locale"),
    (
        ("en", "en_GB"),
        ("en", "en-GB"),
        ("en_GB", None),
        ("en", "GB"),
        ("en", "gb"),
        ("en", "uk"),
        ("en", "en_UK"),
    ),
)
def test_uk_locale_aliases_enable_postcode_rule(lang, locale):
    text = "Exampletown ZZ1 1ZZ"

    assert _uk_postcode_spans(text, lang=lang, locale=locale) == [("ZZ1 1ZZ", 12, 19)]


@pytest.mark.parametrize(
    ("lang", "locale"),
    (
        ("en", None),
        ("en", "en_US"),
        ("en", "en_IE"),
        ("en", "en_AU"),
        ("en", "en_CA"),
        ("en", "en_IN"),
        ("en", "en_ZA"),
        ("uk", None),  # Ukrainian, not the United Kingdom
        ("uk", "uk"),
        ("uk", "uk_UA"),
        ("fr", None),
        ("de", None),
        ("nl", None),
        ("es", None),
        ("pt", "pt_BR"),
    ),
)
def test_non_uk_locales_do_not_detect_uk_postcodes(lang, locale):
    text = "Exampletown ZZ1 1ZZ, Z1A 1ZZ, zz11zz"

    assert _uk_postcode_spans(text, lang=lang, locale=locale) == []


def test_safety_sweep_is_scoped_to_en_gb():
    text = "Exampletown ZZ1 1ZZ"

    gb = safety_sweep(text, [], lang="en", locale="en_GB")
    plain = safety_sweep(text, [], lang="en")

    assert [(e.text, e.label, e.start, e.end) for e in gb] == [
        ("ZZ1 1ZZ", "postcode", 12, 19)
    ]
    assert plain == []


def test_us_zip_rule_is_unchanged_under_en_gb():
    text = "Mailing ZIP code 90210-1234 on file."
    en_units = [
        unit[:4] for unit in find_semantic_units(text, get_patterns_for_language("en"))
    ]
    gb_units = [
        unit[:4]
        for unit in find_semantic_units(
            text, get_patterns_for_language("en", locale="en_GB")
        )
    ]

    start = text.index("90210-1234")
    assert en_units == gb_units
    ((zip_start, zip_end, label, score),) = en_units
    assert (zip_start, zip_end, label) == (start, start + len("90210-1234"), "postcode")
    assert score == pytest.approx(0.85)
    assert all(
        pattern.validator is not validate_uk_postcode for pattern in PII_PATTERNS
    )
