"""Surrogates and shifted dates keep the source span's digit system (#3871)."""

from __future__ import annotations

import json
import random
import re
import warnings
from datetime import datetime, timedelta

import pytest

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.digit_systems import (
    ASCII_DIGITS,
    SourceDigitStyle,
    detect_digit_system,
    detect_source_digit_style,
    digit_system_of,
    fold_digits,
    render_digits,
    render_in_source_digits,
)
from openmed.core.locale_formats import parse_date
from openmed.core.pii import PIIEntity, _redact_entity, _shift_date, reidentify
from openmed.core.pii_i18n import (
    validate_aadhaar,
    validate_bangladesh_nid,
    validate_egyptian_national_id,
    validate_iran_national_id,
    validate_thai_national_id,
)
from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult

ZEROS = {
    "arabic_indic": "\N{ARABIC-INDIC DIGIT ZERO}",
    "extended_arabic_indic": "\N{EXTENDED ARABIC-INDIC DIGIT ZERO}",
    "devanagari": "\N{DEVANAGARI DIGIT ZERO}",
    "bengali": "\N{BENGALI DIGIT ZERO}",
    "thai": "\N{THAI DIGIT ZERO}",
    "myanmar": "\N{MYANMAR DIGIT ZERO}",
    "fullwidth": "\N{FULLWIDTH DIGIT ZERO}",
}
ARABIC_DATE_SEPARATOR = "\N{ARABIC DATE SEPARATOR}"
FULLWIDTH_SOLIDUS = "\N{FULLWIDTH SOLIDUS}"
# 15/3/2018 in additive Ethiopic numerals: 10+5 / 3 / 20*100+10+8.
ETHIOPIC_DATE = (
    "\N{ETHIOPIC NUMBER TEN}\N{ETHIOPIC DIGIT FIVE}/\N{ETHIOPIC DIGIT THREE}/"
    "\N{ETHIOPIC NUMBER TWENTY}\N{ETHIOPIC NUMBER HUNDRED}"
    "\N{ETHIOPIC NUMBER TEN}\N{ETHIOPIC DIGIT EIGHT}"
)


def _native(value: str, system: str) -> str:
    zero = ord(ZEROS[system])
    return "".join(chr(zero + int(char)) if char.isdigit() else char for char in value)


def _digits(value: str) -> str:
    return re.sub(r"[^0-9]", "", fold_digits(value))


def _shape(value: str) -> str:
    return re.sub(r"[0-9]", "#", fold_digits(value))


def _systems(value: str) -> set[str]:
    return {digit_system_of(char).name for char in value if char.isdecimal()}


def _thai_national_id(body: str) -> str:
    total = sum(weight * int(digit) for weight, digit in zip(range(13, 1, -1), body))
    return f"{body}{(11 - total % 11) % 10}"


def _anonymizer(lang: str, seed: int = 11) -> Anonymizer:
    return Anonymizer(lang=lang, consistent=True, seed=seed)


def _surrogate(lang: str, source: str, label: str, seed: int = 11) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _anonymizer(lang, seed).surrogate(source, label)


def _format_preserving(lang: str, source: str, label: str, seed: int = 11) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surrogate = _anonymizer(lang, seed).format_preserving_surrogate(source, label)
    assert surrogate is not None
    return surrogate


def _date_entity(text: str, label: str = "DATE") -> PIIEntity:
    return PIIEntity(
        text=text,
        label=label,
        start=0,
        end=len(text),
        confidence=0.99,
        entity_type=label,
        original_text=text,
        canonical_label=label,
    )


def _shift(text: str, days: int, lang: str, *, keep_year: bool = False) -> str:
    return _redact_entity(
        _date_entity(text),
        "shift_dates",
        keep_year=keep_year,
        date_shift_days=days,
        lang=lang,
    )


def _run(text: str, lang: str, spans: list[tuple[str, str]], **run_kwargs):
    def model_detector(model_text: str, **kwargs):
        entities = []
        for surface, label in spans:
            start = model_text.index(surface)
            entities.append(
                EntityPrediction(
                    text=surface,
                    label=label,
                    start=start,
                    end=start + len(surface),
                    confidence=0.99,
                )
            )
        return PredictionResult(
            text=model_text,
            entities=entities,
            model_name="stub",
            timestamp=datetime.now().isoformat(),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Pipeline(
            model_detector=model_detector,
            lang=lang,
            use_safety_sweep=False,
        ).run(text, **run_kwargs)


# ---------------------------------------------------------------------------
# Digit-system helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "zero"), sorted(ZEROS.items()))
def test_named_digit_systems_fold_and_render_round_trip(name, zero):
    native = _native("0123456789", name)

    system = detect_digit_system(native)

    assert system is not None
    assert (system.name, system.zero) == (name, zero)
    assert fold_digits(native) == "0123456789"
    assert render_digits("0123456789", system) == native
    assert render_digits(native, ASCII_DIGITS) == "0123456789"


@pytest.mark.parametrize(
    "source",
    ["555-123-4567", "+1 (415) 555-1234", "01/15/2020", "AB-1234", "John Smith", ""],
)
def test_ascii_and_digit_free_spans_have_no_source_style(source):
    calls = []

    assert detect_source_digit_style(source) is None
    rendered = render_in_source_digits(
        source,
        lambda value: calls.append(value) or "x",
    )
    assert rendered == "x"
    assert calls == [source]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Most digits wins.
        (
            _native("0912 345 ", "extended_arabic_indic") + "6789",
            "extended_arabic_indic",
        ),
        ("0912 345 " + _native("6789", "extended_arabic_indic"), "ascii"),
        # Ties go to the system whose first digit comes earliest.
        (_native("0912 ", "extended_arabic_indic") + "3456", "extended_arabic_indic"),
        ("0912 " + _native("3456", "extended_arabic_indic"), "ascii"),
        (
            _native("0912 ", "extended_arabic_indic") + _native("3456", "arabic_indic"),
            "extended_arabic_indic",
        ),
        (
            _native("0912 ", "arabic_indic") + _native("3456", "extended_arabic_indic"),
            "arabic_indic",
        ),
    ],
)
def test_mixed_digit_spans_use_the_documented_dominant_system(source, expected):
    style = detect_source_digit_style(source)

    assert style is not None
    assert style.system.name == expected
    surrogate = _format_preserving("fa", source, "PHONE")
    assert _systems(surrogate) == {expected}
    assert _shape(surrogate) == _shape(source)


def test_ethiopic_numerals_are_left_unchanged():
    calls = []

    assert digit_system_of("\N{ETHIOPIC DIGIT ONE}") is None
    assert detect_digit_system(ETHIOPIC_DATE) is None
    assert fold_digits(ETHIOPIC_DATE) == ETHIOPIC_DATE
    assert detect_source_digit_style(ETHIOPIC_DATE) is None
    passthrough = render_in_source_digits(
        ETHIOPIC_DATE,
        lambda value: calls.append(value) or value,
    )
    assert passthrough == ETHIOPIC_DATE
    assert calls == [ETHIOPIC_DATE]
    # The date shifter sees the span exactly as before this change.
    assert _shift(ETHIOPIC_DATE, 30, "am") == _shift_date(ETHIOPIC_DATE, 30, lang="am")


def test_native_date_separator_is_folded_and_restored():
    sep = ARABIC_DATE_SEPARATOR
    source = _native(f"15{sep}03{sep}2026", "arabic_indic")

    style = detect_source_digit_style(source)

    assert style is not None
    assert style.fold(source) == "15/03/2026"
    assert style.render("18/03/2026") == _native(f"18{sep}03{sep}2026", "arabic_indic")


def test_native_separator_is_kept_when_its_ascii_form_also_occurs():
    sep = ARABIC_DATE_SEPARATOR
    source = _native(f"15{sep}03/2026", "arabic_indic")

    style = detect_source_digit_style(source)

    assert style is not None
    assert style.separators == ()
    assert style.fold(source) == f"15{sep}03/2026"


def test_bare_digit_surrogate_is_regrouped_like_the_source():
    style = detect_source_digit_style(_native("5982 5979 1903", "devanagari"))

    assert style is not None
    assert style.render("379496084197") == _native("3794 9608 4197", "devanagari")
    # Different digit counts and already-shaped values are not regrouped.
    assert style.render("37949608419") == _native("37949608419", "devanagari")
    assert style.render("+91 98765 43210") == _native("+91 98765 43210", "devanagari")


def test_source_digit_style_retains_no_source_digits():
    source = _native("0912 345 6789", "extended_arabic_indic")

    style = detect_source_digit_style(source)

    assert isinstance(style, SourceDigitStyle)
    description = repr(style)
    assert source not in description
    assert fold_digits(source) not in description
    assert "0912" not in description and "6789" not in description


# ---------------------------------------------------------------------------
# Surrogates
# ---------------------------------------------------------------------------


NATIONAL_ID_CASES = [
    pytest.param(
        "fa",
        "0012345679",
        "extended_arabic_indic",
        validate_iran_national_id,
        id="iran-persian",
    ),
    pytest.param(
        "ar",
        "32306263173021",
        "arabic_indic",
        validate_egyptian_national_id,
        id="egypt-arabic-indic",
    ),
    pytest.param(
        "bn",
        "55807302157368193",
        "bengali",
        validate_bangladesh_nid,
        id="bangladesh-bengali",
    ),
    pytest.param(
        "th",
        _thai_national_id("110170020345"),
        "thai",
        validate_thai_national_id,
        id="thailand-thai",
    ),
    pytest.param(
        "mr",
        "5982 5979 1903",
        "devanagari",
        validate_aadhaar,
        id="aadhaar-devanagari",
    ),
]


@pytest.mark.parametrize(
    ("lang", "ascii_source", "system", "validator"), NATIONAL_ID_CASES
)
def test_native_digit_ids_get_native_surrogates_that_validate(
    lang, ascii_source, system, validator
):
    assert validator(ascii_source)
    source = _native(ascii_source, system)

    surrogate = _surrogate(lang, source, "national_id")
    ascii_surrogate = _surrogate(lang, ascii_source, "national_id")

    assert _systems(surrogate) == {system}
    assert validator(fold_digits(surrogate))
    assert validator(ascii_surrogate)
    # The same identifier in either digit system maps to one surrogate.
    assert _digits(surrogate) == _digits(ascii_surrogate)
    assert _digits(surrogate) != _digits(source)


def test_grouped_devanagari_aadhaar_keeps_its_grouping():
    source = _native("5982 5979 1903", "devanagari")

    surrogate = _surrogate("mr", source, "aadhaar")

    assert _shape(surrogate) == "#### #### ####"
    assert _systems(surrogate) == {"devanagari"}
    assert validate_aadhaar(fold_digits(surrogate))


@pytest.mark.parametrize(
    ("lang", "ascii_phone", "system"),
    [
        ("fa", "0912 345 6789", "extended_arabic_indic"),
        ("ur", "0300 1234567", "extended_arabic_indic"),
        ("ar", "050 123 4567", "arabic_indic"),
        ("hi", "98765 43210", "devanagari"),
        ("bn", "01712-345678", "bengali"),
        ("th", "081-234-5678", "thai"),
        ("my", "09-123456789", "myanmar"),
        ("ja", "090-1234-5678", "fullwidth"),
    ],
)
def test_native_digit_phones_render_in_the_source_system(lang, ascii_phone, system):
    source = _native(ascii_phone, system)

    replaced = _surrogate(lang, source, "phone")
    preserved = _format_preserving(lang, source, "phone")

    for surrogate in (replaced, preserved):
        assert _systems(surrogate) == {system}
        assert not re.search(r"[0-9]", surrogate)
        assert _digits(surrogate) != _digits(source)
    assert _shape(preserved) == _shape(source)
    assert _digits(replaced) == _digits(_surrogate(lang, ascii_phone, "phone"))


@pytest.mark.parametrize(
    ("lang", "ascii_date", "system"),
    [
        ("ar", "15/03/2026", "arabic_indic"),
        ("hi", "15-03-2026", "devanagari"),
        ("bn", "15/03/2026", "bengali"),
        ("fa", "2026/03/15", "extended_arabic_indic"),
    ],
)
def test_native_digit_date_surrogates_parse_after_folding(lang, ascii_date, system):
    source = _native(ascii_date, system)

    for surrogate in (
        _surrogate(lang, source, "date"),
        _format_preserving(lang, source, "date"),
    ):
        assert _systems(surrogate) == {system}
        assert _shape(surrogate) == _shape(source)
        assert parse_date(fold_digits(surrogate), lang).normalized is not None
        assert _digits(surrogate) != _digits(source)


@pytest.mark.parametrize(
    ("lang", "source", "label", "replaced", "preserved"),
    [
        ("en", "555-123-4567", "PHONE", "698-574-7393", "698-574-7393"),
        ("en", "01/15/1970", "DATE", "11/09/1941", "11/09/1941"),
        ("fa", "0912 345 6789", "PHONE", "2791 654 4694", "2791 654 4694"),
        ("th", "081-234-5678", "PHONE", "848-446-7503", "848-446-7503"),
    ],
)
def test_ascii_inputs_keep_their_existing_surrogates(
    lang, source, label, replaced, preserved
):
    assert detect_source_digit_style(source) is None
    assert _surrogate(lang, source, label, seed=7) == replaced
    assert _format_preserving(lang, source, label, seed=7) == preserved


@pytest.mark.parametrize("system", sorted(ZEROS))
def test_native_surrogates_never_reuse_source_digits(system):
    rng = random.Random(3871)
    anonymizer = _anonymizer("en", seed=5)

    for _ in range(40):
        digits = "".join(rng.choice("0123456789") for _ in range(10))
        source = _native(f"{digits[:3]}-{digits[3:]}", system)

        preserved_id = anonymizer.format_preserving_surrogate(source, "ID_NUM")
        preserved_phone = anonymizer.format_preserving_surrogate(source, "PHONE")
        replaced_phone = anonymizer.surrogate(source, "PHONE")

        for surrogate in (preserved_id, preserved_phone, replaced_phone):
            assert _systems(surrogate) == {system}
            assert _digits(surrogate) != digits
        # Format preservation shares only the shape: every digit changes.
        assert _shape(preserved_id) == _shape(source)
        assert all(
            new != old for new, old in zip(_digits(preserved_id), digits, strict=True)
        )


# ---------------------------------------------------------------------------
# Date shifting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lang", "ascii_date", "system", "expected"),
    [
        ("ar", "15/03/2026", "arabic_indic", "18/03/2026"),
        ("fa", "2026/03/15", "extended_arabic_indic", "2026/03/18"),
        ("hi", "15-03-2026", "devanagari", "18-03-2026"),
        ("bn", "15.03.2026", "bengali", "18.03.2026"),
        ("th", "15/03/2026", "thai", "18/03/2026"),
        ("my", "15/03/2026", "myanmar", "18/03/2026"),
        ("ar", "15 مارس 2026", "arabic_indic", "18 مارس 2026"),
    ],
)
def test_shift_dates_keeps_digit_system_and_separator(
    lang, ascii_date, system, expected
):
    source = _native(ascii_date, system)

    shifted = _shift(source, 3, lang)

    assert shifted == _native(expected, system)
    assert _shift(ascii_date, 3, lang) == expected


@pytest.mark.parametrize(
    ("lang", "source", "expected"),
    [
        (
            "ar",
            _native("15{0}03{0}2026".format(ARABIC_DATE_SEPARATOR), "arabic_indic"),
            _native("18{0}03{0}2026".format(ARABIC_DATE_SEPARATOR), "arabic_indic"),
        ),
        (
            "ja",
            _native("2026{0}03{0}15".format(FULLWIDTH_SOLIDUS), "fullwidth"),
            _native("2026{0}03{0}18".format(FULLWIDTH_SOLIDUS), "fullwidth"),
        ),
    ],
)
def test_shift_dates_restores_native_date_separators(lang, source, expected):
    assert _shift(source, 3, lang) == expected


def test_shift_dates_keep_year_keeps_native_year_digits():
    source = _native("30/12/2025", "arabic_indic")

    assert _shift(source, 5, "ar", keep_year=True) == _native(
        "04/01/2025", "arabic_indic"
    )


def test_shift_dates_ascii_inputs_are_unchanged():
    assert _shift("01/15/2020", 30, "en") == "02/14/2020"
    assert _shift("15.01.2020", 30, "de") == "14.02.2020"
    assert _shift("2026-03-15", 30, "en") == "2026-04-14"


@pytest.mark.parametrize("system", sorted(ZEROS))
def test_shifted_native_dates_keep_intervals_over_a_synthetic_sample(system):
    rng = random.Random(1871)
    base = datetime(1990, 1, 1)

    for _ in range(30):
        original = base + timedelta(days=rng.randint(0, 12_000))
        offset = rng.choice([-1, 1]) * rng.randint(1, 365)
        source = _native(original.strftime("%Y-%m-%d"), system)

        shifted = _shift(source, offset, "en")

        assert _systems(shifted) == {system}
        assert fold_digits(shifted) == (original + timedelta(days=offset)).strftime(
            "%Y-%m-%d"
        )
        assert shifted != source


# ---------------------------------------------------------------------------
# Pipeline, mapping and audit artifacts
# ---------------------------------------------------------------------------


PERSIAN_NOTE_SPANS = [
    (_native("0012345679", "extended_arabic_indic"), "ID_NUM"),
    (_native("0912 345 6789", "extended_arabic_indic"), "PHONE"),
    (_native("2026/03/15", "extended_arabic_indic"), "DATE"),
]
PERSIAN_NOTE = "کد ملی {} تلفن {} تاریخ {}".format(
    *(surface for surface, _ in PERSIAN_NOTE_SPANS)
)


@pytest.mark.parametrize("method", ["replace", "format_preserve"])
def test_pipeline_native_digit_note_has_no_ascii_surrogate_digits(method):
    result = _run(PERSIAN_NOTE, "fa", PERSIAN_NOTE_SPANS, method=method, seed=7)

    entities = result.deidentification_result.pii_entities
    assert not re.search(r"[0-9]", result.redacted_text)
    assert [entity.action for entity in entities] == [method] * 3
    for entity in entities:
        assert _systems(entity.redacted_text) == {"extended_arabic_indic"}
        assert entity.redacted_text not in PERSIAN_NOTE


def test_pipeline_shift_dates_native_digit_note():
    text = "تاريخ الدخول {} م".format(_native("15/03/2026", "arabic_indic"))
    surface = _native("15/03/2026", "arabic_indic")

    result = _run(
        text,
        "ar",
        [(surface, "DATE")],
        method="shift_dates",
        date_shift_days=3,
    )

    assert result.redacted_text == "تاريخ الدخول {} م".format(
        _native("18/03/2026", "arabic_indic")
    )


def test_native_digit_mapping_and_audit_store_nothing_new_in_plaintext():
    result = _run(
        PERSIAN_NOTE,
        "fa",
        PERSIAN_NOTE_SPANS,
        method="replace",
        seed=7,
        keep_mapping=True,
        audit=True,
    )
    deidentified = result.deidentification_result
    surrogates = {entity.redacted_text for entity in deidentified.pii_entities}
    originals = {surface for surface, _ in PERSIAN_NOTE_SPANS}

    # The reversible mapping holds exactly surrogate -> source, as for ASCII.
    assert set(deidentified.mapping) == surrogates
    assert set(deidentified.mapping.values()) == originals
    assert reidentify(result.redacted_text, deidentified.mapping) == PERSIAN_NOTE

    serialized_mapping = json.dumps(deidentified.mapping, ensure_ascii=False)
    serialized_audit = json.dumps(
        [result.audit_record, deidentified.audit_report.to_dict()],
        ensure_ascii=False,
        default=str,
    )
    for original in originals:
        assert fold_digits(original) not in serialized_mapping
        assert original not in serialized_audit
        assert fold_digits(original) not in serialized_audit
        assert _digits(original) not in serialized_audit
