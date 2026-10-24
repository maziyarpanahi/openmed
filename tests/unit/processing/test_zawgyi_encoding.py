"""Acceptance gates for Zawgyi Burmese detection and offset-mapped conversion.

Every fixture is synthetic: short generic Burmese phrases and honorific names
written once in Zawgyi visual order and once in Unicode storage order.
Strings are spelled as code points so each Zawgyi/Unicode difference is
explicit and reviewable.
"""

from __future__ import annotations

import json
import logging
import unicodedata

import pytest

from openmed.core.errors import InputError
from openmed.core.pipeline import Pipeline
from openmed.core.script_detect import normalize_for_pii_detection
from openmed.processing.legacy_encoding import (
    LEGACY_ENCODING_UNCONVERTED,
    LEGACY_ENCODING_UNDETERMINED,
    ZAWGYI_CONFIDENCE_THRESHOLD,
    ZAWGYI_RULE_PROVENANCE,
    convert_legacy_encoding,
    detect_legacy_encoding,
    detect_zawgyi,
    iscii_to_unicode,
    zawgyi_to_unicode,
)
from openmed.processing.zawgyi import convert_zawgyi_with_origins


def _cp(*codepoints: int) -> str:
    return "".join(chr(codepoint) for codepoint in codepoints)


# "Patient Maung Maung arrived at the hospital." Exercises prefix vowel E,
# U+1039 as the killer, short RA (U+1090) and right-shifted dot (U+1094).
_ZG_SENTENCE = _cp(
    *(0x101C, 0x1030, 0x1014, 0x102C, 0x20),
    *(0x1031, 0x1019, 0x102C, 0x1004, 0x1039, 0x1031, 0x1019, 0x102C, 0x1004, 0x1039),
    *(0x20, 0x101E, 0x100A, 0x1039, 0x20),
    *(0x1031, 0x1006, 0x1038, 0x1090, 0x102F, 0x1036, 0x101E, 0x102D, 0x102F, 0x1094),
    *(0x20, 0x1031, 0x101B, 0x102C, 0x1000, 0x1039, 0x101B, 0x103D, 0x102D),
    *(0x101E, 0x100A, 0x1039, 0x104B),
)
_UNI_SENTENCE = _cp(
    *(0x101C, 0x1030, 0x1014, 0x102C, 0x20),
    *(0x1019, 0x1031, 0x102C, 0x1004, 0x103A, 0x1019, 0x1031, 0x102C, 0x1004, 0x103A),
    *(0x20, 0x101E, 0x100A, 0x103A, 0x20),
    *(0x1006, 0x1031, 0x1038, 0x101B, 0x102F, 0x1036, 0x101E, 0x102D, 0x102F, 0x1037),
    *(0x20, 0x101B, 0x1031, 0x102C, 0x1000, 0x103A, 0x101B, 0x103E, 0x102D),
    *(0x101E, 0x100A, 0x103A, 0x104B),
)

# "U Aung Myint and Daw Khin May". Exercises U+1025 U+102E composition,
# pre-base medial RA (U+103B), short NA (U+108F) and U+105A (tall AA + asat).
_ZG_NAMES = _cp(
    *(0x1025, 0x102E, 0x1038, 0x1031, 0x1021, 0x102C, 0x1004, 0x1039),
    *(0x103B, 0x1019, 0x1004, 0x1037, 0x1039, 0x20),
    *(0x108F, 0x103D, 0x1004, 0x1037, 0x1039, 0x20),
    *(0x1031, 0x1012, 0x105A, 0x1001, 0x1004, 0x1039, 0x1031, 0x1019),
)
_UNI_NAMES = _cp(
    *(0x1026, 0x1038, 0x1021, 0x1031, 0x102C, 0x1004, 0x103A),
    *(0x1019, 0x103C, 0x1004, 0x1037, 0x103A, 0x20),
    *(0x1014, 0x103E, 0x1004, 0x1037, 0x103A, 0x20),
    *(0x1012, 0x1031, 0x102B, 0x103A, 0x1001, 0x1004, 0x103A, 0x1019, 0x1031),
)

# Place names and common words. Exercises stacked consonants (U+1072),
# kinzi (U+1064), medial YA (U+103A), wide medial RA (U+107E) and dot below
# before the killer.
_ZG_PLACES = _cp(
    *(0x103B, 0x1019, 0x1014, 0x1039, 0x1019, 0x102C, 0x20),
    *(0x1019, 0x108F, 0x1072, 0x1031, 0x101C, 0x1038, 0x20),
    *(0x1021, 0x1002, 0x1064, 0x101C, 0x102D, 0x1015, 0x1039, 0x20),
    *(0x1031, 0x1000, 0x103A, 0x102C, 0x1004, 0x1039, 0x1038, 0x20),
    *(0x1031, 0x103B, 0x1015, 0x102C, 0x20),
    *(0x107E, 0x1000, 0x100A, 0x1037, 0x1039),
)
_UNI_PLACES = _cp(
    *(0x1019, 0x103C, 0x1014, 0x103A, 0x1019, 0x102C, 0x20),
    *(0x1019, 0x1014, 0x1039, 0x1010, 0x101C, 0x1031, 0x1038, 0x20),
    *(0x1021, 0x1004, 0x103A, 0x1039, 0x1002, 0x101C, 0x102D, 0x1015, 0x103A, 0x20),
    *(0x1000, 0x103B, 0x1031, 0x102C, 0x1004, 0x103A, 0x1038, 0x20),
    *(0x1015, 0x103C, 0x1031, 0x102C, 0x20),
    *(0x1000, 0x103C, 0x100A, 0x1037, 0x103A),
)

_PAIRS = [
    pytest.param(_ZG_SENTENCE, _UNI_SENTENCE, id="sentence"),
    pytest.param(_ZG_NAMES, _UNI_NAMES, id="names"),
    pytest.param(_ZG_PLACES, _UNI_PLACES, id="places"),
]

# Synthetic "Maung Maung" in Zawgyi and Unicode.
_ZG_NAME = _cp(
    0x1031, 0x1019, 0x102C, 0x1004, 0x1039, 0x1031, 0x1019, 0x102C, 0x1004, 0x1039
)
_UNI_NAME = _cp(
    0x1019, 0x1031, 0x102C, 0x1004, 0x103A, 0x1019, 0x1031, 0x102C, 0x1004, 0x103A
)
_NOTE = f"Patient: {_ZG_NAME}, MRN 12345"

# Synthetic Shan Unicode text: Shan letters, vowels and tones in U+1075-U+108F
# overlap Zawgyi presentation forms but carry no Zawgyi structure.
_SHAN_UNICODE = _cp(
    *(0x1075, 0x1083, 0x1087, 0x20, 0x1019, 0x1082, 0x103A, 0x1087, 0x20),
    *(0x101E, 0x102F, 0x1004, 0x103A, 0x20, 0x1076, 0x1083, 0x1088),
)
# Synthetic ISCII-1991 Devanagari fixture shared with the ISCII tests.
_ISCII_NAME = bytes.fromhex("cf cc e1 d5 20 d5 cf e8 cc da")


def test_rule_data_carries_license_and_provenance_record():
    assert ZAWGYI_RULE_PROVENANCE["license"] == "Apache-2.0"
    assert ZAWGYI_RULE_PROVENANCE["third_party_code"] == "none"
    assert "Unicode Technical Note #11" in ZAWGYI_RULE_PROVENANCE["provenance"]
    assert ZAWGYI_RULE_PROVENANCE["version"]


@pytest.mark.parametrize(("zawgyi", "unicode"), _PAIRS)
def test_synthetic_zawgyi_is_detected_and_converted(zawgyi, unicode):
    detection = detect_zawgyi(zawgyi)

    assert detection.decision == "zawgyi"
    assert detection.confidence >= ZAWGYI_CONFIDENCE_THRESHOLD
    assert detect_legacy_encoding(zawgyi) == "zawgyi"
    assert detect_legacy_encoding(zawgyi.encode("utf-8")) == "zawgyi"

    converted = convert_legacy_encoding(zawgyi)

    assert converted.text == unicode
    assert converted.encoding == "zawgyi"
    assert converted.changed
    assert converted.reason is None
    assert converted.zawgyi == detection
    assert unicodedata.normalize("NFC", converted.text) == converted.text
    assert converted.to_original_span(0, len(converted.text)) == (0, len(zawgyi))


@pytest.mark.parametrize(("zawgyi", "unicode"), _PAIRS)
def test_synthetic_unicode_burmese_is_never_converted(zawgyi, unicode):
    detection = detect_zawgyi(unicode)

    assert detection.decision == "unicode"
    assert detection.zawgyi_signals == 0
    assert detect_legacy_encoding(unicode) == "unicode"
    for source in (unicode, unicode.encode("utf-8")):
        converted = convert_legacy_encoding(source)
        assert converted.text == unicode
        assert converted.encoding == "unicode"
        assert not converted.changed
        assert converted.reason is None
        assert converted.zawgyi is not None
        assert converted.zawgyi.decision == "unicode"


def test_converted_name_span_maps_to_exact_source_code_points():
    converted = convert_legacy_encoding(_NOTE)

    assert converted.text == f"Patient: {_UNI_NAME}, MRN 12345"
    assert converted.offset_unit == "code_point"
    start = converted.text.index(_UNI_NAME)
    source_start, source_end = converted.to_original_span(start, start + len(_UNI_NAME))
    assert _NOTE[source_start:source_end] == _ZG_NAME
    assert converted.offset_map.to_converted_span(source_start, source_end) == (
        start,
        start + len(_UNI_NAME),
    )


def test_converted_name_span_maps_to_exact_utf8_byte_offsets():
    source = _NOTE.encode("utf-8")
    converted = convert_legacy_encoding(source)

    assert converted.encoding == "zawgyi"
    assert converted.offset_unit == "byte"
    assert converted.original == source
    start = converted.text.index(_UNI_NAME)
    source_start, source_end = converted.to_original_span(start, start + len(_UNI_NAME))
    assert source[source_start:source_end].decode("utf-8") == _ZG_NAME
    mrn = converted.text.index("MRN 12345")
    assert converted.to_original_span(mrn, mrn + 9) == (
        source.index(b"MRN 12345"),
        len(source),
    )


@pytest.mark.parametrize(("zawgyi", "unicode"), _PAIRS)
def test_offsets_are_monotone_and_cover_their_source_characters(zawgyi, unicode):
    converted, origins = convert_zawgyi_with_origins(zawgyi)

    assert converted == unicode
    assert len(origins) == len(converted)
    for (start, end), (next_start, next_end) in zip(origins, origins[1:]):
        assert start <= next_start
        assert end <= next_end
    # Every source code point is covered by at least one output character.
    covered = {index for start, end in origins for index in range(start, end)}
    assert covered == set(range(len(zawgyi)))
    # Word-level spans map back to the exact source word.
    unicode_words = unicode.split(" ")
    zawgyi_words = zawgyi.split(" ")
    cursor = 0
    for unicode_word, zawgyi_word in zip(unicode_words, zawgyi_words, strict=True):
        start = unicode.index(unicode_word, cursor)
        end = start + len(unicode_word)
        source_start = origins[start][0]
        source_end = origins[end - 1][1]
        assert zawgyi[source_start:source_end] == zawgyi_word
        cursor = end


def test_text_outside_myanmar_runs_keeps_identity_offsets():
    converted = convert_legacy_encoding(_NOTE)
    spans = converted.offset_map.converted_to_original_spans

    prefix = "Patient: "
    assert converted.text.startswith(prefix)
    assert spans[: len(prefix)] == tuple(
        (index, index + 1) for index in range(len(prefix))
    )
    suffix_start = converted.text.index(", MRN")
    source_suffix_start = _NOTE.index(", MRN")
    assert spans[suffix_start:] == tuple(
        (index, index + 1) for index in range(source_suffix_start, len(_NOTE))
    )


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(_cp(0x1031, 0x1019), id="short-zawgyi-syllable"),
        pytest.param(_cp(0x1019, 0x1031), id="short-unicode-syllable"),
        pytest.param(_cp(0x1000, 0x1031, 0x101C), id="valid-in-both"),
        pytest.param(_ZG_NAME[:5], id="short-zawgyi-word"),
        pytest.param(f"{_ZG_SENTENCE} {_UNI_SENTENCE}", id="mixed-encodings"),
    ],
)
def test_short_or_ambiguous_myanmar_is_undetermined_and_unchanged(text):
    detection = detect_zawgyi(text)

    assert detection.decision == "undetermined"
    assert not detection.suspected
    converted = convert_legacy_encoding(text)
    assert converted.text == text
    assert converted.encoding == "unicode"
    assert not converted.changed
    assert converted.reason == LEGACY_ENCODING_UNDETERMINED
    assert converted.zawgyi == detection


def test_myanmar_script_presentation_forms_alone_never_convert():
    detection = detect_zawgyi(_SHAN_UNICODE)

    assert detection.zawgyi_forms > 0
    assert detection.zawgyi_signals == 0
    assert detection.decision != "zawgyi"
    converted = convert_legacy_encoding(_SHAN_UNICODE)
    assert converted.text == _SHAN_UNICODE
    assert converted.encoding == "unicode"
    # Callers that know the language is Unicode can skip detection entirely.
    explicit = convert_legacy_encoding(_SHAN_UNICODE, encoding="unicode")
    assert explicit.zawgyi is None
    assert explicit.text == _SHAN_UNICODE


@pytest.mark.parametrize(
    "text",
    [
        "Patient John Smith visited clinic for BP follow-up.",
        "Crème brûlée, José",
        "रमेश शर्मा",
        "สมชาย ใจดี",
        "患者张伟",
        "",
    ],
    ids=["ascii", "latin-1", "devanagari", "thai", "han", "empty"],
)
def test_non_myanmar_text_is_unchanged(text):
    detection = detect_zawgyi(text)

    assert detection.decision == "not-myanmar"
    assert detection.myanmar_chars == 0
    assert detect_legacy_encoding(text) == "unicode"
    converted = convert_legacy_encoding(text)
    assert converted.text == text
    assert converted.encoding == "unicode"
    assert converted.zawgyi is None
    assert converted.reason is None
    assert convert_zawgyi_with_origins(text) == (
        text,
        tuple((index, index + 1) for index in range(len(text))),
    )


def test_iscii_detection_and_conversion_are_unchanged():
    prefixed = bytes.fromhex("ef 42") + _ISCII_NAME

    assert detect_legacy_encoding(prefixed) == "iscii"
    assert detect_legacy_encoding(_ISCII_NAME.decode("latin-1")) == "iscii"
    converted = convert_legacy_encoding(prefixed)
    assert converted == iscii_to_unicode(prefixed)
    assert converted.text == "रमेश शर्मा"
    assert converted.zawgyi is None
    assert converted.reason is None
    assert converted.offset_unit == "byte"


def test_disabled_conversion_returns_typed_unconverted_reason():
    converted = convert_legacy_encoding(_ZG_SENTENCE, convert_zawgyi=False)

    assert converted.text == _ZG_SENTENCE
    assert converted.encoding == "unicode"
    assert not converted.changed
    assert converted.reason == LEGACY_ENCODING_UNCONVERTED
    assert converted.zawgyi is not None
    assert converted.zawgyi.decision == "zawgyi"
    assert converted.to_metadata()["reason"] == "legacy_encoding_unconverted"


def test_explicit_zawgyi_forces_conversion_of_short_runs():
    short = _cp(0x1031, 0x1019)

    converted = convert_legacy_encoding(short, encoding="zawgyi")

    assert converted.text == _cp(0x1019, 0x1031)
    assert converted.encoding == "zawgyi"
    assert converted.zawgyi is not None
    assert converted.zawgyi.decision == "undetermined"
    assert converted.to_original_span(0, 1) == (0, 2)
    assert zawgyi_to_unicode(short.encode("utf-8")).text == converted.text


@pytest.mark.parametrize(
    ("zawgyi", "unicode"),
    [
        # LETTER U + VOWEL SIGN II composes to LETTER UU under NFC.
        pytest.param(_cp(0x1025, 0x102E), _cp(0x1026), id="uu-composition"),
        # Zawgyi reuses LETTER U for NYA before the killer.
        pytest.param(_cp(0x1025, 0x1039), _cp(0x1009, 0x103A), id="nya-before-killer"),
        # Contraction keeps asat before vowel sign U.
        pytest.param(
            _cp(0x1000, 0x103A, 0x103C, 0x1014, 0x1039, 0x102F, 0x1015, 0x1039),
            _cp(0x1000, 0x103B, 0x103D, 0x1014, 0x103A, 0x102F, 0x1015, 0x103A),
            id="asat-contraction",
        ),
        # Medial HA + U ligature and NNA + DDA ligature.
        pytest.param(
            _cp(0x1019, 0x1088, 0x20, 0x1091),
            _cp(0x1019, 0x103E, 0x102F, 0x20, 0x100F, 0x1039, 0x100D),
            id="ligatures",
        ),
        # Dot below typed after the killer still lands before asat.
        pytest.param(
            _cp(0x101E, 0x100A, 0x1039, 0x1037),
            _cp(0x101E, 0x100A, 0x1037, 0x103A),
            id="dot-below-order",
        ),
    ],
)
def test_contextual_rules_produce_canonical_unicode(zawgyi, unicode):
    converted = zawgyi_to_unicode(zawgyi)

    assert converted.text == unicode
    assert unicodedata.normalize("NFC", converted.text) == converted.text
    assert converted.to_original_span(0, len(converted.text)) == (0, len(zawgyi))


def test_detection_is_deterministic_and_validates_threshold():
    first = detect_zawgyi(_ZG_NAMES)

    assert all(detect_zawgyi(_ZG_NAMES) == first for _ in range(3))
    assert convert_legacy_encoding(_ZG_NAMES) == convert_legacy_encoding(_ZG_NAMES)
    strict = detect_zawgyi(_ZG_NAMES, threshold=0.999)
    assert strict.decision == "undetermined"
    assert strict.confidence == first.confidence
    for threshold in (0.5, 1.5, True):
        with pytest.raises(ValueError, match="threshold"):
            detect_zawgyi(_ZG_NAMES, threshold=threshold)
    with pytest.raises(TypeError):
        detect_zawgyi(_ZG_NAMES.encode("utf-8"))


def test_diagnostics_contain_only_decisions_and_counts(caplog):
    caplog.set_level(logging.DEBUG)
    conversion = convert_legacy_encoding(_NOTE)
    unconverted = convert_legacy_encoding(_NOTE, convert_zawgyi=False)
    normalization = normalize_for_pii_detection(_NOTE)
    document = Pipeline().stage1_normalize(_NOTE)

    payloads = [
        conversion.to_metadata(),
        unconverted.to_metadata(),
        normalization.to_metadata(),
        dict(document.metadata["legacy_encoding"]),
    ]
    for payload in payloads:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        assert not any(0x1000 <= ord(char) <= 0x109F for char in serialized)
        assert "Patient" not in serialized
        assert "12345" not in serialized
    assert set(conversion.zawgyi.to_metadata()) == {
        "decision",
        "confidence",
        "threshold",
        "myanmar_chars",
        "zawgyi_signals",
        "zawgyi_forms",
        "unicode_signals",
    }
    for value in conversion.zawgyi.to_metadata().values():
        assert isinstance(value, (str, int, float))
    for record in caplog.records:
        message = record.getMessage()
        assert not any(0x1000 <= ord(char) <= 0x109F for char in message)


def test_detection_normalizer_converts_zawgyi_and_remaps_spans():
    normalized = normalize_for_pii_detection(_NOTE)

    assert normalized.legacy_encoding == "zawgyi"
    assert normalized.converted_legacy_bytes > 0
    metadata = normalized.to_metadata()
    assert metadata["legacy_encoding_reason"] is None
    assert metadata["zawgyi"]["decision"] == "zawgyi"
    name_start = len("Patient: ")
    name_end = normalized.text.index(",")
    assert normalized.remap_span(name_start, name_end) == (
        _NOTE.index(_ZG_NAME),
        _NOTE.index(_ZG_NAME) + len(_ZG_NAME),
    )
    mrn = normalized.text.index("MRN")
    source_start, source_end = normalized.remap_span(mrn, mrn + 9)
    assert _NOTE[source_start:source_end] == "MRN 12345"


def test_detection_normalizer_reports_undetermined_myanmar():
    text = f"ID {_cp(0x1031, 0x1019)}"

    normalized = normalize_for_pii_detection(text)

    assert normalized.legacy_encoding == "unicode"
    assert normalized.legacy_encoding_reason == LEGACY_ENCODING_UNDETERMINED
    assert normalized.to_metadata()["zawgyi"]["decision"] == "undetermined"
    assert normalize_for_pii_detection("Patient").to_metadata()["zawgyi"] is None


def test_pipeline_converts_zawgyi_before_script_routing():
    pipeline = Pipeline()

    document = pipeline.stage1_normalize(_ZG_SENTENCE)
    route = pipeline.stage2_language_script(document.normalized_text)

    assert document.normalized_text == _UNI_SENTENCE
    assert document.offset_map.normalized_span_to_original_offsets(
        0, len(document.normalized_text)
    ) == (0, len(_ZG_SENTENCE))
    legacy = document.metadata["legacy_encoding"]
    assert legacy["encoding"] == "zawgyi"
    assert legacy["changed"] is True
    assert legacy["converted_bytes"] > 0
    assert legacy["reason"] is None
    assert legacy["zawgyi"]["decision"] == "zawgyi"
    assert route.script == "Myanmar"

    name = _UNI_SENTENCE.split(" ")[1]
    start = document.normalized_text.index(name)
    source_start, source_end = document.offset_map.normalized_span_to_original_offsets(
        start, start + len(name)
    )
    assert _ZG_SENTENCE[source_start:source_end] == _ZG_SENTENCE.split(" ")[1]


def test_pipeline_refuses_to_route_unconverted_zawgyi(monkeypatch):
    def fail_routing(self, text):  # pragma: no cover - must not run
        raise AssertionError("suspected Zawgyi must not be routed")

    monkeypatch.setattr(Pipeline, "stage2_language_script", fail_routing)
    pipeline = Pipeline(lang="auto", convert_zawgyi=False)

    document = pipeline.stage1_normalize(_ZG_SENTENCE)
    assert document.normalized_text == unicodedata.normalize("NFC", _ZG_SENTENCE)
    assert document.metadata["legacy_encoding"]["reason"] == (
        LEGACY_ENCODING_UNCONVERTED
    )
    with pytest.raises(InputError) as error:
        pipeline.run(_ZG_SENTENCE)

    assert error.value.details["reason"] == "legacy_encoding_unconverted"
    assert error.value.details["suspected_encoding"] == "zawgyi"
    rendered = json.dumps(error.value.to_dict(), ensure_ascii=False)
    assert not any(0x1000 <= ord(char) <= 0x109F for char in rendered)


def test_pipeline_routes_undetermined_myanmar_without_rejecting():
    pipeline = Pipeline()
    text = f"ID {_cp(0x1031, 0x1019)}"

    document = pipeline.stage1_normalize(text)

    assert document.metadata["legacy_encoding"]["reason"] == (
        LEGACY_ENCODING_UNDETERMINED
    )
    assert document.metadata["legacy_encoding"]["encoding"] == "unicode"
