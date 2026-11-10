"""Hausa, Yoruba, Igbo and Kinyarwanda month-name date regressions (#3865).

These national-ID-only routes ship no bundled model, so deterministic patterns
are their main protection. Every input is synthetic.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from openmed.core.pii import _shift_date
from openmed.core.pii_i18n import LANGUAGE_MONTH_NAMES, NATIONAL_ID_ONLY_LANGUAGES
from openmed.core.pipeline import Pipeline
from openmed.core.safety_sweep import safety_sweep
from openmed.processing.outputs import PredictionResult

_STUB_MODEL = "synthetic/user-supplied-stub"
_RW_FIXTURE_PATH = Path("openmed/eval/golden/fixtures/i18n/rw.jsonl")

# CLDR gregorian wide month names; Yoruba drops CLDR's leading "Oṣù".
_EXPECTED_MONTH_NAMES = {
    "ha": [
        "Janairu",
        "Faburairu",
        "Maris",
        "Afirilu",
        "Mayu",
        "Yuni",
        "Yuli",
        "Agusta",
        "Satumba",
        "Oktoba",
        "Nuwamba",
        "Disamba",
    ],
    "ig": [
        "Jenụwarị",
        "Febrụwarị",
        "Maachị",
        "Epreel",
        "Mee",
        "Jun",
        "Julaị",
        "Ọgọọst",
        "Septemba",
        "Ọktoba",
        "Novemba",
        "Disemba",
    ],
    "rw": [
        "Mutarama",
        "Gashyantare",
        "Werurwe",
        "Mata",
        "Gicurasi",
        "Kamena",
        "Nyakanga",
        "Kanama",
        "Nzeri",
        "Ukwakira",
        "Ugushyingo",
        "Ukuboza",
    ],
    "yo": [
        "Ṣẹ́rẹ́",
        "Èrèlè",
        "Ẹrẹ̀nà",
        "Ìgbé",
        "Ẹ̀bibi",
        "Òkúdu",
        "Agẹmọ",
        "Ògún",
        "Owewe",
        "Ọ̀wàrà",
        "Bélú",
        "Ọ̀pẹ̀",
    ],
}


def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def _unmarked(text: str) -> str:
    return "".join(
        char for char in _nfd(text) if not unicodedata.category(char).startswith("M")
    )


# (lang, text, exact date surface)
_POSITIVE_CASES = (
    ("ha", "Ranar haihuwa: 15 Satumba 1980.", "15 Satumba 1980"),
    ("ha", "An haife shi 3 ga watan Fabrairu, 1975.", "3 ga watan Fabrairu, 1975"),
    ("ha", "Kwanan wata: 7 ga Faburairu 2001.", "7 ga Faburairu 2001"),
    ("ha", "Ranar haihuwa: Afrilu 21, 1990.", "Afrilu 21, 1990"),
    ("yo", "Ọjọ́ ìbí: 15 Ṣẹ́rẹ́ 1980.", "15 Ṣẹ́rẹ́ 1980"),
    ("yo", _nfd("Ọjọ́ ìbí: 15 Ṣẹ́rẹ́ 1980."), _nfd("15 Ṣẹ́rẹ́ 1980")),
    ("yo", "Ojo ibi: 15 Sere 1980.", "15 Sere 1980"),
    ("yo", "Ọjọ́ ìbí: 2 Ẹrẹna 1984.", "2 Ẹrẹna 1984"),
    ("yo", "Ọjọ́ ìbí: 9 oṣù Ọ̀wàrà ọdún 1969.", "9 oṣù Ọ̀wàrà ọdún 1969"),
    ("yo", "Ojo ibi: 9 osu Owara odun 1969.", "9 osu Owara odun 1969"),
    ("yo", "Ọjọ́ ìbí: Ògún 15, 1980.", "Ògún 15, 1980"),
    ("ig", "Ụbọchị ọmụmụ: 15 Jenụwarị 1980.", "15 Jenụwarị 1980"),
    ("ig", _nfd("Ụbọchị ọmụmụ: 15 Jenụwarị 1980."), _nfd("15 Jenụwarị 1980")),
    ("ig", "Ubochi omumu: 15 Jenuwari 1980.", "15 Jenuwari 1980"),
    ("ig", "Ụbọchị ọmụmụ: 4 Ọgọstụ 1999.", "4 Ọgọstụ 1999"),
    ("ig", "Ụbọchị ọmụmụ: Disemba 25, 2001.", "Disemba 25, 2001"),
    ("rw", "Itariki y'amavuko: 15 Mutarama 1980.", "15 Mutarama 1980"),
    ("rw", "Yavutse ku wa 2 Nzeli 1995.", "2 Nzeli 1995"),
    ("rw", "Date of birth: Mata 12, 1988.", "Mata 12, 1988"),
    ("rw", "Itariki y'amavuko: 30 Ukuboza 2004.", "30 Ukuboza 2004"),
)

# Month words with no attached day and year, including homonyms such as
# Kinyarwanda "mata" (milk) and the Yoruba state name "Ògún".
_NEGATIVE_CASES = (
    ("ha", "Asibiti zai sake buɗe a watan Satumba."),
    ("ha", "Ana duba marasa lafiya daga Maris zuwa Mayu."),
    ("yo", "Ilé-ìwòsàn yóò ṣí ní oṣù Ṣẹ́rẹ́."),
    ("yo", "Aláìsàn náà ń gbé ní ìpínlẹ̀ Ògún."),
    ("yo", _unmarked("Aláìsàn náà ń gbé ní ìpínlẹ̀ Ògún.")),
    ("ig", "Ụlọ ọgwụ ga-emeghe na Jenụwarị."),
    ("ig", "Ọ bịara ụlọ ọgwụ na Mee."),
    ("rw", "Ivuriro rizafungura muri Mutarama."),
    ("rw", "Umurwayi anywa mata buri munsi."),
    ("rw", "Yanyoye mata 2 ku munsi."),
)


def _empty_detector(text: str, **_kwargs: object) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="synthetic-empty-detector",
        timestamp="2026-01-01T00:00:00Z",
    )


def _masked(lang: str, text: str) -> str:
    pipeline = Pipeline(
        lang=lang,
        model_name=_STUB_MODEL,
        model_detector=_empty_detector,
        use_smart_merging=True,
        use_safety_sweep=True,
    )
    return pipeline.run(text, method="mask").deidentification_result.deidentified_text


def _date_spans(lang: str, text: str) -> list[tuple[int, int, str]]:
    return [
        (entity.start, entity.end, entity.text)
        for entity in safety_sweep(text, [], lang=lang)
        if entity.label == "date"
    ]


@pytest.mark.parametrize("lang", sorted(_EXPECTED_MONTH_NAMES))
def test_month_vocabulary_is_registered_as_twelve_cldr_names(lang: str) -> None:
    assert lang in NATIONAL_ID_ONLY_LANGUAGES
    assert LANGUAGE_MONTH_NAMES[lang] == _EXPECTED_MONTH_NAMES[lang]
    assert all(
        unicodedata.is_normalized("NFC", month) for month in LANGUAGE_MONTH_NAMES[lang]
    )


@pytest.mark.parametrize(("lang", "text", "surface"), _POSITIVE_CASES)
def test_month_name_date_is_swept_at_exact_offsets(
    lang: str, text: str, surface: str
) -> None:
    start = text.index(surface)

    assert (start, start + len(surface), surface) in _date_spans(lang, text)


@pytest.mark.parametrize(("lang", "text", "surface"), _POSITIVE_CASES)
def test_empty_detector_pipeline_masks_month_name_date(
    lang: str, text: str, surface: str
) -> None:
    start = text.index(surface)
    end = start + len(surface)

    masked = _masked(lang, text)

    assert surface not in masked
    assert masked.startswith(text[:start])
    assert masked.endswith(text[end:])
    assert masked != text


@pytest.mark.parametrize("lang", ("yo", "ig"))
@pytest.mark.parametrize("form", ("NFC", "NFD"))
def test_every_marked_month_is_grapheme_safe_in_both_normal_forms(
    lang: str, form: str
) -> None:
    for month in LANGUAGE_MONTH_NAMES[lang]:
        for surface in (f"14 {month} 1984", f"{month} 14, 1984"):
            surface = unicodedata.normalize(form, surface)
            text = unicodedata.normalize(form, f"Ọjọ́: {surface}. Ọ̀nà.")
            start = text.index(surface)
            end = start + len(surface)

            assert _date_spans(lang, text) == [(start, end, surface)]
            assert not unicodedata.category(text[start]).startswith("M")
            assert not unicodedata.category(text[end]).startswith("M")


@pytest.mark.parametrize("lang", ("yo", "ig"))
def test_every_unmarked_month_spelling_matches(lang: str) -> None:
    for month in LANGUAGE_MONTH_NAMES[lang]:
        surface = _unmarked(f"14 {month} 1984")
        text = f"Ojo: {surface}."
        start = text.index(surface)

        assert _date_spans(lang, text) == [(start, start + len(surface), surface)]


@pytest.mark.parametrize(("lang", "text"), _NEGATIVE_CASES)
def test_month_word_without_day_and_year_is_not_flagged(lang: str, text: str) -> None:
    assert _date_spans(lang, text) == []
    assert _masked(lang, text) == text


def test_kinyarwanda_fixture_masks_exactly_its_gold_spans_offline() -> None:
    rows = [
        json.loads(line)
        for line in _RW_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert rows
    for row in rows:
        assert row["language"] == "rw"
        assert row["metadata"]["synthetic"] is True
        masked = _masked("rw", row["text"])
        swept = {
            (entity.start, entity.end)
            for entity in safety_sweep(row["text"], [], lang="rw")
        }

        assert swept == {(span["start"], span["end"]) for span in row["gold_spans"]}
        assert not [span for span in row["gold_spans"] if span["text"] in masked]


# Swept (label, start, end) spans and shifted dates recorded on master before
# the four month vocabularies were added. Other languages must not change.
_UNCHANGED_SWEEP_SNAPSHOT = (
    (
        "sw",
        "Tarehe ya kuzaliwa 15 Septemba 1980; kliniki ijayo Machi.",
        [("date", 19, 35)],
    ),
    ("sw", "Alizaliwa 15 Satumba 1980, Mutarama 3, 1999 na 4 Mee 2001.", []),
    (
        "zu",
        "Usuku lokuzalwa 15 Septhemba 1980; ukuvakasha ngoMashi.",
        [("date", 16, 33)],
    ),
    (
        "xh",
        "Umhla wokuzalwa 15 Septemba 1980; utyelelo ngoMatshi.",
        [("date", 16, 32)],
    ),
    (
        "af",
        "Geboortedatum 15 September 1980; volgende besoek in Maart.",
        [("date", 14, 31)],
    ),
    (
        "en",
        "DOB 15 September 1980; review 3 Mata 1999 and Satumba 15, 1980.",
        [("date", 4, 21)],
    ),
    ("fr", "Née le 15 septembre 1980, revue le 3 Mutarama 1999.", [("date", 7, 24)]),
)

_UNCHANGED_SHIFT_SNAPSHOT = (
    ("sw", "15 Septemba 1980", "15 Oktoba 1980"),
    ("af", "15 September 1980", "15 Oktober 1980"),
    ("en", "September 15, 1980", "October 15, 1980"),
    ("fr", "15 septembre 1980", "15 octobre 1980"),
    ("en", "15 Mata 1980", "[DATE_SHIFTED]"),
)


@pytest.mark.parametrize(("lang", "text", "expected"), _UNCHANGED_SWEEP_SNAPSHOT)
def test_other_language_sweep_output_is_unchanged(
    lang: str, text: str, expected: list[tuple[str, int, int]]
) -> None:
    observed = [
        (entity.label, entity.start, entity.end)
        for entity in safety_sweep(text, [], lang=lang)
    ]

    assert observed == expected


@pytest.mark.parametrize(("lang", "value", "expected"), _UNCHANGED_SHIFT_SNAPSHOT)
def test_other_language_date_shift_output_is_unchanged(
    lang: str, value: str, expected: str
) -> None:
    assert _shift_date(value, 30, lang=lang) == expected
