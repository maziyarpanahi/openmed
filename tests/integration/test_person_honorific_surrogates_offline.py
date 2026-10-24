"""Offline end-to-end replacement of honorific-bearing PERSON spans (#3887).

Each synthetic note carries one PERSON span with a gendered honorific. The
spans are fed through the de-identification result builder exactly as a model
prediction would be, without loading a model or touching the network. All
names are synthetic compositions of common given and family names.
"""

from __future__ import annotations

import warnings

import pytest

from openmed.core.pii import _build_deidentification_result
from openmed.processing.outputs import EntityPrediction, PredictionResult

pytestmark = pytest.mark.integration

# (lang, note template with {name}, PERSON span, verbatim honorific prefix)
_NOTES = (
    ("th", "ผู้ป่วย {name} มาตรวจตามนัด", "นางสาวสมหญิง ใจดี", "นางสาว"),
    ("sw", "Mgonjwa {name} amefika kliniki leo.", "Bi. Amina Hassan", "Bi. "),
    ("vi", "Bệnh nhân {name} tái khám hôm nay.", "Chị Lê Thu Hà", "Chị "),
    ("ur", "مریض {name} معائنے کے لیے آئے۔", "جناب عارف قریشی صاحب", "جناب "),
    ("ha", "Mara lafiya {name} ta zo asibiti.", "Hajiya Bilkisu Balarabe", "Hajiya "),
    ("id", "Pasien {name} datang kontrol.", "Ibu Dewi Lestari", "Ibu "),
)


def _replace(lang: str, text: str, spans: list[tuple[int, int]], seed: int):
    entities = [
        EntityPrediction(
            text=text[start:end],
            label="PERSON",
            confidence=0.99,
            start=start,
            end=end,
        )
        for start, end in spans
    ]
    prediction = PredictionResult(
        text=text,
        entities=entities,
        model_name="offline-synthetic-person-spans",
        timestamp="2026-10-24T00:00:00Z",
        metadata={},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _build_deidentification_result(
            text,
            prediction,
            effective_method="replace",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang=lang,
            consistent=True,
            seed=seed,
            locale=None,
            use_safety_sweep=False,
        )


def _note(template: str, name: str, mentions: int = 1):
    text = " ".join([template.format(name=name)] * mentions)
    spans = []
    cursor = 0
    for _ in range(mentions):
        start = text.index(name, cursor)
        spans.append((start, start + len(name)))
        cursor = start + len(name)
    return text, spans


@pytest.mark.parametrize(("lang", "template", "name", "honorific"), _NOTES)
def test_replacement_keeps_honorific_and_context(lang, template, name, honorific):
    text, spans = _note(template, name)

    result = _replace(lang, text, spans, seed=3887)

    before, after = template.split("{name}")
    output = result.deidentified_text
    assert output.startswith(before)
    assert output.endswith(after)
    surrogate = output[len(before) : len(output) - len(after)]
    assert surrogate.startswith(honorific)
    assert surrogate != name
    assert name not in output
    assert name[len(honorific) :] not in output
    assert [entity.surrogate for entity in result.pii_entities] == [surrogate]


@pytest.mark.parametrize(("lang", "template", "name", "honorific"), _NOTES)
def test_repeated_mentions_share_one_surrogate_and_are_deterministic(
    lang, template, name, honorific
):
    text, spans = _note(template, name, mentions=2)

    first = _replace(lang, text, spans, seed=11)
    second = _replace(lang, text, spans, seed=11)

    surrogates = [entity.surrogate for entity in first.pii_entities]
    assert len(surrogates) == 2
    assert surrogates[0] == surrogates[1]
    assert surrogates[0].startswith(honorific)
    assert first.deidentified_text == second.deidentified_text


def test_honorific_only_span_is_masked_in_the_note():
    text, spans = _note("Mgonjwa {name} amefika kliniki leo.", "Bi.")

    result = _replace("sw", text, spans, seed=5)

    assert result.deidentified_text == "Mgonjwa [PERSON] amefika kliniki leo."
