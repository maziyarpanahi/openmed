"""Deterministic synthetic site holdouts for federated evaluation tests.

``build_synthetic_site_holdouts`` builds ``BenchmarkFixture`` documents for
the first ``site_count`` entries of ``SYNTHETIC_SITE_IDS``. One seed selects
each site's language, label mix, and the placeholder values in the note.
The same arguments return equal fixtures. A different seed changes the note
text. Every gold span's offsets slice the placeholder stored on that span.

Names and identifiers are drawn only from ``DOCUMENTED_FAKE_VALUES``. Those
strings are fixed placeholders (``Alex Fixture``, ``FAKE-MRN-0001``,
``555-0100``, ``alex.fixture@example.test``, ``FAKE-DATE-01``). This module
does not read patient data, credentials, private paths, or source payloads.

Assignment for site index ``i`` and document index ``d``:

- language: ``SITE_LANGUAGES[(seed + i) % len(SITE_LANGUAGES)]``
- label mix: ``LABEL_MIXES[(seed + i) % len(LABEL_MIXES)]``
- value: ``pool[(seed + i * 8 + d * 3 + slot) % len(pool)]``

``slot`` is the label's position in ``LABEL_VALUE_LISTS``.
"""

from __future__ import annotations

from collections.abc import Mapping

from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan

# Obviously fake person names. The second token marks every value as a fixture.
FAKE_NAMES: tuple[str, ...] = (
    "Alex Fixture",
    "Blair Specimen",
    "Casey Placeholder",
    "Drew Synthetic",
    "Eden Example",
    "Finn Notreal",
    "Gray Sample",
    "Harper Dummy",
)

# Obviously fake record identifiers. The FAKE-MRN- prefix is intentional.
FAKE_IDENTIFIERS: tuple[str, ...] = (
    "FAKE-MRN-0001",
    "FAKE-MRN-0002",
    "FAKE-MRN-0003",
    "FAKE-MRN-0004",
    "FAKE-MRN-0005",
    "FAKE-MRN-0006",
    "FAKE-MRN-0007",
    "FAKE-MRN-0008",
)

# Reserved fictional US numbers in the 555-0100 block.
FAKE_PHONES: tuple[str, ...] = (
    "555-0100",
    "555-0101",
    "555-0102",
    "555-0103",
)

# example.test is a reserved documentation domain.
FAKE_EMAILS: tuple[str, ...] = (
    "alex.fixture@example.test",
    "blair.specimen@example.test",
    "casey.placeholder@example.test",
    "drew.synthetic@example.test",
)

# Non-calendar tokens so a holdout never embeds a real date.
FAKE_DATES: tuple[str, ...] = (
    "FAKE-DATE-01",
    "FAKE-DATE-02",
    "FAKE-DATE-03",
    "FAKE-DATE-04",
)

LABEL_VALUE_LISTS: Mapping[str, tuple[str, ...]] = {
    "PERSON": FAKE_NAMES,
    "ID_NUM": FAKE_IDENTIFIERS,
    "PHONE": FAKE_PHONES,
    "EMAIL": FAKE_EMAILS,
    "DATE": FAKE_DATES,
}

DOCUMENTED_FAKE_VALUES: frozenset[str] = frozenset(
    value for pool in LABEL_VALUE_LISTS.values() for value in pool
)

SYNTHETIC_SITE_IDS: tuple[str, ...] = (
    "synthetic-site-alpha",
    "synthetic-site-bravo",
    "synthetic-site-charlie",
    "synthetic-site-delta",
    "synthetic-site-echo",
    "synthetic-site-foxtrot",
)

SITE_LANGUAGES: tuple[str, ...] = ("en", "es", "fr", "de", "zh")

LABEL_MIXES: tuple[tuple[str, ...], ...] = (
    ("PERSON",),
    ("ID_NUM",),
    ("PERSON", "ID_NUM"),
    ("PERSON", "PHONE"),
    ("PERSON", "EMAIL"),
    ("ID_NUM", "DATE"),
    ("PERSON", "ID_NUM", "DATE"),
    ("PHONE", "EMAIL"),
)

_LABEL_SLOT: Mapping[str, int] = {
    label: slot for slot, label in enumerate(LABEL_VALUE_LISTS)
}

# Lead sentence plus one sentence per label. "{value}" appears once.
_SENTENCES: Mapping[str, Mapping[str, str]] = {
    "en": {
        "lead": "Synthetic clinic note.",
        "PERSON": "Patient {value} was seen.",
        "ID_NUM": "Record {value} is a placeholder.",
        "PHONE": "Callback {value} is disconnected.",
        "EMAIL": "Mailbox {value} is reserved.",
        "DATE": "Anchor {value} is fictional.",
    },
    "es": {
        "lead": "Nota clinica sintetica.",
        "PERSON": "Paciente {value} acudio.",
        "ID_NUM": "Registro {value} es ficticio.",
        "PHONE": "Telefono {value} no existe.",
        "EMAIL": "Correo {value} esta reservado.",
        "DATE": "Fecha {value} es ficticia.",
    },
    "fr": {
        "lead": "Note clinique synthetique.",
        "PERSON": "Patient {value} est vu.",
        "ID_NUM": "Dossier {value} est fictif.",
        "PHONE": "Telephone {value} est faux.",
        "EMAIL": "Courriel {value} est reserve.",
        "DATE": "Date {value} est fictive.",
    },
    "de": {
        "lead": "Synthetische klinische Notiz.",
        "PERSON": "Patient {value} wurde gesehen.",
        "ID_NUM": "Akte {value} ist ein Platzhalter.",
        "PHONE": "Rufnummer {value} ist frei.",
        "EMAIL": "Postfach {value} ist reserviert.",
        "DATE": "Ankerdatum {value} ist erfunden.",
    },
    "zh": {
        "lead": "合成临床记录。",
        "PERSON": "患者 {value} 到诊。",
        "ID_NUM": "编号 {value} 为占位。",
        "PHONE": "电话 {value} 无效。",
        "EMAIL": "邮箱 {value} 为保留。",
        "DATE": "日期 {value} 为虚构。",
    },
}


def _validate_catalog() -> None:
    if set(_SENTENCES) != set(SITE_LANGUAGES):
        raise RuntimeError("site languages and sentence templates differ")
    for language, sentences in _SENTENCES.items():
        if "lead" not in sentences:
            raise RuntimeError(f"{language} template is missing a lead")
        for label in LABEL_VALUE_LISTS:
            template = sentences.get(label, "")
            if template.count("{value}") != 1:
                raise RuntimeError(
                    f"{language} {label} template must contain one value slot"
                )
    for mix in LABEL_MIXES:
        if not mix or any(label not in LABEL_VALUE_LISTS for label in mix):
            raise RuntimeError("label mix must use documented labels")
    unique_values = sum(len(pool) for pool in LABEL_VALUE_LISTS.values())
    if len(DOCUMENTED_FAKE_VALUES) != unique_values:
        raise RuntimeError("documented fake values must be unique")


_validate_catalog()


def _require_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")


def build_synthetic_site_holdouts(
    site_count: int,
    seed: int,
    *,
    documents_per_site: int = 1,
) -> tuple[BenchmarkFixture, ...]:
    """Return deterministic holdout fixtures for ``site_count`` synthetic sites.

    Args:
        site_count: Number of named sites. Must be from 1 through
            ``len(SYNTHETIC_SITE_IDS)``. Sites are
            ``SYNTHETIC_SITE_IDS[:site_count]``, in that order.
        seed: Non-negative integer that selects language, label mix, and
            placeholder values. The same arguments return equal fixtures.
        documents_per_site: How many notes to emit for each site.

    Returns:
        Fixtures in site order. ``metadata["site_id"]`` is the synthetic
        site id, ``metadata["label_mix"]`` is the gold label tuple, and
        each gold span's text is the documented placeholder sliced by its
        offsets.
    """

    _require_int("site_count", site_count)
    _require_int("seed", seed)
    _require_int("documents_per_site", documents_per_site)
    if not 1 <= site_count <= len(SYNTHETIC_SITE_IDS):
        raise ValueError(f"site_count must be from 1 to {len(SYNTHETIC_SITE_IDS)}")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if documents_per_site < 1:
        raise ValueError("documents_per_site must be at least 1")

    fixtures: list[BenchmarkFixture] = []
    for site_index, site_id in enumerate(SYNTHETIC_SITE_IDS[:site_count]):
        language = SITE_LANGUAGES[(seed + site_index) % len(SITE_LANGUAGES)]
        label_mix = LABEL_MIXES[(seed + site_index) % len(LABEL_MIXES)]
        for document_index in range(documents_per_site):
            values = {
                label: _placeholder(label, seed, site_index, document_index)
                for label in label_mix
            }
            text, spans = _render_note(language, label_mix, values)
            fixtures.append(
                BenchmarkFixture(
                    fixture_id=f"{site_id}-holdout-{document_index:02d}",
                    text=text,
                    gold_spans=spans,
                    language=language,
                    metadata={
                        "document_index": document_index,
                        "generator": "tests.fixtures.synthetic_site_holdouts",
                        "label_mix": label_mix,
                        "seed": seed,
                        "site_id": site_id,
                        "synthetic": True,
                    },
                )
            )
    return tuple(fixtures)


def _placeholder(label: str, seed: int, site_index: int, document_index: int) -> str:
    pool = LABEL_VALUE_LISTS[label]
    slot = _LABEL_SLOT[label]
    index = (seed + site_index * 8 + document_index * 3 + slot) % len(pool)
    return pool[index]


def _render_note(
    language: str,
    label_mix: tuple[str, ...],
    values: Mapping[str, str],
) -> tuple[str, tuple[EvalSpan, ...]]:
    sentences = _SENTENCES[language]
    pieces = [sentences["lead"]]
    spans: list[EvalSpan] = []
    cursor = len(pieces[0])
    for label in label_mix:
        value = values[label]
        if value not in DOCUMENTED_FAKE_VALUES:
            raise ValueError("refusing a value outside the documented fake list")
        sentence = sentences[label].format(value=value)
        if sentence.count(value) != 1:
            raise ValueError(
                f"{language} {label} sentence must contain the placeholder once"
            )
        local = sentence.index(value)
        pieces.append(" ")
        pieces.append(sentence)
        cursor += 1
        start = cursor + local
        end = start + len(value)
        spans.append(
            EvalSpan(
                start=start,
                end=end,
                label=label,
                text=value,
                language=language,
            )
        )
        cursor += len(sentence)
    text = "".join(pieces)
    for span in spans:
        if text[span.start : span.end] != span.text:
            raise ValueError("gold span does not slice the synthetic value")
    return text, tuple(spans)
