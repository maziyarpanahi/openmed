"""Tests for synthetic cross-lingual concept grounding aliases."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pytest

from openmed.clinical.grounding import (
    VocabLoader,
    VocabSource,
    VocabularyIndex,
    generate_candidates,
)
from openmed.clinical.grounding.linkers.hpo import HpoLinker
from openmed.clinical.grounding.linkers.icd10cm import Icd10cmLinker
from openmed.clinical.grounding.linkers.rxnorm import RxNormLinker

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "grounding_crosslingual.jsonl"
)
LANGUAGES = ("es", "fr", "de", "zh")
RESTRICTED_MARKERS = (
    "mimic",
    "i2b2",
    "n2c2",
    "umls",
    "snomed",
    "cpt",
    "medical record number",
    "discharge summary",
)


def _rows() -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _index(system: str) -> VocabularyIndex:
    loader = VocabLoader(registry={system: VocabSource(system=system, path=FIXTURE)})
    return loader.get_index(system)


def _linkers() -> dict[str, object]:
    return {
        "icd10cm": Icd10cmLinker(_index("icd10cm")),
        "rxnorm": RxNormLinker(_index("rxnorm")),
        "hpo": HpoLinker(_index("hpo")),
    }


def _gold_cases() -> Iterable[tuple[str, str, str, str]]:
    for row in _rows():
        expected_code = str(row["expected_code"])
        mentions = row["gold_mentions"]
        assert isinstance(mentions, dict)
        for language in LANGUAGES:
            yield str(row["system"]), language, str(mentions[language]), expected_code


def _sparse_loader() -> VocabLoader:
    registry = {
        system: VocabSource(system=system, path=FIXTURE)
        for system in ("icd10cm", "rxnorm", "hpo")
    }
    return VocabLoader(registry=registry)


def _sparse_gold_cases() -> Iterable[tuple[str, str, str, str]]:
    # The sparse generator emits the raw concept code, so gold here is the
    # unformatted ``code`` field rather than the linker-formatted ``expected_code``.
    for row in _rows():
        raw_code = str(row["code"])
        mentions = row["gold_mentions"]
        assert isinstance(mentions, dict)
        for language in LANGUAGES:
            yield str(row["system"]), language, str(mentions[language]), raw_code


def test_vocab_loader_indexes_language_aliases_from_mixed_system_fixture():
    icd = _index("icd10cm")
    rxnorm = _index("rxnorm")
    hpo = _index("hpo")

    assert icd.get("diabetes mellitus tipo 2", language="es") == "E119"
    assert rxnorm.get("metformine", language="fr") == "6809"
    assert hpo.get("发热", language="zh") == "HP:0001945"
    assert rxnorm.concept_count == 1


def test_crosslingual_lexical_grounding_meets_per_language_top1_accuracy():
    linkers = _linkers()
    totals: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)

    for system, language, mention, expected_code in _gold_cases():
        candidates = linkers[system].link(mention, language=language, fuzzy=False)
        totals[language] += 1
        if candidates and candidates[0].code == expected_code:
            hits[language] += 1
        assert candidates[0].source_language == language

    for language in LANGUAGES:
        assert hits[language] / totals[language] >= 0.80


def test_english_baseline_matches_legacy_language_default():
    linkers = _linkers()

    for row in _rows():
        mentions = row["gold_mentions"]
        assert isinstance(mentions, dict)
        mention = str(mentions["en"])
        system = str(row["system"])

        legacy = linkers[system].link(mention, fuzzy=False)
        explicit_english = linkers[system].link(mention, language="en", fuzzy=False)

        assert legacy == explicit_english
        assert legacy[0].code == row["expected_code"]
        assert legacy[0].source_language == "en"


def test_language_lookup_falls_back_to_english_when_alias_is_missing():
    rxnorm = _index("rxnorm")

    assert rxnorm.get("metformin", language="es") == "6809"


def test_crosslingual_fixture_loads_in_offline_mode(monkeypatch):
    monkeypatch.setenv("OPENMED_OFFLINE", "1")

    index = _index("icd10cm")

    assert index.get("typ-2-diabetes", language="de") == "E119"


def test_crosslingual_fixture_contains_no_restricted_corpus_markers():
    text = FIXTURE.read_text(encoding="utf-8").casefold()

    for marker in RESTRICTED_MARKERS:
        assert marker not in text
    for row in _rows():
        assert row["provenance"] == "synthetic-permissive"


def test_sparse_generator_crosslingual_top1_accuracy_and_source_language():
    loader = _sparse_loader()
    totals: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)

    for system, language, mention, raw_code in _sparse_gold_cases():
        candidates = generate_candidates(
            mention, systems=[system], loader=loader, language=language
        )
        assert candidates
        totals[language] += 1
        if candidates[0].code == raw_code:
            hits[language] += 1
        assert candidates[0].source_language == language

    for language in LANGUAGES:
        assert hits[language] / totals[language] >= 0.80


def test_sparse_generator_stamps_source_language_on_every_candidate():
    loader = _sparse_loader()

    for system, language, mention, _ in _sparse_gold_cases():
        candidates = generate_candidates(
            mention, systems=[system], loader=loader, language=language
        )
        assert candidates
        for candidate in candidates:
            assert candidate.source_language == language
            assert candidate.source == "sparse"


def test_fuzzy_stage_reaches_english_only_concept_under_non_english_language(tmp_path):
    # Mixed-coverage index: concept A carries es+en aliases, concept B is
    # English-only. A misspelled (fuzzy) mention of B under language="es" must
    # still reach B, since the exact stage already falls back to English aliases.
    rows = [
        {
            "system": "icd10cm",
            "code": "A000",
            "display": "Alpha condition",
            "synonyms": ["alpha condition"],
            "language_aliases": {"es": ["afeccion alfa"]},
        },
        {
            "system": "icd10cm",
            "code": "B000",
            "display": "Beta herpesviral infection",
            "synonyms": ["herpesviral infection"],
            "language_aliases": {},
        },
    ]
    path = tmp_path / "mixed_coverage.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loader = VocabLoader(registry={"icd10cm": VocabSource(system="icd10cm", path=path)})

    english = generate_candidates(
        "herpesviral infektion", systems=["icd10cm"], loader=loader, language="en"
    )
    assert any(candidate.code == "B000" for candidate in english)

    spanish = generate_candidates(
        "herpesviral infektion", systems=["icd10cm"], loader=loader, language="es"
    )
    assert any(
        candidate.code == "B000" and candidate.source_language == "es"
        for candidate in spanish
    )


def test_sparse_generator_english_default_is_byte_identical():
    loader = _sparse_loader()

    for row in _rows():
        system = str(row["system"])
        mention = str(row["gold_mentions"]["en"])

        default = generate_candidates(mention, systems=[system], loader=loader)
        english = generate_candidates(
            mention, systems=[system], loader=loader, language="en"
        )

        assert [repr(candidate) for candidate in default] == [
            repr(candidate) for candidate in english
        ]
        assert default[0].code == str(row["code"])
        assert default[0].source_language == "en"


def test_sparse_generator_falls_back_to_english_alias_set():
    loader = _sparse_loader()

    candidates = generate_candidates(
        "metformin", systems=["rxnorm"], loader=loader, language="es"
    )

    assert candidates
    assert candidates[0].code == "6809"
    assert candidates[0].match_kind == "exact"
    assert candidates[0].source_language == "es"
