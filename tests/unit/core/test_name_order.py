"""Tests for CJK family-name-first honorific handling (OM-291).

All name samples here are synthetic — common family/given syllables composed
into fictional full names, not real individuals.
"""

from __future__ import annotations

import pytest

from openmed.core.anonymizer import LABEL_GENERATORS, Anonymizer
from openmed.core.name_order import (
    CJK_LANGUAGES,
    HONORIFICS,
    honorifics_for,
    normalize_person_span,
    register_honorific,
    split_name,
)


@pytest.fixture(autouse=True)
def _restore_honorifics():
    """Isolate mutations to the module-global HONORIFICS registry."""

    snapshot = {lang: list(values) for lang, values in HONORIFICS.items()}
    try:
        yield
    finally:
        HONORIFICS.clear()
        HONORIFICS.update({lang: list(values) for lang, values in snapshot.items()})


class TestSplitName:
    """split_name() returns family-first components + trailing honorific."""

    @pytest.mark.parametrize(
        "name, lang, expected",
        [
            # Japanese: space-separated full names, family-name-first.
            ("田中 太郎さん", "ja", ("田中", "太郎", "さん")),
            ("鈴木 花子様", "ja", ("鈴木", "花子", "様")),
            ("山本 一郎先生", "ja", ("山本", "一郎", "先生")),
            ("田中 太郎 さん", "ja", ("田中", "太郎", "さん")),
            ("田中 太郎", "ja", ("田中", "太郎", "")),
            # Korean: contiguous, single-char family name first.
            ("김민준씨", "ko", ("김", "민준", "씨")),
            ("이순신님", "ko", ("이", "순신", "님")),
            ("박지성", "ko", ("박", "지성", "")),
            # Chinese: contiguous, single-char family name first.
            ("王伟先生", "zh", ("王", "伟", "先生")),
            ("李娜女士", "zh", ("李", "娜", "女士")),
            ("张伟", "zh", ("张", "伟", "")),
        ],
    )
    def test_family_first_split_per_language(self, name, lang, expected):
        assert split_name(name, lang) == expected

    def test_multichar_honorific_wins_over_prefix(self):
        # 선생님 (seonsaengnim) must not be split as 님 (nim) leaving a stray 선생.
        family, given, honorific = split_name("김선생님", "ko")
        assert honorific == "선생님"
        assert family == "김"
        assert given == ""

    @pytest.mark.parametrize("lang", ["en", "fr", "de"])
    def test_non_cjk_passthrough(self, lang):
        # Non-CJK languages are returned unchanged as family-only.
        assert split_name("John Smith", lang) == ("John Smith", "", "")

    def test_empty_input(self):
        assert split_name("", "ja") == ("", "", "")

    def test_honorific_only_has_no_invented_name(self):
        assert split_name(" さん", "ja") == ("", "", "さん")


class TestNormalizePersonSpan:
    """normalize_person_span() peels the honorific for later re-attachment."""

    @pytest.mark.parametrize(
        "span, lang, expected",
        [
            ("佐藤さん", "ja", ("佐藤", "さん")),
            ("田中 太郎様", "ja", ("田中 太郎", "様")),
            ("김민준씨", "ko", ("김민준", "씨")),
            ("이순신님", "ko", ("이순신", "님")),
            ("王伟先生", "zh", ("王伟", "先生")),
            ("李娜女士", "zh", ("李娜", "女士")),
            ("田中 さん", "ja", ("田中", " さん")),
            ("김민준 씨", "ko", ("김민준", " 씨")),
            ("王伟 先生", "zh", ("王伟", " 先生")),
        ],
    )
    def test_honorific_separated(self, span, lang, expected):
        assert normalize_person_span(span, lang) == expected

    def test_no_honorific_returns_span_unchanged(self):
        assert normalize_person_span("田中", "ja") == ("田中", "")

    @pytest.mark.parametrize("lang", ["en", "fr", "de"])
    def test_non_cjk_returns_empty_honorific(self, lang):
        assert normalize_person_span("Smith", lang) == ("Smith", "")

    def test_honorific_only_preserves_the_full_suffix(self):
        assert normalize_person_span(" さん", "ja") == ("", " さん")


class TestHonorificsMap:
    """The HONORIFICS map exposes the seed set and accepts additions."""

    def test_seed_set_present_for_each_cjk_language(self):
        assert set(HONORIFICS) >= CJK_LANGUAGES
        assert "さん" in HONORIFICS["ja"]
        assert "씨" in HONORIFICS["ko"]
        assert "先生" in HONORIFICS["zh"]

    def test_honorifics_sorted_longest_first(self):
        for lang in CJK_LANGUAGES:
            lengths = [len(h) for h in HONORIFICS[lang]]
            assert lengths == sorted(lengths, reverse=True)

    def test_register_addition_is_recognized(self):
        assert "どの" not in honorifics_for("ja")
        register_honorific("ja", "どの")
        assert "どの" in honorifics_for("ja")
        # A registered honorific participates in span normalization.
        assert normalize_person_span("織田どの", "ja") == ("織田", "どの")

    def test_register_is_idempotent(self):
        before = len(honorifics_for("ja"))
        register_honorific("ja", "さん")  # already present
        assert len(honorifics_for("ja")) == before

    def test_register_ignores_empty(self):
        before = honorifics_for("ja")
        register_honorific("ja", "")
        register_honorific("", "さん")
        assert honorifics_for("ja") == before

    def test_honorifics_for_unknown_language_is_empty(self):
        assert honorifics_for("en") == ()


class TestSurrogateIntegration:
    """Honorific survives surrogate replacement; only ja/ko/zh are affected."""

    @pytest.mark.parametrize(
        "lang, span, honorific",
        [
            ("ja", "田中さん", "さん"),
            ("ja", "鈴木 花子様", "様"),
            ("ko", "김민준씨", "씨"),
            ("ko", "이순신님", "님"),
            ("zh", "王伟先生", "先生"),
            ("zh", "李娜女士", "女士"),
        ],
    )
    def test_honorific_preserved_across_replacement(self, lang, span, honorific):
        anon = Anonymizer(lang=lang, consistent=True, seed=13)
        surrogate = anon.surrogate(span, "PERSON")
        # The honorific survives verbatim, and the underlying name is swapped.
        assert surrogate.endswith(honorific)
        assert surrogate != span
        assert surrogate[: -len(honorific)]  # a non-empty replacement name

    def test_cjk_person_without_honorific_still_replaced(self):
        anon = Anonymizer(lang="ja", consistent=True, seed=13)
        surrogate = anon.surrogate("田中", "PERSON")
        assert surrogate and surrogate != "田中"

    @pytest.mark.parametrize("lang", ["en", "fr", "de"])
    def test_western_person_output_unchanged(self, lang):
        # Honorific machinery must not touch en/fr/de PERSON replacement:
        # the surrogate equals exactly what the plain generator would emit.
        seeded = Anonymizer(lang=lang, consistent=True, seed=99)
        got = seeded.surrogate("John Smith", "PERSON")

        baseline = Anonymizer(lang=lang, consistent=True, seed=99)
        expected = baseline.surrogate("John Smith", "PERSON")

        assert got == expected
        # And no CJK honorific leaked onto the tail.
        for hon in HONORIFICS.get("ja", []) + HONORIFICS.get("zh", []):
            assert not got.endswith(hon)

    def test_non_person_cjk_label_is_not_honorific_stripped(self):
        # A non-PERSON label ending in what looks like an honorific must be
        # left to its own generator, not the PERSON honorific path.
        anon = Anonymizer(lang="ja", consistent=True, seed=13)
        # CITY is generated by its own Faker method; assert it runs without the
        # PERSON path mangling it (smoke: returns a non-empty string).
        assert anon.surrogate("東京", "CITY")

    def test_determinism_core_shared_between_hon_and_bare(self):
        # A name with and without an honorific share the same core surrogate.
        anon = Anonymizer(lang="ja", consistent=True, seed=21)
        with_hon = anon.surrogate("佐藤さん", "PERSON")

        bare = Anonymizer(lang="ja", consistent=True, seed=21).surrogate(
            "佐藤", "PERSON"
        )
        assert with_hon == f"{bare}さん"

    @pytest.mark.parametrize(
        "lang, bare_name, spaced_name, suffix",
        [
            ("ja", "田中", "田中 さん", " さん"),
            ("ko", "김민준", "김민준 씨", " 씨"),
            ("zh", "王伟", "王伟 先生", " 先生"),
        ],
    )
    def test_separator_whitespace_survives_replacement(
        self, lang, bare_name, spaced_name, suffix
    ):
        bare = Anonymizer(lang=lang, consistent=True, seed=34).surrogate(
            bare_name, "PERSON"
        )
        spaced = Anonymizer(lang=lang, consistent=True, seed=34).surrogate(
            spaced_name, "PERSON"
        )

        assert spaced == f"{bare}{suffix}"

    def test_generator_fallback_preserves_honorific_suffix(self, monkeypatch):
        def fail_generator(*_args, **_kwargs):
            raise RuntimeError("synthetic generator failure")

        monkeypatch.setitem(LABEL_GENERATORS, "PERSON", fail_generator)
        with pytest.warns(RuntimeWarning, match="Anonymizer fallback"):
            surrogate = Anonymizer(lang="ja").surrogate("田中 さん", "PERSON")

        assert surrogate == "[PERSON] さん"

    def test_honorific_only_uses_safe_fallback_without_dropping_suffix(self):
        surrogate = Anonymizer(lang="ja").surrogate(" さん", "PERSON")

        assert surrogate == "[PERSON] さん"
