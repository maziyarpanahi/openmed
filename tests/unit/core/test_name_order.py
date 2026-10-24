"""Tests for CJK (OM-291) and non-CJK (#3887) honorific and name-order handling.

All name samples here are synthetic — common family/given syllables composed
into fictional full names, not real individuals.
"""

from __future__ import annotations

import warnings

import pytest
from faker import Faker
from faker.providers.person.ar_AA import Provider as ArabicPersonProvider
from faker.providers.person.vi_VN import Provider as VietnamesePersonProvider

from openmed.core.anonymizer import LABEL_GENERATORS, Anonymizer
from openmed.core.anonymizer import registry as anonymizer_registry
from openmed.core.anonymizer.locales import (
    resolve_faker_backend_locale,
    resolve_locale,
)
from openmed.core.anonymizer.person_surrogates import (
    AMHARIC_FEMININE_GIVEN_NAMES,
    AMHARIC_MASCULINE_GIVEN_NAMES,
    URDU_FEMININE_GIVEN_NAMES,
    URDU_MASCULINE_GIVEN_NAMES,
)
from openmed.core.name_order import (
    CJK_LANGUAGES,
    HONORIFICS,
    honorifics_for,
    normalize_person_span,
    register_honorific,
    split_name,
)
from openmed.core.person_honorifics import peel_person_honorifics


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

    def test_register_ignores_non_cjk_languages(self):
        register_honorific("en", "Dr")
        assert honorifics_for("en") == ()

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
    def test_western_person_output_unchanged(self, lang, monkeypatch):
        # Prove the non-CJK path passes the original value through to the
        # existing generator and returns its output without CJK post-processing.
        calls = []

        def baseline_generator(_faker, original, *, locale):
            calls.append((original, locale))
            return "baseline-western-surrogate"

        monkeypatch.setitem(LABEL_GENERATORS, "PERSON", baseline_generator)

        got = Anonymizer(lang=lang, consistent=True, seed=99).surrogate(
            "John Smith", "PERSON"
        )

        assert got == "baseline-western-surrogate"
        assert len(calls) == 1
        assert calls[0][0] == "John Smith"

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


# ---------------------------------------------------------------------------
# Non-CJK honorifics, gender cues and name order (#3887)
# ---------------------------------------------------------------------------

_GENDERED_SPANS = [
    # (lang, span, verbatim prefix, verbatim suffix, gender)
    ("th", "นางสาวสมหญิง ใจดี", "นางสาว", "", "female"),
    ("th", "นายสมชาย ใจดี", "นาย", "", "male"),
    ("am", "ወይዘሮ ሜሮን አለሙ", "ወይዘሮ ", "", "female"),
    ("am", "አቶ ዳዊት ከበደ", "አቶ ", "", "male"),
    ("sw", "Bi. Amina Hassan", "Bi. ", "", "female"),
    ("sw", "Bw. Baraka Mushi", "Bw. ", "", "male"),
    ("ar", "السيدة مريم محمود", "السيدة ", "", "female"),
    ("ar", "السيد يوسف إبراهيم", "السيد ", "", "male"),
    ("fa", "خانم سارا احمدی", "خانم ", "", "female"),
    ("fa", "آقای رضا حسینی", "آقای ", "", "male"),
    ("ur", "محترمہ فاطمہ خان", "محترمہ ", "", "female"),
    ("ur", "جناب عارف قریشی صاحب", "جناب ", " صاحب", "male"),
    ("hi", "श्रीमती अनिता शर्मा", "श्रीमती ", "", "female"),
    ("hi", "श्री राजेश कुमार", "श्री ", "", "male"),
    ("vi", "Chị Lê Thu Hà", "Chị ", "", "female"),
    ("vi", "Ông Phạm Quang Huy", "Ông ", "", "male"),
    ("id", "Ibu Dewi Lestari", "Ibu ", "", "female"),
    ("id", "Bapak Agus Pratama", "Bapak ", "", "male"),
    ("yo", "Arábìnrin Bọ́láńlé Adébáyọ̀", "Arábìnrin ", "", "female"),
    ("yo", "Ọ̀gbẹ́ni Adéọlá Ọládìpọ̀", "Ọ̀gbẹ́ni ", "", "male"),
    ("ha", "Hajiya Bilkisu Balarabe", "Hajiya ", "", "female"),
    ("ha", "Malam Musa Ɗanjuma", "Malam ", "", "male"),
]

# Languages whose PERSON surrogates come from gendered Faker or bundled data
# on every supported path, so the given name must positively match the cue.
# Arabic and Amharic script providers and the synthetic Devanagari provider
# may emit names without a known gender; those must never contradict it.
_STRICT_GENDER_LANGUAGES = {"th", "sw", "fa", "ur", "vi", "id", "yo", "ha"}

_TITLE_FREE_LANGUAGES = (
    "vi",
    "de",
    "tr",
    "bg",
    "or",
    "pl",
    "am",
    "da",
    "hu",
    "id",
    "it",
    "ms",
    "sw",
    "uk",
    "xh",
    "yo",
    "zu",
)


def _faker_person_provider(lang: str):
    locale = resolve_faker_backend_locale(
        resolve_locale(lang, warn_approximation=False)
    )
    return Faker(locale).provider("faker.providers.person")


def _gender_oracle(lang: str):
    """Classify a given name with Faker's and the bundled gendered lists."""

    provider = _faker_person_provider(lang)
    female = set(getattr(provider, "first_names_female", ()))
    male = set(getattr(provider, "first_names_male", ()))
    female.update(
        URDU_FEMININE_GIVEN_NAMES,
        AMHARIC_FEMININE_GIVEN_NAMES,
        ArabicPersonProvider.first_names_female,
    )
    male.update(
        URDU_MASCULINE_GIVEN_NAMES,
        AMHARIC_MASCULINE_GIVEN_NAMES,
        ArabicPersonProvider.first_names_male,
    )

    def classify(name: str):
        if name in female and name not in male:
            return "female"
        if name in male and name not in female:
            return "male"
        return None

    return classify


def _faker_title_vocabulary(lang: str) -> set[tuple[str, ...]]:
    """Return the titles Faker can insert for ``lang`` (independent oracle)."""

    provider = _faker_person_provider(lang)
    titles: set[str] = set()
    for attribute in ("prefixes_female", "prefixes_male", "prefixes_nonbinary"):
        titles.update(getattr(provider, attribute, ()))
    titles.update(
        value for value in getattr(provider, "prefixes", ()) if value.endswith(".")
    )
    return {tuple(title.casefold().split()) for title in titles if title.strip()}


def _starts_with_title(surrogate: str, titles: set[tuple[str, ...]]) -> bool:
    tokens = tuple(token.casefold() for token in surrogate.split())
    return any(tokens[: len(title)] == title for title in titles)


class TestNonCJKHonorificSurrogates:
    """Honorific, gender cue and order survive non-CJK PERSON replacement."""

    @pytest.mark.parametrize(
        ("lang", "span", "prefix", "suffix", "gender"), _GENDERED_SPANS
    )
    def test_honorific_kept_and_given_name_matches_gender(
        self, lang, span, prefix, suffix, gender
    ):
        classify = _gender_oracle(lang)
        opposite = "male" if gender == "female" else "female"
        for seed in range(8):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                surrogate = Anonymizer(lang=lang, consistent=True, seed=seed).surrogate(
                    span, "PERSON"
                )

            assert surrogate.startswith(prefix)
            assert surrogate.endswith(suffix)
            core = surrogate[len(prefix) : len(surrogate) - len(suffix)]
            source_core = span[len(prefix) : len(span) - len(suffix)]
            assert core.strip() and core != source_core
            tokens = core.split()
            given = tokens[-1] if lang == "vi" else tokens[0]
            assert classify(given) != opposite, (lang, seed)
            if lang in _STRICT_GENDER_LANGUAGES:
                assert classify(given) == gender, (lang, seed)

    @pytest.mark.parametrize("span", ["Lê Thu Hà", "Chị Lê Thu Hà"])
    def test_vietnamese_surrogate_is_family_first_without_new_honorific(self, span):
        family_names = set(VietnamesePersonProvider.last_names)
        titles = _faker_title_vocabulary("vi")
        for seed in range(12):
            surrogate = Anonymizer(lang="vi", consistent=True, seed=seed).surrogate(
                span, "PERSON"
            )
            core = surrogate.removeprefix("Chị ")

            assert core.split()[0] in family_names, seed
            assert not _starts_with_title(core, titles), seed

    @pytest.mark.parametrize("lang", _TITLE_FREE_LANGUAGES)
    def test_surrogate_gains_no_honorific_the_source_lacked(self, lang):
        titles = _faker_title_vocabulary(lang)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            surrogates = [
                Anonymizer(lang=lang, consistent=True, seed=seed).surrogate(
                    f"Synthetic Name{seed}", "PERSON"
                )
                for seed in range(10)
            ]

        for surrogate in surrogates:
            assert surrogate.split(), lang
            assert not _starts_with_title(surrogate, titles), (lang, surrogate)

    @pytest.mark.parametrize(
        ("lang", "span"),
        [("sw", "Bi."), ("th", "นางสาว"), ("ur", "جناب"), ("vi", "Bà")],
    )
    def test_honorific_only_span_is_masked(self, lang, span):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert Anonymizer(lang=lang).surrogate(span, "PERSON") == "[PERSON]"

    @pytest.mark.parametrize(
        ("lang", "span"),
        [
            ("sw", "Amina Hassan"),
            ("th", "สมหญิง ใจดี"),
            ("ur", "زہرہ بیگم"),
            ("ar", "محمد السيد"),
            ("hi", "श्रीकांत वर्मा"),
        ],
    )
    def test_span_without_honorific_reaches_generator_unchanged(
        self, lang, span, monkeypatch
    ):
        calls = []

        def baseline_generator(_faker, original, *, locale):
            calls.append(original)
            return "Baseline Surrogate"

        monkeypatch.setitem(LABEL_GENERATORS, "PERSON", baseline_generator)
        monkeypatch.setattr(anonymizer_registry, "LANGUAGE_PACK_GENERATORS", {})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            got = Anonymizer(lang=lang, consistent=True, seed=5).surrogate(
                span, "PERSON"
            )

        assert got == "Baseline Surrogate"
        assert calls == [span]

    def test_neutral_honorific_shares_the_bare_name_surrogate(self):
        bare = Anonymizer(lang="sw", consistent=True, seed=21).surrogate(
            "Amina Hassan", "PERSON"
        )
        titled = Anonymizer(lang="sw", consistent=True, seed=21).surrogate(
            "Dkt. Amina Hassan", "PERSON"
        )

        assert titled == f"Dkt. {bare}"

    def test_generator_fallback_keeps_honorific(self, monkeypatch):
        def fail_generator(*_args, **_kwargs):
            raise RuntimeError("synthetic generator failure")

        monkeypatch.setitem(LABEL_GENERATORS, "PERSON", fail_generator)
        with pytest.warns(RuntimeWarning, match="Anonymizer fallback"):
            surrogate = Anonymizer(lang="sw").surrogate("Bi. Amina Hassan", "PERSON")

        assert surrogate == "Bi. [PERSON]"

    def test_non_person_labels_are_not_peeled(self, monkeypatch):
        calls = []

        def first_name_generator(_faker, original, *, locale):
            calls.append(original)
            return "Baseline"

        monkeypatch.setitem(LABEL_GENERATORS, "FIRST_NAME", first_name_generator)

        got = Anonymizer(lang="sw").surrogate("Bi. Amina", "FIRST_NAME")

        assert got == "Baseline"
        assert calls == ["Bi. Amina"]

    @pytest.mark.parametrize(
        ("lang", "span", "honorific"),
        [
            ("ja", "田中さん", "さん"),
            ("ko", "김민준씨", "씨"),
            ("zh", "王伟先生", "先生"),
        ],
    )
    def test_cjk_spans_are_not_peeled_by_the_non_cjk_registry(
        self, lang, span, honorific
    ):
        assert not peel_person_honorifics(span, lang).has_honorific
        surrogate = Anonymizer(lang=lang, consistent=True, seed=13).surrogate(
            span, "PERSON"
        )
        assert surrogate.endswith(honorific)
        assert surrogate != span
