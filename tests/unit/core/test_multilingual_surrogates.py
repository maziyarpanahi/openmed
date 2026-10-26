"""Script fidelity and corpus consistency gates for multilingual surrogates."""

from __future__ import annotations

import warnings

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer import locales as locale_module
from openmed.core.language_pack import get_language_pack
from openmed.core.script_detect import detect_script
from openmed.core.surrogate_vault import HMAC_SCHEME, SurrogateVault

_SCRIPT_CASES = (
    ("zh", "张伟", "Han"),
    ("hi", "अर्जुन", "Devanagari"),
    ("te", "రామ", "Telugu"),
)

_SYNTHETIC_ALPHABETS = {
    "zh": "东西南北春夏秋冬山川云月",
    "hi": "कखगघचछजझटठडढतथदधनपफबभमयरलवशषसह",
    "te": "కఖగఘచఛజఝటఠడఢతథదధనపఫబభమయరలవశషసహ",
}


@pytest.mark.parametrize(("lang", "original", "expected_script"), _SCRIPT_CASES)
def test_name_surrogate_matches_source_script_and_is_disjoint(
    lang: str,
    original: str,
    expected_script: str,
) -> None:
    surrogate = Anonymizer(lang=lang, consistent=True, seed=42).surrogate(
        original,
        "PERSON",
    )

    assert detect_script(surrogate) == expected_script
    assert surrogate != original
    assert set(surrogate).isdisjoint(original)


def test_han_surrogate_preserves_han_character_count() -> None:
    original = "欧阳明"
    surrogate = Anonymizer(lang="zh", consistent=True, seed=7).surrogate(
        original,
        "PERSON",
    )

    assert len(surrogate) == len(original)
    assert detect_script(surrogate) == "Han"


def test_telugu_native_name_path_suppresses_locale_approximation_warning() -> None:
    locale_module._warned.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        surrogate = Anonymizer(lang="te", consistent=True, seed=11).surrogate(
            "సరోజ",
            "PERSON",
        )

    assert detect_script(surrogate) == "Telugu"
    assert not [
        warning for warning in caught if "language 'te'" in str(warning.message)
    ]


def test_telugu_non_native_fallback_keeps_approximation_warning() -> None:
    locale_module._warned.clear()
    with pytest.warns(UserWarning, match="language 'te'.*no native Faker locale"):
        surrogate = Anonymizer(lang="te", consistent=True, seed=11).surrogate(
            "Saroj",
            "PERSON",
        )

    assert detect_script(surrogate) == "Latin"


@pytest.mark.parametrize(
    ("lang", "original", "label", "expected"),
    (
        ("en", "Alice Example", "PERSON", "Andrea Reed"),
        ("en", "Alice", "FIRST_NAME", "Ethan"),
        ("en", "Example", "LAST_NAME", "Martin"),
        ("fr", "Marie Dupont", "PERSON", "Lucas Clément Le Mercier"),
        ("de", "Erika Muster", "PERSON", "Univ.Prof. Dorothea Gotthard"),
    ),
)
def test_latin_faker_outputs_are_unchanged(
    lang: str,
    original: str,
    label: str,
    expected: str,
) -> None:
    assert (
        Anonymizer(lang=lang, consistent=True, seed=42).surrogate(original, label)
        == expected
    )


def test_builtin_script_providers_are_registered_through_language_packs() -> None:
    assert get_language_pack("zh").scripts == ("Han",)
    assert get_language_pack("hi").scripts == ("Devanagari",)
    assert get_language_pack("te").scripts == ("Telugu",)


def test_cross_document_consistency_gate_on_synthetic_multiscript_corpus() -> None:
    vault = SurrogateVault.in_memory("om-683-synthetic-corpus-secret")
    anonymizers = {
        lang: Anonymizer(lang=lang, consistent=True, seed=683)
        for lang in _SYNTHETIC_ALPHABETS
    }
    corpus = [
        (lang, alphabet[index : index + 3])
        for lang, alphabet in _SYNTHETIC_ALPHABETS.items()
        for index in range(0, 9, 3)
    ]

    def replace(lang: str, original: str) -> str:
        def create_surrogate(attempt: int) -> str:
            source = original if attempt == 0 else f"{original}|{attempt}"
            return anonymizers[lang].surrogate(source, "PERSON")

        return vault.get_or_create(
            original,
            label="PERSON",
            lang=lang,
            create_surrogate=create_surrogate,
        )

    expected = {(lang, name): replace(lang, name) for lang, name in corpus}
    repeated = [
        (expected[(lang, name)], replace(lang, name))
        for _note_index in range(4)
        for lang, name in corpus
    ]
    consistency = sum(first == later for first, later in repeated) / len(repeated)

    assert consistency >= 0.99
    assert len(set(expected.values())) == len(expected)
    assert all(
        entry.key.text_hash.startswith(f"{HMAC_SCHEME}:") for entry in vault.entries()
    )
    assert all(
        original not in entry.key.text_hash
        for entry in vault.entries()
        for _lang, original in corpus
    )


@st.composite
def _synthetic_multiscript_name(draw):
    lang = draw(st.sampled_from(tuple(_SYNTHETIC_ALPHABETS)))
    original = draw(
        st.text(
            alphabet=_SYNTHETIC_ALPHABETS[lang],
            min_size=1,
            max_size=8,
        )
    )
    return lang, original


@given(_synthetic_multiscript_name())
def test_multiscript_name_property_never_reuses_source_codepoints(case) -> None:
    lang, original = case
    surrogate = Anonymizer(lang=lang, consistent=True, seed=683).surrogate(
        original,
        "PERSON",
    )

    assert surrogate != original
    assert set(surrogate).isdisjoint(original)
    assert detect_script(surrogate) == detect_script(original)


# ---------------------------------------------------------------------------
# Arabic and Amharic source-script surrogates
# ---------------------------------------------------------------------------

_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_ETHIOPIC_RANGES = (
    (0x1200, 0x137F),
    (0x1380, 0x139F),
    (0x2D80, 0x2DDF),
    (0xAB00, 0xAB2F),
)
_SCRIPT_RANGES = {"Arabic": _ARABIC_RANGES, "Ethiopic": _ETHIOPIC_RANGES}

# Synthetic source surfaces. None of them is a real patient record; they are
# recombined from the curated synthetic language-pack values.
_AR_NAMES = (
    "أحمد علي حسن",
    "ليلى حسن",
    "مريم محمود",
    "يوسف إبراهيم",
)
_AR_FIRST_NAMES = ("أحمد", "ليلى")
_AR_LAST_NAMES = ("حسن", "محمود")
_AR_STREETS = (
    "شارع النيل 12",
    "طريق الملك فهد 45",
)
_AR_CITIES = (
    "القاهرة",
    "الرياض",
)
_AM_NAMES = ("ወይዘሮ አልማዝ ተስፋዬ", "ሰላም ተስፋዬ", "ዳዊት ከበደ", "ሜሮን አለሙ")
_AM_FIRST_NAMES = ("ሰላም", "ዳዊት")
_AM_LAST_NAMES = ("ተስፋዬ", "ከበደ")
_AM_STREETS = ("አዲስ አበባ ቦሌ 12", "ባሕር ዳር ቀበሌ 5")
_AM_CITIES = ("አዲስ አበባ", "ሀዋሳ")

_SOURCE_SCRIPT_CASES = (
    ("ar", "Arabic", "PERSON", _AR_NAMES),
    ("ar", "Arabic", "FIRST_NAME", _AR_FIRST_NAMES),
    ("ar", "Arabic", "LAST_NAME", _AR_LAST_NAMES),
    ("ar", "Arabic", "STREET_ADDRESS", _AR_STREETS),
    ("ar", "Arabic", "CITY", _AR_CITIES),
    ("ar", "Arabic", "LOCATION", _AR_CITIES),
    ("am", "Ethiopic", "PERSON", _AM_NAMES),
    ("am", "Ethiopic", "FIRST_NAME", _AM_FIRST_NAMES),
    ("am", "Ethiopic", "LAST_NAME", _AM_LAST_NAMES),
    ("am", "Ethiopic", "STREET_ADDRESS", _AM_STREETS),
    ("am", "Ethiopic", "CITY", _AM_CITIES),
    ("am", "Ethiopic", "LOCATION", _AM_CITIES),
)

_SAMPLE_SEEDS = range(25)


def _in_script(value: str, script: str) -> bool:
    """Return whether every letter of ``value`` belongs to ``script``."""

    letters = [character for character in value if character.isalpha()]
    return bool(letters) and all(
        any(start <= ord(character) <= end for start, end in _SCRIPT_RANGES[script])
        for character in letters
    )


def _sample(lang: str, label: str, sources: tuple[str, ...]) -> list[tuple]:
    return [
        (
            source,
            Anonymizer(lang=lang, consistent=True, seed=seed).surrogate(source, label),
        )
        for seed in _SAMPLE_SEEDS
        for source in sources
    ]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(("lang", "script", "label", "sources"), _SOURCE_SCRIPT_CASES)
def test_arabic_and_amharic_surrogates_stay_in_source_script(
    lang: str,
    script: str,
    label: str,
    sources: tuple[str, ...],
) -> None:
    assert all(_in_script(source, script) for source in sources)

    sample = _sample(lang, label, sources)
    in_script = sum(_in_script(surrogate, script) for _source, surrogate in sample)

    assert in_script / len(sample) >= 0.9
    assert all(surrogate != source for source, surrogate in sample)
    assert all(surrogate.strip() for _source, surrogate in sample)
    assert len({surrogate for _source, surrogate in sample}) > 1


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(("lang", "script", "label", "sources"), _SOURCE_SCRIPT_CASES)
def test_arabic_and_amharic_surrogates_are_deterministic_per_seed(
    lang: str,
    script: str,
    label: str,
    sources: tuple[str, ...],
) -> None:
    assert _sample(lang, label, sources) == _sample(lang, label, sources)


@pytest.mark.parametrize(
    ("lang", "original"),
    (
        ("ar", "أحمد علي حسن"),
        ("am", "ወይዘሮ አልማዝ ተስፋዬ"),
    ),
)
def test_arabic_and_amharic_names_do_not_reuse_source_tokens(
    lang: str,
    original: str,
) -> None:
    for seed in _SAMPLE_SEEDS:
        surrogate = Anonymizer(lang=lang, consistent=True, seed=seed).surrogate(
            original,
            "PERSON",
        )
        assert set(surrogate.split()).isdisjoint(original.split())


def test_arabic_street_address_does_not_reuse_source_street_or_number() -> None:
    original = "شارع النيل 12"
    for seed in _SAMPLE_SEEDS:
        surrogate = Anonymizer(lang="ar", consistent=True, seed=seed).surrogate(
            original,
            "STREET_ADDRESS",
        )
        assert "النيل" not in surrogate
        assert "12" not in surrogate.split()


def test_amharic_native_name_path_suppresses_locale_approximation_warning() -> None:
    locale_module._warned.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        surrogate = Anonymizer(lang="am", consistent=True, seed=11).surrogate(
            "ሰላም ተስፋዬ",
            "PERSON",
        )

    assert _in_script(surrogate, "Ethiopic")
    assert not [
        warning for warning in caught if "language 'am'" in str(warning.message)
    ]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("lang", ("ar", "am"))
@pytest.mark.parametrize("label", ("PERSON", "STREET_ADDRESS", "CITY"))
def test_latin_sources_in_arabic_and_amharic_keep_latin_surrogates(
    lang: str,
    label: str,
) -> None:
    # Negative control: routing follows the source script, so a Latin span in
    # an Arabic or Amharic note is not rewritten into a different script.
    original = {
        "PERSON": "Alice Example",
        "STREET_ADDRESS": "12 Example Road",
        "CITY": "Exampleton",
    }[label]
    surrogate = Anonymizer(lang=lang, consistent=True, seed=5).surrogate(
        original,
        label,
    )

    assert detect_script(surrogate) == "Latin"
    assert surrogate != original


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_arabic_regional_override_keeps_native_regional_faker_names() -> None:
    from faker.providers.person.ar_DZ import Provider as AlgerianPersonProvider

    first_names = set(AlgerianPersonProvider.first_names)
    last_names = set(AlgerianPersonProvider.last_names)
    for seed in _SAMPLE_SEEDS:
        anonymizer = Anonymizer(lang="ar", locale="ar-DZ", consistent=True, seed=seed)
        surrogate = anonymizer.surrogate(_AR_NAMES[0], "PERSON")
        assert any(
            surrogate == f"{given} {family}"
            for given in first_names
            for family in last_names
        )
        assert anonymizer.surrogate(_AR_FIRST_NAMES[0], "FIRST_NAME") in first_names


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("locale", ("ar-DZ", "ar-SA", "ar-JO", "ar-AE"))
def test_arabic_regional_overrides_stay_in_arabic_script(locale: str) -> None:
    for seed in _SAMPLE_SEEDS:
        anonymizer = Anonymizer(lang="ar", locale=locale, consistent=True, seed=seed)
        for label, original in (
            ("PERSON", _AR_NAMES[0]),
            ("STREET_ADDRESS", _AR_STREETS[0]),
            ("CITY", _AR_CITIES[0]),
        ):
            surrogate = anonymizer.surrogate(original, label)
            assert _in_script(surrogate, "Arabic"), (locale, label, surrogate)
            assert surrogate != original


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    ("lang", "original"),
    (("ar", "+20 10 1234 5678"), ("am", "+251 911 234 567"), ("am", "0911 765 432")),
)
def test_arabic_and_amharic_phone_surrogates_keep_their_format(
    lang: str,
    original: str,
) -> None:
    for seed in _SAMPLE_SEEDS:
        surrogate = Anonymizer(lang=lang, consistent=True, seed=seed).surrogate(
            original,
            "PHONE",
        )
        assert surrogate != original
        assert len(surrogate) == len(original)
        assert [not character.isdigit() for character in surrogate] == [
            not character.isdigit() for character in original
        ]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    ("lang", "original", "validator_name"),
    (
        ("ar", "29801011234567", "validate_egyptian_national_id"),
        ("am", "123456789012", "validate_ethiopia_fayda"),
    ),
)
def test_arabic_and_amharic_national_id_surrogates_still_validate(
    lang: str,
    original: str,
    validator_name: str,
) -> None:
    from openmed.core import pii_i18n

    validator = getattr(pii_i18n, validator_name)
    for seed in _SAMPLE_SEEDS:
        surrogate = Anonymizer(lang=lang, consistent=True, seed=seed).surrogate(
            original,
            "national_id",
        )
        assert surrogate != original
        assert validator(surrogate)


@pytest.mark.parametrize(
    ("lang", "key", "label", "script"),
    (
        ("am", "FIRST_NAME", "FIRST_NAME", "Ethiopic"),
        ("am", "LAST_NAME", "LAST_NAME", "Ethiopic"),
        ("am", "LOCATION", "CITY", "Ethiopic"),
    ),
)
def test_exhausted_curated_inventory_does_not_reuse_source_tokens(
    lang, key, label, script
):
    from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

    original = " ".join(LANGUAGE_FAKE_DATA[lang][key])
    first = Anonymizer(lang=lang, consistent=True, seed=7).surrogate(original, label)
    second = Anonymizer(lang=lang, consistent=True, seed=7).surrogate(original, label)
    assert first == second
    assert _in_script(first, script)
    assert set(first.split()).isdisjoint(original.split())


def test_exhausted_address_number_inventory_fails_without_source_values():
    from faker import Faker

    from openmed.core.anonymizer.providers.script_places import _draw_number

    with pytest.raises(
        ValueError, match="^synthetic address number inventory exhausted$"
    ):
        _draw_number(Faker(), "1 2 3", bounds=(1, 3))
