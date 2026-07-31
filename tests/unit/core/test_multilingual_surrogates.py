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
