"""Aksara grammar gates for Devanagari and Telugu name surrogates."""

from __future__ import annotations

import hashlib
import random
import unicodedata
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.core import labels as L
from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.providers.aksara_names import (
    AKSARA_INVENTORIES,
    DEVANAGARI_AKSARA_INVENTORY,
    TELUGU_AKSARA_INVENTORY,
    generate_aksara_name,
    generate_devanagari_aksara_name,
    generate_telugu_aksara_name,
    is_well_formed_aksara_name,
    iter_aksara,
)
from openmed.core.anonymizer.registry import (
    LANGUAGE_PACK_GENERATORS,
    _gen_marathi_first_name,
    _gen_marathi_last_name,
    _gen_marathi_middle_name,
    _gen_marathi_person,
)
from openmed.core.decoding.spans import iter_grapheme_cluster_spans
from openmed.core.script_detect import detect_script
from openmed.processing.outputs import EntityPrediction, PredictionResult

_NAME_LABELS = (L.PERSON, L.FIRST_NAME, L.MIDDLE_NAME, L.LAST_NAME)
_SEEDS = tuple(range(40))

# Synthetic name shapes: one to three tokens, initials, and a hyphenated part.
_SYNTHETIC_NAMES = (
    ("hi", "Devanagari", "अनीता शर्मा"),
    ("hi", "Devanagari", "राकेश कुमार वर्मा"),
    ("hi", "Devanagari", "अर्जुन"),
    ("hi", "Devanagari", "उदाहरण व्यक्ति"),
    ("hi", "Devanagari", "ए. के. मिश्रा"),
    ("hi", "Devanagari", "राम-प्रसाद गुप्ता"),
    ("te", "Telugu", "సరోజ రెడ్డి"),
    ("te", "Telugu", "వెంకటేశ్వర్లు"),
    ("te", "Telugu", "రామ"),
    ("te", "Telugu", "కె. లక్ష్మి దేవి"),
)

_SCRIPT_BLOCKS = {
    "Devanagari": (0x0900, 0x097F),
    "Telugu": (0x0C00, 0x0C7F),
}
_CONSONANT_BLOCKS = {
    "Devanagari": (0x0915, 0x0939),
    "Telugu": (0x0C15, 0x0C39),
}
_INDEPENDENT_VOWEL_BLOCKS = {
    "Devanagari": (0x0904, 0x0914),
    "Telugu": (0x0C05, 0x0C14),
}

# SHA-256 of seeded surrogates recorded on master before this generator
# existed. Han, Gujarati, and Latin-script spans in Hindi and Telugu
# documents must not change.
_REGRESSION_CASES = {
    "han": (
        ("zh", "PERSON", "欧阳明"),
        ("zh", "FIRST_NAME", "明"),
        ("zh", "LAST_NAME", "欧阳"),
        ("zh", "MIDDLE_NAME", "明华"),
    ),
    "gujarati": (
        ("gu", "PERSON", "કિરણ પટેલ"),
        ("gu", "FIRST_NAME", "રમેશભાઈ"),
        ("gu", "PERSON", "શ્રીમતી કિરણબેન"),
    ),
    "latin_fallback": (
        ("hi", "PERSON", "Anita Sharma"),
        ("te", "PERSON", "Saroj Kumar"),
        ("te", "FIRST_NAME", "Saroj"),
    ),
}
_REGRESSION_DIGESTS = {
    "han": "c37ce028012927fd4909cf7b83d15f5e771ca367491e204f0b39743f28535a28",
    "gujarati": "056820856553f1bf55660c8b369edb6d5ff9cd691a14e2e9f3f047e5f47ce29c",
    "latin_fallback": (
        "59409432b8307a259b382289c4852710e687072525856522f95e478bf4051d8d"
    ),
}


def _in_block(character: str, block: tuple[int, int]) -> bool:
    return block[0] <= ord(character) <= block[1]


def _script_codepoints(value: str, script: str) -> set[str]:
    return {char for char in value if _in_block(char, _SCRIPT_BLOCKS[script])}


def _consonants(value: str, script: str) -> set[str]:
    return {char for char in value if _in_block(char, _CONSONANT_BLOCKS[script])}


def _surrogate(lang: str, original: str, label: str = "PERSON", seed: int = 0):
    return Anonymizer(lang=lang, consistent=True, seed=seed).surrogate(
        original,
        label,
    )


def _seeded_faker(seed: int) -> SimpleNamespace:
    return SimpleNamespace(random=random.Random(seed))


def _looks_like_letter_noise(value: str) -> bool:
    """Flag four or more base letters with no vowel sign, virama, or anusvara."""

    return any(
        sum(unicodedata.category(char) == "Lo" for char in token) >= 4
        and not any(unicodedata.category(char).startswith("M") for char in token)
        for token in value.split()
    )


def _surrogate_digest(cases) -> str:
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lang, label, original in cases:
            for seed in (3, 11, 42):
                surrogate = _surrogate(lang, original, label, seed)
                rows.append(f"{lang}\t{label}\t{seed}\t{surrogate}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


@pytest.mark.parametrize(("lang", "script", "original"), _SYNTHETIC_NAMES)
def test_every_surrogate_is_well_formed_aksara_with_source_token_count(
    lang: str,
    script: str,
    original: str,
) -> None:
    for seed in _SEEDS:
        surrogate = _surrogate(lang, original, seed=seed)

        assert is_well_formed_aksara_name(surrogate, script=script)
        assert len(surrogate.split()) == len(original.split())
        assert detect_script(surrogate) == script
        assert unicodedata.normalize("NFC", surrogate) == surrogate
        assert not _looks_like_letter_noise(surrogate)


@pytest.mark.parametrize(("lang", "script", "original"), _SYNTHETIC_NAMES)
def test_surrogate_reuses_no_source_code_point_and_never_equals_source(
    lang: str,
    script: str,
    original: str,
) -> None:
    for seed in _SEEDS:
        surrogate = _surrogate(lang, original, seed=seed)

        assert surrogate != original
        assert _consonants(surrogate, script).isdisjoint(_consonants(original, script))
        assert _script_codepoints(surrogate, script).isdisjoint(original)
        assert not set(original.split()) & set(surrogate.split())


@pytest.mark.parametrize("label", _NAME_LABELS)
@pytest.mark.parametrize(
    ("lang", "script", "original"),
    (("hi", "Devanagari", "सुनीता देवी"), ("te", "Telugu", "సరోజ రెడ్డి")),
)
def test_every_name_label_routes_to_the_aksara_grammar(
    lang: str,
    script: str,
    original: str,
    label: str,
) -> None:
    surrogate = _surrogate(lang, original, label, seed=5)

    assert is_well_formed_aksara_name(surrogate, script=script)
    assert len(surrogate.split()) == 2
    assert _consonants(surrogate, script).isdisjoint(original)


@pytest.mark.parametrize(("lang", "script", "original"), _SYNTHETIC_NAMES)
def test_tokens_keep_order_initials_hyphens_and_aksara_shape(
    lang: str,
    script: str,
    original: str,
) -> None:
    inventory = AKSARA_INVENTORIES[script]
    surrogate = _surrogate(lang, original, seed=17)

    for source_token, token in zip(original.split(), surrogate.split(), strict=True):
        assert token.endswith(".") == source_token.endswith(".")
        assert token.count("-") == source_token.count("-")
        for source_part, part in zip(
            source_token.rstrip(".").split("-"),
            token.rstrip(".").split("-"),
            strict=True,
        ):
            source_size = sum(1 for _cluster in iter_aksara(source_part))
            if source_token.endswith(".") and source_size == 1:
                expected = 1
            else:
                expected = min(max(source_size, 2), inventory.max_token_aksara)
            assert sum(1 for _cluster in iter_aksara(part)) == expected


@pytest.mark.parametrize(("lang", "script", "original"), _SYNTHETIC_NAMES)
def test_aksara_follow_the_grapheme_engine_and_word_initial_vowels(
    lang: str,
    script: str,
    original: str,
) -> None:
    vowel_block = _INDEPENDENT_VOWEL_BLOCKS[script]
    virama = AKSARA_INVENTORIES[script].virama
    for seed in _SEEDS[:10]:
        surrogate = _surrogate(lang, original, seed=seed)
        for token in surrogate.split():
            for part in token.rstrip(".").split("-"):
                clusters = [
                    part[start:end] for start, end in iter_grapheme_cluster_spans(part)
                ]
                assert "".join(clusters) == part
                assert list(iter_aksara(part)) == clusters
                for cluster in clusters[1:]:
                    assert not _in_block(cluster[0], vowel_block)
                # Adjacent single-consonant aksara never repeat their consonant.
                for left, right in zip(clusters, clusters[1:], strict=False):
                    if virama not in left + right:
                        assert left[0] != right[0]


@pytest.mark.parametrize(("lang", "_script", "original"), _SYNTHETIC_NAMES)
def test_seeded_surrogates_are_reproducible(
    lang: str,
    _script: str,
    original: str,
) -> None:
    first = [_surrogate(lang, original, seed=seed) for seed in _SEEDS[:8]]
    second = [_surrogate(lang, original, seed=seed) for seed in _SEEDS[:8]]

    assert first == second
    assert len(set(first)) > 1


@pytest.mark.parametrize(
    ("generator", "original"),
    (
        (generate_devanagari_aksara_name, "अनीता शर्मा"),
        (generate_telugu_aksara_name, "సరోజ రెడ్డి"),
    ),
)
def test_generator_depends_only_on_the_seeded_random_stream(
    generator,
    original: str,
) -> None:
    first = generator(_seeded_faker(99), original, locale="unused")
    second = generator(_seeded_faker(99), original, locale="unused")
    other = {generator(_seeded_faker(seed), original, locale="x") for seed in _SEEDS}

    assert first == second
    assert len(other) > len(_SEEDS) // 2


def test_hindi_two_token_name_from_the_issue_reads_as_two_names() -> None:
    surrogate = Anonymizer(lang="hi", consistent=True, seed=11).surrogate(
        "अनीता शर्मा",
        "person",
        lang="hi",
    )

    assert len(surrogate.split()) == 2
    assert is_well_formed_aksara_name(surrogate, script="Devanagari")
    assert not _looks_like_letter_noise(surrogate)


@pytest.mark.parametrize(
    ("letter_noise", "aksara", "original", "script"),
    (
        # Seeded letter-sampling output that master produced for these sources.
        (
            "छहऱषझङथएजक",
            generate_devanagari_aksara_name,
            "अनीता शर्मा",
            "Devanagari",
        ),
        ("తఝయలభఛఴఎబఓ", generate_telugu_aksara_name, "సరోజ రెడ్డి", "Telugu"),
    ),
)
def test_letter_sampling_negative_control_fails_the_shape_gate(
    letter_noise: str,
    aksara,
    original: str,
    script: str,
) -> None:
    aksara_values = [
        aksara(_seeded_faker(seed), original, locale="x") for seed in _SEEDS[:10]
    ]

    assert len(letter_noise.split()) != len(original.split())
    assert _looks_like_letter_noise(letter_noise)
    assert all(len(value.split()) == 2 for value in aksara_values)
    assert not any(_looks_like_letter_noise(value) for value in aksara_values)
    assert all(
        is_well_formed_aksara_name(value, script=script) for value in aksara_values
    )


@pytest.mark.parametrize(
    ("value", "script"),
    (
        ("ािक", "Devanagari"),
        ("काि", "Devanagari"),
        ("अा", "Devanagari"),
        ("क्", "Devanagari"),
        ("राम1", "Devanagari"),
        ("రామ", "Devanagari"),
        ("ాక", "Telugu"),
        ("కాి", "Telugu"),
        ("క్", "Telugu"),
        ("", "Telugu"),
        ("   ", "Devanagari"),
    ),
)
def test_well_formed_validator_rejects_broken_aksara(value: str, script: str) -> None:
    assert not is_well_formed_aksara_name(value, script=script)


def test_well_formed_validator_accepts_conjuncts_signs_and_modifiers() -> None:
    assert is_well_formed_aksara_name("क्षितिज प्रसाद", script="Devanagari")
    assert is_well_formed_aksara_name("दुःशील संदीप", script="Devanagari")
    assert is_well_formed_aksara_name("ए. के. राम-लाल", script="Devanagari")
    assert is_well_formed_aksara_name("లక్ష్మి సుందరం", script="Telugu")
    with pytest.raises(ValueError, match="no aksara inventory"):
        is_well_formed_aksara_name("abc", script="Latin")


def test_inventories_hold_syllable_parts_not_names() -> None:
    for inventory in (DEVANAGARI_AKSARA_INVENTORY, TELUGU_AKSARA_INVENTORY):
        singles = (
            inventory.onset_consonants
            + inventory.heavy_consonants
            + inventory.initial_vowels
            + inventory.vowel_signs
            + inventory.initial_vowel_signs
            + inventory.final_vowel_signs
            + inventory.initialism_signs
        )
        assert all(len(text) == 1 and weight > 0 for text, weight in singles)
        for cluster, weight in inventory.initial_clusters + inventory.medial_clusters:
            assert weight > 0
            assert len(cluster) == 3
            assert cluster[1] == inventory.virama
            assert inventory.aksara_pattern.fullmatch(cluster)
        assert all(len(token) <= 8 for token in inventory.title_tokens)


def test_source_without_letters_still_gets_a_distinct_name() -> None:
    surrogate = generate_aksara_name(
        _seeded_faker(3),
        " - ",
        inventory=DEVANAGARI_AKSARA_INVENTORY,
    )

    assert surrogate != "-"
    assert len(surrogate.split()) == 1
    assert is_well_formed_aksara_name(surrogate, script="Devanagari")


def test_latin_tokens_in_a_script_span_are_replaced_in_script() -> None:
    original = "A. Sharma राकेश"
    surrogate = generate_devanagari_aksara_name(
        _seeded_faker(8),
        original,
        locale="hi_IN",
    )

    assert len(surrogate.split()) == 3
    assert surrogate.split()[0].endswith(".")
    assert is_well_formed_aksara_name(surrogate, script="Devanagari")
    assert "Sharma" not in surrogate


def test_exhausted_inventory_fails_without_echoing_the_source() -> None:
    inventory = DEVANAGARI_AKSARA_INVENTORY
    original = "".join(
        text
        for text, _weight in inventory.onset_consonants + inventory.heavy_consonants
    )

    with pytest.raises(ValueError) as caught:
        generate_aksara_name(_seeded_faker(1), original, inventory=inventory)

    assert original not in str(caught.value)
    assert not set(str(caught.value)) & set(original)


def test_aksara_generators_are_registered_only_for_hindi_and_telugu() -> None:
    aksara_generators = {generate_devanagari_aksara_name, generate_telugu_aksara_name}
    routed = {
        key
        for key, generator in LANGUAGE_PACK_GENERATORS.items()
        if generator in aksara_generators
    }

    assert routed == {
        *(("hi", "Devanagari", label) for label in _NAME_LABELS),
        *(("te", "Telugu", label) for label in _NAME_LABELS),
    }
    assert LANGUAGE_PACK_GENERATORS[("hi", "Devanagari", L.PERSON)] is (
        generate_devanagari_aksara_name
    )
    assert LANGUAGE_PACK_GENERATORS[("te", "Telugu", L.PERSON)] is (
        generate_telugu_aksara_name
    )
    # Marathi keeps its native three-part generators.
    assert LANGUAGE_PACK_GENERATORS[("mr", "Devanagari", L.PERSON)] is (
        _gen_marathi_person
    )
    assert LANGUAGE_PACK_GENERATORS[("mr", "Devanagari", L.FIRST_NAME)] is (
        _gen_marathi_first_name
    )
    assert LANGUAGE_PACK_GENERATORS[("mr", "Devanagari", L.MIDDLE_NAME)] is (
        _gen_marathi_middle_name
    )
    assert LANGUAGE_PACK_GENERATORS[("mr", "Devanagari", L.LAST_NAME)] is (
        _gen_marathi_last_name
    )


@pytest.mark.parametrize("group", sorted(_REGRESSION_CASES))
def test_han_gujarati_and_latin_fallback_outputs_are_unchanged(group: str) -> None:
    assert _surrogate_digest(_REGRESSION_CASES[group]) == _REGRESSION_DIGESTS[group]


@pytest.mark.parametrize(
    ("lang", "script", "note", "name"),
    (
        (
            "hi",
            "Devanagari",
            "रोगी अनीता शर्मा को बुखार है।",
            "अनीता शर्मा",
        ),
        (
            "te",
            "Telugu",
            "రోగి సరోజ రెడ్డి జ్వరంతో వచ్చారు.",
            "సరోజ రెడ్డి",
        ),
    ),
)
def test_deidentify_replace_keeps_a_multi_token_pronounceable_name(
    lang: str,
    script: str,
    note: str,
    name: str,
) -> None:
    from openmed.core.pii import deidentify

    start = note.index(name)
    prediction = PredictionResult(
        text=note,
        entities=[
            EntityPrediction(
                text=name,
                label="NAME",
                start=start,
                end=start + len(name),
                confidence=0.95,
            )
        ],
        model_name="mock",
        timestamp="2026-01-01T00:00:00",
    )

    with patch("openmed.core.pii.extract_pii", return_value=prediction):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = deidentify(note, method="replace", lang=lang, seed=31)
            second = deidentify(note, method="replace", lang=lang, seed=31)

    surrogate = first.pii_entities[0].redacted_text
    assert first.deidentified_text == second.deidentified_text
    assert first.deidentified_text == note.replace(name, surrogate)
    assert len(surrogate.split()) == len(name.split())
    assert is_well_formed_aksara_name(surrogate, script=script)
    assert _consonants(surrogate, script).isdisjoint(name)


_DEVANAGARI_AKSARA = st.sampled_from(
    tuple(
        consonant + sign
        for consonant in "कगतनपमरलवसह"
        for sign in ("", "ा", "ि", "ी", "ु", "े", "ो", "ं")
    )
)
_TELUGU_AKSARA = st.sampled_from(
    tuple(
        consonant + sign
        for consonant in "కగతనపమరలవసహ"
        for sign in ("", "ా", "ి", "ీ", "ు", "ె", "ో", "ం")
    )
)


@st.composite
def _synthetic_aksara_name(draw):
    lang, script, alphabet = draw(
        st.sampled_from(
            (
                ("hi", "Devanagari", _DEVANAGARI_AKSARA),
                ("te", "Telugu", _TELUGU_AKSARA),
            )
        )
    )
    tokens = draw(
        st.lists(
            st.lists(alphabet, min_size=1, max_size=6).map("".join),
            min_size=1,
            max_size=3,
        )
    )
    return lang, script, " ".join(tokens)


@settings(max_examples=60, deadline=None)
@given(_synthetic_aksara_name(), st.integers(min_value=0, max_value=2**16))
def test_property_aksara_surrogates_hold_every_invariant(case, seed: int) -> None:
    lang, script, original = case
    surrogate = _surrogate(lang, original, seed=seed)

    assert surrogate != original
    assert is_well_formed_aksara_name(surrogate, script=script)
    assert len(surrogate.split()) == len(original.split())
    assert _script_codepoints(surrogate, script).isdisjoint(original)
    assert surrogate == _surrogate(lang, original, seed=seed)
