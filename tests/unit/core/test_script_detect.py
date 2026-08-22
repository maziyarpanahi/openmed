import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.core.decoding import (
    iter_grapheme_cluster_spans,
    snap_span_to_grapheme_boundaries,
)
from openmed.core.language_pack_catalog import (
    DEFAULT_PII_MODELS as BUILTIN_DEFAULT_PII_MODELS,
)
from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    NATIONAL_ID_ONLY_LANGUAGES,
    OPTIONAL_PII_MODEL,
    SUPPORTED_LANGUAGES,
    USER_SUPPLIED_MODEL_LANGUAGES,
)
from openmed.core.script_detect import (
    CONFUSABLE_DATA_LICENSE,
    CONFUSABLE_DATA_URL,
    CONFUSABLE_DATA_VERSION,
    INDIC_SCRIPTS,
    SCRIPT_LANGUAGE_HINTS,
    SCRIPT_NORMALIZERS,
    SCRIPT_NUMERAL_SETS,
    SUPPORTED_SCRIPTS,
    UNKNOWN_SCRIPT,
    ScriptRun,
    candidate_languages_for_script,
    candidate_languages_for_text,
    confusable_skeleton,
    detect_mixed_script,
    detect_script,
    is_han_dominant,
    mixed_script_spans,
    normalize_for_pii_detection,
    normalizer_for_script,
    numeral_set_for_script,
    segment_by_script,
    urdu_language_evidence,
)
from openmed.processing.text import (
    INDIC_SCRIPTS as PROCESSING_INDIC_SCRIPTS,
)

# Synthetic Arabic-script material, built from code points so no sample is
# pasted from real text. ``_ARABIC_STEM`` uses only letters shared by Arabic,
# Persian and Urdu, so on its own it carries no language evidence.
_ARABIC_STEM = "".join(chr(codepoint) for codepoint in (0x0627, 0x0644, 0x0645, 0x0631))
_URDU_LETTERS = tuple(
    chr(codepoint) for codepoint in (0x0679, 0x0688, 0x0691, 0x06BA, 0x06BE, 0x06D2)
)
# Persian-exclusive peh/tcheh/jeh/gaf plus farsi yeh and heh: none of these is
# an Urdu-exclusive letter, so Persian must never score.
_PERSIAN_LETTERS = "".join(
    chr(codepoint) for codepoint in (0x067E, 0x0686, 0x0698, 0x06AF, 0x06CC, 0x0647)
)
_EXTENDED_ARABIC_INDIC_DIGITS = "".join(
    chr(codepoint) for codepoint in range(0x06F0, 0x06FA)
)
_ARABIC_INDIC_DIGITS = "".join(chr(codepoint) for codepoint in range(0x0660, 0x066A))
_ARABIC_PRESENTATION_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


@st.composite
def _yoruba_cluster_spans(draw):
    clusters = draw(
        st.lists(
            st.sampled_from(("ọ́", "ẹ̀", "ṣ", "á", "ì", "ń", "ǹ")),
            min_size=1,
            max_size=12,
        )
    )
    use_nfd = draw(
        st.lists(st.booleans(), min_size=len(clusters), max_size=len(clusters))
    )
    rendered = [
        unicodedata.normalize("NFD" if decomposed else "NFC", cluster)
        for cluster, decomposed in zip(clusters, use_nfd, strict=True)
    ]
    span_start = draw(st.integers(min_value=0, max_value=len(rendered) - 1))
    span_end = draw(st.integers(min_value=span_start + 1, max_value=len(rendered)))
    return rendered, span_start, span_end


def _assert_offsets_cover_text(
    segments: list[tuple[int, int, str]],
    text: str,
) -> None:
    cursor = 0
    for start, end, script in segments:
        assert script
        assert start == cursor
        assert start < end
        assert text[start:end]
        cursor = end
    assert cursor == len(text)


def test_detect_script_classifies_single_script_samples():
    samples = {
        "Patient John Smith": "Latin",
        "المريض أحمد علي": "Arabic",
        "ታካሚ ሰላም ተስፋዬ": "Ethiopic",
        "患者 佐藤花子": "Han",
        "かな カタカナ": "Hiragana/Katakana",
        "환자 김민수": "Hangul",
        "Пациент Иван": "Cyrillic",
        "मरीज़ अनिता शर्मा": "Devanagari",
        "রোগী অনিতা": "Bengali",
        "ਮਰੀਜ਼ ਅਨੀਤਾ": "Gurmukhi",
        "દર્દી અનીતા": "Gujarati",
        "ରୋଗୀ ଅନିତା": "Odia",
        "நோயாளி அனிதா": "Tamil",
        "రోగి సీత రెడ్డి": "Telugu",
        "ರೋಗಿ ಅನಿತಾ": "Kannada",
        "രോഗി അനിത": "Malayalam",
        "Ασθενής Νίκος": "Greek",
        "מטופל דוד כהן": "Hebrew",
        "ผู้ป่วย สมชาย": "Thai",
    }

    for text, script in samples.items():
        assert detect_script(text) == script


def test_detect_script_ignores_neutral_characters():
    assert detect_script("  MRN-12345  ") == "Latin"
    assert detect_script("12345 / --") == UNKNOWN_SCRIPT


def test_segment_by_script_mixed_latin_arabic_offsets_cover_text():
    text = "Patient Ahmad راجع العيادة 5mg"
    segments = list(segment_by_script(text))

    _assert_offsets_cover_text(segments, text)
    assert [script for _, _, script in segments] == ["Latin", "Arabic", "Latin"]
    assert "".join(text[start:end] for start, end, _ in segments) == text


def test_segment_by_script_mixed_latin_han_offsets_cover_text():
    text = "MRN 42 患者 佐藤 visited"
    segments = list(segment_by_script(text))

    _assert_offsets_cover_text(segments, text)
    assert [script for _, _, script in segments] == ["Latin", "Han", "Latin"]
    assert "".join(text[start:end] for start, end, _ in segments) == text


def test_detect_script_covers_all_ethiopic_unicode_blocks():
    samples = ("ሀ", "ᎀ", "ⶀ", "ꬁ", "𞟠")

    assert all(detect_script(char) == "Ethiopic" for char in samples)


def test_segment_by_script_mixed_amharic_latin_has_exact_offsets():
    text = "ታካሚ Selam፡ ቀጠሮ"

    assert list(segment_by_script(text)) == [
        (0, 4, "Ethiopic"),
        (4, 11, "Latin"),
        (11, 14, "Ethiopic"),
    ]


def test_script_language_hints_cover_detectable_scripts():
    expected_scripts = set(SUPPORTED_SCRIPTS) | {UNKNOWN_SCRIPT}
    routing_languages = (
        SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES | USER_SUPPLIED_MODEL_LANGUAGES
    )

    assert expected_scripts <= set(SCRIPT_LANGUAGE_HINTS)
    for script in expected_scripts:
        hints = candidate_languages_for_script(script)
        assert hints
        assert set(hints) <= routing_languages


def test_indic_and_arabic_script_language_hints_are_exact():
    expected_hints = {
        "Devanagari": ("hi", "mr", "ne"),
        "Bengali": ("bn", "as"),
        "Gurmukhi": ("pa",),
        "Gujarati": ("gu",),
        "Odia": ("or",),
        "Tamil": ("ta",),
        "Telugu": ("te",),
        "Kannada": ("kn",),
        "Malayalam": ("ml",),
        "Arabic": ("ar", "ha", "ur"),
    }

    for script, languages in expected_hints.items():
        assert candidate_languages_for_script(script) == languages


def test_routing_only_languages_do_not_claim_bundled_models():
    expected_languages = {
        "gu",
        "kn",
        "ml",
        "ne",
        "pa",
        "ur",
    }

    assert USER_SUPPLIED_MODEL_LANGUAGES == expected_languages
    assert USER_SUPPLIED_MODEL_LANGUAGES.isdisjoint(BUILTIN_DEFAULT_PII_MODELS)
    for language in expected_languages - {"ne", "ur"}:
        assert DEFAULT_PII_MODELS[language] == OPTIONAL_PII_MODEL
    assert candidate_languages_for_script("Latin") == (
        "en",
        "fr",
        "de",
        "it",
        "es",
        "nl",
        "pt",
        "tr",
        "cs",
        "sw",
        "ig",
        "yo",
        "zu",
        "xh",
    )


@settings(deadline=None)
@given(case=_yoruba_cluster_spans())
def test_yoruba_normalization_remaps_whole_nfc_nfd_and_mixed_clusters(case):
    clusters, selected_start, selected_end = case
    prefix = unicodedata.normalize("NFD", "Àkọsílẹ̀: ")
    suffix = unicodedata.normalize("NFC", " parí.")
    before = "".join(clusters[:selected_start])
    selected = "".join(clusters[selected_start:selected_end])
    text = prefix + "".join(clusters) + suffix

    normalized = normalize_for_pii_detection(text)
    normalized_start = len(normalize_for_pii_detection(prefix + before).text)
    normalized_end = len(normalize_for_pii_detection(prefix + before + selected).text)
    original_start, original_end = normalized.remap_span(
        normalized_start,
        normalized_end,
    )

    assert text[original_start:original_end] == selected
    assert not unicodedata.category(text[original_start]).startswith("M")
    assert original_end == len(text) or not unicodedata.category(
        text[original_end]
    ).startswith("M")


def test_snap_span_expands_both_sides_of_decomposed_yoruba_clusters():
    text = unicodedata.normalize("NFD", "Bọ́láńlé Adébáyọ̀")
    expected_start = text.index(unicodedata.normalize("NFD", "ọ́"))
    expected_end = len(text)
    inside_first_marks = expected_start + 1
    before_final_marks = expected_end - 2

    assert snap_span_to_grapheme_boundaries(
        inside_first_marks,
        before_final_marks,
        text,
    ) == (expected_start, expected_end)


def test_han_script_routes_to_chinese_candidate_language():
    text = "患者王芳因心房颤动入院"

    script = detect_script(text)

    assert script == "Han"
    assert candidate_languages_for_script(script)[0] == "zh"


def test_han_dominance_detection_supports_language_routing():
    assert is_han_dominant("患者王芳因心房颤动入院")
    assert is_han_dominant("患者A")
    assert not is_han_dominant("患者AB")
    assert not is_han_dominant("Patient John Smith")


def test_cee_script_language_hints_route_to_native_packs():
    assert "cs" in candidate_languages_for_script("Latin")
    assert candidate_languages_for_script("Cyrillic") == ("ru", "uk")
    assert candidate_languages_for_script("Greek") == ("el",)


def test_urdu_exclusive_letters_are_the_only_letter_evidence():
    assert urdu_language_evidence(_ARABIC_STEM) == 0
    for letter in _URDU_LETTERS:
        assert urdu_language_evidence(_ARABIC_STEM + letter) == 1
    assert urdu_language_evidence(_ARABIC_STEM + "".join(_URDU_LETTERS)) == len(
        _URDU_LETTERS
    )


def test_persian_text_yields_no_urdu_evidence():
    # Persian shares peh/tcheh/jeh/gaf with Urdu but none of the six
    # Urdu-exclusive letters, and it uses the same extended Arabic-Indic
    # digits, so neither signal may route Persian to Urdu.
    assert urdu_language_evidence(_PERSIAN_LETTERS) == 0
    assert urdu_language_evidence(_PERSIAN_LETTERS + _EXTENDED_ARABIC_INDIC_DIGITS) == 0
    assert candidate_languages_for_text(_PERSIAN_LETTERS, "Arabic") == (
        candidate_languages_for_script("Arabic")
    )


def test_koranic_stop_sign_ligatures_are_not_urdu_evidence():
    # U+FDF0/U+FDF1 decompose through yeh barree (U+06D2) under NFKC but occur
    # in Arabic religious text, so only single-character decompositions count.
    for codepoint in (0xFDF0, 0xFDF1):
        ligature = chr(codepoint)
        assert chr(0x06D2) in unicodedata.normalize("NFKC", ligature)
        assert urdu_language_evidence(_ARABIC_STEM + ligature) == 0


def test_urdu_presentation_form_evidence_is_pinned_to_sixteen_forms():
    scoring_forms = [
        chr(codepoint)
        for start, end in _ARABIC_PRESENTATION_RANGES
        for codepoint in range(start, end + 1)
        if urdu_language_evidence(chr(codepoint)) > 0
    ]

    # Pinned so a Unicode database upgrade fails loudly instead of silently
    # widening or narrowing the Urdu evidence set.
    assert len(scoring_forms) == 16
    assert all(
        unicodedata.normalize("NFKC", form) in _URDU_LETTERS for form in scoring_forms
    )


def test_extended_arabic_indic_digits_reinforce_but_never_trigger_urdu():
    assert urdu_language_evidence(_EXTENDED_ARABIC_INDIC_DIGITS) == 0
    assert urdu_language_evidence(_ARABIC_STEM + _EXTENDED_ARABIC_INDIC_DIGITS) == 0
    assert urdu_language_evidence(_URDU_LETTERS[0] + _EXTENDED_ARABIC_INDIC_DIGITS) == (
        1 + len(_EXTENDED_ARABIC_INDIC_DIGITS)
    )
    # Arabic-Indic digits are shared with Arabic and never add evidence.
    assert urdu_language_evidence(_URDU_LETTERS[0] + _ARABIC_INDIC_DIGITS) == 1


def test_urdu_evidence_moves_ur_ahead_of_ar_without_dropping_candidates():
    baseline = candidate_languages_for_script("Arabic")
    reordered = candidate_languages_for_text(_ARABIC_STEM + _URDU_LETTERS[0], "Arabic")

    assert baseline == ("ar", "ha", "ur")
    assert reordered == ("ur", "ar", "ha")
    assert set(reordered) == set(baseline)
    assert candidate_languages_for_text(_ARABIC_STEM, "Arabic") == baseline


def test_urdu_disambiguation_leaves_script_runs_and_graphemes_unchanged():
    fatha = chr(0x064E)
    arabic = f"Patient {_ARABIC_STEM}{fatha} stable"
    urdu = f"Patient {_ARABIC_STEM[:-1]}{_URDU_LETTERS[0]}{fatha} stable"

    assert len(arabic) == len(urdu)
    assert list(segment_by_script(arabic)) == list(segment_by_script(urdu))
    for text in (arabic, urdu):
        for start, end, _script in segment_by_script(text):
            assert snap_span_to_grapheme_boundaries(start, end, text) == (start, end)


def test_normalize_for_pii_detection_folds_obfuscation_with_offset_map():
    text = "Patient J\u200bo\u0301hn D\u03bfe"
    normalized = normalize_for_pii_detection(text)

    assert normalized.text == "Patient John Doe"
    assert normalized.changed
    assert normalized.mixed_script
    assert normalized.removed_zero_width == 1
    assert normalized.stripped_combining_marks == 1
    assert normalized.folded_confusables == 1
    assert normalized.remap_span(8, 16) == (8, len(text))
    assert "Patient" not in normalized.to_metadata()


def test_normalize_for_pii_detection_routes_indic_runs_and_preserves_marks():
    text = "Patient न\u093cील ന്\u200d"
    normalized = normalize_for_pii_detection(text)

    assert normalized.text == "Patient ऩील ൻ"
    assert normalized.indic_scripts == ("Devanagari", "Malayalam")
    assert normalized.indic_changes > 0
    assert normalized.removed_zero_width == 1
    assert "ी" in normalized.text
    name_start = normalized.text.index("ऩील")
    assert normalized.remap_span(name_start, name_start + len("ऩील")) == (8, 12)


def test_normalize_for_pii_detection_strips_standalone_ethiopic_mark():
    normalized = normalize_for_pii_detection("\u135f")

    assert normalized.text == ""
    assert normalized.stripped_combining_marks == 1


@given(
    before=st.lists(st.sampled_from(tuple("ሀለሐመሠረሰቀበተነአከወዘየደገጠጸፈፐ")), min_size=1),
    after=st.lists(st.sampled_from(tuple("ሀለሐመሠረሰቀበተነአከወዘየደገጠጸፈፐ")), max_size=8),
    prefix=st.sampled_from(("", "ስም፡ ", "Patient ")),
    suffix=st.sampled_from(("", "።", " visited")),
)
def test_ethiopic_combining_mark_remaps_without_offset_drift(
    before: list[str],
    after: list[str],
    prefix: str,
    suffix: str,
):
    marked_value = f"{''.join(before)}\u135f{''.join(after)}"
    text = f"{prefix}{marked_value}{suffix}"
    normalized = normalize_for_pii_detection(text)
    value_start = len(prefix)
    value_end = value_start + len(marked_value)
    grapheme_start = value_start + len(before) - 1
    grapheme_end = grapheme_start + 2

    assert normalized.text == text
    assert normalized.stripped_combining_marks == 0
    assert normalized.remap_span(value_start, value_end) == (value_start, value_end)
    assert normalized.remap_span(grapheme_start, grapheme_end) == (
        grapheme_start,
        grapheme_end,
    )
    assert text[grapheme_start:grapheme_end].endswith("\u135f")


def test_confusable_skeleton_covers_cross_script_width_and_invisible_attacks():
    attacked = "J\u043ehn D\u03bfe D\u3007E \uff2d\uff32\uff2e A\u200b1001"

    assert confusable_skeleton(attacked) == "John Doe DOE MRN A1001"
    assert CONFUSABLE_DATA_VERSION == "17.0.0"
    assert CONFUSABLE_DATA_LICENSE == "Unicode-3.0"
    assert CONFUSABLE_DATA_URL.endswith("/17.0.0/security/confusables.txt")


def test_mixed_script_detector_flags_only_identifier_local_script_mixing():
    text = "Patient J\u043ehn met \u4f50\u85e4 after discharge"

    findings = mixed_script_spans(text)

    assert detect_mixed_script(text)
    assert len(findings) == 1
    assert findings[0].scripts == ("Cyrillic", "Latin")
    assert text[findings[0].start : findings[0].end] == "J\u043ehn"
    assert findings[0].confusable_count == 1
    assert not detect_mixed_script("Patient John met \u4f50\u85e4 after discharge")


def test_han_confusable_normalization_preserves_original_offsets():
    text = "Patient D\u3007E arrived"
    normalized = normalize_for_pii_detection(text)

    assert normalized.text == "Patient DOE arrived"
    assert normalized.mixed_script
    assert normalized.remap_span(8, 11) == (8, 11)
    assert text[slice(*normalized.remap_span(8, 11))] == "D\u3007E"


# ---------------------------------------------------------------------------
# Grapheme-aligned script-run segmentation
#
# Every fixture below is assembled from explicit Unicode code points in code.
# None of it is real clinical text and none of it contains PHI.
# ---------------------------------------------------------------------------

_LATIN_A = "\u0061"
_LATIN_B = "\u0062"
_LATIN_R = "\u0052"
_LATIN_I = "\u0069"
_DEVANAGARI_KA = "\u0915"
_DEVANAGARI_SSA = "\u0937"
_DEVANAGARI_VIRAMA = "\u094d"
_DEVANAGARI_NUKTA = "\u093c"
_DEVANAGARI_UDATTA = "\u0951"
_BENGALI_KA = "\u0995"
_TAMIL_KA = "\u0b95"
_TAMIL_SSA = "\u0bb7"
_TAMIL_VIRAMA = "\u0bcd"
_ZWJ = "\u200d"
_ZWNJ = "\u200c"
_MAN = "\U0001f468"
_MEDICAL_SYMBOL = "\u2695\ufe0f"
_REGIONAL_I = "\U0001f1ee"
_REGIONAL_N = "\U0001f1f3"

# Clusters whose combining mark belongs to a different script than its base.
# Segmenting these by code point splits the cluster.
_CROSS_SCRIPT_CLUSTER_CASES = (
    ("latin_base_devanagari_udatta", _LATIN_A + _DEVANAGARI_UDATTA + _LATIN_B),
    (
        "latin_base_devanagari_nukta",
        _LATIN_R + _LATIN_A + _DEVANAGARI_NUKTA + _LATIN_I,
    ),
    ("bengali_base_devanagari_udatta", _BENGALI_KA + _DEVANAGARI_UDATTA),
)

# Sequences that code-point segmentation already handled correctly. They are
# retained as regression guards and must stay at zero boundary violations.
_ALREADY_SAFE_CASES = (
    (
        "devanagari_virama_conjunct",
        _DEVANAGARI_KA + _DEVANAGARI_VIRAMA + _DEVANAGARI_SSA,
    ),
    ("tamil_virama_conjunct", _TAMIL_KA + _TAMIL_VIRAMA + _TAMIL_SSA),
    ("emoji_zwj_sequence", "Dr " + _MAN + _ZWJ + _MEDICAL_SYMBOL + " ok"),
    ("regional_indicator_pair", _REGIONAL_I + _REGIONAL_N + " x"),
    (
        "zwnj_inside_devanagari",
        _DEVANAGARI_KA + _DEVANAGARI_VIRAMA + _ZWNJ + _DEVANAGARI_SSA,
    ),
)


def _grapheme_boundaries(text):
    """Return every extended grapheme-cluster boundary offset in ``text``."""

    return {start for start, _ in iter_grapheme_cluster_spans(text)} | {len(text)}


@pytest.mark.parametrize(
    "case",
    _CROSS_SCRIPT_CLUSTER_CASES + _ALREADY_SAFE_CASES,
    ids=lambda case: case[0],
)
def test_segment_by_script_never_splits_a_grapheme_cluster(case):
    _name, text = case
    boundaries = _grapheme_boundaries(text)
    runs = list(segment_by_script(text))

    assert runs
    for run in runs:
        assert run.start in boundaries, "run start bisects a grapheme cluster"
        assert run.end in boundaries, "run end bisects a grapheme cluster"
        assert run.start < run.end
    assert "".join(text[run.start : run.end] for run in runs) == text


@pytest.mark.parametrize("case", _CROSS_SCRIPT_CLUSTER_CASES, ids=lambda case: case[0])
def test_cross_script_cluster_stays_whole_in_one_run(case):
    """A combining mark never drags its base character into another run."""

    _name, text = case
    runs = list(segment_by_script(text))

    for cluster_start, cluster_end in iter_grapheme_cluster_spans(text):
        covering = [
            run for run in runs if run.start <= cluster_start and cluster_end <= run.end
        ]
        assert len(covering) == 1


def test_script_runs_stay_tuple_compatible_for_existing_callers():
    """The richer run type must remain a plain tuple for existing unpackers."""

    text = _DEVANAGARI_KA + " " + _LATIN_A
    runs = list(segment_by_script(text))

    assert runs == [(0, 2, "Devanagari"), (2, 3, "Latin")]
    assert isinstance(runs[0], ScriptRun)
    assert isinstance(runs[0], tuple)
    start, end, script = runs[0]
    assert (start, end, script) == (0, 2, "Devanagari")
    assert runs[0].extract(text) == text[0:2]


def test_indic_script_sets_agree_across_modules():
    """Pin the aliased processing-layer tuple against this module's export.

    ``script_detect`` imports ``openmed.processing.text.INDIC_SCRIPTS`` under an
    alias so it cannot rebind its own identically named public export. This gate
    fails if the two ever diverge or if the alias is collapsed.
    """

    assert frozenset(PROCESSING_INDIC_SCRIPTS) == INDIC_SCRIPTS
    assert len(INDIC_SCRIPTS) == 9


def test_script_routing_metadata_is_exact_and_complete():
    routable = set(SUPPORTED_SCRIPTS) | {UNKNOWN_SCRIPT}
    brahmi = {
        "Bengali",
        "Devanagari",
        "Gujarati",
        "Gurmukhi",
        "Kannada",
        "Malayalam",
        "Odia",
        "Tamil",
        "Telugu",
    }

    assert set(SCRIPT_NORMALIZERS) == routable
    assert set(SCRIPT_NUMERAL_SETS) == routable
    assert {
        script for script, name in SCRIPT_NORMALIZERS.items() if name == "indic-nfc"
    } == brahmi
    assert {
        script for script, name in SCRIPT_NUMERAL_SETS.items() if name != "ascii"
    } == brahmi | {"Arabic", "Thai"}
    for script in brahmi:
        assert SCRIPT_NUMERAL_SETS[script] == script.casefold()

    # Arabic-script text writes numbers with its own digits, so declaring
    # "ascii" here would tell a consumer no native numerals are present.
    assert SCRIPT_NUMERAL_SETS["Arabic"] == "arabic-indic"
    assert SCRIPT_NUMERAL_SETS["Thai"] == "thai"
    assert SCRIPT_NORMALIZERS["Latin"] == "unicode-defense"
    assert SCRIPT_NUMERAL_SETS["Latin"] == "ascii"
    # Fullwidth digits are ASCII digits in a wide presentation form, folded by
    # width normalization, so Han is correctly declared as "ascii".
    assert SCRIPT_NUMERAL_SETS["Han"] == "ascii"


def test_declared_arabic_numeral_set_matches_the_digits_actually_folded():
    """The Arabic declaration must cover both blocks the pipeline folds."""

    from openmed.core.pii_i18n import normalize_arabic_indic_digits

    assert SCRIPT_NUMERAL_SETS["Arabic"] != "ascii"
    for codepoint in (0x0660, 0x06F0):
        native = "".join(chr(codepoint + digit) for digit in range(10))
        assert detect_script(native + "\u0639") == "Arabic"
        assert normalize_arabic_indic_digits(native) == "0123456789"
    assert SCRIPT_NORMALIZERS[UNKNOWN_SCRIPT] == "unicode-defense"
    assert normalizer_for_script("Devanagari") == "indic-nfc"
    assert numeral_set_for_script("Tamil") == "tamil"
    assert normalizer_for_script("NotAScript") == "unicode-defense"
    assert numeral_set_for_script("NotAScript") == "ascii"


def test_segment_by_script_stays_linear_on_adversarial_combining_runs(monkeypatch):
    """Cluster resolution must not rewind once per extending mark.

    A long cluster whose extending marks carry a different script from their
    base used to send every mark back to the cluster start, making segmentation
    quadratic. ``validate_pii_input`` accepts this shape because the combining
    and format-sequence guards reset on each other's characters, so the only
    thing standing between an untrusted document and quadratic work is this
    bound. Predicate calls are counted instead of wall-clock time so the gate is
    deterministic under CI load.
    """

    from openmed.core.decoding import spans as decoding_spans

    calls = 0
    real_checker = decoding_spans.grapheme_break_checker

    def counting_checker(text):
        predicate = real_checker(text)

        def wrapper(index):
            nonlocal calls
            calls += 1
            return predicate(index)

        return wrapper

    monkeypatch.setattr(decoding_spans, "grapheme_break_checker", counting_checker)

    payload = _LATIN_A + (_DEVANAGARI_UDATTA * 63 + _ZWJ) * 40 + _LATIN_B
    runs = list(segment_by_script(payload))

    assert "".join(payload[run.start : run.end] for run in runs) == payload
    # Linear: the memoized rewinds telescope, so total probes stay within a
    # small constant factor of the input length. The pre-fix implementation
    # issued roughly len(cluster) probes per mark, i.e. ~80,000 here.
    assert calls <= 4 * len(payload), f"{calls} probes for {len(payload)} chars"
