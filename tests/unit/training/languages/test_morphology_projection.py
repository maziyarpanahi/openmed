"""Span-integrity, ambiguity and negative-control gates for label projection.

Every text, lemma and profile below is synthetic. The profiles are deliberately
small illustrations of each language's morphology, not linguistic references.
"""

from __future__ import annotations

import json
import re
import socket
import unicodedata

import pytest

from openmed.core.decoding.spans import (
    grapheme_break_checker,
    iter_grapheme_cluster_spans,
)
from openmed.training.languages.morphology_projection import (
    PROJECTED_REASONS,
    PROJECTION_STATUSES,
    REVIEW_REASONS,
    SPAN_POLICIES,
    UNMATCHED_REASONS,
    AffixRule,
    LabelProjection,
    MorphologyProfile,
    ProjectionDecision,
    StemAlternation,
    project_labels,
)


def _nfd(value: str) -> str:
    return unicodedata.normalize("NFD", value)


def _affixes(*surfaces: str, position: str = "suffix", kind: str = "inflection"):
    return [
        {"surface": surface, "position": position, "kind": kind} for surface in surfaces
    ]


def _window(text: str, surface: str) -> tuple[int, int]:
    start = text.index(surface)
    return start, start + len(surface)


PROFILES = {
    "fi": {
        "language": "fi",
        "affixes": _affixes("n", "en", "ssa", "ssä", "sta", "stä", "ta", "tä")
        + _affixes("kin", "kaan", "kään", "han", "hän", kind="clitic"),
        "alternations": [
            {"lemma_edge": "s", "stem_edge": "kse"},
            {"lemma_edge": "nen", "stem_edge": "se"},
        ],
    },
    "tr": {
        "language": "tr",
        "affixes": _affixes(
            *("da", "de", "ta", "te", "dan", "den", "in", "ın", "un", "ün"),
            *("nın", "nin", "ı", "i", "u", "ü", "a", "e", "lar", "ler"),
        ),
        "alternations": [
            {"lemma_edge": lemma_edge, "stem_edge": stem_edge}
            for lemma_edge, stem_edge in (("p", "b"), ("ç", "c"), ("k", "ğ"))
        ],
    },
    "hu": {
        "language": "hu",
        "affixes": _affixes("ben", "ban", "ból", "ből", "t", "ot", "et", "at"),
        "alternations": [
            {"lemma_edge": "e", "stem_edge": "é"},
            {"lemma_edge": "a", "stem_edge": "á"},
        ],
    },
    "sw": {
        "language": "sw",
        "affixes": _affixes("m", "wa", "ma", "ki", "vi", position="prefix")
        + _affixes("ni"),
        "alternations": [{"lemma_edge": "u", "stem_edge": "", "position": "prefix"}],
    },
    "am": {
        "language": "am",
        "affixes": _affixes("በ", "ለ", "ከ", "የ", position="prefix", kind="clitic")
        + _affixes("ው", "ን"),
        "lexicalized_forms": ["በሽታ"],
    },
    "ta": {
        "language": "ta",
        "affixes": _affixes("க்கு", "யில்", "ை"),
        "alternations": [{"lemma_edge": "ு", "stem_edge": ""}],
    },
}

TEXTS = {
    "fi": (
        "Potilaalla on diabetes. Diabeteksen hoito jatkuu, eikä munuaisessa "
        "näkynyt muutoksia. Verensokeri mitattiin, ja diabeteskin kirjattiin."
    ),
    "tr": (
        "İnsülin dozu artırıldı. Hastanın kalbinde üfürüm duyuldu ve EKG'de "
        "ritim düzensizdi. İlacı kesildi."
    ),
    # Decomposed accents exercise NFC matching against NFD source offsets.
    "hu": _nfd("A betegségben szenvedő páciens vesében cisztát láttak."),
    "sw": "Mgonjwa ana magonjwa ya moyo na maumivu kifuani.",
    "am": "ታካሚው በስኳር በሽታ ይሰቃያል። በሽታው በመድኃኒት ቁጥጥር ስር ነው።",
    "ta": "நோயாளிக்கு நீரிழிவுக்கு மருந்தை மருத்துவமனையில் கொடுத்தனர்.",
}

# (language, label, lemma, window surface, policy, status, reason, surface)
# fmt: off
CASES = [
    ("fi", "DISEASE", "diabetes", "Diabeteksen", "host", "projected",
     "alternation_match", "Diabeteksen"),
    ("fi", "DISEASE", "diabetes", "Diabeteksen", "stem", "projected",
     "alternation_match", "Diabetekse"),
    ("fi", "ANATOMY", "munuainen", None, "host", "projected",
     "alternation_match", "munuaisessa"),
    ("fi", "DISEASE", "diabetes", "diabeteskin", "host", "projected",
     "affix_match", "diabetes"),
    ("fi", "DISEASE", "diabetes", "diabeteskin", "token", "projected",
     "affix_match", "diabeteskin"),
    ("fi", "DISEASE", "diabetes", None, "host", "needs_review",
     "multiple_matches", None),
    ("fi", "LAB", "sokeri", None, "host", "needs_review", "unanalyzed_affix", None),
    ("fi", "DISEASE", "astma", None, "host", "unmatched", "no_match", None),
    ("tr", "MEDICATION", "insülin", None, "host", "projected", "exact_match",
     "İnsülin"),
    ("tr", "ANATOMY", "kalp", None, "host", "projected", "alternation_match",
     "kalbinde"),
    ("tr", "ANATOMY", "kalp", None, "stem", "projected", "alternation_match",
     "kalb"),
    ("tr", "MEDICATION", "ilaç", None, "host", "projected", "alternation_match",
     "İlacı"),
    ("tr", "TEST", "EKG", None, "host", "projected", "exact_match", "EKG"),
    ("tr", "ROLE", "hasta", None, "host", "projected", "affix_match", "Hastanın"),
    # Privative "-siz" is unconfigured: "düzensiz" (irregular) is not "düzen".
    ("tr", "FINDING", "düzen", None, "host", "needs_review", "unanalyzed_affix",
     None),
    ("hu", "DISEASE", "betegség", None, "host", "projected", "affix_match",
     _nfd("betegségben")),
    ("hu", "DISEASE", "betegség", None, "stem", "projected", "affix_match",
     _nfd("betegség")),
    ("hu", "ANATOMY", "vese", None, "host", "projected", "alternation_match",
     _nfd("vesében")),
    ("hu", "FINDING", "ciszta", None, "stem", "projected", "alternation_match",
     _nfd("cisztá")),
    ("hu", "DISEASE", "beteg", None, "host", "needs_review", "unanalyzed_affix",
     None),
    ("sw", "DISEASE", "ugonjwa", None, "host", "needs_review",
     "multiple_matches", None),
    ("sw", "DISEASE", "ugonjwa", "magonjwa", "host", "projected",
     "alternation_match", "magonjwa"),
    ("sw", "DISEASE", "ugonjwa", "magonjwa", "stem", "projected",
     "alternation_match", "gonjwa"),
    ("sw", "ANATOMY", "kifua", None, "host", "projected", "affix_match",
     "kifuani"),
    ("sw", "ANATOMY", "moyo", None, "host", "projected", "exact_match", "moyo"),
    ("am", "DISEASE", "ስኳር በሽታ", None, "host", "projected", "affix_match",
     "ስኳር በሽታ"),
    ("am", "DISEASE", "ስኳር በሽታ", None, "token", "projected", "affix_match",
     "በስኳር በሽታ"),
    ("am", "DISEASE", "በሽታ", None, "host", "needs_review", "multiple_matches",
     None),
    ("am", "DISEASE", "በሽታ", "በሽታው", "host", "projected", "affix_match",
     "በሽታው"),
    ("am", "DISEASE", "በሽታ", "በሽታው", "stem", "projected", "affix_match", "በሽታ"),
    ("am", "MEDICATION", "መድኃኒት", None, "host", "projected", "affix_match",
     "መድኃኒት"),
    ("am", "SYMPTOM", "ሽታ", "በሽታው", "host", "needs_review", "unanalyzed_affix",
     None),
    ("ta", "DISEASE", "நீரிழிவு", None, "stem", "projected", "affix_match",
     "நீரிழிவு"),
    ("ta", "MEDICATION", "மருந்து", None, "host", "projected",
     "alternation_match", "மருந்தை"),
    ("ta", "MEDICATION", "மருந்து", None, "stem", "needs_review",
     "grapheme_bisection", None),
    ("ta", "FACILITY", "மருத்துவமனை", None, "host", "projected", "affix_match",
     "மருத்துவமனையில்"),
]
# fmt: on


def _profile(language: str, **overrides) -> MorphologyProfile:
    return MorphologyProfile.from_dict({**PROFILES[language], **overrides})


def _projection(language: str, label: str, lemma: str, window: str | None):
    text = TEXTS[language]
    return LabelProjection(
        label, lemma, _window(text, window) if window is not None else None
    )


def _language_projections(language: str) -> list[LabelProjection]:
    return [_projection(*case[:4]) for case in CASES if case[0] == language]


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[f"{case[0]}-{case[1]}-{case[4]}-{n}" for n, case in enumerate(CASES)],
)
def test_fixture_projection_decisions(case) -> None:
    language, label, lemma, window, policy, status, reason, surface = case
    text = TEXTS[language]
    result = project_labels(
        text,
        [_projection(language, label, lemma, window)],
        _profile(language),
        span_policy=policy,
    )
    (decision,) = result.decisions
    assert (decision.status, decision.reason) == (status, reason)
    if surface is None:
        assert decision.start is None and decision.end is None
    else:
        assert text[decision.start : decision.end] == surface


def test_fixtures_cover_several_morphologically_rich_languages_with_controls():
    languages = {case[0] for case in CASES}
    assert languages == {"fi", "tr", "hu", "sw", "am", "ta"}
    for language in languages:
        statuses = {case[5] for case in CASES if case[0] == language}
        assert "projected" in statuses
        assert statuses - {"projected"}, f"{language} lacks a negative control"


@pytest.mark.parametrize("policy", SPAN_POLICIES)
@pytest.mark.parametrize("language", sorted(TEXTS))
def test_projected_spans_round_trip_grapheme_and_byte_offsets(language, policy):
    text = TEXTS[language]
    encoded = text.encode("utf-8")
    clusters = list(iter_grapheme_cluster_spans(text))
    has_break = grapheme_break_checker(text)
    result = project_labels(
        text, _language_projections(language), _profile(language), span_policy=policy
    )
    assert result.projected
    for decision in result.projected:
        start, end = decision.start, decision.end
        assert has_break(start) and has_break(end)
        assert (
            encoded[decision.byte_start : decision.byte_end].decode("utf-8")
            == (text[start:end])
        )
        assert len(encoded[: decision.byte_start].decode("utf-8")) == start
        assert clusters[decision.grapheme_start][0] == start
        assert clusters[decision.grapheme_end - 1][1] == end
    for decision in result.review_queue:
        for candidate_start, candidate_end in decision.candidate_spans:
            assert has_break(candidate_start) and has_break(candidate_end)


def test_turkic_dotted_capital_i_keeps_exact_byte_offsets():
    text = TEXTS["tr"]
    projection = [LabelProjection("MEDICATION", "insülin")]
    (decision,) = project_labels(text, projection, _profile("tr")).decisions
    assert (decision.start, decision.end) == (0, 7)
    assert (decision.byte_start, decision.byte_end) == (0, 9)
    assert (decision.grapheme_start, decision.grapheme_end) == (0, 7)

    # Negative control: Unicode default folding turns the capital into "i̇".
    unicode_profile = _profile("tr", casing="unicode")
    (control,) = project_labels(text, projection, unicode_profile).decisions
    assert (control.status, control.reason) == ("unmatched", "no_match")


def test_decomposed_accents_map_back_to_every_source_code_point():
    text = TEXTS["hu"]
    result = project_labels(
        text,
        [LabelProjection("DISEASE", "betegség")],
        _profile("hu"),
        span_policy="stem",
    )
    (decision,) = result.decisions
    assert decision.end - decision.start == 9
    assert decision.grapheme_end - decision.grapheme_start == 8
    assert decision.byte_end - decision.byte_start == len(_nfd("betegség").encode())


def test_amharic_byte_offsets_cover_three_byte_ethiopic_syllables():
    text = TEXTS["am"]
    result = project_labels(
        text, [LabelProjection("MEDICATION", "መድኃኒት")], _profile("am")
    )
    (decision,) = result.decisions
    assert text[decision.start : decision.end] == "መድኃኒት"
    assert decision.byte_end - decision.byte_start == 3 * len("መድኃኒት")
    assert decision.byte_start == len(text[: decision.start].encode("utf-8"))


def test_tamil_fused_vowel_sign_stem_is_reviewed_not_split():
    text = TEXTS["ta"]
    projection = [LabelProjection("MEDICATION", "மருந்து")]
    (review,) = project_labels(
        text, projection, _profile("ta"), span_policy="stem"
    ).decisions
    assert review.reason == "grapheme_bisection"
    assert review.candidate_spans == (_window(text, "மருந்தை"),)
    (projected,) = project_labels(text, projection, _profile("ta")).decisions
    assert text[projected.start : projected.end] == "மருந்தை"


def test_case_folding_expansion_never_yields_a_partial_character_label():
    profile = MorphologyProfile(
        language="de", affixes=(AffixRule("s"),), min_stem_graphemes=3
    )
    text = "Das Maß stimmt."
    projection = [LabelProjection("MEASURE", "mas")]
    (decision,) = project_labels(
        text, projection, profile, span_policy="stem"
    ).decisions
    assert decision.reason == "grapheme_bisection"
    assert decision.candidate_spans == (_window(text, "Maß"),)


def test_segmentation_ambiguity_is_reviewed_only_when_policy_depends_on_it():
    ambiguous = _profile(
        "fi",
        alternations=[
            *PROFILES["fi"]["alternations"],
            {"lemma_edge": "s", "stem_edge": "ks"},
        ],
    )
    text = TEXTS["fi"]
    projection = [_projection("fi", "DISEASE", "diabetes", "Diabeteksen")]
    (stem,) = project_labels(text, projection, ambiguous, span_policy="stem").decisions
    assert stem.reason == "ambiguous_segmentation"
    assert [text[start:end] for start, end in stem.candidate_spans] == [
        "Diabeteks",
        "Diabetekse",
    ]
    (host,) = project_labels(text, projection, ambiguous).decisions
    assert host.status == "projected"
    assert text[host.start : host.end] == "Diabeteksen"


def test_lexicalized_form_blocks_clitic_split_that_would_mislabel_a_noun():
    text = TEXTS["am"]
    projection = [_projection("am", "SYMPTOM", "ሽታ", "በሽታው")]
    unguarded = _profile("am", lexicalized_forms=[])
    (hazard,) = project_labels(text, projection, unguarded).decisions
    assert hazard.status == "projected"  # the silent mislabel the guard prevents
    (guarded,) = project_labels(text, projection, _profile("am")).decisions
    assert (guarded.status, guarded.reason) == ("needs_review", "unanalyzed_affix")


def test_noun_class_swap_is_flagged_instead_of_labelled_twice():
    text = TEXTS["sw"]
    (decision,) = project_labels(
        text, [LabelProjection("DISEASE", "ugonjwa")], _profile("sw")
    ).decisions
    assert decision.reason == "multiple_matches"
    assert [text[start:end] for start, end in decision.candidate_spans] == [
        "Mgonjwa",
        "magonjwa",
    ]


def test_overlapping_or_conflicting_projections_are_routed_to_review():
    text = TEXTS["am"]
    first_disease = _window(text, "በሽታ")
    nested = project_labels(
        text,
        [
            LabelProjection("DISEASE", "ስኳር በሽታ"),
            LabelProjection("DISEASE", "በሽታ", first_disease),
        ],
        _profile("am"),
    )
    assert {item.reason for item in nested.decisions} == {"overlapping_projection"}

    window = _window(TEXTS["fi"], "Diabeteksen")
    conflicting = project_labels(
        TEXTS["fi"],
        [
            LabelProjection("DISEASE", "diabetes", window),
            LabelProjection("CONDITION", "diabetes", window),
        ],
        _profile("fi"),
    )
    assert {item.reason for item in conflicting.decisions} == {"overlapping_projection"}

    duplicates = project_labels(
        TEXTS["fi"],
        [LabelProjection("DISEASE", "diabetes", window)] * 2,
        _profile("fi"),
    )
    assert all(item.status == "projected" for item in duplicates.decisions)


def test_multiword_lemmas_require_the_same_spacing():
    text = "COVID-19'a bağlı ve COVID - 19 notu."
    result = project_labels(
        text, [LabelProjection("DISEASE", "COVID-19")], _profile("tr")
    )
    (decision,) = result.decisions
    assert decision.reason == "exact_match"
    assert text[decision.start : decision.end] == "COVID-19"


def test_affix_budget_and_minimum_stem_are_fail_closed():
    text = TEXTS["tr"]
    projection = [LabelProjection("ANATOMY", "kalp")]
    (bounded,) = project_labels(
        text, projection, _profile("tr", max_affixes=1)
    ).decisions
    assert (bounded.status, bounded.reason) == ("needs_review", "unanalyzed_affix")

    short = [LabelProjection("FINDING", "ta")]
    lenient = MorphologyProfile("fi", affixes=(AffixRule("ssa"),))
    strict = MorphologyProfile("fi", affixes=(AffixRule("ssa"),), min_stem_graphemes=3)
    assert project_labels("tassa", short, lenient).decisions[0].status == "projected"
    assert project_labels("tassa", short, strict).decisions[0].status != "projected"


def test_decisions_are_deterministic_and_independent_of_input_order():
    text = TEXTS["am"]
    projections = _language_projections("am")
    first = project_labels(text, projections, _profile("am"))
    assert first == project_labels(text, projections, _profile("am"))
    reversed_result = project_labels(text, projections[::-1], _profile("am"))
    count = len(projections)
    for decision in reversed_result.decisions:
        original = first.decisions[count - 1 - decision.index]
        assert decision.to_dict() | {"index": original.index} == original.to_dict()


def test_projection_runs_without_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    for language in TEXTS:
        result = project_labels(
            TEXTS[language], _language_projections(language), _profile(language)
        )
        assert result.decisions


def test_reports_contain_only_controlled_metadata():
    allowed = {
        *PROJECTION_STATUSES,
        *PROJECTED_REASONS,
        *REVIEW_REASONS,
        *UNMATCHED_REASONS,
        *SPAN_POLICIES,
        *TEXTS,
        *(case[1] for case in CASES),
    }
    digest = re.compile(r"[0-9a-f]{64}\Z")

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    for language in TEXTS:
        projections = _language_projections(language)
        result = project_labels(TEXTS[language], projections, _profile(language))
        report = result.to_dict()
        json.dumps(report)
        keys = set(report) | set(report["decisions"][0])
        for value in strings(report):
            assert value in allowed or value in keys or digest.fullmatch(value)
        rendered = repr(result) + repr(projections)
        for projection in projections:
            assert projection.lemma not in rendered


@pytest.mark.parametrize(
    "build",
    [
        lambda: LabelProjection("Jane Roe", "diabetes"),
        lambda: LabelProjection("DISEASE", "Jane Roe " * 40),
        lambda: AffixRule("Jane Roe"),
        lambda: MorphologyProfile.from_dict({"language": "fi", "Jane Roe": 1}),
        lambda: MorphologyProfile("Jane Roe"),
        lambda: project_labels(
            "Jane Roe", [LabelProjection("DISEASE", "roe", (0, 99))], _profile("fi")
        ),
    ],
)
def test_errors_never_echo_input_values(build):
    with pytest.raises((TypeError, ValueError)) as error:
        build()
    assert "Jane" not in str(error.value) and "Roe" not in str(error.value)


def test_profile_round_trip_and_canonical_digest():
    profile = _profile("fi")
    assert MorphologyProfile.from_dict(profile.to_dict()) == profile
    reordered = dict(PROFILES["fi"], affixes=PROFILES["fi"]["affixes"][::-1])
    assert MorphologyProfile.from_dict(reordered).digest == profile.digest
    assert re.fullmatch(r"[0-9a-f]{64}", profile.digest)
    assert _profile("fi", max_affixes=3).digest != profile.digest
    assert _profile("tr").casing == "turkic"
    assert _profile("fi").casing == "unicode"
    result = project_labels(TEXTS["fi"], [], profile)
    assert result.to_dict()["profile_digest"] == profile.digest


@pytest.mark.parametrize(
    ("build", "error"),
    [
        (lambda: MorphologyProfile("fi", affixes="ssa"), TypeError),
        (lambda: MorphologyProfile("fi", max_affixes=9), ValueError),
        (lambda: MorphologyProfile("fi", max_affixes=True), ValueError),
        (lambda: MorphologyProfile("fi", min_stem_graphemes=0), ValueError),
        (lambda: MorphologyProfile("fi", casing="upper"), ValueError),
        (lambda: MorphologyProfile("fi", lexicalized_forms=("two words",)), ValueError),
        (lambda: StemAlternation("s", "s"), ValueError),
        (lambda: AffixRule("ssa", position="infix"), ValueError),
        (lambda: AffixRule("ssa", kind="derivation"), ValueError),
        (lambda: AffixRule("\x00"), ValueError),
        (lambda: MorphologyProfile.from_dict({"affixes": []}), ValueError),
        (
            lambda: MorphologyProfile.from_dict(
                {"language": "fi", "affixes": [{"surface": "n", "extra": 1}]}
            ),
            ValueError,
        ),
        (lambda: LabelProjection("DISEASE", "diabetes", (4, 4)), ValueError),
        (lambda: LabelProjection("DISEASE", "diabetes", (True, 4)), ValueError),
    ],
)
def test_invalid_configuration_fails_closed(build, error):
    with pytest.raises(error):
        build()


def test_invalid_call_arguments_fail_closed():
    profile = _profile("fi")
    projection = [LabelProjection("DISEASE", "diabetes")]
    with pytest.raises(ValueError):
        project_labels(TEXTS["fi"], projection, profile, span_policy="lemma")
    with pytest.raises(TypeError):
        project_labels(TEXTS["fi"].encode(), projection, profile)
    with pytest.raises(TypeError):
        project_labels(TEXTS["fi"], ["diabetes"], profile)
    with pytest.raises(TypeError):
        project_labels(TEXTS["fi"], projection, PROFILES["fi"])


def test_window_accepts_json_style_pairs_and_empty_inputs_are_safe():
    assert LabelProjection("DISEASE", "diabetes", [1, 3]).window == (1, 3)
    assert project_labels("", [], _profile("fi")).decisions == ()
    (decision,) = project_labels(
        "   ", [LabelProjection("DISEASE", "diabetes")], _profile("fi")
    ).decisions
    assert (decision.status, decision.reason) == ("unmatched", "no_match")


def test_result_views_and_counts():
    result = project_labels(TEXTS["sw"], _language_projections("sw"), _profile("sw"))
    assert len(result.projected) + len(result.review_queue) == len(result.decisions)
    assert sum(result.reason_counts().values()) == len(result.decisions)
    assert result.to_dict()["status_counts"] == {
        "projected": len(result.projected),
        "needs_review": len(result.review_queue),
        "unmatched": 0,
    }


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "projected", "reason": "no_match"},
        {"status": "projected", "reason": "exact_match"},
        {"status": "needs_review", "reason": "multiple_matches"},
        {"status": "unmatched", "reason": "no_match", "start": 0},
        {"status": "unmatched", "reason": "no_match", "candidate_spans": ((0, 1),)},
        {"status": "accepted", "reason": "exact_match"},
        {"status": ["projected"], "reason": "exact_match"},
    ],
)
def test_decision_constructor_rejects_inconsistent_metadata(fields):
    with pytest.raises(ValueError):
        ProjectionDecision(index=0, label="DISEASE", **fields)
