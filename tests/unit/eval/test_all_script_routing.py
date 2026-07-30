"""End-to-end script-aware routing regression over the all-script fixture.

The synthetic ``all_script_routing.jsonl`` fixture is one multi-script clinical
note whose runs span Latin, all nine Brahmi scripts (Devanagari, Bengali,
Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam) and the Arabic
block carrying Urdu. This suite proves the completed routing stack: each run is
classified to the right Unicode script, resolves to the right language path
(either a bundled pack the router selects or the top routing candidate for
languages without a bundled model yet), every run boundary is a grapheme
boundary, the source slices reconstruct byte-for-byte, and Urdu resolves to
``ur`` rather than ``ar``. Focused regressions pin existing Hindi, Telugu, and
Latin routing.
"""

from __future__ import annotations

import json
from pathlib import Path

from openmed.core.decoding.spans import is_grapheme_boundary
from openmed.core.language_pack_catalog import SCRIPT_LANGUAGE_HINTS
from openmed.core.language_router import LanguageRouter
from openmed.core.script_detect import (
    candidate_languages_for_script,
    candidate_languages_for_text,
    detect_script,
    segment_by_script,
    urdu_language_evidence,
)

_FIXTURE_PATH = Path("openmed/eval/golden/fixtures/i18n/all_script_routing.jsonl")

# The eleven required runs, in source order.
_REQUIRED_SCRIPTS = (
    "Latin",
    "Devanagari",
    "Bengali",
    "Gurmukhi",
    "Gujarati",
    "Odia",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Arabic",
)

# Scripts whose language path is served by a bundled pack the router selects.
# The remaining scripts carry their language on the top routing candidate until
# a bundled model or complete language pack lands.
_SELECTED_LANGUAGES = frozenset({"en", "hi", "bn", "or", "ta", "te"})


def _load_fixture() -> dict:
    rows = [
        json.loads(line)
        for line in _FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1, "fixture must hold exactly one multi-script note"
    return rows[0]


def test_fixture_is_synthetic_and_covers_all_eleven_runs():
    fixture = _load_fixture()

    assert fixture["synthetic"] is True
    assert fixture["metadata"]["synthetic"] is True
    runs = fixture["runs"]
    assert len(runs) == 11
    assert tuple(run["script"] for run in runs) == _REQUIRED_SCRIPTS


def test_segmentation_matches_fixture_runs_and_reconstructs_byte_for_byte():
    fixture = _load_fixture()
    text = fixture["text"]

    segments = list(segment_by_script(text))
    assert len(segments) == len(fixture["runs"])

    cursor = 0
    for (start, end, script), run in zip(segments, fixture["runs"]):
        assert (start, end, script) == (run["start"], run["end"], run["script"])
        assert start == cursor
        assert start < end
        assert detect_script(text[start:end]) == run["script"]
        cursor = end

    assert cursor == len(text)
    assert "".join(text[start:end] for start, end, _ in segments) == text


def test_every_run_boundary_is_a_grapheme_boundary():
    fixture = _load_fixture()
    text = fixture["text"]

    for run in fixture["runs"]:
        assert is_grapheme_boundary(run["start"], text)
        assert is_grapheme_boundary(run["end"], text)


def test_each_run_resolves_to_the_expected_language_path():
    fixture = _load_fixture()
    text = fixture["text"]
    router = LanguageRouter(use_optional_lid=False)
    routed = {(run.start, run.end): run for run in router.route_runs(text)}

    for run in fixture["runs"]:
        slice_text = text[run["start"] : run["end"]]
        language = run["language"]

        candidates = candidate_languages_for_text(slice_text)
        assert list(candidates) == run["candidate_languages"]
        assert candidates[0] == language
        # Routing metadata: the resolved language is a declared candidate for
        # its detected script.
        assert language in SCRIPT_LANGUAGE_HINTS[run["script"]]

        decision = routed[(run["start"], run["end"])]
        if run["resolution"] == "selected":
            assert language in _SELECTED_LANGUAGES
            assert decision.language == language
        else:
            # No bundled pack yet: the top candidate carries the language path
            # while the router falls back deterministically and offline.
            assert language not in _SELECTED_LANGUAGES


def test_urdu_run_resolves_to_ur_not_ar():
    fixture = _load_fixture()
    text = fixture["text"]
    urdu_run = next(run for run in fixture["runs"] if run["language"] == "ur")
    slice_text = text[urdu_run["start"] : urdu_run["end"]]

    assert detect_script(slice_text) == "Arabic"
    assert urdu_language_evidence(slice_text) > 0
    candidates = candidate_languages_for_text(slice_text)
    assert candidates[0] == "ur"
    assert candidates.index("ur") < candidates.index("ar")

    # Plain Arabic keeps ``ar`` first: the reorder is evidence-driven only.
    arabic = "المريض أحمد علي"
    assert urdu_language_evidence(arabic) == 0
    assert candidate_languages_for_text(arabic)[0] == "ar"
    assert candidate_languages_for_script("Arabic")[0] == "ar"


def test_existing_hindi_telugu_and_latin_routing_is_unchanged():
    router = LanguageRouter(use_optional_lid=False)

    assert router.route("रोगी को बुखार है।").language == "hi"
    assert router.route("రోగి స్థిరంగా ఉన్నారు.").language == "te"
    assert router.route("Patient stable.").language == "en"

    for text, script, language in (
        ("रोगी को बुखार है।", "Devanagari", "hi"),
        ("రోగి స్థిరంగా ఉన్నారు.", "Telugu", "te"),
        ("Patient stable.", "Latin", "en"),
    ):
        assert detect_script(text) == script
        assert candidate_languages_for_text(text)[0] == language
