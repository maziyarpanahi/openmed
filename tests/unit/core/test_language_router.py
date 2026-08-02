"""Acceptance coverage for exact-offset language and script routing."""

from __future__ import annotations

import importlib
import subprocess
import sys
from types import MappingProxyType

import pytest

from openmed.core.decoding import iter_grapheme_cluster_spans
from openmed.core.language_pack import LanguagePack
from openmed.core.language_router import (
    LanguagePrediction,
    LanguageRouter,
    PyCLD2LanguageIdentifier,
)


class _SyntheticLanguageIdentifier:
    name = "synthetic-lid"

    def identify(self, text, candidates):
        if any("\u0900" <= char <= "\u097f" for char in text):
            language = "hi"
        elif any("\u3040" <= char <= "\u30ff" for char in text):
            language = "ja"
        elif any("\u3400" <= char <= "\u9fff" for char in text):
            language = "zh"
        else:
            language = "en"
        if language not in candidates:
            return None
        return LanguagePrediction(language=language, confidence=0.99)


_ROUTING_CORPUS = (
    ("Patient reports fever and cough.", "en"),
    ("Nurse recorded stable blood pressure.", "en"),
    ("患者发热并伴有咳嗽。", "zh"),
    ("医生记录血压稳定。", "zh"),
    ("患者は発熱があります。", "ja"),
    ("医師が血圧を記録した。", "ja"),
    ("रोगी को बुखार है।", "hi"),
    ("चिकित्सक ने रक्तचाप दर्ज किया।", "hi"),
)


# Synthetic Arabic-script material built from code points, never pasted text.
# The stem uses only letters shared by Arabic, Persian and Urdu.
_ARABIC_STEM = "".join(chr(codepoint) for codepoint in (0x0627, 0x0644, 0x0645, 0x0631))
_ARABIC_TEXT = f"{_ARABIC_STEM} {_ARABIC_STEM}"
# Adds tteh and yeh barree, which only Urdu among the Arabic candidates uses.
_URDU_TEXT = f"{_ARABIC_STEM}{chr(0x0679)} {_ARABIC_STEM}{chr(0x06D2)}"
# Adds peh/tcheh/gaf, shared with Urdu, plus farsi yeh: no Urdu-exclusive letter.
_PERSIAN_TEXT = "".join(
    chr(codepoint) for codepoint in (0x067E, 0x0686, 0x06AF, 0x06CC)
)
# Urdu letters alongside extended Arabic-Indic digits (U+06F0-U+06F9).
_URDU_EXTENDED_DIGIT_TEXT = f"{_URDU_TEXT} {chr(0x06F4)}{chr(0x06F2)}"


def _pack(
    code: str,
    scripts: tuple[str, ...],
    model: str,
    *,
    candidate_priority: dict[str, int] | None = None,
    routing_markers: tuple[str, ...] = (),
) -> LanguagePack:
    return LanguagePack(
        code=code,
        scripts=scripts,
        default_model=model,
        segmenter_id="unicode-sentence",
        recognizers=("builtin-patterns", "model"),
        surrogate_locale="en_US",
        candidate_priority=candidate_priority or {},
        routing_markers=routing_markers,
    )


def _routing_accuracy(router: LanguageRouter) -> float:
    correct = 0
    total = 0
    for text, expected_language in _ROUTING_CORPUS:
        for run in router.route_runs(text):
            correct += run.language == expected_language
            total += 1
    return correct / total


def test_runs_tile_mixed_script_input_without_gaps_or_overlaps():
    text = "Patient stable. 患者は安定。 रोगी स्थिर है।"
    runs = LanguageRouter(use_optional_lid=False).route_runs(text)

    cursor = 0
    for run in runs:
        assert run.start == cursor
        assert run.start < run.end
        cursor = run.end
    assert cursor == len(text)
    assert "".join(text[run.start : run.end] for run in runs) == text


def test_cross_script_accuracy_gates_for_lid_and_stdlib_paths():
    lid_accuracy = _routing_accuracy(
        LanguageRouter(language_identifier=_SyntheticLanguageIdentifier())
    )
    fallback_accuracy = _routing_accuracy(LanguageRouter(use_optional_lid=False))

    assert lid_accuracy >= 0.95
    assert fallback_accuracy >= 0.90


def test_han_uses_kana_context_for_japanese_and_priority_for_chinese():
    router = LanguageRouter(use_optional_lid=False)

    chinese = router.route("患者发热并伴有咳嗽。")
    japanese = router.route("患者は発熱です。")

    assert chinese.language == "zh"
    assert {run.language for run in chinese.runs} == {"zh"}
    assert japanese.language == "ja"
    assert {run.language for run in japanese.runs} == {"ja"}
    assert any(run.source == "stdlib:context-script" for run in japanese.runs)


def test_urdu_cues_select_ur_over_ar_once_an_urdu_pack_is_registered():
    router = LanguageRouter(
        packs=(
            _pack("ar", ("Arabic",), "OpenMed/arabic"),
            _pack("ur", ("Arabic",), "OpenMed/urdu"),
        ),
        use_optional_lid=False,
    )

    urdu = router.route(_URDU_TEXT)
    arabic = router.route(_ARABIC_TEXT)
    persian = router.route(_PERSIAN_TEXT)

    assert urdu.language == "ur"
    assert urdu.runs[0].source == "stdlib:urdu-cues"
    assert urdu.runs[0].confidence == pytest.approx(0.99)
    assert urdu.runs[0].candidates == ("ur", "ar", "ha")
    assert arabic.language == "ar"
    assert all(run.language != "ur" for run in arabic.runs)
    assert arabic.runs[0].candidates == ("ar", "ha", "ur")
    assert persian.language == "ar"
    assert all(run.language != "ur" for run in persian.runs)
    assert persian.runs[0].candidates == ("ar", "ha", "ur")


def test_urdu_disambiguation_never_resolves_the_hausa_candidate():
    # ``ha`` sits in the Arabic hint tuple but is national-ID-only: the input
    # gateway rejects it whenever ``include_national_id`` is false, which is
    # what every public edge uses. Reordering must never promote it into
    # ``language``. The guarantee is ordering, not absence: evidence moves only
    # ``ur``, so ``ar`` keeps its place ahead of ``ha`` and always wins first.
    router = LanguageRouter(
        packs=(
            _pack("ar", ("Arabic",), "OpenMed/arabic"),
            _pack("ha", ("Arabic",), "OpenMed/hausa"),
            _pack("ur", ("Arabic",), "OpenMed/urdu"),
        ),
        use_optional_lid=False,
    )

    for text in (_ARABIC_TEXT, _URDU_TEXT, _PERSIAN_TEXT, _URDU_EXTENDED_DIGIT_TEXT):
        for run in router.route_runs(text):
            assert run.language != "ha"
            # ``ha`` may still be advertised in the advisory candidate list.
            assert "ha" in run.candidates
            assert run.candidates.index("ar") < run.candidates.index("ha")


def test_urdu_evidence_without_a_registered_pack_falls_back_to_arabic():
    # The built-in catalog ships no ``ur`` pack, so Urdu evidence must land on
    # the documented Arabic fallback at a visibly lower confidence.
    router = LanguageRouter(use_optional_lid=False)

    urdu = router.route(_URDU_TEXT)
    arabic = router.route(_ARABIC_TEXT)

    assert urdu.language == "ar"
    assert urdu.runs[0].source == "stdlib:arabic-fallback"
    assert urdu.runs[0].confidence == pytest.approx(0.8)
    assert arabic.language == "ar"
    assert arabic.runs[0].source == "stdlib:script"
    assert arabic.runs[0].confidence == pytest.approx(0.99)

    # The evidence is preserved in the run metadata even though no pack can
    # act on it yet: ``ur`` leads the candidate order while ``language``
    # reports the Arabic fallback that actually handled the run.
    assert urdu.runs[0].candidates == ("ur", "ar", "ha")
    assert urdu.runs[0].candidates[0] != urdu.runs[0].language
    assert arabic.runs[0].candidates == ("ar", "ha", "ur")
    assert arabic.runs[0].candidates[0] == arabic.runs[0].language


def test_arabic_fallback_confidence_weights_the_document_decision():
    # Hangul is the only script whose sole candidate pack routes at 0.99, so it
    # isolates the Arabic fallback's 0.8 in the length-weighted document score.
    router = LanguageRouter(use_optional_lid=False)
    prefix = "".join(chr(codepoint) for codepoint in (0xD658, 0xC790)) + " "
    decision = router.route(prefix + _URDU_TEXT)

    hangul, arabic = decision.runs
    assert (hangul.language, hangul.confidence) == ("ko", pytest.approx(0.99))
    assert (arabic.language, arabic.confidence) == ("ar", pytest.approx(0.8))
    assert arabic.source == "stdlib:arabic-fallback"
    assert arabic.candidates == ("ur", "ar", "ha")
    assert hangul.candidates == ("ko",)

    # The lower per-run confidence must propagate into the length-weighted
    # document score rather than being rounded away.
    expected = ((0.99 * len(prefix)) + (0.8 * len(_URDU_TEXT))) / len(
        prefix + _URDU_TEXT
    )
    assert decision.confidence == pytest.approx(expected)
    assert decision.confidence < 0.99


def test_bengali_evidence_keeps_its_established_routing_source():
    # The Urdu source vocabulary is added per script; Assamese must keep the
    # label that the golden-fixture gate pins.
    router = LanguageRouter(use_optional_lid=False)

    decision = router.route("গগৈ আৰু বৰা")

    assert decision.language == "as"
    assert decision.runs[0].source == "stdlib:assamese-cues"
    assert decision.runs[0].confidence == pytest.approx(0.99)
    assert decision.runs[0].candidates == ("as", "bn")


def test_devanagari_uses_pack_declared_candidate_priority():
    lower = _pack(
        "hi",
        ("Devanagari",),
        "OpenMed/hindi",
        candidate_priority={"Devanagari": 10},
    )
    higher = _pack(
        "mr",
        ("Devanagari",),
        "OpenMed/marathi",
        candidate_priority={"Devanagari": 20},
    )

    decision = LanguageRouter(
        packs=(lower, higher),
        use_optional_lid=False,
    ).route("रुग्ण स्थिर आहे।")

    assert decision.language == "mr"
    assert decision.runs[0].source == "stdlib:pack-priority"


def test_marathi_routing_markers_disambiguate_devanagari_from_hindi():
    router = LanguageRouter(use_optional_lid=False)

    marathi = router.route("रुग्ण स्थिर आहे.")
    hindi = router.route("रोगी को बुखार है।")
    compound_without_marker = router.route("रुग्णालय शांत है।")

    assert marathi.language == "mr"
    assert marathi.runs[0].source == "stdlib:routing-marker"
    assert hindi.language == "hi"
    assert hindi.runs[0].source == "stdlib:pack-priority"
    assert compound_without_marker.language == "hi"


def test_language_pack_freezes_routing_configuration():
    priorities = {"Han": 10}
    markers = ("patient",)
    pack = _pack(
        "zh",
        ("Han",),
        "OpenMed/chinese",
        candidate_priority=priorities,
        routing_markers=markers,
    )
    priorities["Han"] = 1

    assert pack.candidate_priority == {"Han": 10}
    assert pack.routing_markers == ("patient",)
    assert isinstance(pack.candidate_priority, MappingProxyType)
    with pytest.raises(TypeError):
        pack.candidate_priority["Han"] = 2


def test_optional_lid_is_lazy_and_missing_package_falls_back(monkeypatch):
    calls = []
    original_import_module = importlib.import_module

    def missing_pycld2(name, package=None):
        calls.append(name)
        if name == "pycld2":
            raise ModuleNotFoundError("pycld2 is not installed")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_pycld2)
    router = LanguageRouter()
    assert calls == []

    decision = router.route("患者发热。")

    assert calls == ["pycld2"]
    assert decision.language == "zh"
    assert decision.runs[0].source == "stdlib:pack-priority"


def test_core_module_imports_when_optional_lid_is_uninstalled():
    program = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'pycld2' or name.startswith('pycld2.'):
        raise ModuleNotFoundError('pycld2 unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import openmed.core.language_router
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_pycld2_adapter_filters_predictions_to_pack_candidates(monkeypatch):
    class FakeCLD2:
        @staticmethod
        def detect(text, bestEffort):
            assert text == "患者发热。"
            assert bestEffort is True
            return (
                True,
                len(text.encode()),
                (
                    ("Chinese", "zh", 98, 1000.0),
                    ("Unknown", "un", 2, 0.0),
                ),
            )

    monkeypatch.setattr(importlib, "import_module", lambda name: FakeCLD2())
    router = LanguageRouter(
        language_identifier=PyCLD2LanguageIdentifier(),
    )

    run = router.route_runs("患者发热。")[0]

    assert run.language == "zh"
    assert run.confidence == pytest.approx(0.98)
    assert run.source == "pycld2"


def test_document_decision_exposes_only_non_dominant_run_overrides():
    decision = LanguageRouter(use_optional_lid=False).route(
        "Patient stable. 患者发热。"
    )

    assert decision.language == "en"
    assert decision.dominant_pack.code == "en"
    assert decision.overrides
    assert {run.language for run in decision.overrides} == {"zh"}


_ROUTING_LATIN = "Patient stable."
_ROUTING_DEVANAGARI = "\u0930\u094b\u0917\u0940"
_ROUTING_TAMIL = "\u0b95\u0bcd\u0bb7"
_ROUTING_ARABIC = "\u0639\u0644\u064a"


def test_run_metadata_is_complete_for_every_routed_run():
    text = " ".join((_ROUTING_LATIN, _ROUTING_DEVANAGARI, _ROUTING_TAMIL))
    runs = LanguageRouter(use_optional_lid=False).route_runs(text)

    assert runs
    for run in runs:
        assert run.candidates
        assert run.language in run.candidates or run.script == "Unknown"
        assert run.normalizer in {"indic-nfc", "unicode-defense"}
        assert run.tokenizer
        assert run.numeral_set
    by_script = {run.script: run for run in runs}
    assert by_script["Latin"].normalizer == "unicode-defense"
    assert by_script["Latin"].numeral_set == "ascii"
    assert by_script["Devanagari"].normalizer == "indic-nfc"
    assert by_script["Devanagari"].numeral_set == "devanagari"
    assert by_script["Tamil"].normalizer == "indic-nfc"
    assert by_script["Tamil"].numeral_set == "tamil"


def test_routed_run_boundaries_are_grapheme_aligned():
    text = _ROUTING_LATIN + _ROUTING_ARABIC + "\u0951" + _ROUTING_DEVANAGARI
    runs = LanguageRouter(use_optional_lid=False).route_runs(text)
    boundaries = {start for start, _ in iter_grapheme_cluster_spans(text)} | {len(text)}

    for run in runs:
        assert run.start in boundaries
        assert run.end in boundaries
    assert "".join(text[run.start : run.end] for run in runs) == text


def test_route_runs_is_deterministic():
    text = " ".join((_ROUTING_LATIN, _ROUTING_DEVANAGARI, _ROUTING_ARABIC))
    router = LanguageRouter(use_optional_lid=False)

    assert router.route_runs(text) == router.route_runs(text)
