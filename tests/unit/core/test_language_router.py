"""Acceptance coverage for exact-offset language and script routing."""

from __future__ import annotations

import importlib
import subprocess
import sys
from types import MappingProxyType

import pytest

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
