"""Multilingual ConText cue lexicon tests for OM-724-1."""

from __future__ import annotations

import re
from pathlib import Path

from openmed.clinical import (
    AFFIRMED,
    CERTAIN,
    HISTORICAL,
    HYPOTHETICAL,
    NEGATED,
    RECENT,
    UNCERTAIN,
    assert_context_axes,
    resolve_negation,
    resolve_span_context,
    resolve_temporality,
    resolve_uncertainty,
    scan_context_cues,
)
from openmed.clinical.context import ClinicalContextResult
from openmed.clinical.lexicons import (
    ClinicalCueLexicon,
    available_clinical_cue_languages,
    clinical_context_lexicon_stats,
    register_clinical_cue_lexicon,
)
from openmed.eval.harness import (
    DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE,
    load_context_multilingual_fixtures,
    run_context_multilingual_eval,
)

FORBIDDEN_FIXTURE_MARKERS = ("cpt", "dua", "i2b2", "mimic", "n2c2", "snomed", "umls")
REQUIRED_LANGUAGES = {"en", "es", "fr", "de", "zh", "hi"}


def test_context_multilingual_fixture_is_synthetic_and_complete() -> None:
    meta, rows = load_context_multilingual_fixtures()

    assert meta["synthetic"] is True
    assert REQUIRED_LANGUAGES <= {row["language"] for row in rows}
    assert all(row.get("synthetic") is True for row in rows)
    for language in REQUIRED_LANGUAGES:
        traps = {row["trap"] for row in rows if row["language"] == language}
        assert {"affirmed", "negated", "historical", "hypothetical"} <= traps
        assert {"pseudo_negation", "double_negation"} <= traps

    fixture_text = Path(DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE).read_text(
        encoding="utf-8"
    )
    for marker in FORBIDDEN_FIXTURE_MARKERS:
        assert re.search(rf"(?<![a-z0-9]){marker}(?![a-z0-9])", fixture_text) is None


def test_resolvers_accept_language_without_breaking_english_defaults() -> None:
    assert resolve_negation("no evidence of pneumonia") == NEGATED
    assert resolve_temporality("history of MI") == HISTORICAL
    assert resolve_uncertainty("possible pneumonia") == UNCERTAIN

    assert resolve_negation("sin evidencia de neumonía", language="es") == NEGATED
    assert resolve_temporality("antécédent de pneumonie", language="fr") == HISTORICAL
    assert resolve_uncertainty("不能排除肺炎", language="zh") == UNCERTAIN


def test_multilingual_fixture_rows_match_resolver_outputs() -> None:
    _, rows = load_context_multilingual_fixtures()

    for row in rows:
        span = _span_from_row(row)
        context = resolve_span_context(span, language=row["language"])

        assert context == ClinicalContextResult(
            temporality=row["expected"]["temporality"],
            certainty=row["expected"]["certainty"],
            negation=row["expected"]["negation"],
        ), row["case_id"]


def test_pseudo_and_double_negation_are_deterministic_per_language() -> None:
    _, rows = load_context_multilingual_fixtures()

    for row in rows:
        if row["trap"] not in {"pseudo_negation", "double_negation"}:
            continue
        span = _span_from_row(row)
        assert resolve_negation(span, language=row["language"]) == AFFIRMED


def test_scanner_uses_language_specific_conjunction_terminators() -> None:
    text = "Sin evidencia de neumonía pero fiebre presente."
    span = _span(text, "fiebre")

    hits = scan_context_cues(text, [span], language="es")

    assert hits[span] == ()


def test_assert_context_axes_uses_language_pack() -> None:
    assertion = assert_context_axes(
        _span("Si la neumonía regresa, llamar a la clínica.", "neumonía"),
        language="es",
    )

    assert assertion.temporality == HYPOTHETICAL
    assert assertion.certainty == UNCERTAIN
    assert assertion.negation == AFFIRMED


def test_stub_language_pack_loads_without_resolver_logic_changes() -> None:
    register_clinical_cue_lexicon(
        ClinicalCueLexicon(
            language="xx",
            negation=("zz no",),
            pseudo_negation=("zz no maybe",),
            historical=("zz old",),
            hypothetical=("zz if",),
            recent=("zz now",),
            uncertainty=("zz maybe", "zz if"),
            backward=("zz done",),
            scope_terminators=("zz stop",),
            conjunction_terminators=("zz stop",),
        )
    )

    assert "xx" in available_clinical_cue_languages()
    assert resolve_negation("zz no fever", language="xx") == NEGATED
    assert resolve_negation("zz no maybe fever", language="xx") == AFFIRMED
    assert resolve_temporality("zz old fever", language="xx") == HISTORICAL
    assert resolve_uncertainty("zz maybe fever", language="xx") == UNCERTAIN


def test_context_multilingual_eval_gate_and_coverage_stats() -> None:
    report = run_context_multilingual_eval()
    scores = report.metrics["context_macro_f1"]

    assert report.metrics["context_gate_passed"] is True
    for language in REQUIRED_LANGUAGES:
        assert scores[language]["negation"] >= 0.90
        assert scores[language]["temporality"] >= 0.85
        assert scores[language]["uncertainty"] >= 0.85

    coverage = clinical_context_lexicon_stats()
    for language in REQUIRED_LANGUAGES:
        assert coverage[language]["negation"] > 0
        assert coverage[language]["uncertainty"] > 0


def test_unknown_language_falls_back_to_english() -> None:
    context = resolve_span_context("possible pneumonia", language="zz-unknown")

    assert context.temporality == RECENT
    assert context.certainty == UNCERTAIN
    assert context.negation == AFFIRMED


def test_language_specific_recent_values_remain_valid() -> None:
    assertion = assert_context_axes(
        _span("Heute akute Pneumonie.", "Pneumonie"),
        language="de",
    )

    assert assertion.temporality == RECENT
    assert assertion.certainty == CERTAIN
    assert assertion.negation == AFFIRMED


def _span_from_row(row: dict) -> dict[str, object]:
    return _span(row["text"], row["target"]["text"])


def _span(text: str, target: str) -> dict[str, object]:
    start = text.index(target)
    return {
        "text": target,
        "context": text,
        "start": start,
        "end": start + len(target),
    }
