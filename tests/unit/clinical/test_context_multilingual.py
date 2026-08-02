"""Multilingual ConText cue lexicon tests for OM-724-1."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

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
    get_clinical_cue_lexicon,
    register_clinical_cue_lexicon,
)
from openmed.eval.harness import (
    DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE,
    load_context_multilingual_fixtures,
    run_context_multilingual_eval,
)

FORBIDDEN_FIXTURE_MARKERS = ("cpt", "dua", "i2b2", "mimic", "n2c2", "snomed", "umls")
REQUIRED_LANGUAGES = {"en", "es", "fr", "de", "zh", "hi"}
PORTUGUESE = "pt"
CONTEXT_GUIDE = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "clinical"
    / "context-and-extraction.md"
)


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


def test_multilingual_context_docs_cover_safe_contribution_contract() -> None:
    guide = CONTEXT_GUIDE.read_text(encoding="utf-8")
    normalized_guide = re.sub(r"\s+", " ", guide)

    for field in ClinicalCueLexicon.__dataclass_fields__:
        assert f"`{field}`" in guide

    for requirement in (
        "Extending the NegEx Lexicon for Multiple Languages",
        "pyConTextSwe",
        "## Language-Pack Review Checklist",
        "context_multilingual.jsonl",
        '"synthetic": true',
        "No raw PHI",
        "must not require edits to resolver or scanner logic",
    ):
        assert requirement in normalized_guide


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


# --- Portuguese language pack (OM-724-3) ------------------------------------

# Every shipped pt cue needs a behavioral regression case. The table is the
# single source of truth for that, and the exact-set gate below fails when a
# cue is added to the pack without one. `recent` is excluded from the gate on
# purpose: those cues reach only the section-prior suppression path, which does
# not take a language argument yet, so no assertion over them could fail.
# cue, sentence, target, expected negation/temporality/certainty
PORTUGUESE_CUE_CASES = (
    (
        "sem evidência de",
        "Sem evidência de pneumonia hoje.",
        "pneumonia",
        NEGATED,
        RECENT,
        CERTAIN,
    ),
    (
        "sem evidencia de",
        "Sem evidencia de pneumonia hoje.",
        "pneumonia",
        NEGATED,
        RECENT,
        CERTAIN,
    ),
    (
        "sem sinais de",
        "Sem sinais de pneumonia hoje.",
        "pneumonia",
        NEGATED,
        RECENT,
        CERTAIN,
    ),
    (
        "nenhum sinal de",
        "Nenhum sinal de pneumonia.",
        "pneumonia",
        NEGATED,
        RECENT,
        CERTAIN,
    ),
    (
        "negativo para",
        "Exame negativo para pneumonia.",
        "pneumonia",
        NEGATED,
        RECENT,
        CERTAIN,
    ),
    (
        "ausência de",
        "Ausência de pneumonia no exame.",
        "pneumonia",
        NEGATED,
        RECENT,
        CERTAIN,
    ),
    (
        "ausencia de",
        "Ausencia de pneumonia no exame.",
        "pneumonia",
        NEGATED,
        RECENT,
        CERTAIN,
    ),
    ("livre de", "Livre de pneumonia.", "pneumonia", NEGATED, RECENT, CERTAIN),
    ("descartada", "Pneumonia descartada.", "Pneumonia", NEGATED, RECENT, CERTAIN),
    (
        "descartado",
        "Derrame pleural descartado.",
        "Derrame pleural",
        NEGATED,
        RECENT,
        CERTAIN,
    ),
    ("afastada", "Pneumonia afastada.", "Pneumonia", NEGATED, RECENT, CERTAIN),
    (
        "afastado",
        "Derrame pleural afastado.",
        "Derrame pleural",
        NEGATED,
        RECENT,
        CERTAIN,
    ),
    ("nega", "Paciente nega pneumonia.", "pneumonia", NEGATED, RECENT, CERTAIN),
    ("negou", "Paciente negou pneumonia.", "pneumonia", NEGATED, RECENT, CERTAIN),
    (
        "não apresenta",
        "Paciente não apresenta pneumonia.",
        "pneumonia",
        NEGATED,
        RECENT,
        CERTAIN,
    ),
    (
        "nao apresenta",
        "Paciente nao apresenta pneumonia.",
        "pneumonia",
        NEGATED,
        RECENT,
        CERTAIN,
    ),
    ("sem", "Paciente sem pneumonia.", "pneumonia", NEGATED, RECENT, CERTAIN),
    ("não", "Não pneumonia.", "pneumonia", NEGATED, RECENT, CERTAIN),
    ("nao", "Nao pneumonia.", "pneumonia", NEGATED, RECENT, CERTAIN),
    # Hedges: the embedded negation is masked, so these stay affirmed but hedged.
    (
        "não pode ser descartada",
        "Pneumonia não pode ser descartada.",
        "Pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "nao pode ser descartada",
        "Pneumonia nao pode ser descartada.",
        "Pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "não pode ser descartado",
        "Derrame pleural não pode ser descartado.",
        "Derrame pleural",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "nao pode ser descartado",
        "Derrame pleural nao pode ser descartado.",
        "Derrame pleural",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "não se pode descartar",
        "Não se pode descartar pneumonia.",
        "pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "nao se pode descartar",
        "Nao se pode descartar pneumonia.",
        "pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "não pode ser excluída",
        "Pneumonia não pode ser excluída.",
        "Pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "nao pode ser excluida",
        "Pneumonia nao pode ser excluida.",
        "Pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "não pode ser excluído",
        "Derrame pleural não pode ser excluído.",
        "Derrame pleural",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "nao pode ser excluido",
        "Derrame pleural nao pode ser excluido.",
        "Derrame pleural",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "não se pode excluir",
        "Não se pode excluir pneumonia.",
        "pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "nao se pode excluir",
        "Nao se pode excluir pneumonia.",
        "pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "não se exclui",
        "Não se exclui pneumonia.",
        "pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "nao se exclui",
        "Nao se exclui pneumonia.",
        "pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "não se afasta a hipótese de",
        "Não se afasta a hipótese de pneumonia.",
        "pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "nao se afasta a hipotese de",
        "Nao se afasta a hipotese de pneumonia.",
        "pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    # Other hedges.
    ("suspeita de", "Suspeita de pneumonia.", "pneumonia", AFFIRMED, RECENT, UNCERTAIN),
    (
        "suspeito de",
        "Quadro suspeito de pneumonia.",
        "pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    ("possível", "Possível pneumonia.", "pneumonia", AFFIRMED, RECENT, UNCERTAIN),
    ("possivel", "Possivel pneumonia.", "pneumonia", AFFIRMED, RECENT, UNCERTAIN),
    ("provável", "Provável pneumonia.", "pneumonia", AFFIRMED, RECENT, UNCERTAIN),
    ("provavel", "Provavel pneumonia.", "pneumonia", AFFIRMED, RECENT, UNCERTAIN),
    ("improvável", "Pneumonia improvável.", "Pneumonia", AFFIRMED, RECENT, UNCERTAIN),
    ("improvavel", "Pneumonia improvavel.", "Pneumonia", AFFIRMED, RECENT, UNCERTAIN),
    (
        "para descartar",
        "Exame solicitado para descartar pneumonia.",
        "pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    (
        "pode",
        "O paciente pode ter pneumonia.",
        "pneumonia",
        AFFIRMED,
        RECENT,
        UNCERTAIN,
    ),
    # Temporality.
    (
        "antecedente de",
        "Antecedente de pneumonia documentado.",
        "pneumonia",
        AFFIRMED,
        HISTORICAL,
        CERTAIN,
    ),
    (
        "antecedentes de",
        "Antecedentes de pneumonia documentados.",
        "pneumonia",
        AFFIRMED,
        HISTORICAL,
        CERTAIN,
    ),
    (
        "história de",
        "História de pneumonia documentada.",
        "pneumonia",
        AFFIRMED,
        HISTORICAL,
        CERTAIN,
    ),
    (
        "historia de",
        "Historia de pneumonia documentada.",
        "pneumonia",
        AFFIRMED,
        HISTORICAL,
        CERTAIN,
    ),
    (
        "anteriormente",
        "Pneumonia tratada anteriormente.",
        "Pneumonia",
        AFFIRMED,
        HISTORICAL,
        CERTAIN,
    ),
    ("resolvida", "Pneumonia resolvida.", "Pneumonia", AFFIRMED, HISTORICAL, CERTAIN),
    (
        "resolvido",
        "Derrame pleural resolvido.",
        "Derrame pleural",
        AFFIRMED,
        HISTORICAL,
        CERTAIN,
    ),
    (
        "em caso de",
        "Em caso de pneumonia, contatar a clínica.",
        "pneumonia",
        AFFIRMED,
        HYPOTHETICAL,
        UNCERTAIN,
    ),
    (
        "se houver",
        "Se houver pneumonia, contatar a clínica.",
        "pneumonia",
        AFFIRMED,
        HYPOTHETICAL,
        UNCERTAIN,
    ),
    (
        "a menos que",
        "A menos que pneumonia retorne, manter conduta.",
        "pneumonia",
        AFFIRMED,
        HYPOTHETICAL,
        UNCERTAIN,
    ),
)

PORTUGUESE_GATED_AXES = (
    "negation",
    "pseudo_negation",
    "historical",
    "hypothetical",
    "uncertainty",
    "backward",
)


def _normalized(cue: str) -> str:
    return " ".join(cue.casefold().split())


def _pt_span(text: str, target: str) -> dict[str, object]:
    start = text.index(target)
    return {
        "text": target,
        "context": text,
        "start": start,
        "end": start + len(target),
    }


@pytest.mark.parametrize(
    ("cue", "text", "target", "negation", "temporality", "certainty"),
    PORTUGUESE_CUE_CASES,
    ids=[case[0] for case in PORTUGUESE_CUE_CASES],
)
def test_portuguese_cue_resolves_expected_axes(
    cue: str,
    text: str,
    target: str,
    negation: str,
    temporality: str,
    certainty: str,
) -> None:
    context = resolve_span_context(_pt_span(text, target), language=PORTUGUESE)

    assert (context.negation, context.temporality, context.certainty) == (
        negation,
        temporality,
        certainty,
    ), cue


def test_every_portuguese_cue_has_a_regression_case() -> None:
    """A cue added to the pt pack must arrive with a behavioral test."""

    lexicon = get_clinical_cue_lexicon(PORTUGUESE)
    shipped = {
        _normalized(cue)
        for axis in PORTUGUESE_GATED_AXES
        for cue in getattr(lexicon, axis)
    }
    covered = {_normalized(case[0]) for case in PORTUGUESE_CUE_CASES}

    assert shipped == covered


def test_portuguese_recent_cues_are_documented_as_inert() -> None:
    """Recency cues cannot be asserted while section priors ignore language.

    ``apply_section_context_priors`` calls ``_has_explicit_temporality_cue``
    without a language argument, so only the English pack is consulted and no
    non-English ``recent`` entry can change an outcome. The tuple is pinned
    here so it stays small until that call threads the language through.
    """

    assert get_clinical_cue_lexicon(PORTUGUESE).recent == (
        "ativo",
        "ativa",
        "agudo",
        "aguda",
        "atual",
    )


def test_portuguese_hedges_are_not_reported_as_negation() -> None:
    """The excluir and afastar families must not leak as hard negations."""

    for text, target in (
        ("Pneumonia não pode ser excluída.", "Pneumonia"),
        ("Não se pode excluir pneumonia.", "pneumonia"),
        ("Não se afasta a hipótese de pneumonia.", "pneumonia"),
    ):
        context = resolve_span_context(_pt_span(text, target), language=PORTUGUESE)

        assert context.negation == AFFIRMED, text
        assert context.certainty == UNCERTAIN, text


def test_portuguese_pseudo_negation_masking_is_load_bearing() -> None:
    """With masking removed this hedge would count one true negation cue."""

    from openmed.clinical.context import _cue_pattern

    lexicon = get_clinical_cue_lexicon(PORTUGUESE)
    negation_re = _cue_pattern(lexicon.negation, token_boundaries=True)
    unmasked = negation_re.findall("Pneumonia não pode ser excluída.")

    # An odd count means the span would resolve as negated without masking.
    assert len(unmasked) % 2 == 1
    assert (
        resolve_negation("Pneumonia não pode ser excluída.", language=PORTUGUESE)
        == AFFIRMED
    )


def test_portuguese_accent_stripped_notes_resolve_identically() -> None:
    """Every accented cue ships an unaccented twin for accent-less notes."""

    _, rows = load_context_multilingual_fixtures()

    for row in rows:
        if row["language"] != PORTUGUESE:
            continue
        text = _strip_accents(row["text"])
        target = _strip_accents(row["target"]["text"])
        context = resolve_span_context(_pt_span(text, target), language=PORTUGUESE)

        assert context == ClinicalContextResult(
            temporality=row["expected"]["temporality"],
            certainty=row["expected"]["certainty"],
            negation=row["expected"]["negation"],
        ), row["case_id"]


@pytest.mark.parametrize("terminator", ("mas", "porém", "porem", "contudo", "embora"))
def test_portuguese_conjunction_terminators_block_cue_scope(terminator: str) -> None:
    text = f"Sem evidência de pneumonia {terminator} febre presente."
    span = _pt_span(text, "febre")

    hits = scan_context_cues(text, [span], language=PORTUGUESE)

    assert hits[span] == ()


@pytest.mark.parametrize(
    "terminator", ("e", "mas", "porém", "porem", "contudo", "ou", "embora")
)
def test_portuguese_scope_terminators_bound_the_resolver_window(
    terminator: str,
) -> None:
    text = f"Sem evidência de pneumonia {terminator} febre presente."

    assert resolve_negation(_pt_span(text, "febre"), language=PORTUGUESE) == AFFIRMED


def test_portuguese_scope_terminator_negative_case() -> None:
    """Without a terminator the same cue still reaches the later finding."""

    text = "Sem evidência de pneumonia febre."

    assert resolve_negation(_pt_span(text, "febre"), language=PORTUGUESE) == NEGATED


def _strip_accents(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if not unicodedata.combining(char)
    )
