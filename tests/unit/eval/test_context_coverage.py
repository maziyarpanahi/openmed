"""Tests for the multilingual ConText coverage summary and its gate."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openmed.clinical.lexicons import (
    ClinicalCueLexicon,
    available_clinical_cue_languages,
    context_cues,
    register_clinical_cue_lexicon,
)
from openmed.eval import (
    CONTEXT_COVERAGE,
    CUE_AXES,
    REASON_EMPTY_AXIS_CUE_PACK,
    REASON_FIXTURE_LANGUAGE_NOT_REGISTERED,
    REASON_MISSING_REQUIRED_TRAP,
    REASON_REGISTERED_WITHOUT_FIXTURES,
    REQUIRED_CUE_AXES,
    REQUIRED_TRAPS,
    SCORED_AXES,
    assert_context_coverage_gate,
    builtin_clinical_cue_languages,
    context_coverage_metadata,
    load_context_multilingual_fixtures,
    run_context_coverage,
    run_context_multilingual_eval,
)
from openmed.eval.context_coverage import FALLBACK_LANGUAGE

_META = {
    "kind": "meta",
    "suite": "context_multilingual",
    "version": 1,
    "synthetic": True,
    "axes": ["negation", "temporality", "certainty"],
}


@pytest.fixture
def cue_registry() -> object:
    """Snapshot and restore the process-global cue registry around a test."""

    snapshot = dict(context_cues._LEXICONS)
    try:
        yield context_cues
    finally:
        context_cues._LEXICONS.clear()
        context_cues._LEXICONS.update(snapshot)


def _synthetic_pack(code: str) -> ClinicalCueLexicon:
    """Build a synthetic, algorithmically generated cue pack for *code*."""

    return ClinicalCueLexicon(
        language=code,
        negation=(f"{code} zz-neg",),
        pseudo_negation=(f"{code} zz-neg zz-pseudo",),
        historical=(f"{code} zz-old",),
        hypothetical=(f"{code} zz-if",),
        recent=(f"{code} zz-now",),
        uncertainty=(f"{code} zz-maybe",),
        backward=(f"{code} zz-done",),
        scope_terminators=(f"{code} zz-stop",),
        conjunction_terminators=(f"{code} zz-stop",),
    )


def _write_fixture(path: Path, rows: list[dict[str, object]]) -> Path:
    lines = [json.dumps(_META, ensure_ascii=False)]
    lines.extend(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _row(case_id: str, language: str, trap: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "language": language,
        "synthetic": True,
        "trap": trap,
        "text": "zz-target is present today.",
        "target": {"text": "zz-target"},
        "expected": {
            "negation": "affirmed",
            "temporality": "recent",
            "certainty": "certain",
        },
    }


def test_coverage_report_is_deterministic_and_summarises_every_language() -> None:
    report = run_context_coverage()

    assert report == run_context_coverage()
    assert report.to_dict()["suite"] == CONTEXT_COVERAGE
    assert report.eval_suite == "context_multilingual"

    _, fixtures = load_context_multilingual_fixtures()
    assert report.fixture_count == len(fixtures)
    assert report.languages == tuple(sorted(report.languages))
    assert set(builtin_clinical_cue_languages()) <= set(report.languages)

    for coverage in report.per_language:
        assert set(coverage.cue_counts) == set(CUE_AXES)
        assert set(coverage.macro_f1) == set(SCORED_AXES)
        assert coverage.total_cues == sum(coverage.cue_counts.values())
        if coverage.declared:
            assert coverage.fixture_case_count > 0
            assert coverage.total_cues > 0
            for axis in REQUIRED_CUE_AXES:
                assert coverage.cue_counts[axis] > 0
            for axis in SCORED_AXES:
                assert coverage.macro_f1[axis] is not None
                assert coverage.macro_f1[axis] >= report.thresholds[axis]


def test_committed_fixture_passes_the_gate() -> None:
    report = assert_context_coverage_gate()

    assert report.passed is True
    assert report.failures == ()
    assert report.failure_reasons() == ()


def test_registered_language_without_fixtures_fails_gate(cue_registry) -> None:
    """A cue pack nobody exercises must fail closed.

    This is the failure mode the shipped eval cannot see: its
    ``context_gate_passed`` flag is computed only over languages that have
    fixture rows, so a pack with zero fixtures leaves it green while
    ``context_lexicon_coverage`` advertises healthy cue counts. The first two
    assertions pin that stale behaviour so this test fails if the coverage gate
    is ever reimplemented by delegating to it.
    """

    register_clinical_cue_lexicon(_synthetic_pack("qa"))
    declared = (*builtin_clinical_cue_languages(), "qa")

    legacy = run_context_multilingual_eval()
    assert legacy.metrics["context_gate_passed"] is True
    assert legacy.metrics["context_lexicon_coverage"]["qa"]["negation"] > 0
    assert "qa" not in legacy.metrics["context_macro_f1"]

    report = run_context_coverage(languages=declared)

    assert report.passed is False
    assert REASON_REGISTERED_WITHOUT_FIXTURES in report.failure_reasons()

    offenders = [
        failure
        for failure in report.failures
        if failure.reason == REASON_REGISTERED_WITHOUT_FIXTURES
    ]
    assert [failure.language for failure in offenders] == ["qa"]
    assert offenders[0].detail["fixture_case_count"] == 0
    assert offenders[0].detail["registered"] is True
    assert offenders[0].detail["total_cues"] > 0

    coverage = report.coverage_for("qa")
    assert coverage is not None
    assert coverage.declared is True
    assert coverage.registered is True
    assert coverage.fixture_case_count == 0
    assert coverage.missing_traps == REQUIRED_TRAPS
    assert coverage.meets_thresholds is False

    with pytest.raises(AssertionError, match="qa:registered_without_fixtures"):
        assert_context_coverage_gate(languages=declared)


def test_empty_fixture_does_not_score_one(tmp_path: Path) -> None:
    """Zero fixture rows must never macro-average to a perfect score.

    The harness macro-F1 helper returns ``1.0`` when both label sequences are
    empty, so an untested language would otherwise be reported as flawless.
    """

    fixture = _write_fixture(tmp_path / "empty.jsonl", [])

    legacy = run_context_multilingual_eval(fixture)
    assert legacy.fixture_count == 0
    assert legacy.metrics["context_macro_f1"] == {}
    assert legacy.metrics["context_gate_passed"] is True

    report = run_context_coverage(fixture)

    assert report.fixture_count == 0
    assert report.passed is False
    assert report.failure_reasons() == (REASON_REGISTERED_WITHOUT_FIXTURES,)
    assert report.per_language

    for coverage in report.per_language:
        assert coverage.fixture_case_count == 0
        assert coverage.traps == ()
        assert all(coverage.macro_f1[axis] is None for axis in SCORED_AXES)
        assert 1.0 not in set(coverage.macro_f1.values())
        assert coverage.meets_thresholds is False

    markdown = report.to_markdown()
    assert "n/a" in markdown
    assert "1.0000" not in markdown


def test_declared_languages_ignore_runtime_registrations(cue_registry) -> None:
    """The gate's authority must not depend on import or test ordering."""

    before = builtin_clinical_cue_languages()
    register_clinical_cue_lexicon(_synthetic_pack("qb"))

    assert "qb" in available_clinical_cue_languages()
    assert builtin_clinical_cue_languages() == before
    assert "qb" not in before

    report = run_context_coverage()

    assert report.passed is True
    assert report.declared_languages == before
    assert "qb" in report.unscored_languages


def test_unknown_fixture_language_is_reported_not_crashed(tmp_path: Path) -> None:
    rows = [_row(f"qq-{trap}", "qq", trap) for trap in REQUIRED_TRAPS]
    fixture = _write_fixture(tmp_path / "unknown.jsonl", rows)

    report = run_context_coverage(fixture, languages=())

    assert report.declared_languages == ()
    assert report.languages == ("qq",)
    assert report.passed is False
    assert REASON_FIXTURE_LANGUAGE_NOT_REGISTERED in report.failure_reasons()

    coverage = report.coverage_for("qq")
    assert coverage is not None
    assert coverage.declared is False
    assert coverage.registered is False
    assert coverage.fixture_case_count == len(REQUIRED_TRAPS)
    assert coverage.total_cues == 0
    assert coverage.cue_counts == dict.fromkeys(CUE_AXES, 0)


def test_missing_trap_and_empty_cue_axis_are_reported(
    cue_registry, tmp_path: Path
) -> None:
    register_clinical_cue_lexicon(
        ClinicalCueLexicon(
            language="qc",
            negation=(),
            pseudo_negation=(),
            historical=(),
            hypothetical=(),
            recent=(),
            uncertainty=(),
            backward=(),
            scope_terminators=(),
            conjunction_terminators=(),
        )
    )
    fixture = _write_fixture(
        tmp_path / "partial.jsonl", [_row("qc-1", "qc", "affirmed")]
    )

    report = run_context_coverage(fixture, languages=("qc",))

    assert report.passed is False
    reasons = report.failure_reasons()
    assert REASON_MISSING_REQUIRED_TRAP in reasons
    assert REASON_EMPTY_AXIS_CUE_PACK in reasons
    assert REASON_REGISTERED_WITHOUT_FIXTURES not in reasons

    coverage = report.coverage_for("qc")
    assert coverage is not None
    assert coverage.traps == ("affirmed",)
    assert set(coverage.missing_traps) == set(REQUIRED_TRAPS) - {"affirmed"}


def test_markdown_summary_is_byte_stable_and_sorted(tmp_path: Path) -> None:
    report = run_context_coverage()
    markdown = report.to_markdown()

    assert markdown == report.to_markdown()
    assert markdown.startswith("# Multilingual ConText Coverage\n")
    assert markdown.endswith("\n")
    assert "| Verdict | `pass` |" in markdown
    assert "## Coverage by Language" in markdown
    assert "## Failures" in markdown

    rendered = [
        line.split("|")[1].strip().strip("`")
        for line in markdown.splitlines()
        if line.startswith("| `")
    ]
    languages = [code for code in rendered if code in report.languages]
    assert languages == sorted(languages)
    assert set(languages) == set(report.languages)

    # A language may exercise trap kinds beyond the required set; the rendered
    # cell counts required traps only and must never exceed its denominator.
    trap_cells = re.findall(r"\| (\d+)/(\d+) \|", markdown)
    assert len(trap_cells) == len(report.per_language)
    for covered, required in trap_cells:
        assert int(required) == len(REQUIRED_TRAPS)
        assert int(covered) <= int(required)

    written = report.write_markdown(tmp_path / "coverage.md")
    assert written.read_text(encoding="utf-8") == markdown


def test_markdown_states_the_fallback_and_the_scope_boundary() -> None:
    """A language absent from the table must not read as missing work.

    A code with no cue pack of its own resolves through the English pack, so
    its absence is a deliberate fallback rather than a coverage gap, and this
    summary reports fixture coverage rather than cue quality. Both statements
    must reach the rendered artifact, not just the docstring.
    """

    from openmed.clinical.context import resolve_span_context
    from openmed.clinical.lexicons import get_clinical_cue_lexicon

    absent = "ar"
    assert absent not in builtin_clinical_cue_languages()
    # The claim the note makes is true: the absent code still resolves.
    assert get_clinical_cue_lexicon(absent).language == FALLBACK_LANGUAGE
    assert resolve_span_context("possible pneumonia", language=absent).negation

    markdown = run_context_coverage().to_markdown()

    assert f"`{FALLBACK_LANGUAGE}` fallback" in markdown
    assert "not a coverage gap" in markdown
    assert "not\nhow well its cues perform" in markdown or (
        "not how well its cues perform" in markdown
    )


def test_report_is_raw_text_free() -> None:
    report = run_context_coverage()
    serialized = json.dumps(report.to_dict(), ensure_ascii=False)

    _, fixtures = load_context_multilingual_fixtures()
    surfaces = {str(row["text"]) for row in fixtures}
    assert surfaces
    for surface in surfaces:
        assert surface not in serialized


def test_metadata_declares_the_coverage_contract() -> None:
    metadata = context_coverage_metadata()

    assert metadata == {
        "suite": CONTEXT_COVERAGE,
        "schema_version": 1,
        "eval_suite": "context_multilingual",
        "synthetic": True,
        "scored_axes": list(SCORED_AXES),
        "cue_axes": list(CUE_AXES),
        "required_traps": list(REQUIRED_TRAPS),
        "required_cue_axes": list(REQUIRED_CUE_AXES),
        "fallback_language": FALLBACK_LANGUAGE,
    }


def test_harness_entry_points_remain_backward_compatible() -> None:
    report = run_context_multilingual_eval()

    assert report.suite == "context_multilingual"
    assert report.model_name == "deterministic-context"
    assert report.device == "local"
    assert set(report.metrics) == {
        "context_macro_f1",
        "context_thresholds",
        "context_gate_passed",
        "context_lexicon_coverage",
    }
    assert report.metadata["parent_issue"] == "OM-724"
    assert report.metrics["context_gate_passed"] is True
