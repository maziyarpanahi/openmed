"""Tests for the grounding-accuracy eval suite and release gate.

All gold content is synthetic and algorithmically generated; no real patient
data or licensed terminology (UMLS/SNOMED) is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from openmed.clinical.grounding.types import Candidate
from openmed.eval import release_gates as rg
from openmed.eval.grounding_accuracy import (
    DEFAULT_GROUNDING_GOLD_DIR,
    GROUNDING_ACCURACY_LANGUAGES,
    PERMISSIVE_GROUNDING_SYSTEMS,
    RESTRICTED_VOCAB_MARKERS,
    GroundingConcept,
    GroundingGold,
    GroundingMention,
    LanguageGroundingAccuracy,
    SystemGroundingAccuracy,
    evaluate_grounding_accuracy,
    format_grounding_accuracy_table,
    load_grounding_gold,
    restricted_vocab_markers_in,
)

_GOLD = load_grounding_gold()


# ---------------------------------------------------------------------------
# Gold set: size, provenance, license policy
# ---------------------------------------------------------------------------


def test_each_system_has_at_least_150_groundable_pairs():
    assert set(_GOLD) == set(PERMISSIVE_GROUNDING_SYSTEMS)
    for system, gold in _GOLD.items():
        assert gold.groundable_pair_count >= 150, system


def test_gold_is_marked_synthetic_and_permissively_licensed():
    for gold in _GOLD.values():
        for concept in gold.concepts:
            assert concept.metadata.get("synthetic") is True
            assert concept.metadata.get("license") == "CC0-1.0"


def test_license_policy_no_restricted_vocabulary_content():
    # No UMLS/SNOMED-derived strings anywhere in the shipped gold.
    assert restricted_vocab_markers_in(_GOLD) == []
    raw = "\n".join(
        (DEFAULT_GROUNDING_GOLD_DIR / f"{system}.jsonl").read_text(encoding="utf-8")
        for system in PERMISSIVE_GROUNDING_SYSTEMS
    ).lower()
    for marker in RESTRICTED_VOCAB_MARKERS:
        assert marker not in raw


def test_dataset_card_documents_synthetic_provenance():
    card = (DEFAULT_GROUNDING_GOLD_DIR / "README.md").read_text(encoding="utf-8")
    assert "synthetic" in card.lower()
    assert "CC0-1.0" in card
    assert "SNOMED" in card and "UMLS" in card


def test_gold_files_are_byte_stable_across_reads():
    first = load_grounding_gold()
    assert first.keys() == _GOLD.keys()
    for system in PERMISSIVE_GROUNDING_SYSTEMS:
        assert [c.code for c in first[system].concepts] == [
            c.code for c in _GOLD[system].concepts
        ]


# ---------------------------------------------------------------------------
# Metric math with hand-computed fixtures
# ---------------------------------------------------------------------------


def _mini_gold() -> dict[str, GroundingGold]:
    concept = GroundingConcept(
        system="rxnorm",
        code="RX1",
        preferred_term="alpha tablet",
        synonyms=("alpha tab",),
        language_aliases={},
        mentions=(
            GroundingMention("alpha tablet", "en", "RX1", True, "exact"),
            GroundingMention("alpha tab", "en", "RX1", True, "exact"),
            GroundingMention("beta", "en", "RX1", True, "fuzzy"),
            GroundingMention("gamma tablet", "en", "RX1", True, "fuzzy"),
        ),
        metadata={"synthetic": True},
    )
    return {"rxnorm": GroundingGold(system="rxnorm", concepts=(concept,))}


def test_metric_math_matches_hand_computed_values():
    # Deterministic stub: correct top-1 for the two exact mentions, a top-5-only
    # hit for one, and an abstention for the last.
    responses = {
        "alpha tablet": [Candidate("RXNORM", "RX1", "", 1.0)],
        "alpha tab": [Candidate("RXNORM", "RX1", "", 1.0)],
        "beta": [
            Candidate("RXNORM", "RXX", "", 0.9),
            Candidate("RXNORM", "RX1", "", 0.8),
        ],
        "gamma tablet": [],
    }

    def provider(mention, system, language, k):
        return responses[mention]

    report = evaluate_grounding_accuracy(_mini_gold(), provider=provider)
    accuracy = report.system("rxnorm").language("en")
    assert accuracy.support == 4
    assert accuracy.top1_accuracy == pytest.approx(2 / 4)
    assert accuracy.top5_accuracy == pytest.approx(3 / 4)
    assert accuracy.abstention_rate == pytest.approx(1 / 4)


def test_report_serializes_standard_benchmark_schema():
    report = evaluate_grounding_accuracy(_mini_gold(), provider=lambda *_: [])
    benchmark = report.to_benchmark_report()
    assert benchmark.suite == "grounding_accuracy"
    assert benchmark.metadata["synthetic"] is True
    payload = json.loads(benchmark.to_json())
    assert payload["metrics"]["systems"]["rxnorm"]["en"]["support"] == 4


# ---------------------------------------------------------------------------
# Scoring the shipped gold with the real generator
# ---------------------------------------------------------------------------


def test_not_groundable_mentions_are_scored_for_correct_abstention():
    # The not-groundable (truly unmappable) surfaces must be exercised and the
    # generator must correctly abstain on them (return no code).
    report = evaluate_grounding_accuracy()
    scored = False
    for system in report.systems.values():
        for accuracy in system.languages.values():
            if accuracy.not_groundable_support:
                scored = True
                assert accuracy.correct_abstention_rate == 1.0
    assert scored  # the shipped gold actually exercises the abstention path


def test_shipped_gold_meets_all_floors_with_real_generator():
    report = evaluate_grounding_accuracy()
    config = rg.GroundingGateConfig()
    for system in PERMISSIVE_GROUNDING_SYSTEMS:
        english = report.system(system).language("en")
        assert english.top1_accuracy >= config.english_top1_floor, system
        assert english.top5_accuracy >= config.english_top5_floor, system
        for language in config.multilingual_languages:
            metrics = report.system(system).language(language)
            assert metrics.top1_accuracy >= config.multilingual_top1_floor
            assert metrics.top5_accuracy >= config.multilingual_top5_floor


def test_every_language_is_scored_for_every_system():
    report = evaluate_grounding_accuracy()
    for system in PERMISSIVE_GROUNDING_SYSTEMS:
        for language in GROUNDING_ACCURACY_LANGUAGES:
            assert report.system(system).language(language) is not None


def test_shipped_gold_gate_passes_and_report_is_deterministic():
    first = evaluate_grounding_accuracy().to_benchmark_report().to_json()
    second = evaluate_grounding_accuracy().to_benchmark_report().to_json()
    assert first == second
    checks = rg.evaluate_grounding_accuracy_gate()
    assert {check.gate for check in checks} == {
        f"grounding_accuracy:{system}" for system in PERMISSIVE_GROUNDING_SYSTEMS
    }
    assert all(check.passed for check in checks)


# ---------------------------------------------------------------------------
# Regression injection: a broken linker must trip the gate
# ---------------------------------------------------------------------------


def _broken_provider(mention, system, language, k):
    # Deliberately wrong crosswalk: always emit a code that is never expected.
    return [Candidate(system.upper(), "WRONG-CODE", "wrong", 1.0)]


def test_broken_linker_trips_every_system_gate():
    report = evaluate_grounding_accuracy(provider=_broken_provider)
    checks = rg.evaluate_grounding_accuracy_gate(report)
    assert checks
    for check in checks:
        assert check.passed is False
        assert "below floor" in check.reason
    gate_report = rg.build_grounding_gate_report(report)
    assert gate_report.decision == rg.QUARANTINED


def test_cli_grounding_gate_fails_closed_on_regression(tmp_path, monkeypatch):
    from openmed.eval import grounding_accuracy as grounding_module

    broken = grounding_module.evaluate_grounding_accuracy(provider=_broken_provider)
    monkeypatch.setattr(
        grounding_module, "evaluate_grounding_accuracy", lambda *a, **k: broken
    )

    output = tmp_path / "grounding-gate.json"
    exit_code = rg.main(["--grounding", "--output", str(output)])
    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"] == rg.QUARANTINED
    assert any(not check["passed"] for check in payload["gate_results"])


def test_cli_grounding_gate_passes_on_shipped_gold(tmp_path):
    output = tmp_path / "grounding-gate.json"
    exit_code = rg.main(["--grounding", "--output", str(output)])
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"] == rg.RELEASABLE


# ---------------------------------------------------------------------------
# GateReport round-trips through sign()/verify() with grounding checks
# ---------------------------------------------------------------------------


def test_gate_report_round_trips_sign_and_verify():
    report = evaluate_grounding_accuracy()
    gate_report = rg.build_grounding_gate_report(report)
    gate_names = {check.gate for check in gate_report.gate_results}
    assert gate_names == {
        f"grounding_accuracy:{system}" for system in PERMISSIVE_GROUNDING_SYSTEMS
    }

    signed = gate_report.sign("unit-test-key")
    assert signed.verify("unit-test-key") is True
    assert signed.verify("other-key") is False

    restored = rg.GateReport.from_json(signed.to_json())
    assert restored.verify("unit-test-key") is True
    assert {check.gate for check in restored.gate_results} == gate_names


# ---------------------------------------------------------------------------
# CLI table hook
# ---------------------------------------------------------------------------


def test_format_table_lists_every_system_and_language():
    report = evaluate_grounding_accuracy()
    table = format_grounding_accuracy_table(report)
    for system in PERMISSIVE_GROUNDING_SYSTEMS:
        assert system in table
    assert "top1" in table and "top5" in table
    assert "abstain" in table


def test_cli_grounding_command_prints_table(capsys):
    from openmed.cli.gates import handle_gates_grounding

    args = argparse.Namespace(strict=False, format="text", output=None)
    exit_code = handle_gates_grounding(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "rxnorm" in out and "icd10cm" in out


def test_cli_grounding_command_strict_passes_on_shipped_gold():
    from openmed.cli.gates import handle_gates_grounding

    args = argparse.Namespace(strict=True, format="text", output=None)
    assert handle_gates_grounding(args) == 0
