"""Per-language scorecard tests for multilingual relation extraction."""

from __future__ import annotations

from openmed.clinical import extract_relations
from openmed.eval.harness import run_relation_benchmark
from openmed.eval.scorecard import ModelScorecard
from openmed.eval.suites.relations import load_multilingual_relation_fixtures


def test_relation_harness_and_model_scorecard_report_each_language() -> None:
    fixtures = load_multilingual_relation_fixtures()

    def runner(fixture, model_name, device):
        assert model_name == "deterministic-multilingual-relations"
        assert device == "cpu"
        return extract_relations(
            fixture.text,
            fixture.entities.values(),
            language=fixture.language,
        )

    report = run_relation_benchmark(
        fixtures,
        suite="relations-i18n",
        model_name="deterministic-multilingual-relations",
        runner=runner,
        ci_resamples=20,
        ci_seed=17,
    )

    per_language = report.metrics["relation_extraction"]["per_language"]
    assert per_language["hi"]["strict"]["f1"] == 1.0
    assert per_language["zh"]["strict"]["f1"] == 1.0

    scorecard = ModelScorecard.from_reports([report])
    row = scorecard.to_dict()["device_tiers"][0]
    assert row["relation_per_language_f1"] == {
        "hi": {"relaxed": 1.0, "strict": 1.0},
        "zh": {"relaxed": 1.0, "strict": 1.0},
    }
    markdown = scorecard.to_markdown()
    assert "Per-Language RE-F1" in markdown
    assert "hi: strict 100.00%, relaxed 100.00%" in markdown
