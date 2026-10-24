"""Offline tests for the GLiNER2.5 language-decision evaluation harness.

The scoring and report code runs against scripted fake backends, so no model
is downloaded. The real checkpoint run lives in
``tests/integration/test_gliner2_language_decisions.py``.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pytest

from openmed.eval import language_decisions as ld
from openmed.ner.families.gliner2 import GLiNER2Handle
from openmed.structured.decision import DecisionMode, EncoderDecisionBackend

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = ROOT / "openmed" / "eval" / "golden"
SUMMARY_PATH = ROOT / "docs" / "benchmarks" / "gliner2-language-decisions.json"
FAKE_REVISION = "0123456789abcdef0123456789abcdef01234567"

GERMAN = "Der Patient hat seit gestern Fieber und Husten."
HINDI = "रोगी को बुखार है।"
TAMIL = "நோயாளி நிலையாக உள்ளார்."
HINGLISH = "Patient ka naam रवि hai."
RUSSIAN = "Пациент стабилен."
PORTUGUESE = "Paciente com febre desde ontem."
LANGUAGE_OPTIONS = ("de", "en", "hi", "ta")


class ScriptedScorer:
    """Answers each synthetic text with a fixed label, uniform scores or an error."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, tuple[str, ...], DecisionMode]] = []

    def __call__(
        self, text: str, options: tuple[str, ...], mode: DecisionMode
    ) -> tuple[float, ...]:
        self.calls.append((text, options, mode))
        answer = self.answers.get(text)
        if answer == "raise":
            raise RuntimeError("scripted backend failure")
        if answer is None:
            return tuple(1.0 / len(options) for _ in options)
        label, confidence = answer
        rest = (1.0 - confidence) / (len(options) - 1)
        return tuple(confidence if option == label else rest for option in options)


def _backend(scorer: ScriptedScorer) -> EncoderDecisionBackend:
    return EncoderDecisionBackend(
        backend_id="test.scripted.v1",
        model_id="test/scripted",
        revision=FAKE_REVISION,
        license_id="Apache-2.0",
        scorer=scorer,
        runtime="test",
    )


def _sample(
    kind: str,
    text: str,
    gold: str,
    options: tuple[str, ...],
    *,
    language: Optional[str] = None,
    mix: str = ld.MIX_NONE,
) -> ld.LanguageDecisionSample:
    return ld.LanguageDecisionSample(
        sample_id=f"{kind}:test-{len(text)}-{gold}",
        kind=kind,
        language=language or gold,
        gold=gold,
        options=options,
        source="test",
        start=0,
        end=len(text),
        mix=mix,
        text=text,
    )


def _scripted_samples() -> list[ld.LanguageDecisionSample]:
    return [
        _sample(ld.LANGUAGE, GERMAN, "de", LANGUAGE_OPTIONS),
        _sample(ld.LANGUAGE, HINDI, "hi", LANGUAGE_OPTIONS),
        _sample(ld.LANGUAGE, TAMIL, "ta", LANGUAGE_OPTIONS),
        _sample(
            ld.CODE_SWITCH,
            HINGLISH,
            "yes",
            ld.BOOLEAN_OPTIONS,
            language="hi",
            mix=ld.MIX_DECLARED,
        ),
        _sample(ld.DIALECT, PORTUGUESE, "pt_BR", ("pt_BR", "pt_PT"), language="pt"),
        _sample(ld.SCRIPT, RUSSIAN, "Cyrillic", ("Cyrillic", "Latin"), language="ru"),
    ]


def _scripted_evaluation() -> tuple[ld.LanguageDecisionEvaluation, ScriptedScorer]:
    scorer = ScriptedScorer(
        {
            GERMAN: ("German", 0.9),
            HINDI: None,
            TAMIL: ("Hindi", 0.95),
            HINGLISH: ("no", 0.8),
            PORTUGUESE: ("Brazilian Portuguese", 0.95),
            RUSSIAN: "raise",
        }
    )
    backend = _backend(scorer)
    evaluation = ld.evaluate_language_decisions(
        _scripted_samples(),
        {kind: backend for kind in ld.DECISION_KINDS},
        tasks=ld.DECISION_TASKS,
    )
    return evaluation, scorer


@pytest.fixture(scope="module")
def samples() -> tuple[ld.LanguageDecisionSample, ...]:
    return ld.build_language_decision_samples()


@pytest.fixture(scope="module")
def committed_summary() -> dict[str, Any]:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _fixture_texts() -> dict[tuple[str, str], str]:
    texts: dict[tuple[str, str], str] = {}
    paths = [*sorted((GOLDEN_DIR / "fixtures" / "i18n").glob("*.jsonl"))]
    paths += [GOLDEN_DIR / name for name in ("be.jsonl", "ch.jsonl", "es_mx.jsonl")]
    paths += [GOLDEN_DIR / "ie.jsonl"]
    paths += [GOLDEN_DIR / "fixtures" / "code_mixed_hinglish.jsonl"]
    for path in paths:
        source = path.relative_to(GOLDEN_DIR).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                texts[(source, payload["id"])] = payload["text"]
    return texts


# ---------------------------------------------------------------------------
# Evaluation set
# ---------------------------------------------------------------------------


def test_evaluation_set_is_deterministic_and_covers_every_kind(samples) -> None:
    rebuilt = ld.build_language_decision_samples()

    assert ld.dataset_digest(rebuilt) == ld.dataset_digest(samples)
    assert [sample.sample_id for sample in rebuilt] == [
        sample.sample_id for sample in samples
    ]
    counts = Counter(sample.kind for sample in samples)
    assert set(counts) == set(ld.DECISION_KINDS)
    assert len({sample.sample_id for sample in samples}) == len(samples)
    for sample in samples:
        assert sample.gold in sample.options
        assert len(set(sample.options)) == len(sample.options) >= 2
        assert len(ld.option_labels(sample)) == len(sample.options)


def test_every_sample_is_an_exact_slice_of_its_synthetic_fixture(samples) -> None:
    texts = _fixture_texts()

    for sample in samples:
        record_id = sample.sample_id.split(":", 1)[1].split("#run", 1)[0]
        source_text = texts[(sample.source, record_id)]
        assert source_text[sample.start : sample.end] == sample.text


def test_gold_labels_come_from_fixture_declarations(samples) -> None:
    by_id = {sample.sample_id: sample for sample in samples}

    # Declared code-mixing in metadata is a positive; a declared monolingual
    # register is a negative.
    assert by_id["code_switch:golden-i18n-sw-code-switched-clinical-pii"].gold == "yes"
    assert by_id["code_switch:golden-i18n-sw-swahili-clinical-pii"].gold == "no"
    assert by_id["code_switch:hinglish-clinical-context"].gold == "yes"
    assert by_id["code_switch:all-script-routing-001"].language == "mul"
    # Undeclared mixed-script notes are ambiguous and left out of code-switch.
    assert "code_switch:golden-i18n-hi-clinical-pii" not in by_id
    assert by_id["language:golden-i18n-hi-clinical-pii"].mix == ld.MIX_UNDECLARED
    # Fixture locales drive dialect gold; the pack default is always an option.
    dialect = by_id["dialect:golden-i18n-pt-nif"]
    assert (dialect.gold, dialect.options) == ("pt_PT", ("pt_BR", "pt_PT"))
    # Annotated runs carry their own script and language.
    run = by_id["script:all-script-routing-001#run1"]
    assert (run.gold, run.language) == ("Devanagari", "hi")
    # Script-conditioned candidates exclude other scripts' languages.
    scoped = by_id["language_in_script:golden-i18n-uk-clinical-pii"]
    assert scoped.options == ("bg", "ru", "sr", "uk")


def test_romanized_text_is_excluded_from_script_conditioned_candidates(
    samples,
) -> None:
    ids = {sample.sample_id for sample in samples}

    assert "language:india-clinical-hinglish-001" in ids
    assert "language_in_script:india-clinical-hinglish-001" not in ids


def test_resource_tiers_come_from_the_pack_registry() -> None:
    assert ld.resource_tier("de") == "dedicated_model"
    assert ld.resource_tier("sw") == "shared_model"
    assert ld.resource_tier("af") == "no_pack"
    assert ld.resource_tier(ld.MULTIPLE_LANGUAGES) == "multiple"


def test_every_option_has_a_model_facing_label(samples) -> None:
    for kind in (ld.LANGUAGE, ld.SCRIPT, ld.DIALECT):
        options = {
            code for sample in samples if sample.kind == kind for code in sample.options
        }
        table = {
            ld.LANGUAGE: ld.LANGUAGE_NAMES,
            ld.SCRIPT: ld.SCRIPT_LABELS,
            ld.DIALECT: ld.LOCALE_LABELS,
        }[kind]
        assert options <= set(table)
    assert set(ld.LANGUAGE_SCRIPTS) == set(ld.LANGUAGE_NAMES)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"gold": "fr"}, "gold answer is not an option"),
        ({"options": ("de",)}, "two or more distinct options"),
        ({"end": 3}, "invalid offsets"),
        ({"kind": "unknown"}, "unknown decision kind"),
        ({"mix": "maybe"}, "unknown code-mixing category"),
    ],
)
def test_samples_reject_invalid_questions_without_echoing_text(
    changes, message
) -> None:
    fields = {**_valid_fields(), **changes}

    with pytest.raises(ld.LanguageDecisionEvalError, match=message) as error:
        ld.LanguageDecisionSample(**fields)
    assert "Fieber" not in str(error.value)


def test_sample_repr_hides_text() -> None:
    sample = ld.LanguageDecisionSample(**_valid_fields())

    assert "Fieber" not in repr(sample)
    assert sample.text_digest.startswith("sha256:")
    assert "text" not in sample.manifest_entry()


def _valid_fields() -> dict[str, Any]:
    return {
        "sample_id": "language:test",
        "kind": ld.LANGUAGE,
        "language": "de",
        "gold": "de",
        "options": ("de", "en"),
        "source": "test",
        "start": 0,
        "end": len(GERMAN),
        "mix": ld.MIX_NONE,
        "text": GERMAN,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_scripted_backend_outcomes_and_policies() -> None:
    evaluation, scorer = _scripted_evaluation()
    records = {record.kind + ":" + record.gold: record for record in evaluation.records}

    german = records["language:de"]
    assert german.router == ld.RouterOutcome(prediction="en", decisive=False)
    assert (german.model.state, german.model.choice) == ("success", "de")
    assert german.policy("router_always") == (True, "en")
    assert german.policy("router_evidence") == (False, "en")
    assert german.policy("model_agreement") == (False, "en")
    assert german.policy("model_fallback") == (True, "de")

    hindi = records["language:hi"]
    assert hindi.router == ld.RouterOutcome(prediction="hi", decisive=False)
    assert hindi.model.state == "abstained"
    assert hindi.model.choice is None and hindi.model.top1 == "de"
    assert hindi.policy("model_fallback") == (False, None)

    tamil = records["language:ta"]
    assert tamil.router == ld.RouterOutcome(prediction="ta", decisive=True)
    assert tamil.policy("model_only") == (True, "hi")
    assert tamil.policy("model_fallback") == (True, "ta")

    mixed = records["code_switch:yes"]
    assert mixed.router.prediction == "yes"
    assert mixed.model.choice == "no"

    dialect = records["dialect:pt_BR"]
    assert dialect.router == ld.RouterOutcome(prediction="pt_PT", decisive=False)
    assert dialect.model.choice == "pt_BR"
    assert dialect.policy("router_evidence") == (False, "pt_PT")
    assert dialect.policy("model_fallback") == (True, "pt_BR")

    script = records["script:Cyrillic"]
    assert script.router == ld.RouterOutcome(prediction="Cyrillic", decisive=True)
    assert script.model.state == "failure" and script.model.top1 is None

    sent = {text: (options, mode) for text, options, mode in scorer.calls}
    assert sent[GERMAN] == (
        ("German", "English", "Hindi", "Tamil"),
        DecisionMode.FIXED_CHOICE,
    )
    assert sent[HINGLISH] == (("yes", "no"), DecisionMode.BOOLEAN_CHOICE)
    assert sent[PORTUGUESE][0] == ("Brazilian Portuguese", "European Portuguese")


def test_report_counts_rates_and_language_rows() -> None:
    evaluation, _ = _scripted_evaluation()
    report = evaluation.to_dict()
    language = report["kinds"][ld.LANGUAGE]
    summary = language["summary"]

    assert summary["samples"] == 3
    assert summary["router"] == {"correct": 2, "decisive": 1, "decisive_correct": 1}
    assert summary["model"] == {
        "abstained": 1,
        "agrees_with_router": 0,
        "covered": 2,
        "covered_correct": 1,
        "failed": 0,
        "top1_correct": 1,
    }
    assert summary["policies"] == {
        "model_agreement": {"accepted": 0, "correct": 0},
        "model_fallback": {"accepted": 2, "correct": 2},
        "model_only": {"accepted": 2, "correct": 1},
        "router_always": {"accepted": 3, "correct": 2},
        "router_evidence": {"accepted": 1, "correct": 1},
    }
    rates = summary["rates"]
    assert rates["router_accuracy"] == pytest.approx(0.6667)
    assert rates["model_selective_accuracy"] == 0.5
    assert rates["policies"]["model_only"]["silent_error_rate"] == pytest.approx(0.3333)
    assert rates["policies"]["model_fallback"]["silent_error_rate"] == 0.0

    # ``ta`` packs share the multilingual fallback model; ``de`` and ``hi``
    # ship dedicated models.
    assert language["low_resource"]["samples"] == 1
    assert set(language["by_tier"]["rows"]) == {"dedicated_model", "shared_model"}
    table = language["languages"]
    assert table["columns"] == list(ld.LANGUAGE_ROW_COLUMNS)
    assert table["rows"]["de"] == ["dedicated_model", 1, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    assert table["rows"]["ta"] == ["shared_model", 1, 1, 1, 0, 1, 0, 0, 0, 1, 1]

    sweep = {item["min_confidence"]: item for item in language["confidence_sweep"]}
    assert sweep[0.9]["covered"] == 2 and sweep[0.9]["correct"] == 1

    script = report["kinds"][ld.SCRIPT]["summary"]
    assert script["model"]["failed"] == 1
    assert script["rates"]["model_selective_accuracy"] is None


def test_report_metadata_is_value_free() -> None:
    evaluation, _ = _scripted_evaluation()
    report = evaluation.to_dict()
    text = ld.format_report_json(report)

    assert report["contains_source_text"] is False
    assert report["dataset"]["samples"] == 6
    assert report["router"]["lid_backend"] is None
    assert report["backends"][ld.LANGUAGE]["revision"] == FAKE_REVISION
    assert report["tasks"][ld.CODE_SWITCH] == "mixes_languages"
    assert json.loads(text) == report
    for sample in _scripted_samples():
        assert sample.text not in text
    for fragment in ("Fieber", "रोगी", "நோயாளி", "Пациент", "febre", "naam"):
        assert fragment not in text
        assert fragment not in ld.format_language_decision_summary(report)
        assert fragment not in ld.format_language_table(report)


def test_evaluation_requires_a_backend_for_every_kind() -> None:
    backend = _backend(ScriptedScorer({}))

    with pytest.raises(ld.LanguageDecisionEvalError, match="no decision backend"):
        ld.evaluate_language_decisions(_scripted_samples(), {ld.LANGUAGE: backend})
    with pytest.raises(ld.LanguageDecisionEvalError, match="requires samples"):
        ld.evaluate_language_decisions([], {ld.LANGUAGE: backend})


def test_uniform_backend_never_widens_coverage() -> None:
    backend = _backend(ScriptedScorer({}))
    samples = _scripted_samples()
    report = ld.evaluate_language_decisions(
        samples, {kind: backend for kind in ld.DECISION_KINDS}
    ).to_dict()

    for kind, sections in report["kinds"].items():
        policies = sections["summary"]["policies"]
        assert sections["summary"]["model"]["covered"] == 0
        assert policies["model_only"]["accepted"] == 0
        assert policies["model_fallback"] == policies["router_evidence"], kind


# ---------------------------------------------------------------------------
# Committed summary and reproducibility helpers
# ---------------------------------------------------------------------------


def test_committed_summary_matches_the_current_evaluation_set(
    samples, committed_summary
) -> None:
    dataset = committed_summary["dataset"]

    assert committed_summary["schema_version"] == ld.SCHEMA_VERSION
    assert committed_summary["contains_source_text"] is False
    assert dataset["digest"] == ld.dataset_digest(samples)
    assert dataset["samples"] == len(samples)
    assert dataset["kinds"] == dict(Counter(sample.kind for sample in samples))
    assert dataset["synthetic"] is True
    for backend in committed_summary["backends"].values():
        assert backend["model_id"] == ld.GLINER2_LANGUAGE_DECISION_MODEL
        assert backend["revision"] == ld.GLINER2_LANGUAGE_DECISION_REVISION
        assert backend["license_id"] == "Apache-2.0"
        assert backend["local"] is True
    assert committed_summary["router"]["lid_backend"] is None


def test_committed_router_counts_reproduce_offline(samples, committed_summary) -> None:
    backend = _backend(ScriptedScorer({}))
    fresh = ld.evaluate_language_decisions(
        samples,
        {kind: backend for kind in ld.DECISION_KINDS},
        tasks=ld.DECISION_TASKS,
    ).to_dict()

    for kind, sections in committed_summary["kinds"].items():
        assert (
            fresh["kinds"][kind]["summary"]["router"] == (sections["summary"]["router"])
        )
        router_columns = slice(0, 4)
        for language, row in sections["languages"]["rows"].items():
            assert (
                fresh["kinds"][kind]["languages"]["rows"][language][router_columns]
                == row[router_columns]
            )


def test_committed_summary_contains_no_fixture_text_or_values(
    samples, committed_summary
) -> None:
    text = SUMMARY_PATH.read_text(encoding="utf-8")

    assert text == ld.format_report_json(committed_summary)
    for sample in samples:
        assert sample.text not in text
    values = set()
    for path in [*(GOLDEN_DIR / "fixtures" / "i18n").glob("*.jsonl")]:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                for span in json.loads(line).get("gold_spans", ()):
                    if len(span["text"]) >= 6:
                        values.add(span["text"])
    assert values
    assert not [value for value in values if value in text]


def test_report_differences_tolerate_model_drift_only(committed_summary) -> None:
    assert ld.report_differences(committed_summary, committed_summary) == []

    drifted = copy.deepcopy(committed_summary)
    summary = drifted["kinds"][ld.LANGUAGE_IN_SCRIPT]["summary"]
    summary["model"]["covered"] += 2
    row = drifted["kinds"][ld.LANGUAGE]["languages"]["rows"]["de"]
    row[4] += 1
    assert ld.report_differences(committed_summary, drifted) == []

    summary["model"]["covered"] += 1
    assert ld.report_differences(committed_summary, drifted) == [
        "kinds.language_in_script.summary.model.covered differs by 3"
    ]

    router_drift = copy.deepcopy(committed_summary)
    router_drift["kinds"][ld.LANGUAGE]["summary"]["router"]["correct"] += 1
    router_drift["kinds"][ld.SCRIPT]["languages"]["rows"]["ru"][2] += 1
    router_drift["dataset"]["digest"] = "sha256:" + "0" * 64
    router_drift["backends"][ld.DIALECT]["revision"] = "main"
    assert ld.report_differences(committed_summary, router_drift) == [
        "dataset differs",
        "backends.dialect.revision differs",
        "kinds.language.summary.router.correct differs",
        "kinds.script.languages.ru.router_correct differs",
    ]


def test_run_loads_the_pinned_revision_and_reports_offline() -> None:
    calls: list[dict[str, Any]] = []

    class FakeClassifier:
        def classify_text(self, text: str, tasks: Any, **kwargs: Any) -> Any:
            return {
                task: [
                    {"label": label, "confidence": 0.9 if index == 0 else 0.0}
                    for index, label in enumerate(config["labels"])
                ]
                for task, config in tasks.items()
            }

    def loader(model_id: str, **kwargs: Any) -> GLiNER2Handle:
        calls.append({"model_id": model_id, **kwargs})
        return GLiNER2Handle(
            model_id=model_id,
            model=FakeClassifier(),
            revision=kwargs["revision"],
            license_id="Apache-2.0",
            architecture="boundary",
        )

    report = ld.run_gliner2_language_decision_eval(loader=loader)

    assert calls == [
        {
            "model_id": ld.GLINER2_LANGUAGE_DECISION_MODEL,
            "revision": ld.GLINER2_LANGUAGE_DECISION_REVISION,
            "device": "cpu",
            "cache_dir": None,
            "local_files_only": None,
        }
    ]
    assert report["tasks"] == dict(ld.DECISION_TASKS)
    assert {backend["runtime"] for backend in report["backends"].values()} == {
        "gliner2"
    }
    assert report["dataset"]["samples"] == sum(report["dataset"]["kinds"].values())
    assert report["contains_source_text"] is False


def test_benchmark_script_pins_the_evaluated_revision() -> None:
    from scripts.benchmarks import gliner2_language_decisions as script

    args = script.parse_args([])

    assert args.model_id == ld.GLINER2_LANGUAGE_DECISION_MODEL
    assert args.revision == ld.GLINER2_LANGUAGE_DECISION_REVISION
    assert len(args.revision) == 40
    assert args.device == "cpu"
    assert args.tolerance == 2
