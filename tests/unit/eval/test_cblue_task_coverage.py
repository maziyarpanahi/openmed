"""CBLUE task-shape coverage, license-boundary, and provenance-gate tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from openmed.core.labels import CONDITION, LAB_TEST, MEDICATION, PROCEDURE
from openmed.eval.datasets.cblue import (
    CBLUE_PATH_ENV,
    CBLUE_TASKS,
    CBLUE_UNSUPPORTED_TASKS,
    CHIP_CDN,
    CMEIE,
    IMCS_V2_NER,
    cblue_task_metadata,
    cblue_task_shape,
    configured_cblue_task_path,
    load_cblue_task,
    load_cblue_task_fixtures,
    map_cblue_label,
    synthetic_cblue_fixture_path,
)
from openmed.eval.datasets.multilingual_ner import MultilingualNerCorpusRequired
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.suites import (
    CBLUE_TASK_COVERAGE,
    DEFAULT_SUITES,
    load_suite_fixtures,
    suite_metadata,
)
from openmed.eval.suites.cblue_coverage import (
    SYNTHETIC_SOURCE,
    USER_SUPPLIED_SOURCE,
    CblueProvenanceError,
    load_cblue_task_coverage_fixtures,
    run_cblue_task_coverage,
    run_synthetic_cblue_task_coverage_smoke,
)
from scripts.benchmarks.generate_cblue_synthetic_fixtures import (
    synthetic_fixture_payloads,
)

IMCS_EXPECTED = {
    "Symptom": CONDITION,
    "Drug": MEDICATION,
    "Drug_Category": MEDICATION,
    "Medical_Examination": LAB_TEST,
    "Operation": PROCEDURE,
}


@pytest.fixture(autouse=True)
def _isolated_cblue_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test hermetic against a developer's exported CBLUE paths."""

    monkeypatch.delenv(CBLUE_PATH_ENV, raising=False)
    for task in CBLUE_TASKS:
        monkeypatch.delenv(cblue_task_shape(task).path_env, raising=False)


def _synthetic(task: str) -> list[BenchmarkFixture]:
    return load_cblue_task_fixtures(
        task,
        synthetic_cblue_fixture_path(task),
        split="synthetic",
        allow_repo_path=True,
    )


def test_committed_fixtures_are_reproducible_from_the_generator() -> None:
    """The data-provenance claim must be reproducible, not merely asserted.

    The generator draws no random numbers and reads no external state, so a
    reviewer can regenerate both fixtures and get identical bytes.
    """

    payloads = synthetic_fixture_payloads()

    assert set(payloads) == {
        "cblue_chip_cdn_synthetic.jsonl",
        "cblue_imcs_v2_ner_synthetic.jsonl",
    }
    for name, payload in payloads.items():
        committed = (synthetic_cblue_fixture_path(CHIP_CDN).parent / name).read_text(
            encoding="utf-8"
        )
        assert committed == payload
    assert synthetic_fixture_payloads() == payloads


def test_supported_task_shapes_are_exactly_the_entity_relevant_ones() -> None:
    assert CBLUE_TASKS == (CHIP_CDN, IMCS_V2_NER)
    assert cblue_task_shape(CHIP_CDN).shape == "entity_normalization"
    assert cblue_task_shape(IMCS_V2_NER).shape == "dialogue_ner"


def test_cmeie_is_not_registered() -> None:
    """Relation decoding stays out of scope and must fail loudly, not silently."""

    assert CMEIE in CBLUE_UNSUPPORTED_TASKS
    assert CMEIE not in CBLUE_TASKS
    with pytest.raises(ValueError, match="relation decoding"):
        cblue_task_shape(CMEIE)
    with pytest.raises(ValueError, match="relation decoding"):
        load_cblue_task(CMEIE, allow_repo_path=True)


@pytest.mark.parametrize("task", CBLUE_TASKS)
def test_each_task_requires_an_explicit_external_path(
    task: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CBLUE_PATH_ENV, raising=False)
    monkeypatch.delenv(cblue_task_shape(task).path_env, raising=False)

    with pytest.raises(MultilingualNerCorpusRequired, match="explicit local"):
        load_cblue_task(task)


@pytest.mark.parametrize("task", CBLUE_TASKS)
def test_each_task_rejects_a_repository_internal_path(task: str) -> None:
    with pytest.raises(MultilingualNerCorpusRequired, match="inside the repository"):
        load_cblue_task(task, synthetic_cblue_fixture_path(task))


@pytest.mark.parametrize("task", CBLUE_TASKS)
def test_each_task_rejects_a_missing_path(task: str, tmp_path: Path) -> None:
    with pytest.raises(MultilingualNerCorpusRequired, match="does not exist"):
        load_cblue_task(task, tmp_path / "absent.jsonl")


def test_chip_cdn_fixtures_round_trip_offsets_and_standard_terms() -> None:
    fixtures = _synthetic(CHIP_CDN)

    assert len(fixtures) == 3
    for fixture in fixtures:
        assert fixture.language == "zh"
        assert fixture.metadata["synthetic"] is True
        assert fixture.metadata["contains_real_phi"] is False
        assert fixture.metadata["cblue_task"] == CHIP_CDN
        assert len(fixture.gold_spans) == 1
        span = fixture.gold_spans[0]
        assert (span.start, span.end) == (0, len(fixture.text))
        assert fixture.text[span.start : span.end] == span.text
        assert span.label == CONDITION
        assert len(fixture.metadata["normalized_terms"]) == 2


def test_imcs_v2_ner_fixtures_decode_every_source_category() -> None:
    fixtures = _synthetic(IMCS_V2_NER)
    observed = {
        span.metadata["source_label"]
        for fixture in fixtures
        for span in fixture.gold_spans
    }

    assert observed == set(IMCS_EXPECTED)
    for source_label, expected in IMCS_EXPECTED.items():
        mapping = map_cblue_label(IMCS_V2_NER, source_label)
        assert mapping.canonical_label == expected
        assert mapping.mapped is True
    for fixture in fixtures:
        assert all(
            fixture.text[span.start : span.end] == span.text
            for span in fixture.gold_spans
        )
        assert not fixture.metadata["unmapped_labels"]


def test_imcs_v2_ner_accepts_the_native_string_release_shape(tmp_path: Path) -> None:
    """The upstream release ships sentence and BIO_label as strings."""

    source = tmp_path / "imcs_v2_ner_test.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "external-1",
                "sentence": "甲司林",
                "BIO_label": "B-Drug I-Drug I-Drug",
                "metadata": {"synthetic": True},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_cblue_task(IMCS_V2_NER, source)
    record = result.records[0]

    assert record.text == "甲司林"
    assert [(span.start, span.end, span.canonical_label) for span in record.spans] == [
        (0, 3, MEDICATION)
    ]


def test_chip_cdn_accepts_the_native_double_hash_release_shape(tmp_path: Path) -> None:
    source = tmp_path / "chip_cdn_test.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "external-1",
                "text": "甲区甲型热症",
                "normalized_result": "甲型热症##甲区病变",
                "metadata": {"synthetic": True},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_cblue_task(CHIP_CDN, source)
    record = result.records[0]

    assert record.metadata["normalized_terms"] == ["甲型热症", "甲区病变"]
    assert record.metadata["license"]["redistribution"] == "user-supplied"


def test_chip_cdn_validation_error_omits_the_source_mention(tmp_path: Path) -> None:
    source = tmp_path / "chip_cdn_invalid.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "external-invalid",
                "text": "甲区敏感诊断",
                "normalized_result": "",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="requires at least one standard term"
    ) as caught:
        load_cblue_task(CHIP_CDN, source)

    assert "甲区敏感诊断" not in str(caught.value)


@pytest.mark.parametrize("task", CBLUE_TASKS)
def test_task_metadata_reports_availability_without_reading_content(
    task: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CBLUE_PATH_ENV, raising=False)
    monkeypatch.delenv(cblue_task_shape(task).path_env, raising=False)
    skipped = cblue_task_metadata(task)

    assert skipped["availability"]["status"] == "skipped"
    assert skipped["license"]["license_id"] == "CBLUE-access-controlled"
    assert skipped["license"]["redistribution"] == "user-supplied"

    monkeypatch.setenv(CBLUE_PATH_ENV, str(tmp_path))
    configured = cblue_task_metadata(task)

    assert configured["availability"]["status"] == "configured"
    assert configured_cblue_task_path(task) == tmp_path / task


def test_registry_exposes_the_cblue_suite_and_runs_it_offline() -> None:
    assert CBLUE_TASK_COVERAGE in DEFAULT_SUITES

    fixtures = load_suite_fixtures(CBLUE_TASK_COVERAGE)
    metadata = suite_metadata(CBLUE_TASK_COVERAGE)

    assert len(fixtures) == 6
    assert {fixture.metadata["cblue_task"] for fixture in fixtures} == set(CBLUE_TASKS)
    assert metadata["suite"] == CBLUE_TASK_COVERAGE
    assert metadata["unsupported_tasks"] == {CMEIE: "relation decoding is out of scope"}


def test_synthetic_suite_reports_per_task_metrics_and_passes_provenance() -> None:
    report = run_synthetic_cblue_task_coverage_smoke()
    tasks = report.metrics["tasks"]

    assert report.suite == CBLUE_TASK_COVERAGE
    assert report.metrics["gate"]["passed"] is True
    assert report.metrics["gate"]["failures"] == []
    assert set(tasks) == set(CBLUE_TASKS)
    assert tasks[CHIP_CDN]["exact_span_f1"]["f1"] == 1.0
    assert tasks[CHIP_CDN]["normalization_accuracy"] == 1.0
    assert tasks[CHIP_CDN]["normalized_term_count"] == 6
    assert tasks[IMCS_V2_NER]["exact_span_f1"]["f1"] == 1.0
    assert tasks[IMCS_V2_NER]["span_count"] == 13
    assert tasks[IMCS_V2_NER]["label_counts"] == {
        CONDITION: 3,
        LAB_TEST: 3,
        MEDICATION: 4,
        PROCEDURE: 3,
    }


def test_provenance_gate_fails_closed_without_license_evidence() -> None:
    fixtures = load_cblue_task_coverage_fixtures()
    stripped = [
        replace(
            fixture,
            metadata={
                key: value
                for key, value in fixture.metadata.items()
                if key not in {"license", "source_path_hash"}
            },
        )
        for fixture in fixtures
    ]

    with pytest.raises(CblueProvenanceError):
        run_cblue_task_coverage(
            stripped,
            model_name="synthetic-oracle",
            runner=_identity_runner,
        )

    report = run_cblue_task_coverage(
        stripped,
        model_name="synthetic-oracle",
        runner=_identity_runner,
        fail_on_provenance=False,
    )
    reasons = {failure["reason"] for failure in report.metrics["gate"]["failures"]}

    assert report.metrics["gate"]["passed"] is False
    assert reasons == {"missing_license_block", "missing_source_path_hash"}


def test_provenance_gate_fails_when_a_task_shape_is_missing() -> None:
    only_chip_cdn = [
        fixture
        for fixture in load_cblue_task_coverage_fixtures()
        if fixture.metadata["cblue_task"] == CHIP_CDN
    ]

    report = run_cblue_task_coverage(
        only_chip_cdn,
        model_name="synthetic-oracle",
        runner=_identity_runner,
        fail_on_provenance=False,
    )
    failures = report.metrics["gate"]["failures"]

    assert [failure["reason"] for failure in failures] == ["no_task_fixtures"]
    assert failures[0]["task"] == IMCS_V2_NER


def test_report_retains_no_benchmark_text() -> None:
    report = run_synthetic_cblue_task_coverage_smoke()
    serialized = report.to_json()

    for fixture in load_cblue_task_coverage_fixtures():
        assert fixture.text not in serialized


def test_duplicate_fixture_ids_are_rejected() -> None:
    fixtures = load_cblue_task_coverage_fixtures()

    with pytest.raises(ValueError, match="duplicate fixture id"):
        run_cblue_task_coverage(
            [*fixtures, fixtures[0]],
            model_name="synthetic-oracle",
            runner=_identity_runner,
        )


def _write_external_corpus(tmp_path: Path) -> Path:
    """Write a licensed-shaped corpus outside the repository tree.

    Both files number their rows from 1, which is how CBLUE ships them.
    """

    root = tmp_path / "cblue"
    (root / CHIP_CDN).mkdir(parents=True)
    (root / IMCS_V2_NER).mkdir(parents=True)
    (root / CHIP_CDN / "data.jsonl").write_text(
        json.dumps(
            {
                "id": "1",
                "text": "戊区戊型咳症",
                "normalized_result": "戊型咳症##戊区病变",
                "metadata": {"synthetic": True},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / IMCS_V2_NER / "data.jsonl").write_text(
        json.dumps(
            {
                "id": "1",
                "sentence": "戊司林",
                "BIO_label": "B-Drug I-Drug I-Drug",
                "metadata": {"synthetic": True},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_root_env_var_is_honoured_by_the_documented_entry_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the suite must read configured data, not the smoke fixture.

    Setting OPENMED_CBLUE_PATH and running the documented command previously
    scored the bundled synthetic rows while reporting ``configured``.
    """

    root = _write_external_corpus(tmp_path)
    monkeypatch.setenv(CBLUE_PATH_ENV, str(root))

    fixtures = load_suite_fixtures(CBLUE_TASK_COVERAGE)

    assert [fixture.fixture_id for fixture in fixtures] == [
        f"{CHIP_CDN}/1",
        f"{IMCS_V2_NER}/1",
    ]
    assert all(
        fixture.metadata["source_kind"] == USER_SUPPLIED_SOURCE for fixture in fixtures
    )
    assert any("戊" in fixture.text for fixture in fixtures)


def test_task_specific_env_var_overrides_the_root_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_external_corpus(tmp_path)
    monkeypatch.delenv(CBLUE_PATH_ENV, raising=False)
    monkeypatch.setenv(cblue_task_shape(CHIP_CDN).path_env, str(root / CHIP_CDN))
    monkeypatch.delenv(cblue_task_shape(IMCS_V2_NER).path_env, raising=False)

    fixtures = load_suite_fixtures(CBLUE_TASK_COVERAGE)
    by_task = {
        fixture.metadata["cblue_task"]: fixture.metadata["source_kind"]
        for fixture in fixtures
    }

    assert by_task[CHIP_CDN] == USER_SUPPLIED_SOURCE
    assert by_task[IMCS_V2_NER] == SYNTHETIC_SOURCE


def test_report_availability_describes_what_was_scored_not_the_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both directions of the availability/scored correlation."""

    synthetic = run_synthetic_cblue_task_coverage_smoke()
    synthetic_availability = synthetic.metadata["tasks"][CHIP_CDN]["availability"]

    assert synthetic_availability["configured"] is False
    assert synthetic_availability["status"] == "synthetic"
    assert synthetic_availability["source_kind"] == SYNTHETIC_SOURCE

    monkeypatch.setenv(CBLUE_PATH_ENV, str(_write_external_corpus(tmp_path)))
    real = run_cblue_task_coverage(
        load_suite_fixtures(CBLUE_TASK_COVERAGE),
        model_name="synthetic-oracle",
        runner=_identity_runner,
    )
    real_availability = real.metadata["tasks"][CHIP_CDN]["availability"]

    assert real_availability["configured"] is True
    assert real_availability["status"] == "configured"
    assert real_availability["source_kind"] == USER_SUPPLIED_SOURCE


def _reasons(fixtures: list[BenchmarkFixture]) -> set[str]:
    report = run_cblue_task_coverage(
        fixtures,
        model_name="synthetic-oracle",
        runner=_identity_runner,
        normalizer=_gold_normalizer,
        fail_on_provenance=False,
    )
    return {str(failure["reason"]) for failure in report.metrics["gate"]["failures"]}


def _corrupt_first(task: str, **metadata: Any) -> list[BenchmarkFixture]:
    fixtures = load_cblue_task_coverage_fixtures()
    for index, fixture in enumerate(fixtures):
        if fixture.metadata["cblue_task"] == task:
            fixtures[index] = replace(
                fixture, metadata={**dict(fixture.metadata), **metadata}
            )
            break
    return fixtures


@pytest.mark.parametrize(
    ("reason", "task", "metadata"),
    [
        ("unknown_task", CHIP_CDN, {"cblue_task": CMEIE}),
        (
            "incomplete_license_block",
            CHIP_CDN,
            {"license": {"dataset": "chip_cdn", "license_id": "x", "source_url": ""}},
        ),
        (
            "unexpected_redistribution",
            CHIP_CDN,
            {
                "license": {
                    "dataset": "chip_cdn",
                    "license_id": "x",
                    "redistribution": "download-on-demand",
                    "source_url": "https://example.invalid",
                }
            },
        ),
        ("unexpected_script", CHIP_CDN, {"script": "Latin"}),
        ("unmapped_source_label", CHIP_CDN, {"unmapped_labels": ("mystery",)}),
        (
            "raw_source_path_in_metadata",
            CHIP_CDN,
            {"corpus_file": "/data/cblue/chip_cdn_test.jsonl"},
        ),
        ("missing_normalized_terms", CHIP_CDN, {"normalized_terms": []}),
    ],
)
def test_each_documented_gate_reason_code_is_reachable(
    reason: str,
    task: str,
    metadata: dict[str, Any],
) -> None:
    assert reason in _reasons(_corrupt_first(task, **metadata))


def test_unexpected_language_is_reachable() -> None:
    fixtures = load_cblue_task_coverage_fixtures()
    fixtures[0] = replace(fixtures[0], language="en")

    assert "unexpected_language" in _reasons(fixtures)


def test_documented_reason_codes_match_the_implementation() -> None:
    """Every reason code named in the docs must be emitted by some input."""

    documented = {
        "incomplete_license_block",
        "missing_license_block",
        "missing_normalized_terms",
        "missing_source_path_hash",
        "no_task_fixtures",
        "raw_source_path_in_metadata",
        "unexpected_language",
        "unexpected_redistribution",
        "unexpected_script",
        "unknown_task",
        "unmapped_source_label",
    }
    doc = (Path(__file__).resolve().parents[3] / "docs" / "eval-harness.md").read_text(
        encoding="utf-8"
    )

    for reason in documented:
        assert f"`{reason}`" in doc


def _halving_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[Any, ...]:
    """Return every other gold span, so recall must drop below 1.0."""

    _ = (model_name, device)
    return tuple(fixture.gold_spans[::2])


def _mislabelling_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[Any, ...]:
    _ = (model_name, device)
    return tuple(replace(span, label=PROCEDURE) for span in fixture.gold_spans)


def test_metrics_are_not_tautological_under_an_imperfect_runner() -> None:
    """A perfect-oracle-only suite could be arbitrarily wrong and still pass."""

    report = run_cblue_task_coverage(
        load_cblue_task_coverage_fixtures(),
        model_name="imperfect",
        runner=_halving_runner,
        normalizer=_gold_normalizer,
    )
    imcs = report.metrics["tasks"][IMCS_V2_NER]["exact_span_f1"]

    assert imcs["recall"] == pytest.approx(7 / 13, abs=1e-3)
    assert imcs["precision"] == 1.0
    assert imcs["true_positives"] == 7
    assert imcs["false_negatives"] == 6


def test_span_scoring_is_label_sensitive() -> None:
    report = run_cblue_task_coverage(
        load_cblue_task_coverage_fixtures(),
        model_name="mislabelling",
        runner=_mislabelling_runner,
        normalizer=_gold_normalizer,
    )
    imcs = report.metrics["tasks"][IMCS_V2_NER]["exact_span_f1"]

    # Only the genuinely PROCEDURE-labelled spans survive relabelling.
    assert imcs["recall"] < 1.0
    assert imcs["true_positives"] == 3


def test_normalization_accuracy_reacts_to_a_wrong_normalizer() -> None:
    def wrong_for_one(fixture: BenchmarkFixture) -> tuple[str, ...]:
        if fixture.fixture_id.endswith("-1"):
            return ("完全不同的词",)
        return _gold_normalizer(fixture)

    report = run_cblue_task_coverage(
        load_cblue_task_coverage_fixtures(),
        model_name="partial",
        runner=_identity_runner,
        normalizer=wrong_for_one,
    )

    assert report.metrics["tasks"][CHIP_CDN]["normalization_accuracy"] == pytest.approx(
        2 / 3
    )


def _gold_normalizer(fixture: BenchmarkFixture) -> tuple[str, ...]:
    return tuple(sorted(fixture.metadata.get("normalized_terms") or ()))


def _identity_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[Any, ...]:
    _ = (model_name, device)
    return tuple(fixture.gold_spans)
