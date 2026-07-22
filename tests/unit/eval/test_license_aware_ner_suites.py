"""Tests for the CMeEE and Naamapadam license-aware eval suites."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from openmed.core.labels import (
    BODY_SITE,
    CONDITION,
    JOB_DEPARTMENT,
    LAB_TEST,
    LOCATION,
    MEDICATION,
    MICROORGANISM,
    ORGANIZATION,
    OTHER,
    PERSON,
    PROCEDURE,
)
from openmed.eval.datasets import (
    CMEEE,
    CMEEE_ENTITY_TYPES,
    CMEEE_PATH_ENV,
    NAAMAPADAM,
    NAAMAPADAM_PATH_ENV,
    license_for,
    load_cmeee,
    load_naamapadam,
    map_cmeee_label,
    map_naamapadam_label,
)
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.suites import (
    DEFAULT_SUITES,
    load_suite_fixtures,
    run_script_ner_benchmark,
    suite_metadata,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "license_aware_ner"
ZH_FIXTURE = FIXTURE_ROOT / "zh_standin.json"


def test_cmeee_nested_code_point_offsets_round_trip() -> None:
    result = load_cmeee(ZH_FIXTURE, allow_repo_path=True)
    fixture = result.to_benchmark_fixtures()[0]

    assert [(span.start, span.end) for span in fixture.gold_spans[:2]] == [
        (0, 3),
        (0, 5),
    ]
    assert [span.label for span in fixture.gold_spans] == [
        BODY_SITE,
        CONDITION,
        PROCEDURE,
    ]
    assert fixture.metadata["license"]["redistribution"] == "user-supplied"
    assert fixture.metadata["script"] == "Han"
    _assert_offset_round_trip(fixture)


def test_cmeee_documents_all_nine_source_categories() -> None:
    assert CMEEE_ENTITY_TYPES == {
        "bod": BODY_SITE,
        "dep": JOB_DEPARTMENT,
        "dis": CONDITION,
        "dru": MEDICATION,
        "equ": OTHER,
        "ite": LAB_TEST,
        "mic": MICROORGANISM,
        "pro": PROCEDURE,
        "sym": CONDITION,
    }
    for source_label, canonical in CMEEE_ENTITY_TYPES.items():
        assert map_cmeee_label(source_label).canonical_label == canonical


def test_cmeee_directory_selects_only_the_requested_split(tmp_path: Path) -> None:
    train_path = tmp_path / "CMeEE_train.json"
    test_path = tmp_path / "CMeEE_test.json"
    train_path.write_text(
        json.dumps(
            [
                {
                    "id": "train-row",
                    "text": "肺炎",
                    "entities": [
                        {
                            "start_idx": 0,
                            "end_idx": 1,
                            "type": "dis",
                            "entity": "肺炎",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    test_path.write_text(ZH_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    result = load_cmeee(tmp_path, split="test")

    assert [record.record_id for record in result.records] == ["zh-standin-1"]


def test_naamapadam_official_token_tags_load_hi_and_te() -> None:
    result = load_naamapadam(FIXTURE_ROOT, allow_repo_path=True)
    fixtures = result.to_benchmark_fixtures()

    assert {fixture.language for fixture in fixtures} == {"hi", "te"}
    assert result.unmapped_labels == ()
    for fixture in fixtures:
        assert [span.label for span in fixture.gold_spans] == [PERSON, LOCATION]
        assert fixture.metadata["license"]["license_id"] == "CC0-1.0"
        _assert_offset_round_trip(fixture)
    assert {fixture.metadata["script"] for fixture in fixtures} == {
        "Devanagari",
        "Telugu",
    }


@pytest.mark.parametrize(
    ("source_label", "expected"),
    [("PER", PERSON), ("B-LOC", LOCATION), ("I-ORG", ORGANIZATION)],
)
def test_naamapadam_per_loc_org_mapping(source_label: str, expected: str) -> None:
    mapping = map_naamapadam_label(source_label, language="hi")

    assert mapping.canonical_label == expected
    assert mapping.mapped is True


@pytest.mark.parametrize(
    ("suite", "path_env"),
    [(CMEEE, CMEEE_PATH_ENV), (NAAMAPADAM, NAAMAPADAM_PATH_ENV)],
)
def test_unconfigured_suites_auto_skip_with_clear_message(
    suite: str,
    path_env: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(path_env, raising=False)

    with pytest.warns(UserWarning, match=rf"Skipping {suite}: {path_env} is not set"):
        assert load_suite_fixtures(suite) == []

    metadata = suite_metadata(suite)
    assert metadata["availability"] == {
        "configured": False,
        "path_env": path_env,
        "reason": f"{path_env} is not set",
        "status": "skipped",
    }


def test_suites_run_from_environment_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CMEEE_PATH_ENV, str(ZH_FIXTURE))
    monkeypatch.setenv(NAAMAPADAM_PATH_ENV, str(FIXTURE_ROOT))

    cmeee = load_suite_fixtures(CMEEE, allow_repo_path=True)
    naamapadam = load_suite_fixtures(
        NAAMAPADAM,
        languages="hi",
        allow_repo_path=True,
    )

    assert len(cmeee) == 1
    assert cmeee[0].metadata["suite"] == CMEEE
    assert len(naamapadam) == 1
    assert naamapadam[0].language == "hi"


@pytest.mark.parametrize(
    ("suite", "path_env"),
    [(CMEEE, CMEEE_PATH_ENV), (NAAMAPADAM, NAAMAPADAM_PATH_ENV)],
)
def test_configured_empty_suites_fail_instead_of_silently_passing(
    suite: str,
    path_env: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(path_env, str(tmp_path))

    with pytest.raises(ValueError, match="configured.*contains no benchmark records"):
        load_suite_fixtures(suite)


def test_cross_script_harness_reports_per_suite_micro_f1() -> None:
    cmeee_report = run_script_ner_benchmark(
        CMEEE,
        model_name="synthetic-perfect",
        runner=_identity_runner,
        load_kwargs={"path": ZH_FIXTURE, "allow_repo_path": True},
    )
    naamapadam_report = run_script_ner_benchmark(
        NAAMAPADAM,
        model_name="synthetic-perfect",
        runner=_identity_runner,
        load_kwargs={
            "path": FIXTURE_ROOT,
            "languages": "hi",
            "allow_repo_path": True,
        },
    )

    assert cmeee_report.suite == CMEEE
    assert cmeee_report.metrics["micro_f1"] == pytest.approx(1.0)
    assert cmeee_report.metrics["micro_f1_by_script"] == {"Han": 1.0}
    assert naamapadam_report.suite == NAAMAPADAM
    assert naamapadam_report.metrics["micro_f1"] == pytest.approx(1.0)
    assert naamapadam_report.metrics["micro_f1_by_script"] == {"Devanagari": 1.0}


def test_license_registry_covers_cblue_cmeee_and_naamapadam() -> None:
    cblue = license_for("cblue")
    cmeee = license_for(CMEEE)
    naamapadam = license_for(NAAMAPADAM)

    assert cblue.license_id == "CBLUE-access-controlled"
    assert cblue.redistribution == "user-supplied"
    assert cmeee.license_id == "CBLUE-access-controlled"
    assert cmeee.redistribution == "user-supplied"
    assert naamapadam.license_id == "CC0-1.0"
    assert naamapadam.redistribution == "reference-only"


def test_default_registry_contains_both_license_aware_suites() -> None:
    assert CMEEE in DEFAULT_SUITES
    assert NAAMAPADAM in DEFAULT_SUITES


def test_no_real_benchmark_payload_filenames_are_committed() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    benchmark_name = re.compile(r"cblue|cmeee|naamapadam|indicglue", re.IGNORECASE)
    payload_suffixes = {".bio", ".conll", ".csv", ".iob", ".json", ".jsonl", ".tsv"}
    offenders: list[str] = []

    for scan_root in (repo_root / "openmed", repo_root / "tests"):
        for path in scan_root.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() not in payload_suffixes
                or benchmark_name.search(path.name) is None
            ):
                continue
            if _is_synthetic_payload(path):
                continue
            offenders.append(str(path.relative_to(repo_root)))

    assert offenders == []


def _identity_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[Any, ...]:
    _ = (model_name, device)
    return tuple(fixture.gold_spans)


def _assert_offset_round_trip(fixture: BenchmarkFixture) -> None:
    for span in fixture.gold_spans:
        assert fixture.text[span.start : span.end] == span.text


def _is_synthetic_payload(path: Path) -> bool:
    try:
        if path.suffix.lower() == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(rows, dict):
                rows = [rows]
        else:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return bool(rows) and all(
        isinstance(row, dict) and row.get("metadata", {}).get("synthetic") is True
        for row in rows
    )
