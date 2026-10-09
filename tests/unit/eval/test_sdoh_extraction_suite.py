"""Synthetic SDOH scoring and native SHAC adapter controls; no restricted data."""

from __future__ import annotations

import json
import re
import socket
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.clinical.sdoh import SDOHFinding, available_determinant_extractors
from openmed.eval.report import BenchmarkReport
from openmed.eval.suites import (
    REGISTERED_EVAL_SUITES,
    load_suite_fixtures,
    suite_metadata,
)
from openmed.eval.suites.sdoh_extraction import (
    SDOH_DETERMINANTS,
    SDOH_EXTRACTION,
    SDOH_GOLD_PATH,
    SDOH_REPORT_PATH,
    SDOHBenchmarkCase,
    SDOHBenchmarkError,
    SDOHBenchmarkUnavailable,
    SDOHGoldFinding,
    SDOHUnavailableReason,
    load_sdoh_extraction_fixtures,
    run_sdoh_extraction_benchmark,
    run_shac_sdoh_benchmark,
    sdoh_report_digest,
)


def _finding(
    category="tobacco", status="current", start=0, end=4, value="PRIVATE_VALUE"
):
    return SDOHFinding(category, value, status, None, None, (start, end), 0.8)


def _case(*gold):
    return SDOHBenchmarkCase("0123456789 PRIVATE_NOTE", tuple(gold))


def _metrics(case, predicted):
    return run_sdoh_extraction_benchmark(
        [case], extractor=lambda text: predicted
    ).metrics


def _native_row(category="Tobacco", subtype="past"):
    texts = {
        "Tobacco": ("The synthetic patient quit tobacco.", "tobacco", "quit"),
        "Alcohol": ("The synthetic patient drinks alcohol.", "alcohol", "drinks"),
        "Drug": ("The synthetic patient denies cocaine.", "cocaine", "denies"),
        "Employment": ("The synthetic patient is retired.", "retired", "retired"),
        "LivingStatus": ("The synthetic patient lives alone.", "lives", "lives"),
    }
    text, trigger, status = texts[category]
    status_type = "StatusEmploy" if category == "Employment" else "StatusTime"
    entities = []
    for identity, label, surface in (
        ("T1", category, trigger),
        ("T2", status_type, status),
    ):
        start = text.index(surface)
        entities.append(
            dict(id=identity, label=label, start=start, end=start + len(surface))
        )
    arguments = [dict(role="Status", target="T2")]
    attributes = [dict(name=status_type + "Val", target="T2", value=subtype)]
    if category == "LivingStatus":
        start = text.index("alone")
        entities.append(dict(id="T3", label="TypeLiving", start=start, end=start + 5))
        arguments.append(dict(role="Type", target="T3"))
        attributes.append(dict(name="TypeLivingVal", target="T3", value="alone"))
    return dict(
        id="PRIVATE_RECORD_ID",
        text=text,
        entities=entities,
        events=[dict(id="E1", type=category, trigger="T1", arguments=arguments)],
        attributes=attributes,
    )


def _write_shac(root: Path, row, mode="json"):
    root.mkdir(exist_ok=True)
    if mode == "json":
        path = root / "PRIVATE_PATH.json"
        path.write_text(json.dumps(row), encoding="utf-8")
    else:
        path = root / "PRIVATE_PATH.txt"
        path.write_text(row["text"], encoding="utf-8")
        lines = [
            f"{x['id']}\t{x['label']} {x['start']} {x['end']}\t{row['text'][x['start'] : x['end']]}"
            for x in row["entities"]
        ]
        for event in row["events"]:
            args = " ".join(f"{x['role']}:{x['target']}" for x in event["arguments"])
            lines.append(f"{event['id']}\t{event['type']}:{event['trigger']} {args}")
        lines.extend(
            f"A{i}\t{x['name']} {x['target']} {x['value']}"
            for i, x in enumerate(row["attributes"], 1)
        )
        path.with_suffix(".ann").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_suite_registered_and_synthetic_report_reproducible(monkeypatch):
    from openmed.core.models import ModelLoader

    monkeypatch.setattr(
        ModelLoader, "load_model", lambda *args, **kwargs: pytest.fail("model loading")
    )
    assert SDOH_EXTRACTION in REGISTERED_EVAL_SUITES
    cases = load_suite_fixtures(SDOH_EXTRACTION)
    assert cases == load_sdoh_extraction_fixtures()
    assert set(SDOH_DETERMINANTS) == set(available_determinant_extractors())
    assert suite_metadata(SDOH_EXTRACTION)["task"] == "sdoh_extraction"
    report = run_sdoh_extraction_benchmark(cases)
    assert report.to_json() + "\n" == SDOH_REPORT_PATH.read_text()
    assert (
        report.to_json()
        == run_sdoh_extraction_benchmark(list(reversed(cases))).to_json()
    )
    assert report.fixture_count == 33
    assert report.metrics["overall"]["gold_count"] == 26
    assert report.metrics["overall"]["false_negatives"] == 4
    assert report.metrics["overall"]["false_positives"] == 7
    assert report.metadata["report_digest"] == sdoh_report_digest(report)
    assert re.fullmatch("sha256:[0-9a-f]{64}", sdoh_report_digest(report))
    assert BenchmarkReport.from_dict(report.to_dict()).to_json() == report.to_json()
    for case in cases:
        assert case.text not in report.to_json()


@pytest.mark.parametrize(
    "text,category,predicted,correct",
    (
        ("Smokes 1 pack per day.", "tobacco", 0, 0),
        ("Uses marijuana occasionally.", "drug", 0, 0),
        ("He smokes cigarettes daily.", "tobacco", 1, 0),
    ),
)
def test_known_misses_stay_in_denominators(text, category, predicted, correct):
    case = next(x for x in load_sdoh_extraction_fixtures() if x.text == text)
    counts = run_sdoh_extraction_benchmark([case]).metrics["by_determinant"][category]
    assert counts["gold_count"] == 1
    assert counts["prediction_count"] == predicted
    assert counts["status_matches"] == correct
    assert counts["status_accuracy"] == 0
    assert counts["recall"] == predicted


def test_one_to_one_overlap_matching_finds_maximum_not_greedy():
    case = _case(
        SDOHGoldFinding("tobacco", "current", 0, 4),
        SDOHGoldFinding("tobacco", "current", 5, 8),
    )
    result = _metrics(case, [_finding(end=8), _finding(end=4)])["overall"]
    assert result["overlap_matches"] == 2
    assert result["recall"] == result["precision"] == result["status_accuracy"] == 1
    assert result["exact_matches"] == 1
    assert result["exact_offset_match"] == 0.5


def test_duplicate_predictions_are_false_positives_and_wrong_status_is_failure():
    case = _case(SDOHGoldFinding("tobacco", "past", 0, 4))
    result = _metrics(case, [_finding(), _finding()])["overall"]
    assert result["overlap_matches"] == result["exact_matches"] == 1
    assert result["false_positives"] == 1
    assert result["precision"] == 0.5
    assert result["status_matches"] == result["status_accuracy"] == 0
    status = _metrics(case, [_finding()])["by_status"]["tobacco"]
    assert status["past"]["gold_count"] == status["past"]["false_negatives"] == 1
    assert status["current"]["false_positives"] == 1


@pytest.mark.parametrize(
    "prediction", (_finding(start=4, end=6), _finding(category="drug"))
)
def test_touching_offsets_and_wrong_categories_do_not_match(prediction):
    result = _metrics(_case(SDOHGoldFinding("tobacco", "current", 0, 4)), [prediction])[
        "overall"
    ]
    assert (
        result["overlap_matches"]
        == result["exact_matches"]
        == result["status_matches"]
        == 0
    )
    assert result["false_negatives"] == result["false_positives"] == 1


def test_zero_denominators_are_unobserved_and_empty_inputs_refused():
    result = _metrics(_case(), [])
    assert result["overall"]["recall"] is None
    assert result["overall"]["precision"] is None
    assert result["by_determinant"]["food_insecurity"]["status_accuracy"] is None
    with pytest.raises(SDOHBenchmarkError, match="sdoh_cases_empty"):
        run_sdoh_extraction_benchmark([])


def test_reports_ignore_value_score_paths_ids_and_hide_text(capsys, tmp_path):
    case = _case(SDOHGoldFinding("tobacco", "current", 0, 4))
    a = run_sdoh_extraction_benchmark([case], extractor=lambda text: [_finding()])
    b = run_sdoh_extraction_benchmark(
        [case],
        extractor=lambda text: [
            replace(
                _finding(),
                value="ANOTHER_PRIVATE_VALUE",
                score=0.01,
                extent="PRIVATE_EXTENT",
            )
        ],
    )
    assert a.to_json() == b.to_json()
    assert "PRIVATE" not in a.to_json() + a.to_markdown() + repr(case)
    assert capsys.readouterr() == ("", "")
    path = _write_shac(tmp_path / "corpus", _native_row())
    report = run_shac_sdoh_benchmark(path)
    assert isinstance(report, BenchmarkReport)
    assert "PRIVATE" not in report.to_json() + report.to_markdown()
    assert str(path) not in report.to_json()
    assert set(report.metadata) == {
        "schema_version",
        "fixture_digest",
        "prediction_digest",
        "report_digest",
    }


@pytest.mark.parametrize(
    "failure", ("exception", "network", "label", "status", "offset", "shape")
)
def test_extractor_failures_refuse_without_context_or_partial_report(failure):
    def extractor(text):
        if failure == "exception":
            raise RuntimeError("PRIVATE_NOTE")
        if failure == "network":
            socket.create_connection(("127.0.0.1", 9))
        if failure == "label":
            return [_finding(category="PRIVATE_CATEGORY")]
        if failure == "status":
            return [_finding(status=None)]
        if failure == "offset":
            return [_finding(end=1000)]
        return [{"PRIVATE_NOTE": text}]

    with pytest.raises(
        SDOHBenchmarkError, match="sdoh_extractor_contract_failed"
    ) as caught:
        run_sdoh_extraction_benchmark([_case()], extractor=extractor)
    assert caught.value.__context__ is None
    assert "PRIVATE_NOTE" not in str(caught.value)
    assert "RuntimeError" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    "change",
    (
        "restricted",
        "version",
        "float",
        "bool",
        "outside",
        "category",
        "status",
        "extra",
        "empty",
    ),
)
def test_malformed_gold_is_refused_without_source_or_path(change, tmp_path):
    payload = json.loads(SDOH_GOLD_PATH.read_text())
    payload["cases"][0]["text"] += " PRIVATE_NOTE"
    item = payload["cases"][0]["gold"][0]
    if change == "restricted":
        payload["synthetic"] = False
    elif change == "version":
        payload["schema_version"] = 2
    elif change in {"float", "bool", "outside"}:
        item["end"] = {"float": 2.1, "bool": True, "outside": 1000}[change]
    elif change == "category":
        item["category"] = "PRIVATE_CATEGORY"
    elif change == "status":
        item["status"] = "PRIVATE_STATUS"
    elif change == "extra":
        item["text"] = "PRIVATE_NOTE"
    else:
        payload["cases"] = []
    path = tmp_path / "PRIVATE_FILE.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(
        SDOHBenchmarkError, match="sdoh_synthetic_gold_invalid"
    ) as caught:
        load_sdoh_extraction_fixtures(path)
    assert caught.value.__context__ is None
    assert "PRIVATE" not in str(caught.value)


def test_shac_unconfigured_and_absent_paths_are_typed_unavailable(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("OPENMED_SHAC_PATH", raising=False)
    result = run_shac_sdoh_benchmark()
    assert result == SDOHBenchmarkUnavailable(SDOHUnavailableReason.NOT_CONFIGURED)
    result = run_shac_sdoh_benchmark(tmp_path / "PRIVATE_MISSING")
    assert result == SDOHBenchmarkUnavailable(SDOHUnavailableReason.PATH_UNAVAILABLE)
    assert set(result.to_dict()) == {"suite", "availability", "reason"}
    assert "PRIVATE" not in json.dumps(result.to_dict())
    monkeypatch.setenv("OPENMED_SHAC_PATH", str(tmp_path / "PRIVATE_MISSING"))
    assert run_shac_sdoh_benchmark() == result


@pytest.mark.parametrize("mode", ("json", "brat"))
@pytest.mark.parametrize(
    "category,native,status",
    (
        ("Tobacco", "past", "past"),
        ("Alcohol", "current", "current"),
        ("Drug", "none", "none"),
        ("Employment", "retired", "retired"),
        ("Employment", "on_disability", "disabled"),
        ("Employment", "homemaker", "homemaker"),
        ("LivingStatus", "current", "lives_alone"),
        ("LivingStatus", "past", "former"),
        ("LivingStatus", "future", "future"),
    ),
)
def test_native_json_and_brat_explicit_status_gold(
    mode, category, native, status, tmp_path
):
    row = _native_row(category, native)
    path = _write_shac(tmp_path / "corpus", row, mode)
    trigger = row["entities"][0]
    key = "living_status" if category == "LivingStatus" else category.lower()
    report = run_shac_sdoh_benchmark(
        path,
        extractor=lambda text: [
            _finding(key, status, trigger["start"], trigger["end"])
        ],
    )
    assert isinstance(report, BenchmarkReport)
    assert (
        report.metrics["overall"]["gold_count"]
        == report.metrics["overall"]["status_matches"]
        == 1
    )
    assert (
        report.metrics["overall"]["recall"]
        == report.metrics["overall"]["exact_offset_match"]
        == 1
    )
    assert report.metrics["by_determinant"]["food_insecurity"]["recall"] is None


@pytest.mark.parametrize(
    "change",
    (
        "missing",
        "conflict",
        "wrong-target",
        "unknown-label",
        "missing-event",
        "discontinuous",
        "bad-role",
        "ambiguous",
        "float",
        "private-subtype",
    ),
)
def test_native_gold_is_never_guessed_or_silently_dropped(change, tmp_path):
    row = _native_row()
    if change == "missing":
        row["attributes"] = []
    elif change == "conflict":
        row["attributes"].append(
            dict(name="StatusTimeVal", target="T2", value="current")
        )
    elif change == "wrong-target":
        row["attributes"][0]["target"] = "T1"
    elif change == "unknown-label":
        row["entities"][1]["label"] = "Status"
    elif change == "missing-event":
        row["events"] = []
    elif change == "discontinuous":
        row["entities"][0]["discontinuous"] = True
    elif change == "bad-role":
        row["events"][0]["arguments"][0]["role"] = "History"
    elif change == "ambiguous":
        row["events"][0]["arguments"] *= 2
    elif change == "float":
        row["entities"][0]["start"] += 0.5
    else:
        row["attributes"][0]["value"] = "PRIVATE_SUBTYPE"
    path = _write_shac(tmp_path / "corpus", row)
    result = run_shac_sdoh_benchmark(path)
    assert isinstance(result, SDOHBenchmarkUnavailable)
    assert result.reason in {
        SDOHUnavailableReason.GOLD_INVALID,
        SDOHUnavailableReason.GOLD_UNSUPPORTED,
    }
    assert "PRIVATE" not in json.dumps(result.to_dict())


def test_shared_native_trigger_keeps_two_event_denominators(tmp_path):
    row = _native_row()
    entity = dict(row["entities"][1], id="T3")
    row["entities"].append(entity)
    row["attributes"].append(dict(name="StatusTimeVal", target="T3", value="current"))
    row["events"].append(
        dict(
            id="E2",
            type="Tobacco",
            trigger="T1",
            arguments=[dict(role="Status", target="T3")],
        )
    )
    path = _write_shac(tmp_path / "corpus", row)
    trigger = row["entities"][0]
    report = run_shac_sdoh_benchmark(
        path,
        extractor=lambda text: [
            _finding(status="past", start=trigger["start"], end=trigger["end"])
        ],
    )
    assert isinstance(report, BenchmarkReport)
    assert report.metrics["overall"]["gold_count"] == 2
    assert report.metrics["overall"]["false_negatives"] == 1
    assert report.metrics["overall"]["status_accuracy"] == 0.5


def test_digest_changes_on_gold_or_prediction_and_cannot_certify_modified_report():
    case = _case(SDOHGoldFinding("tobacco", "current", 0, 4))
    a = run_sdoh_extraction_benchmark([case], extractor=lambda text: [_finding()])
    b = run_sdoh_extraction_benchmark([case], extractor=lambda text: [])
    c = run_sdoh_extraction_benchmark(
        [replace(case, text=case.text + "x")], extractor=lambda text: [_finding()]
    )
    assert a.metadata["prediction_digest"] != b.metadata["prediction_digest"]
    assert a.metadata["fixture_digest"] != c.metadata["fixture_digest"]
    modified = replace(a, fixture_count=100)
    assert modified.metadata["report_digest"] != sdoh_report_digest(modified)


def test_gold_duplicate_keys_and_boolean_version_are_invalid(tmp_path):
    original = SDOH_GOLD_PATH.read_text()
    for text in (
        original.replace('"schema_version": 1', '"schema_version": true'),
        original.replace('"synthetic": true', '"synthetic": false, "synthetic": true'),
    ):
        path = tmp_path / "manifest.json"
        path.write_text(text)
        with pytest.raises(SDOHBenchmarkError, match="sdoh_synthetic_gold_invalid"):
            load_sdoh_extraction_fixtures(path)


@pytest.mark.parametrize(
    "native,status",
    (
        ("alone", "lives_alone"),
        ("with_family", "lives_with_family"),
        ("with_others", "lives_with_others"),
        ("homeless", "homeless"),
    ),
)
def test_native_living_type_labels_are_retained(native, status, tmp_path):
    row = _native_row("LivingStatus", "current")
    row["attributes"][1]["value"] = native
    trigger = row["entities"][0]
    path = _write_shac(tmp_path / "corpus", row)
    report = run_shac_sdoh_benchmark(
        path,
        extractor=lambda text: [
            _finding("living_status", status, trigger["start"], trigger["end"])
        ],
    )
    assert isinstance(report, BenchmarkReport)
    assert report.metrics["by_status"]["living_status"][status]["status_matches"] == 1


def test_shac_digest_is_independent_of_private_path_and_record_id(tmp_path):
    row = _native_row()
    first = run_shac_sdoh_benchmark(_write_shac(tmp_path / "first", row))
    row["id"] = "ANOTHER_PRIVATE_ID"
    second = run_shac_sdoh_benchmark(_write_shac(tmp_path / "second", row))
    assert isinstance(first, BenchmarkReport) and isinstance(second, BenchmarkReport)
    assert first.to_json() == second.to_json()


def test_native_inline_subtype_and_argument_attribute_exports(tmp_path):
    row = _native_row()
    row["attributes"] = []
    row["entities"][1]["subtype"] = "past"
    first = run_shac_sdoh_benchmark(_write_shac(tmp_path / "inline", row))
    row["entities"][1].pop("subtype")
    row["entities"][1]["attributes"] = {"StatusTimeVal": "past"}
    second = run_shac_sdoh_benchmark(_write_shac(tmp_path / "attributes", row))
    assert isinstance(first, BenchmarkReport) and isinstance(second, BenchmarkReport)
    assert first.to_json() == second.to_json()


def test_corpus_symlink_escape_and_repository_paths_stay_unavailable(tmp_path):
    source = _write_shac(tmp_path / "outside", _native_row())
    root = tmp_path / "root"
    root.mkdir()
    (root / "corpus.json").symlink_to(source)
    result = run_shac_sdoh_benchmark(root)
    assert result == SDOHBenchmarkUnavailable(SDOHUnavailableReason.PATH_UNAVAILABLE)
    result = run_shac_sdoh_benchmark(SDOH_GOLD_PATH)
    assert result == SDOHBenchmarkUnavailable(SDOHUnavailableReason.PATH_UNAVAILABLE)


def test_non_english_native_records_refuse_instead_of_dropping(tmp_path):
    row = _native_row()
    row["language"] = "fr"
    result = run_shac_sdoh_benchmark(_write_shac(tmp_path / "corpus", row))
    assert result == SDOHBenchmarkUnavailable(SDOHUnavailableReason.GOLD_UNSUPPORTED)


@pytest.mark.parametrize("suffix", ("json", "jsonl"))
def test_native_duplicate_json_labels_are_refused(tmp_path, suffix):
    row = _native_row()
    encoded = json.dumps(row).replace(
        '"value": "past"', '"value": "current", "value": "past"'
    )
    path = tmp_path / ("corpus." + suffix)
    path.write_text(encoded + "\n")
    result = run_shac_sdoh_benchmark(path)
    assert result == SDOHBenchmarkUnavailable(SDOHUnavailableReason.GOLD_INVALID)


def test_native_duplicate_event_and_brat_annotation_ids_refuse(tmp_path):
    row = _native_row()
    row["events"] *= 2
    result = run_shac_sdoh_benchmark(_write_shac(tmp_path / "json", row))
    assert result == SDOHBenchmarkUnavailable(SDOHUnavailableReason.GOLD_INVALID)
    path = _write_shac(tmp_path / "brat", _native_row(), "brat")
    annotation = path.with_suffix(".ann")
    first_line = annotation.read_text().splitlines()[0]
    annotation.write_text(annotation.read_text() + first_line + "\n")
    result = run_shac_sdoh_benchmark(path)
    assert result == SDOHBenchmarkUnavailable(SDOHUnavailableReason.GOLD_INVALID)


def test_documented_examples_execute_offline_without_models(monkeypatch, capsys):
    from openmed.core.models import ModelLoader
    from openmed.core.offline import network_blocked_if_offline

    monkeypatch.delenv("OPENMED_SHAC_PATH", raising=False)
    monkeypatch.setattr(
        ModelLoader, "load_model", lambda *args, **kwargs: pytest.fail("model loading")
    )
    doc = Path(__file__).parents[3] / "docs" / "evaluation" / "sdoh-extraction.md"
    examples = re.findall(r"```python\n(.*?)```", doc.read_text(), re.S)
    assert len(examples) == 3
    with network_blocked_if_offline(local_only=True):
        for code in examples:
            exec(compile(code, "sdoh-extraction-doc-example", "exec"), {})
    output = capsys.readouterr()
    assert output.err == ""
    assert "shac_not_configured" in output.out
    assert "PRIVATE" not in output.out
