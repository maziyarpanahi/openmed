from __future__ import annotations

import json

import pytest

from openmed.eval.datasets import (
    DUA_GATED_CORPORA,
    DUACredentialRequired,
    PUBLIC_DATASETS,
    assert_no_gated_content_committed,
    license_for,
    load_dua_corpus,
    load_public_dataset,
    map_public_label,
)


def test_public_adapter_loads_common_schema_and_maps_labels(tmp_path):
    text = "Patient Jordan Smith called 555-111-2222."
    payload = {
        "records": [
            {
                "id": "shield-1",
                "text": text,
                "split": "sample",
                "spans": [
                    {"start": 8, "end": 20, "label": "patient", "text": "Jordan Smith"},
                    {"start": 28, "end": 40, "label": "phone", "text": "555-111-2222"},
                ],
                "metadata": {"source": "unit-test"},
            }
        ]
    }
    path = tmp_path / "shield.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = load_public_dataset("shield", path)

    assert result.skipped is False
    assert len(result.records) == 1
    record = result.records[0]
    assert record.license == license_for("shield")
    assert [span.label for span in record.spans] == ["PERSON", "PHONE"]
    fixture = record.to_benchmark_fixture()
    assert fixture.fixture_id == "shield-1"
    assert fixture.gold_spans[0].label == "PERSON"
    assert fixture.metadata["license"]["redistribution"] == "reference-only"


def test_public_adapters_skip_cleanly_when_source_absent(tmp_path):
    for dataset in PUBLIC_DATASETS:
        result = load_public_dataset(dataset, tmp_path / f"{dataset}.json")
        assert result.skipped is True
        assert result.records == ()
        assert result.license is not None


def test_public_label_maps_are_canonical():
    assert map_public_label("shield", "id") == "ID_NUM"
    assert map_public_label("drugprot", "CHEMICAL") == "OTHER"
    assert map_public_label("ncbi_disease", "Disease") == "OTHER"
    assert map_public_label("bc5cdr", "chemical") == "OTHER"


def test_dua_stubs_refuse_without_credentialed_path():
    for corpus in DUA_GATED_CORPORA:
        with pytest.raises(DUACredentialRequired):
            load_dua_corpus(corpus)


def test_dua_stub_accepts_existing_credentialed_path_as_eval_only(tmp_path):
    result = load_dua_corpus("i2b2", tmp_path)

    assert result.skipped is True
    assert result.reason.startswith("eval-only gated corpus stub")


def test_guard_rejects_gated_payload_markers(tmp_path):
    data_file = tmp_path / "payload.jsonl"
    data_file.write_text('{"label": "UMLS"}\n', encoding="utf-8")

    with pytest.raises(AssertionError, match="gated dataset content"):
        assert_no_gated_content_committed(tmp_path)


def test_guard_ignores_python_source_and_clean_payloads(tmp_path):
    (tmp_path / "adapter.py").write_text('MARKER = "UMLS"\n', encoding="utf-8")
    (tmp_path / "payload.jsonl").write_text('{"label": "PERSON"}\n', encoding="utf-8")

    assert_no_gated_content_committed(tmp_path)
