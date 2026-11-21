"""Exercise native temporal parsing and aggregation with invented data."""

import json
import socket

import pytest

from openmed.cli.main import main
from openmed.eval.datasets.i2b2_2012 import (
    EVENT_TYPES,
    TLINK_FOLD,
    TemporalCorpusError,
    load_i2b2_2012,
)
from openmed.eval.suites import i2b2_2012_temporal as suite
from tests.fixtures.eval.i2b2_2012_synthetic import write_temporal_fixture


def test_all_source_types_and_fail_closed_offsets(tmp_path):
    path = tmp_path / "private.xml"
    write_temporal_fixture(path, tuple(TLINK_FOLD))
    (doc,) = load_i2b2_2012(tmp_path)
    assert len(doc.events) == 6 and len(doc.timexes) == len(doc.sectimes) == 1
    assert [link[0] for link in doc.links] == list(TLINK_FOLD.values())
    original = path.read_text()
    for changed, code in (
        (
            original.replace('text="rash"', 'text="private-invalid"'),
            "temporal_span_mismatch",
        ),
        (
            original.replace('type="BEFORE"', 'type="private-invalid"'),
            "unknown_tlink_type",
        ),
    ):
        path.write_text(changed)
        with pytest.raises(TemporalCorpusError, match=f"^{code}$"):
            load_i2b2_2012(tmp_path)


def test_real_extractors_and_aggregate_only_cli(tmp_path, monkeypatch):
    corpus = tmp_path / "authorized"
    corpus.mkdir()
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("network"))
    monkeypatch.setenv("OPENMED_I2B2_2012_PATH", str(corpus))
    write_temporal_fixture(corpus / "private.xml")
    seen = set()
    original = suite.extract_tlink_candidates

    def inspect(text, spans):
        seen.update(
            row["metadata"]["source_type"] for row in spans if row["label"] == "EVENT"
        )
        return original(text, spans)

    monkeypatch.setattr(suite, "extract_tlink_candidates", inspect)
    result = suite.score_i2b2_2012(load_i2b2_2012(corpus))
    assert seen == EVENT_TYPES
    assert result["cells"]["timex_value_accuracy"] == {"status": "suppressed"}
    for n in range(4):
        write_temporal_fixture(corpus / f"doc{n}.xml")
    target = tmp_path / "result.json"
    assert (
        main(
            [
                "eval",
                "credentialed",
                "--suite",
                "i2b2-2012-temporal",
                "--output",
                str(target),
            ]
        )
        == 0
    )
    report = json.loads(target.read_bytes())
    assert report["cells"]["timex_value_accuracy"]["rate"] == 1
    assert report["cells"]["timex_type_accuracy"]["gold_count"] == 5
    assert report["cells"]["event_polarity_agreement"]["gold_count"] == 30
    assert report["cells"]["tlinks"]["gold_count"] == 5
    assert report["cells"]["tlinks"]["closure_violations"] == 0
    assert all(
        word not in target.read_text()
        for word in ("therapy", "2031-06-01", "private.xml", '"start"', '"fromID"')
    )


def test_preflight_refusals_precede_corpus_reads(tmp_path, monkeypatch):
    from pathlib import Path

    from openmed.eval.datasets import i2b2_2012

    monkeypatch.setattr(i2b2_2012, "_parse", lambda path: pytest.fail("read corpus"))
    monkeypatch.delenv("OPENMED_I2B2_2012_PATH", raising=False)
    args = ["eval", "credentialed", "--suite", "i2b2-2012-temporal", "--output"]
    assert main([*args, str(tmp_path / "output.json")]) == 2
    write_temporal_fixture(tmp_path / "private.xml")
    monkeypatch.setenv("OPENMED_I2B2_2012_PATH", str(tmp_path))
    assert main([*args, str(Path.cwd() / "credentialed-test-output.json")]) == 2
