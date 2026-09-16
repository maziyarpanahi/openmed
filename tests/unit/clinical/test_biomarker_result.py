from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from openmed.clinical import (
    BIOMARKER_RESULT_ADVISORY,
    assemble_biomarker_results,
    normalize_biomarker_method,
    normalize_result_polarity,
)
from openmed.core.decoding import decode_span_graph
from openmed.eval import biomarker_result_tuple_f1

_ROOT = Path(__file__).resolve().parents[3]
_CLINICAL_FIXTURE = _ROOT / "tests/fixtures/clinical/biomarker_result.jsonl"
_EVAL_FIXTURE = _ROOT / "openmed/eval/golden/fixtures/biomarker_result.jsonl"
_TUPLE_FIELDS = (
    "gene",
    "variant_or_finding",
    "result_value",
    "method",
    "result_polarity",
)


def _rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_assemble_biomarker_results_matches_every_synthetic_gold_row() -> None:
    for row in _rows(_CLINICAL_FIXTURE):
        predicted = assemble_biomarker_results(row["text"], row["mentions"])
        assert predicted == row["gold"], row["id"]


def test_assembler_uses_shared_span_graph_decoder() -> None:
    row = _rows(_CLINICAL_FIXTURE)[0]
    with patch(
        "openmed.clinical.biomarker_result.decode_span_graph",
        wraps=decode_span_graph,
    ) as decoder:
        assemble_biomarker_results(row["text"], row["mentions"])
    decoder.assert_called_once()


def test_every_populated_field_has_round_trip_utf8_byte_provenance() -> None:
    for row in _rows(_CLINICAL_FIXTURE):
        source = row["text"]
        encoded = source.encode("utf-8")
        for result in assemble_biomarker_results(source, row["mentions"]):
            populated_fields = {
                field for field in _TUPLE_FIELDS if result[field] is not None
            }
            assert set(result["provenance_spans"]) == populated_fields
            for field, span in result["provenance_spans"].items():
                assert span["unit"] == "utf8_byte"
                surface = encoded[span["start"] : span["end"]].decode("utf-8")
                if field == "method":
                    assert normalize_biomarker_method(surface) == result[field]
                elif field == "result_polarity":
                    assert normalize_result_polarity(surface) == result[field]
                else:
                    assert surface == result[field]


def test_negated_equivocal_and_detected_polarities_are_normalized() -> None:
    results = [
        result
        for row in _rows(_CLINICAL_FIXTURE)
        for result in assemble_biomarker_results(row["text"], row["mentions"])
    ]
    assert {
        result["result_value"]: result["result_polarity"] for result in results
    } == {
        "detected": "detected",
        "not detected": "not_detected",
        "3+": "detected",
        "equivocal": "equivocal",
    }


def test_unknown_polarity_fails_closed_unless_upstream_supplies_it() -> None:
    text = "EGFR expression 80% by IHC."
    mentions = [
        {"label": "gene", "start": 0, "end": 4},
        {"label": "result", "start": 16, "end": 19},
        {"label": "method", "start": 23, "end": 26},
    ]
    with pytest.raises(ValueError, match="provide an explicit polarity"):
        assemble_biomarker_results(text, mentions)

    mentions[1]["polarity"] = "detected"
    assert (
        assemble_biomarker_results(text, mentions)[0]["result_polarity"] == "detected"
    )


def test_clause_boundaries_prevent_cross_result_linking() -> None:
    text = "EGFR detected by NGS. KRAS not detected by PCR."
    mentions = [
        {"label": "gene", "start": 0, "end": 4},
        {"label": "result", "start": 5, "end": 13},
        {"label": "method", "start": 17, "end": 20},
        {"label": "gene", "start": 22, "end": 26},
        {"label": "result", "start": 27, "end": 39},
        {"label": "method", "start": 43, "end": 46},
    ]
    with patch(
        "openmed.clinical.biomarker_result.decode_span_graph",
        wraps=decode_span_graph,
    ) as decoder:
        results = assemble_biomarker_results(text, mentions)
    assert [(result["gene"], result["method"]) for result in results] == [
        ("EGFR", "NGS"),
        ("KRAS", "PCR"),
    ]
    candidate_edges = decoder.call_args.args[1]
    assert len(candidate_edges) == 4


def test_advisory_is_emitted_on_every_result() -> None:
    for row in _rows(_CLINICAL_FIXTURE):
        for result in assemble_biomarker_results(row["text"], row["mentions"]):
            assert result["advisory"] == BIOMARKER_RESULT_ADVISORY


def test_offline_eval_fixture_meets_exact_tuple_f1_gate() -> None:
    predicted: list[dict[str, object]] = []
    gold: list[dict[str, object]] = []
    with patch("socket.create_connection", side_effect=AssertionError("network used")):
        for row in _rows(_EVAL_FIXTURE):
            predicted.extend(assemble_biomarker_results(row["text"], row["mentions"]))
            gold.extend(row["gold"])

    metrics = biomarker_result_tuple_f1(predicted, gold)
    assert metrics.f1 >= 0.85
    assert metrics.f1 == 1.0
    assert metrics.true_positives == 6
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_exact_tuple_metric_uses_multiset_semantics() -> None:
    item = {
        "gene": "EGFR",
        "variant_or_finding": "L858R",
        "result_value": "detected",
        "method": "NGS",
        "result_polarity": "detected",
    }
    metrics = biomarker_result_tuple_f1([item], [item, item])
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(2 / 3)


def test_committed_clinical_and_eval_fixtures_stay_identical() -> None:
    assert _CLINICAL_FIXTURE.read_bytes() == _EVAL_FIXTURE.read_bytes()


@pytest.mark.parametrize(
    ("mentions", "error", "match"),
    [
        ([{"label": "gene", "start": -1, "end": 4}], ValueError, "offsets"),
        ([{"label": "therapy", "start": 0, "end": 4}], ValueError, "unknown"),
        ([{"label": "gene", "start": 0, "end": 4, "text": "KRAS"}], ValueError, "text"),
    ],
)
def test_invalid_mentions_are_rejected(
    mentions: list[dict[str, object]],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        assemble_biomarker_results("EGFR detected", mentions)
