import json
from importlib.resources import files

import jsonschema
import pytest

from openmed.core.schemas.span import OpenMedSpan
from openmed.eval.shadow_compare import compare_shadow_spans


def span(start, end, label="PERSON", doc="private-document"):
    return OpenMedSpan(doc, start, end, "hmac-sha256:" + "0" * 64, label, label)


def test_exact_and_value_free_schema():
    spans = [span(40, 52), span(70, 82, "EMAIL")]
    report = compare_shadow_spans(spans, reversed(spans))
    counts = report["by_policy_label"]["DIRECT_IDENTIFIER"]
    assert counts == {"exact": 2, "overlap": 0, "active_only": 0, "candidate_only": 0}
    assert report["missed_direct_identifiers"] == 0
    schema = json.loads(
        files("openmed.core.schemas.json")
        .joinpath("shadow_comparison.schema.json")
        .read_text()
    )
    jsonschema.validate(report, schema)
    serialized = json.dumps(report)
    for value in ("private-document", "hmac", "40", "52", "70", "82"):
        assert value not in serialized


def test_missing_person_and_candidate_only():
    result = compare_shadow_spans([span(2, 6)], [span(6, 9)])
    assert result["missed_direct_identifiers"] == 1
    assert result["by_canonical_label"]["PERSON"] == {
        "active_only": 1,
        "candidate_only": 1,
        "exact": 0,
        "overlap": 0,
    }


def test_one_to_one_overlap_prefers_exact_and_deduplicates():
    active = [span(0, 5), span(0, 5), span(1, 8), span(3, 9)]
    result = compare_shadow_spans(active, [span(0, 5), span(2, 6)])
    assert result["by_canonical_label"]["PERSON"] == {
        "active_only": 1,
        "candidate_only": 0,
        "exact": 1,
        "overlap": 1,
    }


def test_distinct_documents_rejected_without_echoing_values():
    with pytest.raises(ValueError, match="^expected nonempty spans from one document$"):
        compare_shadow_spans([span(0, 1)], [span(0, 1, doc="another")])
