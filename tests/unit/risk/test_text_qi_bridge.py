import json

import pytest

from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash
from openmed.processing.outputs import EntityPrediction
from openmed.risk.kanon import kanon_report
from openmed.risk.text_qi_bridge import TextQIDocument, TextQIPolicy, bridge_text_qis

POLICY = TextQIPolicy(10, {"Fictionville": "region_a"}, {"weld artist": "trade"})
KEY = b"synthetic"
VALUES = [
    ("37", "AGE"),
    ("2022-03-05", "DATE"),
    ("12345", "ZIPCODE"),
    ("Fictionville", "LOCATION"),
    ("weld artist", "OCCUPATION"),
]


def test_all_columns_and_direct_tabular_input():
    text = " / ".join(v for v, _ in VALUES)
    spans = [
        EntityPrediction(v, label, 1.0, text.index(v), text.index(v) + len(v))
        for v, label in VALUES
    ]
    doc = TextQIDocument(text, spans)
    result = bridge_text_qis([doc, doc], POLICY)
    expected = {
        "age_band": "30-39",
        "year": "2022",
        "zip3": "123",
        "region": "region_a",
        "occupation_group": "trade",
    }
    assert result.rows == (expected, expected)
    assert result.profile["columns"]["year"] == {"span_count": 2, "unmapped_count": 0}
    assert kanon_report(result.rows, quasi_identifiers=list(expected))["k"] == 2
    for value, _ in VALUES:
        assert value not in json.dumps([result.rows, result.profile])


def test_keyed_hash_binding_and_grouping():
    token = hmac_text_hash("subject", KEY)

    def document(age):
        span = OpenMedSpan("doc", 0, len(age), hmac_text_hash(age, KEY), "AGE", "AGE")
        return TextQIDocument(age, [span], token)

    result = bridge_text_qis(
        [document("23"), document("48")], POLICY, by_unit=True, span_hmac_key=KEY
    )
    assert len(result.rows) == 1 and result.rows[0]["age_band"] == "20-29|40-49"
    assert token not in json.dumps(result.rows)
    with pytest.raises(ValueError, match="does not match"):
        bridge_text_qis([document("23")], POLICY, span_hmac_key=b"wrong")


@pytest.mark.parametrize(
    "spans",
    [
        [EntityPrediction("37", "AGE", 1.0, 1, 3)],
        [EntityPrediction("37", "AGE", 1.0, 0, 3)],
        [EntityPrediction("73", "AGE", 1.0, 0, 2)],
        [
            EntityPrediction("37", "AGE", 1.0, 0, 2),
            EntityPrediction("7", "AGE", 1.0, 1, 2),
        ],
    ],
)
def test_bad_spans_fail_before_rows_are_returned(spans):
    with pytest.raises(ValueError):
        bridge_text_qis([TextQIDocument("37", spans)], POLICY)


def test_unmapped_surface_is_counted_and_dropped():
    doc = TextQIDocument(
        "unmapped", [EntityPrediction("unmapped", "LOCATION", 1.0, 0, 8)]
    )
    result = bridge_text_qis([doc], POLICY)
    assert result.rows[0]["region"] is None
    assert result.profile["columns"]["region"]["unmapped_count"] == 1
