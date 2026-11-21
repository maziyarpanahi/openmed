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


def _risk_doc(unit, age="37", zipcode="12345"):
    return TextQIDocument(
        f"{age} {zipcode}",
        [
            EntityPrediction(age, "AGE", 1.0, 0, len(age)),
            EntityPrediction(
                zipcode, "ZIPCODE", 1.0, len(age) + 1, len(age) + 1 + len(zipcode)
            ),
        ],
        hmac_text_hash(unit, KEY),
    )


def test_text_risk_uses_one_row_per_unit_and_no_profile_keys():
    from openmed.risk.text_qi_bridge import text_qi_risk

    a, b = _risk_doc("unit-a"), _risk_doc("unit-b", "58", "98765")
    reference = [a, b, _risk_doc("unit-c"), _risk_doc("unit-d", "58", "98765")]
    result = text_qi_risk([a, a, b], POLICY, reference, 2, 2)
    assert result["profile"]["row_count"] == 2
    assert result["kanon"] == {
        "k": 1,
        "target_k": 2,
        "class_count": 2,
        "meets_k": False,
    }
    assert result["population"]["achieved_k_map"] == 2
    assert result["population"]["meets_k_map"]
    encoded = json.dumps(result)
    for value in (
        a.unit_digest,
        "unit-a",
        "987",
        "50-59",
        "members",
        "equivalence_classes",
    ):
        assert value not in encoded
    assert text_qi_risk([a], POLICY, target_k=1)["population"] is None
    unmatched = text_qi_risk([a, b], POLICY, [a], 1, 1)["population"]
    assert unmatched["achieved_k_map"] == 0 and not unmatched["meets_k_map"]


def test_text_risk_requires_explicit_thresholds_and_units():
    from openmed.risk.text_qi_bridge import text_qi_risk

    doc = _risk_doc("unit-a")
    for args in ((None, None, None), ([doc], 1, None), (None, 1, 1), ([doc], True, 1)):
        with pytest.raises(ValueError):
            text_qi_risk([doc], POLICY, *args)
    with pytest.raises(ValueError, match="keyed privacy unit"):
        text_qi_risk([TextQIDocument(doc.text, doc.spans)], POLICY, target_k=1)


def test_local_concept_rates_lower_k_without_disclosing_concepts():
    from openmed.risk.text_qi_bridge import text_qi_risk

    def doc(unit, concept):
        return TextQIDocument(
            concept,
            [EntityPrediction(concept, "CONDITION", 1, 0, len(concept))],
            hmac_text_hash(unit, KEY),
        )

    rare, common = "Toy Condition A", "Toy Condition B"
    docs = [doc("a", rare), doc("a", common), doc("b", common)]
    table = {("CONDITION", rare): 0.01, ("CONDITION", common): 0.10}
    policy = TextQIPolicy(10, concept_frequencies=table, rare_frequency_threshold=0.10)
    assert text_qi_risk(docs, POLICY, target_k=2)["kanon"]["k"] == 2
    assert "rare_attribute" not in bridge_text_qis(docs, POLICY).rows[0]
    result = bridge_text_qis(docs, policy, by_unit=True)
    assert [r["rare_attribute"] for r in result.rows] == ["1", "0"]
    risk = text_qi_risk(docs, policy, target_k=2)
    assert risk["kanon"]["k"] == 1
    for raw in (rare, common):
        assert raw not in json.dumps([result.rows, result.profile, risk])
        assert raw not in repr(policy)
    unknown = bridge_text_qis([doc("c", "Unlisted")], policy)
    assert unknown.profile["columns"]["rare_attribute"]["unmapped_count"] == 1


@pytest.mark.parametrize(
    "table,threshold",
    [
        (None, 0.1),
        ({}, None),
        ({("CONDITION", "secret"): float("nan")}, 0.1),
        ({"secret": 0.1}, 0.3),
    ],
)
def test_invalid_concept_rates_are_value_free(table, threshold):
    with pytest.raises(ValueError) as error:
        TextQIPolicy(10, concept_frequencies=table, rare_frequency_threshold=threshold)
    assert "secret" not in str(error.value)
