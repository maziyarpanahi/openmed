import hashlib
import hmac
import json

from openmed.risk.release_composition import ReleasedTable, assess_release_composition


def table(groups):
    return ReleasedTable(
        [
            {
                "token": hmac.new(
                    b"synthetic", str(i).encode(), hashlib.sha256
                ).hexdigest(),
                "region": "sensitive-" + str(group),
            }
            for i, group in enumerate(groups)
        ],
        ("region",),
        "token",
    )


def test_each_release_k_five_but_combined_k_one():
    left = table([i // 5 for i in range(25)])
    right = table([i % 5 for i in range(25)])
    result = assess_release_composition([left, right], target_k=5)
    assert result.composed_k == 1 and result.single_release_k == (5, 5)
    assert result.units_below_target == 25 and result.drop_pairs == ((1, 2),)
    assert result.evidence.prior_release_overlap_assessed
    assert result.evidence.risk_status == "increase_observed"
    serialized = json.dumps(result.to_dict())
    assert "sensitive-" not in serialized and left.rows[0]["token"] not in serialized


def test_identical_releases_and_repeated_rows_count_distinct_units():
    left = table([0] * 5)
    repeat = ReleasedTable(list(left.rows) * 2, ("region",), "token")
    result = assess_release_composition([left, repeat, left], target_k=5)
    assert result.composed_k == 5 and result.units_below_target == 0
    assert (
        result.drop_pairs == ()
        and result.evidence.risk_status == "no_material_increase_observed"
    )


def test_missing_and_disjoint_keys_return_unassessed():
    left = table([0, 0])
    for right, reason in [
        (ReleasedTable(left.rows, ("region",), None), "missing_linking_key"),
        (
            ReleasedTable(
                [{"token": "a" * 64, "region": "private"}], ("region",), "token"
            ),
            "no_common_units",
        ),
    ]:
        result = assess_release_composition([left, right], target_k=2)
        assert result.evidence.risk_status == "not_assessed" and result.reason == reason
        assert not result.evidence.prior_release_overlap_assessed
        assert result.composed_k is None and result.units_below_target is None


def test_declared_age_generalization_is_applied():
    left = table([0] * 5)
    age = ReleasedTable(
        [{"token": r["token"], "age": 20 + i} for i, r in enumerate(left.rows)],
        ("age",),
        "token",
        {"age": 1},
    )
    assert assess_release_composition([left, age], target_k=5).composed_k == 5
