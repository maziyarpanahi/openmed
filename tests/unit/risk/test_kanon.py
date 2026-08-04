"""Tests for k-anonymity / l-diversity / t-closeness measurement (issue #500)."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from openmed.risk import enforce_kanon, kanon_report


def _classes_by_size(report):
    return sorted(c["size"] for c in report["equivalence_classes"])


class TestKAnonymity:
    RECORDS = [
        {"age": 30, "zip": "1000", "disease": "flu"},
        {"age": 30, "zip": "1000", "disease": "cold"},
        {"age": 41, "zip": "2000", "disease": "flu"},  # singleton
    ]

    def test_k_min_and_singleton_class(self):
        report = kanon_report(
            self.RECORDS,
            quasi_identifiers=["age", "zip"],
            sensitive_attributes=["disease"],
        )
        assert report["k"] == 1
        assert report["class_count"] == 2
        assert _classes_by_size(report) == [1, 2]
        singletons = [c for c in report["equivalence_classes"] if c["size"] == 1]
        assert len(singletons) == 1

    def test_higher_k_when_all_classes_share_key(self):
        records = [
            {"age": 30, "zip": "1000", "disease": "flu"},
            {"age": 30, "zip": "1000", "disease": "cold"},
        ]
        report = kanon_report(
            records, quasi_identifiers=["age", "zip"], sensitive_attributes=["disease"]
        )
        assert report["k"] == 2
        assert report["class_count"] == 1

    def test_missing_null_empty_and_unnormalized_qis_do_not_inflate_k(self):
        records = [
            {"disease": "a"},
            {"age": None, "disease": "b"},
            {"age": "", "disease": "c"},
            {"age": "unknown", "disease": "d"},
        ]

        report = kanon_report(records, quasi_identifiers=["age"])

        assert report["k"] == 1
        assert report["class_count"] == 4


class TestLDiversity:
    @pytest.mark.parametrize("value", [None, "", float("nan")])
    def test_missing_or_nonfinite_sensitive_values_fail_closed(self, value):
        with pytest.raises(ValueError, match="Sensitive attribute"):
            kanon_report(
                [{"g": "A", "disease": "flu"}, {"g": "A", "disease": value}],
                quasi_identifiers=["g"],
                sensitive_attributes=["disease"],
            )

        with pytest.raises(ValueError, match="Sensitive attribute"):
            kanon_report(
                [{"g": "A", "disease": "flu"}, {"g": "A"}],
                quasi_identifiers=["g"],
                sensitive_attributes=["disease"],
            )

    def test_distinct_is_one_when_each_class_has_single_value(self):
        records = [
            {"age": 30, "zip": "1000", "disease": "flu"},
            {"age": 30, "zip": "1000", "disease": "flu"},
            {"age": 41, "zip": "2000", "disease": "cold"},
        ]
        report = kanon_report(
            records, quasi_identifiers=["age", "zip"], sensitive_attributes=["disease"]
        )
        for cls in report["equivalence_classes"]:
            assert cls["l_diversity"]["disease"]["distinct"] == 1
            assert cls["l_diversity"]["disease"]["entropy"] == 0.0
        assert report["l"]["disease"] == 1
        assert report["l_diversity"]["disease"]["min_distinct"] == 1

    def test_distinct_counts_multiple_sensitive_values(self):
        records = [
            {"g": "A", "disease": "flu"},
            {"g": "A", "disease": "cold"},
        ]
        report = kanon_report(
            records, quasi_identifiers=["g"], sensitive_attributes=["disease"]
        )
        cls = report["equivalence_classes"][0]
        assert cls["l_diversity"]["disease"]["distinct"] == 2
        assert cls["l_diversity"]["disease"]["entropy"] == pytest.approx(1.0)

    def test_entropy_metric_selects_overall_entropy(self):
        records = [
            {"g": "A", "disease": "flu"},
            {"g": "A", "disease": "cold"},
        ]
        report = kanon_report(
            records,
            quasi_identifiers=["g"],
            sensitive_attributes=["disease"],
            l_metric="entropy",
        )
        assert report["l"]["disease"] == pytest.approx(1.0)
        assert report["l_diversity"]["disease"]["min_entropy"] == pytest.approx(1.0)


class TestTCloseness:
    def test_class_matching_global_distribution_is_zero(self):
        records = [
            {"g": "A", "disease": "flu"},
            {"g": "A", "disease": "cold"},
            {"g": "B", "disease": "flu"},
            {"g": "B", "disease": "cold"},
        ]
        report = kanon_report(
            records, quasi_identifiers=["g"], sensitive_attributes=["disease"]
        )
        for cls in report["equivalence_classes"]:
            assert cls["t_closeness"]["disease"] == pytest.approx(0.0, abs=1e-9)
        assert report["t_closeness"]["disease"] == pytest.approx(0.0, abs=1e-9)

    def test_skewed_class_has_positive_distance(self):
        records = [
            {"g": "A", "disease": "flu"},
            {"g": "A", "disease": "flu"},
            {"g": "B", "disease": "cold"},
            {"g": "B", "disease": "cold"},
        ]
        report = kanon_report(
            records, quasi_identifiers=["g"], sensitive_attributes=["disease"]
        )
        # Global is 50/50; each class is 100% one value -> TV distance 0.5.
        assert report["t_closeness"]["disease"] == pytest.approx(0.5)


class TestContract:
    RECORDS = [
        {"age": 30, "zip": "1000", "disease": "flu"},
        {"age": 30, "zip": "1000", "disease": "cold"},
        {"age": 41, "zip": "2000", "disease": "flu"},
    ]

    def test_deterministic_and_json_serializable(self):
        kwargs = dict(
            quasi_identifiers=["age", "zip"], sensitive_attributes=["disease"]
        )
        first = kanon_report(self.RECORDS, **kwargs)
        second = kanon_report(self.RECORDS, **kwargs)
        assert first == second
        assert json.loads(json.dumps(first)) == first

    def test_importable_and_exported(self):
        import openmed.risk as risk

        assert hasattr(risk, "kanon_report")
        assert "kanon_report" in risk.__all__

    def test_no_sensitive_attributes_reports_k_only(self):
        report = kanon_report(self.RECORDS, quasi_identifiers=["age", "zip"])
        assert report["k"] == 1
        assert report["l"] == {}
        assert report["l_diversity"] == {}
        assert report["t_closeness"] == {}

    def test_auto_quasi_identifier_detection_runs(self):
        # Without explicit QIs, fall back to risk_report-consistent profiling.
        records = [
            {"text": "SSN 123-45-6789"},
            {"text": "SSN 987-65-4321"},
        ]
        report = kanon_report(records, sensitive_attributes=None)
        assert report["k"] >= 1
        assert json.loads(json.dumps(report)) == report

    def test_unsupported_l_metric_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported l_metric"):
            kanon_report(self.RECORDS, quasi_identifiers=["age"], l_metric="ratio")


def test_structured_rows_preserve_reserved_and_container_named_columns() -> None:
    records = [
        {"rows": "shared", "items": ["first"], "content": "one"},
        {"rows": "shared", "items": ["second"], "content": "two"},
    ]

    report = kanon_report(records, quasi_identifiers=["rows"])

    assert report["k"] == 2
    assert report["class_count"] == 1
    enforced = enforce_kanon(
        records,
        quasi_identifiers=["rows"],
        target_k=2,
        remove_direct_identifiers=False,
    )
    assert enforced["records"] == records


def test_typed_clinical_scalars_are_supported_without_type_collisions() -> None:
    records = [
        {
            "visit_date": date(2026, 7, 26),
            "amount": Decimal("1.20"),
            "payload": b"\x01",
            "disease": 1,
        },
        {
            "visit_date": date(2026, 7, 26),
            "amount": Decimal("1.20"),
            "payload": b"\x01",
            "disease": "1",
        },
    ]

    report = kanon_report(
        records,
        quasi_identifiers=["visit_date", "amount", "payload"],
        sensitive_attributes=["disease"],
    )

    assert report["k"] == 2
    assert report["l"]["disease"] == 2
    assert json.loads(json.dumps(report)) == report

    typed_qi_report = kanon_report(
        [{"value": 1}, {"value": "1"}],
        quasi_identifiers=["value"],
    )
    assert typed_qi_report["k"] == 1
    assert typed_qi_report["class_count"] == 2

    precise_values = [
        Decimal("1.0000000000000000000000000000000000000001"),
        Decimal("1.0000000000000000000000000000000000000002"),
    ]
    precise_report = kanon_report(
        [
            {"group": "A", "measurement": precise_values[0]},
            {"group": "A", "measurement": precise_values[1]},
        ],
        quasi_identifiers=["group"],
        sensitive_attributes=["measurement"],
    )
    assert precise_report["l"]["measurement"] == 2


def test_signed_float_zero_is_one_sensitive_value() -> None:
    report = kanon_report(
        [
            {"group": "A", "measurement": -0.0},
            {"group": "A", "measurement": 0.0},
        ],
        quasi_identifiers=["group"],
        sensitive_attributes=["measurement"],
    )

    assert report["l"]["measurement"] == 1


def test_signed_float_zero_qis_preserve_published_representation() -> None:
    report = kanon_report(
        [{"measurement": -0.0}, {"measurement": 0.0}],
        quasi_identifiers=["measurement"],
    )

    assert report["k"] == 1
    assert report["class_count"] == 2


def test_unicode_canonical_equivalents_are_one_sensitive_value() -> None:
    report = kanon_report(
        [
            {"group": "A", "condition": "café"},
            {"group": "A", "condition": "cafe\u0301"},
        ],
        quasi_identifiers=["group"],
        sensitive_attributes=["condition"],
    )

    assert report["l"]["condition"] == 1


def test_explicit_qis_preserve_published_case_and_punctuation() -> None:
    report = kanon_report(
        [{"city": "Paris"}, {"city": "paris"}],
        quasi_identifiers=["city"],
    )

    assert report["k"] == 1
    assert report["class_count"] == 2


def test_explicit_qis_preserve_published_unicode_representation() -> None:
    report = kanon_report(
        [{"city": "café"}, {"city": "cafe\u0301"}],
        quasi_identifiers=["city"],
    )

    assert report["k"] == 1
    assert report["class_count"] == 2


def test_aware_datetime_qis_preserve_published_offset_representation() -> None:
    first = datetime(2020, 1, 1, tzinfo=timezone.utc)
    second = datetime(
        2020,
        1,
        1,
        1,
        tzinfo=timezone(timedelta(hours=1)),
    )
    assert first == second
    report = kanon_report(
        [
            {"event_time": first, "measurement": first},
            {"event_time": second, "measurement": second},
        ],
        quasi_identifiers=["event_time"],
        sensitive_attributes=["measurement"],
    )

    assert report["k"] == 1
    assert report["class_count"] == 2
    assert report["l"]["measurement"] == 1


def test_t_closeness_measures_published_sensitive_representations() -> None:
    first = datetime(2020, 1, 1, tzinfo=timezone.utc)
    second = datetime(
        2020,
        1,
        1,
        1,
        tzinfo=timezone(timedelta(hours=1)),
    )
    report = kanon_report(
        [
            {"group": "A", "measurement": first},
            {"group": "A", "measurement": first},
            {"group": "B", "measurement": second},
            {"group": "B", "measurement": second},
        ],
        quasi_identifiers=["group"],
        sensitive_attributes=["measurement"],
    )

    assert report["l"]["measurement"] == 1
    assert report["t_closeness"]["measurement"] == pytest.approx(0.5)


def test_internal_missing_marker_cannot_be_injected_as_a_literal_qi() -> None:
    report = kanon_report(
        [{}, {"value": "__OPENMED_INTERNAL_QI__:state:missing"}],
        quasi_identifiers=["value"],
    )

    assert report["k"] == 1
    assert report["class_count"] == 2


@pytest.mark.parametrize("value", [" ", "\t\n", " flu ", b""])
def test_ambiguous_sensitive_values_fail_closed(value: object) -> None:
    with pytest.raises(ValueError, match="Sensitive attribute"):
        kanon_report(
            [{"group": "A", "disease": value}],
            quasi_identifiers=["group"],
            sensitive_attributes=["disease"],
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("quasi_identifiers", "age"),
        ("sensitive_attributes", "disease"),
    ],
)
def test_column_arguments_reject_bare_strings(argument: str, value: str) -> None:
    with pytest.raises(TypeError, match="sequence of column names"):
        kanon_report(
            [{"age": 30, "disease": "flu"}],
            **{argument: value},
        )


def test_direct_dataframe_scalars_are_normalized_to_python_values() -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "visit_date": [
                pd.Timestamp("2026-07-26"),
                pd.Timestamp("2026-07-26"),
            ],
            "count": pd.Series([np.int64(2), np.int64(2)], dtype=object),
        }
    )

    report = kanon_report(
        frame,
        quasi_identifiers=["visit_date", "count"],
    )

    assert report["k"] == 2
    assert json.loads(json.dumps(report)) == report

    enforced = enforce_kanon(
        frame,
        quasi_identifiers=["visit_date", "count"],
        target_k=2,
        remove_direct_identifiers=False,
    )
    assert type(enforced["records"][0]["visit_date"]) is datetime
    assert type(enforced["records"][0]["count"]) is int


@pytest.mark.parametrize("temporal_kind", ["datetime_ns", "time_ns"])
def test_polars_temporal_precision_cannot_collapse_equivalence_classes(
    temporal_kind: str,
) -> None:
    pl = pytest.importorskip("polars", exc_type=ImportError)
    dtype = pl.Datetime("ns", "UTC") if temporal_kind == "datetime_ns" else pl.Time
    frame = pl.DataFrame({"event_time": pl.Series("event_time", [1, 2], dtype=dtype)})

    with pytest.raises(ValueError, match="sub-microsecond precision"):
        kanon_report(frame, quasi_identifiers=["event_time"])


def test_direct_dataframe_rejects_sub_microsecond_timestamp_collapse() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "visit_date": [
                pd.Timestamp("2026-07-26T00:00:00.000000001"),
                pd.Timestamp("2026-07-26T00:00:00.000000002"),
            ]
        }
    )

    with pytest.raises(ValueError, match="sub-microsecond precision"):
        kanon_report(frame, quasi_identifiers=["visit_date"])
