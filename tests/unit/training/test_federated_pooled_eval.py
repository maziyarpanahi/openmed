"""Pooled federated evaluation through an injected secure-sum port."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from statistics import NormalDist
from typing import Any

import pytest

from openmed.eval.metrics import EvalSpan, compute_exact_span_f1
from openmed.training.federated_metrics import (
    FederatedParticipantCountBand,
    FederatedPrivacyMechanism,
    FederatedUncertaintyMethod,
)
from openmed.training.federated_pooled_eval import (
    POOLED_EVALUATION_SCHEMA_VERSION,
    PooledEvaluationError,
    PooledEvaluationErrorCode,
    PooledEvaluationLayout,
    PooledEvaluationReport,
    SecureSumOutcome,
    SecureSumRequest,
    SiteCountVector,
    build_site_count_vector,
    encode_site_counts,
    pool_federated_evaluation,
)

Code = PooledEvaluationErrorCode
P = FederatedPrivacyMechanism
EVALUATION_REF = "sha256:" + "c" * 64
REPORT_FIELDS = {
    "direct_identifier_metrics",
    "evaluation_ref",
    "layout_digest",
    "minimum_label_support",
    "minimum_sites",
    "other_label_metrics",
    "schema_version",
    "site_count_band",
}
ENVELOPE_FIELDS = {
    "aggregate_value",
    "clipping_lower_bound",
    "clipping_upper_bound",
    "confidence_level",
    "metric_id",
    "metric_kind",
    "minimum_group_size",
    "participant_count_band",
    "privacy_mechanism",
    "privacy_mechanism_version",
    "schema_version",
    "uncertainty_lower_bound",
    "uncertainty_method",
    "uncertainty_upper_bound",
}
LABELS = ("EMAIL", "PHONE", "AGE", "DISEASE", "CITY")

# Distinctive per-site sentinel counts. CITY stays tiny at every site so its
# pooled slice is below the minimum label support and must be suppressed.
SITE_COUNTS: tuple[dict[str, tuple[int, int, int]], ...] = (
    {
        "EMAIL": (7331, 1213, 2417),
        "PHONE": (6007, 1031, 1999),
        "AGE": (5101, 907, 1601),
        "DISEASE": (8317, 2003, 1297),
        "CITY": (1, 0, 0),
    },
    {
        "EMAIL": (4919, 883, 3301),
        "PHONE": (3779, 619, 2111),
        "AGE": (6421, 1427, 1093),
        "DISEASE": (9011, 1801, 1789),
        "CITY": (0, 1, 0),
    },
    {
        "EMAIL": (5927, 1709, 1171),
        "PHONE": (4483, 2251, 1129),
        "AGE": (7187, 1129, 2203),
        "DISEASE": (7547, 1531, 2381),
        "CITY": (0, 0, 1),
    },
    {
        "EMAIL": (6661, 947, 1913),
        "PHONE": (5303, 1153, 1433),
        "AGE": (4801, 1381, 1709),
        "DISEASE": (8669, 1063, 1867),
        "CITY": (1, 0, 0),
    },
    {
        "EMAIL": (3881, 1279, 2687),
        "PHONE": (6823, 1621, 991),
        "AGE": (5689, 2137, 1453),
        "DISEASE": (9341, 1217, 2029),
        "CITY": (0, 0, 0),
    },
)


class FakeSecureSumPort:
    """Test double: holds site vectors privately and releases only their sum."""

    def __init__(
        self,
        contributions: Sequence[SiteCountVector | tuple[int, ...]],
        *,
        claim_verified: bool | None = None,
        echo_mechanism: tuple[FederatedPrivacyMechanism, str] | None = None,
        layout_digest: str | None = None,
    ) -> None:
        self._contributions = list(contributions)
        self._claim_verified = claim_verified
        self._echo_mechanism = echo_mechanism
        self._layout_digest = layout_digest
        self.requests: list[SecureSumRequest] = []

    def secure_sum(self, request: SecureSumRequest) -> SecureSumOutcome:
        self.requests.append(request)
        vectors = [
            item.values if isinstance(item, SiteCountVector) else tuple(item)
            for item in self._contributions
        ]
        in_bounds = all(
            len(vector) == request.vector_length
            and all(0 <= value <= request.entry_upper_bound for value in vector)
            for vector in vectors
        )
        summed = tuple(sum(column) for column in zip(*vectors)) or (
            (0,) * request.vector_length
        )
        mechanism, version = self._echo_mechanism or (
            request.privacy_mechanism,
            request.privacy_mechanism_version,
        )
        return SecureSumOutcome(
            layout_digest=self._layout_digest or request.layout_digest,
            summed_vector=summed,
            contributor_count=len(vectors),
            bounds_verified=(
                in_bounds if self._claim_verified is None else self._claim_verified
            ),
            privacy_mechanism=mechanism,
            privacy_mechanism_version=version,
        )

    def leaky_debug_dump(self) -> str:
        """Negative control: what a transcript would look like if it leaked."""

        return json.dumps(
            [
                list(item.values if isinstance(item, SiteCountVector) else item)
                for item in self._contributions
            ]
        )


class RaisingPort:
    def secure_sum(self, request: SecureSumRequest) -> SecureSumOutcome:
        raise RuntimeError("site-alpha vector 7331 rejected at /private/site/path")


def _layout(**overrides: object) -> PooledEvaluationLayout:
    arguments: dict[str, object] = {"labels": LABELS, "max_count_per_entry": 20000}
    arguments.update(overrides)
    return PooledEvaluationLayout(**arguments)


def _site_vectors(layout: PooledEvaluationLayout) -> list[SiteCountVector]:
    return [encode_site_counts(layout, counts) for counts in SITE_COUNTS]


def _pool(
    layout: PooledEvaluationLayout,
    port: object,
    **overrides: object,
) -> PooledEvaluationReport:
    arguments: dict[str, object] = {
        "layout": layout,
        "port": port,
        "evaluation_ref": EVALUATION_REF,
        "minimum_sites": 5,
        "minimum_label_support": 10,
    }
    arguments.update(overrides)
    return pool_federated_evaluation(**arguments)


def _plaintext_totals(labels: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    totals: dict[str, tuple[int, int, int]] = {}
    for label in labels:
        tp = sum(site[label][0] for site in SITE_COUNTS)
        fp = sum(site[label][1] for site in SITE_COUNTS)
        fn = sum(site[label][2] for site in SITE_COUNTS)
        totals[label] = (tp, fp, fn)
    return totals


def _metric(report: PooledEvaluationReport, metric_id: str):
    matches = [
        envelope
        for envelope in (
            *report.direct_identifier_metrics,
            *report.other_label_metrics,
        )
        if envelope.metric_id == metric_id
    ]
    assert len(matches) == 1, metric_id
    return matches[0]


def _reference_wilson(successes: int, trials: int, level: float) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    center = (phat + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * trials)) / trials)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


# -- pooled metrics equal plaintext-sum metrics ------------------------------


def test_pooled_metrics_equal_plaintext_sum_metrics() -> None:
    layout = _layout()
    report = _pool(layout, FakeSecureSumPort(_site_vectors(layout)))
    totals = _plaintext_totals(LABELS)

    for label in ("EMAIL", "PHONE"):
        tp, fp, fn = totals[label]
        lowered = label.lower()
        precision = _metric(report, f"pooled.di.label.{lowered}.precision")
        recall = _metric(report, f"pooled.di.label.{lowered}.recall")
        assert precision.aggregate_value == pytest.approx(tp / (tp + fp))
        assert recall.aggregate_value == pytest.approx(tp / (tp + fn))
    for label in ("AGE", "DISEASE"):
        tp, fp, fn = totals[label]
        lowered = label.lower()
        precision = _metric(report, f"pooled.other.label.{lowered}.precision")
        recall = _metric(report, f"pooled.other.label.{lowered}.recall")
        assert precision.aggregate_value == pytest.approx(tp / (tp + fp))
        assert recall.aggregate_value == pytest.approx(tp / (tp + fn))


def test_pooled_rates_carry_wilson_intervals_from_pooled_counts() -> None:
    layout = _layout()
    report = _pool(
        layout, FakeSecureSumPort(_site_vectors(layout)), confidence_level=0.9
    )
    tp, _, fn = _plaintext_totals(("EMAIL",))["EMAIL"]
    recall = _metric(report, "pooled.di.label.email.recall")
    lower, upper = _reference_wilson(tp, tp + fn, 0.9)

    assert recall.uncertainty_method is FederatedUncertaintyMethod.CONFIDENCE_INTERVAL
    assert recall.uncertainty_lower_bound == pytest.approx(lower)
    assert recall.uncertainty_upper_bound == pytest.approx(upper)
    assert recall.confidence_level == 0.9
    assert recall.privacy_mechanism is P.THRESHOLD_ONLY
    assert recall.privacy_mechanism_version == "v1"


def test_partition_micro_metrics_pool_only_their_own_labels() -> None:
    layout = _layout()
    report = _pool(layout, FakeSecureSumPort(_site_vectors(layout)))
    totals = _plaintext_totals(LABELS)

    di_tp = sum(totals[label][0] for label in ("EMAIL", "PHONE"))
    di_fn = sum(totals[label][2] for label in ("EMAIL", "PHONE"))
    other = ("AGE", "DISEASE", "CITY")
    other_tp = sum(totals[label][0] for label in other)
    other_fp = sum(totals[label][1] for label in other)

    di_recall = _metric(report, "pooled.di.micro.recall")
    other_precision = _metric(report, "pooled.other.micro.precision")
    assert di_recall.aggregate_value == pytest.approx(di_tp / (di_tp + di_fn))
    assert other_precision.aggregate_value == pytest.approx(
        other_tp / (other_tp + other_fp)
    )


def test_site_vectors_from_spans_match_eval_exact_span_counts() -> None:
    layout = _layout(max_count_per_entry=50)
    documents = [
        (
            [
                EvalSpan(0, 5, "EMAIL"),
                EvalSpan(10, 14, "AGE"),
                EvalSpan(20, 28, "DISEASE"),
            ],
            [
                EvalSpan(0, 5, "EMAIL"),
                EvalSpan(10, 13, "AGE"),
                EvalSpan(30, 34, "PHONE"),
            ],
        ),
        (
            [EvalSpan(2, 9, "PHONE"), EvalSpan(12, 20, "DISEASE")],
            [EvalSpan(2, 9, "PHONE"), EvalSpan(12, 20, "DISEASE")],
        ),
    ]
    vector = build_site_count_vector(layout, documents)

    expected: list[int] = []
    for label in LABELS:
        tp = fp = fn = 0
        for gold, predicted in documents:
            metrics = compute_exact_span_f1(
                [span for span in gold if span.label == label],
                [span for span in predicted if span.label == label],
            )
            tp += metrics.true_positives
            fp += metrics.false_positives
            fn += metrics.false_negatives
        expected.extend((tp, fp, fn))
    assert vector.values == tuple(expected)
    assert vector.layout_digest == layout.digest


def test_end_to_end_span_sites_pool_to_plaintext_metrics() -> None:
    layout = _layout(max_count_per_entry=50)
    site_documents = []
    for site_index in range(5):
        gold = [EvalSpan(offset, offset + 4, "EMAIL") for offset in range(0, 80, 8)]
        predicted = gold[: 6 + site_index % 3] + [EvalSpan(200, 204, "EMAIL")]
        site_documents.append([(gold, predicted)])
    vectors = [build_site_count_vector(layout, docs) for docs in site_documents]
    report = _pool(layout, FakeSecureSumPort(vectors))

    tp = sum(6 + index % 3 for index in range(5))
    recall = _metric(report, "pooled.di.label.email.recall")
    precision = _metric(report, "pooled.di.label.email.precision")
    assert recall.aggregate_value == pytest.approx(tp / 50)
    assert precision.aggregate_value == pytest.approx(tp / (tp + 5))


# -- coordinator transcript sentinel ------------------------------------------


def _sentinel_tokens() -> set[str]:
    tokens: set[str] = set()
    for site in SITE_COUNTS:
        for tp, fp, fn in site.values():
            for value in (tp, fp, fn):
                if value >= 100:
                    tokens.add(str(value))
            # Tiny CITY slices give rates such as 1.0 that are not distinctive.
            if tp + fp >= 100:
                tokens.add(repr(tp / (tp + fp)))
            if tp + fn >= 100:
                tokens.add(repr(tp / (tp + fn)))
        tokens.add(json.dumps([list(item) for item in site.values()]))
    return tokens


def _contains_any(text: str, tokens: set[str]) -> list[str]:
    return [
        token
        for token in tokens
        if re.search(rf"(?<![0-9A-Za-z.]){re.escape(token)}(?![0-9A-Za-z])", text)
    ]


def test_coordinator_transcript_has_no_per_site_counts_or_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    layout = _layout()
    port = FakeSecureSumPort(_site_vectors(layout))
    caplog.set_level(logging.DEBUG)
    report = _pool(layout, port)

    transcript = "\n".join(
        [
            report.to_json(),
            repr(report),
            *(repr(request) for request in port.requests),
            caplog.text,
        ]
    )
    tokens = _sentinel_tokens()

    assert _contains_any(transcript, tokens) == []
    payload = report.to_dict()
    assert set(payload) == REPORT_FIELDS
    for envelope in (
        *payload["direct_identifier_metrics"],
        *payload["other_label_metrics"],
    ):
        assert set(envelope) == ENVELOPE_FIELDS


def test_sentinel_detector_negative_control_catches_a_leaky_dump() -> None:
    layout = _layout()
    port = FakeSecureSumPort(_site_vectors(layout))
    tokens = _sentinel_tokens()

    assert _contains_any(port.leaky_debug_dump(), tokens)
    single_site = SITE_COUNTS[0]["EMAIL"]
    assert _contains_any(repr(single_site[0] / sum(single_site[:2])), tokens)


def test_site_vector_repr_hides_counts() -> None:
    layout = _layout()
    vector = encode_site_counts(layout, SITE_COUNTS[0])

    assert "7331" not in repr(vector)
    assert layout.digest in repr(vector)
    with pytest.raises(FrozenInstanceError):
        vector.values = ()  # type: ignore[misc]


def test_secure_sum_outcome_repr_hides_pooled_vector() -> None:
    outcome = SecureSumOutcome(
        layout_digest=_layout().digest,
        summed_vector=(424242,) * 15,
        contributor_count=5,
        bounds_verified=True,
        privacy_mechanism=P.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
    )
    assert "424242" not in repr(outcome)


# -- fail-closed quorum and bounds --------------------------------------------


def test_below_quorum_fails_closed_without_metrics() -> None:
    layout = _layout()
    port = FakeSecureSumPort(_site_vectors(layout)[:4])

    with pytest.raises(PooledEvaluationError) as caught:
        _pool(layout, port)

    assert caught.value.code is Code.QUORUM_NOT_MET
    assert _contains_any(str(caught.value), _sentinel_tokens()) == []


def test_quorum_exactly_met_releases_metrics_negative_control() -> None:
    layout = _layout()
    report = _pool(layout, FakeSecureSumPort(_site_vectors(layout)), minimum_sites=5)

    assert report.site_count_band is (
        FederatedParticipantCountBand.MINIMUM_TO_UNDER_DOUBLE
    )


def test_site_encoder_rejects_out_of_bound_counts_locally() -> None:
    layout = _layout(max_count_per_entry=1000)

    with pytest.raises(PooledEvaluationError) as caught:
        encode_site_counts(layout, SITE_COUNTS[0])

    assert caught.value.code is Code.CONTRIBUTION_OUT_OF_BOUNDS
    assert "7331" not in str(caught.value)
    assert "EMAIL" not in str(caught.value)


def test_site_encoder_accepts_values_at_the_bound_negative_control() -> None:
    layout = _layout(max_count_per_entry=7)
    vector = encode_site_counts(layout, {"EMAIL": (7, 0, 7)})

    assert vector.values[:3] == (7, 0, 7)
    assert vector.values[3:] == (0,) * 12


def test_port_reported_bound_violation_fails_closed() -> None:
    layout = _layout(max_count_per_entry=10000)
    vectors: list[SiteCountVector | tuple[int, ...]] = list(_site_vectors(layout))
    rogue = (10001,) + (0,) * (layout.vector_length - 1)
    port = FakeSecureSumPort([*vectors, rogue])

    with pytest.raises(PooledEvaluationError) as caught:
        _pool(layout, port)

    assert caught.value.code is Code.CONTRIBUTION_OUT_OF_BOUNDS
    assert "10001" not in str(caught.value)


def test_sum_exceeding_contributor_bound_fails_closed_even_if_port_claims_ok() -> None:
    layout = _layout(max_count_per_entry=10)
    rogue = [(11,) * 15 for _ in range(5)]
    port = FakeSecureSumPort(rogue, claim_verified=True)

    with pytest.raises(PooledEvaluationError) as caught:
        _pool(layout, port, minimum_label_support=2)

    assert caught.value.code is Code.CONTRIBUTION_OUT_OF_BOUNDS


@pytest.mark.parametrize(
    "counts",
    [
        {"EMAIL": (-1, 0, 0)},
        {"EMAIL": (True, 0, 0)},
        {"EMAIL": (1.0, 0, 0)},
        {"EMAIL": (1, 0)},
        {"EMAIL": "123"},
    ],
)
def test_site_encoder_rejects_malformed_counts(counts: dict[str, object]) -> None:
    with pytest.raises(PooledEvaluationError) as caught:
        encode_site_counts(_layout(), counts)  # type: ignore[arg-type]

    assert caught.value.code is Code.CONTRIBUTION_MALFORMED


def test_site_encoder_rejects_undeclared_labels() -> None:
    with pytest.raises(PooledEvaluationError) as caught:
        encode_site_counts(_layout(), {"SSN": (1, 0, 0)})

    assert caught.value.code is Code.UNDECLARED_LABEL
    assert "SSN" not in str(caught.value)


def test_span_builder_rejects_undeclared_labels() -> None:
    with pytest.raises(PooledEvaluationError) as caught:
        build_site_count_vector(
            _layout(), [([EvalSpan(0, 3, "SSN")], [EvalSpan(0, 3, "SSN")])]
        )

    assert caught.value.code is Code.UNDECLARED_LABEL


@pytest.mark.parametrize(
    ("outcome_overrides", "code"),
    [
        ({"summed_vector": (1,) * 14}, Code.SECURE_SUM_MALFORMED),
        ({"summed_vector": (-1,) + (1,) * 14}, Code.SECURE_SUM_MALFORMED),
        ({"summed_vector": (1.0,) * 15}, Code.SECURE_SUM_MALFORMED),
        ({"contributor_count": True}, Code.SECURE_SUM_MALFORMED),
        ({"bounds_verified": "yes"}, Code.CONTRIBUTION_OUT_OF_BOUNDS),
        ({"layout_digest": "sha256:" + "f" * 64}, Code.LAYOUT_MISMATCH),
        ({"privacy_mechanism": P.LAPLACE}, Code.MECHANISM_MISMATCH),
        ({"privacy_mechanism_version": "v2"}, Code.MECHANISM_MISMATCH),
    ],
)
def test_malformed_port_outcomes_fail_closed(
    outcome_overrides: dict[str, object], code: PooledEvaluationErrorCode
) -> None:
    layout = _layout()

    class Port:
        def secure_sum(self, request: SecureSumRequest) -> SecureSumOutcome:
            arguments: dict[str, object] = {
                "layout_digest": request.layout_digest,
                "summed_vector": (1,) * request.vector_length,
                "contributor_count": 5,
                "bounds_verified": True,
                "privacy_mechanism": request.privacy_mechanism,
                "privacy_mechanism_version": request.privacy_mechanism_version,
            }
            arguments.update(outcome_overrides)
            return SecureSumOutcome(**arguments)  # type: ignore[arg-type]

    with pytest.raises(PooledEvaluationError) as caught:
        _pool(layout, Port(), minimum_label_support=2)

    assert caught.value.code is code


def test_port_returning_wrong_type_fails_closed() -> None:
    class Port:
        def secure_sum(self, request: SecureSumRequest) -> object:
            return {"summed_vector": [1] * request.vector_length}

    with pytest.raises(PooledEvaluationError) as caught:
        _pool(_layout(), Port())

    assert caught.value.code is Code.SECURE_SUM_MALFORMED


def test_port_exception_text_is_not_propagated() -> None:
    with pytest.raises(PooledEvaluationError) as caught:
        _pool(_layout(), RaisingPort())

    assert caught.value.code is Code.SECURE_SUM_FAILED
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    message = str(caught.value)
    assert "7331" not in message
    assert "private" not in message
    assert "site-alpha" not in message


def test_request_declares_bounds_quorum_and_mechanism() -> None:
    layout = _layout()
    port = FakeSecureSumPort(_site_vectors(layout))
    _pool(
        layout,
        port,
        minimum_sites=5,
        privacy_mechanism=P.GAUSSIAN,
        privacy_mechanism_version="v3",
    )

    (request,) = port.requests
    assert request.evaluation_ref == EVALUATION_REF
    assert request.layout_digest == layout.digest
    assert request.vector_length == 3 * len(LABELS)
    assert request.entry_upper_bound == 20000
    assert request.minimum_contributors == 5
    assert request.privacy_mechanism is P.GAUSSIAN
    assert request.privacy_mechanism_version == "v3"


@pytest.mark.parametrize(
    "overrides",
    [
        {"evaluation_ref": "holdout-at-site-a"},
        {"minimum_sites": 1},
        {"minimum_sites": True},
        {"minimum_label_support": 1},
        {"confidence_level": 1.0},
        {"confidence_level": 0},
        {"privacy_mechanism": "threshold_only"},
        {"privacy_mechanism_version": "latest"},
        {"layout": object()},
    ],
)
def test_invalid_coordinator_configuration_fails_closed(
    overrides: dict[str, object],
) -> None:
    layout = _layout()
    port = FakeSecureSumPort(_site_vectors(layout))
    arguments: dict[str, object] = {
        "layout": layout,
        "port": port,
        "evaluation_ref": EVALUATION_REF,
        **overrides,
    }
    with pytest.raises(PooledEvaluationError) as caught:
        pool_federated_evaluation(**arguments)  # type: ignore[arg-type]

    assert port.requests == []

    assert caught.value.code is Code.CONFIGURATION_INVALID


# -- small label slice suppression ---------------------------------------------


def test_small_label_slices_are_suppressed() -> None:
    layout = _layout()
    report = _pool(layout, FakeSecureSumPort(_site_vectors(layout)))

    for name in ("precision", "recall"):
        envelope = _metric(report, f"pooled.other.label.city.{name}")
        assert envelope.aggregate_value is None
        assert envelope.participant_count_band is (
            FederatedParticipantCountBand.SUPPRESSED
        )
        assert envelope.uncertainty_method is FederatedUncertaintyMethod.SUPPRESSED
        assert envelope.uncertainty_lower_bound is None


def test_label_slice_at_minimum_support_is_released_negative_control() -> None:
    layout = _layout(max_count_per_entry=10)
    sites = [encode_site_counts(layout, {"DISEASE": (1, 1, 1)}) for _ in range(5)]
    report = _pool(layout, FakeSecureSumPort(sites), minimum_label_support=11)
    below = _metric(report, "pooled.other.label.disease.recall")
    assert below.aggregate_value is None

    report = _pool(layout, FakeSecureSumPort(sites), minimum_label_support=5)
    released = _metric(report, "pooled.other.label.disease.recall")
    assert released.aggregate_value == pytest.approx(0.5)
    assert released.participant_count_band is (
        FederatedParticipantCountBand.DOUBLE_TO_UNDER_FOURFOLD
    )


def test_empty_label_slice_is_suppressed_not_zero() -> None:
    layout = _layout(max_count_per_entry=10)
    sites = [encode_site_counts(layout, {"EMAIL": (2, 0, 0)}) for _ in range(5)]
    report = _pool(layout, FakeSecureSumPort(sites), minimum_label_support=5)

    phone = _metric(report, "pooled.di.label.phone.recall")
    assert phone.aggregate_value is None
    assert phone.participant_count_band is FederatedParticipantCountBand.SUPPRESSED


# -- direct identifiers are separate and non-compensable -----------------------


def test_direct_identifier_labels_come_from_the_label_catalog() -> None:
    layout = _layout()

    assert layout.direct_identifier_labels == ("EMAIL", "PHONE")
    assert layout.other_labels == ("AGE", "DISEASE", "CITY")


def test_custom_direct_identifier_labels_can_be_added() -> None:
    layout = PooledEvaluationLayout(
        labels=("EMAIL", "BADGE_CODE", "DISEASE"),
        max_count_per_entry=10,
        extra_direct_identifier_labels=frozenset({"BADGE_CODE"}),
    )

    assert layout.direct_identifier_labels == ("EMAIL", "BADGE_CODE")


def test_report_partitions_direct_identifier_metrics_separately() -> None:
    layout = _layout()
    report = _pool(layout, FakeSecureSumPort(_site_vectors(layout)))

    di_ids = {envelope.metric_id for envelope in report.direct_identifier_metrics}
    other_ids = {envelope.metric_id for envelope in report.other_label_metrics}
    assert di_ids == {
        "pooled.di.label.email.precision",
        "pooled.di.label.email.recall",
        "pooled.di.label.phone.precision",
        "pooled.di.label.phone.recall",
        "pooled.di.micro.precision",
        "pooled.di.micro.recall",
    }
    assert all(metric_id.startswith("pooled.other.") for metric_id in other_ids)
    assert not di_ids & other_ids
    assert set(report.to_dict()) == REPORT_FIELDS


def test_other_micro_metrics_ignore_direct_identifier_counts() -> None:
    layout = _layout()
    baseline = _pool(layout, FakeSecureSumPort(_site_vectors(layout)))
    degraded_counts = [dict(site, EMAIL=(1, 4000, 9000)) for site in SITE_COUNTS]
    degraded = _pool(
        layout,
        FakeSecureSumPort(
            [encode_site_counts(layout, counts) for counts in degraded_counts]
        ),
    )

    for metric_id in ("pooled.other.micro.precision", "pooled.other.micro.recall"):
        assert (
            _metric(degraded, metric_id).aggregate_value
            == _metric(baseline, metric_id).aggregate_value
        )
    assert (
        _metric(degraded, "pooled.di.micro.recall").aggregate_value
        < _metric(baseline, "pooled.di.micro.recall").aggregate_value
    )


def test_direct_identifier_recall_gate_is_non_compensable() -> None:
    layout = _layout()
    weak_email = [dict(site, EMAIL=(300, 5, 700)) for site in SITE_COUNTS]
    report = _pool(
        layout,
        FakeSecureSumPort([encode_site_counts(layout, c) for c in weak_email]),
    )
    recalls = [
        envelope.aggregate_value
        for envelope in (*report.direct_identifier_metrics, *report.other_label_metrics)
        if envelope.metric_id.endswith(".recall")
        and ".label." in envelope.metric_id
        and envelope.aggregate_value is not None
    ]
    # Negative control: a naive average across all labels would pass 0.6.
    assert sum(recalls) / len(recalls) > 0.6

    gate = report.direct_identifier_recall_gate(minimum_recall=0.6)

    assert gate.passed is False
    assert gate.reason_code == "direct_identifier_recall_below_floor"
    assert gate.failing_metric_ids == ("pooled.di.label.email.recall",)


def test_direct_identifier_recall_gate_passes_when_every_label_passes() -> None:
    layout = _layout()
    report = _pool(layout, FakeSecureSumPort(_site_vectors(layout)))

    gate = report.direct_identifier_recall_gate(minimum_recall=0.6)

    assert gate.passed is True
    assert gate.reason_code is None
    assert gate.failing_metric_ids == ()


def test_direct_identifier_recall_gate_fails_closed_on_suppressed_label() -> None:
    layout = _layout(max_count_per_entry=10)
    sites = [encode_site_counts(layout, {"EMAIL": (2, 0, 0)}) for _ in range(5)]
    report = _pool(layout, FakeSecureSumPort(sites), minimum_label_support=5)

    gate = report.direct_identifier_recall_gate(minimum_recall=0.5)

    assert gate.passed is False
    assert gate.reason_code == "direct_identifier_recall_suppressed"
    assert gate.failing_metric_ids == ("pooled.di.label.phone.recall",)


def test_direct_identifier_recall_gate_fails_closed_without_di_labels() -> None:
    layout = _layout(labels=("AGE", "DISEASE"))
    sites = [encode_site_counts(layout, {"AGE": (9, 1, 1)}) for _ in range(5)]
    report = _pool(layout, FakeSecureSumPort(sites), minimum_label_support=5)

    gate = report.direct_identifier_recall_gate(minimum_recall=0.5)

    assert gate.passed is False
    assert gate.reason_code == "no_direct_identifier_labels"


@pytest.mark.parametrize("value", [True, -0.1, 1.5, float("nan"), "0.9"])
def test_direct_identifier_recall_gate_rejects_invalid_floor(value: Any) -> None:
    layout = _layout()
    report = _pool(layout, FakeSecureSumPort(_site_vectors(layout)))

    with pytest.raises(PooledEvaluationError) as caught:
        report.direct_identifier_recall_gate(minimum_recall=value)

    assert caught.value.code is Code.CONFIGURATION_INVALID


# -- layout and report invariants ----------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"labels": ()},
        {"labels": ("EMAIL", "EMAIL")},
        {"labels": ("EMAIL", "email")},
        {"labels": ("bad label",)},
        {"labels": ("X" * 33,)},
        {"labels": ["EMAIL"]},
        {"max_count_per_entry": 0},
        {"max_count_per_entry": True},
        {"max_count_per_entry": 1.5},
        {"extra_direct_identifier_labels": frozenset({"SSN"})},
        {"extra_direct_identifier_labels": {"EMAIL"}},
    ],
)
def test_invalid_layouts_fail_closed(overrides: dict[str, object]) -> None:
    with pytest.raises(PooledEvaluationError) as caught:
        _layout(**overrides)

    assert caught.value.code is Code.LAYOUT_INVALID


def test_layout_digest_is_stable_and_binds_bound_and_partition() -> None:
    assert _layout().digest == _layout().digest
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", _layout().digest)
    assert _layout().digest != _layout(max_count_per_entry=19999).digest
    assert (
        _layout().digest
        != _layout(extra_direct_identifier_labels=frozenset({"CITY"})).digest
    )


def test_report_json_is_deterministic_and_versioned() -> None:
    layout = _layout()
    first = _pool(layout, FakeSecureSumPort(_site_vectors(layout)))
    second = _pool(layout, FakeSecureSumPort(list(reversed(_site_vectors(layout)))))

    assert first.to_json() == second.to_json()
    assert first.to_json().endswith("\n")
    payload = json.loads(first.to_json())
    assert payload["schema_version"] == POOLED_EVALUATION_SCHEMA_VERSION
    assert payload["evaluation_ref"] == EVALUATION_REF
    assert payload["layout_digest"] == layout.digest
    assert payload["site_count_band"] == "minimum_to_under_double"


def test_report_cannot_be_constructed_directly() -> None:
    with pytest.raises(PooledEvaluationError):
        PooledEvaluationReport(  # type: ignore[call-arg]
            evaluation_ref=EVALUATION_REF,
            layout_digest=_layout().digest,
            site_count_band=FederatedParticipantCountBand.FOURFOLD_OR_MORE,
            minimum_sites=5,
            minimum_label_support=10,
            direct_identifier_metrics=(),
            other_label_metrics=(),
        )


def test_error_messages_are_fixed_per_code() -> None:
    for code in PooledEvaluationErrorCode:
        error = PooledEvaluationError(code)
        assert str(error).startswith(code.value)
        assert error.code is code
