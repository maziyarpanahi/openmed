"""Pooled federated evaluation counts released only as aggregate envelopes.

Each site turns its committed local holdout into a bounded per-label vector of
true-positive, false-positive and false-negative counts. The vectors are summed
by an injected :class:`FederatedSecureSumPort`, so the coordinator only ever
receives the pooled vector and a contributor count. Pooled precision and recall
are then released through :func:`build_federated_metric_envelope`, with
minimum-group suppression applied per label slice.

Direct-identifier labels are reported in their own partition and are never
pooled with other labels. :meth:`PooledEvaluationReport.direct_identifier_recall_gate`
checks every direct-identifier label on its own, so strong results on other
labels cannot compensate for a weak direct-identifier label.

The secure-sum protocol itself, disclosure budgeting and sealed evaluation are
out of scope and are injected or handled elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from enum import Enum
from statistics import NormalDist
from typing import Any, Final, Protocol, runtime_checkable

from .federated_metrics import (
    DEFAULT_FEDERATED_METRIC_MINIMUM_GROUP_SIZE,
    FederatedMetricEnvelope,
    FederatedMetricKind,
    FederatedParticipantCountBand,
    FederatedPrivacyMechanism,
    FederatedUncertaintyMethod,
    _participant_count_band,
    build_federated_metric_envelope,
)
from .federated_status import DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE

POOLED_EVALUATION_SCHEMA_VERSION = "openmed.training.federated_pooled_eval.v1"
POOLED_EVALUATION_LAYOUT_VERSION = "openmed.training.federated_pooled_layout.v1"
DEFAULT_POOLED_EVALUATION_MINIMUM_SITES = DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE
DEFAULT_POOLED_EVALUATION_MINIMUM_LABEL_SUPPORT = (
    DEFAULT_FEDERATED_METRIC_MINIMUM_GROUP_SIZE
)
DEFAULT_POOLED_EVALUATION_CONFIDENCE_LEVEL = 0.95
POOLED_COUNT_FIELDS: Final[tuple[str, str, str]] = (
    "true_positives",
    "false_positives",
    "false_negatives",
)

_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}\Z")
_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MECHANISM_VERSION = re.compile(r"v[1-9][0-9]{0,3}\Z")
_MAX_LABELS: Final = 256
_BUILDER_TOKEN: Final = object()
_DIRECT_IDENTIFIER_PREFIX: Final = "pooled.di"
_OTHER_PREFIX: Final = "pooled.other"


class PooledEvaluationErrorCode(str, Enum):
    """Stable, value-free failure codes for pooled federated evaluation."""

    LAYOUT_INVALID = "layout_invalid"
    CONFIGURATION_INVALID = "configuration_invalid"
    UNDECLARED_LABEL = "undeclared_label"
    CONTRIBUTION_MALFORMED = "contribution_malformed"
    CONTRIBUTION_OUT_OF_BOUNDS = "contribution_out_of_bounds"
    QUORUM_NOT_MET = "quorum_not_met"
    SECURE_SUM_FAILED = "secure_sum_failed"
    SECURE_SUM_MALFORMED = "secure_sum_malformed"
    LAYOUT_MISMATCH = "layout_mismatch"
    MECHANISM_MISMATCH = "mechanism_mismatch"

    def __str__(self) -> str:
        return self.value


_ERROR_MESSAGES: Final[Mapping[PooledEvaluationErrorCode, str]] = {
    PooledEvaluationErrorCode.LAYOUT_INVALID: "pooled evaluation layout is invalid",
    PooledEvaluationErrorCode.CONFIGURATION_INVALID: (
        "pooled evaluation configuration is invalid"
    ),
    PooledEvaluationErrorCode.UNDECLARED_LABEL: (
        "site contribution uses a label outside the declared layout"
    ),
    PooledEvaluationErrorCode.CONTRIBUTION_MALFORMED: (
        "site contribution is not a vector of non-negative integer counts"
    ),
    PooledEvaluationErrorCode.CONTRIBUTION_OUT_OF_BOUNDS: (
        "a contribution exceeds the declared per-entry bound"
    ),
    PooledEvaluationErrorCode.QUORUM_NOT_MET: (
        "fewer sites than the declared minimum contributed"
    ),
    PooledEvaluationErrorCode.SECURE_SUM_FAILED: "secure-sum port failed",
    PooledEvaluationErrorCode.SECURE_SUM_MALFORMED: (
        "secure-sum port returned a malformed outcome"
    ),
    PooledEvaluationErrorCode.LAYOUT_MISMATCH: (
        "secure-sum outcome does not match the declared layout"
    ),
    PooledEvaluationErrorCode.MECHANISM_MISMATCH: (
        "secure-sum outcome does not match the declared privacy mechanism"
    ),
}


class PooledEvaluationError(ValueError):
    """Raised when pooled evaluation cannot proceed safely.

    The message is fixed per code and never contains labels, counts, site
    identities, paths or upstream exception text.

    Attributes:
        code: The stable failure code.
    """

    def __init__(self, code: PooledEvaluationErrorCode) -> None:
        self.code = code
        super().__init__(f"{code.value}: {_ERROR_MESSAGES[code]}")


def _fail(code: PooledEvaluationErrorCode) -> PooledEvaluationError:
    return PooledEvaluationError(code)


@dataclass(frozen=True)
class PooledEvaluationLayout:
    """Closed label layout and per-site bound shared by every participant.

    Direct-identifier labels are derived from the canonical label catalog
    (``openmed.core.labels``). Callers may add custom direct-identifier labels
    through ``extra_direct_identifier_labels`` but can never reclassify a
    catalog direct identifier as an ordinary label.

    Attributes:
        labels: Ordered, unique label names. Each label owns three vector
            entries in :data:`POOLED_COUNT_FIELDS` order.
        max_count_per_entry: Largest count any single site may contribute to
            one vector entry.
        extra_direct_identifier_labels: Additional declared labels to treat as
            direct identifiers.
    """

    labels: tuple[str, ...]
    max_count_per_entry: int
    extra_direct_identifier_labels: frozenset[str] = frozenset()
    _direct_identifiers: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        labels = self.labels
        if type(labels) is not tuple or not 0 < len(labels) <= _MAX_LABELS:
            raise _fail(PooledEvaluationErrorCode.LAYOUT_INVALID)
        lowered: set[str] = set()
        for label in labels:
            if type(label) is not str or _LABEL.fullmatch(label) is None:
                raise _fail(PooledEvaluationErrorCode.LAYOUT_INVALID)
            if label.lower() in lowered:
                raise _fail(PooledEvaluationErrorCode.LAYOUT_INVALID)
            lowered.add(label.lower())
        bound = self.max_count_per_entry
        if type(bound) is not int or bound < 1:
            raise _fail(PooledEvaluationErrorCode.LAYOUT_INVALID)
        extra = self.extra_direct_identifier_labels
        if type(extra) is not frozenset or not extra <= set(labels):
            raise _fail(PooledEvaluationErrorCode.LAYOUT_INVALID)
        object.__setattr__(
            self,
            "_direct_identifiers",
            frozenset(
                label
                for label in labels
                if label in extra or _catalog_direct_identifier(label)
            ),
        )

    @property
    def direct_identifier_labels(self) -> tuple[str, ...]:
        """Return direct-identifier labels in layout order."""

        return tuple(label for label in self.labels if self.is_direct_identifier(label))

    @property
    def other_labels(self) -> tuple[str, ...]:
        """Return labels that are not direct identifiers, in layout order."""

        return tuple(
            label for label in self.labels if not self.is_direct_identifier(label)
        )

    @property
    def vector_length(self) -> int:
        """Return the number of entries in a site count vector."""

        return len(self.labels) * len(POOLED_COUNT_FIELDS)

    @property
    def digest(self) -> str:
        """Return a stable digest binding labels, partition and bound."""

        payload = {
            "direct_identifier_labels": list(self.direct_identifier_labels),
            "count_fields": list(POOLED_COUNT_FIELDS),
            "labels": list(self.labels),
            "layout_version": POOLED_EVALUATION_LAYOUT_VERSION,
            "max_count_per_entry": self.max_count_per_entry,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_direct_identifier(self, label: str) -> bool:
        """Return whether a declared label is reported as a direct identifier.

        Args:
            label: A label from :attr:`labels`.

        Returns:
            ``True`` when the label belongs to the direct-identifier partition.
        """

        return label in self._direct_identifiers


def _catalog_direct_identifier(label: str) -> bool:
    from openmed.core.labels import (
        DIRECT_IDENTIFIER,
        is_recognized_label,
        policy_label_for,
    )

    return is_recognized_label(label) and policy_label_for(label) == (DIRECT_IDENTIFIER)


@dataclass(frozen=True)
class SiteCountVector:
    """A bounded, site-local count vector ready for secure-sum submission.

    Build instances with :func:`encode_site_counts` or
    :func:`build_site_count_vector`. The counts are excluded from ``repr`` and
    are meant to be handed only to the site's secure-sum client, never to the
    coordinator.

    Attributes:
        layout_digest: Digest of the layout the vector was encoded against.
        values: Flattened counts in layout order.
    """

    layout_digest: str
    values: tuple[int, ...] = field(repr=False)
    _builder_token: InitVar[object | None] = None

    def __post_init__(self, _builder_token: object | None) -> None:
        if _builder_token is not _BUILDER_TOKEN:
            raise _fail(PooledEvaluationErrorCode.CONTRIBUTION_MALFORMED)


def encode_site_counts(
    layout: PooledEvaluationLayout,
    counts: Mapping[str, Sequence[int]],
) -> SiteCountVector:
    """Encode per-label counts into a bounded site vector.

    Runs at the site, before anything leaves it. Labels missing from
    ``counts`` contribute zeros.

    Args:
        layout: The shared evaluation layout.
        counts: Mapping from declared label to a ``(tp, fp, fn)`` triple.

    Returns:
        A :class:`SiteCountVector` bound to ``layout``.

    Raises:
        PooledEvaluationError: ``undeclared_label`` for labels outside the
            layout, ``contribution_malformed`` for non-integer, negative or
            mis-shaped counts and ``contribution_out_of_bounds`` when any
            count exceeds ``layout.max_count_per_entry``.
    """

    if not isinstance(layout, PooledEvaluationLayout):
        raise _fail(PooledEvaluationErrorCode.CONFIGURATION_INVALID)
    if not isinstance(counts, Mapping):
        raise _fail(PooledEvaluationErrorCode.CONTRIBUTION_MALFORMED)
    declared = set(layout.labels)
    if any(label not in declared for label in counts):
        raise _fail(PooledEvaluationErrorCode.UNDECLARED_LABEL)
    values: list[int] = []
    out_of_bounds = False
    for label in layout.labels:
        triple = counts.get(label, (0, 0, 0))
        if isinstance(triple, (str, bytes)) or not isinstance(triple, Sequence):
            raise _fail(PooledEvaluationErrorCode.CONTRIBUTION_MALFORMED)
        if len(triple) != len(POOLED_COUNT_FIELDS):
            raise _fail(PooledEvaluationErrorCode.CONTRIBUTION_MALFORMED)
        for value in triple:
            if type(value) is not int or value < 0:
                raise _fail(PooledEvaluationErrorCode.CONTRIBUTION_MALFORMED)
            out_of_bounds = out_of_bounds or value > layout.max_count_per_entry
            values.append(value)
    if out_of_bounds:
        raise _fail(PooledEvaluationErrorCode.CONTRIBUTION_OUT_OF_BOUNDS)
    return SiteCountVector(
        layout_digest=layout.digest,
        values=tuple(values),
        _builder_token=_BUILDER_TOKEN,
    )


def build_site_count_vector(
    layout: PooledEvaluationLayout,
    documents: Iterable[tuple[Iterable[Any], Iterable[Any]]],
) -> SiteCountVector:
    """Count strict exact-span matches on a site's local holdout.

    Each document is a ``(gold_spans, predicted_spans)`` pair. Counts use the
    same label-aware exact matching as
    :func:`openmed.eval.metrics.compute_exact_span_f1`, applied per label and
    per document, then summed across documents.

    Args:
        layout: The shared evaluation layout.
        documents: Iterable of gold and predicted span collections.

    Returns:
        A bounded :class:`SiteCountVector`.

    Raises:
        PooledEvaluationError: ``undeclared_label`` when a span label is not in
            the layout, plus every failure raised by :func:`encode_site_counts`.
    """

    from openmed.eval.metrics import compute_exact_span_f1, normalize_eval_spans

    if not isinstance(layout, PooledEvaluationLayout):
        raise _fail(PooledEvaluationErrorCode.CONFIGURATION_INVALID)
    declared = set(layout.labels)
    totals = {label: [0, 0, 0] for label in layout.labels}
    for document in documents:
        try:
            gold_spans, predicted_spans = document
            gold = normalize_eval_spans(gold_spans)
            predicted = normalize_eval_spans(predicted_spans)
        except (TypeError, ValueError):
            raise _fail(PooledEvaluationErrorCode.CONTRIBUTION_MALFORMED) from None
        if any(span.label not in declared for span in (*gold, *predicted)):
            raise _fail(PooledEvaluationErrorCode.UNDECLARED_LABEL)
        for label in layout.labels:
            metrics = compute_exact_span_f1(
                [span for span in gold if span.label == label],
                [span for span in predicted if span.label == label],
            )
            total = totals[label]
            total[0] += metrics.true_positives
            total[1] += metrics.false_positives
            total[2] += metrics.false_negatives
    return encode_site_counts(
        layout, {label: tuple(value) for label, value in totals.items()}
    )


@dataclass(frozen=True)
class SecureSumRequest:
    """What the coordinator asks of the secure-sum port.

    Attributes:
        evaluation_ref: Opaque ``sha256:`` reference for this evaluation run.
        layout_digest: Digest of the shared layout.
        vector_length: Expected number of entries in every vector.
        entry_upper_bound: Largest count a single site may contribute per
            entry; the port must verify every contribution against it.
        minimum_contributors: Quorum the port should enforce before unmasking.
        privacy_mechanism: Mechanism the port must apply to the sum.
        privacy_mechanism_version: Version of that mechanism.
    """

    evaluation_ref: str
    layout_digest: str
    vector_length: int
    entry_upper_bound: int
    minimum_contributors: int
    privacy_mechanism: FederatedPrivacyMechanism
    privacy_mechanism_version: str


@dataclass(frozen=True)
class SecureSumOutcome:
    """Aggregate result returned by a secure-sum port.

    Attributes:
        layout_digest: Digest of the layout the sum was computed for.
        summed_vector: Entry-wise sum of every contribution. Excluded from
            ``repr``.
        contributor_count: Number of sites whose vectors were summed.
        bounds_verified: ``True`` only when the port verified every individual
            contribution against ``entry_upper_bound``.
        privacy_mechanism: Mechanism the port applied.
        privacy_mechanism_version: Version of that mechanism.
    """

    layout_digest: str
    summed_vector: tuple[int, ...] = field(repr=False)
    contributor_count: int = 0
    bounds_verified: bool = False
    privacy_mechanism: FederatedPrivacyMechanism = (
        FederatedPrivacyMechanism.THRESHOLD_ONLY
    )
    privacy_mechanism_version: str = "v1"


@runtime_checkable
class FederatedSecureSumPort(Protocol):
    """Injected secure-sum transport.

    Implementations collect site vectors out of band and release only their
    entry-wise sum. They must never hand individual vectors to the caller.
    """

    def secure_sum(self, request: SecureSumRequest) -> SecureSumOutcome:
        """Return the aggregate sum for ``request``."""


@dataclass(frozen=True)
class DirectIdentifierRecallGate:
    """Non-compensable verdict over every direct-identifier recall.

    Attributes:
        passed: ``True`` only when every direct-identifier label recall was
            released and meets the floor.
        reason_code: ``None`` on pass, otherwise one of
            ``no_direct_identifier_labels``,
            ``direct_identifier_recall_suppressed`` or
            ``direct_identifier_recall_below_floor``.
        failing_metric_ids: Metric ids of the labels that failed.
    """

    passed: bool
    reason_code: str | None
    failing_metric_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PooledEvaluationReport:
    """Coordinator-visible result of :func:`pool_federated_evaluation`.

    It holds only released or suppressed aggregate envelopes and coarse bands;
    there is no field for site identities, site vectors or per-site metrics.

    Attributes:
        evaluation_ref: Opaque reference for the evaluation run.
        layout_digest: Digest of the shared layout.
        site_count_band: Coarse contributor band relative to ``minimum_sites``.
        minimum_sites: Quorum that was enforced.
        minimum_label_support: Minimum pooled support for a released slice.
        direct_identifier_metrics: Envelopes for direct-identifier labels and
            their partition micro aggregate.
        other_label_metrics: Envelopes for every other label and their
            partition micro aggregate.
        schema_version: Report schema version.
    """

    evaluation_ref: str
    layout_digest: str
    site_count_band: FederatedParticipantCountBand
    minimum_sites: int
    minimum_label_support: int
    direct_identifier_metrics: tuple[FederatedMetricEnvelope, ...]
    other_label_metrics: tuple[FederatedMetricEnvelope, ...]
    schema_version: str = POOLED_EVALUATION_SCHEMA_VERSION
    _builder_token: InitVar[object | None] = None

    def __post_init__(self, _builder_token: object | None) -> None:
        if _builder_token is not _BUILDER_TOKEN:
            raise _fail(PooledEvaluationErrorCode.CONFIGURATION_INVALID)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping of aggregate fields only."""

        return {
            "direct_identifier_metrics": [
                envelope.to_dict() for envelope in self.direct_identifier_metrics
            ],
            "evaluation_ref": self.evaluation_ref,
            "layout_digest": self.layout_digest,
            "minimum_label_support": self.minimum_label_support,
            "minimum_sites": self.minimum_sites,
            "other_label_metrics": [
                envelope.to_dict() for envelope in self.other_label_metrics
            ],
            "schema_version": self.schema_version,
            "site_count_band": self.site_count_band.value,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with a trailing newline."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def direct_identifier_recall_gate(
        self, *, minimum_recall: float
    ) -> DirectIdentifierRecallGate:
        """Check every direct-identifier label recall against a floor.

        Each label is checked on its own. Partition micro aggregates and
        other-label metrics are ignored, so they cannot offset a weak label.
        A suppressed direct-identifier recall fails closed.

        Args:
            minimum_recall: Required recall for every direct-identifier label.

        Returns:
            A :class:`DirectIdentifierRecallGate`.

        Raises:
            PooledEvaluationError: ``configuration_invalid`` when the floor is
                not a finite number between zero and one.
        """

        if (
            type(minimum_recall) not in (int, float)
            or not math.isfinite(minimum_recall)
            or not 0 <= minimum_recall <= 1
        ):
            raise _fail(PooledEvaluationErrorCode.CONFIGURATION_INVALID)
        label_recalls = [
            envelope
            for envelope in self.direct_identifier_metrics
            if envelope.metric_id.startswith(f"{_DIRECT_IDENTIFIER_PREFIX}.label.")
            and envelope.metric_id.endswith(".recall")
        ]
        if not label_recalls:
            return DirectIdentifierRecallGate(
                passed=False, reason_code="no_direct_identifier_labels"
            )
        suppressed = tuple(
            envelope.metric_id
            for envelope in label_recalls
            if envelope.aggregate_value is None
        )
        below = tuple(
            envelope.metric_id
            for envelope in label_recalls
            if envelope.aggregate_value is not None
            and envelope.aggregate_value < minimum_recall
        )
        if suppressed:
            return DirectIdentifierRecallGate(
                passed=False,
                reason_code="direct_identifier_recall_suppressed",
                failing_metric_ids=suppressed + below,
            )
        if below:
            return DirectIdentifierRecallGate(
                passed=False,
                reason_code="direct_identifier_recall_below_floor",
                failing_metric_ids=below,
            )
        return DirectIdentifierRecallGate(passed=True, reason_code=None)


def pool_federated_evaluation(
    *,
    layout: PooledEvaluationLayout,
    port: FederatedSecureSumPort,
    evaluation_ref: str,
    minimum_sites: int = DEFAULT_POOLED_EVALUATION_MINIMUM_SITES,
    minimum_label_support: int = DEFAULT_POOLED_EVALUATION_MINIMUM_LABEL_SUPPORT,
    confidence_level: float = DEFAULT_POOLED_EVALUATION_CONFIDENCE_LEVEL,
    privacy_mechanism: FederatedPrivacyMechanism = (
        FederatedPrivacyMechanism.THRESHOLD_ONLY
    ),
    privacy_mechanism_version: str = "v1",
) -> PooledEvaluationReport:
    """Pool site counts through a secure-sum port and release envelopes.

    The coordinator never receives a per-site vector. It checks quorum before
    reading the pooled vector, requires the port to attest that every
    contribution respected the declared bound and rejects any pooled entry
    that exceeds ``contributor_count * max_count_per_entry``.

    Precision uses ``tp / (tp + fp)`` and recall uses ``tp / (tp + fn)`` on
    pooled counts, each with a Wilson score interval. A slice whose pooled
    support is below ``minimum_label_support`` is suppressed by
    :func:`build_federated_metric_envelope`; the envelope's participant band
    then describes pooled support, not sites.

    Args:
        layout: The shared evaluation layout.
        port: Injected secure-sum port.
        evaluation_ref: Opaque ``sha256:`` reference for this run.
        minimum_sites: Minimum number of contributing sites.
        minimum_label_support: Minimum pooled support per released slice.
        confidence_level: Wilson interval confidence level.
        privacy_mechanism: Mechanism the port applies and the envelopes declare.
        privacy_mechanism_version: Version of that mechanism.

    Returns:
        A :class:`PooledEvaluationReport`.

    Raises:
        PooledEvaluationError: With a stable code for invalid configuration,
            port failures, malformed or mismatched outcomes, unmet quorum or
            out-of-bound contributions.
    """

    _validate_configuration(
        layout=layout,
        evaluation_ref=evaluation_ref,
        minimum_sites=minimum_sites,
        minimum_label_support=minimum_label_support,
        confidence_level=confidence_level,
        privacy_mechanism=privacy_mechanism,
        privacy_mechanism_version=privacy_mechanism_version,
    )
    request = SecureSumRequest(
        evaluation_ref=evaluation_ref,
        layout_digest=layout.digest,
        vector_length=layout.vector_length,
        entry_upper_bound=layout.max_count_per_entry,
        minimum_contributors=minimum_sites,
        privacy_mechanism=privacy_mechanism,
        privacy_mechanism_version=privacy_mechanism_version,
    )
    try:
        outcome = port.secure_sum(request)
    except PooledEvaluationError:
        raise
    except Exception:
        raise _fail(PooledEvaluationErrorCode.SECURE_SUM_FAILED) from None
    pooled = _validated_pooled_vector(outcome, request)
    contributor_count = outcome.contributor_count

    def release(prefix: str, labels: Sequence[str]) -> list[FederatedMetricEnvelope]:
        envelopes: list[FederatedMetricEnvelope] = []
        micro = [0, 0, 0]
        for label in labels:
            index = layout.labels.index(label) * len(POOLED_COUNT_FIELDS)
            tp, fp, fn = pooled[index : index + len(POOLED_COUNT_FIELDS)]
            micro = [micro[0] + tp, micro[1] + fp, micro[2] + fn]
            envelopes.extend(
                _precision_recall(
                    f"{prefix}.label.{label.lower()}",
                    tp,
                    fp,
                    fn,
                    minimum_label_support=minimum_label_support,
                    confidence_level=confidence_level,
                    privacy_mechanism=privacy_mechanism,
                    privacy_mechanism_version=privacy_mechanism_version,
                )
            )
        if labels:
            envelopes.extend(
                _precision_recall(
                    f"{prefix}.micro",
                    *micro,
                    minimum_label_support=minimum_label_support,
                    confidence_level=confidence_level,
                    privacy_mechanism=privacy_mechanism,
                    privacy_mechanism_version=privacy_mechanism_version,
                )
            )
        return envelopes

    return PooledEvaluationReport(
        evaluation_ref=evaluation_ref,
        layout_digest=layout.digest,
        site_count_band=_participant_count_band(contributor_count, minimum_sites),
        minimum_sites=minimum_sites,
        minimum_label_support=minimum_label_support,
        direct_identifier_metrics=tuple(
            release(_DIRECT_IDENTIFIER_PREFIX, layout.direct_identifier_labels)
        ),
        other_label_metrics=tuple(release(_OTHER_PREFIX, layout.other_labels)),
        _builder_token=_BUILDER_TOKEN,
    )


def _validate_configuration(
    *,
    layout: object,
    evaluation_ref: object,
    minimum_sites: object,
    minimum_label_support: object,
    confidence_level: object,
    privacy_mechanism: object,
    privacy_mechanism_version: object,
) -> None:
    invalid = (
        not isinstance(layout, PooledEvaluationLayout)
        or type(evaluation_ref) is not str
        or _SHA256_DIGEST.fullmatch(evaluation_ref) is None
        or type(minimum_sites) is not int
        or minimum_sites < 2
        or type(minimum_label_support) is not int
        or minimum_label_support < 2
        or type(confidence_level) is not float
        or not 0.0 < confidence_level < 1.0
        or not isinstance(privacy_mechanism, FederatedPrivacyMechanism)
        or type(privacy_mechanism_version) is not str
        or _MECHANISM_VERSION.fullmatch(privacy_mechanism_version) is None
    )
    if invalid:
        raise _fail(PooledEvaluationErrorCode.CONFIGURATION_INVALID)


def _validated_pooled_vector(
    outcome: object, request: SecureSumRequest
) -> tuple[int, ...]:
    if not isinstance(outcome, SecureSumOutcome):
        raise _fail(PooledEvaluationErrorCode.SECURE_SUM_MALFORMED)
    if outcome.layout_digest != request.layout_digest:
        raise _fail(PooledEvaluationErrorCode.LAYOUT_MISMATCH)
    if (
        outcome.privacy_mechanism is not request.privacy_mechanism
        or outcome.privacy_mechanism_version != request.privacy_mechanism_version
    ):
        raise _fail(PooledEvaluationErrorCode.MECHANISM_MISMATCH)
    contributors = outcome.contributor_count
    if type(contributors) is not int or contributors < 0:
        raise _fail(PooledEvaluationErrorCode.SECURE_SUM_MALFORMED)
    if contributors < request.minimum_contributors:
        raise _fail(PooledEvaluationErrorCode.QUORUM_NOT_MET)
    if outcome.bounds_verified is not True:
        raise _fail(PooledEvaluationErrorCode.CONTRIBUTION_OUT_OF_BOUNDS)
    vector = outcome.summed_vector
    if (
        type(vector) is not tuple
        or len(vector) != request.vector_length
        or any(type(value) is not int or value < 0 for value in vector)
    ):
        raise _fail(PooledEvaluationErrorCode.SECURE_SUM_MALFORMED)
    ceiling = contributors * request.entry_upper_bound
    if any(value > ceiling for value in vector):
        raise _fail(PooledEvaluationErrorCode.CONTRIBUTION_OUT_OF_BOUNDS)
    return vector


def _precision_recall(
    metric_prefix: str,
    tp: int,
    fp: int,
    fn: int,
    *,
    minimum_label_support: int,
    confidence_level: float,
    privacy_mechanism: FederatedPrivacyMechanism,
    privacy_mechanism_version: str,
) -> tuple[FederatedMetricEnvelope, FederatedMetricEnvelope]:
    common = {
        "minimum_label_support": minimum_label_support,
        "confidence_level": confidence_level,
        "privacy_mechanism": privacy_mechanism,
        "privacy_mechanism_version": privacy_mechanism_version,
    }
    return (
        _rate_envelope(f"{metric_prefix}.precision", tp, tp + fp, **common),
        _rate_envelope(f"{metric_prefix}.recall", tp, tp + fn, **common),
    )


def _rate_envelope(
    metric_id: str,
    successes: int,
    trials: int,
    *,
    minimum_label_support: int,
    confidence_level: float,
    privacy_mechanism: FederatedPrivacyMechanism,
    privacy_mechanism_version: str,
) -> FederatedMetricEnvelope:
    common: dict[str, Any] = {
        "metric_id": metric_id,
        "metric_kind": FederatedMetricKind.RATE,
        "clipping_lower_bound": 0.0,
        "clipping_upper_bound": 1.0,
        "privacy_mechanism": privacy_mechanism,
        "privacy_mechanism_version": privacy_mechanism_version,
        "participant_count": trials,
        "minimum_group_size": minimum_label_support,
    }
    if trials < minimum_label_support:
        # The builder discards the value; a neutral placeholder avoids
        # computing a rate for a slice that will never be released.
        return build_federated_metric_envelope(aggregate_value=0.0, **common)
    value = successes / trials
    lower, upper = _wilson_interval(successes, trials, confidence_level)
    return build_federated_metric_envelope(
        aggregate_value=value,
        uncertainty_method=FederatedUncertaintyMethod.CONFIDENCE_INTERVAL,
        uncertainty_lower_bound=min(lower, value),
        uncertainty_upper_bound=max(upper, value),
        confidence_level=confidence_level,
        **common,
    )


def _wilson_interval(
    successes: int, trials: int, confidence_level: float
) -> tuple[float, float]:
    # Same Wilson score interval as openmed.eval.flaky; kept local so the
    # training package does not import evaluation internals.
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    center = (phat + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * trials)) / trials)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
