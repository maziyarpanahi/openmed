"""No-PHI production drift monitor feeding the retrain-trigger boundary.

The monitor consumes only privacy-safe aggregates produced on the host --
per-label prediction counts, calibrated-score histograms, input length and
script histograms, and input-characteristic hashes/offsets. Raw production
text, detected-span surface strings, and any free-text field are structurally
rejected before an aggregate window is admitted, so no PHI can enter the
monitor's state, its serialized drift record, or the drift signal a retrain
trigger consumes.

Divergence is a deterministic Population Stability Index (PSI) computed over a
committed reference window, so identical committed aggregates always re-derive
identical divergence offline. The monitor only emits a drift signal up to the
trigger boundary (``DriftTriggerSignal``); it deliberately does not implement
the retrain-decision logic itself, which lives in the flywheel trigger.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openmed.core.offline import is_local_only, network_blocked_if_offline

DRIFT_MONITOR_SCHEMA_VERSION = "openmed.drift_monitor.v1"
DRIFT_WINDOW_SCHEMA_VERSION = "openmed.drift_monitor.window.v1"
DRIFT_TRIGGER_SCHEMA_VERSION = "openmed.drift_monitor.trigger.v1"

VERDICT_STABLE = "stable"
VERDICT_WARNING = "drift_warning"
VERDICT_DRIFT = "drift"

LABEL_RATE_FAMILY = "label_rate"
SCORE_HISTOGRAM_FAMILY = "score_histogram"
LENGTH_HISTOGRAM_FAMILY = "length_histogram"
SCRIPT_HISTOGRAM_FAMILY = "script_histogram"

DRIFT_FAMILIES = (
    LABEL_RATE_FAMILY,
    SCORE_HISTOGRAM_FAMILY,
    LENGTH_HISTOGRAM_FAMILY,
    SCRIPT_HISTOGRAM_FAMILY,
)

# Reuse the calibrate.py 10-bin calibrated-score histogram convention so the
# monitor consumes the same aggregate shape the calibration reports emit.
DEFAULT_SCORE_BINS = 10
DEFAULT_WARNING_THRESHOLD = 0.1
DEFAULT_DRIFT_THRESHOLD = 0.25

_PSI_EPSILON = 1e-6

DEFAULT_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "gates" / "drift_reference.json"
)

# Keys whose presence anywhere in an ingested aggregate payload signals that raw
# production text or detected-span surface strings are being smuggled into the
# no-PHI monitor. Their presence is a hard failure, never a silent drop.
RAW_TEXT_FORBIDDEN_KEYS = frozenset(
    {
        "text",
        "texts",
        "raw_text",
        "raw_texts",
        "input_text",
        "source_text",
        "content",
        "contents",
        "snippet",
        "snippets",
        "span",
        "spans",
        "span_text",
        "surface",
        "surface_form",
        "word",
        "words",
        "token",
        "tokens",
        "sentence",
        "sentences",
        "phrase",
        "phrases",
        "document",
        "documents",
        "note",
        "notes",
        "message",
        "messages",
        "example",
        "examples",
        "sample_text",
        "entity_text",
        "detected_text",
        "prediction_text",
        "payload",
        "records",
        "rows",
    }
)

# Category / identifier tokens (label names, script names, length buckets,
# window ids, hashes, ISO timestamps, schema versions) must be short, single
# tokens with no whitespace or sentence punctuation. A raw clinical sentence
# ("Patient Jane Doe, MRN ...") always fails this pattern, which is the
# structural guarantee that free text cannot ride inside a "safe" value.
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,127}\Z")


class DriftPrivacyError(ValueError):
    """Raised when a raw-text/PHI field enters the no-PHI drift monitor."""


class DriftInputError(ValueError):
    """Raised when an aggregate window is malformed or empty."""


def assert_no_raw_text(payload: Any, *, where: str = "drift record") -> None:
    """Assert a payload carries only aggregates/hashes/offsets, never raw text.

    Structurally rejects forbidden text-bearing keys and any string value that
    is not a short single token (label/script/bucket name, hash, timestamp, or
    schema version). Numbers, booleans, and null are always allowed.
    """

    _walk_no_raw_text(payload, where=where, path="")


def _walk_no_raw_text(value: Any, *, where: str, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in RAW_TEXT_FORBIDDEN_KEYS:
                raise DriftPrivacyError(
                    f"{where} contains a forbidden raw-text field "
                    f"{_join_path(path, key_text)!r}"
                )
            _walk_no_raw_text(child, where=where, path=_join_path(path, key_text))
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_no_raw_text(child, where=where, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        if not _SAFE_TOKEN.fullmatch(value):
            raise DriftPrivacyError(
                f"{where} field {path or '<root>'!r} holds a non-aggregate "
                "string value; only counts, hashes, and safe identifiers are "
                "permitted"
            )
        return
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        return
    raise DriftPrivacyError(
        f"{where} field {path or '<root>'!r} holds an unsupported value type "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True)
class DriftAggregateWindow:
    """A privacy-safe aggregate window over one production observation period.

    Every field is an aggregate count, histogram bucket, hash, or offset. No raw
    text, detected-span surface string, or per-record value is ever stored.
    """

    window_id: str
    sample_count: int
    label_counts: Mapping[str, int]
    score_histogram: tuple[float, ...]
    length_histogram: Mapping[str, int]
    script_histogram: Mapping[str, int]
    feature_hashes: Mapping[str, str] = field(default_factory=dict)
    generated_at: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DriftAggregateWindow":
        """Coerce an ingested aggregate mapping, rejecting any raw-text field."""

        if not isinstance(payload, Mapping):
            raise DriftInputError("drift aggregate window must be a mapping")
        assert_no_raw_text(payload, where="drift aggregate window")

        window_id = _safe_identifier(payload.get("window_id"), "window_id")
        label_counts = _coerce_counts(payload.get("label_counts"), "label_counts")
        length_histogram = _coerce_counts(
            payload.get("length_histogram"), "length_histogram"
        )
        script_histogram = _coerce_counts(
            payload.get("script_histogram"), "script_histogram"
        )
        score_histogram = _coerce_score_histogram(payload.get("score_histogram"))
        feature_hashes = _coerce_hashes(payload.get("feature_hashes"))

        declared = payload.get("sample_count")
        derived = sum(label_counts.values())
        if declared is None:
            sample_count = derived
        else:
            sample_count = _coerce_count(declared, "sample_count")
        if sample_count < 0:
            raise DriftInputError("sample_count must be non-negative")

        generated_at = payload.get("generated_at")
        if generated_at is not None:
            generated_at = _safe_identifier(generated_at, "generated_at")

        return cls(
            window_id=window_id,
            sample_count=sample_count,
            label_counts=dict(sorted(label_counts.items())),
            score_histogram=score_histogram,
            length_histogram=dict(sorted(length_histogram.items())),
            script_histogram=dict(sorted(script_histogram.items())),
            feature_hashes=dict(sorted(feature_hashes.items())),
            generated_at=generated_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-free JSON-ready aggregate payload."""

        payload = {
            "schema_version": DRIFT_WINDOW_SCHEMA_VERSION,
            "window_id": self.window_id,
            "sample_count": self.sample_count,
            "label_counts": dict(sorted(self.label_counts.items())),
            "score_histogram": list(self.score_histogram),
            "length_histogram": dict(sorted(self.length_histogram.items())),
            "script_histogram": dict(sorted(self.script_histogram.items())),
            "feature_hashes": dict(sorted(self.feature_hashes.items())),
            "generated_at": self.generated_at,
        }
        assert_no_raw_text(payload, where="drift aggregate window")
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the aggregate window to deterministic JSON."""

        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True
        )


@dataclass(frozen=True)
class DriftFamilyDivergence:
    """PSI divergence and dominant-bucket evidence for one aggregate family."""

    family: str
    divergence: float
    verdict: str
    dominant_bucket: str | None
    bucket_contributions: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-free JSON-ready family payload."""

        return {
            "family": self.family,
            "divergence": self.divergence,
            "verdict": self.verdict,
            "dominant_bucket": self.dominant_bucket,
            "bucket_contributions": [dict(row) for row in self.bucket_contributions],
        }


@dataclass(frozen=True)
class DriftTriggerSignal:
    """Drift signal at the retrain-trigger boundary (trigger-consumable).

    This is the interface a downstream retrain trigger reads directly. It
    carries the per-family divergence map and the dominant drifting label the
    trigger scorer needs, and nothing else. The monitor does not decide whether
    to retrain -- that is the trigger's responsibility.
    """

    schema_version: str
    drift_detected: bool
    verdict: str
    max_divergence: float
    threshold: float
    warning_threshold: float
    per_family_divergence: Mapping[str, float]
    dominant_family: str | None
    dominant_drifting_label: str | None
    reference_window_id: str
    observation_window_id: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-free JSON-ready trigger signal."""

        payload = {
            "schema_version": self.schema_version,
            "drift_detected": self.drift_detected,
            "verdict": self.verdict,
            "max_divergence": self.max_divergence,
            "threshold": self.threshold,
            "warning_threshold": self.warning_threshold,
            "per_family_divergence": {
                family: float(value)
                for family, value in sorted(self.per_family_divergence.items())
            },
            "dominant_family": self.dominant_family,
            "dominant_drifting_label": self.dominant_drifting_label,
            "reference_window_id": self.reference_window_id,
            "observation_window_id": self.observation_window_id,
            "generated_at": self.generated_at,
        }
        assert_no_raw_text(payload, where="drift trigger signal")
        return payload


@dataclass(frozen=True)
class DriftReport:
    """Full no-PHI drift record comparing an observation window to a reference."""

    reference_window_id: str
    observation_window_id: str
    families: tuple[DriftFamilyDivergence, ...]
    threshold: float
    warning_threshold: float
    reference_sample_count: int
    observation_sample_count: int
    generated_at: str
    schema_version: str = DRIFT_MONITOR_SCHEMA_VERSION

    @property
    def per_family_divergence(self) -> dict[str, float]:
        """Return the family -> divergence map in a stable order."""

        return {family.family: family.divergence for family in self.families}

    @property
    def max_divergence(self) -> float:
        """Return the largest per-family divergence."""

        return max((family.divergence for family in self.families), default=0.0)

    @property
    def dominant_family(self) -> str | None:
        """Return the family carrying the largest divergence, if any drifts."""

        drifting = [family for family in self.families if family.divergence > 0.0]
        if not drifting:
            return None
        return max(
            drifting,
            key=lambda family: (family.divergence, family.family),
        ).family

    @property
    def verdict(self) -> str:
        """Return the overall drift verdict from the max divergence."""

        return _verdict_for(
            self.max_divergence,
            threshold=self.threshold,
            warning_threshold=self.warning_threshold,
        )

    @property
    def drift_detected(self) -> bool:
        """Return whether the observation crossed the drift threshold."""

        return self.verdict == VERDICT_DRIFT

    @property
    def dominant_drifting_label(self) -> str | None:
        """Return the label whose prediction-rate shift dominates label drift."""

        for family in self.families:
            if family.family != LABEL_RATE_FAMILY:
                continue
            if family.divergence <= 0.0:
                return None
            return family.dominant_bucket
        return None

    def to_trigger_signal(self) -> DriftTriggerSignal:
        """Project the drift record onto the retrain-trigger boundary shape."""

        return DriftTriggerSignal(
            schema_version=DRIFT_TRIGGER_SCHEMA_VERSION,
            drift_detected=self.drift_detected,
            verdict=self.verdict,
            max_divergence=self.max_divergence,
            threshold=self.threshold,
            warning_threshold=self.warning_threshold,
            per_family_divergence=self.per_family_divergence,
            dominant_family=self.dominant_family,
            dominant_drifting_label=self.dominant_drifting_label,
            reference_window_id=self.reference_window_id,
            observation_window_id=self.observation_window_id,
            generated_at=self.generated_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-free JSON-ready drift record."""

        payload = {
            "schema_version": self.schema_version,
            "reference_window_id": self.reference_window_id,
            "observation_window_id": self.observation_window_id,
            "reference_sample_count": self.reference_sample_count,
            "observation_sample_count": self.observation_sample_count,
            "threshold": self.threshold,
            "warning_threshold": self.warning_threshold,
            "generated_at": self.generated_at,
            "verdict": self.verdict,
            "drift_detected": self.drift_detected,
            "max_divergence": self.max_divergence,
            "dominant_family": self.dominant_family,
            "dominant_drifting_label": self.dominant_drifting_label,
            "families": [family.to_dict() for family in self.families],
            "trigger_signal": self.to_trigger_signal().to_dict(),
        }
        assert_no_raw_text(payload, where="drift report")
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the drift record to deterministic JSON."""

        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the deterministic drift record JSON and return its path."""

        output_path = Path(path)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path


def compute_drift_report(
    reference: DriftAggregateWindow | Mapping[str, Any],
    observation: DriftAggregateWindow | Mapping[str, Any],
    *,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    generated_at: str | None = None,
    local_only: bool | None = None,
) -> DriftReport:
    """Compute a no-PHI drift record from two privacy-safe aggregate windows.

    The computation is pure arithmetic over committed aggregates, so identical
    inputs always yield identical divergence. It runs fully on-device: it is
    wrapped in the offline network guard so that under ``OPENMED_OFFLINE`` (or an
    explicit ``local_only``) any accidental outbound call fails instead of
    silently phoning home.
    """

    if threshold <= 0.0 or warning_threshold <= 0.0:
        raise DriftInputError("thresholds must be positive")
    if warning_threshold > threshold:
        raise DriftInputError("warning_threshold must not exceed threshold")

    offline = is_local_only() if local_only is None else bool(local_only)
    with network_blocked_if_offline(local_only=offline):
        reference_window = _as_window(reference)
        observation_window = _as_window(observation)

        families = tuple(
            _family_divergence(
                family,
                reference_window,
                observation_window,
                threshold=threshold,
                warning_threshold=warning_threshold,
            )
            for family in DRIFT_FAMILIES
        )

    return DriftReport(
        reference_window_id=reference_window.window_id,
        observation_window_id=observation_window.window_id,
        families=families,
        threshold=threshold,
        warning_threshold=warning_threshold,
        reference_sample_count=reference_window.sample_count,
        observation_sample_count=observation_window.sample_count,
        generated_at=generated_at or _utc_now(),
    )


def load_drift_window(
    path: str | Path,
    *,
    local_only: bool | None = None,
) -> DriftAggregateWindow:
    """Load and validate a local aggregate window, honoring offline mode."""

    offline = is_local_only() if local_only is None else bool(local_only)
    with network_blocked_if_offline(local_only=offline):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DriftInputError("drift aggregate window file must be an object")
    return DriftAggregateWindow.from_mapping(payload)


def load_drift_reference(
    path: str | Path = DEFAULT_REFERENCE_PATH,
    *,
    local_only: bool | None = None,
) -> DriftAggregateWindow:
    """Load the committed reference window from ``gates/drift_reference.json``."""

    return load_drift_window(path, local_only=local_only)


def write_drift_report(report: DriftReport, path: str | Path) -> Path:
    """Write a deterministic drift record to ``path`` and return it."""

    return report.write_json(path)


def build_drift_window(
    window_id: str,
    *,
    label_counts: Mapping[str, int],
    score_histogram: Sequence[float],
    length_histogram: Mapping[str, int],
    script_histogram: Mapping[str, int],
    feature_hashes: Mapping[str, str] | None = None,
    generated_at: str | None = None,
) -> DriftAggregateWindow:
    """Build an aggregate window from algorithmic counts (no raw text)."""

    payload: dict[str, Any] = {
        "window_id": window_id,
        "label_counts": dict(label_counts),
        "score_histogram": list(score_histogram),
        "length_histogram": dict(length_histogram),
        "script_histogram": dict(script_histogram),
        "feature_hashes": dict(feature_hashes or {}),
    }
    if generated_at is not None:
        payload["generated_at"] = generated_at
    return DriftAggregateWindow.from_mapping(payload)


def population_stability_index(
    reference: Mapping[str, float],
    observation: Mapping[str, float],
) -> float:
    """Return the PSI between two count distributions over shared buckets.

    ``PSI = sum((o_i - r_i) * ln(o_i / r_i))`` over the union of buckets, with
    epsilon smoothing so empty buckets never divide by zero. PSI is symmetric to
    bucket ordering and deterministic for fixed inputs.
    """

    contributions = _psi_contributions(reference, observation)
    return math.fsum(row["contribution"] for row in contributions)


def _psi_contributions(
    reference: Mapping[str, float],
    observation: Mapping[str, float],
) -> list[dict[str, Any]]:
    reference_proportions = _proportions(reference)
    observation_proportions = _proportions(observation)
    buckets = sorted(set(reference_proportions) | set(observation_proportions))
    contributions: list[dict[str, Any]] = []
    for bucket in buckets:
        ref = reference_proportions.get(bucket, 0.0) + _PSI_EPSILON
        obs = observation_proportions.get(bucket, 0.0) + _PSI_EPSILON
        contribution = (obs - ref) * math.log(obs / ref)
        contributions.append(
            {
                "bucket": bucket,
                "reference_proportion": reference_proportions.get(bucket, 0.0),
                "observation_proportion": observation_proportions.get(bucket, 0.0),
                "contribution": contribution,
            }
        )
    return contributions


def _family_divergence(
    family: str,
    reference: DriftAggregateWindow,
    observation: DriftAggregateWindow,
    *,
    threshold: float,
    warning_threshold: float,
) -> DriftFamilyDivergence:
    reference_counts = _family_counts(family, reference)
    observation_counts = _family_counts(family, observation)
    contributions = _psi_contributions(reference_counts, observation_counts)
    divergence = math.fsum(row["contribution"] for row in contributions)
    divergence = max(divergence, 0.0)

    dominant_bucket: str | None = None
    positive = [row for row in contributions if row["contribution"] > 0.0]
    if positive:
        dominant_bucket = max(
            positive,
            key=lambda row: (row["contribution"], row["bucket"]),
        )["bucket"]

    return DriftFamilyDivergence(
        family=family,
        divergence=divergence,
        verdict=_verdict_for(
            divergence,
            threshold=threshold,
            warning_threshold=warning_threshold,
        ),
        dominant_bucket=dominant_bucket,
        bucket_contributions=tuple(contributions),
    )


def _family_counts(family: str, window: DriftAggregateWindow) -> dict[str, float]:
    if family == LABEL_RATE_FAMILY:
        return {key: float(value) for key, value in window.label_counts.items()}
    if family == LENGTH_HISTOGRAM_FAMILY:
        return {key: float(value) for key, value in window.length_histogram.items()}
    if family == SCRIPT_HISTOGRAM_FAMILY:
        return {key: float(value) for key, value in window.script_histogram.items()}
    if family == SCORE_HISTOGRAM_FAMILY:
        return {
            _score_bucket_name(index): float(value)
            for index, value in enumerate(window.score_histogram)
        }
    raise DriftInputError(f"unknown drift family: {family}")


def _verdict_for(
    divergence: float,
    *,
    threshold: float,
    warning_threshold: float,
) -> str:
    if divergence >= threshold:
        return VERDICT_DRIFT
    if divergence >= warning_threshold:
        return VERDICT_WARNING
    return VERDICT_STABLE


def _as_window(
    window: DriftAggregateWindow | Mapping[str, Any],
) -> DriftAggregateWindow:
    if isinstance(window, DriftAggregateWindow):
        return window
    return DriftAggregateWindow.from_mapping(window)


def _proportions(counts: Mapping[str, float]) -> dict[str, float]:
    total = math.fsum(float(value) for value in counts.values())
    if total <= 0.0:
        return {key: 0.0 for key in counts}
    return {key: float(value) / total for key, value in counts.items()}


def _coerce_counts(value: Any, field_name: str) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DriftInputError(f"{field_name} must be an object")
    counts: dict[str, int] = {}
    for key, raw in value.items():
        bucket = _safe_identifier(key, f"{field_name} bucket")
        counts[bucket] = _coerce_count(raw, f"{field_name}.{bucket}")
    return counts


def _coerce_count(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DriftInputError(f"{field_name} must be a non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise DriftInputError(f"{field_name} must be a non-negative number")
    if number != int(number):
        raise DriftInputError(f"{field_name} must be an integer count")
    return int(number)


def _coerce_score_histogram(value: Any) -> tuple[float, ...]:
    if value is None:
        return tuple(0.0 for _ in range(DEFAULT_SCORE_BINS))
    if not isinstance(value, (list, tuple)):
        raise DriftInputError("score_histogram must be a list of bin counts")
    counts: list[float] = []
    for index, raw in enumerate(value):
        counts.append(float(_coerce_count(raw, f"score_histogram[{index}]")))
    return tuple(counts)


def _coerce_hashes(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DriftInputError("feature_hashes must be an object")
    hashes: dict[str, str] = {}
    for key, raw in value.items():
        name = _safe_identifier(key, "feature_hashes key")
        hashes[name] = _safe_identifier(raw, f"feature_hashes.{name}")
    return hashes


def _safe_identifier(value: Any, field_name: str) -> str:
    if value is None:
        raise DriftInputError(f"{field_name} is required")
    text = str(value)
    if not _SAFE_TOKEN.fullmatch(text):
        raise DriftPrivacyError(
            f"{field_name} must be a short aggregate-safe identifier, hash, or "
            "offset; got a non-token value"
        )
    return text


def _score_bucket_name(index: int) -> str:
    return f"score_bin_{index:02d}"


def _join_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_DRIFT_THRESHOLD",
    "DEFAULT_REFERENCE_PATH",
    "DEFAULT_SCORE_BINS",
    "DEFAULT_WARNING_THRESHOLD",
    "DRIFT_FAMILIES",
    "DRIFT_MONITOR_SCHEMA_VERSION",
    "DRIFT_TRIGGER_SCHEMA_VERSION",
    "DRIFT_WINDOW_SCHEMA_VERSION",
    "LABEL_RATE_FAMILY",
    "LENGTH_HISTOGRAM_FAMILY",
    "SCORE_HISTOGRAM_FAMILY",
    "SCRIPT_HISTOGRAM_FAMILY",
    "VERDICT_DRIFT",
    "VERDICT_STABLE",
    "VERDICT_WARNING",
    "DriftAggregateWindow",
    "DriftFamilyDivergence",
    "DriftInputError",
    "DriftPrivacyError",
    "DriftReport",
    "DriftTriggerSignal",
    "assert_no_raw_text",
    "build_drift_window",
    "compute_drift_report",
    "load_drift_reference",
    "load_drift_window",
    "population_stability_index",
    "write_drift_report",
]
