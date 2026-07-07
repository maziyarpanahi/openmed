"""Reproducible clinical-validation study runner.

The study runner takes **user-supplied** labeled data (referenced by a config
path; nothing sensitive is bundled) and a model or runner, computes the
protocol metrics by reusing :mod:`openmed.eval` (recall/precision/F1, leakage,
subgroup fairness), and emits a signed, reproducible validation report as JSON
and Markdown with provenance hashes.

Design constraints:

* **Local-first** — the runner performs no mandatory network calls.
* **User-supplied data** — the dataset path is taken from the study config. The
  repository only ships a fully synthetic sample for tests and documentation.
* **Leakage-first** — PHI leakage is a gating primary metric.
* **No raw PHI in reports** — the report records offsets, hashes, counts, and
  rates only. The dataset is fingerprinted through a content-addressed manifest
  that never persists raw text.
* **Signed and reproducible** — re-running on identical inputs yields identical
  provenance and repro hashes; the report is HMAC-signed with a caller key.

The scaffold supports internal clinical-validation studies. It does not certify
a model for clinical use and is not a medical device.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed import __version__ as OPENMED_VERSION
from openmed.clinical.validation.protocol import (
    CLINICAL_VALIDATION_DISCLAIMER,
    SUBGROUP_AXES,
    VALIDATION_PROTOCOL_ID,
    VALIDATION_PROTOCOL_SCHEMA_VERSION,
    AcceptanceThreshold,
    coerce_acceptance_thresholds,
)
from openmed.core.audit import AuditSignature, stable_hash
from openmed.eval.cache import eval_code_hash
from openmed.eval.data_provenance import build_training_data_manifest
from openmed.eval.fairness import fairness_report
from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    default_model_runner,
    load_fixtures,
)
from openmed.eval.metrics import compute_exact_span_f1, compute_leakage_rate

VALIDATION_REPORT_SCHEMA_VERSION = "openmed.clinical_validation_report.v1"
_SIGNATURE_ALGORITHM = "HMAC-SHA256"
_DEFAULT_SIGNING_KEY = "openmed-clinical-validation-local-key"

ValidationRunner = Callable[[BenchmarkFixture, str, str], Iterable[Any]]


@dataclass(frozen=True)
class StudyConfig:
    """Configuration for one clinical-validation study run.

    Attributes:
        dataset_path: Filesystem path to the **user-supplied** labeled dataset
            (JSON or JSONL benchmark fixtures). Never bundled beyond the
            synthetic sample.
        model_name: Model identifier recorded in the report and passed to the
            default runner.
        dataset_id: Stable identifier for the labeled dataset, recorded in the
            provenance manifest.
        data_revision: Caller-supplied revision string (for example a git SHA
            or dataset version tag) for the labeled data.
        device: Device tag passed to the runner and metric normalization.
        study_id: Human-readable study identifier for the report.
        subgroup_axes: Fairness axes to break results down by.
        threshold_overrides: Optional per-metric acceptance-threshold overrides.
        signing_key_id: Key identifier stamped into the report signature.
    """

    dataset_path: str | Path
    model_name: str
    dataset_id: str
    data_revision: str
    device: str = "cpu"
    study_id: str = "clinical-validation-study"
    subgroup_axes: Sequence[str] = SUBGROUP_AXES
    threshold_overrides: Mapping[str, Any] | None = None
    signing_key_id: str = "clinical-validation"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StudyConfig":
        """Build a study config from a JSON-ready mapping.

        Raises:
            ValueError: If any required field is missing.
        """

        missing = [
            key
            for key in ("dataset_path", "model_name", "dataset_id", "data_revision")
            if not data.get(key)
        ]
        if missing:
            raise ValueError(
                "study config missing required fields: " + ", ".join(sorted(missing))
            )
        return cls(
            dataset_path=str(data["dataset_path"]),
            model_name=str(data["model_name"]),
            dataset_id=str(data["dataset_id"]),
            data_revision=str(data["data_revision"]),
            device=str(data.get("device", "cpu")),
            study_id=str(data.get("study_id", "clinical-validation-study")),
            subgroup_axes=tuple(data.get("subgroup_axes") or SUBGROUP_AXES),
            threshold_overrides=data.get("threshold_overrides"),
            signing_key_id=str(data.get("signing_key_id", "clinical-validation")),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "StudyConfig":
        """Read a study config from a JSON file."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("study config JSON must be an object")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class AcceptanceResult:
    """Outcome of evaluating one acceptance criterion against an observation."""

    metric: str
    direction: str
    threshold: float
    observed: float
    primary: bool
    passed: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping."""

        return {
            "metric": self.metric,
            "direction": self.direction,
            "threshold": self.threshold,
            "observed": self.observed,
            "primary": self.primary,
            "passed": self.passed,
            "description": self.description,
        }


@dataclass
class ValidationReport:
    """Signed, reproducible clinical-validation report.

    The payload is PHI-free: it records metric rates and counts, subgroup
    breakdowns, acceptance outcomes, and provenance hashes. Raw document text
    and identifiers never enter the report.
    """

    study_id: str
    model_name: str
    device: str
    protocol_id: str
    protocol_schema_version: str
    fixture_count: int
    metrics: Mapping[str, Any]
    subgroups: Mapping[str, Any]
    acceptance: Sequence[AcceptanceResult]
    provenance: Mapping[str, Any]
    disclaimer: str = CLINICAL_VALIDATION_DISCLAIMER
    schema_version: str = VALIDATION_REPORT_SCHEMA_VERSION
    generated_at: str | None = None
    repro_hash: str = ""
    signature: AuditSignature | None = None

    def __post_init__(self) -> None:
        self.acceptance = tuple(self.acceptance)
        if not self.repro_hash:
            self.repro_hash = self.recompute_repro_hash()

    @property
    def accepted(self) -> bool:
        """Return whether every primary acceptance criterion passed."""

        return all(result.passed for result in self.acceptance if result.primary)

    def _payload(
        self,
        *,
        include_repro_hash: bool,
        include_signature: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "acceptance": [result.to_dict() for result in self.acceptance],
            "accepted": self.accepted,
            "device": self.device,
            "disclaimer": self.disclaimer,
            "fixture_count": int(self.fixture_count),
            "generated_at": self.generated_at,
            "metrics": _plain(self.metrics),
            "model_name": self.model_name,
            "protocol_id": self.protocol_id,
            "protocol_schema_version": self.protocol_schema_version,
            "provenance": _plain(self.provenance),
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "subgroups": _plain(self.subgroups),
        }
        if include_repro_hash:
            payload["repro_hash"] = self.repro_hash
        if include_signature:
            payload["signature"] = (
                self.signature.to_dict() if self.signature is not None else None
            )
        return payload

    def recompute_repro_hash(self) -> str:
        """Hash the deterministic, PHI-free report payload."""

        return stable_hash(
            self._payload(include_repro_hash=False, include_signature=False)
        )

    def sign(
        self,
        key: bytes | str,
        *,
        key_id: str = "clinical-validation",
    ) -> "ValidationReport":
        """Sign the report payload with an HMAC key and return ``self``."""

        self.repro_hash = self.recompute_repro_hash()
        message = _canonical_json(
            self._payload(include_repro_hash=True, include_signature=False)
        ).encode("utf-8")
        self.signature = AuditSignature(
            key_id=key_id,
            algorithm=_SIGNATURE_ALGORITHM,
            value=hmac.new(_key_bytes(key), message, hashlib.sha256).hexdigest(),
        )
        return self

    def verify(self, key: bytes | str) -> bool:
        """Verify the signature and reproducibility hash with a non-empty key."""

        if self.recompute_repro_hash() != self.repro_hash:
            return False
        if self.signature is None or self.signature.algorithm != _SIGNATURE_ALGORITHM:
            return False
        message = _canonical_json(
            self._payload(include_repro_hash=True, include_signature=False)
        ).encode("utf-8")
        expected = hmac.new(_key_bytes(key), message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature.value)

    def to_dict(self) -> dict[str, Any]:
        """Return the full JSON-ready report payload."""

        return self._payload(include_repro_hash=True, include_signature=True)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a deterministic, single-page Markdown validation report."""

        payload = self.to_dict()
        metrics = payload["metrics"]
        overall = metrics.get("overall", {})
        lines = [
            f"# Clinical Validation Report: {self.study_id}",
            "",
            f"> {self.disclaimer}",
            "",
            "## Summary",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Study | `{self.study_id}` |",
            f"| Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| Protocol | `{self.protocol_id}` |",
            f"| Fixtures | {self.fixture_count} |",
            f"| Accepted | {'yes' if payload['accepted'] else 'no'} |",
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | `{self.generated_at}` |")

        lines.extend(
            [
                "",
                "## Primary Metrics",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Recall | {_fmt(overall.get('recall'))} |",
                f"| Precision | {_fmt(overall.get('precision'))} |",
                f"| F1 | {_fmt(overall.get('f1'))} |",
                f"| Leakage Rate | {_fmt(overall.get('leakage_rate'))} |",
                "",
                "## Acceptance",
                "",
                "| Metric | Direction | Threshold | Observed | Primary | Result |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for result in payload["acceptance"]:
            lines.append(
                "| "
                f"`{result['metric']}` | "
                f"{result['direction']} | "
                f"{_fmt(result['threshold'])} | "
                f"{_fmt(result['observed'])} | "
                f"{'yes' if result['primary'] else 'no'} | "
                f"{'PASS' if result['passed'] else 'FAIL'} |"
            )

        lines.extend(["", "## Subgroups", ""])
        for axis in sorted(payload["subgroups"]):
            axis_report = payload["subgroups"][axis]
            per_group = axis_report.get("per_group", {})
            lines.extend(
                [
                    f"### Axis: `{axis}`",
                    "",
                    (
                        f"Leakage disparity: {_fmt(axis_report.get('leakage_disparity'))}"
                        f" (worst group: `{axis_report.get('worst_group')}`)"
                    ),
                    "",
                    "| Group | Leakage Rate | Recall | Spans |",
                    "|---|---:|---:|---:|",
                ]
            )
            for group in sorted(per_group):
                group_metrics = per_group[group]
                lines.append(
                    "| "
                    f"`{group}` | "
                    f"{_fmt(group_metrics.get('leakage_rate'))} | "
                    f"{_fmt(group_metrics.get('recall'))} | "
                    f"{group_metrics.get('span_count')} |"
                )
            lines.append("")

        provenance = payload["provenance"]
        signature = payload.get("signature") or {}
        lines.extend(
            [
                "## Provenance",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| OpenMed Version | `{provenance.get('openmed_version')}` |",
                f"| Dataset ID | `{provenance.get('dataset_id')}` |",
                f"| Data Revision | `{provenance.get('data_revision')}` |",
                f"| Dataset Manifest Hash | `{provenance.get('dataset_manifest_hash')}` |",
                f"| Eval Code Hash | `{provenance.get('eval_code_hash')}` |",
                f"| Repro Hash | `{self.repro_hash}` |",
                (
                    "| Signature | "
                    f"`{signature.get('algorithm', 'unsigned')}:"
                    f"{signature.get('value', '')}` |"
                ),
            ]
        )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def load_study_dataset(path: str | Path) -> list[BenchmarkFixture]:
    """Load user-supplied labeled fixtures, tolerating a leading meta row.

    The synthetic sample and typical study exports start with a ``{"kind":
    "meta", ...}`` descriptor row. That row is skipped; every remaining row is a
    labeled benchmark fixture. JSON and JSONL inputs are both accepted.
    """

    dataset_path = Path(path)
    if dataset_path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        fixtures = [
            BenchmarkFixture.from_mapping(row)
            for row in rows
            if row.get("kind") != "meta"
        ]
        _validate_unique(fixtures)
        return fixtures
    return load_fixtures(dataset_path)


def run_validation_study(
    config: StudyConfig,
    *,
    runner: ValidationRunner | None = None,
    signing_key: bytes | str | None = None,
    generated_at: str | None = None,
) -> ValidationReport:
    """Run a clinical-validation study and return a signed report.

    Args:
        config: Study configuration, including the user-supplied dataset path.
        runner: Optional model runner with the harness ``ModelRunner``
            signature. Defaults to the shared PII runtime runner.
        signing_key: HMAC signing key. Defaults to a local key so reports are
            always signed and self-verifiable offline.
        generated_at: Optional timestamp recorded in the report. Left ``None``
            by default to keep the repro hash independent of wall-clock time.

    Returns:
        A :class:`ValidationReport` with primary/subgroup metrics, acceptance
        outcomes, provenance hashes, and an HMAC signature.
    """

    fixtures = load_study_dataset(config.dataset_path)
    if not fixtures:
        raise ValueError("validation dataset contains no labeled fixtures")

    model_runner: ModelRunner = runner or default_model_runner
    thresholds = coerce_acceptance_thresholds(config.threshold_overrides)

    predictions_by_fixture = {
        fixture.fixture_id: _predict(fixture, config, model_runner)
        for fixture in fixtures
    }

    overall = _overall_metrics(fixtures, predictions_by_fixture, config.device)
    subgroups = {
        axis: _subgroup_metrics(
            fixtures,
            config=config,
            runner=model_runner,
            axis=axis,
        )
        for axis in config.subgroup_axes
    }
    acceptance = _evaluate_acceptance(overall, subgroups, thresholds)

    provenance = _build_provenance(fixtures, config)
    report = ValidationReport(
        study_id=config.study_id,
        model_name=config.model_name,
        device=config.device,
        protocol_id=VALIDATION_PROTOCOL_ID,
        protocol_schema_version=VALIDATION_PROTOCOL_SCHEMA_VERSION,
        fixture_count=len(fixtures),
        metrics={"overall": overall},
        subgroups=subgroups,
        acceptance=acceptance,
        provenance=provenance,
        generated_at=generated_at,
    )
    report.sign(
        signing_key if signing_key is not None else _DEFAULT_SIGNING_KEY,
        key_id=config.signing_key_id,
    )
    return report


def _predict(
    fixture: BenchmarkFixture,
    config: StudyConfig,
    runner: ModelRunner,
) -> tuple[Any, ...]:
    from openmed.eval.metrics import normalize_eval_spans

    raw = list(runner(fixture, config.model_name, config.device))
    return tuple(
        normalize_eval_spans(
            raw,
            default_language=fixture.language,
            default_device=config.device,
            source_text=fixture.text,
        )
    )


def _overall_metrics(
    fixtures: Sequence[BenchmarkFixture],
    predictions_by_fixture: Mapping[str, Sequence[Any]],
    device: str,
) -> dict[str, Any]:
    gold_all: list[Any] = []
    pred_all: list[Any] = []
    corpus_parts: list[str] = []
    offset = 0
    for fixture in fixtures:
        predicted = predictions_by_fixture[fixture.fixture_id]
        gold_all.extend(_shift(span, offset) for span in fixture.gold_spans)
        pred_all.extend(_shift(span, offset) for span in predicted)
        corpus_parts.append(fixture.text)
        offset += len(fixture.text) + 1
    corpus_text = "\n".join(corpus_parts)

    f1 = compute_exact_span_f1(
        gold_all, pred_all, default_device=device, source_text=corpus_text
    )
    leakage = compute_leakage_rate(
        gold_all, pred_all, default_device=device, source_text=corpus_text
    )
    return {
        "recall": f1.recall,
        "precision": f1.precision,
        "f1": f1.f1,
        "true_positives": f1.true_positives,
        "false_positives": f1.false_positives,
        "false_negatives": f1.false_negatives,
        "leakage_rate": leakage.overall,
        "leaked_chars": leakage.leaked_chars,
        "total_chars": leakage.total_chars,
    }


def _subgroup_metrics(
    fixtures: Sequence[BenchmarkFixture],
    *,
    config: StudyConfig,
    runner: ModelRunner,
    axis: str,
) -> dict[str, Any]:
    axis_fixtures = [_regroup(fixture, axis) for fixture in fixtures]
    report = fairness_report(
        config.model_name,
        axis_fixtures,
        runner=runner,
        device=config.device,
    )
    payload = report.to_dict()
    payload["axis"] = axis
    return payload


def _evaluate_acceptance(
    overall: Mapping[str, Any],
    subgroups: Mapping[str, Any],
    thresholds: Sequence[AcceptanceThreshold],
) -> tuple[AcceptanceResult, ...]:
    disparities = [
        float(axis_report.get("leakage_disparity", 0.0))
        for axis_report in subgroups.values()
    ]
    worst_disparity = max(disparities) if disparities else 0.0
    observations = {
        "recall": float(overall.get("recall", 0.0)),
        "precision": float(overall.get("precision", 0.0)),
        "f1": float(overall.get("f1", 0.0)),
        "leakage_rate": float(overall.get("leakage_rate", 0.0)),
        "subgroup_leakage_disparity": worst_disparity,
    }
    results: list[AcceptanceResult] = []
    for threshold in thresholds:
        observed = observations.get(threshold.metric)
        if observed is None:
            continue
        results.append(
            AcceptanceResult(
                metric=threshold.metric,
                direction=threshold.direction,
                threshold=threshold.threshold,
                observed=observed,
                primary=threshold.primary,
                passed=threshold.evaluate(observed),
                description=threshold.description,
            )
        )
    return tuple(results)


def _build_provenance(
    fixtures: Sequence[BenchmarkFixture],
    config: StudyConfig,
) -> dict[str, Any]:
    manifest = build_training_data_manifest(
        (_fixture_manifest_row(fixture) for fixture in fixtures),
        dataset_id=config.dataset_id,
        data_revision=config.data_revision,
        source="user-supplied",
    )
    return {
        "openmed_version": OPENMED_VERSION,
        "dataset_id": config.dataset_id,
        "data_revision": config.data_revision,
        "dataset_source": "user-supplied",
        "dataset_manifest_hash": manifest["manifest_hash"],
        "eval_code_hash": f"sha256:{eval_code_hash()}",
        "fixture_count": len(fixtures),
        "acceptance_thresholds": [
            threshold.to_dict()
            for threshold in coerce_acceptance_thresholds(config.threshold_overrides)
        ],
    }


def _fixture_manifest_row(fixture: BenchmarkFixture) -> dict[str, Any]:
    return {
        "fixture_id": fixture.fixture_id,
        "text": fixture.text,
        "language": fixture.language,
        "gold_spans": [
            {"start": span.start, "end": span.end, "label": span.label, "text": ""}
            for span in fixture.gold_spans
        ],
    }


def _regroup(fixture: BenchmarkFixture, axis: str) -> BenchmarkFixture:
    """Return a copy whose gold spans carry the axis value as their group."""

    if axis == "language":
        group_value = fixture.language
    else:
        group_value = str(fixture.metadata.get(axis, "unspecified"))
    from dataclasses import replace as _replace

    regrouped_spans = tuple(
        _with_group(span, group_value) for span in fixture.gold_spans
    )
    return _replace(fixture, gold_spans=regrouped_spans)


def _with_group(span: Any, group_value: str) -> Any:
    from dataclasses import replace as _replace

    metadata = dict(span.metadata or {})
    metadata["group"] = group_value
    return _replace(span, metadata=metadata)


def _shift(span: Any, offset: int) -> Any:
    from dataclasses import replace as _replace

    if offset == 0:
        return span
    return _replace(span, start=span.start + offset, end=span.end + offset)


def _validate_unique(fixtures: Sequence[BenchmarkFixture]) -> None:
    seen: set[str] = set()
    for fixture in fixtures:
        if fixture.fixture_id in seen:
            raise ValueError(f"duplicate fixture id: {fixture.fixture_id}")
        seen.add(fixture.fixture_id)


def _canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _key_bytes(key: bytes | str) -> bytes:
    if isinstance(key, bytes):
        if not key:
            raise ValueError("signing key must be non-empty")
        return key
    if isinstance(key, str):
        if not key:
            raise ValueError("signing key must be non-empty")
        return key.encode("utf-8")
    raise TypeError("signing key must be str or bytes")


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


__all__ = [
    "AcceptanceResult",
    "StudyConfig",
    "VALIDATION_REPORT_SCHEMA_VERSION",
    "ValidationReport",
    "ValidationRunner",
    "load_study_dataset",
    "run_validation_study",
]
