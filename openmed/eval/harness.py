"""Suite runner for OpenMed benchmark fixtures."""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from openmed.core.audit import AuditSignature, stable_hash
from openmed.core.quality_gates import validate_entity_spans
from openmed.core.safety_sweep import hashed_span_surface
from openmed.eval.cache import build_report_key, hash_fixture_set, load_or_compute
from openmed.eval.calibrate import load_calibration_thresholds
from openmed.eval.metrics import (
    PIPELINE_EVAL_STAGES,
    EvalSpan,
    F1Metrics,
    PipelineFact,
    compute_confidence_intervals,
    compute_exact_span_f1,
    compute_fact_level_f1,
    compute_latency_summary,
    compute_metrics_bundle,
    compute_relaxed_span_f1,
    compute_resource_metrics,
    compute_span_grounded_faithfulness,
    expected_calibration_error,
    merge_fact_level_f1,
    merge_faithfulness_metrics,
    normalize_eval_spans,
    normalize_pipeline_facts,
    reliability_bins,
)
from openmed.eval.relation_metrics import (
    EvalRelation,
    compute_relation_confidence_intervals,
    compute_relation_metrics_bundle,
    normalize_eval_relations,
)
from openmed.eval.report import BenchmarkReport

if TYPE_CHECKING:
    from openmed.eval.attacks.reid import SideChannelProbeResult
    from openmed.eval.error_analysis import PipelineAttributionReport

ModelRunner = Callable[["BenchmarkFixture", str, str], Iterable[Any]]
RelationModelRunner = Callable[[Any, str, str], Iterable[Any]]
PipelineStageRunner = Callable[
    ["PipelineEvalFixture", Mapping[str, "PipelineStageOutput"]],
    "PipelineStageOutput | Mapping[str, Any] | Iterable[Any]",
]
_SIGNATURE_ALGORITHM = "HMAC-SHA256"
_DEFAULT_FEDERATED_SIGNING_KEY = "openmed-federated-eval-local-key"
_DEFAULT_RELATION_SCORECARD_SIGNING_KEY = "openmed-relation-scorecard-local-key"
RELATION_SCORECARD_ARTIFACT = "openmed.eval.relation_scorecard"
RELATION_SCORECARD_SCHEMA_VERSION = 1
DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE = (
    Path(__file__).resolve().parent
    / "golden"
    / "fixtures"
    / "context_multilingual.jsonl"
)
DEFAULT_SECTION_MULTILINGUAL_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "section_multilingual.jsonl"
)
DEFAULT_PIPELINE_EVAL_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "pipeline_e2e_synthetic.jsonl"
)
PIPELINE_EVAL_SCHEMA_VERSION = "openmed.eval.pipeline_e2e.v1"
_PIPELINE_RAW_METADATA_KEYS = frozenset(
    {
        "deidentified_text",
        "mention_text",
        "raw_text",
        "source_text",
        "span_text",
        "text",
    }
)


@dataclass(frozen=True)
class BenchmarkFixture:
    """One benchmark document with gold PHI spans."""

    fixture_id: str
    text: str
    gold_spans: tuple[EvalSpan, ...]
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BenchmarkFixture":
        """Build a fixture from a JSON-ready mapping."""
        text = str(data.get("text", ""))
        language = str(data.get("language") or data.get("lang") or "en")
        fixture_id = str(data.get("id") or data.get("fixture_id") or "fixture")
        gold_spans = tuple(
            normalize_eval_spans(
                data.get("gold_spans") or data.get("entities") or [],
                default_language=language,
                source_text=text,
            )
        )
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {"value": metadata}
        return cls(
            fixture_id=fixture_id,
            text=text,
            gold_spans=gold_spans,
            language=language,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class FixtureResult:
    """Predictions and timing for one benchmark fixture."""

    fixture_id: str
    predicted_spans: tuple[EvalSpan, ...]
    latency_ms: float


@dataclass(frozen=True)
class PipelineEvalFixture:
    """One raw-note fixture with exact gold facts for all pipeline stages."""

    fixture_id: str
    text: str
    gold_facts: tuple[PipelineFact, ...]
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PipelineEvalFixture":
        """Build a pipeline fixture from a JSON-ready synthetic mapping."""

        fixture_id = str(data.get("fixture_id") or data.get("id") or "fixture")
        text = str(data.get("text") or "")
        if not fixture_id:
            raise ValueError("pipeline fixtures require a non-empty fixture_id")
        if not text:
            raise ValueError("pipeline fixtures require non-empty text")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise TypeError("pipeline fixture metadata must be a mapping")
        return cls(
            fixture_id=fixture_id,
            text=text,
            gold_facts=normalize_pipeline_facts(data.get("gold_facts") or ()),
            language=str(data.get("language") or data.get("lang") or "en"),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class PipelineStageOutput:
    """Raw-text-free fact snapshots emitted by one evaluated pipeline stage."""

    stage: str
    facts: tuple[PipelineFact, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in PIPELINE_EVAL_STAGES:
            allowed = ", ".join(PIPELINE_EVAL_STAGES)
            raise ValueError(
                f"unknown pipeline stage {self.stage!r}; expected one of {allowed}"
            )
        object.__setattr__(self, "facts", normalize_pipeline_facts(self.facts))

    def to_dict(self) -> dict[str, Any]:
        """Return the intermediate output without raw clinical text."""

        return {
            "facts": [fact.to_dict() for fact in self.facts],
            "metadata": _privacy_safe_pipeline_metadata(self.metadata),
            "stage": self.stage,
        }


@dataclass(frozen=True)
class PipelineFixtureEvalResult:
    """Intermediate outputs, final F1, and attribution for one fixture."""

    fixture_id: str
    source_hash: str
    stage_outputs: tuple[PipelineStageOutput, ...]
    fact_level: F1Metrics
    attribution: "PipelineAttributionReport"

    def stage(self, name: str) -> PipelineStageOutput:
        """Return one captured stage output by canonical name."""

        for output in self.stage_outputs:
            if output.stage == name:
                return output
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, raw-text-free fixture evidence."""

        return {
            "attribution": self.attribution.to_dict(),
            "fact_level": self.fact_level.to_dict(),
            "fixture_id": self.fixture_id,
            "source_hash": self.source_hash,
            "stages": [output.to_dict() for output in self.stage_outputs],
        }


@dataclass(frozen=True)
class PipelineEvalReport:
    """Aggregate end-to-end fact scoring with first-defect attribution."""

    suite: str
    fixture_results: tuple[PipelineFixtureEvalResult, ...]
    fact_level: F1Metrics
    attribution: "PipelineAttributionReport"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PIPELINE_EVAL_SCHEMA_VERSION

    @property
    def fixture_count(self) -> int:
        """Return the number of evaluated fixtures."""

        return len(self.fixture_results)

    def to_metric(self) -> dict[str, Any]:
        """Return the compact metric payload consumed by release gate G15."""

        attribution = self.attribution.to_dict()
        return {
            "artifact_type": PIPELINE_EVAL_SCHEMA_VERSION,
            "attribution": attribution,
            "fact_f1": self.fact_level.f1,
            "fact_level": self.fact_level.to_dict(),
            "fixture_count": self.fixture_count,
            "stage_error_counts": attribution["stage_error_counts"],
            "total_end_to_end_errors": attribution["total_end_to_end_errors"],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the full report with every captured intermediate output."""

        return {
            "attribution": self.attribution.to_dict(),
            "fact_level": self.fact_level.to_dict(),
            "fixture_count": self.fixture_count,
            "fixtures": [result.to_dict() for result in self.fixture_results],
            "metadata": _privacy_safe_pipeline_metadata(self.metadata),
            "schema_version": self.schema_version,
            "stage_order": list(PIPELINE_EVAL_STAGES),
            "suite": self.suite,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize deterministic, privacy-safe end-to-end evidence."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the complete pipeline evaluation report."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def write_attribution_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the standalone per-stage attribution report for evidence bundles."""

        return self.attribution.write_json(path, indent=indent)


@dataclass(frozen=True)
class RelationFixtureResult:
    """Predicted relation triples and timing for one relation fixture."""

    fixture_id: str
    predicted_relations: tuple[EvalRelation, ...]
    latency_ms: float


class RelationGateFailure(RuntimeError):
    """Raised with signed evidence when a relation suite fails its gate."""

    def __init__(self, scorecard: "RelationScorecard") -> None:
        self.scorecard = scorecard
        reason = str(scorecard.gate_result.get("reason") or "relation gate failed")
        super().__init__(reason)


@dataclass
class RelationScorecard:
    """Signed, aggregate-only relation suite evidence for model cards."""

    suite: str
    model_name: str
    device: str
    fixture_count: int
    metrics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    gate_result: Mapping[str, Any]
    gate_passed: bool
    generated_at: str | None = None
    repro_hash: str = ""
    signature: AuditSignature | None = None

    def __post_init__(self) -> None:
        self.metrics = _plain(self.metrics)
        self.provenance = _plain(self.provenance)
        self.gate_result = _plain(self.gate_result)
        if bool(self.gate_result.get("passed", False)) != self.gate_passed:
            raise ValueError("relation scorecard gate result is inconsistent")
        if not self.repro_hash:
            self.repro_hash = self.recompute_repro_hash()

    def _payload(
        self,
        *,
        include_repro_hash: bool,
        include_signature: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_type": RELATION_SCORECARD_ARTIFACT,
            "device": self.device,
            "fixture_count": int(self.fixture_count),
            "gate_passed": bool(self.gate_passed),
            "gate_result": _plain(self.gate_result),
            "generated_at": self.generated_at,
            "metrics": _plain(self.metrics),
            "model_name": self.model_name,
            "provenance": _plain(self.provenance),
            "schema_version": RELATION_SCORECARD_SCHEMA_VERSION,
            "suite": self.suite,
        }
        if include_repro_hash:
            payload["repro_hash"] = self.repro_hash
        if include_signature:
            payload["signature"] = (
                self.signature.to_dict() if self.signature is not None else None
            )
        return payload

    def recompute_repro_hash(self) -> str:
        """Recompute the scorecard evidence hash."""

        return stable_hash(
            self._payload(include_repro_hash=False, include_signature=False)
        )

    def sign(
        self,
        key: bytes | str,
        *,
        key_id: str = "relation-scorecard",
    ) -> "RelationScorecard":
        """Sign the complete scorecard and return ``self``."""

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
        """Verify the evidence hash and HMAC signature."""

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
        """Return the stable JSON-ready scorecard payload."""

        return self._payload(include_repro_hash=True, include_signature=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationScorecard":
        """Restore a relation scorecard from a JSON-compatible mapping."""

        if data.get("artifact_type") != RELATION_SCORECARD_ARTIFACT:
            raise ValueError("unsupported relation scorecard artifact_type")
        if data.get("schema_version") != RELATION_SCORECARD_SCHEMA_VERSION:
            raise ValueError("unsupported relation scorecard schema_version")
        signature_data = data.get("signature")
        return cls(
            suite=str(data.get("suite") or ""),
            model_name=str(data.get("model_name") or ""),
            device=str(data.get("device") or ""),
            fixture_count=int(data.get("fixture_count") or 0),
            metrics=dict(data.get("metrics") or {}),
            provenance=dict(data.get("provenance") or {}),
            gate_result=dict(data.get("gate_result") or {}),
            gate_passed=bool(data.get("gate_passed", False)),
            generated_at=(
                str(data["generated_at"])
                if data.get("generated_at") is not None
                else None
            ),
            repro_hash=str(data.get("repro_hash") or ""),
            signature=(
                AuditSignature.from_dict(signature_data)
                if isinstance(signature_data, Mapping)
                else None
            ),
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "RelationScorecard":
        """Restore a relation scorecard from JSON."""

        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid relation scorecard JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("relation scorecard JSON must contain an object")
        return cls.from_dict(payload)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the signed scorecard to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the signed scorecard JSON artifact."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render aggregate relation evidence as deterministic Markdown."""

        strict = _mapping_value(self.metrics.get("strict"))
        relaxed = _mapping_value(self.metrics.get("relaxed"))
        lines = [
            f"# Relation Scorecard: {self.model_name}",
            "",
            "## Summary",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Suite | `{self.suite}` |",
            f"| Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| Fixtures | {self.fixture_count} |",
            f"| Relation gate | {'passed' if self.gate_passed else 'failed'} |",
            f"| Gate reason | {self.gate_result.get('reason', '')} |",
            "",
            "## Overall Relation Metrics",
            "",
            "| Match | Precision | Recall | F1 |",
            "|---|---:|---:|---:|",
            _relation_markdown_metric_row("Strict", strict),
            _relation_markdown_metric_row("Relaxed", relaxed),
            "",
            "## Per-Relation-Type Metrics",
            "",
            "| Relation type | Strict F1 | Relaxed F1 |",
            "|---|---:|---:|",
        ]
        by_type = _mapping_value(self.metrics.get("by_type"))
        if by_type:
            for relation_type, metric in sorted(
                by_type.items(), key=lambda item: str(item[0])
            ):
                pair = _mapping_value(metric)
                lines.append(
                    "| "
                    f"`{relation_type}` | "
                    f"{_relation_percent(_mapping_value(pair.get('strict')).get('f1'))} | "
                    f"{_relation_percent(_mapping_value(pair.get('relaxed')).get('f1'))} |"
                )
        else:
            lines.append("| `none` | n/a | n/a |")

        trap_summary = _mapping_value(self.provenance.get("trap_summary"))
        configured_traps = _mapping_value(trap_summary.get("by_kind"))
        trap_leaks = _mapping_value(self.metrics.get("trap_leaks"))
        consistency = _mapping_value(self.metrics.get("consistency"))
        lines.extend(
            [
                "",
                "## Zero-Tolerance Trap Summary",
                "",
                "| Trap kind | Configured | Leaks | Consistency | Zero tolerance |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for kind in sorted(set(configured_traps) | set(trap_leaks) | set(consistency)):
            configured = _mapping_value(configured_traps.get(kind))
            leaks = _mapping_value(trap_leaks.get(kind))
            consistency_metric = _mapping_value(consistency.get(kind))
            lines.append(
                "| "
                f"`{kind}` | {int(configured.get('count') or 0)} | "
                f"{int(leaks.get('leak_count') or 0)} | "
                f"{_relation_percent(consistency_metric.get('score'))} | "
                f"{'yes' if configured.get('zero_tolerance') else 'no'} |"
            )

        signature = self.signature
        lines.extend(
            [
                "",
                "## Provenance and Signature",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| Fixture set hash | `{self.provenance.get('fixture_set_hash', '')}` |",
                f"| Fixture schema version | `{self.provenance.get('fixture_schema_version', '')}` |",
                f"| Scorecard hash | `{self.repro_hash}` |",
                f"| Signature key | `{signature.key_id if signature else ''}` |",
                f"| Signature algorithm | `{signature.algorithm if signature else ''}` |",
            ]
        )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the human-readable scorecard artifact."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def model_card_evidence(self) -> dict[str, Any]:
        """Return the signed scorecard as a model-card evidence block."""

        return {"relation_scorecard": self.to_dict()}

    def to_benchmark_report(self) -> BenchmarkReport:
        """Expose the scorecard through the existing model-scorecard input shape."""

        strict = _mapping_value(self.metrics.get("strict"))
        relaxed = _mapping_value(self.metrics.get("relaxed"))
        by_type = _mapping_value(self.metrics.get("by_type"))
        by_language = _mapping_value(self.metrics.get("by_language"))
        counts = _mapping_value(self.metrics.get("counts"))
        consistency = _mapping_value(self.metrics.get("consistency"))
        relation_metrics = {
            "by_scope": _mapping_value(self.metrics.get("by_scope")),
            "consistency": consistency,
            "gold_relation_count": int(counts.get("gold") or 0),
            "per_language": by_language,
            "per_relation_type": by_type,
            "predicted_relation_count": int(counts.get("predicted") or 0),
            "relaxed": relaxed,
            "strict": strict,
        }
        return BenchmarkReport(
            suite=self.suite,
            model_name=self.model_name,
            device=self.device,
            fixture_count=self.fixture_count,
            metrics={
                "relation_extraction": relation_metrics,
                "relation_golden": {
                    "by_type": by_type,
                    "consistency": consistency,
                    "trap_leaks": _mapping_value(self.metrics.get("trap_leaks")),
                },
                "relation_consistency": consistency,
                "relaxed_relation_f1": relaxed,
                "strict_relation_f1": strict,
                "per_language_relation_f1": by_language,
                "per_relation_type_re_f1": by_type,
            },
            generated_at=self.generated_at,
            metadata={
                **dict(self.provenance),
                "relation_gate_passed": self.gate_passed,
                "relation_scorecard_hash": self.repro_hash,
                "relation_trap_leaks": _mapping_value(self.metrics.get("trap_leaks")),
                "relation_traps": _mapping_value(self.provenance.get("trap_summary")),
                "task": "relation",
            },
        )


@dataclass(frozen=True)
class FederatedDetectorSpec:
    """Python subprocess detector entry point for federated evaluation."""

    script_path: str | Path
    python_executable: str | Path = sys.executable
    timeout_s: float = 10.0
    read_roots: tuple[str | Path, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryLeakageFinding:
    """Raw-PHI egress evidence keyed by source offsets and hashes only."""

    fixture_id: str
    sink: str
    artifact: str
    start: int
    end: int
    label: str
    length: int
    text_hash: str
    byte_offsets: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "sink": self.sink,
            "artifact": self.artifact,
            "start": int(self.start),
            "end": int(self.end),
            "label": self.label,
            "length": int(self.length),
            "text_hash": self.text_hash,
            "byte_offsets": [int(offset) for offset in self.byte_offsets],
        }


@dataclass(frozen=True)
class BoundaryLeakageResult:
    """Boundary leakage rate for detector stdout/stderr/files."""

    rate: float
    leaked_bytes: int
    total_phi_bytes: int
    findings: tuple[BoundaryLeakageFinding, ...] = ()
    emitted_bytes_by_sink: Mapping[str, int] = field(default_factory=dict)

    def to_metric(self) -> dict[str, Any]:
        return {
            "rate": float(self.rate),
            "leaked_bytes": int(self.leaked_bytes),
            "total_phi_bytes": int(self.total_phi_bytes),
            "findings": [finding.to_dict() for finding in self.findings],
            "emitted_bytes_by_sink": {
                str(key): int(value)
                for key, value in sorted(self.emitted_bytes_by_sink.items())
            },
        }


@dataclass(frozen=True)
class TrainingEvalOverlapFinding:
    """PHI-free evidence that an eval fixture overlaps a training manifest."""

    fixture_id: str
    benchmark: str
    language: str
    split: str
    overlap_key: str
    manifest_row_hash: str
    manifest_line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "fixture_id": self.fixture_id,
            "language": self.language,
            "manifest_line": int(self.manifest_line),
            "manifest_row_hash": self.manifest_row_hash,
            "overlap_key": self.overlap_key,
            "split": self.split,
        }


@dataclass(frozen=True)
class SandboxViolation:
    """Sandbox policy violation reported without raw host paths."""

    fixture_id: str
    kind: str
    event: str
    operation: str
    path_hash: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "fixture_id": self.fixture_id,
            "kind": self.kind,
            "event": self.event,
            "operation": self.operation,
            "detail": self.detail,
        }
        if self.path_hash is not None:
            payload["path_hash"] = self.path_hash
        return payload


@dataclass
class FederatedEvalReport:
    """Signed federated detector-boundary report."""

    suite: str
    detector_name: str
    fixture_count: int
    boundary_leakage: BoundaryLeakageResult
    side_channel: SideChannelProbeResult
    sandbox_violations: tuple[SandboxViolation, ...]
    resource_accounting: Mapping[str, Any]
    gate_passed: bool
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    repro_hash: str = ""
    signature: AuditSignature | None = None

    def __post_init__(self) -> None:
        self.sandbox_violations = tuple(self.sandbox_violations)
        if not self.repro_hash:
            self.repro_hash = self.recompute_repro_hash()

    def _payload(
        self,
        *,
        include_repro_hash: bool,
        include_signature: bool,
        include_resource_accounting: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "suite": self.suite,
            "detector_name": self.detector_name,
            "fixture_count": int(self.fixture_count),
            "boundary_leakage": self.boundary_leakage.to_metric(),
            "side_channel": self.side_channel.to_metric(),
            "sandbox_violations": [
                violation.to_dict() for violation in self.sandbox_violations
            ],
            "gate_passed": bool(self.gate_passed),
            "generated_at": self.generated_at,
            "metadata": _plain(self.metadata),
        }
        if include_resource_accounting:
            payload["resource_accounting"] = _plain(self.resource_accounting)
        if include_repro_hash:
            payload["repro_hash"] = self.repro_hash
        if include_signature:
            payload["signature"] = (
                self.signature.to_dict() if self.signature is not None else None
            )
        return payload

    def recompute_repro_hash(self) -> str:
        """Hash deterministic leakage evidence, excluding volatile resources."""
        return stable_hash(
            self._payload(
                include_repro_hash=False,
                include_signature=False,
                include_resource_accounting=False,
            )
        )

    def sign(
        self,
        key: bytes | str,
        *,
        key_id: str = "federated-eval",
    ) -> "FederatedEvalReport":
        """Sign the complete report payload and return ``self``."""
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
        """Verify the report signature and deterministic evidence hash."""
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
        return self._payload(include_repro_hash=True, include_signature=True)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_benchmark_report(self) -> BenchmarkReport:
        """Expose federated boundary metrics through the benchmark report shape."""
        return BenchmarkReport(
            suite=self.suite,
            model_name=self.detector_name,
            device="federated-subprocess",
            fixture_count=self.fixture_count,
            generated_at=self.generated_at,
            metrics={
                "boundary_leakage": self.boundary_leakage.to_metric(),
                "federated_boundary_leakage_rate": self.boundary_leakage.rate,
                "side_channel": self.side_channel.to_metric(),
                "sandbox_violation_count": len(self.sandbox_violations),
            },
            metadata={
                **dict(self.metadata),
                "federated_eval": True,
                "federated_gate_passed": self.gate_passed,
                "federated_repro_hash": self.repro_hash,
            },
        )


@dataclass(frozen=True)
class _CapturedArtifact:
    sink: str
    artifact: str
    content: bytes


@dataclass(frozen=True)
class _FederatedFixtureRun:
    fixture_id: str
    predicted_spans: tuple[EvalSpan, ...]
    timing_records: tuple[dict[str, Any], ...]
    artifacts: tuple[_CapturedArtifact, ...]
    sandbox_violations: tuple[SandboxViolation, ...]
    elapsed_ms: float
    exit_code: int


def load_pipeline_eval_fixtures(
    path: str | Path = DEFAULT_PIPELINE_EVAL_FIXTURE,
) -> list[PipelineEvalFixture]:
    """Load committed synthetic end-to-end fixtures from JSON or JSONL."""

    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if isinstance(payload, Mapping):
        rows: Sequence[Any] = (payload,)
    elif isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        rows = payload
    else:
        raise ValueError("pipeline fixture file must contain an object or array")
    fixtures = [
        PipelineEvalFixture.from_mapping(row)
        for row in rows
        if isinstance(row, Mapping)
    ]
    if len(fixtures) != len(rows):
        raise ValueError("every pipeline fixture must be a JSON object")
    _validate_unique_pipeline_fixture_ids(fixtures)
    return fixtures


def run_pipeline_eval_fixture(
    fixture: PipelineEvalFixture | Mapping[str, Any],
    stage_runners: Mapping[str, PipelineStageRunner],
) -> PipelineFixtureEvalResult:
    """Run one fixture through de-id, NER, assertion, grounding, and FHIR.

    Each injected runner receives the fixture plus an immutable-by-convention
    mapping of prior outputs. This keeps component adapters offline and makes
    every intermediate fact snapshot explicit without replacing runtime stages.
    """

    resolved_fixture = (
        fixture
        if isinstance(fixture, PipelineEvalFixture)
        else PipelineEvalFixture.from_mapping(fixture)
    )
    missing = [stage for stage in PIPELINE_EVAL_STAGES if stage not in stage_runners]
    unknown = sorted(set(stage_runners) - set(PIPELINE_EVAL_STAGES))
    if missing or unknown:
        problems: list[str] = []
        if missing:
            problems.append("missing: " + ", ".join(missing))
        if unknown:
            problems.append("unknown: " + ", ".join(unknown))
        raise ValueError("invalid pipeline stage runners (" + "; ".join(problems) + ")")

    outputs: dict[str, PipelineStageOutput] = {}
    for stage in PIPELINE_EVAL_STAGES:
        raw_output = stage_runners[stage](resolved_fixture, dict(outputs))
        outputs[stage] = _coerce_pipeline_stage_output(stage, raw_output)

    from openmed.eval.error_analysis import attribute_pipeline_errors

    final_facts = outputs["fhir"].facts
    fact_level = compute_fact_level_f1(resolved_fixture.gold_facts, final_facts)
    attribution = attribute_pipeline_errors(
        fixture_id=resolved_fixture.fixture_id,
        gold_facts=resolved_fixture.gold_facts,
        stage_outputs=outputs,
    )
    return PipelineFixtureEvalResult(
        fixture_id=resolved_fixture.fixture_id,
        source_hash=stable_hash(
            {
                "fixture_id": resolved_fixture.fixture_id,
                "text": resolved_fixture.text,
            }
        ),
        stage_outputs=tuple(outputs[stage] for stage in PIPELINE_EVAL_STAGES),
        fact_level=fact_level,
        attribution=attribution,
    )


def run_pipeline_eval(
    fixtures: (
        str
        | Path
        | PipelineEvalFixture
        | Mapping[str, Any]
        | Sequence[PipelineEvalFixture | Mapping[str, Any]]
    ),
    stage_runners: Mapping[str, PipelineStageRunner],
    *,
    suite: str = "pipeline-e2e",
    metadata: Mapping[str, Any] | None = None,
) -> PipelineEvalReport:
    """Run and aggregate an offline end-to-end pipeline fixture suite."""

    if isinstance(fixtures, (str, Path)):
        resolved = load_pipeline_eval_fixtures(fixtures)
    elif isinstance(fixtures, PipelineEvalFixture) or isinstance(fixtures, Mapping):
        resolved = [
            fixtures
            if isinstance(fixtures, PipelineEvalFixture)
            else PipelineEvalFixture.from_mapping(fixtures)
        ]
    else:
        resolved = [
            fixture
            if isinstance(fixture, PipelineEvalFixture)
            else PipelineEvalFixture.from_mapping(fixture)
            for fixture in fixtures
        ]
    _validate_unique_pipeline_fixture_ids(resolved)
    fixture_results = tuple(
        run_pipeline_eval_fixture(fixture, stage_runners) for fixture in resolved
    )

    from openmed.eval.error_analysis import merge_pipeline_attribution_reports

    attribution = merge_pipeline_attribution_reports(
        (result.attribution for result in fixture_results),
        suite=suite,
    )
    return PipelineEvalReport(
        suite=suite,
        fixture_results=fixture_results,
        fact_level=merge_fact_level_f1(result.fact_level for result in fixture_results),
        attribution=attribution,
        metadata=dict(metadata or {}),
    )


def _coerce_pipeline_stage_output(
    stage: str,
    value: PipelineStageOutput | Mapping[str, Any] | Iterable[Any],
) -> PipelineStageOutput:
    if isinstance(value, PipelineStageOutput):
        if value.stage != stage:
            raise ValueError(
                f"runner for {stage!r} returned output for {value.stage!r}"
            )
        return value
    metadata: Mapping[str, Any] = {}
    raw_facts: Iterable[Any]
    if isinstance(value, Mapping) and "facts" in value:
        returned_stage = str(value.get("stage") or stage)
        if returned_stage != stage:
            raise ValueError(
                f"runner for {stage!r} returned output for {returned_stage!r}"
            )
        raw_facts = value.get("facts") or ()
        raw_metadata = value.get("metadata") or {}
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("pipeline stage metadata must be a mapping")
        metadata = raw_metadata
    elif isinstance(value, Mapping):
        raw_facts = (value,)
    elif isinstance(value, (str, bytes, bytearray)):
        raise TypeError("pipeline stage runners must return fact records")
    else:
        raw_facts = value
    return PipelineStageOutput(
        stage=stage,
        facts=normalize_pipeline_facts(raw_facts),
        metadata=dict(metadata),
    )


def _validate_unique_pipeline_fixture_ids(
    fixtures: Sequence[PipelineEvalFixture],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for fixture in fixtures:
        if fixture.fixture_id in seen:
            duplicates.add(fixture.fixture_id)
        seen.add(fixture.fixture_id)
    if duplicates:
        raise ValueError(
            "duplicate pipeline fixture id(s): " + ", ".join(sorted(duplicates))
        )


def load_fixtures(path: str | Path) -> list[BenchmarkFixture]:
    """Load benchmark fixtures from a JSON or JSONL file.

    Accepted top-level shapes are either a list of fixture objects or a mapping
    containing a ``fixtures`` list. JSONL files contain one fixture object per
    non-empty line.
    """
    fixture_path = Path(path)
    if fixture_path.suffix.lower() == ".jsonl":
        fixtures = [
            BenchmarkFixture.from_mapping(json.loads(line))
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _validate_unique_fixture_ids(fixtures)
        return fixtures

    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    rows = raw.get("fixtures") if isinstance(raw, Mapping) else raw
    if not isinstance(rows, list):
        raise ValueError(
            "benchmark fixture JSON must be a list or contain a fixtures list"
        )
    fixtures = [BenchmarkFixture.from_mapping(row) for row in rows]
    _validate_unique_fixture_ids(fixtures)
    return fixtures


def load_context_multilingual_fixtures(
    path: str | Path = DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    """Load synthetic multilingual ConText assertion fixtures."""

    fixture_path = Path(path)
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or rows[0].get("kind") != "meta":
        raise ValueError("context multilingual fixture must start with a meta row")
    fixtures = tuple(row for row in rows[1:] if row.get("kind") != "meta")
    case_ids = [str(row.get("case_id", "")) for row in fixtures]
    if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("context multilingual fixtures require unique case_id values")
    return rows[0], fixtures


def run_context_multilingual_eval(
    path: str | Path = DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE,
    *,
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Score deterministic multilingual ConText axes on synthetic fixtures."""

    from openmed.clinical.context import (
        CERTAINTY_VALUES,
        NEGATION_VALUES,
        TEMPORALITY_VALUES,
        clinical_context_lexicon_stats,
        resolve_span_context,
    )

    meta, fixtures = load_context_multilingual_fixtures(path)
    labels_by_axis = {
        "negation": NEGATION_VALUES,
        "temporality": TEMPORALITY_VALUES,
        "uncertainty": CERTAINTY_VALUES,
    }
    expected_by_language: dict[str, dict[str, list[str]]] = {}
    predicted_by_language: dict[str, dict[str, list[str]]] = {}

    for row in fixtures:
        language = str(row.get("language") or "en")
        span = _context_fixture_span(row)
        context = resolve_span_context(span, language=language)
        expected = row.get("expected")
        if not isinstance(expected, Mapping):
            raise ValueError(
                f"context fixture {row.get('case_id')} lacks expected axes"
            )

        axis_predictions = {
            "negation": context.negation,
            "temporality": context.temporality,
            "uncertainty": context.certainty,
        }
        axis_expected = {
            "negation": str(expected["negation"]),
            "temporality": str(expected["temporality"]),
            "uncertainty": str(expected["certainty"]),
        }
        language_expected = expected_by_language.setdefault(language, {})
        language_predicted = predicted_by_language.setdefault(language, {})
        for axis in labels_by_axis:
            language_expected.setdefault(axis, []).append(axis_expected[axis])
            language_predicted.setdefault(axis, []).append(axis_predictions[axis])

    macro_f1 = {
        language: {
            axis: _macro_f1(
                expected_by_language[language][axis],
                predicted_by_language[language][axis],
                labels_by_axis[axis],
            )
            for axis in labels_by_axis
        }
        for language in sorted(expected_by_language)
    }
    thresholds = {"negation": 0.90, "temporality": 0.85, "uncertainty": 0.85}
    gate_passed = all(
        macro_f1[language][axis] >= thresholds[axis]
        for language in macro_f1
        for axis in thresholds
    )
    metrics = {
        "context_macro_f1": macro_f1,
        "context_thresholds": thresholds,
        "context_gate_passed": gate_passed,
        "context_lexicon_coverage": clinical_context_lexicon_stats(),
    }
    return BenchmarkReport(
        suite="context_multilingual",
        model_name="deterministic-context",
        device="local",
        fixture_count=len(fixtures),
        metrics=metrics,
        generated_at=generated_at,
        metadata={
            "fixture_ids": [str(row["case_id"]) for row in fixtures],
            "languages": sorted(expected_by_language),
            "parent_issue": "OM-724",
            "synthetic": bool(meta.get("synthetic")),
        },
    )


def load_section_multilingual_fixtures(
    path: str | Path = DEFAULT_SECTION_MULTILINGUAL_FIXTURE,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    """Load synthetic multilingual clinical section detection fixtures."""

    fixture_path = Path(path)
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or rows[0].get("kind") != "meta":
        raise ValueError("section multilingual fixture must start with a meta row")
    fixtures = tuple(row for row in rows[1:] if row.get("kind") != "meta")
    case_ids = [str(row.get("case_id", "")) for row in fixtures]
    if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("section multilingual fixtures require unique case_id values")
    return rows[0], fixtures


def run_section_multilingual_eval(
    path: str | Path = DEFAULT_SECTION_MULTILINGUAL_FIXTURE,
    *,
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Score deterministic multilingual section detection on synthetic notes."""

    from openmed.clinical.lexicons import section_lexicon_stats
    from openmed.clinical.sections import detect_sections
    from openmed.eval.section_recall import (
        compute_section_detection_metrics,
        compute_section_recall,
    )

    meta, fixtures = load_section_multilingual_fixtures(path)
    label_f1_by_language: dict[str, list[float]] = {}
    boundary_recall_by_language: dict[str, list[float]] = {}
    char_recall_by_language: dict[str, list[float]] = {}
    char_baseline_by_language: dict[str, list[float]] = {}

    for row in fixtures:
        language = str(row.get("language") or "en")
        text = str(row.get("text") or "")
        gold_sections = _section_fixture_gold_sections(row)
        predicted_sections = detect_sections(text, language=language)
        detection_metrics = compute_section_detection_metrics(
            text,
            gold_sections,
            predicted_sections,
        )
        gold_spans = _section_fixture_gold_spans(row)
        character_report = compute_section_recall(
            text,
            predicted_sections,
            gold_spans,
            gold_spans,
            default_language=language,
        )
        baseline_report = compute_section_recall(
            text,
            gold_sections,
            gold_spans,
            gold_spans,
            default_language=language,
        )

        label_f1_by_language.setdefault(language, []).append(detection_metrics.label_f1)
        boundary_recall_by_language.setdefault(language, []).append(
            detection_metrics.boundary_recall
        )
        char_recall_by_language.setdefault(language, []).append(
            character_report.overall.recall
        )
        char_baseline_by_language.setdefault(language, []).append(
            baseline_report.overall.recall
        )

    section_label_f1 = {
        language: _mean(values)
        for language, values in sorted(label_f1_by_language.items())
    }
    section_boundary_recall = {
        language: _mean(values)
        for language, values in sorted(boundary_recall_by_language.items())
    }
    section_character_recall = {
        language: _mean(values)
        for language, values in sorted(char_recall_by_language.items())
    }
    section_character_baseline = {
        language: _mean(values)
        for language, values in sorted(char_baseline_by_language.items())
    }
    label_threshold = 0.85
    gate_passed = all(
        score >= label_threshold for score in section_label_f1.values()
    ) and all(
        section_character_recall[language] >= baseline
        for language, baseline in section_character_baseline.items()
    )

    return BenchmarkReport(
        suite="section_multilingual",
        model_name="deterministic-section-detector",
        device="local",
        fixture_count=len(fixtures),
        metrics={
            "section_boundary_recall": section_boundary_recall,
            "section_character_recall": section_character_recall,
            "section_character_recall_baseline": section_character_baseline,
            "section_gate_passed": gate_passed,
            "section_label_f1": section_label_f1,
            "section_label_f1_threshold": label_threshold,
            "section_lexicon_coverage": section_lexicon_stats(),
        },
        generated_at=generated_at,
        metadata={
            "fixture_ids": [str(row["case_id"]) for row in fixtures],
            "languages": sorted(label_f1_by_language),
            "parent_issue": "OM-729",
            "synthetic": bool(meta.get("synthetic")),
        },
    )


def _context_fixture_span(row: Mapping[str, Any]) -> dict[str, Any]:
    text = str(row.get("text", ""))
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise ValueError(f"context fixture {row.get('case_id')} lacks target")
    target_text = str(target.get("text") or "")
    if not target_text:
        raise ValueError(f"context fixture {row.get('case_id')} has empty target")
    start = text.find(target_text)
    if start == -1:
        raise ValueError(
            f"context fixture {row.get('case_id')} target is absent from text"
        )
    return {
        "text": target_text,
        "context": text,
        "start": start,
        "end": start + len(target_text),
    }


def _section_fixture_gold_sections(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    from openmed.clinical.sections import UNSECTIONED_SECTION

    text = str(row.get("text") or "")
    raw_sections = row.get("gold_sections")
    if not isinstance(raw_sections, Sequence) or isinstance(raw_sections, (str, bytes)):
        raise ValueError(f"section fixture {row.get('case_id')} lacks gold sections")
    if all(
        isinstance(section, Mapping)
        and "start" in section
        and "end" in section
        and "label" in section
        for section in raw_sections
    ):
        return tuple(dict(section) for section in raw_sections)

    header_starts: list[tuple[str, str, int]] = []
    search_from = 0
    for section in raw_sections:
        if not isinstance(section, Mapping):
            raise ValueError("gold section entries must be mappings")
        label = str(section.get("label") or "")
        header = str(section.get("header") or "")
        if not label or not header:
            raise ValueError("gold section entries require label and header")
        start = text.find(header, search_from)
        if start == -1:
            raise ValueError(
                f"section fixture {row.get('case_id')} lacks header {header!r}"
            )
        header_starts.append((label, header, start))
        search_from = start + len(header)

    sections: list[dict[str, Any]] = []
    cursor = 0
    for index, (label, _header, start) in enumerate(header_starts):
        if cursor < start:
            sections.append(
                {"label": UNSECTIONED_SECTION, "start": cursor, "end": start}
            )
        end = (
            header_starts[index + 1][2] if index + 1 < len(header_starts) else len(text)
        )
        sections.append({"label": label, "start": start, "end": end})
        cursor = end
    if cursor < len(text):
        sections.append(
            {"label": UNSECTIONED_SECTION, "start": cursor, "end": len(text)}
        )
    return tuple(sections)


def _section_fixture_gold_spans(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    text = str(row.get("text") or "")
    raw_spans = row.get("gold_spans") or ()
    if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
        raise ValueError("gold_spans must be a sequence")
    spans: list[dict[str, Any]] = []
    search_from = 0
    for raw_span in raw_spans:
        if not isinstance(raw_span, Mapping):
            raise ValueError("gold span entries must be mappings")
        if "start" in raw_span and "end" in raw_span:
            spans.append(dict(raw_span))
            continue
        surface = str(raw_span.get("text") or "")
        if not surface:
            raise ValueError("gold span entries require text or offsets")
        start = text.find(surface, search_from)
        if start == -1:
            raise ValueError(f"gold span surface is absent: {surface!r}")
        search_from = start + len(surface)
        span = dict(raw_span)
        span["start"] = start
        span["end"] = start + len(surface)
        spans.append(span)
    return tuple(spans)


def _macro_f1(
    expected: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> float:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted labels must be the same length")
    scores = [
        _label_f1(expected, predicted, label)
        for label in labels
        if label in expected or label in predicted
    ]
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


def _label_f1(expected: Sequence[str], predicted: Sequence[str], label: str) -> float:
    true_positive = sum(
        1
        for gold, guess in zip(expected, predicted)
        if gold == label and guess == label
    )
    false_positive = sum(
        1
        for gold, guess in zip(expected, predicted)
        if gold != label and guess == label
    )
    false_negative = sum(
        1
        for gold, guess in zip(expected, predicted)
        if gold == label and guess != label
    )
    if true_positive == 0:
        return 0.0 if false_positive or false_negative else 1.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2 * precision * recall / (precision + recall)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    return sum(values) / len(values)


def default_model_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
    *,
    loader: Any | None = None,
) -> Iterable[Any]:
    """Run a fixture through the existing PII runtime."""
    from openmed.core.pii import extract_pii

    result = extract_pii(
        fixture.text,
        model_name=model_name,
        lang=fixture.language,
        loader=loader,
    )
    for entity in result.entities:
        metadata = dict(entity.metadata or {})
        metadata.setdefault("device", device)
        entity.metadata = metadata
    return result.entities


def run_benchmark(
    fixtures: Sequence[BenchmarkFixture],
    *,
    suite: str,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    confidence_intervals: bool = False,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
    calibration: bool = False,
    calibration_bins: int = 10,
    abstention_thresholds: Any | None = None,
    abstention_thresholds_path: str | Path | None = None,
    abstention_confidence_threshold: float = 0.0,
    abstention_target_risk: float | None = None,
    abstention_confidence_level: float | None = None,
    abstention_bootstrap_resamples: int = 0,
    abstention_seed: int = 0,
    cache_dir: str | Path | None = None,
    cache_code_hash: str | None = None,
) -> BenchmarkReport:
    """Run *model_name* over fixtures and return a benchmark report.

    When ``confidence_intervals`` is enabled (off by default to keep fast runs
    cheap), a non-parametric bootstrap over documents attaches a
    ``confidence_interval`` payload to the leakage, character recall, and exact
    and relaxed span F1 metrics. The bootstrap is deterministic for a fixed
    ``ci_seed``. Passing ``cache_dir`` opts into a local filesystem cache keyed
    by model, suite, device, fixture-set hash, and eval code hash.
    """
    _validate_unique_fixture_ids(fixtures)
    active_abstention_thresholds = _resolve_abstention_thresholds(
        abstention_thresholds,
        abstention_thresholds_path,
    )
    if cache_dir is not None:
        effective_code_hash = _abstention_cache_hash(
            cache_code_hash,
            thresholds=active_abstention_thresholds,
            thresholds_path=abstention_thresholds_path,
            confidence_threshold=abstention_confidence_threshold,
            target_risk=abstention_target_risk,
            confidence_level=abstention_confidence_level,
            bootstrap_resamples=abstention_bootstrap_resamples,
            seed=abstention_seed,
        )
        report_key = build_report_key(
            model_name=model_name,
            suite=suite,
            fixture_set_hash=hash_fixture_set(fixtures),
            code_hash=effective_code_hash,
            device=device,
        )
        return load_or_compute(
            report_key,
            lambda: run_benchmark(
                fixtures,
                suite=suite,
                model_name=model_name,
                device=device,
                runner=runner,
                generated_at=generated_at,
                metadata=metadata,
                confidence_intervals=confidence_intervals,
                ci_resamples=ci_resamples,
                ci_alpha=ci_alpha,
                ci_seed=ci_seed,
                calibration=calibration,
                calibration_bins=calibration_bins,
                abstention_thresholds=active_abstention_thresholds,
                abstention_confidence_threshold=abstention_confidence_threshold,
                abstention_target_risk=abstention_target_risk,
                abstention_confidence_level=abstention_confidence_level,
                abstention_bootstrap_resamples=abstention_bootstrap_resamples,
                abstention_seed=abstention_seed,
            ),
            cache_dir=cache_dir,
        )

    model_runner = runner or _shared_default_model_runner()
    results: list[FixtureResult] = []
    peak_rss_start = _peak_rss_bytes()

    for fixture in fixtures:
        started = time.perf_counter()
        raw_predictions = list(model_runner(fixture, model_name, device))
        latency_ms = (time.perf_counter() - started) * 1000.0
        predicted_spans = tuple(
            normalize_eval_spans(
                raw_predictions,
                default_language=fixture.language,
                default_device=device,
                source_text=fixture.text,
            )
        )
        validate_entity_spans(
            [span.to_entity() for span in predicted_spans],
            fixture.text,
        )
        results.append(
            FixtureResult(
                fixture_id=fixture.fixture_id,
                predicted_spans=predicted_spans,
                latency_ms=latency_ms,
            )
        )

    gold_spans, predicted_spans, corpus_text = _corpus_coordinates(fixtures, results)
    peak_rss_end = _peak_rss_bytes()
    rss_values = [
        value for value in (peak_rss_start, peak_rss_end) if value is not None
    ]
    peak_rss = max(rss_values) if rss_values else None
    metrics = compute_metrics_bundle(
        gold_spans,
        predicted_spans,
        latencies_ms=[result.latency_ms for result in results[1:]],
        cold_start_ms=(results[0].latency_ms if results else None),
        peak_rss_bytes=peak_rss,
        abstention_thresholds=active_abstention_thresholds,
        abstention_confidence_threshold=abstention_confidence_threshold,
        abstention_model_id=model_name,
        abstention_target_risk=abstention_target_risk,
        abstention_confidence_level=abstention_confidence_level,
        abstention_bootstrap_resamples=abstention_bootstrap_resamples,
        abstention_seed=abstention_seed,
        default_device=device,
        source_text=corpus_text,
    )
    faithfulness_metrics = [
        compute_span_grounded_faithfulness(
            facts,
            source_text=fixture.text,
            fixture_id=fixture.fixture_id,
        )
        for fixture in fixtures
        if (facts := _fixture_extracted_facts(fixture))
    ]
    if faithfulness_metrics:
        metrics["faithfulness"] = merge_faithfulness_metrics(
            faithfulness_metrics
        ).to_dict()
    if confidence_intervals:
        metrics = _attach_confidence_intervals(
            metrics,
            fixtures,
            results,
            device=device,
            n_resamples=ci_resamples,
            alpha=ci_alpha,
            seed=ci_seed,
        )
    if calibration:
        metrics = _attach_calibration_metrics(
            metrics,
            gold_spans,
            predicted_spans,
            n_bins=calibration_bins,
        )

    report_metadata = dict(metadata or {})
    report_metadata.setdefault(
        "fixture_ids", [fixture.fixture_id for fixture in fixtures]
    )
    return BenchmarkReport(
        suite=suite,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        metrics=metrics,
        generated_at=generated_at,
        metadata=report_metadata,
    )


def run_relation_benchmark(
    fixtures: Sequence[Any],
    *,
    suite: str,
    model_name: str,
    runner: RelationModelRunner,
    device: str = "cpu",
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
) -> BenchmarkReport:
    """Run a relation model over fixtures with typed span relations."""

    report, _ = _run_relation_benchmark(
        fixtures,
        suite=suite,
        model_name=model_name,
        runner=runner,
        device=device,
        generated_at=generated_at,
        metadata=metadata,
        ci_resamples=ci_resamples,
        ci_alpha=ci_alpha,
        ci_seed=ci_seed,
    )
    return report


def _run_relation_benchmark(
    fixtures: Sequence[Any],
    *,
    suite: str,
    model_name: str,
    runner: RelationModelRunner,
    device: str,
    generated_at: str | None,
    metadata: Mapping[str, Any] | None,
    ci_resamples: int,
    ci_alpha: float,
    ci_seed: int,
) -> tuple[BenchmarkReport, tuple[RelationFixtureResult, ...]]:
    """Run relation inference and retain per-fixture results for suite gates."""

    if not fixtures:
        raise ValueError("relation benchmark requires at least one fixture")
    _validate_unique_fixture_ids(fixtures)
    results: list[RelationFixtureResult] = []
    peak_rss_start = _peak_rss_bytes()

    for fixture in fixtures:
        fixture_id = str(getattr(fixture, "fixture_id"))
        text = str(getattr(fixture, "text", ""))
        started = time.perf_counter()
        raw_predictions = list(runner(fixture, model_name, device))
        latency_ms = (time.perf_counter() - started) * 1000.0
        predicted_relations = tuple(
            normalize_eval_relations(
                raw_predictions,
                entity_spans=getattr(fixture, "entities", None),
                fixture_id=fixture_id,
                default_language=str(getattr(fixture, "language", "en")),
                source_text=text,
            )
        )
        for relation in predicted_relations:
            _validate_relation_offsets(relation, text, fixture_id)
        results.append(
            RelationFixtureResult(
                fixture_id=fixture_id,
                predicted_relations=predicted_relations,
                latency_ms=latency_ms,
            )
        )

    gold_relations, predicted_relations = _relation_corpus_relations(
        fixtures,
        results,
    )
    per_document_relations = _per_document_relations(fixtures, results)
    relation_metrics = compute_relation_metrics_bundle(
        gold_relations,
        predicted_relations,
    )
    relation_intervals = compute_relation_confidence_intervals(
        per_document_relations,
        n_resamples=ci_resamples,
        alpha=ci_alpha,
        seed=ci_seed,
    )
    for key, interval in relation_intervals.items():
        metric = relation_metrics.get(key)
        if isinstance(metric, Mapping):
            relation_metrics[key] = {**metric, "confidence_interval": interval}

    peak_rss_end = _peak_rss_bytes()
    rss_values = [
        value for value in (peak_rss_start, peak_rss_end) if value is not None
    ]
    peak_rss = max(rss_values) if rss_values else None
    metrics: dict[str, Any] = {
        "latency": {
            **compute_latency_summary(
                [result.latency_ms for result in results[1:]]
            ).to_dict(),
            "cold_start_ms": results[0].latency_ms if results else None,
        },
        "relation_extraction": relation_metrics,
        "resources": compute_resource_metrics(peak_rss_bytes=peak_rss).to_dict(),
    }
    metrics["strict_relation_f1"] = relation_metrics["strict"]
    metrics["relaxed_relation_f1"] = relation_metrics["relaxed"]
    metrics["per_relation_type_re_f1"] = relation_metrics["per_relation_type"]
    metrics["per_language_relation_f1"] = relation_metrics["per_language"]

    report_metadata = dict(metadata or {})
    report_metadata.setdefault(
        "fixture_ids",
        [str(getattr(fixture, "fixture_id")) for fixture in fixtures],
    )
    report_metadata.setdefault("task", "relation")
    report_metadata.setdefault(
        "relation_types",
        sorted({relation.relation_type for relation in gold_relations}),
    )
    report_metadata.setdefault(
        "languages",
        sorted({relation.head.language for relation in gold_relations}),
    )
    return (
        BenchmarkReport(
            suite=suite,
            model_name=model_name,
            device=device,
            fixture_count=len(fixtures),
            metrics=metrics,
            generated_at=generated_at,
            metadata=report_metadata,
        ),
        tuple(results),
    )


def run_relation_suite(
    fixture_path: str | Path | None = None,
    *,
    model_name: str,
    runner: RelationModelRunner,
    suite: str = "relations",
    device: str = "cpu",
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    family: str = "Relation",
    baseline: Mapping[str, Any] | None = None,
    baseline_path: str | Path | None = None,
    signing_key: bytes | str | None = None,
    key_id: str = "relation-scorecard",
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
) -> RelationScorecard:
    """Run synthetic relation gold, sign its scorecard, and fail closed.

    A failing gate is still serialized to requested output paths before
    :class:`RelationGateFailure` is raised, so automation can archive the
    signed failure evidence without treating the run as successful.
    """

    from openmed.core.baseline import (
        BASELINE_PATH,
        BaselineError,
        load_baseline_store,
    )
    from openmed.eval.release_gates import evaluate_relation_golden_regression_gate
    from openmed.eval.suites.relations import (
        DEFAULT_RELATION_GOLD_PATH,
        RELATION_GOLD_SCHEMA_VERSION,
        load_relation_fixtures,
        relation_suite_metadata,
        relation_trap_summary,
    )

    resolved_path = (
        Path(fixture_path) if fixture_path is not None else DEFAULT_RELATION_GOLD_PATH
    )
    fixtures = load_relation_fixtures(resolved_path)
    report, results = _run_relation_benchmark(
        fixtures,
        suite=suite,
        model_name=model_name,
        runner=runner,
        device=device,
        generated_at=generated_at,
        metadata=metadata,
        ci_resamples=ci_resamples,
        ci_alpha=ci_alpha,
        ci_seed=ci_seed,
    )

    fixture_set_hash = _sha256_path(resolved_path)
    fixture_hashes = _relation_fixture_hashes(fixtures)
    trap_summary = relation_trap_summary(fixtures)
    trap_leaks = _relation_trap_leak_summary(fixtures, results)
    consistency = _relation_consistency_scores(trap_leaks)
    relation_metrics = _mapping_value(report.metrics.get("relation_extraction"))
    by_type = _mapping_value(relation_metrics.get("per_relation_type"))
    relation_golden = {
        "by_type": by_type,
        "consistency": consistency,
        "fixture_set_hash": fixture_set_hash,
        "trap_leaks": trap_leaks,
    }
    report_metrics = {
        **dict(report.metrics),
        "relation_golden": relation_golden,
    }
    report_metadata = {
        **dict(report.metadata),
        "family": family,
        "fixture_hashes": fixture_hashes,
        "fixture_set_hash": fixture_set_hash,
        "relation_golden_regression_required": True,
        "relation_suite": relation_suite_metadata(),
        "relation_trap_leaks": trap_leaks,
        "relation_traps": trap_summary,
    }
    report = replace(report, metrics=report_metrics, metadata=report_metadata)

    resolved_baseline = baseline
    if resolved_baseline is None:
        try:
            resolved_baseline = load_baseline_store(baseline_path or BASELINE_PATH)
        except (OSError, BaselineError, json.JSONDecodeError):
            resolved_baseline = {}
    gate = evaluate_relation_golden_regression_gate(
        report.metrics,
        resolved_baseline,
        family=family,
        metadata=report.metadata,
    )
    scorecard_metrics = {
        "by_language": _mapping_value(relation_metrics.get("per_language")),
        "by_scope": _mapping_value(relation_metrics.get("by_scope")),
        "by_type": by_type,
        "consistency": consistency,
        "counts": {
            "gold": int(relation_metrics.get("gold_relation_count") or 0),
            "predicted": int(relation_metrics.get("predicted_relation_count") or 0),
        },
        "relaxed": _mapping_value(relation_metrics.get("relaxed")),
        "strict": _mapping_value(relation_metrics.get("strict")),
        "trap_leaks": trap_leaks,
    }
    provenance = {
        "fixture_hashes": fixture_hashes,
        "fixture_schema_version": RELATION_GOLD_SCHEMA_VERSION,
        "fixture_set_hash": fixture_set_hash,
        "relation_suite": relation_suite_metadata(),
        "trap_summary": trap_summary,
    }
    active_signing_key = (
        signing_key
        if signing_key is not None
        else os.environ.get(
            "OPENMED_RELATION_SCORECARD_KEY",
            _DEFAULT_RELATION_SCORECARD_SIGNING_KEY,
        )
    )
    scorecard = RelationScorecard(
        suite=report.suite,
        model_name=report.model_name,
        device=report.device,
        fixture_count=report.fixture_count,
        metrics=scorecard_metrics,
        provenance=provenance,
        gate_result=gate.to_dict(),
        gate_passed=gate.passed,
        generated_at=report.generated_at,
    ).sign(active_signing_key, key_id=key_id)
    if output_json is not None:
        scorecard.write_json(output_json)
    if output_markdown is not None:
        scorecard.write_markdown(output_markdown)
    if not scorecard.gate_passed:
        raise RelationGateFailure(scorecard)
    return scorecard


def run_suite(
    fixture_path: str | Path,
    *,
    suite: str,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    confidence_intervals: bool = False,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
    calibration: bool = False,
    calibration_bins: int = 10,
    abstention_thresholds: Any | None = None,
    abstention_thresholds_path: str | Path | None = None,
    abstention_confidence_threshold: float = 0.0,
    abstention_target_risk: float | None = None,
    abstention_confidence_level: float | None = None,
    abstention_bootstrap_resamples: int = 0,
    abstention_seed: int = 0,
    cache_dir: str | Path | None = None,
    cache_code_hash: str | None = None,
) -> BenchmarkReport | RelationScorecard:
    """Load fixtures, run the benchmark, and optionally write reports."""

    if _is_relation_suite(suite):
        if runner is None:
            raise ValueError("relation suite requires an explicit relation runner")
        return run_relation_suite(
            fixture_path,
            suite=suite,
            model_name=model_name,
            device=device,
            runner=runner,
            output_json=output_json,
            output_markdown=output_markdown,
            generated_at=generated_at,
            metadata=metadata,
            ci_resamples=ci_resamples,
            ci_alpha=ci_alpha,
            ci_seed=ci_seed,
        )
    report = run_benchmark(
        load_fixtures(fixture_path),
        suite=suite,
        model_name=model_name,
        device=device,
        runner=runner,
        generated_at=generated_at,
        metadata=metadata,
        confidence_intervals=confidence_intervals,
        ci_resamples=ci_resamples,
        ci_alpha=ci_alpha,
        ci_seed=ci_seed,
        calibration=calibration,
        calibration_bins=calibration_bins,
        abstention_thresholds=abstention_thresholds,
        abstention_thresholds_path=abstention_thresholds_path,
        abstention_confidence_threshold=abstention_confidence_threshold,
        abstention_target_risk=abstention_target_risk,
        abstention_confidence_level=abstention_confidence_level,
        abstention_bootstrap_resamples=abstention_bootstrap_resamples,
        abstention_seed=abstention_seed,
        cache_dir=cache_dir,
        cache_code_hash=cache_code_hash,
    )
    if output_json is not None:
        report.write_json(output_json)
    if output_markdown is not None:
        report.write_markdown(output_markdown)
    return report


def run_multilingual_ner_scorecard(
    fixtures: Sequence[BenchmarkFixture],
    *,
    suite: str = "multilingual-clinical-ner",
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    min_exact_span_f1: float = 0.85,
    training_manifest_path: str | Path | None = None,
) -> BenchmarkReport:
    """Run multilingual clinical NER fixtures and emit sliced F1 scorecards.

    The scorecard groups fixtures by benchmark and language, then computes
    exact and relaxed span F1 using the shared metrics module. When a training
    manifest is supplied, the report includes a PHI-free overlap check and the
    gate fails if any eval fixture appears in the training inputs.
    """

    if not fixtures:
        raise ValueError("multilingual NER scorecard requires at least one fixture")
    _validate_unique_fixture_ids(fixtures)
    model_runner = runner or _shared_default_model_runner()
    results = _run_fixture_predictions(
        fixtures,
        model_runner=model_runner,
        model_name=model_name,
        device=device,
    )
    by_benchmark_language = _multilingual_group_rows(
        fixtures,
        results,
        group_keys=("benchmark", "language"),
    )
    per_benchmark = _multilingual_group_rows(
        fixtures,
        results,
        group_keys=("benchmark",),
    )
    per_language = _multilingual_group_rows(
        fixtures,
        results,
        group_keys=("language",),
    )
    overlap_findings = (
        check_training_manifest_overlap(fixtures, training_manifest_path)
        if training_manifest_path is not None
        else ()
    )
    gate_failures = _multilingual_gate_failures(
        by_benchmark_language,
        min_exact_span_f1=min_exact_span_f1,
        overlap_findings=overlap_findings,
    )
    report_metadata = dict(metadata or {})
    report_metadata.setdefault(
        "fixture_ids", [fixture.fixture_id for fixture in fixtures]
    )
    report_metadata.setdefault(
        "unmapped_labels", _unmapped_labels_by_benchmark(fixtures)
    )
    metrics = {
        "gate": {
            "failures": gate_failures,
            "min_exact_span_f1": min_exact_span_f1,
            "passed": not gate_failures,
        },
        "per_benchmark": per_benchmark,
        "per_benchmark_language": by_benchmark_language,
        "per_language": per_language,
        "scorecard": by_benchmark_language,
        "train_eval_overlap": {
            "finding_count": len(overlap_findings),
            "findings": [finding.to_dict() for finding in overlap_findings],
            "manifest_path_hash": (
                _hash_path(training_manifest_path)
                if training_manifest_path is not None
                else None
            ),
            "passed": not overlap_findings,
        },
        "unmapped_labels": _unmapped_labels_by_benchmark(fixtures),
    }
    return BenchmarkReport(
        suite=suite,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        metrics=metrics,
        generated_at=generated_at,
        metadata=report_metadata,
    )


def run_masakhaner_scorecard(
    fixtures: Sequence[BenchmarkFixture],
    *,
    suite: str = "masakhaner",
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    languages: Sequence[str] | None = None,
    checkpoint_path: str | Path | None = None,
) -> BenchmarkReport:
    """Run an inference-only MasakhaNER scorecard sliced by language.

    ``checkpoint_path`` is an optional caller-supplied local multilingual
    encoder checkpoint. It is passed to the configured runner as the model
    name and is never downloaded, trained, or modified by the harness.
    """

    if not fixtures:
        raise ValueError("MasakhaNER scorecard requires at least one fixture")
    selected = list(fixtures)
    requested_languages: tuple[str, ...] | None = None
    if languages is not None:
        language_values = (languages,) if isinstance(languages, str) else languages
        requested_languages = tuple(
            dict.fromkeys(str(language).strip().lower() for language in language_values)
        )
        selected = [
            fixture
            for fixture in fixtures
            if fixture.language.lower() in requested_languages
        ]
        present = {fixture.language.lower() for fixture in selected}
        missing = [
            language for language in requested_languages if language not in present
        ]
        if missing:
            raise ValueError(
                "MasakhaNER fixtures missing requested language(s): "
                + ", ".join(missing)
            )
    if not selected:
        raise ValueError("MasakhaNER language selection produced no fixtures")

    effective_model_name = model_name
    checkpoint_hash: str | None = None
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"user-supplied MasakhaNER checkpoint does not exist: {checkpoint}"
            )
        checkpoint = checkpoint.resolve()
        effective_model_name = str(checkpoint)
        checkpoint_hash = _hash_path(checkpoint)

    _validate_unique_fixture_ids(selected)
    model_runner = runner or _shared_default_model_runner()
    results = _run_fixture_predictions(
        selected,
        model_runner=model_runner,
        model_name=effective_model_name,
        device=device,
    )
    per_language = _multilingual_group_rows(
        selected,
        results,
        group_keys=("language",),
    )
    overall_rows = _multilingual_group_rows(
        selected,
        results,
        group_keys=("benchmark",),
    )
    report_metadata = dict(metadata or {})
    report_metadata.update(
        {
            "checkpoint_mode": (
                "user-supplied-local" if checkpoint_path is not None else "configured"
            ),
            "checkpoint_path_hash": checkpoint_hash,
            "configured_model_name": model_name,
            "inference_only": True,
            "languages": [row["language"] for row in per_language],
            "license_ids": sorted(
                {
                    str(fixture.metadata.get("license_id"))
                    for fixture in selected
                    if fixture.metadata.get("license_id")
                }
            ),
            "training_performed": False,
        }
    )
    metrics = {
        "overall": overall_rows[0] if overall_rows else {},
        "per_language": per_language,
        "scorecard": per_language,
    }
    return BenchmarkReport(
        suite=suite,
        model_name=effective_model_name,
        device=device,
        fixture_count=len(selected),
        metrics=metrics,
        generated_at=generated_at,
        metadata=report_metadata,
    )


def check_training_manifest_overlap(
    fixtures: Sequence[BenchmarkFixture],
    manifest_path: str | Path,
) -> tuple[TrainingEvalOverlapFinding, ...]:
    """Flag PHI-free train/eval overlap against a JSON or JSONL manifest.

    The checker compares text hashes and hashed record identifiers. Raw text,
    spans, and manifest paths are not returned.
    """

    rows = _load_manifest_rows(manifest_path)
    manifest_keys: dict[str, tuple[int, str]] = {}
    for line_number, row in rows:
        row_hash = _manifest_row_hash(row, line_number)
        for key in _manifest_overlap_keys(row):
            manifest_keys.setdefault(key, (line_number, row_hash))

    findings: list[TrainingEvalOverlapFinding] = []
    seen: set[tuple[str, str]] = set()
    for fixture in fixtures:
        benchmark = str(
            fixture.metadata.get("benchmark") or fixture.metadata.get("dataset") or ""
        )
        split = str(fixture.metadata.get("split") or "")
        for key in _fixture_overlap_keys(fixture):
            match = manifest_keys.get(key)
            if match is None:
                continue
            dedupe_key = (fixture.fixture_id, key)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            line_number, row_hash = match
            findings.append(
                TrainingEvalOverlapFinding(
                    fixture_id=fixture.fixture_id,
                    benchmark=benchmark,
                    language=fixture.language,
                    split=split,
                    overlap_key=key,
                    manifest_row_hash=row_hash,
                    manifest_line=line_number,
                )
            )
    return tuple(findings)


def run_cross_lingual_transfer(
    fixtures: Sequence[BenchmarkFixture],
    *,
    suite: str,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    languages: Sequence[str] | None = None,
    leakage_floors: Mapping[str, float] | None = None,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
) -> Any:
    """Run the cross-lingual transfer matrix over benchmark fixtures.

    The returned report is PHI-free and byte-stable. Its source-language
    calibration context is passed to the runner through fixture metadata.
    """
    from openmed.eval.fairness import cross_lingual_transfer_report

    report = cross_lingual_transfer_report(
        model_name,
        fixtures,
        runner=runner,
        device=device,
        languages=languages,
        leakage_floors=leakage_floors,
        ci_resamples=ci_resamples,
        ci_alpha=ci_alpha,
        ci_seed=ci_seed,
    )
    report = replace(report, suite=suite)
    if output_json is not None:
        report.write_json(output_json)
    if output_markdown is not None:
        report.write_markdown(output_markdown)
    return report


def run_cross_lingual_transfer_suite(
    fixture_path: str | Path,
    *,
    suite: str,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    languages: Sequence[str] | None = None,
    leakage_floors: Mapping[str, float] | None = None,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
) -> Any:
    """Load fixtures and run a cross-lingual transfer-matrix report."""
    return run_cross_lingual_transfer(
        load_fixtures(fixture_path),
        suite=suite,
        model_name=model_name,
        device=device,
        runner=runner,
        output_json=output_json,
        output_markdown=output_markdown,
        languages=languages,
        leakage_floors=leakage_floors,
        ci_resamples=ci_resamples,
        ci_alpha=ci_alpha,
        ci_seed=ci_seed,
    )


def run_federated_leakage_eval(
    fixtures: Sequence[BenchmarkFixture],
    *,
    detector: FederatedDetectorSpec | str | Path,
    suite: str = "federated",
    detector_name: str | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    max_boundary_leakage_rate: float = 0.0,
    side_channel_threshold_bits: float = 0.30,
    side_channel_min_samples: int = 4,
    signing_key: bytes | str | None = None,
    key_id: str = "federated-eval",
    work_dir: str | Path | None = None,
) -> FederatedEvalReport:
    """Run an untrusted detector out-of-process and gate boundary leakage.

    The detector protocol is intentionally narrow: the child receives a JSON
    fixture path and output path through environment variables, writes
    predictions to that output JSON, and any stdout/stderr/side files are
    treated as monitored egress. Gold spans are never passed to the child.
    """
    from openmed.eval.attacks.reid import probe_span_timing_side_channel

    _validate_unique_fixture_ids(fixtures)
    spec = _coerce_federated_spec(detector)
    script_path = Path(spec.script_path).expanduser().resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"detector script does not exist: {script_path}")

    temp_root = Path(
        tempfile.mkdtemp(
            prefix="openmed-federated-",
            dir=str(work_dir) if work_dir is not None else None,
        )
    )
    wrapper_path = temp_root / "federated_child.py"
    wrapper_path.write_text(_FEDERATED_CHILD_WRAPPER, encoding="utf-8")

    runs: list[_FederatedFixtureRun] = []
    try:
        for fixture in fixtures:
            runs.append(
                _run_federated_fixture(
                    fixture,
                    spec=spec,
                    script_path=script_path,
                    wrapper_path=wrapper_path,
                    root=temp_root,
                )
            )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    leakage = _scan_boundary_egress(fixtures, runs)
    timing_records = [record for run in runs for record in run.timing_records]
    side_channel = probe_span_timing_side_channel(
        fixtures,
        timing_records,
        threshold_bits=side_channel_threshold_bits,
        min_samples=side_channel_min_samples,
    )
    sandbox_violations = tuple(
        violation for run in runs for violation in run.sandbox_violations
    )
    gate_passed = (
        leakage.rate <= max_boundary_leakage_rate
        and not side_channel.flagged
        and not sandbox_violations
    )
    report = FederatedEvalReport(
        suite=suite,
        detector_name=detector_name or script_path.stem,
        fixture_count=len(fixtures),
        boundary_leakage=leakage,
        side_channel=side_channel,
        sandbox_violations=sandbox_violations,
        resource_accounting=_resource_accounting(runs),
        gate_passed=gate_passed,
        generated_at=generated_at,
        metadata={
            **dict(metadata or {}),
            "protocol": "openmed.federated-detector.v1",
            "detector_path_hash": _path_hash(script_path),
        },
    )
    key = signing_key or os.environ.get(
        "OPENMED_FEDERATED_EVAL_KEY",
        _DEFAULT_FEDERATED_SIGNING_KEY,
    )
    return report.sign(key, key_id=key_id)


def _coerce_federated_spec(
    detector: FederatedDetectorSpec | str | Path,
) -> FederatedDetectorSpec:
    if isinstance(detector, FederatedDetectorSpec):
        return detector
    return FederatedDetectorSpec(script_path=detector)


def _run_fixture_predictions(
    fixtures: Sequence[BenchmarkFixture],
    *,
    model_runner: ModelRunner,
    model_name: str,
    device: str,
) -> list[FixtureResult]:
    results: list[FixtureResult] = []
    for fixture in fixtures:
        started = time.perf_counter()
        raw_predictions = list(model_runner(fixture, model_name, device))
        latency_ms = (time.perf_counter() - started) * 1000.0
        predicted_spans = tuple(
            normalize_eval_spans(
                raw_predictions,
                default_language=fixture.language,
                default_device=device,
                source_text=fixture.text,
            )
        )
        validate_entity_spans(
            [span.to_entity() for span in predicted_spans],
            fixture.text,
        )
        results.append(
            FixtureResult(
                fixture_id=fixture.fixture_id,
                predicted_spans=predicted_spans,
                latency_ms=latency_ms,
            )
        )
    return results


def _multilingual_group_rows(
    fixtures: Sequence[BenchmarkFixture],
    results: Sequence[FixtureResult],
    *,
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[BenchmarkFixture]] = defaultdict(list)
    for fixture in fixtures:
        grouped[_fixture_group_key(fixture, group_keys)].append(fixture)

    result_by_id = {result.fixture_id: result for result in results}
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_fixtures = grouped[key]
        group_results = [
            result_by_id[fixture.fixture_id]
            for fixture in group_fixtures
            if fixture.fixture_id in result_by_id
        ]
        gold, predicted, text = _corpus_coordinates(group_fixtures, group_results)
        exact = compute_exact_span_f1(gold, predicted, source_text=text)
        relaxed = compute_relaxed_span_f1(gold, predicted, source_text=text)
        row = {
            "exact_span_f1": exact.to_dict(),
            "f1": exact.f1,
            "fixture_count": len(group_fixtures),
            "precision": exact.precision,
            "recall": exact.recall,
            "relaxed_span_f1": relaxed.to_dict(),
            "span_count": sum(len(fixture.gold_spans) for fixture in group_fixtures),
        }
        for name, value in zip(group_keys, key, strict=True):
            row[name] = value
        rows.append(row)
    return rows


def _fixture_group_key(
    fixture: BenchmarkFixture,
    group_keys: tuple[str, ...],
) -> tuple[str, ...]:
    values: list[str] = []
    for key in group_keys:
        if key == "benchmark":
            values.append(
                str(
                    fixture.metadata.get("benchmark")
                    or fixture.metadata.get("dataset")
                    or "unknown"
                )
            )
        elif key == "language":
            values.append(str(fixture.language))
        else:
            values.append(str(fixture.metadata.get(key) or "unknown"))
    return tuple(values)


def _multilingual_gate_failures(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_exact_span_f1: float,
    overlap_findings: Sequence[TrainingEvalOverlapFinding],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        exact = row.get("exact_span_f1")
        f1 = exact.get("f1") if isinstance(exact, Mapping) else None
        if f1 is not None and float(f1) < min_exact_span_f1:
            failures.append(
                {
                    "benchmark": row.get("benchmark"),
                    "f1": float(f1),
                    "language": row.get("language"),
                    "reason": "exact_span_f1_below_threshold",
                    "threshold": min_exact_span_f1,
                }
            )
    if overlap_findings:
        failures.append(
            {
                "finding_count": len(overlap_findings),
                "reason": "train_eval_overlap",
            }
        )
    return failures


def _unmapped_labels_by_benchmark(
    fixtures: Sequence[BenchmarkFixture],
) -> dict[str, list[str]]:
    labels: defaultdict[str, set[str]] = defaultdict(set)
    for fixture in fixtures:
        benchmark = str(
            fixture.metadata.get("benchmark") or fixture.metadata.get("dataset") or ""
        )
        metadata_labels = fixture.metadata.get("unmapped_labels") or ()
        if isinstance(metadata_labels, str):
            labels[benchmark].add(metadata_labels)
        else:
            labels[benchmark].update(str(label) for label in metadata_labels)
        for span in fixture.gold_spans:
            if span.metadata.get("unmapped_label"):
                labels[benchmark].add(str(span.metadata.get("source_label") or ""))
    return {
        benchmark: sorted(label for label in values if label)
        for benchmark, values in sorted(labels.items())
        if values
    }


def _load_manifest_rows(path: str | Path) -> list[tuple[int, Mapping[str, Any]]]:
    manifest_path = Path(path)
    if manifest_path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: list[tuple[int, Mapping[str, Any]]] = []
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append((line_number, payload))
        return rows

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        values = (
            raw.get("records")
            or raw.get("documents")
            or raw.get("rows")
            or raw.get("examples")
            or [raw]
        )
    else:
        values = raw
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"training manifest must contain row objects: {path}")
    return [
        (index, row)
        for index, row in enumerate(values, start=1)
        if isinstance(row, Mapping)
    ]


def _manifest_overlap_keys(row: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key in (
        "content_hash",
        "document_hash",
        "sha256",
        "source_hash",
        "text_hash",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            keys.add(_hash_key(value))
    text = row.get("text") or row.get("document_text") or row.get("note_text")
    if isinstance(text, str) and text:
        keys.add(_hash_text(text))
    for key in ("fixture_id", "id", "record_id", "source_record_id"):
        value = row.get(key)
        if value is not None:
            keys.add(_id_key(str(value)))
    return keys


def _fixture_overlap_keys(fixture: BenchmarkFixture) -> set[str]:
    keys = {_hash_text(fixture.text), _id_key(fixture.fixture_id)}
    text_hash = fixture.metadata.get("text_hash")
    if isinstance(text_hash, str) and text_hash.strip():
        keys.add(_hash_key(text_hash))
    for key in ("source_record_id", "record_id", "document_id"):
        value = fixture.metadata.get(key)
        if value is not None:
            keys.add(_id_key(str(value)))
    return keys


def _manifest_row_hash(row: Mapping[str, Any], line_number: int) -> str:
    stable_row = {
        str(key): value
        for key, value in row.items()
        if str(key).lower()
        not in {"text", "document_text", "note", "note_text", "raw_text"}
    }
    if not stable_row:
        stable_row = {"line": line_number}
    return _hash_bytes(
        json.dumps(stable_row, sort_keys=True, default=str).encode("utf-8")
    )


def _hash_key(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned.startswith("sha256:"):
        return cleaned
    if re.fullmatch(r"[0-9a-f]{64}", cleaned):
        return f"sha256:{cleaned}"
    return _hash_text(value)


def _hash_text(value: str) -> str:
    return _hash_bytes(value.encode("utf-8"))


def _id_key(value: str) -> str:
    return "id:" + _hash_text(value)


def _hash_path(path: str | Path) -> str:
    return _hash_bytes(str(Path(path).expanduser().resolve()).encode("utf-8"))


def _run_federated_fixture(
    fixture: BenchmarkFixture,
    *,
    spec: FederatedDetectorSpec,
    script_path: Path,
    wrapper_path: Path,
    root: Path,
) -> _FederatedFixtureRun:
    run_dir = root / _slug_fixture_id(fixture.fixture_id)
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "input.json"
    output_path = output_dir / "detector_output.json"
    violation_log = run_dir / "sandbox_violations.jsonl"
    input_path.write_text(
        _canonical_json(
            {
                "fixture_id": fixture.fixture_id,
                "language": fixture.language,
                "metadata": _plain(fixture.metadata),
                "text": fixture.text,
            }
        ),
        encoding="utf-8",
    )

    command = [
        str(spec.python_executable),
        str(wrapper_path),
        str(script_path),
        str(input_path),
        str(output_path),
        str(output_dir),
        str(violation_log),
        _path_list([script_path.parent, *spec.read_roots]),
    ]
    started = time.perf_counter()
    stdout = b""
    stderr = b""
    exit_code = 0
    timeout_violation: SandboxViolation | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=output_dir,
            env=_sandbox_env(spec, input_path, output_path, output_dir),
            capture_output=True,
            timeout=spec.timeout_s,
            check=False,
        )
        stdout = _bytes_output(completed.stdout)
        stderr = _bytes_output(completed.stderr)
        exit_code = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        stdout = _bytes_output(exc.stdout)
        stderr = _bytes_output(exc.stderr)
        exit_code = -9
        timeout_violation = SandboxViolation(
            fixture_id=fixture.fixture_id,
            kind="timeout",
            event="process.timeout",
            operation="terminated",
            detail=f"detector exceeded {spec.timeout_s:.3f}s timeout",
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    artifacts = list(_collect_artifacts(output_dir, stdout=stdout, stderr=stderr))
    payload = _read_detector_payload(output_path)
    predicted_spans = tuple(
        normalize_eval_spans(
            _extract_detector_spans(payload),
            default_language=fixture.language,
            default_device="federated-subprocess",
            source_text=fixture.text,
        )
    )
    validate_entity_spans([span.to_entity() for span in predicted_spans], fixture.text)
    timing_records = tuple(
        _extract_timing_records(payload, fixture_id=fixture.fixture_id)
    )
    violations = list(_read_sandbox_violations(violation_log, fixture.fixture_id))
    if timeout_violation is not None:
        violations.append(timeout_violation)
    if exit_code != 0 and not violations:
        violations.append(
            SandboxViolation(
                fixture_id=fixture.fixture_id,
                kind="process",
                event="process.exit",
                operation="nonzero_exit",
                detail=f"detector exited with status {exit_code}",
            )
        )
    return _FederatedFixtureRun(
        fixture_id=fixture.fixture_id,
        predicted_spans=predicted_spans,
        timing_records=timing_records,
        artifacts=tuple(artifacts),
        sandbox_violations=tuple(violations),
        elapsed_ms=elapsed_ms,
        exit_code=exit_code,
    )


def _sandbox_env(
    spec: FederatedDetectorSpec,
    input_path: Path,
    output_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OPENMED_DETECTOR_INPUT": str(input_path),
        "OPENMED_DETECTOR_OUTPUT": str(output_path),
        "OPENMED_DETECTOR_OUTPUT_DIR": str(output_dir),
    }
    env.update({str(key): str(value) for key, value in spec.env.items()})
    return env


def _collect_artifacts(
    output_dir: Path,
    *,
    stdout: bytes,
    stderr: bytes,
) -> Iterable[_CapturedArtifact]:
    yield _CapturedArtifact("stdout", "stdout", stdout)
    yield _CapturedArtifact("stderr", "stderr", stderr)
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir).as_posix()
        artifact_id = f"file:{_hash_bytes(relative.encode('utf-8'))}"
        yield _CapturedArtifact("file_path", artifact_id, relative.encode("utf-8"))
        try:
            content = path.read_bytes()
        except OSError:
            content = b""
        yield _CapturedArtifact("file", artifact_id, content)


def _read_detector_payload(output_path: Path) -> Any:
    if not output_path.exists():
        return {"spans": [], "timings": []}
    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"spans": [], "timings": []}


def _extract_detector_spans(payload: Any) -> Sequence[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("spans", "predictions", "entities"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes, bytearray),
            ):
                return value
    return ()


def _extract_timing_records(
    payload: Any,
    *,
    fixture_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        for key in ("timings", "timing_records", "span_timings"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes, bytearray),
            ):
                records.extend(_timing_record(item, fixture_id) for item in value)
    for span in _extract_detector_spans(payload):
        data = span if isinstance(span, Mapping) else vars(span)
        metadata = data.get("metadata") if isinstance(data, Mapping) else None
        if not isinstance(metadata, Mapping) or "duration_ms" not in metadata:
            continue
        records.append(
            _timing_record(
                {
                    "start": data.get("start"),
                    "end": data.get("end"),
                    "duration_ms": metadata.get("duration_ms"),
                },
                fixture_id,
            )
        )
    return [record for record in records if record]


def _timing_record(value: Any, fixture_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    record = {
        "fixture_id": fixture_id,
        "start": value.get("start"),
        "end": value.get("end"),
        "duration_ms": value.get("duration_ms", value.get("elapsed_ms")),
    }
    if value.get("label") is not None:
        record["label"] = str(value["label"])
    return record


def _read_sandbox_violations(
    violation_log: Path,
    fixture_id: str,
) -> Iterable[SandboxViolation]:
    if not violation_log.exists():
        return ()
    violations: list[SandboxViolation] = []
    for line in violation_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        violations.append(
            SandboxViolation(
                fixture_id=fixture_id,
                kind=str(payload.get("kind") or "sandbox"),
                event=str(payload.get("event") or ""),
                operation=str(payload.get("operation") or "blocked"),
                path_hash=(
                    str(payload["path_hash"])
                    if payload.get("path_hash") is not None
                    else None
                ),
                detail=str(payload.get("detail") or ""),
            )
        )
    return tuple(violations)


def _scan_boundary_egress(
    fixtures: Sequence[BenchmarkFixture],
    runs: Sequence[_FederatedFixtureRun],
) -> BoundaryLeakageResult:
    surfaces = _gold_surfaces(fixtures)
    total_phi_bytes = sum(len(surface["bytes"]) for surface in surfaces)
    leaked_keys: set[tuple[str, int, int, str]] = set()
    findings: list[BoundaryLeakageFinding] = []
    emitted_by_sink: dict[str, int] = {}

    for run in runs:
        for artifact in run.artifacts:
            emitted_by_sink[artifact.sink] = emitted_by_sink.get(
                artifact.sink, 0
            ) + len(artifact.content)
            for surface in surfaces:
                offsets = tuple(_find_all(artifact.content, surface["bytes"]))
                if not offsets:
                    continue
                key = (
                    str(surface["fixture_id"]),
                    int(surface["start"]),
                    int(surface["end"]),
                    str(surface["label"]),
                )
                leaked_keys.add(key)
                findings.append(
                    BoundaryLeakageFinding(
                        fixture_id=str(surface["fixture_id"]),
                        sink=artifact.sink,
                        artifact=artifact.artifact,
                        start=int(surface["start"]),
                        end=int(surface["end"]),
                        label=str(surface["label"]),
                        length=int(surface["length"]),
                        text_hash=str(surface["text_hash"]),
                        byte_offsets=offsets,
                    )
                )

    leaked_bytes = sum(
        len(surface["bytes"])
        for surface in surfaces
        if (
            str(surface["fixture_id"]),
            int(surface["start"]),
            int(surface["end"]),
            str(surface["label"]),
        )
        in leaked_keys
    )
    rate = leaked_bytes / total_phi_bytes if total_phi_bytes else 0.0
    return BoundaryLeakageResult(
        rate=rate,
        leaked_bytes=leaked_bytes,
        total_phi_bytes=total_phi_bytes,
        findings=tuple(findings),
        emitted_bytes_by_sink=emitted_by_sink,
    )


def _gold_surfaces(fixtures: Sequence[BenchmarkFixture]) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for fixture in fixtures:
        for span in fixture.gold_spans:
            if not (0 <= span.start < span.end <= len(fixture.text)):
                continue
            surface = fixture.text[span.start : span.end]
            if not surface:
                continue
            key = (fixture.fixture_id, span.start, span.end, span.label)
            if key in seen:
                continue
            seen.add(key)
            evidence = hashed_span_surface(
                fixture.text,
                span.start,
                span.end,
                label=span.label,
            )
            surfaces.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "label": span.label,
                    "bytes": surface.encode("utf-8"),
                    **evidence,
                }
            )
    return surfaces


def _find_all(content: bytes, needle: bytes) -> list[int]:
    if not needle:
        return []
    offsets: list[int] = []
    start = 0
    while True:
        index = content.find(needle, start)
        if index < 0:
            break
        offsets.append(index)
        start = index + max(len(needle), 1)
    return offsets


def _resource_accounting(runs: Sequence[_FederatedFixtureRun]) -> dict[str, Any]:
    return {
        "process_count": len(runs),
        "total_elapsed_ms": sum(run.elapsed_ms for run in runs),
        "max_fixture_elapsed_ms": max((run.elapsed_ms for run in runs), default=0.0),
        "exit_codes": [run.exit_code for run in runs],
        "stdout_bytes": sum(
            len(artifact.content)
            for run in runs
            for artifact in run.artifacts
            if artifact.sink == "stdout"
        ),
        "stderr_bytes": sum(
            len(artifact.content)
            for run in runs
            for artifact in run.artifacts
            if artifact.sink == "stderr"
        ),
        "file_bytes": sum(
            len(artifact.content)
            for run in runs
            for artifact in run.artifacts
            if artifact.sink == "file"
        ),
    }


def _bytes_output(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8", errors="replace")


def _path_list(paths: Iterable[str | Path]) -> str:
    return os.pathsep.join(str(Path(path).expanduser().resolve()) for path in paths)


def _path_hash(path: str | Path) -> str:
    return _hash_bytes(str(Path(path).expanduser().resolve()).encode("utf-8"))


def _hash_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _slug_fixture_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return safe[:80] or "fixture"


_FEDERATED_CHILD_WRAPPER = r"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

detector_path = str(Path(sys.argv[1]).resolve())
input_path = str(Path(sys.argv[2]).resolve())
output_path = str(Path(sys.argv[3]).resolve())
output_dir = str(Path(sys.argv[4]).resolve())
violation_log = str(Path(sys.argv[5]).resolve())
extra_read_roots = [
    str(Path(item).resolve())
    for item in sys.argv[6].split(os.pathsep)
    if item
]
violation_fd = os.open(
    violation_log,
    os.O_CREAT | os.O_WRONLY | os.O_APPEND,
    0o600,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(
    kind: str,
    event: str,
    operation: str,
    path: object | None = None,
    detail: str = "",
) -> None:
    payload = {
        "kind": kind,
        "event": event,
        "operation": operation,
        "detail": detail,
    }
    if path is not None:
        payload["path_hash"] = _hash(str(Path(path).expanduser().resolve()))
    os.write(
        violation_fd,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )


def _roots(values: list[str]) -> list[str]:
    roots: list[str] = []
    for value in values:
        if not value:
            continue
        try:
            roots.append(str(Path(value).expanduser().resolve()))
        except OSError:
            continue
    return roots


READ_ROOTS = _roots(
    [
        detector_path,
        str(Path(detector_path).parent),
        input_path,
        output_path,
        output_dir,
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        *extra_read_roots,
    ]
)
WRITE_ROOTS = _roots([output_dir])


def _inside(path: object, roots: list[str]) -> bool:
    if not isinstance(path, (str, bytes, os.PathLike)):
        return True
    try:
        resolved = str(Path(path).expanduser().resolve())
    except OSError:
        resolved = os.path.abspath(os.fspath(path))
    for root in roots:
        try:
            if os.path.commonpath([resolved, root]) == root:
                return True
        except ValueError:
            continue
    return False


def _file_operation(mode: object, flags: object) -> str:
    if isinstance(mode, str):
        if any(marker in mode for marker in ("w", "a", "x", "+")):
            return "write"
        return "read"
    if isinstance(flags, int):
        write_mask = (
            os.O_WRONLY
            | os.O_RDWR
            | os.O_CREAT
            | os.O_TRUNC
            | os.O_APPEND
        )
        if flags & write_mask:
            return "write"
    return "read"


def _audit(event: str, args: tuple[object, ...]) -> None:
    if event.startswith("socket"):
        _record("network", event, "blocked")
        raise PermissionError("OPENMED_SANDBOX_VIOLATION network")
    if event in {
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.spawn",
        "os.fork",
        "os.exec",
    }:
        _record("process", event, "blocked")
        raise PermissionError("OPENMED_SANDBOX_VIOLATION process")
    if event == "open" and args:
        path = args[0]
        operation = _file_operation(
            args[1] if len(args) > 1 else None,
            args[2] if len(args) > 2 else None,
        )
        roots = WRITE_ROOTS if operation == "write" else READ_ROOTS
        if not _inside(path, roots):
            _record("filesystem", event, operation, path)
            raise PermissionError("OPENMED_SANDBOX_VIOLATION filesystem")
    if event in {"os.remove", "os.unlink", "os.rmdir"} and args:
        path = args[0]
        if not _inside(path, WRITE_ROOTS):
            _record("filesystem", event, "write", path)
            raise PermissionError("OPENMED_SANDBOX_VIOLATION filesystem")
    if event == "os.rename" and len(args) >= 2:
        for path in args[:2]:
            if not _inside(path, WRITE_ROOTS):
                _record("filesystem", event, "write", path)
                raise PermissionError("OPENMED_SANDBOX_VIOLATION filesystem")


sys.addaudithook(_audit)
os.environ["OPENMED_DETECTOR_INPUT"] = input_path
os.environ["OPENMED_DETECTOR_OUTPUT"] = output_path
os.environ["OPENMED_DETECTOR_OUTPUT_DIR"] = output_dir
runpy.run_path(detector_path, run_name="__main__")
""".lstrip()


def _shared_default_model_runner() -> ModelRunner:
    shared_loader: Any | None = None
    accepts_loader = _runner_accepts_loader(default_model_runner)

    def run_fixture(
        fixture: BenchmarkFixture,
        model_name: str,
        device: str,
    ) -> Iterable[Any]:
        nonlocal shared_loader
        if not accepts_loader:
            return default_model_runner(fixture, model_name, device)
        if shared_loader is None:
            from openmed.core.models import ModelLoader

            shared_loader = ModelLoader()
        return default_model_runner(
            fixture,
            model_name,
            device,
            loader=shared_loader,
        )

    return run_fixture


def _runner_accepts_loader(runner: Callable[..., Iterable[Any]]) -> bool:
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return True

    return any(
        parameter.name == "loader" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _attach_confidence_intervals(
    metrics: Mapping[str, Any],
    fixtures: Sequence[BenchmarkFixture],
    results: Sequence[FixtureResult],
    *,
    device: str,
    n_resamples: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap per-document CIs and merge them into the metric bundle."""
    result_by_id = {result.fixture_id: result for result in results}
    per_document_spans = [
        (
            fixture.gold_spans,
            getattr(result_by_id.get(fixture.fixture_id), "predicted_spans", ()),
        )
        for fixture in fixtures
    ]
    intervals = compute_confidence_intervals(
        per_document_spans,
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
        default_device=device,
    )
    merged = dict(metrics)
    for key, interval in intervals.items():
        metric = merged.get(key)
        if isinstance(metric, Mapping):
            merged[key] = {**metric, "confidence_interval": interval}
    return merged


def _attach_calibration_metrics(
    metrics: Mapping[str, Any],
    gold_spans: Sequence[EvalSpan],
    predicted_spans: Sequence[EvalSpan],
    *,
    n_bins: int,
) -> dict[str, Any]:
    """Merge reliability diagram data into the metric bundle."""
    bins = reliability_bins(
        _prediction_confidence_records(gold_spans, predicted_spans),
        n_bins=n_bins,
    )
    merged = dict(metrics)
    merged["calibration"] = {
        "expected_calibration_error": expected_calibration_error(bins),
        "reliability_bins": bins,
        "n_bins": n_bins,
    }
    return merged


def _resolve_abstention_thresholds(
    thresholds: Any | None,
    thresholds_path: str | Path | None,
) -> Any | None:
    if thresholds is not None:
        return thresholds
    if thresholds_path is not None:
        return load_calibration_thresholds(thresholds_path)
    return None


def _abstention_cache_hash(
    base_hash: str | None,
    *,
    thresholds: Any | None,
    thresholds_path: str | Path | None,
    confidence_threshold: float,
    target_risk: float | None,
    confidence_level: float | None,
    bootstrap_resamples: int,
    seed: int,
) -> str | None:
    if thresholds is None and thresholds_path is None:
        return base_hash
    digest = hashlib.sha256()
    digest.update(str(base_hash or "").encode("utf-8"))
    digest.update(b"\0abstention\0")
    payload = {
        "confidence_level": confidence_level,
        "confidence_threshold": confidence_threshold,
        "seed": seed,
        "target_risk": target_risk,
        "bootstrap_resamples": bootstrap_resamples,
        "thresholds": _jsonable_thresholds(thresholds),
        "thresholds_path": str(thresholds_path) if thresholds_path else None,
    }
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _jsonable_thresholds(thresholds: Any | None) -> Any:
    if thresholds is None:
        return None
    if hasattr(thresholds, "source_path") and thresholds.source_path:
        return {"source_path": thresholds.source_path}
    if hasattr(thresholds, "thresholds"):
        return {
            ".".join(key): value for key, value in sorted(thresholds.thresholds.items())
        }
    return thresholds


def _prediction_confidence_records(
    gold_spans: Sequence[EvalSpan],
    predicted_spans: Sequence[EvalSpan],
) -> list[dict[str, Any]]:
    matched_gold: set[int] = set()
    records: list[dict[str, Any]] = []
    for predicted in predicted_spans:
        correct = False
        for index, gold in enumerate(gold_spans):
            if index in matched_gold:
                continue
            if _exact_span_match(gold, predicted):
                matched_gold.add(index)
                correct = True
                break
        records.append(
            {
                "confidence": predicted.metadata.get("confidence", 1.0),
                "correct": correct,
            }
        )
    return records


def _exact_span_match(gold_span: EvalSpan, predicted_span: EvalSpan) -> bool:
    return (
        gold_span.label == predicted_span.label
        and gold_span.start == predicted_span.start
        and gold_span.end == predicted_span.end
    )


def _corpus_coordinates(
    fixtures: Sequence[BenchmarkFixture],
    results: Sequence[FixtureResult],
) -> tuple[list[EvalSpan], list[EvalSpan], str]:
    result_by_id = {result.fixture_id: result for result in results}
    gold: list[EvalSpan] = []
    predicted: list[EvalSpan] = []
    text_parts: list[str] = []
    offset = 0
    for fixture in fixtures:
        text_parts.append(fixture.text)
        gold.extend(_shift_spans(fixture.gold_spans, offset))
        result = result_by_id.get(fixture.fixture_id)
        if result is not None:
            predicted.extend(_shift_spans(result.predicted_spans, offset))
        offset += len(fixture.text) + 1
    return gold, predicted, "\n".join(text_parts)


def _fixture_extracted_facts(fixture: BenchmarkFixture) -> list[Any]:
    for key in (
        "extracted_facts",
        "facts",
        "clinical_facts",
        "entities",
        "relations",
        "grounded_concepts",
    ):
        value = fixture.metadata.get(key)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return list(value)
    return []


def _relation_corpus_relations(
    fixtures: Sequence[Any],
    results: Sequence[RelationFixtureResult],
) -> tuple[list[EvalRelation], list[EvalRelation]]:
    result_by_id = {result.fixture_id: result for result in results}
    gold: list[EvalRelation] = []
    predicted: list[EvalRelation] = []
    for fixture in fixtures:
        fixture_id = str(getattr(fixture, "fixture_id"))
        text = str(getattr(fixture, "text", ""))
        gold.extend(
            normalize_eval_relations(
                _fixture_gold_relations(fixture),
                entity_spans=getattr(fixture, "entities", None),
                fixture_id=fixture_id,
                default_language=str(getattr(fixture, "language", "en")),
                source_text=text,
            )
        )
        result = result_by_id.get(fixture_id)
        if result is not None:
            predicted.extend(result.predicted_relations)
    return gold, predicted


def _per_document_relations(
    fixtures: Sequence[Any],
    results: Sequence[RelationFixtureResult],
) -> list[tuple[tuple[EvalRelation, ...], tuple[EvalRelation, ...]]]:
    result_by_id = {result.fixture_id: result for result in results}
    documents: list[tuple[tuple[EvalRelation, ...], tuple[EvalRelation, ...]]] = []
    for fixture in fixtures:
        fixture_id = str(getattr(fixture, "fixture_id"))
        result = result_by_id.get(fixture_id)
        documents.append(
            (
                tuple(
                    normalize_eval_relations(
                        _fixture_gold_relations(fixture),
                        entity_spans=getattr(fixture, "entities", None),
                        fixture_id=fixture_id,
                        default_language=str(getattr(fixture, "language", "en")),
                        source_text=str(getattr(fixture, "text", "")),
                    )
                ),
                result.predicted_relations if result is not None else (),
            )
        )
    return documents


def _fixture_gold_relations(fixture: Any) -> Iterable[Any]:
    relations = getattr(fixture, "relations", None)
    if relations is not None:
        return relations
    return getattr(fixture, "gold_relations", ())


def _relation_fixture_hashes(fixtures: Sequence[Any]) -> dict[str, str]:
    """Hash each normalized fixture without carrying its source text."""

    fixture_hashes: dict[str, str] = {}
    for fixture in fixtures:
        fixture_id = str(getattr(fixture, "fixture_id"))
        payload = fixture.to_dict() if hasattr(fixture, "to_dict") else fixture
        fixture_hashes[fixture_id] = stable_hash(_plain(payload))
    return dict(sorted(fixture_hashes.items()))


def _relation_trap_leak_summary(
    fixtures: Sequence[Any],
    results: Sequence[RelationFixtureResult],
) -> dict[str, dict[str, Any]]:
    """Count conflicting predictions for zero-tolerance relation traps."""

    result_by_id = {result.fixture_id: result for result in results}
    leaked_by_kind: dict[str, set[str]] = {
        "assertion": set(),
        "temporal": set(),
    }
    trap_count = {"assertion": 0, "temporal": 0}
    evaluated_relation_count = {"assertion": 0, "temporal": 0}
    for fixture in fixtures:
        fixture_id = str(getattr(fixture, "fixture_id"))
        predicted = result_by_id.get(fixture_id)
        predictions = predicted.predicted_relations if predicted is not None else ()
        gold_by_id = {
            relation.relation_id: relation
            for relation in normalize_eval_relations(
                _fixture_gold_relations(fixture),
                entity_spans=getattr(fixture, "entities", None),
                fixture_id=fixture_id,
                default_language=str(getattr(fixture, "language", "en")),
                source_text=str(getattr(fixture, "text", "")),
            )
        }
        for trap in getattr(fixture, "traps", ()):
            kind = str(getattr(trap, "kind", "")).strip().lower()
            if kind not in trap_count:
                continue
            trap_count[kind] += 1
            for relation_id in getattr(trap, "relation_ids", ()):
                gold = gold_by_id.get(str(relation_id))
                if gold is None:
                    continue
                evaluated_relation_count[kind] += 1
                if not _relation_trap_conflict(
                    gold,
                    predictions,
                    kind=kind,
                ):
                    continue
                leaked_by_kind[kind].add(
                    stable_hash(
                        {
                            "fixture_id": fixture_id,
                            "relation_id": str(relation_id),
                            "trap_id": str(getattr(trap, "trap_id", "")),
                        }
                    )
                )

    return {
        kind: {
            "evaluated_relation_count": evaluated_relation_count[kind],
            "leak_count": len(leaked_by_kind[kind]),
            "leaked_relation_hashes": sorted(leaked_by_kind[kind]),
            "trap_count": trap_count[kind],
        }
        for kind in ("assertion", "temporal")
    }


def _relation_consistency_scores(
    trap_leaks: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return assertion and temporal consistency over trapped relations."""

    scores: dict[str, dict[str, Any]] = {}
    for kind in ("assertion", "temporal"):
        leak_evidence = _mapping_value(trap_leaks.get(kind))
        evaluated_relation_count = int(
            leak_evidence.get("evaluated_relation_count") or 0
        )
        leak_count = int(leak_evidence.get("leak_count") or 0)
        denominator = max(evaluated_relation_count, 1)
        score = max(0.0, 1.0 - (leak_count / denominator))
        scores[kind] = {
            "evaluated_relation_count": evaluated_relation_count,
            "leak_count": leak_count,
            "score": score,
        }
    return scores


def _relation_trap_conflict(
    gold: EvalRelation,
    predictions: Sequence[EvalRelation],
    *,
    kind: str,
) -> bool:
    for predicted in predictions:
        same_id = bool(
            gold.relation_id
            and predicted.relation_id
            and gold.relation_id == predicted.relation_id
        )
        same_direction = _relation_arguments_overlap(gold, predicted)
        reverse_direction = kind == "temporal" and _relation_arguments_overlap(
            gold,
            predicted,
            reverse=True,
        )
        if not (same_id or same_direction or reverse_direction):
            continue
        if same_direction and predicted.relation_type == gold.relation_type:
            continue
        return True
    return False


def _relation_arguments_overlap(
    gold: EvalRelation,
    predicted: EvalRelation,
    *,
    reverse: bool = False,
) -> bool:
    predicted_head = predicted.tail if reverse else predicted.head
    predicted_tail = predicted.head if reverse else predicted.tail
    return _eval_spans_overlap(gold.head, predicted_head) and _eval_spans_overlap(
        gold.tail,
        predicted_tail,
    )


def _eval_spans_overlap(left: EvalSpan, right: EvalSpan) -> bool:
    return min(left.end, right.end) > max(left.start, right.start)


def _validate_relation_offsets(
    relation: EvalRelation,
    text: str,
    fixture_id: str,
) -> None:
    for argument_name, argument in (("head", relation.head), ("tail", relation.tail)):
        if (
            argument.start < 0
            or argument.end < argument.start
            or argument.end > len(text)
        ):
            raise ValueError(
                "invalid relation argument offsets "
                f"{fixture_id}:{argument_name} "
                f"{argument.start}:{argument.end} for text length {len(text)}"
            )


def _is_relation_suite(suite: str) -> bool:
    normalized = suite.strip().lower().replace("_", "-")
    return normalized in {"relation", "relation-gold", "relation-golden", "relations"}


def _sha256_path(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _mapping_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _relation_markdown_metric_row(
    label: str,
    metric: Mapping[str, Any],
) -> str:
    return (
        f"| {label} | {_relation_percent(metric.get('precision'))} | "
        f"{_relation_percent(metric.get('recall'))} | "
        f"{_relation_percent(metric.get('f1'))} |"
    )


def _relation_percent(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.2%}"


def _shift_spans(spans: Iterable[EvalSpan], offset: int) -> list[EvalSpan]:
    return [
        replace(span, start=span.start + offset, end=span.end + offset)
        for span in spans
    ]


def _validate_unique_fixture_ids(fixtures: Sequence[BenchmarkFixture]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for fixture in fixtures:
        if fixture.fixture_id in seen and fixture.fixture_id not in duplicates:
            duplicates.append(fixture.fixture_id)
        seen.add(fixture.fixture_id)
    if duplicates:
        quoted = ", ".join(repr(value) for value in duplicates)
        raise ValueError(f"duplicate benchmark fixture id(s): {quoted}")


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _privacy_safe_pipeline_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for raw_key in sorted(value, key=str):
            key = str(raw_key)
            item = value[raw_key]
            if key.lower() in _PIPELINE_RAW_METADATA_KEYS:
                safe[f"{key}_hash"] = stable_hash({"value": str(item)})
            else:
                safe[key] = _privacy_safe_pipeline_metadata(item)
        return safe
    if isinstance(value, (list, tuple)):
        return [_privacy_safe_pipeline_metadata(item) for item in value]
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _key_bytes(key: bytes | str) -> bytes:
    if isinstance(key, bytes):
        if not key:
            raise ValueError("signing key must be non-empty")
        return key
    if isinstance(key, str):
        if not key:
            raise ValueError("signing key must be non-empty")
        return key.encode("utf-8")
    raise TypeError("signing key must be bytes or str")


__all__ = [
    "ModelRunner",
    "PipelineStageRunner",
    "RelationModelRunner",
    "BenchmarkFixture",
    "DEFAULT_PIPELINE_EVAL_FIXTURE",
    "PIPELINE_EVAL_SCHEMA_VERSION",
    "BoundaryLeakageFinding",
    "BoundaryLeakageResult",
    "FederatedDetectorSpec",
    "FederatedEvalReport",
    "FixtureResult",
    "PipelineEvalFixture",
    "PipelineEvalReport",
    "PipelineFixtureEvalResult",
    "PipelineStageOutput",
    "RELATION_SCORECARD_ARTIFACT",
    "RELATION_SCORECARD_SCHEMA_VERSION",
    "RelationFixtureResult",
    "RelationGateFailure",
    "RelationScorecard",
    "SandboxViolation",
    "TrainingEvalOverlapFinding",
    "check_training_manifest_overlap",
    "load_fixtures",
    "load_pipeline_eval_fixtures",
    "default_model_runner",
    "run_federated_leakage_eval",
    "run_pipeline_eval",
    "run_pipeline_eval_fixture",
    "run_benchmark",
    "run_relation_benchmark",
    "run_relation_suite",
    "run_cross_lingual_transfer",
    "run_cross_lingual_transfer_suite",
    "run_masakhaner_scorecard",
    "run_multilingual_ner_scorecard",
    "run_suite",
]
