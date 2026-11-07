"""Family-neutral qualification gates for exported token classifiers.

An exported Core ML package or ONNX graph is only *qualified* when it
reproduces its PyTorch reference checkpoint on a committed synthetic PHI suite.
This module owns the parts of that decision that do not depend on a model
family or a runtime:

- stable, value-free :class:`ExportReason` codes and
  :class:`ExportQualificationError`;
- the declared runtime support table (:class:`RuntimeSupport`) and tensor
  contract (:class:`ExportIOContract`), both checked before inference;
- per-variant gates for logit parity, span parity, direct-identifier recall,
  critical leakage, and quantization delta (reusing
  :mod:`openmed.eval.quant_delta`);
- identity digests for the source revision, tokenizer, label map, and artifact
  files, plus offline verification of a written report.

Family adapters (ModernBERT today; Longformer, Privacy Filter, and GLiNER
later) supply the runners. A runner receives one padded feed such as
``{"input_ids": array[1, S], "attention_mask": array[1, S]}`` and returns
logits shaped ``[1, S, num_labels]``. Reports carry only revisions, versions,
digests, dimensions, counts, aggregate metrics, timings, and reason codes:
never fixture text, decoded values, token ids, logits, or local paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.core.labels import (
    DIRECT_IDENTIFIER,
    RISK_HIGH,
    normalize_label,
    policy_label_for,
    risk_level_for,
)
from openmed.eval.quant_delta import (
    evaluate_onnx_logit_parity,
    evaluate_quant_recall_delta,
    limit_for_format,
)

EXPORT_QUALIFICATION_SCHEMA_VERSION = 1
EXPORT_QUALIFICATION_FORMAT = "openmed-export-qualification"
EXPORT_QUALIFICATION_FILENAME = "openmed-export-qualification.json"
DEFAULT_QUALIFICATION_SUITE = (
    Path(__file__).resolve().parent / "fixtures" / "export_qualification_phi.jsonl"
)
FLOAT32_LOGIT_ATOL = 1e-4
FLOAT32_LOGIT_RTOL = 1e-4
FLOAT16_LOGIT_ATOL = 5e-2
FLOAT16_LOGIT_RTOL = 1e-2
OFFSET_CONTRACT: Mapping[str, str] = {
    "source": "tokenizer_offset_mapping",
    "unit": "unicode_code_point",
    "interval": "half_open",
    "special_tokens": "zero_width",
    "padding": "zero_width",
}
DEFAULT_ENVIRONMENT_PACKAGES = (
    "openmed",
    "numpy",
    "torch",
    "transformers",
    "tokenizers",
    "coremltools",
    "onnx",
    "onnxruntime",
    "onnxscript",
)
QUANTIZATION_NONE = "none"
SHAPE_MODE_STATIC = "static"
SHAPE_MODE_DYNAMIC = "dynamic"
_SHAPE_MODES = (SHAPE_MODE_STATIC, SHAPE_MODE_DYNAMIC)
_VOLATILE_VARIANT_KEYS = ("latency_ms", "process_peak_rss_bytes")
_REPORT_REQUIRED_KEYS = (
    "family",
    "runtime",
    "source",
    "tokenizer",
    "label_map",
    "io_contract",
    "offset_contract",
    "environment",
    "suite",
    "variants",
    "artifacts",
)


class ExportReason(StrEnum):
    """Stable, value-free reason codes for export qualification decisions."""

    UNSUPPORTED_FAMILY = "unsupported_family"
    UNSUPPORTED_RUNTIME = "unsupported_runtime"
    UNSUPPORTED_PROVIDER = "unsupported_provider"
    UNSUPPORTED_PRECISION = "unsupported_precision"
    UNSUPPORTED_QUANTIZATION = "unsupported_quantization"
    UNSUPPORTED_SEQUENCE_LENGTH = "unsupported_sequence_length"
    UNSUPPORTED_SEQUENCE_SHAPE = "unsupported_sequence_shape"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    UNSUPPORTED_CONVERTER_ENVIRONMENT = "unsupported_converter_environment"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    MISSING_LABEL_MAP = "missing_label_map"
    MISSING_OFFSET_MAPPING = "missing_offset_mapping"
    MISSING_EVIDENCE = "missing_evidence"
    SOURCE_REVISION_UNAVAILABLE = "source_revision_unavailable"
    SOURCE_REVISION_MISMATCH = "source_revision_mismatch"
    TOKENIZER_IDENTITY_MISMATCH = "tokenizer_identity_mismatch"
    LABEL_MAP_MISMATCH = "label_map_mismatch"
    IO_CONTRACT_MISMATCH = "io_contract_mismatch"
    ARTIFACT_MISSING = "artifact_missing"
    ARTIFACT_DIGEST_MISMATCH = "artifact_digest_mismatch"
    REPORT_INVALID = "report_invalid"
    VARIANT_NOT_QUALIFIED = "variant_not_qualified"
    INFERENCE_FAILED = "inference_failed"
    LOGIT_PARITY_FAILED = "logit_parity_failed"
    SPAN_PARITY_FAILED = "span_parity_failed"
    DIRECT_IDENTIFIER_RECALL_FAILED = "direct_identifier_recall_failed"
    CRITICAL_LEAKAGE_DETECTED = "critical_leakage_detected"
    QUANTIZATION_DELTA_EXCEEDED = "quantization_delta_exceeded"
    # Label-conditioned (zero-shot) profiles.
    UNSUPPORTED_TASK = "unsupported_task"
    UNSUPPORTED_MODEL_VARIANT = "unsupported_model_variant"
    UNSUPPORTED_REMOTE_CODE = "unsupported_remote_code"
    UNSUPPORTED_LABEL_COUNT = "unsupported_label_count"
    UNSUPPORTED_LABEL = "unsupported_label"
    UNSUPPORTED_CONTROL_MARKER = "unsupported_control_marker"
    UNSUPPORTED_SPAN_WIDTH = "unsupported_span_width"
    UNSUPPORTED_WORD_COUNT = "unsupported_word_count"
    UNSUPPORTED_MEMORY_BUDGET = "unsupported_memory_budget"
    LABEL_ORDER_PARITY_FAILED = "label_order_parity_failed"
    DECISION_PARITY_FAILED = "decision_parity_failed"


_REASON_MESSAGES: Mapping[ExportReason, str] = {
    ExportReason.UNSUPPORTED_FAMILY: "model family has no qualified export path",
    ExportReason.UNSUPPORTED_RUNTIME: "runtime is not declared for this family",
    ExportReason.UNSUPPORTED_PROVIDER: "execution provider is not qualified",
    ExportReason.UNSUPPORTED_PRECISION: "precision is not qualified",
    ExportReason.UNSUPPORTED_QUANTIZATION: "quantization is not qualified",
    ExportReason.UNSUPPORTED_SEQUENCE_LENGTH: (
        "sequence length is outside the declared buckets"
    ),
    ExportReason.UNSUPPORTED_SEQUENCE_SHAPE: "sequence shape mode is not qualified",
    ExportReason.UNSUPPORTED_OPERATOR: "graph uses an operator without coverage",
    ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT: (
        "installed converter dependencies are not qualified for this family"
    ),
    ExportReason.RUNTIME_UNAVAILABLE: "runtime cannot execute on this host",
    ExportReason.MISSING_LABEL_MAP: "label map is missing",
    ExportReason.MISSING_OFFSET_MAPPING: "tokenizer does not return offsets",
    ExportReason.MISSING_EVIDENCE: "required qualification evidence is missing",
    ExportReason.SOURCE_REVISION_UNAVAILABLE: "source revision cannot be pinned",
    ExportReason.SOURCE_REVISION_MISMATCH: "source revision does not match the pin",
    ExportReason.TOKENIZER_IDENTITY_MISMATCH: "tokenizer identity does not match",
    ExportReason.LABEL_MAP_MISMATCH: "label map identity does not match",
    ExportReason.IO_CONTRACT_MISMATCH: "tensor contract does not match",
    ExportReason.ARTIFACT_MISSING: "declared artifact file is missing",
    ExportReason.ARTIFACT_DIGEST_MISMATCH: "artifact digest does not match",
    ExportReason.REPORT_INVALID: "qualification report is malformed",
    ExportReason.VARIANT_NOT_QUALIFIED: "artifact variant is not qualified",
    ExportReason.INFERENCE_FAILED: "runtime inference failed",
    ExportReason.LOGIT_PARITY_FAILED: "logits exceed the declared tolerance",
    ExportReason.SPAN_PARITY_FAILED: "decoded spans differ from the reference",
    ExportReason.DIRECT_IDENTIFIER_RECALL_FAILED: (
        "direct-identifier recall fell below the reference"
    ),
    ExportReason.CRITICAL_LEAKAGE_DETECTED: (
        "candidate leaks critical identifiers the reference caught"
    ),
    ExportReason.QUANTIZATION_DELTA_EXCEEDED: (
        "quantized recall delta exceeds the documented limit"
    ),
    ExportReason.UNSUPPORTED_TASK: "task has no qualified export or runtime path",
    ExportReason.UNSUPPORTED_MODEL_VARIANT: (
        "model configuration is outside the declared export profile"
    ),
    ExportReason.UNSUPPORTED_REMOTE_CODE: (
        "checkpoint requires repository code, which is never executed"
    ),
    ExportReason.UNSUPPORTED_LABEL_COUNT: (
        "candidate label count is outside the qualified bounds"
    ),
    ExportReason.UNSUPPORTED_LABEL: "candidate label violates the label contract",
    ExportReason.UNSUPPORTED_CONTROL_MARKER: (
        "input contains a reserved prompt control marker"
    ),
    ExportReason.UNSUPPORTED_SPAN_WIDTH: "span width exceeds the qualified bound",
    ExportReason.UNSUPPORTED_WORD_COUNT: "word count exceeds the qualified bound",
    ExportReason.UNSUPPORTED_MEMORY_BUDGET: (
        "estimated activation memory exceeds the configured budget"
    ),
    ExportReason.LABEL_ORDER_PARITY_FAILED: (
        "candidate label ranking differs from the reference"
    ),
    ExportReason.DECISION_PARITY_FAILED: (
        "thresholded decisions differ from the reference"
    ),
}


class ExportQualificationError(ValueError):
    """Raised when an export request or artifact fails qualification.

    The message never contains model inputs, decoded values, credentials, or
    local paths. Callers branch on :attr:`reason`, a stable
    :class:`ExportReason` value.

    Args:
        reason: The stable reason code.
        message: Optional value-free message; defaults to a static description
            of *reason*.
    """

    def __init__(self, reason: ExportReason | str, message: str | None = None):
        self.reason = ExportReason(reason)
        super().__init__(
            message or f"{self.reason.value}: {_REASON_MESSAGES[self.reason]}"
        )

    @property
    def reason_code(self) -> str:
        """Return the reason as a plain string."""
        return self.reason.value


@dataclass(frozen=True)
class RuntimeSupport:
    """Declared, qualifiable combinations for one family on one runtime.

    Attributes:
        runtime: Runtime name, for example ``"coreml"`` or ``"onnx"``.
        precisions: Compute or weight precisions qualification may claim.
        quantizations: Weight quantizations; ``"none"`` is the float graph.
        providers: Execution providers or Core ML compute-unit selectors.
        shape_modes: ``"static"`` buckets and/or ``"dynamic"`` sequence axes.
        target: Deployment target or minimum opset, for the manifest.
    """

    runtime: str
    precisions: tuple[str, ...]
    quantizations: tuple[str, ...] = (QUANTIZATION_NONE,)
    providers: tuple[str, ...] = ()
    shape_modes: tuple[str, ...] = (SHAPE_MODE_STATIC,)
    target: str | None = None

    def validate(
        self,
        *,
        precision: str | None = None,
        quantization: str | None = None,
        provider: str | None = None,
        shape_mode: str | None = None,
    ) -> None:
        """Reject an unsupported combination before any inference runs.

        Raises:
            ExportQualificationError: With the reason for the first dimension
                that is not declared.
        """

        if precision is not None and precision not in self.precisions:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_PRECISION)
        if quantization is not None and quantization not in self.quantizations:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_QUANTIZATION)
        if provider is not None and self.providers and provider not in self.providers:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_PROVIDER)
        if shape_mode is not None and shape_mode not in self.shape_modes:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_SEQUENCE_SHAPE)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready support row."""
        return {
            "runtime": self.runtime,
            "precisions": list(self.precisions),
            "quantizations": list(self.quantizations),
            "providers": list(self.providers),
            "shape_modes": list(self.shape_modes),
            "target": self.target,
        }


@dataclass(frozen=True)
class ExportIOContract:
    """Stable tensor names and sequence bounds of an exported graph.

    Attributes:
        input_names: Graph input names in feed order.
        output_names: Graph output names; the first is the logits tensor.
        sequence_buckets: Padded sequence lengths the artifact accepts.
        max_sequence_length: Hard upper bound; longer inputs are rejected.
        shape_mode: ``"static"`` (pad to the next bucket) or ``"dynamic"``.
        batch_size: Fixed batch size, or ``None`` for a dynamic batch axis.
        input_dtype: Integer dtype of every input tensor.
    """

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    sequence_buckets: tuple[int, ...]
    max_sequence_length: int
    shape_mode: str = SHAPE_MODE_STATIC
    batch_size: int | None = 1
    input_dtype: str = "int32"

    def __post_init__(self) -> None:
        buckets = tuple(sorted({int(bucket) for bucket in self.sequence_buckets}))
        object.__setattr__(self, "sequence_buckets", buckets)
        object.__setattr__(self, "input_names", tuple(self.input_names))
        object.__setattr__(self, "output_names", tuple(self.output_names))
        if (
            not self.input_names
            or not self.output_names
            or self.shape_mode not in _SHAPE_MODES
            or self.max_sequence_length <= 0
            or any(bucket <= 0 for bucket in buckets)
            or any(bucket > self.max_sequence_length for bucket in buckets)
            or (self.shape_mode == SHAPE_MODE_STATIC and not buckets)
        ):
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)

    def bucket_for(self, sequence_length: int) -> int:
        """Return the padded length for *sequence_length* tokens.

        Raises:
            ExportQualificationError: ``unsupported_sequence_length`` when the
                input is empty or exceeds the declared bound.
        """

        length = int(sequence_length)
        if length <= 0 or length > self.max_sequence_length:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_SEQUENCE_LENGTH)
        for bucket in self.sequence_buckets:
            if length <= bucket:
                return bucket
        if self.shape_mode == SHAPE_MODE_DYNAMIC:
            return length
        raise ExportQualificationError(ExportReason.UNSUPPORTED_SEQUENCE_LENGTH)

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest form of the contract."""
        batch: int | str = self.batch_size if self.batch_size is not None else "batch"
        sequence: str = "bucket" if self.shape_mode == SHAPE_MODE_STATIC else "sequence"
        return {
            "inputs": [
                {"name": name, "dtype": self.input_dtype, "shape": [batch, sequence]}
                for name in self.input_names
            ],
            "outputs": [
                {
                    "name": name,
                    "dtype": "float32",
                    "shape": [batch, sequence, "num_labels"],
                }
                for name in self.output_names
            ],
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "sequence_buckets": list(self.sequence_buckets),
            "max_sequence_length": self.max_sequence_length,
            "shape_mode": self.shape_mode,
            "overflow": "reject",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExportIOContract":
        """Rebuild a contract from :meth:`to_dict` output."""
        try:
            inputs = payload.get("inputs") or []
            batch = inputs[0]["shape"][0] if inputs else 1
            return cls(
                input_names=tuple(payload["input_names"]),
                output_names=tuple(payload["output_names"]),
                sequence_buckets=tuple(payload.get("sequence_buckets") or ()),
                max_sequence_length=int(payload["max_sequence_length"]),
                shape_mode=str(payload.get("shape_mode") or SHAPE_MODE_STATIC),
                batch_size=batch if isinstance(batch, int) else None,
                input_dtype=str(inputs[0]["dtype"]) if inputs else "int32",
            )
        except (KeyError, IndexError, TypeError, ValueError):
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH) from None


@dataclass(frozen=True)
class QualificationPolicy:
    """Declared tolerances for one exported variant.

    Float variants must reproduce reference logits and spans. Quantized
    variants are judged on direct-identifier recall delta (the documented
    INT8/INT4 limits in :mod:`openmed.eval.quant_delta`) and critical leakage;
    their logit and span drift is recorded but does not gate.

    Attributes:
        logit_atol: Absolute logit tolerance.
        logit_rtol: Relative logit tolerance.
        gate_logits: Whether logit drift fails the variant.
        gate_spans: Whether any decoded-span difference fails the variant.
        span_tolerance: Allowed character drift per span boundary.
        recall_delta_limit: Allowed per-label direct-identifier recall loss.
        max_new_critical_leaks: Allowed critical gold spans the candidate leaks
            but the reference caught.
        direct_identifier_recall_floor: Optional absolute recall floor, used
            for release evidence on real checkpoints.
    """

    logit_atol: float
    logit_rtol: float
    gate_logits: bool = True
    gate_spans: bool = True
    span_tolerance: int = 0
    recall_delta_limit: float = 0.0
    max_new_critical_leaks: int = 0
    direct_identifier_recall_floor: float | None = None

    @classmethod
    def for_variant(
        cls,
        *,
        precision: str,
        quantization: str = QUANTIZATION_NONE,
        format_name: str | None = None,
        direct_identifier_recall_floor: float | None = None,
    ) -> "QualificationPolicy":
        """Return the documented policy for a precision/quantization pair.

        Raises:
            ExportQualificationError: For an undeclared precision or a
                quantization without a documented recall-delta limit.
        """

        if quantization != QUANTIZATION_NONE:
            limit = limit_for_format(quantization)
            if limit is None and format_name is not None:
                limit = limit_for_format(format_name)
            if limit is None:
                raise ExportQualificationError(ExportReason.UNSUPPORTED_QUANTIZATION)
            return cls(
                logit_atol=FLOAT16_LOGIT_ATOL,
                logit_rtol=FLOAT16_LOGIT_RTOL,
                gate_logits=False,
                gate_spans=False,
                recall_delta_limit=limit,
                direct_identifier_recall_floor=direct_identifier_recall_floor,
            )
        if precision == "float32":
            return cls(
                logit_atol=FLOAT32_LOGIT_ATOL,
                logit_rtol=FLOAT32_LOGIT_RTOL,
                direct_identifier_recall_floor=direct_identifier_recall_floor,
            )
        if precision == "float16":
            return cls(
                logit_atol=FLOAT16_LOGIT_ATOL,
                logit_rtol=FLOAT16_LOGIT_RTOL,
                direct_identifier_recall_floor=direct_identifier_recall_floor,
            )
        raise ExportQualificationError(ExportReason.UNSUPPORTED_PRECISION)

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest form of the policy."""
        return {
            "logit_atol": self.logit_atol,
            "logit_rtol": self.logit_rtol,
            "gate_logits": self.gate_logits,
            "gate_spans": self.gate_spans,
            "span_tolerance": self.span_tolerance,
            "recall_delta_limit": self.recall_delta_limit,
            "max_new_critical_leaks": self.max_new_critical_leaks,
            "direct_identifier_recall_floor": self.direct_identifier_recall_floor,
        }


@dataclass(frozen=True)
class EncodedFixture:
    """One synthetic fixture tokenized and padded to its bucket.

    Encoded fixtures stay in memory only; they are never serialized.
    """

    fixture_id: str
    feed: Mapping[str, Any]
    offsets: tuple[tuple[int, int], ...]
    attention: tuple[int, ...]
    token_count: int
    bucket: int


LogitRunner = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class VariantCandidate:
    """One exported artifact to qualify against the reference runner.

    Attributes:
        name: Variant name, for example ``"coreml-fp16"`` or ``"onnx-int8"``.
        runner: Callable returning ``[1, S, num_labels]`` logits for a feed.
        precision: Compute or weight precision of the artifact.
        quantization: ``"none"``, ``"int8"``, or ``"int4"``.
        provider: Provider or compute-unit selector used for qualification.
        artifact_paths: Files or directories that make up the artifact.
        target_provider: Deployment provider when qualification ran on a
            different, CPU proxy provider. A proxy run is evidence for the
            graph, never a supported claim for the target provider.
        evidence: Extra value-free evidence (operator coverage, residency).
    """

    name: str
    runner: LogitRunner
    precision: str
    quantization: str = QUANTIZATION_NONE
    provider: str | None = None
    artifact_paths: tuple[Path, ...] = ()
    target_provider: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantQualification:
    """Gate outcomes and value-free evidence for one exported variant."""

    name: str
    runtime: str
    precision: str
    quantization: str
    provider: str | None
    target_provider: str | None
    passed: bool
    reasons: tuple[str, ...]
    policy: QualificationPolicy
    logit_parity: Mapping[str, Any]
    span_parity: Mapping[str, Any]
    direct_identifier_recall: Mapping[str, Any]
    critical_leakage: Mapping[str, Any]
    quantization_delta: Mapping[str, Any] | None
    buckets_evaluated: tuple[int, ...]
    artifacts: tuple[str, ...] = ()
    size_bytes: int | None = None
    latency_ms: Mapping[str, Any] = field(default_factory=dict)
    process_peak_rss_bytes: int | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def claim(self) -> str:
        """Return ``qualified``, ``cpu_proxy``, or ``rejected``."""
        if not self.passed:
            return "rejected"
        if self.target_provider and self.target_provider != self.provider:
            return "cpu_proxy"
        return "qualified"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready variant evidence."""
        return {
            "name": self.name,
            "runtime": self.runtime,
            "precision": self.precision,
            "quantization": self.quantization,
            "provider": self.provider,
            "target_provider": self.target_provider,
            "claim": self.claim,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "policy": self.policy.to_dict(),
            "logit_parity": dict(self.logit_parity),
            "span_parity": dict(self.span_parity),
            "direct_identifier_recall": dict(self.direct_identifier_recall),
            "critical_leakage": dict(self.critical_leakage),
            "quantization_delta": (
                dict(self.quantization_delta)
                if self.quantization_delta is not None
                else None
            ),
            "buckets_evaluated": list(self.buckets_evaluated),
            "artifacts": list(self.artifacts),
            "size_bytes": self.size_bytes,
            "latency_ms": dict(self.latency_ms),
            "process_peak_rss_bytes": self.process_peak_rss_bytes,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class ExportQualificationReport:
    """Complete qualification evidence for one family on one runtime."""

    family: str
    adapter_version: int
    runtime: str
    support: RuntimeSupport
    source: Mapping[str, Any]
    tokenizer: Mapping[str, Any]
    label_map: Mapping[str, Any]
    io_contract: ExportIOContract
    environment: Mapping[str, Any]
    suite: Mapping[str, Any]
    variants: tuple[VariantQualification, ...]
    artifacts: Mapping[str, str]
    task: str = "token-classification"

    @property
    def passed(self) -> bool:
        """Return whether every variant passed."""
        return bool(self.variants) and all(item.passed for item in self.variants)

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return the distinct failing reason codes in variant order."""
        seen: list[str] = []
        for variant in self.variants:
            for reason in variant.reasons:
                if reason not in seen:
                    seen.append(reason)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready report including its stable digest."""
        payload = self._payload()
        payload["stable_sha256"] = _stable_report_digest(payload)
        return payload

    def _payload(self) -> dict[str, Any]:
        return {
            "format": EXPORT_QUALIFICATION_FORMAT,
            "schema_version": EXPORT_QUALIFICATION_SCHEMA_VERSION,
            "task": self.task,
            "family": self.family,
            "adapter_version": self.adapter_version,
            "runtime": self.runtime,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "qualified_variants": [
                item.name for item in self.variants if item.claim == "qualified"
            ],
            "support": self.support.to_dict(),
            "source": dict(self.source),
            "tokenizer": dict(self.tokenizer),
            "label_map": dict(self.label_map),
            "io_contract": self.io_contract.to_dict(),
            "offset_contract": dict(OFFSET_CONTRACT),
            "environment": dict(self.environment),
            "suite": dict(self.suite),
            "variants": [item.to_dict() for item in self.variants],
            "artifacts": dict(sorted(self.artifacts.items())),
        }

    def raise_for_failure(self) -> None:
        """Raise for the first failing reason, if any variant failed.

        Raises:
            ExportQualificationError: When qualification did not pass.
        """

        if self.passed:
            return
        reasons = self.reasons
        raise ExportQualificationError(
            reasons[0] if reasons else ExportReason.MISSING_EVIDENCE
        )


# ---------------------------------------------------------------------------
# Identity and digests
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of one file, streamed in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_file_digests(
    paths: Iterable[str | Path],
    *,
    root: str | Path,
    exclude_names: Iterable[str] = (),
) -> dict[str, str]:
    """Return ``{relative posix path: sha256}`` for files under *root*.

    Directories (for example ``.mlpackage`` bundles) are walked recursively.
    Paths outside *root* are rejected so reports never record private paths.

    Raises:
        ExportQualificationError: ``artifact_missing`` for a missing path or
            one that is not under *root*.
    """

    root_path = Path(root).resolve()
    excluded = set(exclude_names)
    digests: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.exists():
            raise ExportQualificationError(ExportReason.ARTIFACT_MISSING)
        try:
            path.relative_to(root_path)
        except ValueError:
            raise ExportQualificationError(ExportReason.ARTIFACT_MISSING) from None
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file_path in files:
            if not file_path.is_file() or file_path.name in excluded:
                continue
            relative = file_path.relative_to(root_path).as_posix()
            digests[relative] = sha256_file(file_path)
    return dict(sorted(digests.items()))


def label_map_identity(id2label: Mapping[Any, Any] | None) -> dict[str, Any]:
    """Return the digest and size of a canonical ``{id: label}`` map.

    Raises:
        ExportQualificationError: ``missing_label_map`` for an empty map or
            non-integer ids.
    """

    if not id2label:
        raise ExportQualificationError(ExportReason.MISSING_LABEL_MAP)
    try:
        canonical = {str(int(key)): str(value) for key, value in id2label.items()}
    except (TypeError, ValueError):
        raise ExportQualificationError(ExportReason.MISSING_LABEL_MAP) from None
    ordered = dict(sorted(canonical.items(), key=lambda item: int(item[0])))
    return {
        "sha256": _canonical_digest(ordered),
        "num_labels": len(ordered),
    }


def tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    """Return a file-independent identity digest for a fast tokenizer.

    The digest covers the backend tokenizer JSON (vocabulary, normalizer,
    pre-tokenizer, model, post-processor) plus the special-token map, so a
    consumer can recompute it from whatever copy of the tokenizer it loads.

    Raises:
        ExportQualificationError: ``missing_offset_mapping`` when the
            tokenizer is not a fast tokenizer that can report offsets.
    """

    backend = getattr(tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        raise ExportQualificationError(ExportReason.MISSING_OFFSET_MAPPING)
    serialized = json.loads(to_str())
    if isinstance(serialized, dict):
        # Transformers toggles backend padding/truncation per call; neither is
        # part of the tokenizer's identity.
        serialized.pop("padding", None)
        serialized.pop("truncation", None)
    special = getattr(tokenizer, "special_tokens_map", None) or {}
    payload = {
        "backend": serialized,
        "special_tokens": {str(key): str(value) for key, value in special.items()},
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    model = serialized.get("model") if isinstance(serialized, Mapping) else None
    vocab = model.get("vocab") if isinstance(model, Mapping) else None
    return {
        "sha256": _canonical_digest(payload),
        "kind": "fast_tokenizer_json",
        "model": str(model.get("type")) if isinstance(model, Mapping) else None,
        "vocab_size": len(vocab) if isinstance(vocab, (Mapping, list)) else None,
    }


def config_identity(config: Any) -> str:
    """Return a digest of the architecture-defining model configuration."""
    payload = _config_dict(config)
    volatile = {"_name_or_path", "_commit_hash", "transformers_version", "dtype"}
    stable = {
        key: value
        for key, value in payload.items()
        if key not in volatile and not key.startswith("_attn")
    }
    return _canonical_digest(stable)


def source_identity(
    model_id: str | Path,
    *,
    config: Any,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Return the pinned source revision for a reference checkpoint.

    Hub checkpoints record the resolved commit from the loaded configuration.
    Local directories record a content digest of their weight and config
    files instead of the (private) path.

    Args:
        model_id: Hub repository id or local checkpoint directory.
        config: Loaded Hugging Face configuration.
        expected_revision: Optional pin; a different resolved revision fails.

    Raises:
        ExportQualificationError: ``source_revision_unavailable`` when no
            revision can be resolved, or ``source_revision_mismatch``.
    """

    path = Path(str(model_id)).expanduser()
    commit = _config_value(config, "_commit_hash")
    if path.is_dir():
        weight_files = [
            item
            for item in sorted(path.iterdir())
            if item.is_file()
            and (
                item.suffix in {".safetensors", ".bin", ".pt"}
                or item.name == "config.json"
            )
        ]
        if not any(item.name != "config.json" for item in weight_files):
            raise ExportQualificationError(ExportReason.SOURCE_REVISION_UNAVAILABLE)
        listing = {item.name: sha256_file(item) for item in weight_files}
        revision = f"sha256:{_canonical_digest(listing)}"
        kind = "content_sha256"
        public_id = None
    elif commit:
        revision = str(commit)
        kind = "hub_commit"
        public_id = str(model_id)
    else:
        raise ExportQualificationError(ExportReason.SOURCE_REVISION_UNAVAILABLE)

    if expected_revision is not None and expected_revision not in {revision, commit}:
        raise ExportQualificationError(ExportReason.SOURCE_REVISION_MISMATCH)
    return {
        "kind": kind,
        "revision": revision,
        "model_id": public_id,
        "config_sha256": config_identity(config),
        "model_type": _config_value(config, "model_type"),
    }


def converter_environment(
    packages: Sequence[str] = DEFAULT_ENVIRONMENT_PACKAGES,
) -> dict[str, Any]:
    """Return installed converter versions without importing heavy modules."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "packages": versions,
    }


def load_qualification_fixtures(
    path: str | Path | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Load a synthetic qualification suite and its value-free identity.

    Args:
        path: JSON or JSONL benchmark fixture file. Defaults to the packaged
            synthetic PHI suite.

    Returns:
        ``(fixtures, suite_identity)`` where the identity records only the
        file digest, fixture count, labels, and whether it is the default.

    Raises:
        ExportQualificationError: ``missing_evidence`` for an empty suite.
    """

    from openmed.eval.harness import load_fixtures

    suite_path = Path(path) if path is not None else DEFAULT_QUALIFICATION_SUITE
    fixtures = load_fixtures(suite_path)
    if not fixtures:
        raise ExportQualificationError(ExportReason.MISSING_EVIDENCE)
    labels = sorted(
        {
            normalize_label(span.label, fixture.language)
            for fixture in fixtures
            for span in fixture.gold_spans
        }
    )
    identity = {
        "default": path is None
        or suite_path.resolve() == DEFAULT_QUALIFICATION_SUITE.resolve(),
        "sha256": sha256_file(suite_path),
        "fixture_count": len(fixtures),
        "gold_span_count": sum(len(fixture.gold_spans) for fixture in fixtures),
        "labels": labels,
        "languages": sorted({fixture.language for fixture in fixtures}),
        "synthetic": True,
    }
    return fixtures, identity


# ---------------------------------------------------------------------------
# Encoding, decoding, and gates
# ---------------------------------------------------------------------------


def encode_fixtures(
    fixtures: Sequence[Any],
    tokenizer: Any,
    contract: ExportIOContract,
) -> tuple[EncodedFixture, ...]:
    """Tokenize fixtures and pad each one to its declared bucket.

    Raises:
        ExportQualificationError: ``missing_offset_mapping`` when the tokenizer
            cannot report offsets, or ``unsupported_sequence_length`` for a
            fixture longer than the contract allows.
    """

    import numpy as np

    dtype = np.dtype(contract.input_dtype)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = int(pad_id) if pad_id is not None else 0
    encoded: list[EncodedFixture] = []
    for fixture in fixtures:
        try:
            tokens = tokenizer(
                fixture.text,
                add_special_tokens=True,
                return_offsets_mapping=True,
                truncation=False,
            )
        except (NotImplementedError, TypeError, ValueError):
            raise ExportQualificationError(
                ExportReason.MISSING_OFFSET_MAPPING
            ) from None
        offsets = tokens.get("offset_mapping")
        if offsets is None:
            raise ExportQualificationError(ExportReason.MISSING_OFFSET_MAPPING)
        input_ids = [int(value) for value in tokens["input_ids"]]
        token_count = len(input_ids)
        bucket = contract.bucket_for(token_count)
        padding = bucket - token_count
        attention = [1] * token_count + [0] * padding
        feed: dict[str, Any] = {}
        for name in contract.input_names:
            if name == "input_ids":
                values = input_ids + [pad_id] * padding
            elif name == "attention_mask":
                values = attention
            elif name == "token_type_ids":
                values = [0] * bucket
            else:
                raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
            feed[name] = np.asarray([values], dtype=dtype)
        padded_offsets = [(int(start), int(end)) for start, end in offsets] + [
            (0, 0)
        ] * padding
        encoded.append(
            EncodedFixture(
                fixture_id=str(fixture.fixture_id),
                feed=feed,
                offsets=tuple(padded_offsets),
                attention=tuple(attention),
                token_count=token_count,
                bucket=bucket,
            )
        )
    return tuple(encoded)


def decode_token_spans(
    logits: Any,
    *,
    offsets: Sequence[Sequence[int]],
    attention: Sequence[int],
    id2label: Mapping[Any, str],
) -> list[dict[str, Any]]:
    """Decode ``[S, num_labels]`` logits into character spans.

    Uses the same BIOES decoder as the OpenMed runtimes. Special and padding
    tokens (zero-width offsets or a zero attention mask) never open spans.
    The returned spans carry labels and offsets only, never text.
    """

    from openmed.core.decoding import build_label_info, labels_to_token_spans

    rows = _as_rows(logits)
    label_info = build_label_info(
        {int(key): str(value) for key, value in id2label.items()}
    )
    labels_by_index: dict[int, int] = {}
    for index, row in enumerate(rows):
        if index >= len(offsets) or index >= len(attention):
            break
        start, end = int(offsets[index][0]), int(offsets[index][1])
        if not attention[index] or start == end:
            continue
        labels_by_index[index] = _argmax(row)
    spans: list[dict[str, Any]] = []
    for span_label, token_start, token_end in labels_to_token_spans(
        labels_by_index, label_info
    ):
        spans.append(
            {
                "label": label_info.span_class_names[span_label],
                "start": int(offsets[token_start][0]),
                "end": int(offsets[token_end - 1][1]),
            }
        )
    return spans


def collect_runner_outputs(
    runner: LogitRunner,
    encoded: Sequence[EncodedFixture],
    *,
    id2label: Mapping[Any, str],
) -> tuple[dict[str, list[list[float]]], dict[str, list[dict[str, Any]]], list[float]]:
    """Run *runner* on every encoded fixture.

    Returns:
        ``(logits, spans, latencies_ms)``: real-token logit rows and decoded
        spans keyed by fixture id, plus per-call latency.

    Raises:
        ExportQualificationError: ``inference_failed`` when the runner raises
            (the original message is suppressed because runtime errors may echo
            inputs), or ``io_contract_mismatch`` for a wrong output shape.
    """

    num_labels = len(id2label)
    logits_by_fixture: dict[str, list[list[float]]] = {}
    spans_by_fixture: dict[str, list[dict[str, Any]]] = {}
    latencies: list[float] = []
    for item in encoded:
        started = time.perf_counter()
        try:
            output = runner(item.feed)
        except ExportQualificationError:
            raise
        except Exception:
            raise ExportQualificationError(ExportReason.INFERENCE_FAILED) from None
        latencies.append((time.perf_counter() - started) * 1000.0)
        batch = _as_nested(output)
        if (
            not isinstance(batch, list)
            or len(batch) != 1
            or len(batch[0]) != item.bucket
            or any(len(row) != num_labels for row in batch[0])
        ):
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
        rows = [[float(value) for value in row] for row in batch[0]]
        logits_by_fixture[item.fixture_id] = [
            rows[index] for index in range(item.bucket) if item.attention[index]
        ]
        spans_by_fixture[item.fixture_id] = decode_token_spans(
            rows,
            offsets=item.offsets,
            attention=item.attention,
            id2label=id2label,
        )
    return logits_by_fixture, spans_by_fixture, latencies


def direct_identifier_labels(fixtures: Sequence[Any]) -> tuple[str, ...]:
    """Return canonical gold labels classified as direct identifiers."""
    labels = set()
    for fixture in fixtures:
        for span in fixture.gold_spans:
            label = normalize_label(span.label, fixture.language)
            if policy_label_for(label, fixture.language) == DIRECT_IDENTIFIER:
                labels.add(label)
    return tuple(sorted(labels))


def recall_by_label(
    fixtures: Sequence[Any],
    spans_by_fixture: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    labels: Sequence[str],
) -> dict[str, float]:
    """Return grapheme recall for *labels* over the whole suite."""
    from openmed.eval.metrics import compute_recall_slices

    gold: list[dict[str, Any]] = []
    predicted: list[dict[str, Any]] = []
    offset = 0
    for fixture in fixtures:
        for span in fixture.gold_spans:
            gold.append(
                {
                    "label": span.label,
                    "start": span.start + offset,
                    "end": span.end + offset,
                    "text": span.text,
                    "language": fixture.language,
                }
            )
        for span in spans_by_fixture.get(str(fixture.fixture_id), ()):
            start, end = int(span["start"]), int(span["end"])
            predicted.append(
                {
                    "label": str(span["label"]),
                    "start": start + offset,
                    "end": end + offset,
                    "text": fixture.text[start:end],
                    "language": fixture.language,
                }
            )
        offset += len(fixture.text) + 1
    by_label = compute_recall_slices(gold, predicted).by_label
    return {label: float(by_label.get(label, 0.0)) for label in labels}


def critical_leaks(
    fixtures: Sequence[Any],
    spans_by_fixture: Mapping[str, Sequence[Mapping[str, Any]]],
) -> set[tuple[str, int, int, str]]:
    """Return high-risk gold spans not fully covered by any predicted span.

    Coverage is label-agnostic: a redaction that hides every character of a
    critical identifier under any label is not a leak. The returned keys stay
    in memory only; reports record counts per label.
    """

    leaks: set[tuple[str, int, int, str]] = set()
    for fixture in fixtures:
        fixture_id = str(fixture.fixture_id)
        covered: set[int] = set()
        for span in spans_by_fixture.get(fixture_id, ()):
            covered.update(range(int(span["start"]), int(span["end"])))
        for span in fixture.gold_spans:
            label = normalize_label(span.label, fixture.language)
            if risk_level_for(label, fixture.language) != RISK_HIGH:
                continue
            positions = {
                index
                for index in range(span.start, span.end)
                if not fixture.text[index].isspace()
            }
            if not positions <= covered:
                leaks.add((fixture_id, span.start, span.end, label))
    return leaks


def qualify_variant(
    *,
    candidate: VariantCandidate,
    runtime: str,
    support: RuntimeSupport,
    fixtures: Sequence[Any],
    encoded: Sequence[EncodedFixture],
    id2label: Mapping[Any, str],
    reference_logits: Mapping[str, Sequence[Sequence[float]]],
    reference_spans: Mapping[str, Sequence[Mapping[str, Any]]],
    artifact_root: str | Path | None = None,
    direct_identifier_recall_floor: float | None = None,
) -> VariantQualification:
    """Run every gate for one candidate against precomputed reference outputs.

    Unsupported precision, quantization, or provider combinations are rejected
    by *support* before the candidate runs. Inference failures and wrong
    output shapes are recorded as failing reasons rather than raised, so one
    broken variant never hides the evidence for its siblings.

    Raises:
        ExportQualificationError: For combinations *support* does not declare.
    """

    support.validate(
        precision=candidate.precision,
        quantization=candidate.quantization,
        provider=candidate.provider,
    )
    policy = QualificationPolicy.for_variant(
        precision=candidate.precision,
        quantization=candidate.quantization,
        format_name=candidate.name,
        direct_identifier_recall_floor=direct_identifier_recall_floor,
    )
    relpaths = _relative_artifacts(candidate.artifact_paths, artifact_root)
    size_bytes = _artifacts_size(candidate.artifact_paths)
    buckets = tuple(sorted({item.bucket for item in encoded}))
    base: dict[str, Any] = {
        "name": candidate.name,
        "runtime": runtime,
        "precision": candidate.precision,
        "quantization": candidate.quantization,
        "provider": candidate.provider,
        "target_provider": candidate.target_provider,
        "policy": policy,
        "buckets_evaluated": buckets,
        "artifacts": relpaths,
        "size_bytes": size_bytes,
        "evidence": dict(candidate.evidence),
    }

    rss_before = _peak_rss_bytes()
    try:
        logits, spans, latencies = collect_runner_outputs(
            candidate.runner, encoded, id2label=id2label
        )
    except ExportQualificationError as exc:
        return VariantQualification(
            passed=False,
            reasons=(exc.reason_code,),
            logit_parity={},
            span_parity={},
            direct_identifier_recall={},
            critical_leakage={},
            quantization_delta=None,
            **base,
        )
    rss_after = _peak_rss_bytes()

    reasons: list[str] = []
    reference_rows = [
        row for key in sorted(reference_logits) for row in reference_logits[key]
    ]
    candidate_rows = [
        row for key in sorted(reference_logits) for row in logits.get(key, [])
    ]
    parity = evaluate_onnx_logit_parity(
        [reference_rows],
        [candidate_rows],
        atol=policy.logit_atol,
        rtol=policy.logit_rtol,
    )
    logit_payload = {
        "within_tolerance": parity.logits_within_tolerance,
        "max_abs_diff": _finite_or_none(parity.max_abs_diff),
        "max_rel_diff": _finite_or_none(parity.max_rel_diff),
        "argmax_mismatches": parity.token_mismatches,
        "tokens_evaluated": parity.tokens_evaluated,
        "gated": policy.gate_logits,
    }
    if policy.gate_logits and not parity.logits_within_tolerance:
        reasons.append(ExportReason.LOGIT_PARITY_FAILED.value)

    span_payload = _span_parity(reference_spans, spans, tolerance=policy.span_tolerance)
    span_payload["gated"] = policy.gate_spans
    if policy.gate_spans and not span_payload["identical"]:
        reasons.append(ExportReason.SPAN_PARITY_FAILED.value)

    labels = direct_identifier_labels(fixtures)
    reference_recall = recall_by_label(fixtures, reference_spans, labels=labels)
    candidate_recall = recall_by_label(fixtures, spans, labels=labels)
    deltas = {
        label: max(reference_recall[label] - candidate_recall[label], 0.0)
        for label in labels
    }
    max_delta = max(deltas.values()) if deltas else None
    floor_failed = direct_identifier_recall_floor is not None and any(
        value < direct_identifier_recall_floor for value in candidate_recall.values()
    )
    delta_failed = max_delta is None or max_delta > policy.recall_delta_limit + 1e-12
    recall_payload = {
        "labels": list(labels),
        "reference": reference_recall,
        "candidate": candidate_recall,
        "per_label_delta": deltas,
        "max_delta": max_delta,
        "limit": policy.recall_delta_limit,
        "floor": direct_identifier_recall_floor,
        "passed": not (delta_failed or floor_failed),
    }

    quant_payload: dict[str, Any] | None = None
    if candidate.quantization != QUANTIZATION_NONE:
        quant = evaluate_quant_recall_delta(
            format_name=f"{runtime}-{candidate.quantization}",
            candidate_recall=candidate_recall,
            parent_recall=reference_recall,
            labels=labels or None,
        )
        quant_payload = quant.to_dict()
        if not quant.passed:
            reasons.append(
                ExportReason.MISSING_EVIDENCE.value
                if quant.source == "missing_evidence"
                else ExportReason.QUANTIZATION_DELTA_EXCEEDED.value
            )
        if floor_failed:
            reasons.append(ExportReason.DIRECT_IDENTIFIER_RECALL_FAILED.value)
    elif delta_failed or floor_failed:
        reasons.append(
            ExportReason.MISSING_EVIDENCE.value
            if max_delta is None
            else ExportReason.DIRECT_IDENTIFIER_RECALL_FAILED.value
        )

    reference_leaks = critical_leaks(fixtures, reference_spans)
    candidate_leaks = critical_leaks(fixtures, spans)
    new_leaks = candidate_leaks - reference_leaks
    leak_payload = {
        "reference_leaked_spans": len(reference_leaks),
        "candidate_leaked_spans": len(candidate_leaks),
        "new_leaks": len(new_leaks),
        "new_leaks_by_label": _count_by_label(new_leaks),
        "limit": policy.max_new_critical_leaks,
        "passed": len(new_leaks) <= policy.max_new_critical_leaks,
    }
    if not leak_payload["passed"]:
        reasons.append(ExportReason.CRITICAL_LEAKAGE_DETECTED.value)

    return VariantQualification(
        passed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        logit_parity=logit_payload,
        span_parity=span_payload,
        direct_identifier_recall=recall_payload,
        critical_leakage=leak_payload,
        quantization_delta=quant_payload,
        latency_ms=_latency_summary(latencies),
        process_peak_rss_bytes=max(rss_before or 0, rss_after or 0) or None,
        **base,
    )


def qualify_export(
    *,
    family: str,
    adapter_version: int,
    runtime: str,
    support: RuntimeSupport,
    io_contract: ExportIOContract,
    tokenizer: Any,
    id2label: Mapping[Any, str],
    reference_runner: LogitRunner,
    candidates: Sequence[VariantCandidate],
    source: Mapping[str, Any],
    artifact_root: str | Path,
    artifact_paths: Iterable[str | Path] = (),
    fixtures_path: str | Path | None = None,
    exclude_artifact_names: Iterable[str] = (),
    direct_identifier_recall_floor: float | None = None,
    environment: Mapping[str, Any] | None = None,
) -> ExportQualificationReport:
    """Qualify every candidate variant of one export against its reference.

    Args:
        family: Family adapter name, for example ``"modernbert"``.
        adapter_version: Version of the family adapter that traced the graph.
        runtime: ``"coreml"`` or ``"onnx"``.
        support: Declared support table for this family and runtime.
        io_contract: Tensor names, buckets, and bounds of the artifacts.
        tokenizer: Fast tokenizer that produced the reference encodings.
        id2label: Label map shared by the reference and the artifacts.
        reference_runner: PyTorch (or MLX) reference logits runner.
        candidates: Exported variants to qualify.
        source: Output of :func:`source_identity`.
        artifact_root: Directory every artifact path is reported relative to.
        artifact_paths: Extra files (tokenizer, config, labels) to digest.
        fixtures_path: Synthetic suite; defaults to the packaged PHI suite.
        exclude_artifact_names: Metadata filenames excluded from digests, such
            as manifests rewritten after qualification.
        direct_identifier_recall_floor: Optional absolute recall floor.
        environment: Converter environment; defaults to the installed one.

    Returns:
        The complete report. Call :meth:`ExportQualificationReport.raise_for_failure`
        to fail closed.

    Raises:
        ExportQualificationError: For contract, tokenizer, label-map, or
            support violations detected before inference.
    """

    label_identity = label_map_identity(id2label)
    token_identity = tokenizer_identity(tokenizer)
    fixtures, suite = load_qualification_fixtures(fixtures_path)
    encoded = encode_fixtures(fixtures, tokenizer, io_contract)
    reference_logits, reference_spans, _ = collect_runner_outputs(
        reference_runner, encoded, id2label=id2label
    )
    variants = tuple(
        qualify_variant(
            candidate=candidate,
            runtime=runtime,
            support=support,
            fixtures=fixtures,
            encoded=encoded,
            id2label=id2label,
            reference_logits=reference_logits,
            reference_spans=reference_spans,
            artifact_root=artifact_root,
            direct_identifier_recall_floor=direct_identifier_recall_floor,
        )
        for candidate in candidates
    )
    all_paths = [
        *artifact_paths,
        *(path for candidate in candidates for path in candidate.artifact_paths),
    ]
    digests = artifact_file_digests(
        all_paths,
        root=artifact_root,
        exclude_names={EXPORT_QUALIFICATION_FILENAME, *exclude_artifact_names},
    )
    return ExportQualificationReport(
        family=family,
        adapter_version=int(adapter_version),
        runtime=runtime,
        support=support,
        source=dict(source),
        tokenizer=token_identity,
        label_map=label_identity,
        io_contract=io_contract,
        environment=dict(environment or converter_environment()),
        suite=suite,
        variants=variants,
        artifacts=digests,
    )


def write_qualification_report(
    report: ExportQualificationReport | Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Write *report* as sorted, indented JSON and return the path."""
    payload = (
        report.to_dict()
        if isinstance(report, ExportQualificationReport)
        else dict(report)
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def summarize_report(report: ExportQualificationReport) -> dict[str, Any]:
    """Return the compact block embedded in converter manifests."""
    payload = report.to_dict()
    return {
        "format": payload["format"],
        "schema_version": payload["schema_version"],
        "family": payload["family"],
        "adapter_version": payload["adapter_version"],
        "passed": payload["passed"],
        "reasons": payload["reasons"],
        "qualified_variants": payload["qualified_variants"],
        "source_revision": payload["source"].get("revision"),
        "tokenizer_sha256": payload["tokenizer"].get("sha256"),
        "label_map_sha256": payload["label_map"].get("sha256"),
        "sequence_buckets": payload["io_contract"]["sequence_buckets"],
        "stable_sha256": payload["stable_sha256"],
        "variants": {
            item["name"]: {
                "claim": item["claim"],
                "passed": item["passed"],
                "reasons": item["reasons"],
            }
            for item in payload["variants"]
        },
    }


# ---------------------------------------------------------------------------
# Offline verification and consumer-side checks
# ---------------------------------------------------------------------------


def load_qualification_report(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a qualification report.

    Raises:
        ExportQualificationError: ``report_invalid`` for a malformed report.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ExportQualificationError(ExportReason.REPORT_INVALID) from None
    _validate_report_structure(payload)
    return payload


def verify_qualification_report(
    report: str | Path | Mapping[str, Any],
    *,
    artifact_root: str | Path,
    tokenizer: Any | None = None,
    id2label: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """Verify a qualification report against the artifacts on disk.

    Checks that every required identity field is present, recomputes every
    artifact digest, the stable report digest, and optionally the tokenizer
    and label-map identities.

    Raises:
        ExportQualificationError: ``report_invalid``, ``artifact_missing``,
            ``artifact_digest_mismatch``, ``tokenizer_identity_mismatch``, or
            ``label_map_mismatch``.
    """

    payload = (
        load_qualification_report(report)
        if isinstance(report, (str, Path))
        else dict(report)
    )
    _validate_report_structure(payload)
    expected_stable = payload.get("stable_sha256")
    body = {key: value for key, value in payload.items() if key != "stable_sha256"}
    if expected_stable != _stable_report_digest(body):
        raise ExportQualificationError(ExportReason.REPORT_INVALID)

    root = Path(artifact_root)
    for relative, expected in payload["artifacts"].items():
        candidate = root / str(relative)
        if ".." in Path(str(relative)).parts or not candidate.is_file():
            raise ExportQualificationError(ExportReason.ARTIFACT_MISSING)
        if sha256_file(candidate) != expected:
            raise ExportQualificationError(ExportReason.ARTIFACT_DIGEST_MISMATCH)
    if tokenizer is not None:
        if tokenizer_identity(tokenizer)["sha256"] != payload["tokenizer"]["sha256"]:
            raise ExportQualificationError(ExportReason.TOKENIZER_IDENTITY_MISMATCH)
    if id2label is not None:
        if label_map_identity(id2label)["sha256"] != payload["label_map"]["sha256"]:
            raise ExportQualificationError(ExportReason.LABEL_MAP_MISMATCH)
    return payload


def check_qualified_request(
    report: Mapping[str, Any],
    *,
    runtime: str,
    variant: str | None = None,
    artifact: str | None = None,
    providers: Sequence[str] | None = None,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Reject an inference request the report does not qualify.

    Call this before creating a runtime session. A variant qualified only on a
    CPU proxy provider is accepted for that proxy provider, never for its
    target provider.

    Args:
        report: Loaded qualification report.
        runtime: Runtime the caller is about to use.
        variant: Variant name to run, or ``None`` to match by *artifact*.
        artifact: Artifact path relative to the report's root.
        providers: Execution providers the caller will request.
        sequence_length: Padded sequence length the caller will feed.

    Returns:
        The qualified variant entry.

    Raises:
        ExportQualificationError: ``unsupported_runtime``,
            ``variant_not_qualified``, ``unsupported_provider``, or
            ``unsupported_sequence_length``.
    """

    _validate_report_structure(report)
    if report["runtime"] != runtime:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_RUNTIME)
    entry = _find_variant(report, variant=variant, artifact=artifact)
    if entry is None or not entry.get("passed"):
        raise ExportQualificationError(ExportReason.VARIANT_NOT_QUALIFIED)
    qualified_provider = entry.get("provider")
    for provider in providers or ():
        if qualified_provider is not None and provider != qualified_provider:
            raise ExportQualificationError(ExportReason.UNSUPPORTED_PROVIDER)
    if sequence_length is not None:
        ExportIOContract.from_dict(report["io_contract"]).bucket_for(sequence_length)
    return dict(entry)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_report_structure(payload: Any) -> None:
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != EXPORT_QUALIFICATION_FORMAT
        or payload.get("schema_version") != EXPORT_QUALIFICATION_SCHEMA_VERSION
        or any(key not in payload for key in _REPORT_REQUIRED_KEYS)
        or not isinstance(payload.get("artifacts"), Mapping)
        or not payload["artifacts"]
        or not isinstance(payload.get("variants"), list)
        or not payload["variants"]
        or not (payload.get("source") or {}).get("revision")
        or not (payload.get("tokenizer") or {}).get("sha256")
        or not (payload.get("label_map") or {}).get("sha256")
    ):
        raise ExportQualificationError(ExportReason.REPORT_INVALID)
    contract = payload["io_contract"]
    if not isinstance(contract, Mapping) or not contract.get("input_names"):
        raise ExportQualificationError(ExportReason.REPORT_INVALID)
    for entry in payload["variants"]:
        if not isinstance(entry, Mapping) or not entry.get("precision"):
            raise ExportQualificationError(ExportReason.REPORT_INVALID)


def _find_variant(
    report: Mapping[str, Any],
    *,
    variant: str | None,
    artifact: str | None,
) -> Mapping[str, Any] | None:
    for entry in report["variants"]:
        if variant is not None and entry.get("name") == variant:
            return entry
        if artifact is not None:
            names = [str(item) for item in entry.get("artifacts") or ()]
            if artifact in names or any(
                artifact.startswith(f"{name}/") for name in names
            ):
                return entry
    return None


def _span_parity(
    reference: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    tolerance: int,
) -> dict[str, Any]:
    fixtures_mismatched = 0
    spans_mismatched = 0
    reference_count = 0
    candidate_count = 0
    for fixture_id in sorted(set(reference) | set(candidate)):
        left = sorted(_span_key(span) for span in reference.get(fixture_id, ()))
        right = sorted(_span_key(span) for span in candidate.get(fixture_id, ()))
        reference_count += len(left)
        candidate_count += len(right)
        mismatches = abs(len(left) - len(right)) + sum(
            1
            for a, b in zip(left, right)
            if a[0] != b[0]
            or abs(a[1] - b[1]) > tolerance
            or abs(a[2] - b[2]) > tolerance
        )
        if mismatches:
            fixtures_mismatched += 1
            spans_mismatched += mismatches
    return {
        "identical": fixtures_mismatched == 0,
        "fixtures_compared": len(set(reference) | set(candidate)),
        "fixtures_mismatched": fixtures_mismatched,
        "spans_mismatched": spans_mismatched,
        "reference_span_count": reference_count,
        "candidate_span_count": candidate_count,
        "tolerance": tolerance,
    }


def _span_key(span: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(span["label"]), int(span["start"]), int(span["end"]))


def _count_by_label(keys: Iterable[tuple[str, int, int, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, _, _, label in keys:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"measured": False}
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "measured": True,
        "calls": len(values),
        "p50": statistics.median(values),
        "p95": ordered[index],
    }


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows has no resource module.
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _relative_artifacts(
    paths: Sequence[Path], root: str | Path | None
) -> tuple[str, ...]:
    if root is None:
        return tuple(Path(path).name for path in paths)
    root_path = Path(root).resolve()
    relpaths: list[str] = []
    for path in paths:
        try:
            relpaths.append(Path(path).resolve().relative_to(root_path).as_posix())
        except ValueError:
            relpaths.append(Path(path).name)
    return tuple(relpaths)


def _artifacts_size(paths: Sequence[Path]) -> int | None:
    total = 0
    found = False
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            total += sum(
                item.stat().st_size for item in path.rglob("*") if item.is_file()
            )
            found = True
        elif path.is_file():
            total += path.stat().st_size
            found = True
            sidecar = path.with_name(f"{path.name}.data")
            if sidecar.is_file():
                total += sidecar.stat().st_size
    return total if found else None


def _stable_report_digest(payload: Mapping[str, Any]) -> str:
    stable = json.loads(json.dumps(payload, sort_keys=True, default=str))
    stable.pop("stable_sha256", None)
    for entry in stable.get("variants") or ():
        for key in _VOLATILE_VARIANT_KEYS:
            entry.pop(key, None)
    return _canonical_digest(stable)


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return dict(getattr(config, "__dict__", {}))


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _as_nested(value: Any) -> Any:
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return to_list()
    if isinstance(value, (list, tuple)):
        return [_as_nested(item) for item in value]
    return value


def _as_rows(value: Any) -> list[list[float]]:
    nested = _as_nested(value)
    if (
        nested
        and isinstance(nested[0], list)
        and nested[0]
        and isinstance(nested[0][0], list)
    ):
        nested = nested[0]
    return [[float(item) for item in row] for row in nested]


def _argmax(row: Sequence[float]) -> int:
    best_index = 0
    best_value = -math.inf
    for index, value in enumerate(row):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


__all__ = [
    "DEFAULT_QUALIFICATION_SUITE",
    "EXPORT_QUALIFICATION_FILENAME",
    "EXPORT_QUALIFICATION_FORMAT",
    "EXPORT_QUALIFICATION_SCHEMA_VERSION",
    "OFFSET_CONTRACT",
    "EncodedFixture",
    "ExportIOContract",
    "ExportQualificationError",
    "ExportQualificationReport",
    "ExportReason",
    "LogitRunner",
    "QualificationPolicy",
    "RuntimeSupport",
    "VariantCandidate",
    "VariantQualification",
    "artifact_file_digests",
    "check_qualified_request",
    "collect_runner_outputs",
    "config_identity",
    "converter_environment",
    "critical_leaks",
    "decode_token_spans",
    "direct_identifier_labels",
    "encode_fixtures",
    "label_map_identity",
    "load_qualification_fixtures",
    "load_qualification_report",
    "qualify_export",
    "qualify_variant",
    "recall_by_label",
    "sha256_file",
    "source_identity",
    "summarize_report",
    "tokenizer_identity",
    "verify_qualification_report",
    "write_qualification_report",
]
