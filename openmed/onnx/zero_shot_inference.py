"""ONNX Runtime loader for qualified GLiNER span and GLiClass bundles.

:class:`ZeroShotOnnxModel` opens a bundle written by
:mod:`openmed.onnx.zero_shot_export` only when its qualification report is
present, structurally valid, matches every file digest on disk, and
qualifies the requested provider. Every request is then checked against the
bundle's declared contract *before* inference: candidate-label count and
validity, reserved prompt markers, word count, span width, sequence length
(inputs are never truncated), and an optional activation-memory budget. A
request outside the contract returns a typed
:class:`~openmed.ner.families.zero_shot_export.ZeroShotUnsupported` result
with a value-free reason code instead of running.

Only ``numpy``, ``onnxruntime``, and ``tokenizers`` are needed
(``pip install openmed[onnx-runtime]``). Nothing is downloaded, and source
text, labels, token ids, and logits are never logged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.capabilities import raise_missing_backend
from openmed.eval.export_qualification import (
    EXPORT_QUALIFICATION_FILENAME,
    ExportQualificationError,
    ExportReason,
    check_qualified_request,
    sha256_file,
    verify_qualification_report,
)
from openmed.eval.zero_shot_qualification import zero_shot_tokenizer_identity
from openmed.ner.families.zero_shot_export import (
    ONNX_CPU_PROVIDER,
    ONNX_RUNTIME,
    TASK_ZERO_SHOT_NER,
    ZERO_SHOT_MANIFEST_FILENAME,
    ZERO_SHOT_MANIFEST_FORMAT,
    ZERO_SHOT_MANIFEST_SCHEMA_VERSION,
    ZERO_SHOT_PROFILES,
    ZeroShotBounds,
    ZeroShotClassificationResult,
    ZeroShotModelSpec,
    ZeroShotPromptSpec,
    ZeroShotRequestError,
    ZeroShotSpanResult,
    ZeroShotUnsupported,
    check_labels,
    check_memory_budget,
    decode_gliclass,
    decode_gliner_spans,
    encode_gliclass,
    encode_gliner_span,
)
from openmed.onnx.zero_shot_export import (
    TOKENIZER_FILENAME,
    ZERO_SHOT_MODEL_FILENAME,
    check_session_contract,
    load_zero_shot_tokenizer,
)

_MANIFEST_REQUIRED_KEYS = (
    "profile",
    "family",
    "task",
    "source",
    "architecture",
    "prompt",
    "labels",
    "tokenizer",
    "tensors",
    "bounds",
    "decoder",
    "offset_contract",
    "files",
    "qualification",
)


class ZeroShotOnnxModel:
    """Run a qualified GLiNER span or GLiClass ONNX bundle.

    Args:
        bundle_dir: Directory written by
            :func:`~openmed.onnx.zero_shot_export.export_zero_shot_onnx`.
        providers: Execution providers to request; only the qualified CPU
            provider is accepted.
        memory_budget_bytes: Optional per-request activation budget. Requests
            whose deterministic estimate exceeds it are refused with
            ``unsupported_memory_budget``. An
            :class:`~openmed.onnx.ram_budget.RamBudget` is also accepted.
        session_options: Optional ``onnxruntime.SessionOptions``.

    Raises:
        ExportQualificationError: ``missing_evidence`` without a report,
            ``report_invalid`` for a malformed manifest or report,
            ``artifact_missing`` / ``artifact_digest_mismatch`` for changed
            files, ``unsupported_task`` for a non-zero-shot report,
            ``variant_not_qualified``, ``unsupported_provider``,
            ``tokenizer_identity_mismatch``, or ``io_contract_mismatch``.
    """

    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        providers: Sequence[str] = (ONNX_CPU_PROVIDER,),
        memory_budget_bytes: int | Any | None = None,
        session_options: Any | None = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest = _load_manifest(self.bundle_dir)
        report_path = self.bundle_dir / EXPORT_QUALIFICATION_FILENAME
        if not report_path.is_file():
            raise ExportQualificationError(ExportReason.MISSING_EVIDENCE)
        report = verify_qualification_report(report_path, artifact_root=self.bundle_dir)
        profile = ZERO_SHOT_PROFILES.get(str(self.manifest["family"]))
        if (
            profile is None
            or report.get("task") != profile.task
            or self.manifest["task"] != profile.task
            or report.get("family") != profile.family
        ):
            raise ExportQualificationError(ExportReason.UNSUPPORTED_TASK)
        self.profile = profile
        variant = check_qualified_request(
            report,
            runtime=ONNX_RUNTIME,
            artifact=ZERO_SHOT_MODEL_FILENAME,
            providers=list(providers),
        )
        _verify_manifest_files(self.bundle_dir, self.manifest)
        self.bounds = ZeroShotBounds.from_dict(self.manifest["bounds"])
        self.prompt = ZeroShotPromptSpec.from_dict(self.manifest["prompt"])
        self.spec = _model_spec(profile, self.manifest["architecture"])
        self.threshold = float(self.manifest["decoder"]["threshold"])
        self.memory_budget_bytes = _budget_bytes(memory_budget_bytes)
        self.tokenizer = load_zero_shot_tokenizer(self.bundle_dir / TOKENIZER_FILENAME)
        identity = zero_shot_tokenizer_identity(self.tokenizer)["sha256"]
        if identity != report["tokenizer"]["sha256"]:
            raise ExportQualificationError(ExportReason.TOKENIZER_IDENTITY_MISMATCH)
        self.qualification = {
            "profile": profile.name,
            "family": profile.family,
            "task": profile.task,
            "variant": variant.get("name"),
            "claim": variant.get("claim"),
            "provider": variant.get("provider"),
            "bounds": self.bounds.to_dict(),
        }
        self._session = _create_session(
            self.bundle_dir / ZERO_SHOT_MODEL_FILENAME,
            providers=list(providers),
            session_options=session_options,
        )
        check_session_contract(self._session, profile)

    def predict_entities(
        self,
        text: str,
        labels: Sequence[str],
        *,
        threshold: float | None = None,
        flat_ner: bool = True,
        span_width: int | None = None,
    ) -> ZeroShotSpanResult | ZeroShotUnsupported:
        """Detect spans of *labels* in *text* with a GLiNER span bundle.

        Args:
            text: Source text; never logged or stored.
            labels: Candidate labels in priority order (ties keep this order).
            threshold: Score threshold; defaults to the qualified one.
            flat_ner: Resolve overlaps greedily (default) or keep nested spans.
            span_width: Optional narrower span width.

        Returns:
            Spans with code-point offsets into *text*, or a typed refusal.
        """

        if self.profile.task != TASK_ZERO_SHOT_NER:
            return ZeroShotUnsupported(
                reason=ExportReason.UNSUPPORTED_TASK, dimension="task"
            )
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        cutoff = self._threshold(threshold)
        try:
            checked = check_labels(labels, bounds=self.bounds, prompt=self.prompt)
            encoding = encode_gliner_span(
                self.tokenizer,
                text,
                checked,
                prompt=self.prompt,
                bounds=self.bounds,
                graph_span_width=int(self.spec.span_width or 1),
                span_width=span_width,
            )
            check_memory_budget(
                self.spec,
                budget_bytes=self.memory_budget_bytes,
                sequence_length=encoding.token_count,
                num_labels=encoding.num_labels,
                num_words=len(encoding.word_offsets),
                span_width=encoding.span_width,
            )
        except ZeroShotRequestError as error:
            return ZeroShotUnsupported.from_error(error)
        if not encoding.word_offsets:
            return ZeroShotSpanResult(spans=())
        logits = self._run(encoding.feed)
        return decode_gliner_spans(
            logits,
            word_offsets=encoding.word_offsets,
            labels=checked,
            threshold=cutoff,
            flat_ner=flat_ner,
            span_width=encoding.span_width,
        )

    def classify(
        self,
        text: str,
        labels: Sequence[str],
        *,
        threshold: float | None = None,
        instruction: str | None = None,
    ) -> ZeroShotClassificationResult | ZeroShotUnsupported:
        """Score candidate *labels* for *text* with a GLiClass bundle.

        Args:
            text: Source text; never logged or stored.
            labels: Candidate labels in caller order.
            threshold: Acceptance threshold; defaults to the qualified one.
            instruction: Optional instruction prompt placed after the labels.

        Returns:
            Ranked label scores, or a typed refusal.
        """

        if self.profile.task == TASK_ZERO_SHOT_NER:
            return ZeroShotUnsupported(
                reason=ExportReason.UNSUPPORTED_TASK, dimension="task"
            )
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if instruction is not None and not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        cutoff = self._threshold(threshold)
        try:
            checked = check_labels(labels, bounds=self.bounds, prompt=self.prompt)
            encoding = encode_gliclass(
                self.tokenizer,
                text,
                checked,
                prompt=self.prompt,
                bounds=self.bounds,
                instruction=instruction,
            )
            check_memory_budget(
                self.spec,
                budget_bytes=self.memory_budget_bytes,
                sequence_length=encoding.token_count,
                num_labels=encoding.num_labels,
            )
        except ZeroShotRequestError as error:
            return ZeroShotUnsupported.from_error(error)
        logits = self._run(encoding.feed)
        return decode_gliclass(logits, labels=checked, threshold=cutoff)

    def _threshold(self, threshold: float | None) -> float:
        value = self.threshold if threshold is None else float(threshold)
        if not 0.0 < value < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return value

    def _run(self, feed: Mapping[str, Any]) -> Any:
        import numpy as np

        prepared = {name: feed[name] for name in self.profile.input_names}
        try:
            output = self._session.run(["logits"], prepared)[0]
        except Exception:
            # Runtime errors can echo tensor contents; keep them out.
            raise ExportQualificationError(ExportReason.INFERENCE_FAILED) from None
        return np.asarray(output, dtype=np.float32)


def load_zero_shot_onnx_model(
    bundle_dir: str | Path, **kwargs: Any
) -> ZeroShotOnnxModel:
    """Return a :class:`ZeroShotOnnxModel` for a local bundle."""
    return ZeroShotOnnxModel(bundle_dir, **kwargs)


def _load_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / ZERO_SHOT_MANIFEST_FILENAME
    if not path.is_file():
        raise ExportQualificationError(ExportReason.MISSING_EVIDENCE)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ExportQualificationError(ExportReason.REPORT_INVALID) from None
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("format") != ZERO_SHOT_MANIFEST_FORMAT
        or manifest.get("schema_version") != ZERO_SHOT_MANIFEST_SCHEMA_VERSION
        or any(key not in manifest for key in _MANIFEST_REQUIRED_KEYS)
        or not isinstance(manifest.get("files"), Mapping)
    ):
        raise ExportQualificationError(ExportReason.REPORT_INVALID)
    return dict(manifest)


def _verify_manifest_files(bundle_dir: Path, manifest: Mapping[str, Any]) -> None:
    for relative, expected in manifest["files"].items():
        path = bundle_dir / str(relative)
        if ".." in Path(str(relative)).parts or not path.is_file():
            raise ExportQualificationError(ExportReason.ARTIFACT_MISSING)
        if sha256_file(path) != expected:
            raise ExportQualificationError(ExportReason.ARTIFACT_DIGEST_MISMATCH)


def _model_spec(profile: Any, payload: Mapping[str, Any]) -> ZeroShotModelSpec:
    try:
        return ZeroShotModelSpec(
            profile=profile,
            hidden_size=int(payload["hidden_size"]),
            encoder_hidden_size=int(payload["encoder_hidden_size"]),
            num_hidden_layers=int(payload["num_hidden_layers"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            intermediate_size=int(payload["intermediate_size"]),
            max_position_embeddings=int(payload["max_position_embeddings"]),
            marker_token_index=int(payload["marker_token_index"]),
            text_token_index=payload.get("text_token_index"),
            example_token_index=payload.get("example_token_index"),
            span_width=payload.get("span_width"),
            max_words=payload.get("max_words"),
            options=dict(payload.get("options") or {}),
        )
    except (KeyError, TypeError, ValueError):
        raise ExportQualificationError(ExportReason.REPORT_INVALID) from None


def _budget_bytes(value: Any) -> int | None:
    if value is None:
        return None
    limit = getattr(value, "max_bytes", value)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("memory budget must be a positive number of bytes")
    return limit


def _create_session(
    path: Path, *, providers: Sequence[str], session_options: Any | None
) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise_missing_backend("onnx", feature="ONNX zero-shot inference", cause=exc)
    session = ort.InferenceSession(
        str(path), sess_options=session_options, providers=list(providers)
    )
    if ONNX_CPU_PROVIDER not in session.get_providers():
        raise ExportQualificationError(ExportReason.UNSUPPORTED_PROVIDER)
    return session


__all__ = ["ZeroShotOnnxModel", "load_zero_shot_onnx_model"]
