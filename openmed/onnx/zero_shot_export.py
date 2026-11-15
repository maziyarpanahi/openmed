"""Export GLiNER span and GLiClass checkpoints to qualified ONNX graphs.

The source is a local OpenMed MLX artifact directory (``config.json``,
``weights.safetensors``, ``tokenizer.json``) for one of the declared
profiles in :mod:`openmed.ner.families.zero_shot_export`. The export:

1. resolves the profile and rejects relation extraction, quantized sources,
   repository-code requirements, and configuration variants the portable
   graph does not reproduce, all before any weights load;
2. validates the requested bounds, precision, provider, and opset against
   the profile's declared ONNX Runtime support;
3. loads the weights into the PyTorch reference graph
   (:mod:`openmed.torch.zero_shot_export`) and traces it to ONNX with
   dynamic batch, sequence, word, span, and label axes;
4. checks the graph structurally (ONNX checker, tensor names and ranks,
   operator inventory) and qualifies it on ``CPUExecutionProvider`` against
   the PyTorch reference with :mod:`openmed.eval.zero_shot_qualification`;
5. writes ``openmed-zero-shot.json`` (profile, contracts, bounds, digests,
   evidence references) and ``openmed-export-qualification.json``.

A failing gate raises :class:`~openmed.eval.export_qualification.ExportQualificationError`
after the evidence is written. Loading never touches the network, never runs
repository code, and never falls back to a remote model. Core ML has no
qualified zero-shot profile and fails with ``unsupported_runtime``.

Example::

    python -m openmed.onnx.zero_shot_export --source ./gliner-mlx \\
        --output ./dist/gliner-onnx --max-labels 16 --max-sequence-length 384
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.capabilities import raise_missing_backend
from openmed.eval.export_qualification import (
    EXPORT_QUALIFICATION_FILENAME,
    QUANTIZATION_NONE,
    SHAPE_MODE_DYNAMIC,
    ExportQualificationError,
    ExportReason,
    VariantCandidate,
    artifact_file_digests,
    source_identity,
    summarize_report,
    write_qualification_report,
)
from openmed.eval.zero_shot_qualification import (
    ZeroShotQualificationReport,
    qualify_zero_shot_export,
    zero_shot_tokenizer_identity,
)
from openmed.ner.families.zero_shot_export import (
    DEFAULT_THRESHOLD,
    GLINER_WORD_SPLITTER,
    ONNX_CPU_PROVIDER,
    ONNX_RUNTIME,
    TASK_ZERO_SHOT_NER,
    ZERO_SHOT_CONTRACT_VERSION,
    ZERO_SHOT_MANIFEST_FILENAME,
    ZERO_SHOT_MANIFEST_FORMAT,
    ZERO_SHOT_MANIFEST_SCHEMA_VERSION,
    ZERO_SHOT_OFFSET_CONTRACTS,
    ZERO_SHOT_ONNX_OPSET,
    ZeroShotBounds,
    ZeroShotModelSpec,
    ZeroShotProfile,
    ZeroShotPromptSpec,
    default_bounds,
    encode_gliclass,
    encode_gliner_span,
    estimate_activation_bytes,
    label_schema_identity,
    reject_raw_first_token_pooling,
    resolve_prompt_spec,
    resolve_zero_shot_profile,
    validate_zero_shot_config,
)

logger = logging.getLogger(__name__)

ZERO_SHOT_MODEL_FILENAME = "model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"
CONFIG_FILENAME = "config.json"
QUALIFIED_VARIANT_NAME = "onnx-fp32"
MEMORY_ESTIMATOR = "openmed-float32-activations-v1"
# Synthetic tracing request; it never contains caller data.
_TRACE_TEXT = "synthetic trace sample for the export graph ."
_TRACE_LABELS = ("alpha", "beta")
_UNSUPPORTED_SURFACES = {
    "coreml": ExportReason.UNSUPPORTED_RUNTIME.value,
    "relation_extraction": ExportReason.UNSUPPORTED_TASK.value,
    "providers_other_than_cpu": ExportReason.UNSUPPORTED_PROVIDER.value,
    "quantized_variants": ExportReason.UNSUPPORTED_QUANTIZATION.value,
    "float16": ExportReason.UNSUPPORTED_PRECISION.value,
}


@dataclass(frozen=True)
class ZeroShotSource:
    """A validated local zero-shot checkpoint, before any weights load.

    Attributes:
        directory: Local MLX artifact directory.
        raw_config: ``config.json`` as written.
        config: Configuration with the MLX family defaults applied.
        profile: Matched export profile.
        spec: Validated architecture facts.
        tokenizer: ``tokenizers.Tokenizer`` from ``tokenizer.json``.
        prompt: Prompt contract checked against the tokenizer.
        weights_path: ``weights.safetensors`` path.
    """

    directory: Path
    raw_config: Mapping[str, Any]
    config: Mapping[str, Any]
    profile: ZeroShotProfile
    spec: ZeroShotModelSpec
    tokenizer: Any
    prompt: ZeroShotPromptSpec
    weights_path: Path


@dataclass(frozen=True)
class ZeroShotExportResult:
    """Paths and evidence produced by :func:`export_zero_shot_onnx`."""

    output_dir: Path
    model_path: Path
    manifest_path: Path
    report_path: Path | None
    report: ZeroShotQualificationReport | None

    @property
    def passed(self) -> bool:
        """Return whether the export was qualified."""
        return self.report is not None and self.report.passed


def load_zero_shot_source(source_dir: str | Path) -> ZeroShotSource:
    """Resolve and validate a local zero-shot checkpoint without loading weights.

    Raises:
        ExportQualificationError: ``artifact_missing`` for a missing directory,
            config, tokenizer, or ``weights.safetensors`` (Hub ids are never
            resolved here); plus every profile and variant refusal of
            :func:`~openmed.ner.families.zero_shot_export.resolve_zero_shot_profile`
            and :func:`~openmed.ner.families.zero_shot_export.validate_zero_shot_config`.
    """

    from openmed.mlx.artifact import load_artifact_config
    from openmed.mlx.models import normalize_model_config

    directory = Path(source_dir)
    if not directory.is_dir() or not (directory / CONFIG_FILENAME).is_file():
        raise ExportQualificationError(ExportReason.ARTIFACT_MISSING)
    try:
        manifest, raw_config = load_artifact_config(directory)
    except (OSError, ValueError):
        raise ExportQualificationError(ExportReason.ARTIFACT_MISSING) from None
    profile = resolve_zero_shot_profile(raw_config, manifest)
    if profile.task != TASK_ZERO_SHOT_NER:
        reject_raw_first_token_pooling(raw_config)
    config = normalize_model_config(dict(raw_config), manifest=manifest)
    spec = validate_zero_shot_config(profile, config)
    weights_path = directory / "weights.safetensors"
    tokenizer_path = directory / TOKENIZER_FILENAME
    if not weights_path.is_file() or not tokenizer_path.is_file():
        raise ExportQualificationError(ExportReason.ARTIFACT_MISSING)
    tokenizer = load_zero_shot_tokenizer(tokenizer_path)
    prompt = resolve_prompt_spec(
        profile,
        config,
        spec,
        tokenizer,
        prompt_spec=(manifest or {}).get("prompt_spec"),
    )
    return ZeroShotSource(
        directory=directory,
        raw_config=dict(raw_config),
        config=config,
        profile=profile,
        spec=spec,
        tokenizer=tokenizer,
        prompt=prompt,
        weights_path=weights_path,
    )


def load_zero_shot_tokenizer(path: str | Path) -> Any:
    """Load ``tokenizer.json`` with truncation and padding disabled.

    Raises:
        MissingBackendError: When ``tokenizers`` is not installed.
    """

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise_missing_backend(
            "onnx", feature="GLiNER and GLiClass tokenization", cause=exc
        )
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def export_zero_shot_onnx(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    max_labels: int | None = None,
    max_sequence_length: int | None = None,
    max_words: int | None = None,
    max_span_width: int | None = None,
    max_label_chars: int | None = None,
    precision: str = "float32",
    quantization: str = QUANTIZATION_NONE,
    providers: Sequence[str] = (ONNX_CPU_PROVIDER,),
    opset: int = ZERO_SHOT_ONNX_OPSET,
    threshold: float = DEFAULT_THRESHOLD,
    qualify: bool = True,
    qualification_fixtures_path: str | Path | None = None,
    label_prompts: Mapping[str, str] | None = None,
    expected_source_revision: str | None = None,
    direct_identifier_recall_floor: float | None = None,
) -> ZeroShotExportResult:
    """Export one GLiNER span or GLiClass checkpoint to a qualified ONNX bundle.

    Args:
        source_dir: Local MLX artifact directory.
        output_dir: Bundle directory to write.
        max_labels: Candidate-label bound; defaults to 32 (at most 64).
        max_sequence_length: Token bound including the prompt; defaults to
            ``min(512, max_position_embeddings)``.
        max_words: GLiNER word bound; defaults to the checkpoint ``max_len``.
        max_span_width: GLiNER span-width bound; defaults to (and may not
            exceed) the checkpoint ``max_width``.
        max_label_chars: Code-point bound for one label.
        precision: Only ``"float32"`` is declared.
        quantization: Only ``"none"`` is declared.
        providers: Execution providers to qualify; only the CPU provider is
            declared.
        opset: ONNX opset, at least 18.
        threshold: Decoder threshold recorded in the contract.
        qualify: ``False`` writes the graph with ``status: not_run`` and
            claims nothing; the runtime then refuses to load it.
        qualification_fixtures_path: Synthetic suite; defaults to the
            packaged synthetic PHI suite (GLiNER) or zero-shot
            classification suite (GLiClass).
        label_prompts: GLiNER only: canonical label to candidate label.
        expected_source_revision: Optional pinned source revision.
        direct_identifier_recall_floor: Optional absolute recall floor.

    Returns:
        The written bundle and its qualification report.

    Raises:
        ExportQualificationError: For any unsupported request, before weights
            load; or for a failing gate, after the evidence is written.
    """

    source = load_zero_shot_source(source_dir)
    profile, spec = source.profile, source.spec
    support = profile.support_for(ONNX_RUNTIME)
    support.validate(
        precision=precision, quantization=quantization, shape_mode=SHAPE_MODE_DYNAMIC
    )
    for provider in providers:
        support.validate(provider=provider)
    if type(opset) is not int or opset < ZERO_SHOT_ONNX_OPSET:
        raise ExportQualificationError(ExportReason.UNSUPPORTED_OPERATOR)
    if not 0.0 < float(threshold) < 1.0:
        raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
    bounds = default_bounds(
        spec,
        max_labels=max_labels,
        max_sequence_length=max_sequence_length,
        max_words=max_words,
        max_span_width=max_span_width,
        max_label_chars=max_label_chars,
    )
    revision = source_identity(
        source.directory,
        config=source.raw_config,
        expected_revision=expected_source_revision,
    )

    try:
        import torch
    except ImportError as exc:
        raise_missing_backend("onnx", feature="GLiNER and GLiClass export", cause=exc)
    from openmed.torch.zero_shot_export import (
        build_zero_shot_module,
        load_reference_weights,
        read_mlx_weights,
        zero_shot_reference_runner,
    )

    module = build_zero_shot_module(torch, source.config, spec)
    load_reference_weights(torch, module, read_mlx_weights(source.weights_path))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Evidence from an earlier export into this directory never carries over.
    (output / EXPORT_QUALIFICATION_FILENAME).unlink(missing_ok=True)
    model_path = output / ZERO_SHOT_MODEL_FILENAME
    shutil.copyfile(source.directory / TOKENIZER_FILENAME, output / TOKENIZER_FILENAME)
    (output / CONFIG_FILENAME).write_text(
        json.dumps(portable_config(source.raw_config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _trace_to_onnx(torch, module, source, bounds, model_path, opset=opset)
    evidence = _structural_evidence(model_path, profile, opset=opset)

    model_files = [model_path]
    sidecar = model_path.with_name(f"{model_path.name}.data")
    if sidecar.is_file():
        model_files.append(sidecar)
    support_files = [output / TOKENIZER_FILENAME, output / CONFIG_FILENAME]
    manifest_path = write_zero_shot_manifest(
        output,
        source=source,
        bounds=bounds,
        revision=revision,
        threshold=threshold,
        opset=opset,
        files=[*model_files, *support_files],
        qualification={
            "status": "not_run",
            "passed": False,
            "reasons": [ExportReason.MISSING_EVIDENCE.value],
            "qualified_variants": [],
        },
    )
    if not qualify:
        return ZeroShotExportResult(
            output_dir=output,
            model_path=model_path,
            manifest_path=manifest_path,
            report_path=None,
            report=None,
        )

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise_missing_backend("onnx", feature="ONNX qualification", cause=exc)
    candidate = VariantCandidate(
        name=QUALIFIED_VARIANT_NAME,
        runner=onnx_logits_runner(ort, model_path, profile),
        precision=precision,
        quantization=quantization,
        provider=ONNX_CPU_PROVIDER,
        artifact_paths=tuple(model_files),
        evidence=evidence,
    )
    report = qualify_zero_shot_export(
        profile=profile,
        spec=spec,
        prompt=source.prompt,
        bounds=bounds,
        tokenizer=source.tokenizer,
        reference_runner=zero_shot_reference_runner(torch, module),
        candidates=[candidate],
        source=revision,
        artifact_root=output,
        artifact_paths=support_files,
        fixtures_path=qualification_fixtures_path,
        label_prompts=label_prompts,
        threshold=threshold,
        exclude_artifact_names=(ZERO_SHOT_MANIFEST_FILENAME,),
        direct_identifier_recall_floor=direct_identifier_recall_floor,
    )
    report_path = write_qualification_report(
        report, output / EXPORT_QUALIFICATION_FILENAME
    )
    summary = summarize_report(report)
    summary["status"] = "passed" if report.passed else "failed"
    summary["report"] = EXPORT_QUALIFICATION_FILENAME
    _attach_qualification(manifest_path, summary)
    if not report.passed:
        logger.warning(
            "Zero-shot ONNX qualification failed: %s", ", ".join(report.reasons)
        )
        report.raise_for_failure()
    return ZeroShotExportResult(
        output_dir=output,
        model_path=model_path,
        manifest_path=manifest_path,
        report_path=report_path,
        report=report,
    )


def onnx_logits_runner(ort_module: Any, path: Path, profile: ZeroShotProfile) -> Any:
    """Return a ``CPUExecutionProvider`` logits runner for an exported graph.

    Raises:
        ExportQualificationError: ``unsupported_provider`` when the CPU
            provider is unavailable, or ``io_contract_mismatch`` when the
            graph's tensor names differ from the profile.
    """

    import numpy as np

    session = ort_module.InferenceSession(str(path), providers=[ONNX_CPU_PROVIDER])
    if ONNX_CPU_PROVIDER not in session.get_providers():
        raise ExportQualificationError(ExportReason.UNSUPPORTED_PROVIDER)
    check_session_contract(session, profile)

    def run(feed: Mapping[str, Any]) -> Any:
        prepared = {name: np.asarray(feed[name]) for name in profile.input_names}
        return np.asarray(session.run(["logits"], prepared)[0], dtype=np.float32)

    return run


def check_session_contract(session: Any, profile: ZeroShotProfile) -> None:
    """Reject a session whose tensor names or ranks differ from the profile.

    Raises:
        ExportQualificationError: ``io_contract_mismatch``.
    """

    contract = profile.tensor_contract()
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    if list(inputs) != list(profile.input_names) or list(outputs) != list(
        profile.output_names
    ):
        raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
    for spec in (*contract["inputs"], *contract["outputs"]):
        item = inputs.get(spec["name"]) or outputs.get(spec["name"])
        shape = getattr(item, "shape", None)
        if shape is not None and len(shape) != len(spec["shape"]):
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)


def write_zero_shot_manifest(
    output_dir: Path,
    *,
    source: ZeroShotSource,
    bounds: ZeroShotBounds,
    revision: Mapping[str, Any],
    threshold: float,
    opset: int,
    files: Sequence[Path],
    qualification: Mapping[str, Any],
) -> Path:
    """Write ``openmed-zero-shot.json`` and return its path.

    The manifest records the profile, source revision, adapter version,
    prompt and label schema, tokenizer identity, tensor contract, bounds,
    decoder and threshold, offset contract, precision, runtime
    compatibility, file digests, and evidence references. It never contains
    caller labels, text, token ids, logits, or local paths.
    """

    profile, spec = source.profile, source.spec
    graph_width = spec.span_width if profile.task == TASK_ZERO_SHOT_NER else None
    prompt = source.prompt.to_dict()
    if profile.task == TASK_ZERO_SHOT_NER:
        prompt["layout"] = "marker_label_pairs_separator_then_words"
        prompt["word_splitter"] = dict(GLINER_WORD_SPLITTER)
    else:
        prompt["layout"] = (
            "marker_labels_separator_instruction_then_text"
            if source.prompt.prompt_first
            else "text_then_marker_labels_separator_instruction"
        )
        prompt["few_shot_examples"] = "unsupported"
    manifest = {
        "format": ZERO_SHOT_MANIFEST_FORMAT,
        "schema_version": ZERO_SHOT_MANIFEST_SCHEMA_VERSION,
        "contract_version": ZERO_SHOT_CONTRACT_VERSION,
        "profile": profile.name,
        "family": profile.family,
        "task": profile.task,
        "adapter_version": profile.adapter_version,
        "runtime": ONNX_RUNTIME,
        "source": dict(revision),
        "architecture": spec.to_dict(),
        "prompt": prompt,
        "labels": {
            **label_schema_identity(profile, source.prompt, bounds),
            "max_labels": bounds.max_labels,
            "max_label_chars": bounds.max_label_chars,
            "duplicates": "reject",
            "reserved_markers": "reject",
        },
        "tokenizer": {
            **zero_shot_tokenizer_identity(source.tokenizer),
            "file": TOKENIZER_FILENAME,
            "truncation": "never",
        },
        "tensors": profile.tensor_contract(span_width=graph_width),
        "bounds": bounds.to_dict(),
        "memory": {
            "estimator": MEMORY_ESTIMATOR,
            "activation_bytes_at_bounds": estimate_activation_bytes(
                spec,
                sequence_length=bounds.max_sequence_length,
                num_labels=bounds.max_labels,
                num_words=bounds.max_words or 0,
                span_width=bounds.max_span_width,
            ),
        },
        "decoder": profile.decoder_contract(threshold),
        "offset_contract": dict(ZERO_SHOT_OFFSET_CONTRACTS[profile.task]),
        "precision": "float32",
        "quantization": QUANTIZATION_NONE,
        "opset": opset,
        "runtime_compatibility": [
            {
                "runtime": "onnxruntime",
                "provider": ONNX_CPU_PROVIDER,
                "precision": "float32",
                "variant": QUALIFIED_VARIANT_NAME,
            }
        ],
        "unsupported": dict(_UNSUPPORTED_SURFACES),
        "files": artifact_file_digests(files, root=output_dir),
        "evidence": {"qualification_report": EXPORT_QUALIFICATION_FILENAME},
        "intended_use": {
            "clinical_claim": "unchanged_from_source",
            "autonomous_clinical_decisions": False,
        },
        "qualification": dict(qualification),
    }
    path = output_dir / ZERO_SHOT_MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    return path


def portable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return *config* without fields that can carry local paths.

    Hugging Face configurations record ``_name_or_path`` (often a private
    checkout path) and ``_commit_hash``; any other top-level string that looks
    like an absolute filesystem path is dropped as well.
    """

    portable: dict[str, Any] = {}
    for key, value in config.items():
        if key in {"_name_or_path", "_commit_hash"}:
            continue
        if isinstance(value, str) and _looks_like_path(value):
            continue
        portable[key] = value
    return portable


def _looks_like_path(value: str) -> bool:
    text = value.strip()
    return text.startswith(("/", "~", "\\\\", "file:")) or (
        len(text) > 2 and text[1] == ":" and text[2] in "\\/"
    )


def _attach_qualification(path: Path, qualification: Mapping[str, Any]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["qualification"] = dict(qualification)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")


def _trace_to_onnx(
    torch_module: Any,
    module: Any,
    source: ZeroShotSource,
    bounds: ZeroShotBounds,
    path: Path,
    *,
    opset: int,
) -> None:
    """Trace *module* with a synthetic request and dynamic axes."""

    from openmed.onnx.convert import _check_onnx_model, _consolidate_external_onnx_data
    from openmed.torch.zero_shot_export import example_inputs

    profile, spec = source.profile, source.spec
    # The graph does not depend on the declared bounds (every axis stays
    # dynamic), so the synthetic tracing request only has to fit the model.
    bounds = replace(
        bounds,
        max_sequence_length=spec.max_position_embeddings,
        max_words=None,
    )
    if profile.task == TASK_ZERO_SHOT_NER:
        encoding = encode_gliner_span(
            source.tokenizer,
            _TRACE_TEXT,
            _TRACE_LABELS,
            prompt=source.prompt,
            bounds=bounds,
            graph_span_width=int(spec.span_width or 1),
        )
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "words_mask": {0: "batch", 1: "sequence"},
            "text_lengths": {0: "batch"},
            "span_idx": {0: "batch", 1: "spans"},
            "span_mask": {0: "batch", 1: "spans"},
            "logits": {0: "batch", 1: "words", 3: "labels"},
        }
    else:
        encoding = encode_gliclass(
            source.tokenizer,
            _TRACE_TEXT,
            _TRACE_LABELS,
            prompt=source.prompt,
            bounds=bounds,
        )
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "labels"},
        }
    feed = {name: encoding.feed[name] for name in profile.input_names}
    with torch_module.no_grad():
        torch_module.onnx.export(
            module,
            example_inputs(torch_module, feed),
            str(path),
            input_names=list(profile.input_names),
            output_names=list(profile.output_names),
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    _consolidate_external_onnx_data(path)
    _check_onnx_model(path)


def _structural_evidence(
    path: Path, profile: ZeroShotProfile, *, opset: int
) -> dict[str, Any]:
    """Return value-free structural evidence for an exported graph.

    Raises:
        ExportQualificationError: ``io_contract_mismatch`` when the graph's
            tensor names or ranks differ from the profile.
    """

    import onnx

    from openmed.onnx.convert import _onnx_operator_evidence

    graph = onnx.load(str(path), load_external_data=False).graph
    initializers = {item.name for item in graph.initializer}
    inputs = [item for item in graph.input if item.name not in initializers]
    names = [item.name for item in inputs]
    outputs = [item.name for item in graph.output]
    contract = profile.tensor_contract()
    ranks = {spec["name"]: len(spec["shape"]) for spec in contract["inputs"]}
    ranks.update({spec["name"]: len(spec["shape"]) for spec in contract["outputs"]})
    observed = {
        item.name: len(item.type.tensor_type.shape.dim)
        for item in [*inputs, *graph.output]
    }
    if (
        names != list(profile.input_names)
        or outputs != list(profile.output_names)
        or observed != ranks
    ):
        raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
    evidence = _onnx_operator_evidence(path)
    evidence["structural_validation"] = {
        "onnx_checker": "passed",
        "input_names": names,
        "output_names": outputs,
        "ranks_match_contract": True,
        "opset": opset,
    }
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns a process exit code."""

    parser = argparse.ArgumentParser(
        description=(
            "Export a local GLiNER span or GLiClass MLX artifact to a qualified "
            "ONNX bundle"
        )
    )
    parser.add_argument("--source", required=True, help="Local MLX artifact dir")
    parser.add_argument("--output", required=True, help="Bundle output directory")
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-span-width", type=int, default=None)
    parser.add_argument("--max-label-chars", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--opset", type=int, default=ZERO_SHOT_ONNX_OPSET)
    parser.add_argument(
        "--qualification-fixtures",
        default=None,
        help="Synthetic qualification suite (JSON or JSONL)",
    )
    parser.add_argument(
        "--expected-source-revision",
        default=None,
        help="Fail unless the source resolves to this revision",
    )
    parser.add_argument("--direct-identifier-recall-floor", type=float, default=None)
    parser.add_argument(
        "--no-qualify",
        action="store_true",
        help="Write the graph without qualification; it then claims nothing",
    )
    args = parser.parse_args(argv)
    try:
        result = export_zero_shot_onnx(
            args.source,
            args.output,
            max_labels=args.max_labels,
            max_sequence_length=args.max_sequence_length,
            max_words=args.max_words,
            max_span_width=args.max_span_width,
            max_label_chars=args.max_label_chars,
            opset=args.opset,
            threshold=args.threshold,
            qualify=not args.no_qualify,
            qualification_fixtures_path=args.qualification_fixtures,
            expected_source_revision=args.expected_source_revision,
            direct_identifier_recall_floor=args.direct_identifier_recall_floor,
        )
    except ExportQualificationError as exc:
        print(f"export refused: {exc.reason_code}")
        return 2
    status = "qualified" if result.passed else "not qualified"
    print(f"wrote {ZERO_SHOT_MANIFEST_FILENAME} ({status})")
    return 0


__all__ = [
    "CONFIG_FILENAME",
    "MEMORY_ESTIMATOR",
    "QUALIFIED_VARIANT_NAME",
    "TOKENIZER_FILENAME",
    "ZERO_SHOT_MODEL_FILENAME",
    "ZeroShotExportResult",
    "ZeroShotSource",
    "check_session_contract",
    "export_zero_shot_onnx",
    "load_zero_shot_source",
    "load_zero_shot_tokenizer",
    "main",
    "onnx_logits_runner",
    "portable_config",
    "write_zero_shot_manifest",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
