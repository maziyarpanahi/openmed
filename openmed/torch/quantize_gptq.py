"""GPTQ 4-bit quantization recipe for Hugging Face checkpoints."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibration import (
    QUANTIZATION_CALIBRATION_SOURCE,
    calibration_texts_sha256,
    load_quantization_calibration_texts,
)

QUANT_CONFIG_FILENAME = "quant_config.json"
GPTQ_FORMAT = "openmed-gptq"
GPTQ_FORMAT_VERSION = 1


@dataclass(frozen=True)
class GptqQuantizationResult:
    """Paths and metadata produced by :func:`quantize_gptq`."""

    output_dir: Path
    quant_config_path: Path
    source_model_id: str
    source_revision: str
    calibration_sample_count: int


def quantize_gptq(
    model_name: str,
    calib_texts: Iterable[str] | None,
    out_dir: str | Path,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    device_map: str | Mapping[str, Any] | None = "auto",
    max_calib_seq_len: int = 512,
    calib_batch_size: int = 1,
    use_safetensors: bool = True,
) -> GptqQuantizationResult:
    """Quantize a Hugging Face checkpoint with AutoGPTQ and record the recipe.

    Args:
        model_name: Hugging Face model ID or local model directory.
        calib_texts: Calibration samples. When ``None``, OpenMed's committed
            synthetic clinical-note calibration set is used.
        out_dir: Directory that receives the GPTQ checkpoint and metadata.
        bits: Weight bit width. OpenMed's recipe defaults to 4-bit GPTQ.
        group_size: GPTQ quantization group size.
        desc_act: Whether AutoGPTQ should use activation-order quantization.
        revision: Optional source model revision. When omitted, the Hugging Face
            config commit hash is used when available.
        trust_remote_code: Passed through to Hugging Face and AutoGPTQ loaders.
        local_files_only: Restrict Hugging Face resolution to local cache/files.
        device_map: Device map passed to AutoGPTQ.
        max_calib_seq_len: Maximum sequence length for calibration samples.
        calib_batch_size: Batch size passed to AutoGPTQ calibration.
        use_safetensors: Save GPTQ weights with safetensors when supported.

    Raises:
        ImportError: If the optional GPTQ dependencies are not installed.
        ValueError: If quantization parameters or calibration texts are invalid.
    """

    _validate_quant_params(bits=bits, group_size=group_size)
    if calib_batch_size <= 0:
        raise ValueError("calib_batch_size must be a positive integer")
    if max_calib_seq_len <= 0:
        raise ValueError("max_calib_seq_len must be a positive integer")

    calibration_texts = _normalize_calibration_texts(calib_texts)

    AutoGPTQForCausalLM, BaseQuantizeConfig = _require_autogptq()
    AutoConfig, AutoTokenizer = _require_transformers()

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        hf_kwargs["revision"] = revision

    config = AutoConfig.from_pretrained(model_name, **hf_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)

    autogptq_quant_config = {
        "bits": bits,
        "group_size": group_size,
        "desc_act": desc_act,
    }
    quantize_config = BaseQuantizeConfig(**autogptq_quant_config)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        model_kwargs["revision"] = revision
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config,
        **model_kwargs,
    )
    calibration_examples = _build_calibration_examples(
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        max_calib_seq_len=max_calib_seq_len,
    )
    model.quantize(calibration_examples, batch_size=calib_batch_size)
    model.save_quantized(str(output_dir), use_safetensors=use_safetensors)
    tokenizer.save_pretrained(output_dir)

    source_revision = _resolve_source_revision(
        config=config,
        model_name=model_name,
        explicit_revision=revision,
    )
    quant_config_path = write_quant_config(
        output_dir,
        source_model_id=model_name,
        source_revision=source_revision,
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        calibration_texts=calibration_texts,
        autogptq_quant_config=autogptq_quant_config,
        max_calib_seq_len=max_calib_seq_len,
        calib_batch_size=calib_batch_size,
        config=config,
    )
    return GptqQuantizationResult(
        output_dir=output_dir,
        quant_config_path=quant_config_path,
        source_model_id=model_name,
        source_revision=source_revision,
        calibration_sample_count=len(calibration_texts),
    )


def write_quant_config(
    output_dir: str | Path,
    *,
    source_model_id: str,
    source_revision: str,
    bits: int,
    group_size: int,
    desc_act: bool,
    calibration_texts: Iterable[str],
    autogptq_quant_config: Mapping[str, Any],
    max_calib_seq_len: int = 512,
    calib_batch_size: int = 1,
    config: Any | None = None,
) -> Path:
    """Write OpenMed GPTQ recipe metadata into ``quant_config.json``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_samples = _normalize_calibration_texts(calibration_texts)
    config_dict = _config_to_dict(config)

    quant_config = {
        "format": GPTQ_FORMAT,
        "format_version": GPTQ_FORMAT_VERSION,
        "quantization_method": "gptq",
        "source_model_id": source_model_id,
        "source_revision": source_revision,
        "task": str(config_dict.get("task") or "token-classification"),
        "family": str(config_dict.get("model_type") or "unknown"),
        "bits": bits,
        "group_size": group_size,
        "desc_act": desc_act,
        "calibration_sample_count": len(calibration_samples),
        "calibration_source": QUANTIZATION_CALIBRATION_SOURCE,
        "calibration_sha256": calibration_texts_sha256(calibration_samples),
        "max_calib_seq_len": max_calib_seq_len,
        "calib_batch_size": calib_batch_size,
        "autogptq_quant_config": dict(autogptq_quant_config),
    }

    label_map = config_dict.get("id2label")
    if isinstance(label_map, Mapping):
        quant_config["label_count"] = len(label_map)

    quant_config_path = output_dir / QUANT_CONFIG_FILENAME
    with quant_config_path.open("w", encoding="utf-8") as handle:
        json.dump(quant_config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return quant_config_path


def _require_autogptq() -> tuple[Any, Any]:
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError as exc:
        raise ImportError(
            "GPTQ quantization requires the optional `auto-gptq` dependency. "
            "Install it with: pip install openmed[gptq]"
        ) from exc
    return AutoGPTQForCausalLM, BaseQuantizeConfig


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "GPTQ quantization requires Hugging Face Transformers. "
            "Install with: pip install openmed[gptq]"
        ) from exc
    return AutoConfig, AutoTokenizer


def _build_calibration_examples(
    *,
    tokenizer: Any,
    calibration_texts: Iterable[str],
    max_calib_seq_len: int,
) -> list[Any]:
    return [
        tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_calib_seq_len,
        )
        for text in calibration_texts
    ]


def _normalize_calibration_texts(calib_texts: Iterable[str] | None) -> list[str]:
    texts = (
        load_quantization_calibration_texts()
        if calib_texts is None
        else list(calib_texts)
    )
    normalized = [text.strip() for text in texts if text and text.strip()]
    if not normalized:
        raise ValueError("calib_texts must contain at least one non-empty sample")
    return normalized


def _validate_quant_params(*, bits: int, group_size: int) -> None:
    if bits != 4:
        raise ValueError("OpenMed GPTQ export currently supports bits=4")
    if group_size <= 0:
        raise ValueError("group_size must be a positive integer")


def _resolve_source_revision(
    *,
    config: Any,
    model_name: str,
    explicit_revision: str | None,
) -> str:
    if explicit_revision:
        return explicit_revision
    commit_hash = getattr(config, "_commit_hash", None)
    if commit_hash:
        return str(commit_hash)
    config_dict = _config_to_dict(config)
    for key in ("_commit_hash", "commit_hash", "revision"):
        value = config_dict.get(key)
        if value:
            return str(value)
    if Path(model_name).exists():
        return "local"
    return "unresolved"


def _config_to_dict(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    return {}


__all__ = [
    "GPTQ_FORMAT",
    "GPTQ_FORMAT_VERSION",
    "GptqQuantizationResult",
    "QUANT_CONFIG_FILENAME",
    "quantize_gptq",
    "write_quant_config",
]
