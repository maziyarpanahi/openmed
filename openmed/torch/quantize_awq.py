"""AWQ 4-bit quantization recipe for Hugging Face checkpoints."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.processing.tokenizer_cache import get_tokenizer_with_loader

from .calibration import (
    QUANTIZATION_CALIBRATION_SOURCE,
    calibration_texts_sha256,
    load_quantization_calibration_texts,
)

QUANT_CONFIG_FILENAME = "quant_config.json"
AWQ_FORMAT = "openmed-awq"
AWQ_FORMAT_VERSION = 1
DEFAULT_AWQ_VERSION = "GEMM"


@dataclass(frozen=True)
class AwqQuantizationResult:
    """Paths and metadata produced by :func:`quantize_awq`."""

    output_dir: Path
    quant_config_path: Path
    source_model_id: str
    source_revision: str
    calibration_sample_count: int


def quantize_awq(
    model_name: str,
    calib_texts: Iterable[str] | None,
    out_dir: str | Path,
    w_bit: int = 4,
    group_size: int = 128,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    device_map: str | Mapping[str, Any] | None = "auto",
    zero_point: bool = True,
    version: str = DEFAULT_AWQ_VERSION,
    max_calib_seq_len: int = 512,
) -> AwqQuantizationResult:
    """Quantize a Hugging Face checkpoint with AutoAWQ and record the recipe.

    Args:
        model_name: Hugging Face model ID or local model directory.
        calib_texts: Calibration samples. When ``None``, OpenMed's committed
            synthetic clinical-note calibration set is used.
        out_dir: Directory that receives the AWQ checkpoint and metadata.
        w_bit: Weight bit width. OpenMed's recipe defaults to 4-bit AWQ.
        group_size: AWQ quantization group size.
        revision: Optional source model revision. When omitted, the Hugging Face
            config commit hash is used when available.
        trust_remote_code: Passed through to Hugging Face and AutoAWQ loaders.
        local_files_only: Restrict Hugging Face resolution to local cache/files.
        device_map: Device map passed to AutoAWQ.
        zero_point: Whether to use zero-point quantization.
        version: AutoAWQ kernel layout, such as ``"GEMM"``.
        max_calib_seq_len: Maximum sequence length for calibration samples.

    Raises:
        ImportError: If the optional AWQ dependencies are not installed.
        ValueError: If quantization parameters or calibration texts are invalid.
    """

    _validate_quant_params(w_bit=w_bit, group_size=group_size)
    calibration_texts = _normalize_calibration_texts(calib_texts)

    AutoAWQForCausalLM = _require_autoawq()
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
    tokenizer = get_tokenizer_with_loader(
        model_name,
        AutoTokenizer.from_pretrained,
        **hf_kwargs,
    )

    download_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if revision is not None:
        download_kwargs["revision"] = revision

    model = AutoAWQForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        safetensors=True,
        device_map=device_map,
        download_kwargs=download_kwargs,
    )

    autoawq_quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": w_bit,
        "version": version,
    }
    model.quantize(
        tokenizer,
        quant_config=autoawq_quant_config,
        calib_data=calibration_texts,
        max_calib_samples=len(calibration_texts),
        max_calib_seq_len=max_calib_seq_len,
    )
    model.save_quantized(str(output_dir))
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
        w_bit=w_bit,
        group_size=group_size,
        calibration_texts=calibration_texts,
        autoawq_quant_config=autoawq_quant_config,
        max_calib_seq_len=max_calib_seq_len,
        config=config,
    )
    return AwqQuantizationResult(
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
    w_bit: int,
    group_size: int,
    calibration_texts: Iterable[str],
    autoawq_quant_config: Mapping[str, Any],
    max_calib_seq_len: int = 512,
    config: Any | None = None,
) -> Path:
    """Write OpenMed AWQ recipe metadata into ``quant_config.json``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_samples = _normalize_calibration_texts(calibration_texts)
    config_dict = _config_to_dict(config)

    quant_config = {
        "format": AWQ_FORMAT,
        "format_version": AWQ_FORMAT_VERSION,
        "quantization_method": "awq",
        "source_model_id": source_model_id,
        "source_revision": source_revision,
        "task": str(config_dict.get("task") or "token-classification"),
        "family": str(config_dict.get("model_type") or "unknown"),
        "w_bit": w_bit,
        "group_size": group_size,
        "q_group_size": group_size,
        "zero_point": bool(autoawq_quant_config.get("zero_point", True)),
        "version": str(autoawq_quant_config.get("version", DEFAULT_AWQ_VERSION)),
        "calibration_sample_count": len(calibration_samples),
        "calibration_source": QUANTIZATION_CALIBRATION_SOURCE,
        "calibration_sha256": calibration_texts_sha256(calibration_samples),
        "max_calib_seq_len": max_calib_seq_len,
        "autoawq_quant_config": dict(autoawq_quant_config),
    }

    label_map = config_dict.get("id2label")
    if isinstance(label_map, Mapping):
        quant_config["label_count"] = len(label_map)

    quant_config_path = output_dir / QUANT_CONFIG_FILENAME
    with quant_config_path.open("w", encoding="utf-8") as handle:
        json.dump(quant_config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return quant_config_path


def _require_autoawq() -> Any:
    try:
        from awq import AutoAWQForCausalLM
    except ImportError as exc:
        raise ImportError(
            "AWQ quantization requires the optional `autoawq` dependency. "
            "Install it with: pip install openmed[awq]"
        ) from exc
    return AutoAWQForCausalLM


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "AWQ quantization requires Hugging Face Transformers. "
            "Install with: pip install openmed[awq]"
        ) from exc
    return AutoConfig, AutoTokenizer


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


def _validate_quant_params(*, w_bit: int, group_size: int) -> None:
    if w_bit != 4:
        raise ValueError("OpenMed AWQ export currently supports w_bit=4")
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
    "AWQ_FORMAT",
    "AWQ_FORMAT_VERSION",
    "AwqQuantizationResult",
    "QUANT_CONFIG_FILENAME",
    "quantize_awq",
    "write_quant_config",
]
