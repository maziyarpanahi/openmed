"""Convert Hugging Face token-classification models to OpenMed MLX artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

from openmed.core.hf_publish import publish_artifact
from openmed.mlx.artifact import find_tokenizer_files, write_manifest
from openmed.mlx.models import (
    build_model,
    normalize_model_config,
    normalize_model_type,
    resolve_model_type,
)

logger = logging.getLogger(__name__)

_RECALL_DELTA_REPORT_FILENAME = "recall_delta.json"
_RECALL_DELTA_REPORT_VERSION = 1
_DEFAULT_QUANTIZE_GROUP_SIZE = 64

QuantEvalRunner = Callable[[Any, str, str], Iterable[Any]]

# ---- Weight key remapping ---------------------------------------------------

_BERT_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    (".attention.self.query.", ".attention.query_proj."),
    (".attention.self.key.", ".attention.key_proj."),
    (".attention.self.value.", ".attention.value_proj."),
    (".attention.output.dense.", ".attention.out_proj."),
    (".attention.output.LayerNorm.", ".ln1."),
    (".intermediate.dense.", ".linear1."),
    (".output.dense.", ".linear2."),
    (".output.LayerNorm.", ".ln2."),
    ("bert.encoder.layer.", "encoder.layers."),
    ("bert.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("bert.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("bert.embeddings.token_type_embeddings.", "embeddings.token_type_embeddings."),
    ("bert.embeddings.LayerNorm.", "embeddings.norm."),
    ("classifier.", "classifier."),
    ("bert.pooler.", "_pooler."),
]

_DEBERTA_V2_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    (".attention.output.dense.", ".attention.out_proj."),
    (".attention.output.LayerNorm.", ".ln1."),
    (".intermediate.dense.", ".linear1."),
    (".output.dense.", ".linear2."),
    (".output.LayerNorm.", ".ln2."),
]

_ROBERTA_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    ("roberta.encoder.layer.", "encoder.layers."),
    ("xlm_roberta.encoder.layer.", "encoder.layers."),
    ("roberta.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("roberta.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("roberta.embeddings.token_type_embeddings.", "embeddings.token_type_embeddings."),
    ("roberta.embeddings.LayerNorm.", "embeddings.norm."),
    ("xlm_roberta.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("xlm_roberta.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    (
        "xlm_roberta.embeddings.token_type_embeddings.",
        "embeddings.token_type_embeddings.",
    ),
    ("xlm_roberta.embeddings.LayerNorm.", "embeddings.norm."),
    ("roberta.pooler.", "_pooler."),
    ("xlm_roberta.pooler.", "_pooler."),
]

_DISTILBERT_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    (".attention.q_lin.", ".attention.query_proj."),
    (".attention.k_lin.", ".attention.key_proj."),
    (".attention.v_lin.", ".attention.value_proj."),
    (".attention.out_lin.", ".attention.out_proj."),
    (".sa_layer_norm.", ".ln1."),
    (".output_layer_norm.", ".ln2."),
    (".ffn.lin1.", ".linear1."),
    (".ffn.lin2.", ".linear2."),
    ("distilbert.transformer.layer.", "encoder.layers."),
    ("distilbert.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("distilbert.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("distilbert.embeddings.LayerNorm.", "embeddings.norm."),
]

_ELECTRA_KEY_REPLACEMENTS: list[Tuple[str, str]] = [
    (".attention.self.query.", ".attention.query_proj."),
    (".attention.self.key.", ".attention.key_proj."),
    (".attention.self.value.", ".attention.value_proj."),
    (".attention.output.dense.", ".attention.out_proj."),
    (".attention.output.LayerNorm.", ".ln1."),
    (".intermediate.dense.", ".linear1."),
    (".output.dense.", ".linear2."),
    (".output.LayerNorm.", ".ln2."),
    ("electra.encoder.layer.", "encoder.layers."),
    ("electra.embeddings.word_embeddings.", "embeddings.word_embeddings."),
    ("electra.embeddings.position_embeddings.", "embeddings.position_embeddings."),
    ("electra.embeddings.token_type_embeddings.", "embeddings.token_type_embeddings."),
    ("electra.embeddings.LayerNorm.", "embeddings.norm."),
    ("classifier.", "classifier."),
]

_OPF_MODEL_TYPES = {
    "openai-privacy-filter",
    "privacy-filter",
    "privacy-filter-nemotron",
    "nemotron-privacy-filter",
    "privacy-filter-multilingual",
    "multilingual-privacy-filter",
}


def _convert_opf_weights(state_dict: dict) -> dict:
    """Remap and reshape HuggingFace OPF weights to the MLX namespace.

    The HF ``openai_privacy_filter`` model differs from the MLX layout in:

    1. Top-level prefix: ``score.*`` → ``unembedding.*``
    2. Layer container: ``model.layers.N.*`` → ``block.N.*``
    3. QKV fusion: separate q/k/v_proj → single fused ``attn.qkv`` linear
    4. Sub-key names: ``input_layernorm`` → ``attn.norm``, ``router`` → ``gate``
    5. RMSNorm param: ``weight`` → ``scale``
    6. Expert weight layout: HF stores as ``[E, in, out]``, same as MLX (no transpose needed)
    """
    import re

    import numpy as np

    out: dict[str, np.ndarray] = {}
    # layer_idx → proj ("q"|"k"|"v") → param ("weight"|"bias") → array
    qkv: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for hf_key, arr in state_dict.items():
        # classifier head
        if hf_key == "score.weight":
            out["unembedding.weight"] = arr
            continue
        if hf_key == "score.bias":
            out["unembedding.bias"] = arr
            continue

        # top-level
        if hf_key == "model.embed_tokens.weight":
            out["embedding.weight"] = arr
            continue
        if hf_key == "model.norm.weight":
            out["norm.scale"] = arr
            continue

        m = re.match(r"^model\.layers\.(\d+)\.(.*)", hf_key)
        if m is None:
            continue
        n, rest = m.group(1), m.group(2)

        # RMSNorms
        if rest == "input_layernorm.weight":
            out[f"block.{n}.attn.norm.scale"] = arr
            continue
        if rest == "post_attention_layernorm.weight":
            out[f"block.{n}.mlp.norm.scale"] = arr
            continue

        # attention sinks
        if rest == "self_attn.sinks":
            out[f"block.{n}.attn.sinks"] = arr
            continue

        # output projection
        if rest == "self_attn.o_proj.weight":
            out[f"block.{n}.attn.out.weight"] = arr
            continue
        if rest == "self_attn.o_proj.bias":
            out[f"block.{n}.attn.out.bias"] = arr
            continue

        # Q / K / V — collect for fusion
        m2 = re.match(r"self_attn\.(q|k|v)_proj\.(weight|bias)", rest)
        if m2:
            proj, param = m2.group(1), m2.group(2)
            qkv.setdefault(n, {}).setdefault(proj, {})[param] = arr
            continue

        # MLP router → gate
        if rest == "mlp.router.weight":
            out[f"block.{n}.mlp.gate.weight"] = arr
            continue
        if rest == "mlp.router.bias":
            out[f"block.{n}.mlp.gate.bias"] = arr
            continue

        # expert weights: HF stores as [E, in_features, out_features] — same as MLX, no transpose needed
        if rest == "mlp.experts.gate_up_proj":
            out[f"block.{n}.mlp.swiglu.weight"] = arr
            continue
        if rest == "mlp.experts.gate_up_proj_bias":
            out[f"block.{n}.mlp.swiglu.bias"] = arr
            continue
        if rest == "mlp.experts.down_proj":
            out[f"block.{n}.mlp.out.weight"] = arr
            continue
        if rest == "mlp.experts.down_proj_bias":
            out[f"block.{n}.mlp.out.bias"] = arr
            continue

    # fuse Q / K / V into a single QKV linear per layer
    for n, projs in qkv.items():
        for param in ("weight", "bias"):
            parts = [
                projs[p][param] for p in ("q", "k", "v") if param in projs.get(p, {})
            ]
            if parts:
                out[f"block.{n}.attn.qkv.{param}"] = np.concatenate(parts, axis=0)

    return out


def _expected_opf_weight_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return the MLX weight shapes expected by the OPF runtime."""
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_hidden_layers"])
    num_labels = int(config["num_labels"])
    vocab_size = int(config["vocab_size"])
    intermediate_size = int(config["intermediate_size"])
    num_experts = int(config["num_experts"])
    head_dim = int(config["head_dim"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    qkv_dim = head_dim * (num_attention_heads + 2 * num_key_value_heads)
    attn_out_dim = head_dim * num_attention_heads

    shapes: dict[str, tuple[int, ...]] = {
        "embedding.weight": (vocab_size, hidden_size),
        "norm.scale": (hidden_size,),
        "unembedding.weight": (num_labels, hidden_size),
    }
    if bool(config.get("classifier_bias", config.get("unembedding_bias", False))):
        shapes["unembedding.bias"] = (num_labels,)

    for layer_idx in range(num_layers):
        prefix = f"block.{layer_idx}"
        shapes.update(
            {
                f"{prefix}.attn.norm.scale": (hidden_size,),
                f"{prefix}.attn.qkv.weight": (qkv_dim, hidden_size),
                f"{prefix}.attn.qkv.bias": (qkv_dim,),
                f"{prefix}.attn.out.weight": (hidden_size, attn_out_dim),
                f"{prefix}.attn.out.bias": (hidden_size,),
                f"{prefix}.attn.sinks": (num_attention_heads,),
                f"{prefix}.mlp.norm.scale": (hidden_size,),
                f"{prefix}.mlp.gate.weight": (num_experts, hidden_size),
                f"{prefix}.mlp.gate.bias": (num_experts,),
                f"{prefix}.mlp.swiglu.weight": (
                    num_experts,
                    hidden_size,
                    intermediate_size * 2,
                ),
                f"{prefix}.mlp.swiglu.bias": (num_experts, intermediate_size * 2),
                f"{prefix}.mlp.out.weight": (
                    num_experts,
                    intermediate_size,
                    hidden_size,
                ),
                f"{prefix}.mlp.out.bias": (num_experts, hidden_size),
            }
        )
    return shapes


def _validate_opf_weights(weights: dict[str, Any], config: dict[str, Any]) -> None:
    """Fail early when OPF conversion misses keys or produces wrong shapes."""
    expected_shapes = _expected_opf_weight_shapes(config)
    expected_keys = set(expected_shapes)
    actual_keys = set(weights)

    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {missing[:8]}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected[:8]}")
        raise ValueError("Invalid OPF MLX weight mapping (" + "; ".join(details) + ")")

    bad_shapes: list[str] = []
    for key, expected_shape in expected_shapes.items():
        actual_shape = getattr(weights[key], "shape", None)
        if actual_shape is not None and tuple(actual_shape) != expected_shape:
            bad_shapes.append(
                f"{key}: expected {expected_shape}, got {tuple(actual_shape)}"
            )

    if bad_shapes:
        raise ValueError("Invalid OPF MLX weight shapes: " + "; ".join(bad_shapes[:8]))


def _infer_source_model_type(key: str, model_type: str | None) -> str:
    normalized = normalize_model_type(model_type)
    if normalized is not None:
        return normalized

    if key.startswith("deberta."):
        return "deberta-v2"
    if key.startswith("distilbert."):
        return "distilbert"
    if key.startswith("roberta.") or key.startswith("xlm_roberta."):
        return "roberta"
    if key.startswith("electra."):
        return "electra"
    return "bert"


def remap_key(key: str, model_type: str | None = None) -> str:
    """Remap a HuggingFace state-dict key to the MLX model namespace."""
    source_model_type = _infer_source_model_type(key, model_type)
    resolved_model_type = resolve_model_type(source_model_type)

    if resolved_model_type == "deberta-v2":
        replacements = _DEBERTA_V2_KEY_REPLACEMENTS
    elif source_model_type == "distilbert":
        replacements = _DISTILBERT_KEY_REPLACEMENTS
    elif source_model_type in {"roberta", "xlm-roberta", "xlm_roberta"}:
        replacements = _ROBERTA_KEY_REPLACEMENTS + _BERT_KEY_REPLACEMENTS
    elif source_model_type == "electra":
        replacements = _ELECTRA_KEY_REPLACEMENTS
    else:
        replacements = _BERT_KEY_REPLACEMENTS

    for hf_pattern, mlx_pattern in replacements:
        key = key.replace(hf_pattern, mlx_pattern)
    return key


def _to_numpy(tensor: Any) -> Any:
    if hasattr(tensor, "detach"):
        t = tensor.detach().cpu()
        # PyTorch CPU cannot expose bfloat16 tensors through numpy().
        # Cast to float32 explicitly; ``to(float)`` would promote to float64.
        if hasattr(t, "dtype") and str(t.dtype) == "torch.bfloat16":
            t = t.float()
        return t.numpy()
    return tensor


def convert_weights(
    model_id: str,
    cache_dir: str | None = None,
) -> Tuple[Dict[str, Any], dict[str, Any]]:
    """Load HF token-classification weights and config, then remap for MLX."""
    try:
        from transformers import AutoConfig, AutoModelForTokenClassification
    except ImportError:
        raise ImportError(
            "transformers is required for model conversion. "
            "Install with: pip install transformers"
        )

    logger.info("Loading Hugging Face token-classification model %s ...", model_id)
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )

    source_model_type = normalize_model_type(config.model_type)
    state_dict = model.state_dict()
    config_dict = normalize_model_config(config.to_dict())
    config_dict["_mlx_model_type"] = resolve_model_type(config_dict)
    config_dict["_mlx_task"] = "token-classification"
    if config_dict.get("num_labels") is None:
        config_dict["num_labels"] = config.num_labels

    if source_model_type in _OPF_MODEL_TYPES:
        # OPF requires QKV fusion and weight remapping — use dedicated converter
        if "score.bias" in state_dict:
            config_dict["classifier_bias"] = True
        numpy_state = {k: _to_numpy(v) for k, v in state_dict.items()}
        mlx_weights = _convert_opf_weights(numpy_state)
        _validate_opf_weights(mlx_weights, config_dict)
    else:
        mlx_weights = {}
        skipped = []
        for hf_key, tensor in state_dict.items():
            mlx_key = remap_key(hf_key, source_model_type)
            if mlx_key.startswith("_"):
                skipped.append(hf_key)
                continue
            mlx_weights[mlx_key] = _to_numpy(tensor)
        if skipped:
            logger.info("Skipped %d keys (pooler, etc.): %s", len(skipped), skipped[:5])

    return mlx_weights, config_dict


def save_mlx_model(
    weights: Dict[str, Any],
    config: dict[str, Any],
    output_dir: str | Path,
    quantize_bits: int | None = None,
    source_model_id: str | None = None,
    cache_dir: str | None = None,
    quantize_group_size: int | None = _DEFAULT_QUANTIZE_GROUP_SIZE,
) -> Path:
    """Save converted weights and config to *output_dir*."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
    except ImportError:
        raise ImportError("MLX is required. Install with: pip install openmed[mlx]")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_to_save = dict(config)

    mlx_weights = {k: mx.array(v) for k, v in weights.items()}

    if quantize_bits is not None:
        logger.info("Quantizing to %d bits ...", quantize_bits)
        model = build_model(config)
        model.load_weights(list(mlx_weights.items()))
        quantize_kwargs = {"bits": quantize_bits}
        if quantize_group_size is not None:
            quantize_kwargs["group_size"] = quantize_group_size
        nn.quantize(model, **quantize_kwargs)
        mlx_weights = dict(tree_flatten(model.parameters()))
        config_to_save["_mlx_quantization"] = {
            "bits": quantize_bits,
            "format": _publish_format(quantize_bits),
            "group_size": quantize_group_size,
        }
    else:
        config_to_save.pop("_mlx_quantization", None)

    def _cleanup_other_weight_files(keep_path: Path) -> None:
        for candidate in (
            output_dir / "weights.safetensors",
            output_dir / "weights.npz",
        ):
            if candidate != keep_path and candidate.exists():
                candidate.unlink()

    weights_format = "npz"
    weights_path = output_dir / "weights.npz"
    metadata = {
        "format": "mlx",
        "openmed_task": config_to_save.get("_mlx_task", "token-classification"),
    }
    try:
        weights_path = output_dir / "weights.safetensors"
        mx.save_safetensors(weights_path, mlx_weights, metadata=metadata)
        weights_format = "safetensors"
    except Exception as exc:
        logger.warning(
            "Could not save MLX weights as safetensors; falling back to npz: %s",
            exc,
        )
        weights_path = output_dir / "weights.npz"
        mx.savez(str(weights_path), **mlx_weights)
        weights_format = "npz"

    _cleanup_other_weight_files(weights_path)
    config_to_save["_mlx_weights_format"] = weights_format

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    if "id2label" in config_to_save:
        with open(output_dir / "id2label.json", "w") as f:
            json.dump(config_to_save["id2label"], f, indent=2)

    _finalize_artifact(
        output_dir,
        source_model_id=source_model_id,
        config=config_to_save,
        cache_dir=cache_dir,
    )

    logger.info("Saved MLX model to %s", output_dir)
    return output_dir


def save_numpy_model(
    weights: Dict[str, Any],
    config: dict[str, Any],
    output_dir: str | Path,
    source_model_id: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Save converted weights without MLX, preferring ``.safetensors``."""
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_to_save = dict(config)

    def _cleanup_other_weight_files(keep_path: Path) -> None:
        for candidate in (
            output_dir / "weights.safetensors",
            output_dir / "weights.npz",
        ):
            if candidate != keep_path and candidate.exists():
                candidate.unlink()

    weights_format = "npz"
    weights_path = output_dir / "weights.npz"
    metadata = {
        "format": "mlx",
        "openmed_task": config_to_save.get("_mlx_task", "token-classification"),
    }
    try:
        from safetensors.numpy import save_file

        weights_path = output_dir / "weights.safetensors"
        safe_weights = {k: np.ascontiguousarray(v) for k, v in weights.items()}
        save_file(safe_weights, str(weights_path), metadata=metadata)
        weights_format = "safetensors"
    except Exception as exc:
        logger.warning(
            "Could not save NumPy weights as safetensors; falling back to npz: %s",
            exc,
        )
        weights_path = output_dir / "weights.npz"
        np.savez(str(weights_path), **weights)
        weights_format = "npz"

    _cleanup_other_weight_files(weights_path)
    config_to_save["_mlx_weights_format"] = weights_format

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    if "id2label" in config_to_save:
        with open(output_dir / "id2label.json", "w") as f:
            json.dump(config_to_save["id2label"], f, indent=2)

    _finalize_artifact(
        output_dir,
        source_model_id=source_model_id,
        config=config_to_save,
        cache_dir=cache_dir,
    )

    logger.info(
        "Saved MLX-compatible model to %s using %s weights",
        output_dir,
        weights_format,
    )
    return output_dir


def write_quant_recall_delta_report(
    *,
    source_model_id: str,
    artifact_dir: str | Path,
    eval_suite_path: str | Path,
    format_name: str = "mlx-4bit",
    quantize_bits: int = 4,
    quantize_group_size: int | None = _DEFAULT_QUANTIZE_GROUP_SIZE,
    output_path: str | Path | None = None,
    cache_dir: str | None = None,
    parent_runner: QuantEvalRunner | None = None,
    candidate_runner: QuantEvalRunner | None = None,
) -> dict[str, Any]:
    """Run FP-parent versus quantized artifact recall and write evidence JSON."""
    from openmed.eval.harness import load_fixtures, run_benchmark
    from openmed.eval.quant_delta import (
        evaluate_quant_recall_delta,
        limit_for_format,
    )

    artifact_dir = Path(artifact_dir)
    eval_suite_path = Path(eval_suite_path)
    report_path = (
        Path(output_path)
        if output_path is not None
        else artifact_dir / _RECALL_DELTA_REPORT_FILENAME
    )
    generated_at = _utc_now()
    fixtures = load_fixtures(eval_suite_path)

    parent_runner = parent_runner or _hf_token_classification_runner(
        source_model_id,
        cache_dir=cache_dir,
    )
    candidate_runner = candidate_runner or _mlx_artifact_runner(artifact_dir)

    suite_name = eval_suite_path.stem or "eval-suite"
    parent_report = run_benchmark(
        fixtures,
        suite=suite_name,
        model_name=source_model_id,
        device="fp-parent",
        runner=parent_runner,
        generated_at=generated_at,
        metadata={"format": "mlx-fp", "source_model_id": source_model_id},
    )
    candidate_report = run_benchmark(
        fixtures,
        suite=suite_name,
        model_name=str(artifact_dir),
        device=format_name,
        runner=candidate_runner,
        generated_at=generated_at,
        metadata={
            "format": format_name,
            "source_model_id": source_model_id,
            "quantization": {
                "bits": quantize_bits,
                "group_size": quantize_group_size,
            },
        },
    )

    parent_recall = _per_label_recall(parent_report.metrics)
    candidate_recall = _per_label_recall(candidate_report.metrics)
    delta = evaluate_quant_recall_delta(
        format_name=format_name,
        candidate_recall=candidate_recall,
        parent_recall=parent_recall,
    )
    delta_payload = delta.to_dict()
    delta_payload["blocking_format"] = delta.blocking_format
    limit = limit_for_format(format_name)
    report_relpath = _artifact_relative_path(report_path, artifact_dir)

    payload: dict[str, Any] = {
        "schema_version": _RECALL_DELTA_REPORT_VERSION,
        "generated_at": generated_at,
        "source_model_id": source_model_id,
        "artifact_path": str(artifact_dir),
        "format": format_name,
        "quantization": {
            "bits": quantize_bits,
            "group_size": quantize_group_size,
        },
        "eval_suite_path": str(eval_suite_path),
        "fixture_count": len(fixtures),
        "metric": "character_recall",
        "limit": limit,
        "certified": bool(delta.passed),
        "quant_recall_delta": delta.max_delta,
        "per_label": _per_label_recall_comparison(
            parent_recall,
            candidate_recall,
            delta.per_label_delta,
            quantized_label=f"int{quantize_bits}_recall",
        ),
        "fp_parent_per_label_recall": parent_recall,
        "candidate_per_label_recall": candidate_recall,
        "delta": delta_payload,
        "report_path": report_relpath,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _apply_quant_recall_certification(
        artifact_dir,
        payload,
        report_relpath=report_relpath,
    )
    return payload


def _hf_token_classification_runner(
    model_id: str,
    *,
    cache_dir: str | None = None,
) -> QuantEvalRunner:
    pipeline_instance: Any | None = None

    def run_fixture(fixture: Any, model_name: str, device: str) -> Iterable[Any]:
        del model_name, device
        nonlocal pipeline_instance
        if pipeline_instance is None:
            try:
                from transformers import (
                    AutoModelForTokenClassification,
                    AutoTokenizer,
                    pipeline,
                )
            except ImportError as exc:
                raise ImportError(
                    "transformers is required to certify MLX quantization. "
                    "Install with: pip install transformers"
                ) from exc

            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
            model = AutoModelForTokenClassification.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
            pipeline_instance = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
        return pipeline_instance(fixture.text)

    return run_fixture


def _mlx_artifact_runner(artifact_dir: str | Path) -> QuantEvalRunner:
    pipeline_instance: Any | None = None
    artifact_dir = Path(artifact_dir)

    def run_fixture(fixture: Any, model_name: str, device: str) -> Iterable[Any]:
        del model_name, device
        nonlocal pipeline_instance
        if pipeline_instance is None:
            from openmed.mlx.inference import create_mlx_pipeline

            pipeline_instance = create_mlx_pipeline(
                str(artifact_dir),
                aggregation_strategy="simple",
            )
        return pipeline_instance(fixture.text)

    return run_fixture


def _per_label_recall(metrics: Mapping[str, Any]) -> dict[str, float]:
    recall_slices = metrics.get("recall_slices")
    if isinstance(recall_slices, Mapping):
        by_label = recall_slices.get("by_label")
        if isinstance(by_label, Mapping):
            return _float_map(by_label)

    per_label = metrics.get("per_label_recall")
    if isinstance(per_label, Mapping):
        return _float_map(per_label)
    return {}


def _per_label_recall_comparison(
    parent_recall: Mapping[str, float],
    candidate_recall: Mapping[str, float],
    per_label_delta: Mapping[str, float],
    *,
    quantized_label: str,
) -> dict[str, dict[str, float | None]]:
    labels = sorted(set(parent_recall) | set(candidate_recall) | set(per_label_delta))
    return {
        label: {
            "fp_recall": parent_recall.get(label),
            quantized_label: candidate_recall.get(label),
            "delta": per_label_delta.get(label),
        }
        for label in labels
    }


def _apply_quant_recall_certification(
    artifact_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    report_relpath: str,
) -> None:
    artifact_dir = Path(artifact_dir)
    config_path = artifact_dir / "config.json"
    if not config_path.exists():
        return

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    quantization = dict(config.get("_mlx_quantization") or {})
    quantization.update(
        {
            "bits": payload["quantization"]["bits"],
            "certification_limit": payload["limit"],
            "certified": payload["certified"],
            "format": payload["format"],
            "group_size": payload["quantization"].get("group_size"),
            "quant_recall_delta": payload["quant_recall_delta"],
            "recall_delta_path": report_relpath,
        }
    )
    config["_mlx_quantization"] = quantization
    config["quant_recall_delta"] = payload["quant_recall_delta"]
    config["certified"] = payload["certified"]

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    _update_manifest_quant_certification(
        artifact_dir,
        quantization=quantization,
        payload=payload,
        report_relpath=report_relpath,
    )


def _update_manifest_quant_certification(
    artifact_dir: Path,
    *,
    quantization: Mapping[str, Any],
    payload: Mapping[str, Any],
    report_relpath: str,
) -> None:
    from openmed.mlx.artifact import MANIFEST_FILENAME

    manifest_path = artifact_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    format_name = str(payload["format"])
    manifest["formats"] = _dedupe_keep_order(
        [format_name, *[str(item) for item in manifest.get("formats", [])]]
    )
    manifest["quantization"] = dict(quantization)
    manifest["quant_recall_delta"] = payload["quant_recall_delta"]
    manifest["certified"] = payload["certified"]
    manifest["recall_delta_path"] = report_relpath
    manifest["certification"] = {
        "gate": "G4",
        "limit": payload["limit"],
        "metric": payload["metric"],
        "report_path": report_relpath,
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def _float_map(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        result[str(key)] = parsed
    return result


def _artifact_relative_path(path: Path, artifact_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(artifact_dir.resolve()))
    except (OSError, ValueError):
        return path.name


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _finalize_artifact(
    output_dir: str | Path,
    *,
    source_model_id: str | None,
    config: dict[str, Any],
    cache_dir: str | None,
) -> None:
    if source_model_id is None:
        return

    output_dir = Path(output_dir)
    tokenizer_files: list[str] = []

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source_model_id, cache_dir=cache_dir)
        tokenizer.save_pretrained(output_dir)
        tokenizer_files = find_tokenizer_files(output_dir)
    except Exception as exc:
        logger.warning(
            "Could not save tokenizer assets for %s into %s: %s",
            source_model_id,
            output_dir,
            exc,
        )

    write_manifest(
        output_dir,
        source_model_id=source_model_id,
        config=config,
        tokenizer_files=tokenizer_files,
    )


def convert(
    model_id: str,
    output_dir: str | Path,
    quantize_bits: int | None = None,
    cache_dir: str | None = None,
    publish_to_hub: bool = False,
    publish_repo_id: str | None = None,
    publish_org: str = "OpenMed",
    publish_version: int = 1,
    publish_manifest_path: str | Path | None = None,
    publish_token_env: str = "HF_WRITE_TOKEN",
    publish_private: bool = False,
    publish_overwrite_existing: bool = False,
    quantize_group_size: int | None = _DEFAULT_QUANTIZE_GROUP_SIZE,
    eval_suite_path: str | Path | None = None,
    recall_delta_report_path: str | Path | None = None,
) -> Path:
    """End-to-end: download a model, remap it, and save an OpenMed MLX artifact."""
    weights, config = convert_weights(model_id, cache_dir=cache_dir)
    quantized = False

    try:
        import mlx.core  # noqa: F401

        output_path = save_mlx_model(
            weights=weights,
            config=config,
            output_dir=output_dir,
            quantize_bits=quantize_bits,
            source_model_id=model_id,
            cache_dir=cache_dir,
            quantize_group_size=quantize_group_size,
        )
        quantized = quantize_bits is not None
    except ImportError:
        if quantize_bits is not None:
            logger.warning(
                "MLX not available — skipping quantization. "
                "Install mlx for quantization support."
            )
        output_path = save_numpy_model(
            weights,
            config,
            output_dir,
            source_model_id=model_id,
            cache_dir=cache_dir,
        )

    if eval_suite_path is not None:
        if quantize_bits is None:
            raise ValueError("eval_suite_path requires a quantized MLX artifact")
        if not quantized:
            raise ImportError(
                "MLX is required to certify a quantized artifact with eval fixtures"
            )
        write_quant_recall_delta_report(
            source_model_id=model_id,
            artifact_dir=output_path,
            eval_suite_path=eval_suite_path,
            format_name=_publish_format(quantize_bits),
            quantize_bits=quantize_bits,
            quantize_group_size=quantize_group_size,
            output_path=recall_delta_report_path,
            cache_dir=cache_dir,
        )

    if publish_to_hub:
        result = publish_artifact(
            artifact_dir=output_path,
            source_model_id=model_id,
            format_name=_publish_format(quantize_bits),
            repo_id=publish_repo_id,
            org=publish_org,
            version=publish_version,
            token_env=publish_token_env,
            manifest_path=publish_manifest_path,
            private=publish_private,
            skip_existing=not publish_overwrite_existing,
        )
        if result.skipped:
            logger.info("Skipped existing Hub repo %s", result.repo_id)
        else:
            logger.info("Published MLX artifact to %s", result.repo_id)

    return output_path


def _publish_format(quantize_bits: int | None) -> str:
    if quantize_bits is None:
        return "mlx-fp"
    return f"mlx-{quantize_bits}bit"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face token-classification model to OpenMed MLX format",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Source model ID or local directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for MLX model files",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantize weights to N bits (4 or 8)",
    )
    parser.add_argument(
        "--quantize-group-size",
        type=int,
        default=_DEFAULT_QUANTIZE_GROUP_SIZE,
        help="Group size to use for MLX weight quantization",
    )
    parser.add_argument(
        "--eval-suite",
        default=None,
        help=(
            "Benchmark fixture JSON used to certify quantized recall against "
            "the full-precision parent"
        ),
    )
    parser.add_argument(
        "--recall-delta-report",
        default=None,
        help="Optional output path for the quantization recall-delta JSON report",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face cache directory",
    )
    parser.add_argument(
        "--publish-to-hub",
        action="store_true",
        help="Publish the converted artifact after a successful conversion",
    )
    parser.add_argument(
        "--publish-repo-id",
        default=None,
        help="Explicit target repo id for publishing",
    )
    parser.add_argument(
        "--publish-org",
        default="OpenMed",
        help="Target organization for derived publish repo ids",
    )
    parser.add_argument(
        "--publish-version",
        type=int,
        default=1,
        help="Version suffix used when the source repo is not already versioned",
    )
    parser.add_argument(
        "--publish-manifest",
        default=None,
        help="JSONL manifest path to append or update after publishing",
    )
    parser.add_argument(
        "--publish-token-env",
        default="HF_WRITE_TOKEN",
        help="Environment variable containing the Hub write token",
    )
    parser.add_argument(
        "--publish-private",
        action="store_true",
        help="Create the target repo as private when it does not exist",
    )
    parser.add_argument(
        "--publish-overwrite-existing",
        action="store_true",
        help="Upload into an existing target repo instead of skipping it",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convert(
        model_id=args.model,
        output_dir=args.output,
        quantize_bits=args.quantize,
        cache_dir=args.cache_dir,
        eval_suite_path=args.eval_suite,
        recall_delta_report_path=args.recall_delta_report,
        quantize_group_size=args.quantize_group_size,
        publish_to_hub=args.publish_to_hub,
        publish_repo_id=args.publish_repo_id,
        publish_org=args.publish_org,
        publish_version=args.publish_version,
        publish_manifest_path=args.publish_manifest,
        publish_token_env=args.publish_token_env,
        publish_private=args.publish_private,
        publish_overwrite_existing=args.publish_overwrite_existing,
    )


if __name__ == "__main__":
    main()
