"""Build Transformers.js token-classification bundles from ONNX exports."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

TRANSFORMERSJS_FORMAT = "transformersjs"
DEFAULT_BUNDLE_DIRNAME = "transformersjs"
DEFAULT_ONNX_FILENAME = "model.onnx"
DEFAULT_QUANTIZED_ONNX_FILENAME = "model_quantized.onnx"
OPENMED_ONNX_MANIFEST_FILENAME = "openmed-onnx.json"
CONTRACT_FILENAME = "transformersjs-contract.json"
QUANTIZE_CONFIG_FILENAME = "quantize_config.json"
MINIMUM_TOKEN_CLASSIFICATION_OPSET = 18

EXPECTED_INPUT_NAMES = ("input_ids", "attention_mask")
OPTIONAL_INPUT_NAMES = ("token_type_ids",)
EXPECTED_OUTPUT_NAMES = ("logits",)
REQUIRED_BUNDLE_FILES = (
    "onnx/model.onnx",
    "onnx/model_quantized.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    QUANTIZE_CONFIG_FILENAME,
)
TOKENIZER_ASSET_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
)


@dataclass(frozen=True)
class TransformersJsBundleResult:
    """Paths and contract data emitted for a Transformers.js bundle."""

    output_dir: Path
    model_path: Path
    quantized_model_path: Path
    config_path: Path
    tokenizer_path: Path
    tokenizer_config_path: Path
    quantize_config_path: Path
    contract_path: Path
    manifest_path: Path | None
    contract: Mapping[str, Any]

    @property
    def files(self) -> tuple[str, ...]:
        """Return the required bundle files relative to ``output_dir``."""

        return REQUIRED_BUNDLE_FILES


def export_transformersjs_bundle(
    onnx_export_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    tokenizer_source: str | Path | None = None,
    config_source: str | Path | None = None,
    onnx_filename: str = DEFAULT_ONNX_FILENAME,
    quantize: bool = True,
    update_manifest: bool = True,
    minimum_opset: int = MINIMUM_TOKEN_CLASSIFICATION_OPSET,
) -> TransformersJsBundleResult:
    """Emit a Transformers.js-compatible token-classification directory.

    Args:
        onnx_export_dir: Directory produced by the OpenMed ONNX converter.
        output_dir: Destination bundle directory. Defaults to a
            ``transformersjs`` subdirectory under ``onnx_export_dir``.
        tokenizer_source: Saved tokenizer directory or Hugging Face model id.
            Defaults to ``onnx_export_dir``.
        config_source: Path to a source ``config.json`` file. Defaults to the
            config written by the ONNX converter in ``onnx_export_dir``.
        onnx_filename: Source ONNX filename inside ``onnx_export_dir``.
        quantize: If true, create ``onnx/model_quantized.onnx`` with
            onnxruntime dynamic INT8 weight quantization.
        update_manifest: If true and ``openmed-onnx.json`` exists, add the
            ``transformersjs`` format and artifact entry.
    """

    source_dir = Path(onnx_export_dir)
    source_model_path = source_dir / onnx_filename
    if not source_model_path.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {source_model_path}")

    bundle_dir = (
        Path(output_dir)
        if output_dir is not None
        else (source_dir / DEFAULT_BUNDLE_DIRNAME)
    )
    onnx_dir = bundle_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model_path = onnx_dir / DEFAULT_ONNX_FILENAME
    _copy_file(source_model_path, model_path)

    quantized_model_path = onnx_dir / DEFAULT_QUANTIZED_ONNX_FILENAME
    if quantize:
        _quantize_dynamic_int8(model_path, quantized_model_path)
    else:
        _copy_file(model_path, quantized_model_path)

    config = _load_and_normalize_config(
        Path(config_source) if config_source is not None else source_dir / "config.json"
    )
    _copy_or_save_tokenizer_assets(
        tokenizer_source if tokenizer_source is not None else source_dir,
        bundle_dir,
        config=config,
    )
    config_path = bundle_dir / "config.json"
    _write_json(config_path, config)

    quantize_config_path = bundle_dir / QUANTIZE_CONFIG_FILENAME
    _write_quantize_config(quantize_config_path, quantized=quantize)

    contract = validate_transformersjs_contract(
        model_path,
        minimum_opset=minimum_opset,
    )
    contract_path = bundle_dir / CONTRACT_FILENAME
    _write_json(contract_path, contract)

    validate_transformersjs_bundle(bundle_dir, minimum_opset=minimum_opset)
    manifest_path = (
        _update_openmed_onnx_manifest(source_dir, bundle_dir, quantized=quantize)
        if update_manifest
        else None
    )

    return TransformersJsBundleResult(
        output_dir=bundle_dir,
        model_path=model_path,
        quantized_model_path=quantized_model_path,
        config_path=config_path,
        tokenizer_path=bundle_dir / "tokenizer.json",
        tokenizer_config_path=bundle_dir / "tokenizer_config.json",
        quantize_config_path=quantize_config_path,
        contract_path=contract_path,
        manifest_path=manifest_path,
        contract=contract,
    )


def validate_transformersjs_bundle(
    bundle_dir: str | Path,
    *,
    minimum_opset: int = MINIMUM_TOKEN_CLASSIFICATION_OPSET,
) -> Mapping[str, Any]:
    """Validate bundle files, label metadata, and ONNX pipeline contract."""

    bundle_path = Path(bundle_dir)
    missing = find_missing_bundle_files(bundle_path)
    if missing:
        raise ValueError(
            "Transformers.js bundle is missing required files: " + ", ".join(missing)
        )

    config = _read_json(bundle_path / "config.json")
    _normalize_id2label(config.get("id2label"))
    return validate_transformersjs_contract(
        bundle_path / "onnx" / DEFAULT_ONNX_FILENAME,
        minimum_opset=minimum_opset,
    )


def find_missing_bundle_files(bundle_dir: str | Path) -> list[str]:
    """Return required Transformers.js bundle files missing under ``bundle_dir``."""

    root = Path(bundle_dir)
    return [name for name in REQUIRED_BUNDLE_FILES if not (root / name).exists()]


def validate_transformersjs_contract(
    model_path: str | Path,
    *,
    minimum_opset: int = MINIMUM_TOKEN_CLASSIFICATION_OPSET,
) -> dict[str, Any]:
    """Validate ONNX tensor names and dynamic axes for token classification."""

    model = _load_onnx_model(Path(model_path))
    model_opset = _default_domain_opset(model)
    if model_opset is not None and model_opset < minimum_opset:
        raise ValueError(
            "Transformers.js token-classification ONNX model requires opset "
            f">= {minimum_opset}; got {model_opset}"
        )
    graph = model.graph
    initializer_names = {getattr(item, "name", "") for item in graph.initializer}
    graph_inputs = [
        item
        for item in graph.input
        if getattr(item, "name", "") not in initializer_names
    ]
    input_by_name = {item.name: item for item in graph_inputs}
    output_by_name = {item.name: item for item in graph.output}

    missing_inputs = [
        name for name in EXPECTED_INPUT_NAMES if name not in input_by_name
    ]
    if missing_inputs:
        raise ValueError(
            "Transformers.js token-classification ONNX model is missing inputs: "
            + ", ".join(missing_inputs)
        )

    allowed_inputs = set(EXPECTED_INPUT_NAMES) | set(OPTIONAL_INPUT_NAMES)
    unexpected_inputs = sorted(set(input_by_name) - allowed_inputs)
    if unexpected_inputs:
        raise ValueError(
            "Transformers.js token-classification ONNX model has unexpected "
            f"inputs: {', '.join(unexpected_inputs)}"
        )

    if list(output_by_name) != list(EXPECTED_OUTPUT_NAMES):
        raise ValueError(
            "Transformers.js token-classification ONNX model must expose only "
            f"{', '.join(EXPECTED_OUTPUT_NAMES)} output"
        )

    for name, value_info in input_by_name.items():
        _validate_axes(value_info, ("batch", "sequence"))
    _validate_axes(output_by_name["logits"], ("batch", "sequence", "labels"))

    ordered_input_names = [
        name
        for name in (*EXPECTED_INPUT_NAMES, *OPTIONAL_INPUT_NAMES)
        if name in input_by_name
    ]
    return {
        "task": "token-classification",
        "format": TRANSFORMERSJS_FORMAT,
        "minimum_opset": minimum_opset,
        "model_opset": model_opset,
        "inputs": [
            _value_info_contract(input_by_name[name], ("batch", "sequence"))
            for name in ordered_input_names
        ],
        "outputs": [
            _value_info_contract(
                output_by_name[name],
                ("batch", "sequence", "labels"),
            )
            for name in EXPECTED_OUTPUT_NAMES
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Transformers.js export CLI."""

    parser = argparse.ArgumentParser(
        description="Build a Transformers.js bundle from an OpenMed ONNX export",
    )
    parser.add_argument(
        "--onnx-export-dir",
        required=True,
        help="Directory containing model.onnx, config.json, and tokenizer assets",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination bundle directory, defaults to <onnx-export-dir>/transformersjs",
    )
    parser.add_argument(
        "--tokenizer-source",
        default=None,
        help="Saved tokenizer directory or model id. Defaults to the ONNX export dir.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to source config.json. Defaults to <onnx-export-dir>/config.json.",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Copy model.onnx to model_quantized.onnx instead of INT8 quantizing it.",
    )
    parser.add_argument(
        "--no-manifest-update",
        action="store_true",
        help="Do not update openmed-onnx.json in the source export directory.",
    )
    parser.add_argument(
        "--minimum-opset",
        type=int,
        default=MINIMUM_TOKEN_CLASSIFICATION_OPSET,
        help="Minimum ONNX opset required for the bundle contract.",
    )
    args = parser.parse_args(argv)

    result = export_transformersjs_bundle(
        args.onnx_export_dir,
        args.output,
        tokenizer_source=args.tokenizer_source,
        config_source=args.config,
        quantize=not args.no_quantize,
        update_manifest=not args.no_manifest_update,
        minimum_opset=args.minimum_opset,
    )
    print(result.output_dir)


def _copy_or_save_tokenizer_assets(
    source: str | Path,
    output_dir: Path,
    *,
    config: Mapping[str, Any],
) -> None:
    source_path = Path(source)
    copied = False
    if source_path.exists():
        for filename in TOKENIZER_ASSET_FILENAMES:
            candidate = source_path / filename
            if candidate.exists():
                _copy_file(candidate, output_dir / filename)
                copied = True

    if not (output_dir / "tokenizer.json").exists():
        if copied:
            raise ValueError(
                f"tokenizer.json is required for Transformers.js: {source_path}"
            )
        _save_tokenizer_from_pretrained(str(source), output_dir)

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        tokenizer_config = {
            "model_max_length": config.get("max_sequence_length")
            or config.get("max_position_embeddings")
            or 512,
        }
        tokenizer_class = config.get("tokenizer_class")
        if tokenizer_class:
            tokenizer_config["tokenizer_class"] = tokenizer_class
        _write_json(tokenizer_config_path, tokenizer_config)


def _save_tokenizer_from_pretrained(model_id: str, output_dir: Path) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required when tokenizer_source is a model id. "
            "Install with: pip install openmed[hf]"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.save_pretrained(output_dir)
    if not (output_dir / "tokenizer.json").exists():
        raise ValueError(
            "Transformers.js requires a fast tokenizer that saves tokenizer.json"
        )


def _load_and_normalize_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config.json does not exist: {path}")
    config = _read_json(path)
    id2label = _normalize_id2label(config.get("id2label"))
    config["id2label"] = id2label
    config["label2id"] = {label: int(index) for index, label in id2label.items()}
    config["task"] = str(
        config.get("_mlx_task") or config.get("task") or "token-classification"
    )
    return config


def _normalize_id2label(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("config.json must contain a non-empty id2label mapping")
    return {str(key): str(label) for key, label in value.items()}


def _quantize_dynamic_int8(model_path: Path, quantized_model_path: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required to create model_quantized.onnx. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    quantize_dynamic(
        str(model_path),
        str(quantized_model_path),
        weight_type=QuantType.QInt8,
    )
    if not quantized_model_path.exists():
        raise RuntimeError(f"quantized model was not written: {quantized_model_path}")


def _load_onnx_model(path: Path) -> Any:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx is required to validate Transformers.js bundle contracts. "
            "Install with: pip install openmed[onnx]"
        ) from exc
    return onnx.load(str(path))


def _default_domain_opset(model: Any) -> int | None:
    for item in getattr(model, "opset_import", []) or []:
        domain = getattr(item, "domain", "")
        if domain in {"", "ai.onnx"}:
            version = getattr(item, "version", None)
            return int(version) if version is not None else None
    return None


def _validate_axes(value_info: Any, axes: Sequence[str]) -> None:
    dims = _tensor_dims(value_info)
    if len(dims) != len(axes):
        raise ValueError(
            f"{value_info.name} must have rank {len(axes)} for axes {', '.join(axes)}"
        )

    for index, axis in enumerate(axes):
        dim = _dim_contract(dims[index])
        if axis in {"batch", "sequence"} and dim["kind"] != "dynamic":
            raise ValueError(f"{value_info.name} axis {axis} must be dynamic")
        if axis == "labels" and dim["kind"] != "static":
            raise ValueError(f"{value_info.name} axis labels must be static")


def _value_info_contract(value_info: Any, axes: Sequence[str]) -> dict[str, Any]:
    return {
        "name": value_info.name,
        "axes": list(axes),
        "shape": [_dim_contract(dim) for dim in _tensor_dims(value_info)],
    }


def _tensor_dims(value_info: Any) -> Sequence[Any]:
    try:
        return value_info.type.tensor_type.shape.dim
    except AttributeError as exc:
        raise ValueError(f"{value_info.name} must have tensor shape metadata") from exc


def _dim_contract(dim: Any) -> dict[str, Any]:
    dim_param = getattr(dim, "dim_param", "")
    dim_value = getattr(dim, "dim_value", 0)
    if dim_param:
        return {"kind": "dynamic", "name": str(dim_param)}
    if dim_value:
        return {"kind": "static", "value": int(dim_value)}
    return {"kind": "dynamic", "name": None}


def _write_quantize_config(path: Path, *, quantized: bool) -> None:
    _write_json(
        path,
        {
            "format": TRANSFORMERSJS_FORMAT,
            "source": "onnx/model.onnx",
            "target": "onnx/model_quantized.onnx",
            "algorithm": "dynamic",
            "weight_type": "qint8" if quantized else "copied",
        },
    )


def _update_openmed_onnx_manifest(
    source_dir: Path,
    bundle_dir: Path,
    *,
    quantized: bool,
) -> Path | None:
    manifest_path = source_dir / OPENMED_ONNX_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None

    manifest = _read_json(manifest_path)
    formats = list(manifest.get("formats") or [])
    if TRANSFORMERSJS_FORMAT not in formats:
        formats.append(TRANSFORMERSJS_FORMAT)
    manifest["formats"] = formats

    artifacts = list(manifest.get("artifacts") or [])
    if not any(item.get("format") == TRANSFORMERSJS_FORMAT for item in artifacts):
        artifacts.append(
            {
                "format": TRANSFORMERSJS_FORMAT,
                "path": _relative_path(bundle_dir, source_dir),
                "precision": "int8" if quantized else "float32",
            }
        )
    manifest["artifacts"] = artifacts
    _write_json(manifest_path, manifest)
    return manifest_path


def _relative_path(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, target)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
