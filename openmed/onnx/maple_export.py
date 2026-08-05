"""Reproducible Maple decoder export for ONNX Runtime Mobile and WebGPU.

This module deliberately exports the immutable BF16 source checkpoint instead
of pretending that MLX safetensors are ONNX graphs.  It adapts the official
``onnxruntime-genai`` model builder for Maple's alternating RoPE/NoPE attention
and fused, clamped SwiGLU experts.  The resulting single decoder graph accepts
empty KV caches for prefill and populated caches for decode.

The full checkpoint is large (roughly 40 GB before quantization).  Unit tests
therefore exercise source validation, planning, and a tiny QMoE graph; a full
conversion remains an explicit release operation on a high-memory machine.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.machinery
import importlib.metadata
import json
import re
import shutil
import sys
import tempfile
import types
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence

from openmed.onnx.maple_bundle import (
    MAPLE_SOURCE_MODEL,
    MAPLE_SOURCE_REVISION,
    MapleBundleBuild,
    build_maple_onnx_bundle,
)
from openmed.onnx.maple_qmoe_smoke import validate_maple_qmoe_runtime

MAPLE_MLX_MODEL = "deepgrove/maple-preview-2bit-mlx"
MAPLE_MLX_REVISION = "361db5da5e74ff6fcdd852d478e1f266ce11013a"
MAPLE_SOURCE_RECEIPT_FILENAME = "openmed-maple-source.json"
MAPLE_ONNX_RECEIPT_FILENAME = "openmed-maple-onnx-export.json"
MAPLE_ONNX_SCHEMA_VERSION = 1
MAPLE_ONNX_GRAPH_FILENAME = "model.onnx"
MAPLE_ONNX_QUANTIZATION = "qmoe-4bit-blockwise-128"
MAPLE_QMOE_BLOCK_SIZE = 128
MAPLE_ONNX_RUNTIME_MIN_VERSION = "1.25.1"
MAPLE_ONNX_VERSION = "1.21.0"
MAPLE_ONNX_RUNTIME_EXPORT_VERSION = "1.25.1"
MAPLE_ORT_GENAI_VERSION = "0.12.0"
MAPLE_ONNX_IR_VERSION = "0.2.0"
MAPLE_TORCH_VERSION = "2.9.1"
MAPLE_TRANSFORMERS_VERSION = "4.57.1"
MAPLE_NUMPY_VERSION = "2.2.6"
MAPLE_HUGGINGFACE_HUB_VERSION = "0.35.3"
MAPLE_ACCELERATE_VERSION = "1.11.0"
MAPLE_SAFETENSORS_VERSION = "0.6.2"
MAPLE_TOKENIZERS_VERSION = "0.22.1"
MAPLE_TQDM_VERSION = "4.67.1"
MAPLE_ONNX_EXPORT_REQUIREMENTS = (
    f"accelerate=={MAPLE_ACCELERATE_VERSION}",
    f"huggingface-hub=={MAPLE_HUGGINGFACE_HUB_VERSION}",
    f"numpy=={MAPLE_NUMPY_VERSION}",
    f"onnx=={MAPLE_ONNX_VERSION}",
    f"onnx-ir=={MAPLE_ONNX_IR_VERSION}",
    f"onnxruntime=={MAPLE_ONNX_RUNTIME_EXPORT_VERSION}",
    f"onnxruntime-genai=={MAPLE_ORT_GENAI_VERSION}",
    f"safetensors=={MAPLE_SAFETENSORS_VERSION}",
    f"tokenizers=={MAPLE_TOKENIZERS_VERSION}",
    f"torch=={MAPLE_TORCH_VERSION}",
    f"tqdm=={MAPLE_TQDM_VERSION}",
    f"transformers=={MAPLE_TRANSFORMERS_VERSION}",
)

MapleOnnxTarget = Literal["mobile", "webgpu"]

_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "sliding_attention"
    for index in range(24)
)
_SOURCE_ALLOW_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.model",
    "*.py",
    "*.safetensors",
    "*.txt",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_LAYER_ID_PATTERN = re.compile(r"/model/layers\.(\d+)/")


class MapleOnnxExportError(RuntimeError):
    """Raised when Maple source or an ONNX export violates the contract."""


@dataclass(frozen=True)
class MapleOnnxSourceInspection:
    """Validated metadata for a local Maple BF16 source snapshot."""

    source_directory: Path
    source_model: str
    source_revision: str
    revision_verified: bool
    weight_shards: tuple[str, ...]
    total_weight_bytes: int


@dataclass(frozen=True)
class MapleOnnxExportPlan:
    """An immutable, dependency-free plan for one Maple ONNX conversion."""

    source_directory: Path
    output_directory: Path
    target: MapleOnnxTarget
    execution_provider: str
    io_dtype: str
    graph_filename: str
    quantization: str
    source_model: str
    source_revision: str
    revision_verified: bool
    source_weight_bytes: int
    qmoe_block_size: int
    runtime_min_version: str
    builder_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable export plan."""

        result = asdict(self)
        result["source_directory"] = str(self.source_directory)
        result["output_directory"] = str(self.output_directory)
        return result


@dataclass(frozen=True)
class MapleOnnxExportResult:
    """Result and validation receipt for a completed Maple ONNX export."""

    output_directory: Path
    graph_path: Path
    receipt_path: Path
    validation: Mapping[str, Any]


def maple_onnx_export_requirements() -> tuple[str, ...]:
    """Return the exact dependency set used for reproducible conversion."""

    return MAPLE_ONNX_EXPORT_REQUIREMENTS


def download_maple_onnx_source(destination: str | Path) -> Path:
    """Download the immutable BF16 Maple snapshot and bind it to a receipt.

    Args:
        destination: New or empty local directory for the source snapshot.

    Returns:
        Resolved snapshot directory.

    Raises:
        FileExistsError: If the destination already contains files.
        MapleOnnxExportError: If ``huggingface_hub`` is unavailable.
    """

    output = Path(destination).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty source directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MapleOnnxExportError(
            "huggingface_hub is required; install the versions printed by "
            "`python -m openmed.onnx.maple_export requirements`"
        ) from exc

    try:
        resolved = Path(
            snapshot_download(
                repo_id=MAPLE_SOURCE_MODEL,
                revision=MAPLE_SOURCE_REVISION,
                local_dir=str(output),
                allow_patterns=list(_SOURCE_ALLOW_PATTERNS),
            )
        ).resolve()
        _write_json_exclusive(
            resolved / MAPLE_SOURCE_RECEIPT_FILENAME,
            {
                "schema_version": 1,
                "source_model": MAPLE_SOURCE_MODEL,
                "source_revision": MAPLE_SOURCE_REVISION,
            },
        )
        inspect_maple_onnx_source(resolved, require_verified_revision=True)
    except Exception:
        receipt = output / MAPLE_SOURCE_RECEIPT_FILENAME
        receipt.unlink(missing_ok=True)
        raise
    return resolved


def inspect_maple_onnx_source(
    source_directory: str | Path,
    *,
    require_verified_revision: bool = True,
) -> MapleOnnxSourceInspection:
    """Validate a local Maple BF16 snapshot without loading its tensors.

    A source receipt written by :func:`download_maple_onnx_source` is required
    by default.  This prevents a mutable or manually assembled directory from
    being silently described as the pinned upstream revision.
    """

    root = Path(source_directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Maple source directory does not exist: {root}")
    config = _read_json_object(root / "config.json")
    _validate_maple_config(config)

    for filename in ("configuration_maple.py", "modeling_maple.py", "fa3.py"):
        _require_regular_file(root, filename)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        _require_regular_file(root, filename)

    index = _read_json_object(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise MapleOnnxExportError("Maple weight index needs a non-empty weight_map")
    shards = tuple(sorted(set(weight_map.values())))
    if any(not isinstance(item, str) for item in shards):
        raise MapleOnnxExportError("Maple weight index contains invalid shard names")
    total_bytes = 0
    for shard in shards:
        _validate_flat_filename(shard)
        total_bytes += _require_regular_file(root, shard).stat().st_size

    receipt_path = root / MAPLE_SOURCE_RECEIPT_FILENAME
    revision_verified = False
    if receipt_path.is_file():
        receipt = _read_json_object(receipt_path)
        revision_verified = (
            receipt.get("schema_version") == 1
            and receipt.get("source_model") == MAPLE_SOURCE_MODEL
            and receipt.get("source_revision") == MAPLE_SOURCE_REVISION
        )
    if require_verified_revision and not revision_verified:
        raise MapleOnnxExportError(
            f"source revision is unverified; acquire it with "
            f"download_maple_onnx_source() or pass "
            f"require_verified_revision=False explicitly"
        )

    declared_size = index.get("metadata", {}).get("total_size")
    if not isinstance(declared_size, int) or declared_size <= 0:
        raise MapleOnnxExportError("Maple weight index needs a positive total_size")
    # Safetensors indexes declare tensor payload bytes; files also contain
    # headers, so their exact byte sizes are expected to be slightly larger.
    maximum_header_bytes = max(64 * 1024**2, declared_size // 100)
    if not declared_size <= total_bytes <= declared_size + maximum_header_bytes:
        raise MapleOnnxExportError(
            "Maple source shard sizes do not match model.safetensors.index.json"
        )
    return MapleOnnxSourceInspection(
        source_directory=root,
        source_model=MAPLE_SOURCE_MODEL,
        source_revision=MAPLE_SOURCE_REVISION,
        revision_verified=revision_verified,
        weight_shards=shards,
        total_weight_bytes=total_bytes,
    )


def plan_maple_onnx_export(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    target: MapleOnnxTarget = "mobile",
    require_verified_revision: bool = True,
) -> MapleOnnxExportPlan:
    """Create a validated 4-bit Maple export plan without importing ONNX."""

    if target not in {"mobile", "webgpu"}:
        raise ValueError("target must be 'mobile' or 'webgpu'")
    source = inspect_maple_onnx_source(
        source_directory,
        require_verified_revision=require_verified_revision,
    )
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite export directory: {output}")
    return MapleOnnxExportPlan(
        source_directory=source.source_directory,
        output_directory=output,
        target=target,
        execution_provider="cpu" if target == "mobile" else "webgpu",
        io_dtype="float32" if target == "mobile" else "float16",
        graph_filename=MAPLE_ONNX_GRAPH_FILENAME,
        quantization=MAPLE_ONNX_QUANTIZATION,
        source_model=source.source_model,
        source_revision=source.source_revision,
        revision_verified=source.revision_verified,
        source_weight_bytes=source.total_weight_bytes,
        qmoe_block_size=MAPLE_QMOE_BLOCK_SIZE,
        runtime_min_version=MAPLE_ONNX_RUNTIME_MIN_VERSION,
        builder_version=MAPLE_ORT_GENAI_VERSION,
    )


def export_maple_onnx(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    target: MapleOnnxTarget = "mobile",
    require_verified_revision: bool = True,
) -> MapleOnnxExportResult:
    """Export Maple to one cached int4/QMoE ONNX Runtime decoder graph.

    The graph is intentionally shared between prefill and decode.  Callers
    provide zero-length KV caches for the first invocation and feed ``present``
    outputs back as ``past_key_values`` on subsequent invocations.

    This function validates graph structure but does not claim output parity or
    target-device compatibility.  Those release gates are recorded as false in
    the generated receipt until separately exercised.
    """

    plan = plan_maple_onnx_export(
        source_directory,
        output_directory,
        target=target,
        require_verified_revision=require_verified_revision,
    )
    dependencies = _load_export_dependencies()
    qmoe_smoke = validate_maple_qmoe_runtime()
    ir = dependencies["ir"]
    torch = dependencies["torch"]
    auto_config = dependencies["AutoConfig"]
    base_model = dependencies["BaseModel"]

    output = plan.output_directory
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.maple-onnx-", dir=output.parent)
    )
    try:
        _install_flash_attention_import_stubs()
        config = auto_config.from_pretrained(
            str(plan.source_directory),
            trust_remote_code=True,
            local_files_only=True,
        )
        config._name_or_path = str(plan.source_directory)
        extra_options: dict[str, Any] = {
            "filename": plan.graph_filename,
            "hf_remote": True,
            "hf_token": "false",
            "int4_block_size": MAPLE_QMOE_BLOCK_SIZE,
            "qmoe_block_size": MAPLE_QMOE_BLOCK_SIZE,
            "int4_is_symmetric": True,
            "int4_op_types_to_quantize": ("MatMul", "Gather"),
            "int4_nodes_to_exclude": [
                f"/model/layers.{index}/moe/router/MatMul" for index in range(24)
            ],
            "disable_qkv_fusion": True,
        }
        io_dtype = ir.DataType.FLOAT if target == "mobile" else ir.DataType.FLOAT16
        builder_type = _make_maple_builder_type(base_model, ir, torch)
        cache_dir = staging / ".builder-cache"
        cache_dir.mkdir()
        builder = builder_type(
            config,
            io_dtype,
            ir.DataType.INT4,
            plan.execution_provider,
            str(cache_dir),
            extra_options,
        )
        builder.make_model(str(plan.source_directory))
        builder.save_model(str(staging))
        builder.make_genai_config(
            str(plan.source_directory),
            {"local_files_only": True},
            str(staging),
        )
        builder.save_processing(
            str(plan.source_directory),
            {"local_files_only": True},
            str(staging),
        )
        validation = validate_maple_onnx_graph(
            staging / plan.graph_filename,
            target=target,
        )
        artifacts = _record_export_artifacts(staging)
        receipt = {
            "schema_version": MAPLE_ONNX_SCHEMA_VERSION,
            "source_model": plan.source_model,
            "source_revision": plan.source_revision,
            "source_revision_verified": plan.revision_verified,
            "architecture": "MapleForCausalLM",
            "target": target,
            "execution_provider": plan.execution_provider,
            "io_dtype": plan.io_dtype,
            "quantization": plan.quantization,
            "qmoe_block_size": plan.qmoe_block_size,
            "onnxruntime_min_version": plan.runtime_min_version,
            "onnxruntime_genai_version": plan.builder_version,
            "export_toolchain": _installed_export_versions(),
            "graph": {
                "path": plan.graph_filename,
                "prefill_path": plan.graph_filename,
                "decode_path": plan.graph_filename,
            },
            "validation": {
                **validation,
                "synthetic_qmoe_cpu_completed": True,
                "synthetic_qmoe_cpu_runtime": qmoe_smoke.onnxruntime_version,
                "full_checkpoint_conversion_completed": True,
                "runtime_inference_completed": False,
                "source_parity_completed": False,
                "target_device_completed": False,
            },
            "files": artifacts,
        }
        receipt_path = staging / MAPLE_ONNX_RECEIPT_FILENAME
        _write_json_exclusive(receipt_path, receipt)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return MapleOnnxExportResult(
        output_directory=output,
        graph_path=output / plan.graph_filename,
        receipt_path=output / MAPLE_ONNX_RECEIPT_FILENAME,
        validation=validation,
    )


def validate_maple_onnx_graph(
    graph_path: str | Path,
    *,
    target: MapleOnnxTarget,
) -> dict[str, Any]:
    """Validate Maple-specific QMoE, attention, cache, and external-data shape."""

    if target not in {"mobile", "webgpu"}:
        raise ValueError("target must be 'mobile' or 'webgpu'")
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MapleOnnxExportError(
            "onnx>=1.21 is required to validate the graph"
        ) from exc

    path = Path(graph_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Maple ONNX graph does not exist: {path}")
    # Passing the path lets the checker resolve external tensors without
    # materializing the multi-gigabyte sidecar in this process.
    onnx.checker.check_model(str(path), full_check=False)
    model = onnx.load(str(path), load_external_data=False)
    nodes = list(model.graph.node)
    unsupported_normalizations = {
        "SimplifiedLayerNormalization",
        "SkipLayerNormalization",
        "SkipSimplifiedLayerNormalization",
    }
    if any(node.op_type in unsupported_normalizations for node in nodes):
        raise MapleOnnxExportError(
            "Maple graph must lower normalization to portable ONNX operators"
        )
    qmoe_nodes = [node for node in nodes if node.op_type == "QMoE"]
    attention_nodes = [node for node in nodes if node.op_type == "GroupQueryAttention"]
    if len(qmoe_nodes) != 24:
        raise MapleOnnxExportError("Maple graph must contain exactly 24 QMoE nodes")
    if len(attention_nodes) != 24:
        raise MapleOnnxExportError(
            "Maple graph must contain exactly 24 GroupQueryAttention nodes"
        )

    for node in qmoe_nodes:
        if len(node.input) != 7 or node.input[4] != "":
            raise MapleOnnxExportError(
                "Maple QMoE must use fused interleaved gate/up weights without FC3"
            )
        attrs = _onnx_attributes(onnx, node)
        required = {
            "expert_weight_bits": 4,
            "block_size": MAPLE_QMOE_BLOCK_SIZE,
            "k": 8,
            "normalize_routing_weights": 1,
            "swiglu_fusion": 1,
            "activation_type": "swiglu",
            "swiglu_limit": 7.0,
        }
        for key, expected in required.items():
            if attrs.get(key) != expected:
                raise MapleOnnxExportError(
                    f"{node.name or 'QMoE'} has invalid {key}: {attrs.get(key)!r}"
                )

    seen_layers: set[int] = set()
    for node in attention_nodes:
        match = _LAYER_ID_PATTERN.search(node.name)
        if match is None:
            raise MapleOnnxExportError("Maple attention node has no layer id")
        layer_id = int(match.group(1))
        seen_layers.add(layer_id)
        attrs = _onnx_attributes(onnx, node)
        if attrs.get("num_heads") != 16 or attrs.get("kv_num_heads") != 4:
            raise MapleOnnxExportError(
                f"Maple layer {layer_id} has the wrong GQA head contract"
            )
        local = _LAYER_TYPES[layer_id] == "sliding_attention"
        if attrs.get("do_rotary") != int(local):
            raise MapleOnnxExportError(
                f"Maple layer {layer_id} has the wrong RoPE/NoPE setting"
            )
        expected_window = 512 if local else -1
        if attrs.get("local_window_size") != expected_window:
            raise MapleOnnxExportError(
                f"Maple layer {layer_id} has the wrong attention window"
            )
    if seen_layers != set(range(24)):
        raise MapleOnnxExportError("Maple attention layer ids are incomplete")

    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    if "input_ids" not in inputs or "attention_mask" not in inputs:
        raise MapleOnnxExportError("Maple graph is missing token or mask inputs")
    if "position_ids" in inputs:
        raise MapleOnnxExportError("fused Maple GQA must not require position_ids")
    _validate_tensor_contract(
        onnx,
        inputs["input_ids"],
        dtype=onnx.TensorProto.INT64,
        rank=2,
    )
    _validate_tensor_contract(
        onnx,
        inputs["attention_mask"],
        dtype=onnx.TensorProto.INT64,
        rank=2,
    )
    io_dtype = (
        onnx.TensorProto.FLOAT if target == "mobile" else onnx.TensorProto.FLOAT16
    )
    for layer_id in range(24):
        for kind in ("key", "value"):
            past_name = f"past_key_values.{layer_id}.{kind}"
            present_name = f"present.{layer_id}.{kind}"
            if past_name not in inputs:
                raise MapleOnnxExportError("Maple graph has incomplete KV-cache inputs")
            if present_name not in outputs:
                raise MapleOnnxExportError(
                    "Maple graph has incomplete KV-cache outputs"
                )
            _validate_cache_tensor(onnx, inputs[past_name], dtype=io_dtype)
            _validate_cache_tensor(onnx, outputs[present_name], dtype=io_dtype)
    if "logits" not in outputs:
        raise MapleOnnxExportError("Maple graph is missing logits")
    logits_shape = _validate_tensor_contract(
        onnx,
        outputs["logits"],
        dtype=io_dtype,
        rank=3,
    )
    if logits_shape[-1] != 151_936:
        raise MapleOnnxExportError("Maple logits must use the 151936-token vocabulary")

    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    router_nodes = {
        node.name: node for node in nodes if node.name.endswith("/moe/router/MatMul")
    }
    if len(router_nodes) != 24:
        raise MapleOnnxExportError("Maple graph must preserve 24 FP32 router MatMuls")
    for layer_id in range(24):
        name = f"/model/layers.{layer_id}/moe/router/MatMul"
        node = router_nodes.get(name)
        if node is None or len(node.input) != 2:
            raise MapleOnnxExportError(f"Maple layer {layer_id} router is incomplete")
        weight = initializers.get(node.input[1])
        if weight is None or weight.data_type != onnx.TensorProto.FLOAT:
            raise MapleOnnxExportError(
                f"Maple layer {layer_id} router weight must remain FP32"
            )

    external_files = _external_data_files(onnx, model, path.parent)
    if not external_files:
        raise MapleOnnxExportError("Maple graph must store weights as external data")
    return {
        "onnx_checker_completed": True,
        "maple_graph_contract_completed": True,
        "qmoe_nodes": len(qmoe_nodes),
        "group_query_attention_nodes": len(attention_nodes),
        "external_data_files": [item.name for item in external_files],
    }


def build_maple_onnx_export_bundle(
    export_directory: str | Path,
    output_path: str | Path,
) -> MapleBundleBuild:
    """Package a completed exporter directory using its bound receipt.

    The same cached graph is declared for both prefill and decode, while its
    external-data file and tokenizer assets are integrity-bound as payloads.
    """

    root = Path(export_directory).expanduser().resolve()
    receipt = _read_json_object(root / MAPLE_ONNX_RECEIPT_FILENAME)
    if receipt.get("schema_version") != MAPLE_ONNX_SCHEMA_VERSION:
        raise MapleOnnxExportError("unsupported Maple ONNX export receipt")
    if receipt.get("source_model") != MAPLE_SOURCE_MODEL:
        raise MapleOnnxExportError("Maple ONNX receipt has the wrong source model")
    if receipt.get("source_revision") != MAPLE_SOURCE_REVISION:
        raise MapleOnnxExportError("Maple ONNX receipt has the wrong source revision")
    target = receipt.get("target")
    if target not in {"mobile", "webgpu"}:
        raise MapleOnnxExportError("Maple ONNX receipt has an invalid target")
    graph = receipt.get("graph")
    if not isinstance(graph, dict) or graph.get("path") != MAPLE_ONNX_GRAPH_FILENAME:
        raise MapleOnnxExportError("Maple ONNX receipt has an invalid graph contract")
    validate_maple_onnx_graph(root / MAPLE_ONNX_GRAPH_FILENAME, target=target)

    declared = receipt.get("files")
    if not isinstance(declared, list) or not declared:
        raise MapleOnnxExportError("Maple ONNX receipt has no artifact list")
    paths: list[str] = []
    for item in declared:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise MapleOnnxExportError("Maple ONNX receipt has an invalid artifact")
        path = item["path"]
        _validate_flat_filename(path)
        source = _require_regular_file(root, path)
        if source.stat().st_size != item.get("size_bytes"):
            raise MapleOnnxExportError(f"Maple ONNX artifact size changed: {path}")
        expected_digest = item.get("sha256")
        if (
            not isinstance(expected_digest, str)
            or not _SHA256_PATTERN.fullmatch(expected_digest)
            or not hmac.compare_digest(_sha256_file(source), expected_digest)
        ):
            raise MapleOnnxExportError(f"Maple ONNX artifact hash changed: {path}")
        paths.append(path)
    receipt_name = MAPLE_ONNX_RECEIPT_FILENAME
    if receipt_name not in paths:
        paths.append(receipt_name)

    graph_name = MAPLE_ONNX_GRAPH_FILENAME
    tokenizer_name = "tokenizer.json"
    extras = tuple(path for path in paths if path not in {graph_name, tokenizer_name})
    runtime = "onnxruntime-mobile" if target == "mobile" else "onnxruntime-web"
    return build_maple_onnx_bundle(
        root,
        output_path,
        prefill_path=graph_name,
        decode_path=graph_name,
        tokenizer_path=tokenizer_name,
        extra_files=extras,
        runtime=runtime,
        quantization=MAPLE_ONNX_QUANTIZATION,
        source_revision=MAPLE_SOURCE_REVISION,
    )


def _make_maple_builder_type(base_model: type, ir: Any, torch: Any) -> type:
    """Create the narrow Maple adapter around the pinned official builder."""

    class MapleOnnxModel(base_model):
        def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, options):
            config.num_local_experts = config.num_experts
            config.swiglu_limit = 7.0
            super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, options)
            # ORT GenAI does not register Maple as a model family. Its Qwen2
            # decoder pipeline is the compatible generic cached-decoder path;
            # the receipt continues to identify the real Maple architecture.
            self.model_type = "Qwen2ForCausalLM"
            self.layer_types = tuple(config.layer_types)
            if self.layer_types != _LAYER_TYPES:
                raise MapleOnnxExportError("unexpected Maple attention layer pattern")
            self.maple_window_size = int(config.sliding_window)
            self.attention_attrs["q_norm"] = True
            self.attention_attrs["k_norm"] = True
            self.attention_attrs["use_packed_matmul"] = False
            self.moe_attrs.update(
                {
                    "activation_alpha": 1.0,
                    "activation_beta": 0.0,
                    "activation_type": "swiglu",
                    "expert_weight_bits": 4,
                    "normalize_routing_weights": True,
                    "swiglu_fusion": 1,
                    "swiglu_limit": 7.0,
                    "use_sparse_mixer": False,
                }
            )

        def make_layernorm(self, layer_id, layernorm, skip, simple, location):
            # Android and Web builds do not consistently register ORT's
            # standalone/skip normalization contrib operators. The pinned
            # builder already provides an equivalent standard-ONNX lowering.
            self._make_layernorm_op(layer_id, layernorm, skip, simple, location)

        def load_weights(self, input_path):
            # The validated input is a complete local BF16 snapshot. Keep the
            # release conversion offline and avoid a second full-size loading
            # allocation where Transformers supports meta-device loading.
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(
                input_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                local_files_only=True,
                dtype="auto",
                low_cpu_mem_usage=True,
            )

        def make_node(
            self,
            op_type,
            inputs,
            outputs,
            *,
            name,
            domain="",
            **kwargs,
        ):
            # Q/K RMSNorm is emitted directly by the pinned builder rather
            # than through make_layernorm(). Lower it to standard operators as
            # well so the same graph can load in ORT Mobile and ORT Web.
            if op_type == "SimplifiedLayerNormalization":
                if len(inputs) != 2 or len(outputs) != 1:
                    raise MapleOnnxExportError(
                        "unexpected standalone Maple RMSNorm signature"
                    )
                root = self.values[inputs[0]]
                self._make_simplified_layer_norm(
                    name,
                    inputs[0],
                    inputs[1],
                    outputs[0],
                    root.dtype,
                    root.shape,
                )
                return None
            return super().make_node(
                op_type,
                inputs,
                outputs,
                name=name,
                domain=domain,
                **kwargs,
            )

        def make_layer(self, layer_id, layer):
            self.make_layernorm(
                layer_id,
                layer.input_layernorm,
                skip=not self.layernorm_attrs["first_layernorm"],
                simple=True,
                location="input",
            )
            self.make_attention(
                layer_id,
                layer.self_attn,
                root_input=self.layernorm_attrs["output_0"],
            )
            self.make_layernorm(
                layer_id,
                layer.post_attention_layernorm,
                skip=True,
                simple=True,
                location="post_attention",
            )
            self.make_maple_moe(
                layer_id,
                layer.mlp,
                root_input=self.layernorm_attrs["output_0"],
            )
            self.layernorm_attrs["first_layernorm"] = False
            if layer_id == self.num_layers - 1:
                self.layernorm_attrs["last_layernorm"] = True

        def make_attention(self, layer_id, attention, root_input, **kwargs):
            local = self.layer_types[layer_id] == "sliding_attention"
            old_window = self.window_size
            old_rope = self.attention_attrs["use_rope_in_attn"]
            self.window_size = self.maple_window_size if local else -1
            self.attention_attrs["use_rope_in_attn"] = local
            try:
                self.make_attention_input_proj(
                    layer_id, attention, root_input, **kwargs
                )
                self.make_maple_attention_qk(layer_id, attention, **kwargs)
                self.make_attention_output_proj(
                    layer_id, attention, root_input, **kwargs
                )
            finally:
                self.window_size = old_window
                self.attention_attrs["use_rope_in_attn"] = old_rope

        def make_maple_attention_qk(self, layer_id, attention, **kwargs):
            self.make_qk_norm(layer_id, attention)
            cos_cache_name, sin_cache_name = "", ""
            if self.attention_attrs["use_rope_in_attn"]:
                cos_cache_name, sin_cache_name = self.make_rotary_embedding_caches()

            past_k = f"past_key_values.{layer_id}.key"
            past_v = f"past_key_values.{layer_id}.value"
            present_k = f"present.{layer_id}.key"
            present_v = f"present.{layer_id}.value"
            attn_name = (
                f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
            )
            self.make_attention_op(
                attn_name,
                q_path=self.attention_attrs["q_path"],
                k_path=self.attention_attrs["k_path"],
                v_path=self.attention_attrs["v_path"],
                past_k=past_k,
                past_v=past_v,
                present_k=present_k,
                present_v=present_v,
                cos_cache=cos_cache_name,
                sin_cache=sin_cache_name,
                sinks="",
                **kwargs,
            )

        def make_maple_moe(self, layer_id, moe, root_input):
            basename = f"/model/layers.{layer_id}/moe"
            router_probs = self._make_maple_router(layer_id, moe, root_input)
            fc1_weights, fc1_scales = [], []
            fc2_weights, fc2_scales = [], []
            for expert in moe.experts:
                gate_up = torch.stack(
                    (expert.gate_proj.weight, expert.up_proj.weight), dim=1
                ).reshape(-1, expert.gate_proj.weight.shape[-1])
                q_fc1, s_fc1 = self.make_qmoe_weights(gate_up)
                q_fc2, s_fc2 = self.make_qmoe_weights(expert.down_proj.weight)
                fc1_weights.append(q_fc1)
                fc1_scales.append(s_fc1)
                fc2_weights.append(q_fc2)
                fc2_scales.append(s_fc2)

            fc1_weight_name = f"model.layers.{layer_id}.moe.fc1.qweight"
            fc1_scale_name = f"model.layers.{layer_id}.moe.fc1.scales"
            fc2_weight_name = f"model.layers.{layer_id}.moe.fc2.qweight"
            fc2_scale_name = f"model.layers.{layer_id}.moe.fc2.scales"
            self.make_initializer(
                torch.stack(fc1_weights), fc1_weight_name, to=ir.DataType.UINT8
            )
            self.make_initializer(
                torch.stack(fc1_scales), fc1_scale_name, to=self.io_dtype
            )
            self.make_initializer(
                torch.stack(fc2_weights), fc2_weight_name, to=ir.DataType.UINT8
            )
            self.make_initializer(
                torch.stack(fc2_scales), fc2_scale_name, to=self.io_dtype
            )

            output = f"{basename}/output_0"
            self.make_node(
                "QMoE",
                inputs=[
                    root_input,
                    router_probs,
                    fc1_weight_name,
                    fc1_scale_name,
                    "",
                    fc2_weight_name,
                    fc2_scale_name,
                ],
                outputs=[output],
                name=basename,
                domain="com.microsoft",
                activation_alpha=1.0,
                activation_beta=0.0,
                activation_type="swiglu",
                expert_weight_bits=4,
                k=self.moe_attrs["top_k"],
                normalize_routing_weights=1,
                swiglu_fusion=1,
                swiglu_limit=7.0,
                use_sparse_mixer=0,
                block_size=MAPLE_QMOE_BLOCK_SIZE,
            )
            self.make_value(
                output,
                self.io_dtype,
                shape=["batch_size", "sequence_length", self.hidden_size],
            )
            self.layernorm_attrs["skip_input"] = output

        def _make_maple_router(self, layer_id, moe, root_input):
            basename = f"/model/layers.{layer_id}/moe/router"
            router_input = root_input
            if self.io_dtype != ir.DataType.FLOAT:
                cast_name = f"{basename}/CastInput"
                self.make_cast(
                    cast_name,
                    root_input,
                    dtype=ir.DataType.FLOAT,
                    shape=["batch_size", "sequence_length", self.hidden_size],
                )
                router_input = f"{cast_name}/output_0"
            weight_name = f"model.layers.{layer_id}.moe.router.weight"
            self.make_initializer(moe.gate.weight.T, weight_name, to=ir.DataType.FLOAT)
            matmul_name = f"{basename}/MatMul"
            matmul_output = f"{matmul_name}/output_0"
            self.make_node(
                "MatMul",
                inputs=[router_input, weight_name],
                outputs=[matmul_output],
                name=matmul_name,
            )
            self.make_value(
                matmul_output,
                ir.DataType.FLOAT,
                shape=["batch_size", "sequence_length", self.moe_attrs["num_experts"]],
            )
            router_output = matmul_output
            if self.io_dtype != ir.DataType.FLOAT:
                cast_name = f"{basename}/CastOutput"
                self.make_cast(
                    cast_name,
                    matmul_output,
                    dtype=self.io_dtype,
                    shape=[
                        "batch_size",
                        "sequence_length",
                        self.moe_attrs["num_experts"],
                    ],
                )
                router_output = f"{cast_name}/output_0"
            shape_name = f"{basename}/Shape"
            self.make_shape(shape_name, router_output, shape=[3])
            gather_name = f"{basename}/Gather"
            self.make_gather(
                gather_name,
                [f"{shape_name}/output_0", "/model/constants/INT64/2"],
                dtype=ir.DataType.INT64,
                shape=[],
                axis=0,
            )
            unsqueeze_name = f"{basename}/Unsqueeze"
            self.make_unsqueeze(
                unsqueeze_name,
                [f"{gather_name}/output_0", "/model/constants/INT64/[0]"],
                dtype=ir.DataType.INT64,
                shape=[1],
            )
            concat_name = f"{basename}/Concat"
            self.make_concat(
                concat_name,
                ["/model/constants/INT64/[-1]", f"{unsqueeze_name}/output_0"],
                dtype=ir.DataType.INT64,
                shape=[2],
                axis=0,
            )
            reshape_name = f"{basename}/Reshape"
            self.make_reshape(
                reshape_name,
                [router_output, f"{concat_name}/output_0"],
                dtype=self.io_dtype,
                shape=["num_rows", self.moe_attrs["num_experts"]],
            )
            return f"{reshape_name}/output_0"

    MapleOnnxModel.__name__ = "MapleOnnxModel"
    return MapleOnnxModel


def _load_export_dependencies() -> dict[str, Any]:
    required = _required_export_versions()
    problems: list[str] = []
    for package, exact in required.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{package} is not installed")
            continue
        if installed != exact:
            problems.append(f"{package}=={exact} is required (found {installed})")
    if problems:
        raise MapleOnnxExportError(
            "Maple ONNX export dependencies are incompatible: " + "; ".join(problems)
        )
    try:
        import onnx_ir as ir
        import torch
        from onnxruntime_genai.models.builders.base import Model as BaseModel
        from transformers import AutoConfig
    except ImportError as exc:
        raise MapleOnnxExportError(
            "install the pinned ONNX exporter stack before converting Maple"
        ) from exc
    return {
        "ir": ir,
        "torch": torch,
        "BaseModel": BaseModel,
        "AutoConfig": AutoConfig,
    }


def _install_flash_attention_import_stubs() -> None:
    """Let the pinned remote architecture load without executing FlashAttention."""

    if "flash_attn" in sys.modules or "flash_attn_interface" in sys.modules:
        return

    # Let Transformers make its own availability decisions before the stubs
    # exist. Otherwise importlib.find_spec() can mistake the in-memory package
    # for an installed CUDA FlashAttention distribution.
    importlib.import_module("transformers.activations")
    importlib.import_module("transformers.modeling_utils")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("FlashAttention is unavailable in the ONNX export process")

    interface = types.ModuleType("flash_attn_interface")
    interface.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn_interface", loader=None
    )
    interface.flash_attn_func = unavailable
    interface.flash_attn_varlen_func = unavailable
    package = types.ModuleType("flash_attn")
    package.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None, is_package=True
    )
    package.__path__ = []
    package.flash_attn_func = unavailable
    package.flash_attn_varlen_func = unavailable
    padding = types.ModuleType("flash_attn.bert_padding")
    padding.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn.bert_padding", loader=None
    )
    padding.index_first_axis = unavailable
    padding.pad_input = unavailable
    padding.unpad_input = unavailable
    sys.modules["flash_attn_interface"] = interface
    sys.modules["flash_attn"] = package
    sys.modules["flash_attn.bert_padding"] = padding


def _validate_maple_config(config: Mapping[str, Any]) -> None:
    expected = {
        "architectures": ["MapleForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "vocab_size": 151_936,
        "sliding_window": 512,
        "partial_rotary_factor": 0.5,
        "layer_types": list(_LAYER_TYPES),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise MapleOnnxExportError(
                f"unexpected Maple config value for {key}: {config.get(key)!r}"
            )
    max_positions = config.get("max_position_embeddings")
    if not isinstance(max_positions, int) or max_positions < 128_000:
        raise MapleOnnxExportError("Maple source context length is unexpectedly small")
    if "quantization" in config or "quantization_config" in config:
        raise MapleOnnxExportError(
            "full ONNX export requires the BF16 source, not an MLX quantized config"
        )
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict) or auto_map.get("AutoModelForCausalLM") != (
        "modeling_maple.MapleForCausalLM"
    ):
        raise MapleOnnxExportError("Maple source has an unexpected remote-code map")


def _external_data_files(onnx: Any, model: Any, root: Path) -> tuple[Path, ...]:
    locations: set[str] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for item in tensor.external_data:
            if item.key == "location":
                locations.add(item.value)
    result: list[Path] = []
    for location in sorted(locations):
        _validate_flat_filename(location)
        path = _require_regular_file(root, location)
        result.append(path)
    return tuple(result)


def _onnx_attributes(onnx: Any, node: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in node.attribute:
        value = onnx.helper.get_attribute_value(attribute)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result[attribute.name] = value
    return result


def _validate_cache_tensor(
    onnx: Any,
    value: Any,
    *,
    dtype: int,
) -> None:
    shape = _validate_tensor_contract(onnx, value, dtype=dtype, rank=4)
    if shape[1] != 4 or shape[3] != 128:
        raise MapleOnnxExportError(
            f"{value.name} must have cache shape [batch, 4, sequence, 128]"
        )


def _validate_tensor_contract(
    onnx: Any,
    value: Any,
    *,
    dtype: int,
    rank: int,
) -> tuple[int | str | None, ...]:
    tensor_type = value.type.tensor_type
    if tensor_type.elem_type != dtype:
        expected = onnx.TensorProto.DataType.Name(dtype)
        actual = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        raise MapleOnnxExportError(f"{value.name} must use {expected}, found {actual}")
    dimensions: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    if len(dimensions) != rank:
        raise MapleOnnxExportError(f"{value.name} must have rank {rank}")
    return tuple(dimensions)


def _record_export_artifacts(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name == MAPLE_ONNX_RECEIPT_FILENAME:
            continue
        result.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return result


def _required_export_versions() -> dict[str, str]:
    return {
        "accelerate": MAPLE_ACCELERATE_VERSION,
        "huggingface-hub": MAPLE_HUGGINGFACE_HUB_VERSION,
        "numpy": MAPLE_NUMPY_VERSION,
        "onnx": MAPLE_ONNX_VERSION,
        "onnx-ir": MAPLE_ONNX_IR_VERSION,
        "onnxruntime": MAPLE_ONNX_RUNTIME_EXPORT_VERSION,
        "onnxruntime-genai": MAPLE_ORT_GENAI_VERSION,
        "safetensors": MAPLE_SAFETENSORS_VERSION,
        "tokenizers": MAPLE_TOKENIZERS_VERSION,
        "torch": MAPLE_TORCH_VERSION,
        "tqdm": MAPLE_TQDM_VERSION,
        "transformers": MAPLE_TRANSFORMERS_VERSION,
    }


def _installed_export_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in _required_export_versions()
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MapleOnnxExportError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise MapleOnnxExportError(f"JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _require_regular_file(root: Path, relative: str) -> Path:
    _validate_flat_filename(relative)
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise MapleOnnxExportError(f"missing regular Maple artifact: {relative}")
    return path


def _validate_flat_filename(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise MapleOnnxExportError("artifact paths must be flat POSIX filenames")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {".", ".."}:
        raise MapleOnnxExportError(f"unsafe artifact path: {value!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "requirements", help="print the exact reproducible export dependencies"
    )

    download = subparsers.add_parser("download", help="download pinned BF16 source")
    download.add_argument("destination", type=Path)

    inspect = subparsers.add_parser("inspect", help="inspect a local source snapshot")
    inspect.add_argument("source_directory", type=Path)
    inspect.add_argument("--accept-unverified-source", action="store_true")

    export = subparsers.add_parser("export", help="run the full QMoE conversion")
    export.add_argument("source_directory", type=Path)
    export.add_argument("output_directory", type=Path)
    export.add_argument("--target", choices=("mobile", "webgpu"), default="mobile")
    export.add_argument("--accept-unverified-source", action="store_true")
    export.add_argument("--dry-run", action="store_true")

    bundle = subparsers.add_parser("bundle", help="package a completed export")
    bundle.add_argument("export_directory", type=Path)
    bundle.add_argument("output_path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reproducible Maple ONNX export command line."""

    arguments = _build_parser().parse_args(argv)
    if arguments.command == "requirements":
        print("\n".join(maple_onnx_export_requirements()))
        return 0
    if arguments.command == "download":
        print(
            json.dumps(
                {"source": str(download_maple_onnx_source(arguments.destination))}
            )
        )
        return 0
    if arguments.command == "inspect":
        inspection = inspect_maple_onnx_source(
            arguments.source_directory,
            require_verified_revision=not arguments.accept_unverified_source,
        )
        payload = asdict(inspection)
        payload["source_directory"] = str(inspection.source_directory)
        print(json.dumps(payload, indent=2))
        return 0
    if arguments.command == "bundle":
        bundle = build_maple_onnx_export_bundle(
            arguments.export_directory, arguments.output_path
        )
        print(
            json.dumps({"bundle": str(bundle.bundle_path), **bundle.manifest}, indent=2)
        )
        return 0

    plan = plan_maple_onnx_export(
        arguments.source_directory,
        arguments.output_directory,
        target=arguments.target,
        require_verified_revision=not arguments.accept_unverified_source,
    )
    if arguments.dry_run:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0
    result = export_maple_onnx(
        arguments.source_directory,
        arguments.output_directory,
        target=arguments.target,
        require_verified_revision=not arguments.accept_unverified_source,
    )
    print(
        json.dumps(
            {
                "output_directory": str(result.output_directory),
                "graph": str(result.graph_path),
                "receipt": str(result.receipt_path),
                "validation": result.validation,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main tests
    raise SystemExit(main())


__all__ = [
    "MAPLE_MLX_MODEL",
    "MAPLE_MLX_REVISION",
    "MAPLE_ONNX_GRAPH_FILENAME",
    "MAPLE_ONNX_EXPORT_REQUIREMENTS",
    "MAPLE_ONNX_QUANTIZATION",
    "MAPLE_ONNX_RECEIPT_FILENAME",
    "MAPLE_ONNX_RUNTIME_MIN_VERSION",
    "MAPLE_ORT_GENAI_VERSION",
    "MapleOnnxExportError",
    "MapleOnnxExportPlan",
    "MapleOnnxExportResult",
    "MapleOnnxSourceInspection",
    "build_maple_onnx_export_bundle",
    "download_maple_onnx_source",
    "export_maple_onnx",
    "inspect_maple_onnx_source",
    "maple_onnx_export_requirements",
    "plan_maple_onnx_export",
    "validate_maple_onnx_graph",
]
