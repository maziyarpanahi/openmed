"""Build and execute a tiny Maple-shaped QMoE compatibility probe.

The probe is intentionally synthetic and contains no model weights.  It tests
the exact four-bit, block-wise, fused SwiGLU QMoE form used by the Maple ONNX
exporter against an installed ONNX Runtime CPU kernel.  Passing this probe is
not evidence that the full checkpoint has been converted or validated on a
phone or in a browser.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

MAPLE_QMOE_SMOKE_MIN_ORT_VERSION = "1.25.1"
MAPLE_QMOE_SMOKE_BLOCK_SIZE = 16


class MapleQMoESmokeError(RuntimeError):
    """Raised when the synthetic QMoE compatibility probe cannot run."""


@dataclass(frozen=True)
class MapleQMoESmokeResult:
    """Result of one synthetic QMoE CPU execution."""

    model_path: Path
    onnxruntime_version: str
    output_shape: tuple[int, ...]
    maximum_absolute_output: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""

        payload = asdict(self)
        payload["model_path"] = str(self.model_path)
        return payload


def write_maple_qmoe_smoke_model(output_path: str | Path) -> Path:
    """Write a tiny four-bit QMoE graph with Maple's expert layout.

    The packed nibbles use code 8, the zero point for signed INT4 weights, so
    the expected output is exactly zero.  The first expert matrix contains
    interleaved gate/up rows and uses ``swiglu_fusion=1`` just like Maple.

    Args:
        output_path: New ``.onnx`` path. Existing files are never overwritten.

    Returns:
        Resolved path to the written graph.
    """

    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - optional export dependencies
        raise MapleQMoESmokeError(
            "onnx>=1.21 and numpy are required to write the QMoE probe"
        ) from exc

    path = Path(output_path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite QMoE probe: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    experts = 2
    hidden_size = 64
    intermediate_size = 64
    packed_zero = np.uint8(0x88)
    fc1_weight = np.full(
        (experts, 2 * intermediate_size, hidden_size // 2),
        packed_zero,
        dtype=np.uint8,
    )
    fc1_scale = np.ones(
        (
            experts,
            2 * intermediate_size,
            hidden_size // MAPLE_QMOE_SMOKE_BLOCK_SIZE,
        ),
        dtype=np.float32,
    )
    fc2_weight = np.full(
        (experts, hidden_size, intermediate_size // 2),
        packed_zero,
        dtype=np.uint8,
    )
    fc2_scale = np.ones(
        (
            experts,
            hidden_size,
            intermediate_size // MAPLE_QMOE_SMOKE_BLOCK_SIZE,
        ),
        dtype=np.float32,
    )
    initializers = [
        numpy_helper.from_array(fc1_weight, name="fc1_weight"),
        numpy_helper.from_array(fc1_scale, name="fc1_scale"),
        numpy_helper.from_array(fc2_weight, name="fc2_weight"),
        numpy_helper.from_array(fc2_scale, name="fc2_scale"),
    ]
    node = helper.make_node(
        "QMoE",
        [
            "hidden_states",
            "router_logits",
            "fc1_weight",
            "fc1_scale",
            "",
            "fc2_weight",
            "fc2_scale",
        ],
        ["output"],
        name="maple_qmoe_smoke",
        domain="com.microsoft",
        activation_alpha=1.0,
        activation_beta=0.0,
        activation_type="swiglu",
        block_size=MAPLE_QMOE_SMOKE_BLOCK_SIZE,
        expert_weight_bits=4,
        k=1,
        normalize_routing_weights=1,
        swiglu_fusion=1,
        swiglu_limit=7.0,
        use_sparse_mixer=0,
    )
    graph = helper.make_graph(
        [node],
        "maple_qmoe_smoke",
        [
            helper.make_tensor_value_info(
                "hidden_states", TensorProto.FLOAT, [1, 1, hidden_size]
            ),
            helper.make_tensor_value_info(
                "router_logits", TensorProto.FLOAT, [1, experts]
            ),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 64])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
        producer_name="openmed-maple-qmoe-smoke",
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def run_maple_qmoe_smoke(model_path: str | Path) -> MapleQMoESmokeResult:
    """Execute a synthetic Maple QMoE graph with ONNX Runtime CPU.

    Args:
        model_path: Probe created by :func:`write_maple_qmoe_smoke_model`.

    Returns:
        Runtime version, output shape, and maximum absolute output.
    """

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional runtime dependencies
        raise MapleQMoESmokeError(
            "onnxruntime>=1.25.1 and numpy are required to run the QMoE probe"
        ) from exc

    runtime_version = importlib.metadata.version("onnxruntime")
    if _version_tuple(runtime_version) < _version_tuple(
        MAPLE_QMOE_SMOKE_MIN_ORT_VERSION
    ):
        raise MapleQMoESmokeError(
            "the Maple QMoE probe requires onnxruntime>="
            f"{MAPLE_QMOE_SMOKE_MIN_ORT_VERSION}; found {runtime_version}"
        )
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"QMoE probe does not exist: {path}")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    hidden_states = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(1, 1, 64)
    router_logits = np.array([[8.0, -8.0]], dtype=np.float32)
    output = session.run(
        ["output"],
        {"hidden_states": hidden_states, "router_logits": router_logits},
    )[0]
    maximum = float(np.max(np.abs(output)))
    if output.shape != (1, 1, 64) or not np.all(np.isfinite(output)):
        raise MapleQMoESmokeError("QMoE probe returned an invalid output tensor")
    if maximum != 0.0:
        raise MapleQMoESmokeError(
            f"zero-encoded QMoE weights produced a non-zero output: {maximum}"
        )
    return MapleQMoESmokeResult(
        model_path=path,
        onnxruntime_version=runtime_version,
        output_shape=tuple(int(item) for item in output.shape),
        maximum_absolute_output=maximum,
    )


def validate_maple_qmoe_runtime() -> MapleQMoESmokeResult:
    """Build and execute the synthetic QMoE probe in a temporary directory."""

    with tempfile.TemporaryDirectory(prefix="openmed-maple-qmoe-") as temporary:
        model_path = write_maple_qmoe_smoke_model(Path(temporary) / "qmoe.onnx")
        result = run_maple_qmoe_smoke(model_path)
        return MapleQMoESmokeResult(
            model_path=Path("<temporary>/qmoe.onnx"),
            onnxruntime_version=result.onnxruntime_version,
            output_shape=result.output_shape,
            maximum_absolute_output=result.maximum_absolute_output,
        )


def _version_tuple(version: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", version)
    return tuple(int(item) for item in match.group(1).split(".")) if match else (0,)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the synthetic Maple QMoE compatibility probe."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="keep the generated probe at this new path instead of using a temp file",
    )
    arguments = parser.parse_args(argv)
    if arguments.output is None:
        result = validate_maple_qmoe_runtime()
    else:
        result = run_maple_qmoe_smoke(write_maple_qmoe_smoke_model(arguments.output))
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main tests
    raise SystemExit(main())


__all__ = [
    "MAPLE_QMOE_SMOKE_BLOCK_SIZE",
    "MAPLE_QMOE_SMOKE_MIN_ORT_VERSION",
    "MapleQMoESmokeError",
    "MapleQMoESmokeResult",
    "run_maple_qmoe_smoke",
    "validate_maple_qmoe_runtime",
    "write_maple_qmoe_smoke_model",
]
