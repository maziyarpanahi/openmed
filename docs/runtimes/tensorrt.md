# TensorRT Runtime

OpenMed builds device-specific TensorRT engines from exported ONNX
token-classification models. The builder supports variable sequence lengths,
FP16, fail-closed INT8 calibration, ONNX-reference certification,
reproducibility hashes, and device-tier benchmark records.

TensorRT engines contain executable GPU code and are tied to the GPU and
software platform that built them. Build engines on the deployment target and
only deserialize engines you built locally or received from a trusted source.
OpenMed does not bundle TensorRT, CUDA, or prebuilt engines.

## Prerequisites

Export the model to ONNX first, then install NVIDIA TensorRT and a CUDA-enabled
PyTorch build for the target platform. Follow the
[NVIDIA TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/overview.html)
for the matching GPU, CUDA, and TensorRT versions.

TensorRT 10 and earlier use the entropy-calibrator API for INT8. TensorRT 11
removed implicit calibration and weak precision flags, so OpenMed invokes
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) to create an
explicit Q/DQ INT8 graph or FP16-cast graph before building the engine:

```bash
python -m pip install "nvidia-modelopt[all]"
```

Model Optimizer is optional and is not installed or distributed by OpenMed.

## Build an Engine

```python
from openmed.onnx import TensorRTShapeProfile, build_tensorrt_engine

result = build_tensorrt_engine(
    "dist/example/model.onnx",
    "dist/example/model.engine",
    family="deberta-v2",
    precision="fp16",
    shape_profile=TensorRTShapeProfile(
        min_sequence_length=8,
        opt_sequence_length=128,
        max_sequence_length=512,
    ),
)

print(result.engine_sha256)
print(result.metadata_path)
```

TensorRT optimization profiles define the minimum, optimum, and maximum batch
and sequence shapes accepted by dynamic rank-two inputs such as `input_ids`,
`attention_mask`, and `token_type_ids`. Runtime input shapes outside this range
are rejected by TensorRT. See NVIDIA's
[dynamic-shape guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dynamic-shapes.html)
for profile behavior and memory tradeoffs.

## INT8 Calibration and the G4 Gate

INT8 always uses OpenMed's shared deterministic synthetic clinical calibration
loader unless `calibration_texts` is explicitly supplied. The tokenizer converts
those texts to fixed optimum-profile shapes. Raw calibration text is never
written into the engine metadata.

Every INT8 build is fail-closed: per-family recall evidence must pass the G4
INT8 recall-delta limit before calibration or engine construction begins.

```python
from transformers import AutoTokenizer

from openmed.onnx import build_tensorrt_engine

tokenizer = AutoTokenizer.from_pretrained("OpenMed/example-model")

result = build_tensorrt_engine(
    "dist/example/model.onnx",
    "dist/example/model-int8.engine",
    family="deberta-v2",
    precision="int8",
    calibration_tokenizer=tokenizer,
    parent_recall={"PERSON": 0.994, "ID_NUM": 0.997},
    candidate_recall={"PERSON": 0.992, "ID_NUM": 0.996},
)
```

Missing evidence, an empty calibration set, or a recall loss at or above the
INT8 threshold raises `TensorRTQuantizationRejected`. No engine is published.
The build metadata records `gate: G4`, the per-label gate result, the calibration
digest, and the TensorRT version.

TensorRT 10 and earlier may reuse a local calibration cache with
`calibration_cache_path`. Reuse the cache only when the ONNX graph, calibration
set, TensorRT version, and target device are unchanged.

## Synthetic ONNX Parity

Pass a tokenized synthetic note, its ONNX Runtime logits, and the model label map
to certify a newly built engine before it replaces the destination path:

```python
result = build_tensorrt_engine(
    "dist/example/model.onnx",
    "dist/example/model.engine",
    family="bert",
    precision="fp16",
    sample_inputs={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    },
    reference_logits=onnx_logits,
    id2label={0: "O", 1: "B-NAME", 2: "I-NAME"},
)
```

The verifier requires matching output shapes, logits within tolerance, and
identical decoded token spans. A failure raises `TensorRTVerificationError` and
does not replace the destination engine. The GPU parity test skips with a clear
message on runners without TensorRT or CUDA.

## Reproducibility Hashes

Each `.engine.build.json` records two reproducibility checks:

- `build_input_sha256` covers the source and effective ONNX digests, family,
  precision, shape profile, workspace limit, TensorRT version, calibration
  digest, and G4 evidence.
- `engine_sha256` covers the serialized engine bytes.

Pin either value during a rebuild:

```python
result = build_tensorrt_engine(
    "dist/example/model.onnx",
    "dist/example/model.engine",
    family="bert",
    expected_build_input_sha256="<recorded-build-input-sha256>",
    expected_engine_sha256="<recorded-engine-sha256>",
)
```

A mismatch raises `TensorRTReproducibilityError` before the engine is published.
Exact engine hashes require the same GPU, TensorRT/CUDA stack, build inputs, and
builder behavior. Use the build-input hash to distinguish input drift from a
different tactic selection or platform.

## Runtime Session

```python
from openmed.onnx import TensorRTTokenClassificationSession

session = TensorRTTokenClassificationSession("dist/example/model.engine")
logits = session.run(
    input_ids=input_ids,
    attention_mask=attention_mask,
)
```

The session supports TensorRT's named-I/O execution API and the older binding
API. It allocates CUDA buffers with PyTorch, resolves dynamic output shapes, and
returns logits as a NumPy array.

## Device-Tier Benchmarks

Record latency and throughput separately for each deployment tier:

```python
from openmed.onnx import (
    TensorRTBenchmarkRecord,
    write_tensorrt_benchmark_report,
)

write_tensorrt_benchmark_report(
    "dist/example/tensorrt-benchmark.report.json",
    model_name="OpenMed/example-model",
    records=[
        TensorRTBenchmarkRecord(
            device_tier="jetson-orin",
            device="Jetson AGX Orin",
            precision="fp16",
            latency_ms=4.0,
            throughput_items_per_second=250.0,
            sample_count=50,
            sequence_length=128,
        )
    ],
)
```

The standard `BenchmarkReport` output stores the tier, exact device name,
precision, latency, throughput, batch size, sequence length, and sample count.
