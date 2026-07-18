# OpenVINO Runtime

OpenMed can export ONNX token-classification graphs to OpenVINO IR for Intel
CPU, GPU, and NPU edge deployments:

```bash
.venv/bin/python -m openmed.onnx.convert \
  --model dslim/bert-base-NER \
  --output dist/example-openvino \
  --profile openvino
```

The profile writes the ONNX source graph plus an OpenVINO IR directory:

```text
dist/example-openvino/
  model.onnx
  config.json
  id2label.json
  openmed-onnx.json
  openvino/
    model.xml
    model.bin
```

`openmed-onnx.json` records the `openvino-ir` artifact and includes synthetic
verification metadata. The verifier runs a synthetic note through ONNX Runtime
and the exported OpenVINO graph, compares logits within a fixed tolerance, and
checks that decoded token spans match.

## Runtime Session

Use `OpenVinoTokenClassificationSession` when loading an exported IR graph:

```python
from openmed.onnx import OpenVinoTokenClassificationSession

session = OpenVinoTokenClassificationSession(
    "dist/example-openvino/openvino/model.xml",
    device="NPU",
)

logits = session.run(input_ids=input_ids, attention_mask=attention_mask)
```

Device selection is deterministic. The requested device is used when present;
otherwise the runtime falls back through `CPU`, `GPU`, then `NPU`, and records
whether fallback was used. If OpenVINO reports no devices, session creation
fails instead of guessing.

## INT8 Quantization

INT8 export uses NNCF post-training quantization and is fail-closed by the G4
recall-delta gate. Callers must provide calibration samples plus per-family
recall evidence from the synthetic eval or a precomputed recall-delta payload:

```python
from openmed.onnx import quantize_openvino_int8

result = quantize_openvino_int8(
    "dist/example-openvino/openvino/model.xml",
    "dist/example-openvino/openvino_int8",
    calibration_data=[{"input_ids": input_ids, "attention_mask": attention_mask}],
    family="bert",
    candidate_recall={"PERSON": 0.990},
    parent_recall={"PERSON": 0.992},
)
```

If recall evidence is missing, or any evaluated G1/G2 label loses at least the
INT8 threshold, `OpenVinoQuantizationRejected` is raised and no INT8 artifact is
accepted.

## Benchmark Records

OpenVINO device results are written with the standard `BenchmarkReport` schema:

```python
from openmed.onnx import OpenVinoBenchmarkRecord, write_openvino_benchmark_report

write_openvino_benchmark_report(
    "dist/example-openvino/openvino-benchmark.report.json",
    model_name="dslim/bert-base-NER",
    records=[
        OpenVinoBenchmarkRecord(
            device="CPU",
            precision="float32",
            latency_ms=4.0,
            throughput_items_per_second=250.0,
            sample_count=3,
            sequence_length=128,
        )
    ],
)
```

Each device record contains latency, throughput, precision, batch size, and
optional sequence length under `metrics.devices.<device>`.
