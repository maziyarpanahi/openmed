# AWQ grounding embedder

OpenMed can export a 4-bit AWQ grounding embedder and certify it against its fp16 parent before the artifact is admitted to the grounding path. The certificate measures top-k passage overlap on committed synthetic retrieval fixtures. If the recall delta exceeds the tolerance, quantization raises an error and the loader refuses the artifact.

Install the optional export dependencies in an isolated environment:

```bash
pip install "openmed[awq]"
```

The Linux-only `awq` extra pins AutoAWQ 0.2.9 to Transformer 5.5.0: the first
Transformer 5 release containing all current HIGH-severity fixes and the exact
pair covered by OpenMed's import smoke. Use a dedicated environment because the
archived quantization stack is large and hardware-specific. Use MLX or CoreML
on macOS.

## Quantize and certify

Use a model revision or immutable local snapshot when exporting. Calibration always comes from OpenMed's shared, committed synthetic calibration loader; the recipe does not accept a terminology or restricted-vocabulary corpus.

```python
from openmed.torch import quantize_awq_grounding

result = quantize_awq_grounding(
    "your-org/grounding-embedder",
    "artifacts/grounding-awq",
    revision="<immutable-commit-sha>",
    top_k=3,
    recall_delta_tolerance=0.05,
)

print(result.recall_gate.passed)
print(result.benchmark_report_path)
```

The recipe records the ordered calibration digest, source revision, group size, and synthetic retrieval-fixture digest. Given the same source snapshot and committed inputs, these fields pin the quantization inputs for reproducible reruns.

## Recall gate

For each synthetic query, the certificate retrieves the top-k passage indexes from both the fp16 parent and the AWQ artifact. Per-query overlap is the fraction of fp16 top-k indexes retained by AWQ. The gate uses:

```text
recall_delta = 1 - mean(per_query_top_k_overlap)
```

The artifact passes only when `recall_delta <= recall_delta_tolerance`. Empty fixtures, invalid or non-finite embeddings, missing evidence, inconsistent metadata, and excessive recall loss all fail closed.

## Benchmark evidence

`grounding_awq_benchmark.json` uses OpenMed's `BenchmarkReport` shape. It records:

- `metrics.retrieval`: top-k, per-query overlap, mean overlap, recall delta, tolerance, and pass/fail status.
- `metrics.latency.fp16` and `metrics.latency.awq`: p50, p95, p99, and sample count in milliseconds per encoded item.
- `metrics.resources.model_size_bytes`: the AWQ artifact size at certification time.
- `metadata.calibration`: the shared synthetic source, sample count, and SHA-256 digest.
- `metadata.grounding_fixture_sha256`: the ordered synthetic retrieval-fixture digest.

The report stores no calibration passages, queries, terminology rows, or raw clinical text.

## Load on the grounding path

```python
from openmed.torch import load_awq_grounding_embedder

embedder = load_awq_grounding_embedder("artifacts/grounding-awq")
vectors = embedder.encode(["synthetic mention", "synthetic concept label"])
```

The returned adapter exposes `encode(texts)` and produces L2-normalized, attention-mask-aware mean-pooled hidden states. `load_awq_grounding_embedder` validates both `quant_config.json` and `grounding_awq_benchmark.json` before importing the quantized model. There is no flag to bypass a failed or missing certificate.

The loader disables fixed-cache fused generation modules so grounding can use
variable batch sizes and obtain the hidden states required for mean pooling.

## Data and runtime boundaries

- The bundled calibration and certification text is synthetic and permissive. OpenMed does not bundle UMLS, SNOMED CT, CPT, MIMIC, i2b2, n2c2, or other restricted vocabularies or datasets.
- Quantization and model resolution are explicit, user-initiated operations. Set `local_files_only=True` during export when the source snapshot is already cached.
- The loader reads the local artifact and tokenizer only. It does not download terminology data or make telemetry calls.
- AWQ support is intended for compatible GPU-backed AutoAWQ model families; model-family compatibility remains a property of the selected upstream checkpoint.
