# CPU INT8 Classification Fast Path

OpenMed provides a local INT8 fast path for the token-classification head that
runs after a backbone produces hidden states. It quantizes each token row,
executes the pre-quantized classification-head matmul with int32 accumulation,
and fuses winning-label scoring with BIO/BIOES span decoding. The backbone stays
inside its existing runtime.

Install the ONNX Runtime extra so NumPy and the local inference dependencies are
available:

```bash
pip install "openmed[onnx-runtime]"
```

## Prepare and certify a head

Quantize the float classifier once, then run the numerical guard on a synthetic
calibration note before using the hot path:

```python
from openmed.onnx import (
    quantize_classification_head,
    run_cpu_fastpath,
    verify_cpu_fastpath,
)

head = quantize_classification_head(classifier_weight, classifier_bias)

verification = verify_cpu_fastpath(
    synthetic_hidden_states,
    classifier_weight,
    classifier_bias,
    head,
    synthetic_offsets,
    id2label,
    threshold=0.5,
)
assert verification.passed

result = run_cpu_fastpath(
    hidden_states,
    head,
    offsets,
    id2label,
    threshold=0.5,
)
```

The fixed default logit tolerance is `0.125`. Certification fails closed with
`CpuFastPathVerificationError` if the maximum absolute logit delta exceeds that
limit, any winning label changes, or any decoded span boundary or label differs
from the float32 reference. Run certification when loading or building an
artifact, not for every note, because it intentionally executes both paths.

`CpuFastPathResult` reports the selected kernel, dequantized logits, winning
label IDs, winning probabilities, and source-offset spans. It does not retain
input text or emit telemetry.

## CPU dispatch

`detect_cpu_features()` selects the strongest kernel that is safe for the
current process:

1. AVX-512 on x86 when both `AVX512F` and `AVX512BW` are reported
2. AVX2 on x86
3. NEON/ASIMD on ARM
4. deterministic scalar fallback

Linux reads processor flags from `/proc/cpuinfo`; macOS uses `sysctl`. Unknown
architectures, unavailable flag sources, and x86 CPUs without AVX2 stay on the
scalar kernel. Architecture gating prevents ARM flags from enabling an x86
kernel and vice versa. Explicit `CpuFeatures` values are intended for tests and
reproducible benchmark profiles.

All tiers use the same symmetric INT8 values and int32 accumulation. SIMD tiers
run aligned, vectorized NumPy tiles; the scalar tier uses a portable elementwise
accumulator. That makes scalar-versus-SIMD decisions directly comparable.

## Benchmark record

Run the bundled synthetic benchmark on the target CPU:

```bash
python -m openmed.onnx.cpu_fastpath \
  --sequence-length 48 \
  --hidden-size 256 \
  --labels 9 \
  --iterations 7 \
  --warmup 2 \
  --require-speedup
```

The command emits a `CpuFastPathBenchmarkRecord` as JSON. It records the exact
detected tier, shape, iteration count, median scalar and selected-kernel
latencies, speedup, and the `1.05x` minimum-speedup decision. It exits
unsuccessfully when a SIMD tier does not clear that minimum.

The committed acceptance run below used the fixed seed `479`; both measurements
include per-token quantization and fused BIO decoding:

| Date | CPU | Tier | Shape | Iterations | Scalar | Fast path | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| 2026-07-19 | Apple M2 Max, arm64 | NEON | `48x256x9` | 7 | 11.522 ms | 0.173 ms | 66.75x |

`tests/unit/onnx/test_cpu_fastpath.py` also runs the same wall-clock comparison
when live detection selects AVX2 as the strongest tier and requires a
non-trivial reduction. Other machines skip that hardware-specific assertion.
Deterministic unit coverage separately checks the benchmark record schema and
the `1.05x` gate without relying on wall-clock timing.

The benchmark compares the vectorized kernel with the portable scalar INT8
kernel. Use `verify_cpu_fastpath()` for the separate float32 numerical and span
parity check.
