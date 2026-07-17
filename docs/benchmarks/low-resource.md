# Low-resource de-identification benchmark

The `low_resource` profile is the CPU-only preset for machines with limited
memory. It selects the official 44M-parameter PII model's INT8 ONNX artifact,
uses `CPUExecutionProvider`, processes one note at a time, loads the model only
on first use, and refuses to fall back to a Torch backend.

## Resource envelope

The regression workload de-identifies 100 deterministic synthetic clinical
notes. Each note contains fabricated names, dates, medical record numbers,
phone numbers, and email addresses; no real patient data is used or written to
the benchmark report.

| Metric | Recorded baseline | Required gate |
|---|---:|---:|
| Peak RSS | 1,620.875 MiB | Less than 2,560 MiB (2.5 GiB) |
| Peak RSS growth | 1,523.875 MiB | Informational |
| Throughput | 29.609 notes/s | Informational |
| Duration | 3.377 s for 100 notes | Informational |
| Detected entities | 579 | At least one per note |

The recorded baseline was measured on 2026-07-17 with Python 3.11.10 on
Apple arm64 after the model artifact was cached. The pull-request workflow runs
the same command in a fresh Linux container constrained to 4 GiB RAM, 4 GiB
combined memory/swap (so swap is disabled), two CPUs, and no visible GPU. The
committed JSON baseline is
[`low-resource-baseline.json`](low-resource-baseline.json).

CI enforces two independent limits:

1. Absolute peak RSS must stay below 2.5 GiB.
2. Peak RSS must not exceed the committed baseline by more than 10% (currently
   1,782.963 MiB).

## Reproduce locally

Install the project with the CPU-only runtime; the `onnx-runtime` extra does not
install Torch:

```bash
uv sync --frozen --extra dev --extra onnx-runtime --python 3.11
```

Cache the exact model files used by the benchmark:

```bash
.venv/bin/hf download \
  OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-onnx-android \
  model_int8.onnx config.json id2label.json \
  tokenizer.json tokenizer_config.json \
  --cache-dir ~/.cache/openmed
```

Run the 100-note gate:

```bash
OPENMED_PROFILE=low_resource CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python scripts/benchmarks/low_resource_deid.py \
  --notes 100 \
  --baseline docs/benchmarks/low-resource-baseline.json \
  --max-regression-percent 10 \
  --max-peak-rss-mib 2560 \
  --require-no-gpu
```

The command prints aggregate counts, timing, throughput, cgroup limit, peak RSS,
runtime variant, and whether Torch entered `sys.modules`. It never prints input
notes or detected surface text.

## Reproduce the 4 GiB CI envelope

From the repository root with Docker available:

```bash
docker run --rm \
  --memory 4g --memory-swap 4g --cpus 2 \
  -e OPENMED_PROFILE=low_resource \
  -e CUDA_VISIBLE_DEVICES='' \
  -e UV_PROJECT_ENVIRONMENT=/tmp/openmed-venv \
  -v "$PWD:/workspace" -w /workspace python:3.11-slim \
  sh -c 'pip install uv && \
    uv sync --frozen --extra dev --extra onnx-runtime --python 3.11 && \
    /tmp/openmed-venv/bin/hf download \
      OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-onnx-android \
      model_int8.onnx config.json id2label.json \
      tokenizer.json tokenizer_config.json \
      --cache-dir /root/.cache/openmed && \
    /tmp/openmed-venv/bin/python \
      scripts/benchmarks/low_resource_deid.py \
      --notes 100 \
      --baseline docs/benchmarks/low-resource-baseline.json \
      --max-regression-percent 10 \
      --max-peak-rss-mib 2560 \
      --require-cgroup-limit-gib 4 \
      --require-no-gpu'
```

`openmed doctor` reports the effective physical or cgroup-limited RAM, whether
the machine meets the 4 GiB envelope, and suggests
`OPENMED_PROFILE=low_resource` whenever available RAM is below 8 GiB.
