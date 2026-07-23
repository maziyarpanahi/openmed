# ARM SMS Latency Budget

OpenMed gates cached INT8 ONNX PII inference over a committed corpus of
synthetic clinical messages. Every message is 280 characters or shorter, the
report contains aggregate measurements only, and the measured command blocks
Python socket connections for its entire model-load and inference window. It
also verifies the exact `model_int8.onnx` bytes against the committed SHA-256
digest before any warm-up or measured inference.

## Raspberry Pi 5 reference envelope

The clinic-grade reference is a Raspberry Pi 5 with 8 GB RAM and its 4-core Arm
Cortex-A76 CPU. Issue #1456 records a p95 target of 1.5 seconds per SMS-scale
text. The committed budget applies the allowed 20% regression tolerance only
at gate time.

| Metric | Reference number |
|---|---:|
| p95 single-text latency budget | 1,500 ms |
| Throughput implied by the p95 budget | 0.667 texts/s |
| Permitted regression | 20% |
| Gated maximum p95 | 1,800 ms |

The source specification did not include a raw Raspberry Pi capture for p50 or
peak RSS, so those values are deliberately not invented here. Each JSON report
records the measured p50, p95, throughput, and peak RSS for the machine that ran
it, together with OS, architecture, CPU count, model revision, ONNX artifact,
and quantization metadata. A Raspberry Pi reference run should retain that JSON
alongside the exact OpenMed and model revisions before its measured p50 and RSS
are promoted into this table.

## Exact reproduction

Install the runtime and download the pinned model while networking is allowed:

```bash
python -m pip install -e ".[onnx-runtime]"
export OPENMED_ARM_MODEL_DIR="${PWD}/.cache/openmed-pii-int8"
hf download OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android \
  model_int8.onnx config.json id2label.json tokenizer.json tokenizer_config.json \
  --revision 79f7db205869b1be4be23ac4f42aa95bdedc5aee \
  --local-dir "$OPENMED_ARM_MODEL_DIR"
```

Then disconnect networking if desired and run the cache-only benchmark:

```bash
OPENMED_OFFLINE=1 openmed benchmark latency \
  --model "$OPENMED_ARM_MODEL_DIR" \
  --revision 79f7db205869b1be4be23ac4f42aa95bdedc5aee \
  --output arm-latency-report.json
```

The command always requests `model_int8.onnx`, passes
`local_files_only=True`, and activates OpenMed's socket guard even when
`OPENMED_OFFLINE` was not already set. Before measuring, it requires SHA-256
`48a0b2e9269933bef0cf8913239d07996fa2afb107cd223ced95c8decd24ae6b`,
which pins the downloaded INT8 graph in addition to the model revision. It
exits non-zero when measured p95 is greater than 1,800 ms, while still writing
and printing the JSON report.

The default run performs one excluded warm-up inference and measures three
passes over `openmed/eval/fixtures/sms_clinical.jsonl`. Override `--repeat` or
`--warmup-runs` only when recording an explicitly different experiment.

## Report contract

The JSON document includes these stable top-level measurements:

```json
{
  "benchmark": "sms_arm_latency",
  "latency_ms": {"p50": 0.0, "p95": 0.0},
  "throughput_texts_per_second": 0.0,
  "peak_rss_mib": 0.0,
  "model": {
    "artifact": "model_int8.onnx",
    "artifact_sha256": "48a0b2e9269933bef0cf8913239d07996fa2afb107cd223ced95c8decd24ae6b"
  },
  "offline": true,
  "passed": true
}
```

Values above are schema placeholders, not claimed measurements. The real
report also includes the corpus bounds, machine and model provenance, sample
count, total duration, and the complete budget verdict. It never includes the
synthetic source messages.

## CI gate and negative test

`.github/workflows/container-multiarch.yml` runs the benchmark on GitHub's
native 4-core `ubuntu-24.04-arm` runner. The job downloads the pinned artifact
before the measured phase, sets `OPENMED_OFFLINE=1`, executes the command, and
uploads `arm-latency-report.json`. Container publication depends on this job.
The hosted runner is the reproducible regression sentinel; it does not pretend
to be a Raspberry Pi measurement.

The intentionally slowed fixture proves that the 20% envelope fails closed:

```bash
python -m pytest \
  tests/unit/eval/test_arm_latency.py::test_intentionally_slowed_fixture_trips_gate \
  -q
```

The CLI socket-guard assertion is covered separately:

```bash
python -m pytest \
  tests/unit/cli/test_benchmark_cli.py::test_latency_command_emits_offline_int8_json_and_blocks_sockets \
  -q
```
