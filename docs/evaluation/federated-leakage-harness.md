# Federated leakage harness

Use `openmed.eval.run_federated_leakage_eval` to check whether a local detector
exposes known identifier text across its subprocess boundary. Give it synthetic
`BenchmarkFixture` objects with unique IDs, text, and gold spans. The harness
starts one Python subprocess per fixture and keeps the gold annotations in the
parent process.

## Detector protocol

Pass the path to a Python script as `detector`. Use `FederatedDetectorSpec` to
set a different Python executable, per-fixture timeout, or allowed read roots.
The default timeout is 10 seconds. The child starts in a temporary output
directory and receives these paths through environment variables.

| Variable | Path and use |
| --- | --- |
| `OPENMED_DETECTOR_INPUT` | Read this UTF-8 JSON file. It contains `fixture_id`, `language`, `metadata`, and `text`. |
| `OPENMED_DETECTOR_OUTPUT` | Write UTF-8 JSON here with a `spans` array. Each span should contain `start`, `end`, and `label`. Use zero-based character offsets into `text` with an exclusive `end`. |
| `OPENMED_DETECTOR_OUTPUT_DIR` | Put any additional files here. The harness monitors their contents and relative paths. |

Return offsets and labels without copying source text into the output. The
Python audit hook blocks monitored socket and process-launch operations and
checks selected filesystem operations against read and write allowlists. The
report records violations, including timeouts, and captures process counts,
elapsed times, exit codes, and emitted byte counts.

## What the harness measures

The harness searches stdout, stderr, the output JSON, and side-file contents and
relative paths for exact UTF-8 byte matches to gold span text. The leakage rate
is the number of leaked gold bytes divided by the total gold bytes. A gold span
counts once toward leaked bytes even if it appears in multiple outputs. Findings
record labels, source offsets, match byte offsets, and hashes of matched text.
File findings use a hashed artifact identifier.

For timing evaluation, add a top-level `timings` array to the output JSON with
`start`, `end`, and `duration_ms` for each sample. The probe aligns these
detector-reported timings with gold spans and estimates mutual information
between timing buckets split at the median and a bit derived from the gold span
text. It flags estimates at or above `side_channel_threshold_bits` (default
`0.30`) when at least `side_channel_min_samples` (default `4`) align with gold
spans. Fewer usable samples produce an unflagged result with
`insufficient_timing_samples`. Process elapsed time is resource accounting.

## Signing and the boundary gate

Supply a non-empty `signing_key` (bytes or a string) or set
`OPENMED_FEDERATED_EVAL_KEY` for the parent process. An explicit key takes
precedence. Since v4.0 there is no built-in key. If neither source supplies one,
the harness raises `ValueError` before starting the detector. The parent
uses the key to sign the report. Keep signing material out of detector
environment overrides, fixture metadata, and reports.

The harness returns a signed `FederatedEvalReport`. Use `report.verify(key)`
with the same key to verify its HMAC-SHA256 signature and reproducibility hash.
The signature covers resource accounting, which the reproducibility hash
excludes. Use `report.to_json()` to serialize the signed report.

`report.gate_passed` requires leakage at or below `max_boundary_leakage_rate`
(default zero), no timing flag, and no sandbox violations. Call
`openmed.eval.evaluate_federated_boundary_gate(report)` for a `GateCheck` with
`passed`, `reason`, and `details`. This helper enforces zero boundary leakage
even if the run used a higher allowance. It also rejects timing flags and
sandbox violations. It checks the evidence independently of F1 and leaves
signature verification to the caller. An empty prediction set can pass the
boundary checks.

## Run a synthetic example

Save this as `detector.py` in an otherwise empty directory.

```python
import json
import os
from pathlib import Path

input_path = Path(os.environ["OPENMED_DETECTOR_INPUT"])
payload = json.loads(input_path.read_text(encoding="utf-8"))
name = "Example Person"
start = payload["text"].index(name)
spans = [{"start": start, "end": start + len(name), "label": "PERSON"}]
output = Path(os.environ["OPENMED_DETECTOR_OUTPUT"])
output.write_text(json.dumps({"spans": spans}), encoding="utf-8")
```

With OpenMed installed, save this as `evaluate.py` alongside `detector.py` and
run `python evaluate.py` from that directory. The example uses a fresh key.

```python
from secrets import token_bytes

from openmed.eval import (
    BenchmarkFixture,
    EvalSpan,
    evaluate_federated_boundary_gate,
    run_federated_leakage_eval,
)

name = "Example Person"
text = f"Patient {name} arrived for follow-up."
start = text.index(name)
fixture = BenchmarkFixture(
    fixture_id="synthetic-1",
    text=text,
    gold_spans=(EvalSpan(start=start, end=start + len(name), label="PERSON"),),
)
key = token_bytes(32)
report = run_federated_leakage_eval([fixture], detector="detector.py", signing_key=key)
assert report.verify(key)
check = evaluate_federated_boundary_gate(report)
print(report.boundary_leakage.rate, check.passed)  # 0.0 True
```

## Scope limits

Use synthetic text and non-sensitive identifiers and metadata. The child
receives the fixture text, and the audit hook provides Python-level checks only.
The exact-byte scan and timing probe cover the captured artifacts and usable
samples described above. A pass provides no OS-level isolation guarantee,
assurance for secrets or patient data, clinical certification, or authorization
to release a detector.
