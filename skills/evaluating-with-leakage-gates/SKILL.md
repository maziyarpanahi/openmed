---
name: evaluating-with-leakage-gates
description: "Evaluate an OpenMed de-identification or clinical NER model against the leakage-first release gates G1a through G8, which gate releases on residual PHI leakage rather than on F1. Use when the user wants to run the OpenMed eval harness on a synthetic golden set, decide whether a de-id model is RELEASABLE or QUARANTINED, enforce direct-identifier recall floors, require zero critical leakage, fit calibration thresholds, or produce a signed gate report. Trigger on \"release gate\", \"leakage\", \"is this model safe to ship\", \"G1a\", \"G3\", \"quarantine\", \"recall floor\", or \"calibration thresholds\" in an OpenMed de-id context."
license: Apache-2.0
metadata:
  project: OpenMed
  category: evaluation-quality
  pairs: adjacent
  version: "1.0"
---

# Evaluating with Leakage Gates

OpenMed's release gates answer one question: **did any PHI leak?** A de-id model
with a beautiful F1 can still leak a single SSN — and that one leak is a HIPAA
breach. So `openmed.eval` gates on *residual leakage* and *per-label recall
floors*, not on aggregate F1. The candidate is either `RELEASABLE` or
`QUARANTINED`; there is no partial credit.

## When to use this skill

- You have a candidate de-id or PII model and need a ship / no-ship decision.
- You want to run the benchmark harness over a **synthetic** golden suite.
- You need to enforce direct-identifier recall floors and `critical_leakage == 0`.
- You need calibration thresholds (`thresholds.json`) before the gate will pass.
- You want a signed, reproducible gate report for governance.

This is the flagship eval skill. For a pure NER scorecard see
`benchmarking-clinical-ner`; for CI wiring see `gating-deid-leakage`.

## The gates (G1a–G8)

| Gate | Checks | Floor / rule |
| --- | --- | --- |
| **G1a** | Direct & quasi identifiers (PERSON, EMAIL, PHONE, SSN, ID_NUM, DATE_OF_BIRTH, ...) | recall ≥ 0.990 (v1.6) / 0.995 (v2.0); strict-no-leak policies raise the floor |
| **G1b** | Structured secrets (API_KEY, ACCOUNT_NUMBER, CREDIT_CARD, IBAN) | recall ≥ 0.995 |
| **G2** | Free-text names/locations/dates | recall ≥ 0.980 (v1.6) / 0.990 (v2.0) |
| **G3** | Critical leakage (SSN, CREDIT_CARD, CVV, API_KEY, PIN, IBAN, ...) | count **must be exactly 0** |
| **G4** | Quantized recall delta vs fp parent | within INT8 / INT4 limits |
| **G5** | Latency & RAM vs device tier budget | p50/p95/RAM under tier budget |
| **G6** | p50/p95 latency documented | must be present and finite |
| **G7** | Baseline regression | recall drop ≤ 0.002/label; leakage ≤ soft ceiling 0.005 and ≤ steward target; no leakage regression vs last-green |
| **G8** | Span integrity | predicted spans validate (no overlaps/out-of-range) |

Constants live in `openmed.eval.release_gates` (`G1A_V16_RECALL_FLOOR`,
`G1B_RECALL_FLOOR`, `G7_RECALL_DROP_LIMIT`, `RESIDUAL_LEAKAGE_SOFT_CEILING`, ...).
Confirm them there rather than hardcoding — they move per milestone.

## Quick start

Run a candidate benchmark over a synthetic golden suite, then gate it:

```python
from openmed.eval import run_suite, ReleaseGate, RELEASABLE

# 1) Produce a candidate BenchmarkReport from a SYNTHETIC fixtures file.
#    Each fixture carries gold PHI spans; no real patient text is committed.
report = run_suite(
    "eval/golden/phi_synthetic.json",     # user-supplied synthetic fixtures
    suite="golden",
    model_name="OpenMed/Privacy-PII-Detection",
    device="cpu",
    metadata={
        "family": "PII",
        "tier": "base",
        "policy": "hipaa_safe_harbor",
        # calibration artifacts are required for mask/replace policies (see below)
        "thresholds_path": "eval/artifacts/thresholds.json",
        "calibration_report_path": "eval/artifacts/calibration_report.json",
    },
)

# 2) Gate it. The gate reads the last-green baseline store read-only and
#    returns a signed GateReport.
gate = ReleaseGate(milestone="v1.6", policy="hipaa_safe_harbor")
decision = gate.evaluate(report)

print(decision.decision)                  # "RELEASABLE" or "QUARANTINED"
for check in decision.gate_results:
    if not check.passed:
        print(check.gate, "->", check.reason, check.details)

assert decision.decision == RELEASABLE, "do not ship a quarantined model"
```

CLI equivalent (fails closed, exit code 1 on quarantine):

```bash
python -m openmed.eval.release_gates \
  --candidate eval/out/candidate_report.json \
  --milestone v1.6 --policy hipaa_safe_harbor \
  --output release-gate-report.json
```

## Workflow

1. **Build a synthetic golden suite.** Fixtures are JSON with `text` and
   `gold_spans` (offsets + labels). Use `building-gold-corpus` to scaffold one.
   Committed gold must be synthetic; DUA corpora (i2b2/n2c2) are eval-only and
   never committed.
2. **Fit calibration thresholds** for any policy that masks or replaces:

   ```python
   from openmed.eval import write_calibration_artifacts

   paths = write_calibration_artifacts(
       calibration_samples,                 # held-out score/target samples
       artifact_dir="eval/artifacts",
       model_id="OpenMed/Privacy-PII-Detection",
       suite="golden",
       target_leakage=0.0,                   # leakage-first: drive leakage to 0
   )
   # writes thresholds.json + calibration_report.json the gate looks for
   ```

   The gate's `calibration_present` check fails the build if these are missing
   for a mask/replace policy.
3. **Run the suite** (`run_suite` / `run_benchmark`) to get a `BenchmarkReport`.
4. **Evaluate** with `ReleaseGate(...).evaluate(report)`.
5. **Read the per-gate results.** Each `GateCheck` carries `gate`, `passed`,
   `reason`, and `details` (e.g. which labels fell below the recall floor).
6. **Fail closed.** Treat anything other than `RELEASABLE` as a hard stop.
7. **Audit subgroups** with `fairness_report` (see `auditing-subgroup-fairness`)
   so an aggregate pass doesn't hide an under-protected group.

## Hand-off to / from OpenMed

- **From** `building-with-openmed` and the de-id pipeline: you evaluate the model
  produced by `openmed.deidentify` / `openmed.extract_pii`.
- **To** `gating-deid-leakage`: wrap `ReleaseGate.evaluate(...)` in a pytest/CLI
  gate so CI fails closed on regression.
- **To** `authoring-model-cards`: feed `GateReport`, `fairness_report`, and
  `error_report` outputs into the model card's metrics and limitations sections.
- **Pairs with** `auditing-subgroup-fairness` (`fairness_report`) and
  `benchmarking-clinical-ner` (`error_report`).

## Edge cases & gotchas

- **F1 is not a gate.** A model can have higher F1 and still be quarantined if it
  leaks one critical identifier (G3) or drops a label below its floor (G1a/G1b).
- **Calibration is mandatory for mask/replace policies.** No `thresholds.json` →
  `calibration_present` fails → `QUARANTINED`.
- **Baselines are read, never written, by the gate.** The gate compares against
  the last-green baseline store without mutating it (G7). Promote baselines in a
  separate, deliberate step.
- **Strict-no-leak policies raise the G1a floor** and force the leakage target to
  0. Don't assume the default floor.
- **Reports must carry identity metadata** (`family`, `tier`, `format`,
  `eval_set_hash`, `leakage_fixture_hash`); `manifest_coherence` fails without it.
- **Reports are signed** (HMAC-SHA256). Set `OPENMED_RELEASE_GATE_KEY` for a real
  signing key; `GateReport.verify(key)` checks the repro hash and signature.
- **No raw PHI in the report.** Gate evidence is offsets, hashes, and labels —
  never plaintext identifiers. Keep it that way in any wrapper you write.

## Standards & references

- HIPAA Safe Harbor / Expert Determination (45 CFR 164.514):
  https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/
- NIST SP 800-188, *De-Identification of Personal Information*:
  https://csrc.nist.gov/pubs/sp/800/188/final
- i2b2 2014 de-identification shared task (recall-first evaluation tradition):
  https://doi.org/10.1016/j.jbi.2015.06.007
- OpenMed eval source of truth: `openmed/eval/release_gates.py`,
  `openmed/eval/harness.py`, `openmed/eval/calibrate.py`.
