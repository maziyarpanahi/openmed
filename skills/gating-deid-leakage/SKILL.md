---
name: gating-deid-leakage
description: "Add a CI gate that fails the build when an OpenMed de-identification model's recall on a held-out PHI set drops below threshold or any critical identifier leaks. Use when the user wants a pytest test or CLI step that exits nonzero on de-id regression, wants to wire OpenMed's leakage-first release gates into GitHub Actions / CI, needs a recall floor plus zero-leakage assertion against a synthetic held-out set, or wants to block merges that weaken de-identification. Trigger on \"CI gate\", \"fail the build\", \"regression test\", \"de-id recall threshold\", \"block the merge\", \"exit nonzero\", or \"leakage check in CI\" for OpenMed."
license: Apache-2.0
metadata:
  project: OpenMed
  category: evaluation-quality
  pairs: adjacent
  version: "1.0"
---

# Gating De-id Leakage in CI

Logs, baselines, and models drift. The only durable defense is a gate that runs
on every change and **fails closed** when de-identification regresses. This skill
operationalizes OpenMed's leakage-first ethos into a CI check: recall must stay
above the floor and **critical leakage must be exactly zero**, or the build goes
red.

## When to use this skill

- You want a pytest test or CLI step that exits nonzero on de-id regression.
- You need to block PRs that drop PHI recall or introduce a leak.
- You want OpenMed's release gates (`ReleaseGate`, G1a–G8) enforced in CI.
- You maintain a synthetic held-out PHI set and want it checked automatically.

For the full gate semantics see `evaluating-with-leakage-gates`; this skill is
about *wiring it into CI so it fails the build*.

## Quick start — a pytest gate

```python
# tests/eval/test_deid_leakage_gate.py
import pytest
from openmed.eval import run_suite, ReleaseGate, RELEASABLE

RECALL_FLOOR = 0.99          # direct-identifier recall floor
HELD_OUT = "eval/heldout/phi_synthetic.json"   # SYNTHETIC, committed

@pytest.fixture(scope="module")
def gate_report():
    report = run_suite(
        HELD_OUT,
        suite="golden",
        model_name="OpenMed/Privacy-PII-Detection",
        device="cpu",
        metadata={"family": "PII", "tier": "base", "policy": "hipaa_safe_harbor"},
    )
    return ReleaseGate(milestone="v1.6", policy="hipaa_safe_harbor").evaluate(report)

def test_no_critical_leakage(gate_report):
    # Hard zero: one leaked SSN/credit-card is a breach, full stop.
    assert gate_report.critical_leakage_count == 0, "critical PHI leaked"

def test_recall_floor(gate_report):
    low = {
        label: r
        for label, r in gate_report.per_label_recall.items()
        if r < RECALL_FLOOR
    }
    assert not low, f"recall below floor: {low}"

def test_releasable(gate_report):
    # The structural decision: any failed gate -> QUARANTINED -> red build.
    failed = [c.gate for c in gate_report.gate_results if not c.passed]
    assert gate_report.decision == RELEASABLE, f"quarantined; failed gates: {failed}"
```

`pytest` exits nonzero on any failure, so CI turns red automatically.

## Quick start — a CLI gate

The harness ships a `main()` that **fails closed** (exit 1 on quarantine):

```bash
# Produce a candidate report, then gate it. Nonzero exit blocks the job.
python -m openmed.eval.release_gates \
  --candidate eval/out/candidate_report.json \
  --baseline-store eval/baselines/last_green.json \
  --milestone v1.6 --policy hipaa_safe_harbor \
  --output release-gate-report.json
```

Exit codes: `0` RELEASABLE, `1` QUARANTINED, `2` evaluation error before a
report. CI should treat `1` and `2` as failures.

## Wire it into GitHub Actions

```yaml
# .github/workflows/deid-gate.yml
name: de-id leakage gate
on: [pull_request]
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv pip install --system -e ".[hf]"
      - name: Run de-id leakage gate
        run: uv run pytest tests/eval/test_deid_leakage_gate.py -q
      # The job fails (red) automatically if pytest exits nonzero.
```

## Workflow

1. **Curate a synthetic held-out PHI set** and commit it (offsets + labels, no
   real patient text). Use `building-gold-corpus`. It must be disjoint from any
   calibration/training data.
2. **Pick the floor from policy.** Don't hardcode a magic number you invented —
   align to the gate's published floors (`G1A_V16_RECALL_FLOOR` etc. in
   `openmed.eval.release_gates`) and the active policy profile.
3. **Run the suite → gate the report** (pytest fixture above).
4. **Assert two invariants**: `critical_leakage_count == 0` and per-label recall
   ≥ floor. Optionally assert `decision == RELEASABLE` for the full G1a–G8 check.
5. **Make it required.** Mark the job a required status check so a red gate
   blocks merge — a passing-but-not-required gate protects nothing.
6. **On failure, the harness CLI can open/refresh a tracking issue**
   (`--issue-on-failure`) so the regression is visible, not silently retried.

## Hand-off to / from OpenMed

- **From** `evaluating-with-leakage-gates`: this skill is the CI wrapper around
  the same `ReleaseGate.evaluate(...)` call — reuse its gate semantics.
- **From** `building-gold-corpus`: supplies the synthetic held-out fixtures the
  gate runs against.
- **To** `authoring-model-cards`: a green gate's `GateReport` is the evidence the
  model card cites under "evaluation".
- **Pairs with** `enforcing-nophi-logging`: the gate proves the *model* doesn't
  leak; the logging guard proves your *runtime* doesn't leak.

## Edge cases & gotchas

- **Fail closed, never open.** If the candidate report is missing or the eval
  errors, treat it as a failure. Don't `|| true` the step.
- **A passing F1 is not a passing gate.** Recall floor + zero critical leakage are
  the load-bearing assertions; assert them explicitly even if you also check
  `decision`.
- **Held-out must stay held-out.** If the gate set leaks into calibration or
  training, the gate measures memorization, not generalization. Keep splits
  disjoint (see `building-gold-corpus`).
- **Pin the model and milestone.** Floors change per milestone (v1.6 vs v2.0);
  pin both so a "passing" gate doesn't silently weaken.
- **No real PHI in CI artifacts or logs.** The gate report is offsets/hashes
  only; don't print fixture text in CI output.
- **Baselines are inputs, not outputs of the gate.** Promote last-green baselines
  in a separate, reviewed step — never auto-write them from the gate job.

## Standards & references

- HIPAA Safe Harbor de-identification standard (45 CFR 164.514):
  https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/
- NIST SP 800-188, *De-Identification of Personal Information*:
  https://csrc.nist.gov/pubs/sp/800/188/final
- GitHub Actions required status checks:
  https://docs.github.com/actions/using-workflows/required-status-checks
- OpenMed eval source of truth: `openmed/eval/release_gates.py` (`main`,
  `ReleaseGate`), `openmed/eval/harness.py`.
