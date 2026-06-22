# Compliance Posture

OpenMed provides evidence for local validation: audit reports, leakage metrics,
residual-risk scores, reproducible benchmark runs, and policy-profile outputs.
OpenMed does not self-certify compliance. Users validate against their own data,
jurisdiction, deployment controls, and counsel.

No de-identification tool can guarantee compliance or zero residual risk. Validate locally before any production or clinical use.

## Framework Split

| Framework | What OpenMed supplies | What the user still owns |
|---|---|---|
| **HIPAA Safe Harbor** (`policy="hipaa_safe_harbor"`) | Profile mapping `CANONICAL_LABELS` to the 18 identifier classes, many-to-one; per-class leakage metrics; audit trail. | Confirming all 18 classes for their data; the "no actual knowledge" judgment. |
| **HIPAA Expert Determination** (`policy="hipaa_expert_review_assist"`) | Reproducible leakage/residual-risk and adversarial re-id results as documentation. | Engaging a qualified expert; the determination. |
| **GDPR pseudonymization** (`policy="gdpr_pseudonymization"`) | Reversible `replace` with separable key mapping; deterministic surrogates; transform audit. | Key custody, re-id governance, lawful basis, DPIA. |
| **EU AI Act high-risk** | Logging through audit reports, documentation through cards, human oversight through review bundles, robustness through harness and golden suites, and cybersecurity through signing, local-only execution, and no raw logging. | Conformity assessment, registration, and deployment obligations. |

## Evidence Links

- [Audit reports](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/audit.py)
  provide deterministic span provenance, residual-risk snapshots, manifest
  hashes, and optional signatures for de-identification runs.
- [Leakage metrics](./eval-harness.md#metric-bundle) are emitted in
  `BenchmarkReport` output for reproducible benchmark runs.
- [Residual re-identification risk](https://github.com/maziyarpanahi/openmed/blob/master/openmed/risk/reid.py)
  reports leakage and auxiliary linkage risk for text or table records.
- [Policy profiles](https://github.com/maziyarpanahi/openmed/tree/master/openmed/core/policies)
  capture the canonical policy literals and default action posture used by the
  runtime.

## Policy Literals

Use these exact literals in docs, examples, and deployment configuration:

| Literal | Posture | Default action bias | Arbitration mode |
|---|---|---|---|
| `hipaa_safe_harbor` | Redact all 18 HIPAA identifiers, no exceptions. | `remove`/`mask` | High-recall union; safety sweep mandatory; residual-risk ~0. |
| `hipaa_expert_review_assist` | Flag + surrogate, leave clinical content; produce expert-review audit. | `replace` surrogates | Balanced, with reviewer escalation on low confidence. |
| `gdpr_pseudonymization` | Reversible pseudonyms, mapping retained under key. | `replace` + `reversible_id` | Balanced; `keep_mapping=True`; HMAC reversible IDs. |
