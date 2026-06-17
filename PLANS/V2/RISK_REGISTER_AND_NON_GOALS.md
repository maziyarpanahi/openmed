# OpenMed V2 Risk Register and Non-Goals

This register turns roadmap sections 8.5 through 8.8 into operational checks for
planning, release review, and pull request scope.

## Hard Invariants

These are never traded for cadence:

- Critical leakage is zero on the curated golden fixtures and critical-identifier
  suite.
- The direct-identifier recall floor is enforced as a release gate.
- No quantized model is published if its recall delta exceeds the gate.
- No raw PHI is logged, cached, or telemetered.
- The default runtime remains local-first after model weights are present.
- The core package and required dependencies remain MIT, Apache-2.0, or BSD
  compatible.

## Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Release machinery is incomplete | Critical | Track build-vs-existing gaps explicitly; do not claim daily release capability until scheduler, publish, manifest, and gate automation exist |
| Daily-release regression | High | Run tests, benchmark non-regression, manifest diff checks, and stable-channel promotion gates |
| PHI leakage | Critical | Enforce zero critical leakage, deterministic safety sweep, no-raw-PHI logging guard, audit provenance, and residual-risk reporting |
| Quantization recall loss | High | Gate INT8 and INT4 artifacts against their full-precision parent before publication |
| DUA evaluation availability | Medium | Keep daily gates public/synthetic; use credentialed DUA evaluation as periodic stable-promotion evidence |
| Dataset redistribution mistake | High | Public/DUA/synthetic tiering, `NOTICE` review, manifest license rows, and no bundled restricted corpora |
| Over-redaction | Medium | Track precision and over-redaction ceilings; use policy profiles to tune utility versus recall |
| Scope creep | High | Prioritize product-core, gate, and manifest work before new model count growth |
| Stale external facts | Medium | Date-stamp and cite external claims before using them in release or benchmark material |
| Naming drift | Medium | Use the committed manifest as the source of truth for family, tier, parameter count, and generated surfaces |

## Non-Goals

OpenMed will not:

1. Send raw PHI to remote generative services for redaction or extraction.
2. Ship generative-only de-identification.
3. Run a closed benchmark or private-only leaderboard.
4. Add mandatory network calls, default telemetry, or a license server to the
   core runtime.
5. Bundle DUA-gated data, restricted vocabularies, or non-permissive components.
6. Become dependent on wrapped libraries in the default path; adapters remain
   optional interop.

## Review Checklist

Before a task that touches data, releases, adapters, or model policy is merged,
reviewers should confirm:

- data sources are assigned to the correct tier;
- new dependencies pass the license policy gate;
- ported rule files include upstream attribution headers;
- any benchmark or model-card claim has a source and date;
- raw PHI is absent from logs, fixtures, generated reports, and docs examples;
- the change does not weaken a hard invariant.
