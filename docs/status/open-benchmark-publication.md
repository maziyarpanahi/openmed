# Open Benchmark Publication Plan

This plan defines the open, reproducible publication surface for the trust
status epic. It is not a closed leaderboard: every published number must trace
back to committed result JSON, a reproducibility hash, and the renderer that
turned those inputs into a page.

## Current Public Artifacts

| Artifact | Role |
|---|---|
| `models.jsonl` | Canonical model manifest and release metadata. |
| `gates/baseline.json` | Last-green baseline store per family, tier, and format. |
| `docs/benchmarks/golden.report.json` | Committed benchmark report evidence. |
| `docs/benchmarks/golden.md` | Human-readable benchmark card generated from the report. |
| `docs/status/index.md` | Trust status page generated from manifest, baseline, and reports. |
| `docs/leaderboard/index.md` | Open benchmark publication table generated from the same sources. |
| `scripts/status/generate_status.py` | Renderer that keeps status, benchmark cards, and publication rows aligned. |

## Publication Principles

- Source data is committed and inspectable.
- Result pages are generated from JSON inputs, not hand-edited claims.
- Rows represent reproducible benchmark evidence rather than private rankings.
- Competitor or baseline rows without source reports must say that evidence is
  unavailable rather than implying a measured result.
- No raw clinical text, private corpus rows, or unreleasable identifiers are
  committed.

## Result Row Contract

Open publication rows reuse the status contract in
`docs/status/trust-status-contract.md`.
Each published row must include or derive:

| Field | Meaning |
|---|---|
| `family` | Model family. |
| `tier` | Model tier, or `null`. |
| `device` | Runtime used by the harness. |
| `format` | Published artifact format. |
| `current_leakage` | Decimal leakage rate from `BenchmarkReport.metrics`. |
| `last_green_release` | Last passing release date from `gates/baseline.json`. |
| `last_regression` | Most recent known regression date, if any. |
| `last_rollback` | Rollback date for the latest known regression, if any. |
| `harness_freshness` | `BenchmarkReport.generated_at` for the newest successful run. |
| `evidence` | Report suite or committed artifact path. |
| `reproducibility_hash` | `sha256:<64 lower hex>` hash for the recipe, data manifest, base model, and `git_sha`. |

## Reproducibility Hash Convention

Use `openmed.core.repro_hash.compute_reproducibility_hash` and store the
result as `sha256:<64 lower hex>`. The canonical payload has exactly these
top-level inputs:

```json
{
  "base_model": "...",
  "data_manifest": {},
  "git_sha": "abc123",
  "recipe": {}
}
```

The function sorts mapping keys, preserves list order, sorts sets by
representation, hashes bytes, hashes file paths by content, and hashes
directories by their sorted file list. This makes equivalent recipe and data
manifest dictionaries produce the same hash even when keys are written in a
different order.

## Publish And Refresh Flow

1. Produce or refresh one or more `BenchmarkReport` JSON files.
2. Update `models.jsonl` and `gates/baseline.json` only through the release
   helpers that validate schema and hash format.
3. Regenerate the public artifacts:

   ```bash
   .venv/bin/python scripts/status/generate_status.py \
     --report docs/benchmarks/golden.report.json
   ```

4. Review the generated diff for source paths, hashes, leakage, last-green
   dates, regression and rollback metadata, and freshness.
5. Run the repository test suite:

   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```

6. Publish only the committed JSON and generated pages. Do not publish manually
   edited result rows.

## Scheduled Trust Refresh

GitHub Pages is the public hosting surface. `.github/workflows/pages.yml` runs
at 02:17 UTC each day, and can also be rerun through its manual dispatch. The
scheduled job runs the full offline test suite, builds a fresh synthetic harness
control report, then calls the same `scripts/status/generate_status.py` renderer
used locally. The generated report JSON, status page, and leaderboard are staged
in the Pages artifact and deployed together. Push deployments refresh the
control report as well, after the documentation checks pass.
Browser validation and its 14-day evidence upload remain on PR and push runs;
the nightly status refresh skips those steps so it creates no new browser
evidence storage.

The control is a deterministic detector that emits no spans over committed
Apache-2.0 synthetic golden fixtures. It verifies harness execution and
publication freshness. Its 100% leakage is the expected negative-control result and is not
a model-performance number. The report records fixture hash, source rights,
model and configuration revision, source commit, reproducibility hash, and
limitations. Model benchmark rows keep their own original timestamps.

The renderer rejects a missing, malformed, future-dated, or older-than-36-hour
control report. Those errors fail the scheduled job before deployment; the
previous successful Pages deployment remains public. The workflow log names the
failed command and the input error.

To rerun, dispatch **Deploy Docs to GitHub Pages** against `master`, or inspect
the output locally with the current commit SHA:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/status/generate_nightly_control.py \
  --source-revision "$(git rev-parse HEAD)" \
  --output docs/status/evidence/nightly-control.json
.venv/bin/python scripts/status/generate_status.py \
  --report docs/benchmarks/golden.report.json \
  --nightly-report docs/status/evidence/nightly-control.json
```

For rollback, revert the offending source or input commit through a normal PR
to `master`; the Pages push workflow redeploys the corrected tree. A release
pointer regression uses the separate `release-gates.yml` rollback job. Do not
edit the generated page by hand or publish a failed nightly report.

## Child Issue Handoff

The epic has been decomposed into independently mergeable slices:

- #374 defines the nightly trust-results schema and publication contract.
- #375 renders the trust status page from nightly results.
- #376 selects the public status hosting surface.
- #377 adds the nightly trust status refresh.
- #378 publishes the first open SHIELD baseline results.

The current repository already publishes the committed golden baseline through
`docs/benchmarks/golden.report.json`, `docs/benchmarks/golden.md`,
`docs/status/index.md`, and `docs/leaderboard/index.md`. #378 is responsible
for adding the first SHIELD baseline numbers once the scheduled status path and
hosting decision are in place.

## Review Checklist

Before a publication update is accepted:

- every new result has committed JSON evidence;
- every measured row has `harness_freshness`;
- every release or benchmark row has a valid `reproducibility_hash`;
- every hash input is documented enough for another maintainer to recompute it;
- generated pages are reproducible from the committed inputs;
- the update does not convert the open results table into a closed or gated
  leaderboard.
