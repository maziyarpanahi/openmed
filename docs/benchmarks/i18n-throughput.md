# Chinese and Indic throughput gates

The release-candidate benchmark measures two steady-state paths over committed,
fully synthetic Chinese (`zh`), Hindi (`hi`), and Tamil (`ta`) corpora:

- segmentation throughput in characters per second;
- end-to-end `deidentify()` throughput in redacted spans per second.

Each corpus contains at least 100,000 characters generated deterministically
with Faker, seeded per language and using the fixed synthetic visit-date window
from 2021-07-23 through 2026-07-23. The JSON report also records first-call
latency on the first newline-aligned chunk for each operation, including
Chinese dictionary initialization, separately from steady-state full-corpus
throughput.
De-identification processes every record in bounded chunks to avoid making
sentence segmentation quadratic in corpus size. It uses deterministic regex
and pattern detectors with an empty local model adapter, so no model weights or
network access are required.

## Run locally

Install the development and language extras, then run:

```bash
uv sync --extra dev --extra zh --extra indic
.venv/bin/python -m openmed.eval.i18n_throughput \
  --output i18n-throughput-report.json
.venv/bin/python -m openmed.eval.release_gates \
  --throughput-candidate i18n-throughput-report.json \
  --baseline-store gates/baseline.json \
  --output i18n-throughput-gate.json
```

The complete benchmark has a five-minute limit. Its reports contain only
aggregate metrics, corpus sizes, and hashes; they do not copy fixture text or
detected surfaces.

## Refresh the baseline

The committed entries in `gates/baseline.json` use family
`i18n-throughput`, tiers `zh`, `hi`, and `ta`, and format `pattern-only`.
Each entry records the two steady-state metrics and a regression threshold of
`0.2`. A candidate fails when either metric is more than 20% below its
language baseline.

Baselines never update automatically. The release workflow runs on GitHub's
hosted Ubuntu x86_64 runner, so its committed baseline must come from that same
runner class rather than a developer workstation. The current values are the
median of six consecutive hosted runs; the run identifiers and aggregate
calibration hash are recorded with every entry. The calibration-set digest is
the SHA-256 of a compact, key-sorted JSON list of `{run_id, report}` objects,
ordered by run identifier, so reviewers can reproduce it from the named
workflow artifacts.

Refresh them only through a reviewed pull request:

1. Collect at least six completed benchmark artifacts from the designated
   GitHub-hosted Ubuntu x86_64 workflow.
2. Investigate bimodal or unstable results, then use the median of each
   steady-state metric rather than a single fast runner.
3. Update only the six steady-state metrics plus their measurement metadata,
   source run identifiers, sample count, and aggregate reproducibility hash in
   `gates/baseline.json`.
4. Review the JSON diff together with the benchmark report. Do not accept an
   unexplained lower baseline merely to make the gate pass.
5. Run the throughput gate and the full test suite before merging the baseline
   change.

Corpus regeneration is similarly review-only:

```bash
.venv/bin/python scripts/benchmarks/generate_i18n_throughput_fixtures.py
```

Review the fixture metadata, hashes, sizes, and synthetic-only tests before
committing regenerated corpora.
