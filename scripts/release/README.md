# Release budget checklist

OpenMed keeps package-size and core-import budgets in
`gates/release_budgets.json`. The build job enforces these values on
`ubuntu-latest` after creating the distribution artifacts.

## Before a release

1. Build the wheel into an empty `dist/` directory:

   ```bash
   uv run --with build python -m build
   ```

2. Enforce the wheel budget and record installed footprints for the core,
   Chinese, and Indic profiles:

   ```bash
   python scripts/release/check_size_budget.py \
     --skip-build \
     --wheel-dir dist \
     --report size-budget-report.json
   ```

   The script can also build the wheel itself when `build` is available in the
   selected Python environment. Omit `--skip-build` and point `--wheel-dir` at
   an empty directory.

3. Install the built core wheel, then enforce the fresh-import budget:

   ```bash
   uv pip install --system dist/*.whl
   python scripts/release/check_import_budget.py
   ```

The committed wheel baseline is 7,848,985 bytes. Its maximum is 8,633,884
bytes, which provides 10% headroom. A fresh `import openmed` must remain at or
below 300,000 cumulative microseconds on `ubuntu-latest`, and it must not load
`jieba`, `opencc`, `pypinyin`, or `indicnlp`.

The baseline records `master` commit
`ac3b2df82fa547a395d6b24722e556d32b2b6c2e`. The
[Ubuntu CI build](https://github.com/maziyarpanahi/openmed/actions/runs/40999415395/job/123886537270)
and a clean archived macOS rebuild both measured 7,848,985 bytes. The local
wheel has SHA-256
`c6aac5686e941ebcd5ea6c563c807e9f9bcc27473dfe8e1d8e87efa61c38b166`.
The refresh PR's build job independently rebuilds the frozen base on
`ubuntu-latest` and requires its measured size to equal the proposed baseline;
the local reproduction is not a substitute for that hosted check.

The prior baseline, `a88fe2feb699fd1945d053859998e54e00237825`, rebuilds with
its documented Hatchling 1.32.4 at 7,036,059 bytes and 1,495 wheel members.
Current master has 1,637 members and 812,926 bytes of accepted growth,
principally synthetic generators, FHIR adapters, anonymization and language
bundles, clinical exporters, agent workflows, schemas and evaluation code.
Every packaged source/resource payload matches its frozen Git revision,
including the explicitly remapped adversarial corpus. The remaining members
are generated distribution metadata. There are no bundled model weights or
build-cache files. This refresh changes no wheel payload, compression, 10%
headroom policy or import limit.

The JSON size report records the wheel size plus total site-packages bytes for
`openmed`, `openmed[zh]`, and `openmed[indic]`. Each language profile includes
its byte delta from the core installation. CI uploads this report beside the
wheel and source distribution in the `dist-packages` artifact.

## Bumping a budget

Budget changes require a reviewed edit to `gates/release_budgets.json`; CI does
not accept environment-variable overrides.

1. Rebuild the wheel from current `master` on `ubuntu-latest`.
2. Set `baseline_bytes` to the measured wheel size.
3. Set `maximum_bytes` to `ceil(baseline_bytes * 1.10)`, retaining
   `headroom_percent: 10`.
4. For an import-budget change, update
   `maximum_cumulative_microseconds` and explain the regression or intentional
   startup work in the pull request.
5. Run both checks above and include the resulting measurements in the review.

For a pull request that changes the budget file, CI independently archives and
builds the event's frozen base commit. A different measured wheel size fails
the build, and CI retains the baseline artifacts. Re-measure and update the
baseline if `master` has changed; do not use an environment override or disable
the comparison.

Do not raise a budget merely to make an unexplained regression pass.

## V3 Journey release packet

The v3 Journey release decision aggregates ten typed evidence lanes, verifies
the tagged checkout and frozen input digests, checks artifact freshness and
license boundaries, and writes a signed value-free packet. Before running it,
set `OPENMED_JOURNEY_RELEASE_KEY` from a secret manager to a secret of at least
32 bytes; do not put the secret in a command, shell history, or repository file.

```bash
python scripts/release/journey_release_gate.py \
  --manifest journey-release-manifest.json \
  --output journey-release-packet.json
```

The command returns `0` only for `READY`, `1` for a signed `NOT_READY` packet,
and `2` when no safe decision can be produced. It does not publish, promote,
download, or mutate release inputs. See
`docs/release/v3.0-journey-release-gate.md` for the complete manifest,
performance, licensing, exception, signing, and verification contracts.

## Retraining recipe proposals

`retrain_queue.py` consumes the committed aggregate-only input contract at
`gates/retrain_trigger_inputs.json`. It writes the complete decision evidence
and queued JSONL records separately, then updates `recipes/<family>.yaml` only
for families whose weighted score reaches the configured threshold:

```bash
python scripts/release/retrain_queue.py \
  --queue-output artifacts/retrain-trigger/retrain_queue.jsonl \
  --evidence-output artifacts/retrain-trigger/decision_evidence.json \
  --summary-output artifacts/retrain-trigger/dispatch_summary.json
```

The scheduled workflow uploads queue and decision evidence as workflow
artifacts. It opens a configuration-only pull request when a recipe actually
changes. The workflow never trains, converts, or publishes model artifacts;
the normal downstream release gates remain mandatory before promotion.

## v3.1 agent candidate decision

`python scripts/release/agent_release_gate.py` shares the exact-source runner
with `openmed gates agent-release`. Supply a full source SHA, local wheel/sdist,
tool catalog, policy, source-bound aggregate evidence, private signing-key file
and a new output path outside the checkout. See the [candidate input and
verification contract](../../docs/evaluation/v3.1-agent-gates.md#exact-source-candidate-runner).
The command never builds, tags, publishes or creates a release; its exit codes
are 0 for READY, 1 for signed NOT_READY, and 2 for configuration/output errors.
