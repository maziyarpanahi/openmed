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

The committed wheel baseline is 7,036,059 bytes. Its maximum is 7,739,665
bytes, which provides 10% headroom. A fresh `import openmed` must remain at or
below 300,000 cumulative microseconds on `ubuntu-latest`, and it must not load
`jieba`, `opencc`, `pypinyin`, or `indicnlp`.

The baseline records `master` commit
`a88fe2feb699fd1945d053859998e54e00237825`. An Ubuntu 24.04 x86-64 rebuild
with uv 0.11.28, Python 3.11.15 and Hatchling 1.32.4 produced a 7,036,059-byte
wheel with SHA-256
`670169af4a50f40db81b05eab20272fe3f1946dbdf95dbe25fb80c1ad09bd5e4`,
identical to the macOS rebuild. The PR build job also rebuilds the frozen base
on `ubuntu-latest` and requires its measured size to equal the proposed baseline
before accepting a budget refresh; the local reproduction is not a substitute
for that hosted check.

The previous 6,319,377-byte baseline predates already-merged clinical brief,
SDOH, local NLI, language, evaluation and agent capabilities. Unchanged master
exceeded the old 6,951,315-byte maximum by 84,744 bytes. The installed-SDK
candidate at `88b12440f2a05a2f982e1ff98f364db88dc117e2` added only 238 bytes,
measured at 7,036,297 bytes in
[CI run 37135994309](https://github.com/maziyarpanahi/openmed/actions/runs/37135994309).
Both wheels have the same 1,495 members, with every source/resource payload
matching its Git revision. This refresh records accepted source growth; it
does not change package contents, compression, 10% headroom or import limits.

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
