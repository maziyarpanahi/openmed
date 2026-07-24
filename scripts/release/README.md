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

The committed wheel baseline is 2,331,240 bytes. Its maximum is 2,564,364
bytes, which provides 10% headroom. A fresh `import openmed` must remain at or
below 300,000 cumulative microseconds on `ubuntu-latest`, and it must not load
`jieba`, `opencc`, `pypinyin`, or `indicnlp`.

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

Do not raise a budget merely to make an unexplained regression pass.
