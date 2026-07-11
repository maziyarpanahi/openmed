# Contributing to OpenMed

Thanks for helping improve OpenMed — bug reports, features, docs, and
translations are all welcome. By participating you agree to our
[Code of Conduct](CODE_OF_CONDUCT.md).

> [!IMPORTANT]
> **Security and privacy come first.** OpenMed de-identifies PHI. **Never** open
> a public issue for a vulnerability, and **never** include real patient data in
> an issue, pull request, test, or fixture — see [SECURITY.md](SECURITY.md). Any
> test data committed to the repository must be synthetic.

This page is a quick start. The full tooling, documentation, and release guide
lives in **[docs/contributing.md](docs/contributing.md)**.

## Development setup

```bash
uv pip install -e ".[dev]"      # editable install with test + lint deps
# stack extras as needed, e.g. ".[dev,hf]" for Hugging Face model work
pre-commit install              # auto-format staged files on commit
```

The pre-commit hooks also run the Gitleaks secret scanner on staged changes.
See [Secret Scanning](docs/security/secret-scanning.md) for local use, CI
behavior, and baseline maintenance.

## Run the checks before opening a PR

```bash
.venv/bin/python -m pytest tests/ -q     # run the test suite
make format                              # apply canonical Ruff formatting
make lint                                # lint
make format-check                        # CI-parity format check
```

Ruff is the single source of truth for Python lint, import ordering, and
formatting — do not run Black, isort, or flake8. For Swift changes under
`swift/OpenMedKit`, run `make format-swift` and `make lint-swift`.

## Pull requests

- Keep each PR focused on a single feature or fix; avoid unrelated formatting
  churn.
- Write clear, imperative commit messages (the project uses concise prefixes
  such as `fix:`, `feat:`, and `docs:`).
- Reference the issue you are closing and complete the
  [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
- Update `CHANGELOG.md` and the docs when behavior changes.
- For privacy-sensitive paths, add tests for direct-identifier recall, critical
  leakage, and span integrity (see [docs/contributing.md](docs/contributing.md)).

## Reporting issues

- **Bugs and features:** use the
  [issue templates](https://github.com/maziyarpanahi/openmed/issues/new/choose).
- **Security vulnerabilities** (including redaction bypass or PHI/PII leakage):
  report privately per [SECURITY.md](SECURITY.md) — never as a public issue.

For the complete contributor and release workflow, see
**[docs/contributing.md](docs/contributing.md)**.
