# Contributing & Releases

Short feedback cycles keep OpenMed shippable. This page captures the tooling you need to edit docs, cut releases, and
publish the package to PyPI or GitHub Pages.

## Local workflows

- `make help` prints a list of scripted tasks (build, publish, release, docs, etc.).
- `make format` applies the canonical Ruff import ordering and formatting.
- `make lint` and `make format-check` run the same style gates used by CI.
- `make type-check` runs the same scoped, pinned mypy gate used by CI.
- `make format-swift` and `make lint-swift` apply the canonical Swift format checks for OpenMedKit.
- `pre-commit install` enables the local hooks that auto-format staged Python files before commit.
- `make docs-serve` starts the MkDocs preview with hot reload at `http://127.0.0.1:8008`.
- `make docs-build` runs `mkdocs build --strict` for CI parity.
- `uv pip install ".[dev]"` pulls in pytest + coverage; `uv pip install ".[dev,hf]"` stacks extras.
- Follow the [no-raw-PHI logging policy](security/no-raw-phi-logging.md) for every PII, de-identification,
  text-processing, service, and batch change.

## Code style

Ruff is the single source of truth for Python linting, import ordering, and formatting. Do not run Black, isort, flake8,
or editor-specific formatters over the repository. Before opening a pull request, run:

```bash
uv pip install -e ".[dev]"
make format
make lint
make type-check
make format-check
```

CI enforces `ruff check .`, `ruff format --check .`, and the scoped mypy configuration in `pyproject.toml`; pull
requests should not include unrelated formatting-only changes outside the files needed for the feature or fix.

Swift package code uses Apple `swift-format` with the checked-in `.swift-format` configuration. For changes under
`swift/OpenMedKit`, run:

```bash
make format-swift
make lint-swift
```

## Public API docstrings and export inventory

Every function and class exported from `openmed.__all__` must carry a meaningful
docstring. This keeps runtime help, IDE inspection, and explicitly configured
mkdocstrings entries useful without imposing coverage requirements on private or
internal modules. Data exports cannot carry symbol-specific instance docstrings,
so the gate resolves them into an explicit inventory instead of scoring them as
functions or classes.

```bash
python scripts/check_public_api_docstrings.py            # prints coverage + offenders
pytest tests/unit/test_public_api_docstrings.py -q       # CI gate
```

The standalone checker is stdlib-only (`ast` parsing, no runtime import). The
pytest gate adds a separate runtime parity check: it imports `openmed`, requires
the live `__all__` order to match the static inventory, verifies meaningful
docstrings on exported functions/classes, and checks the exact allow-list of data
exports without relying on generic built-in instance documentation. Function and
class coverage must remain at 100%; add a docstring in the same pull request as a
new public callable or class.

## Release outline

1. Bump the version via `make bump-patch` (or `bump-minor` / `bump-major`). These commands update `openmed/__about__.py`.
2. Run `python3 -m build` (or `make build`) to produce wheels and sdists.
3. Confirm the PyPI upload setup in [PyPI Publishing](release/trusted-publishing.md), then publish by pushing a tag (`vX.Y.Z`) to trigger `.github/workflows/publish.yml`.
4. Update `CHANGELOG.md` with release notes before tagging.

## Documentation deploys

The `pages.yml` workflow builds MkDocs on every push to `master`, bundles the marketing site with the docs (served at
`https://openmed.life/docs/`), and deploys the combined `site/` artifact via GitHub Pages. To test locally:

```bash
uv pip install ".[docs]"
uv run mkdocs serve -a 127.0.0.1:8008
make docs-stage
python3 -m http.server --directory site 9000  # inspect the marketing+docs bundle
```

Open build logs to confirm the same warnings would fail CI (we run `mkdocs build --strict` in automation). When you need
to publish outside CI, run `make docs-deploy`; it mirrors the workflow by building into `site/docs`, copying
`docs/website/` into `site/`, and force-pushing the bundle to `gh-pages`.

## Issue triage

- Keep user-facing docs inside `docs/`; new guides only require Markdown and optional front matter.
- Reference exact file + section when filing doc bugs so we can reproduce quickly.
- Prefer small pull requests that focus on a single guide or feature; CI + Pages runs on every PR.
- **Security issues are different:** a redaction bypass or PHI/PII leak must be
  reported privately, never as a public issue. Follow the
  [Security & Disclosure policy](security/disclosure-policy.md).
- Run `pytest tests/unit/test_no_raw_text_logging.py -q` when touching logging, text processing, service request
  handling, PII extraction, or de-identification code.

## Governance references

- [Release Streams & Channels](release/semver-and-channels.md) defines model artifact and library release cadence.
- [PyPI Publishing](release/trusted-publishing.md) documents package publishing, provenance, and token handling.
- [Generative Model Policy](generative-model-policy.md) defines approved and prohibited model-assisted workflows.

Ported rule-set files must start with an upstream attribution header naming the
source project, source URL, license, port date, and local modifications.
