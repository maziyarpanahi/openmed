# Contributing & Releases

Short feedback cycles keep OpenMed shippable. This page captures the tooling you need to edit docs, cut releases, and
publish the package to PyPI or GitHub Pages.

## Local workflows

- `make help` prints a list of scripted tasks (build, publish, release, docs, etc.).
- `make docs-serve` starts the MkDocs preview with hot reload at `http://127.0.0.1:8008`.
- `make docs-build` runs `mkdocs build --strict` for CI parity.
- `uv pip install ".[dev]"` pulls in pytest + coverage; `uv pip install ".[dev,hf]"` stacks extras.

## Release outline

1. Bump the version via `make bump-patch` (or `bump-minor` / `bump-major`). These commands call `scripts/release/release.py`.
2. Run `python -m build` (or just `make build`) to produce wheels and sdists.
3. Publish with `make publish` (wraps `hatch publish`) once smoke tests pass.
4. Tag the release in GitHub so Pages + automation deployments tie back to the version.

## Documentation deploys

The `pages.yml` workflow builds MkDocs on every push to `master`, bundles the marketing site with the docs (served at
`https://openmed.life/docs/`), and deploys the combined `site/` artifact via GitHub Pages. To test locally:

```bash
uv pip install ".[docs]"
uv run mkdocs serve -a 127.0.0.1:8008
make docs-stage
python -m http.server --directory site 9000  # inspect the marketing+docs bundle
```

Open the CLI log to confirm the same warnings would fail CI (we run `mkdocs build --strict` in automation). When you need
to publish outside CI, run `make docs-deploy`; it mirrors the workflow by building into `site/docs`, copying
`docs/website/` into `site/`, and force-pushing the bundle to `gh-pages`.

## Issue triage

- Keep user-facing docs inside `docs/`—new guides only require Markdown and optional front matter.
- Reference the exact file + section when filing doc bugs so we can reproduce and fix them quickly.
- Prefer small pull requests that focus on a single guide or feature; the automation suite (CI + Pages) runs on every PR.
