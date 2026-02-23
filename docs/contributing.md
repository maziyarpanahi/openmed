# Contributing & Releases

Short feedback cycles keep OpenMed shippable. This page captures the tooling you need to edit docs, cut releases, and
publish the package to PyPI or GitHub Pages.

## Local workflows

- `make help` prints a list of scripted tasks (build, publish, release, docs, etc.).
- `make docs-serve` starts the MkDocs preview with hot reload at `http://127.0.0.1:8008`.
- `make docs-build` runs `mkdocs build --strict` for CI parity.
- `uv pip install ".[dev]"` pulls in pytest + coverage; `uv pip install ".[dev,hf]"` stacks extras.

## Release outline

1. Bump the version via `make bump-patch` (or `bump-minor` / `bump-major`). These commands update `openmed/__about__.py`.
2. Run `python3 -m build` (or `make build`) to produce wheels and sdists.
3. Publish by pushing a tag (`vX.Y.Z`) to trigger `.github/workflows/publish.yml`.
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
