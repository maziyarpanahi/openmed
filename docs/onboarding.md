# New Contributor Onboarding

Welcome! This guide walks you through your first contribution to OpenMed, from finding a `good first issue` to opening a clean pull request.

## 1. Find a good first issue

Browse the [issues page](https://github.com/maziyarpanahi/openmed/issues) and filter by the [`good first issue`](https://github.com/maziyarpanahi/openmed/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) label. Pick one that:

- Has a clear description and acceptance criteria
- Is narrowly scoped (ideally 1–2 files)
- Does not require maintainer-only credentials or access to private data
- Has no open PRs claiming it (search `repo:maziyarpanahi/openmed "in:body #<ISSUE_NUMBER>"`)

### Claiming an issue

Comment on the issue to let the maintainers know you're working on it. A short "I'd like to take this" is enough — no formal assignment is required.

If the issue does not match the available templates, open a new item from the
[issue forms](https://github.com/maziyarpanahi/openmed/issues/new/choose) and
include the smallest reproducible scope you can.

## 2. Set up your environment

```bash
# Fork and clone the repo
git clone https://github.com/<your-username>/openmed.git
cd openmed

# Create the local virtual environment and install dev dependencies
uv venv
uv pip install -e ".[dev]"

# (Optional) Install pre-commit hooks
pre-commit install
```

## 3. Run the tests

Before making changes, confirm the baseline passes:

```bash
# Run the full test suite
.venv/bin/python -m pytest tests/ -q

# Or run a scoped subset (replace with the relevant test path)
.venv/bin/python -m pytest tests/unit/test_pii.py -q
```

## 4. Make your change

- Create a branch from `master`:

  ```bash
  git checkout master
  git pull origin master
  git checkout -b fix/<short-description>
  ```

- Implement your change. Keep it focused on the issue scope.
- Add or update tests if the issue asks for them.
- Run `make format` and `make lint` before committing.

## 5. Open a pull request

```bash
git add <changed-files>
git commit -m "fix(scope): brief description"
git push -u origin fix/<short-description>
```

Then open a PR using `gh pr create` or via the GitHub web UI. Fill in the PR template with:

- A clear description of what you changed and why
- Reference the issue number (`Closes #<number>`)
- Your test plan (what you ran to verify the change)

## Important rules

- **No real PHI.** Never include real patient data, names, or identifiable health information in tests, examples, or documentation. Use synthetic data only.
- **One PR per task.** Keep your pull request narrowly scoped to a single issue.
- **Read the full contributing guide.** See [Contributing & Releases](contributing.md) for code style, release procedures, and documentation deployment details.

## Need help?

- Check the [FAQ](faq.md) for common questions.
- File a new issue if you're stuck or need clarification on scope.
