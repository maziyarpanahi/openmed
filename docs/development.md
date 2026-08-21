# Development Workflow

OpenMed uses [uv](https://docs.astral.sh/uv/) as the canonical local development
workflow. The committed `uv.lock` keeps the editable package, test tools, and
lint tools reproducible across machines and CI.

## uv quickstart

Install uv using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/),
then run these commands from the repository root:

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev pre-commit install
```

`uv sync` creates `.venv`, installs OpenMed in editable mode, and uses the
committed lockfile. Add optional capabilities to the same command, for example:

```bash
uv sync --frozen --extra dev --extra hf
```

Run tools through the locked environment:

```bash
uv run --frozen --extra dev pytest tests/unit/test_offline_mode.py -q
make lint
make format-check
make type-check
```

When changing dependencies, regenerate `uv.lock`, review both dependency files,
and run `make lock-check`. CI runs `uv lock --check` and the complete test
matrix; local iteration should use the smallest relevant test file or test case.

## Pip fallback

Pip remains supported for environments that cannot install uv. It does not
provide the lockfile-based workflow used by CI:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pre-commit install
```

On Windows, use `.venv\\Scripts\\python.exe` in place of
`.venv/bin/python`. The fallback supports the same editable package and extras;
the uv workflow remains the canonical path for contributor and CI commands.

## Nix development

OpenMed also includes a Nix flake for a pinned package build and a reproducible
development shell. This is an additional build path for NixOS and nix-darwin
users; it is not required for the uv workflow above.

## Enter the development shell

Install Nix with flakes enabled, clone the repository, and run:

```bash
nix develop
```

The shell provides Python 3.12, OpenMed, the tools and Python packages from the
`dev` extra, and the test-only PyArrow dependency used by the collected suite.
Optional ML and platform extras such as MLX and Core ML are not part of the Nix
shell.

Run the complete repository suite before opening a pull request:

```bash
python -m pytest tests/ -q
```

For a single non-interactive command, use:

```bash
nix develop --command python -m pytest tests/ -q
```

## Build the package

Build the default OpenMed package with:

```bash
nix build
```

The `result` symlink points to the package in the Nix store. The package uses
`buildPythonPackage` with Hatchling and contains the `openmed` command and
Python package.

## Validate and update the pin

Run the flake checks before submitting a Nix-related change:

```bash
nix flake check --print-build-logs
```

This builds both the OpenMed package and development shell. CI also runs the
complete test suite inside that shell on Linux.

`flake.lock` records the exact nixpkgs 26.05 revision and content hash. Update
that pin only as an intentional dependency-maintenance change, then validate
and commit both lock and flake files together:

```bash
nix flake update
nix flake check --print-build-logs
nix develop --command python -m pytest tests/ -q
```
