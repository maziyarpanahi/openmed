# Nix and Pixi Development

OpenMed includes a Nix flake for a pinned package build and a reproducible
development shell, plus a Pixi manifest for conda-centric and HPC workflows.
These are additional build paths for NixOS, nix-darwin, and cross-platform
development; the existing uv and pip workflows remain supported.

## Pixi development

Pixi provides a locked Python 3.12 environment for Linux x86_64, Intel Macs,
and Apple Silicon Macs running macOS 14 or newer. It complements uv by
resolving the conda system layer while installing OpenMed and its Python
dependencies from the local project. The checked-in `pixi.lock` contains
resolutions for `linux-64`, `osx-64`, and `osx-arm64`.

Install Pixi using the instructions for your operating system, then from the
repository root run:

```bash
pixi install
pixi run test
pixi run lint
pixi run docs
```

The default environment includes the `dev` and `docs` extras. The manifest
also exposes named environments that mirror the optional-dependency extras in
`pyproject.toml`:

```bash
pixi install --environment hf
pixi install --environment service
pixi install --environment mlx  # Apple Silicon only
```

Use `pixi run --environment <name> <command>` to run a command in one of these
feature environments. Keep `pixi.lock` committed when changing the manifest;
`pixi lock` refreshes all configured platforms and environments. Pixi's local
environments live under `.pixi/`, which is intentionally ignored by Git.

## Enter the development shell

Install Nix with flakes enabled, clone the repository, and run:

```bash
nix develop
```

The shell provides Python 3.12, OpenMed, the tools and Python packages from the
`dev` extra, and the test-only PyArrow dependency used by the collected suite.
Optional ML and platform extras such as MLX and Core ML are not part of the Nix
shell.

Run the same test command used by the repository gate:

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
