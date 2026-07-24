# Nix Development

OpenMed includes a Nix flake for a pinned package build and a reproducible
development shell. This is an additional build path for NixOS and nix-darwin
users; the existing uv and pip workflows remain supported.

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
