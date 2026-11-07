# Software Bill of Materials (SBOM)

OpenMed provides complementary CycloneDX 1.6 software bills of materials for
the checked-in Python dependency manifests, an installed Python environment,
and release container images.

## Deterministic source evidence

Generate a source-manifest SBOM without resolving or installing packages:

```bash
python scripts/licenses/sbom.py --output sbom.cdx.json
```

The generator reads bounded copies of `pyproject.toml`, `uv.lock`, and, for the
dynamic package version, `openmed/__about__.py`. It follows only the base
`[project].dependencies` closure, so optional extras are excluded. It does not
inspect an installed environment, contact a package index, or require network
access.

When `--source-revision` is omitted, the generator reads the local Git `HEAD`.
Supply the revision explicitly for an exported source tree:

```bash
python scripts/licenses/sbom.py \
  --source-revision <commit-sha> \
  --output sbom.cdx.json
```

The JSON omits timestamps and random serial numbers. Repeated runs over the
same revision and manifest bytes are byte-identical. The
`metadata.properties` section records:

- `openmed:source-revision`
- `openmed:pyproject-sha256`
- `openmed:lockfile-sha256`
- `openmed:manifest-sha256`
- `openmed:version-source-sha256` when the version is dynamic

Package names, versions, PURLs, dependency edges, and artifact hashes come from
the local manifests. Reviewed SPDX identifiers and expressions are preserved;
an absent, malformed, or unreviewed license becomes `NOASSERTION`. Lock-file
download URLs, credentials, local source paths, build paths, timestamps, and
environment details are omitted. Inputs and output are bounded, and output is
replaced atomically after successful rendering.

## Installed environment

Generate the existing installed-environment SBOM with:

```bash
make sbom
```

This syncs the locked base runtime environment and writes `sbom.cdx.json` at
the repository root. To capture a particular installation profile, sync the
extras first and run the generator directly:

```bash
uv sync --frozen --extra service --extra hf
uv run --no-project --with 'cyclonedx-bom>=4.6,<7' \
  python scripts/security/generate_sbom.py
```

The installed-environment generator validates the document against the
CycloneDX 1.6 schema. CI regenerates it on every push and pull request and
uploads the `sbom` artifact. Tagged release workflows attach it to the GitHub
release and retain it as a workflow artifact.

`sbom.cdx.json` is generated and is not committed. Downstream tools can ingest
either Python SBOM, for example:

```bash
grype sbom:sbom.cdx.json     # or: trivy sbom sbom.cdx.json
```

Container releases publish a separate [image SBOM](../supply-chain/sbom.md)
covering operating-system packages and image contents. See also
[Supply Chain Controls](supply-chain.md) and the
[Dependency Policy](dependency-policy.md).
