# Software Bill of Materials (SBOM)

OpenMed publishes a [CycloneDX](https://cyclonedx.org/) software bill of
materials so downstream healthcare integrators can inventory the dependency tree
and answer a supply-chain audit. The SBOM names `openmed` as the root component
and lists the dependencies resolved in the target environment.

## Generate it locally

```bash
make sbom
```

This syncs the locked **runtime** environment and writes `sbom.cdx.json`
(CycloneDX 1.6 JSON) to the repository root. Equivalent one-liner:

```bash
uv sync --frozen
uv run --no-project --with 'cyclonedx-bom>=4.6,<7' \
  python scripts/security/generate_sbom.py
```

The generator
([`scripts/security/generate_sbom.py`](https://github.com/maziyarpanahi/openmed/blob/master/scripts/security/generate_sbom.py))
introspects the environment with `cyclonedx-py environment`, then stamps the
package version onto the root component — hatch resolves the version
dynamically from `openmed/__about__.py`, which PEP 621 metadata alone cannot
express.

`sbom.cdx.json` is a generated artifact and is **not** committed (see
`.gitignore`).

## Cover specific extras

By default the SBOM reflects the base runtime dependencies (`openmed` plus
`pysbd` and `Faker`). To capture a particular install profile, sync that extra
first, then regenerate:

```bash
uv sync --frozen --extra service --extra hf
uv run --no-project --with 'cyclonedx-bom>=4.6,<7' \
  python scripts/security/generate_sbom.py
```

## Where it is published

- **CI** — the `sbom` job in `.github/workflows/ci.yml` regenerates the SBOM on
  every push and pull request and uploads `sbom.cdx.json` as the `sbom`
  artifact.
- **Releases** — `.github/workflows/publish.yml` regenerates and uploads the
  SBOM alongside each tagged PyPI publish. SBOM generation there is fail-open and
  never blocks a release.

## Consume and verify

The generator validates the document against the CycloneDX 1.6 schema before
writing it. Downstream, load `sbom.cdx.json` into any CycloneDX-aware tool — for
example [Dependency-Track](https://dependencytrack.org/) — or scan it for known
vulnerabilities:

```bash
grype sbom:sbom.cdx.json     # or: trivy sbom sbom.cdx.json
```

To confirm the inventory yourself, regenerate it with `make sbom` and diff the
`components` list; only the timestamp and serial number change between runs.

See also [Supply Chain Controls](supply-chain.md) and the
[Dependency Policy](dependency-policy.md).
