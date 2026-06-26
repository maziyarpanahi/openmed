# Supply Chain Controls

OpenMed commits `uv.lock` so Python dependency resolution is reproducible
across local development and CI. Pull requests and pushes run a dedicated
lockfile drift gate:

```bash
uv lock --check
```

The gate verifies that `uv.lock` still matches `pyproject.toml` without
installing the full development environment. If a dependency or optional extra
changes, regenerate the lockfile locally before pushing:

```bash
uv lock
```

Commit the updated `uv.lock` with the dependency change. The CI failure message
uses the same remediation: `Run 'uv lock' and commit the updated uv.lock.`
Do not edit the lockfile by hand.
