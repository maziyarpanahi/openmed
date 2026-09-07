# Project scaffolding with `openmed init`

`openmed init` creates a small, deterministic OpenMed project without contacting
the network, downloading a model, reading clinical data, or copying credentials
from your environment. It provides a validated configuration, a persona-specific
starter pipeline, offline environment defaults, and setup guidance.

## Create a project

Pass a destination and one of the three persona presets:

```bash
openmed init my-openmed-project --preset researcher
cd my-openmed-project
python pipeline.py --check
```

The destination defaults to the current directory. `--persona` is an alias for
`--preset`:

```bash
openmed init ./service-demo --persona app-developer
openmed init ./batch-demo --preset data-engineer
```

The command is non-interactive and works fully offline. `pipeline.py --check`
also stays model-free: it loads the generated config and reports only the preset,
policy, offline setting, and synthetic-record count.

## Persona presets

All presets use clearly marked synthetic input and set `local_only = true`.
They differ in their starter flow and resource configuration:

| Preset | Starter flow | Policy |
| --- | --- | --- |
| `researcher` | De-identify one synthetic research note | `research_limited_dataset` |
| `app-developer` | Adapt a synthetic request mapping to the library API | `strict_no_leak` |
| `data-engineer` | Process a synthetic in-memory batch with `BatchProcessor` | `strict_no_leak` |

These tracks reuse the API paths introduced in the
[persona quickstarts](../quickstarts.md). The generated examples are starting
points, not production release decisions: validate model recall and residual
disclosure risk on approved local evaluation data before releasing any output.

## Generated files

Every preset owns the same five relative paths:

| File | Purpose |
| --- | --- |
| `openmed.toml` | Minimal cache-only `OpenMedConfig` values |
| `pipeline.py` | Persona-specific pipeline over synthetic input |
| `.env.example` | Offline environment flags; contains no credential fields |
| `.gitignore` | Excludes `.env`, local environments, caches, and outputs |
| `README.md` | Local validation and model-cache instructions |

Before writing, OpenMed validates the rendered config mapping against the
bundled Draft 2020-12 schema at `openmed/core/config.schema.json`. The schema is
packaged with wheels, so validation does not fetch a remote `$schema` URL.

The generated `.env.example` is not loaded automatically. Export it through
your process manager or shell when you want the additional environment-level
offline guards. The project config already makes the starter pipeline
cache-only.

## Safe reruns and overwrites

An identical rerun is a no-op: existing matching files are reported as
unchanged and their modification times are preserved.

If any managed file differs, the command checks all five paths and then exits
without writing anything:

```text
Refusing to overwrite existing scaffold files: pipeline.py.
Re-run with --force to replace only these files.
```

Use `--force` only after reviewing the named conflicts:

```bash
openmed init my-openmed-project --preset researcher --force
```

Forced initialization replaces only differing regular files in the five-file
managed set. It never deletes unrelated files. Symbolic-link destinations and
symbolic-link managed paths are rejected even with `--force`, preventing the
scaffold from writing through a link to another location.

## Scriptable output

Like every standard OpenMed CLI leaf command, `init` supports `--json`:

```bash
openmed init ./demo --preset data-engineer --json
```

The success envelope lists created, overwritten, and unchanged relative paths;
it never includes file contents or environment values:

```json
{
  "ok": true,
  "command": "init",
  "data": {
    "destination": "demo",
    "preset": "data-engineer",
    "created": [
      "openmed.toml",
      "pipeline.py",
      ".env.example",
      ".gitignore",
      "README.md"
    ],
    "overwritten": [],
    "unchanged": []
  }
}
```

A collision uses the stable error code `scaffold_conflict`; an invalid or
unsafe destination uses `invalid_scaffold`.

## Run with a cached model

Initialization deliberately does not download models. Prepare the PII model in
an explicitly approved connected environment using the
[model-cache workflow](../model-registry.md), then return to an offline runtime:

```bash
python pipeline.py
```

If the required model is absent, the generated `local_only = true` setting
fails closed instead of attempting a download. For a complete disconnected
installation, use the [air-gapped installation guide](../offline-install.md).

Do not place real patient text, tokens, private keys, or reversible mappings in
the generated repository. Store operational secrets outside source control and
follow the [configuration guidance](../configuration.md) for local paths and
runtime overrides.
