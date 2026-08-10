# Configuration precedence audit

OpenMed can receive a setting from built-in defaults, a local TOML file, the
process environment, or a command-line parser. The precedence audit gives
these sources one deterministic order and explains which source supplied each
effective key without copying any configuration value into the report.

The order is:

```text
default < file < environment < cli
```

Later sources win. The report uses the stable source classes `default`, `file`,
`environment`, and `cli`. A key has one of these conflict categories:

- `none`: only one source supplied the key.
- `same_value`: multiple sources supplied equivalent values.
- `overridden`: at least one lower-precedence source supplied a different value.

## Resolve local sources

Pass mappings for deterministic tests and adapters. A file source can also be a
local TOML path; no network or remote service is consulted.

```python
from openmed.core.config_provenance import resolve_configuration

resolution = resolve_configuration(
    defaults={"timeout": 300, "local_only": False},
    file_config="./openmed.toml",
    environment={"OPENMED_TIMEOUT": "120"},
    cli={"timeout": 60},
)

assert resolution.values["timeout"] == 60
audit = resolution.provenance_report
```

`environment=None` audits the current process environment. Pass an explicit
snapshot, including `{}`, when reproducibility matters. Generic names use the
`OPENMED_` prefix. Known compatibility aliases include `HF_TOKEN`,
`OPENMED_TORCH_DEVICE`, `OPENMED_DEVICE`, and `OPENMED_OFFLINE`.

The effective values are intentionally separate from the report because they
may contain credentials, paths, or other sensitive settings. The value-free
report has this shape:

```json
{
  "schema_version": 1,
  "precedence": ["default", "file", "environment", "cli"],
  "keys": {
    "timeout": {
      "source_class": "cli",
      "conflict_category": "overridden",
      "sources": ["default", "file", "environment", "cli"],
      "overridden_sources": ["default", "file", "environment"]
    }
  }
}
```

The report contains no selected values, file paths, environment payloads, or
command-line arguments. Store or log `resolution.provenance_report`, not
`resolution.to_dict()`, when producing an audit artifact.

## Auditing existing OpenMed defaults

If `defaults` is omitted, `resolve_configuration()` reads the dataclass field
defaults from `OpenMedConfig` without applying environment overrides first. This
keeps the default layer distinct from the environment layer. Use
`default_config_values()` when an adapter needs to inspect or extend that
baseline explicitly.

The audit describes configuration provenance; it is not a compliance
certification or a clinical decision guarantee.
