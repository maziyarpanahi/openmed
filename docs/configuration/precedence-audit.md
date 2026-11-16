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

Inspect the sources for the current CLI configuration without printing values:

```bash
openmed config provenance
openmed config provenance --json
```

The text output lists each key, its source class, and any conflict category.
Use `OPENMED_CONFIG` to select the local configuration file for the audit.

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
command-line arguments. Both `resolution.provenance_report` and
`resolution.to_dict()` return this value-free report. Effective values remain
available separately through `resolution.values` for in-process use and must
not be written to audit artifacts.

## Service settings

The REST, gRPC and GraphQL service reads its `OPENMED_SERVICE_*` settings from
the environment only, so `resolve_service_configuration()` resolves the
`service` section of `openmed/core/settings.schema.json` against an
environment snapshot instead:

```python
from openmed.core.config_provenance import resolve_service_configuration

provenance = resolve_service_configuration(
    {
        "OPENMED_SERVICE_AUTH_ENABLED": "true",
        "OPENMED_SERVICE_DIGEST_KEY_FILE": "/run/secrets/digest-key",
    }
)
report = provenance.provenance_report
assert report["keys"]["OPENMED_SERVICE_AUTH_ENABLED"]["source_class"] == (
    "environment"
)
assert report["keys"]["OPENMED_SERVICE_DIGEST_KEY"]["source_class"] == (
    "secret_file"
)
```

The report lists every declared service setting, sorted by name, and uses
this order:

```text
default < environment < secret_file
```

- `default`: the variable is unset or blank, so the declared default applies.
  Blank counts as unset because the service parsers treat it that way.
- `environment`: the variable is set. Its parsed value is compared in memory
  with the declared default to choose `same_value` or `overridden`; a value
  that does not parse by its declared type reports `overridden`.
- `secret_file`: a credential whose `NAME_FILE` companion names a mounted
  secret file. `openmed.service.secret_files` reads it. When the inline value
  is set as well, the entry reports `conflicting_secret_sources`, which the
  service refuses at startup.

Secret settings report presence only: their source class, `"secret": true`,
and `"present"`. They are never compared with anything, and secret files are
never opened. A new credential setting gets the `secret_file` source by
declaring its companion with `x-secret-file-for` in the settings schema. Each
key in the report has this shape:

```json
{
  "source_class": "secret_file",
  "conflict_category": "none",
  "sources": ["secret_file"],
  "overridden_sources": [],
  "secret": true,
  "present": true
}
```

The report, its JSON form (`provenance.to_json()`) and `repr(provenance)`
contain no values, paths or environment payloads. The same environment
snapshot always produces the same report, whatever the key order.

## Auditing existing OpenMed defaults

If `defaults` is omitted, `resolve_configuration()` reads the dataclass field
defaults from `OpenMedConfig` without applying environment overrides first. This
keeps the default layer distinct from the environment layer. Use
`default_config_values()` when an adapter needs to inspect or extend that
baseline explicitly.

The audit describes configuration provenance; it is not a compliance
certification or a clinical decision guarantee.
