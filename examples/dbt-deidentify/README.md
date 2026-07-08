# OpenMed de-identification for dbt

A small [dbt](https://www.getdbt.com/) package that expresses PHI redaction as a
reusable **macro** and a **staging model**, so analytics-engineering teams can
de-identify free-text columns declaratively instead of writing ad-hoc SQL.

The macros wrap a warehouse **de-identification UDF** — `openmed_deidentify(text,
policy)` — that must be registered on your warehouse connection. This package
generates SQL only; the redaction runs in-warehouse inside the UDF, so PHI never
leaves the database.

## Layout

```
dbt-deidentify/
├── dbt_project.yml
├── macros/redact.sql            # redact() / redact_columns()
├── models/stg_redacted.sql      # applies redact() to configured columns
├── models/schema.yml            # documents the model and its columns
└── seeds/synthetic_patients.csv # fully synthetic sample data (no real PHI)
```

## Macro signature

```sql
{{ redact(column, policy='hipaa_safe_harbor') }}
-- renders: openmed_deidentify(<column>, '<policy>')

{{ redact_columns(['notes', 'reason_for_visit'], policy='gdpr_pseudonymization') }}
-- renders one redacted, aliased column per input column
```

`policy` accepts any bundled OpenMed policy profile (for example
`hipaa_safe_harbor`, `gdpr_pseudonymization`, `strict_no_leak`).

## Prerequisite: register the UDF

The model fails with a clear "function `openmed_deidentify` does not exist" error
until the UDF is registered on the connection. Register it once per session.

**DuckDB (local dev / `dbt-duckdb`):**

```python
import duckdb
from openmed.interop.duckdb_udf import register_openmed_udfs

con = duckdb.connect("dev.duckdb")
register_openmed_udfs(con)  # exposes openmed_deidentify(text, policy)
```

With `dbt-duckdb`, register the UDF through its Python
[plugin hook](https://github.com/duckdb/dbt-duckdb#using-local-python-modules) so
every model run has the function available.

**Snowflake / BigQuery:** register the equivalent remote-function / UDF for
`openmed_deidentify(text, policy)` (see the warehouse UDF tasks) and point this
package's profile at that warehouse — the models are unchanged.

## Run it locally

dbt reads connection settings from a `profiles.yml` (by default `~/.dbt/profiles.yml`).
Add a DuckDB profile named to match this project:

```yaml
# ~/.dbt/profiles.yml
openmed_deidentify:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: dev.duckdb
```

Then, with the `openmed_deidentify` UDF registered on that DuckDB file:

```bash
pip install "openmed[duckdb]" dbt-duckdb
cd examples/dbt-deidentify
dbt seed && dbt run          # UDF must be registered on the connection first
```

`stg_redacted` materializes with every configured text column replaced by its
de-identified value; `patient_id` and other structured columns pass through.

## Testing

`tests/unit/integrations/test_dbt_package.py` compiles the macro and executes the
generated SQL against DuckDB (the local adapter's engine) with a stubbed UDF,
asserting the configured columns are redacted, PHI is absent from the output, and
a missing UDF raises a clear error. It needs no warehouse and no network.
