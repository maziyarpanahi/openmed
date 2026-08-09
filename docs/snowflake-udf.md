# Snowflake Python De-identification UDFs

OpenMed can register a Snowpark Python UDF that de-identifies text inside a
Snowflake warehouse. The handler is a plain Python callable, while Snowpark is
an optional dependency: importing `openmed` or `openmed.interop` does not
import Snowpark.

Install the optional extra in the environment that creates the registration:

```bash
pip install "openmed[snowflake]"
```

## Register with Snowpark

```python
from openmed.interop.snowflake_udf import register_udf
from snowflake.snowpark import Session

session = Session.builder.configs(connection_parameters).create()
register_udf(session)
```

The default registration creates `OPENMED_DEIDENTIFY(text)` and supplies the
published `openmed` package through Snowpark's `packages` argument. The Python
handler defaults to `method="mask"` and OpenMed's default policy, so a query
can redact a text column directly:

```sql
SELECT OPENMED_DEIDENTIFY(note) AS redacted_note
FROM clinical_notes;
```

Use `packages` to add or pin Snowflake package references and `imports` for
stage paths containing additional Python modules. Both values are forwarded to
`session.udf.register`:

```python
register_udf(
    session,
    name="OPENMED_DEIDENTIFY_V2",
    packages=["openmed", "some-anaconda-package"],
    imports=["@my_stage/python/helpers.py"],
)
```

For a permanent UDF, pass `is_permanent=True` and a Snowflake stage through
`stage_location`:

```python
register_udf(
    session,
    name="OPENMED_DEIDENTIFY_PERMANENT",
    is_permanent=True,
    stage_location="@my_stage/openmed_udfs",
)
```

## Generate SQL

Deployments that use SQL rather than Snowpark registration can generate the
same function definition:

```python
from openmed.interop.snowflake_udf import generate_create_function_sql

sql = generate_create_function_sql(
    name="OPENMED_DEIDENTIFY",
    packages=["openmed"],
)
session.sql(sql).collect()
```

The generated statement includes `LANGUAGE PYTHON`, the configured
`RUNTIME_VERSION`, `PACKAGES`, and the
`openmed.interop.snowflake_udf.deidentify_udf` `HANDLER`.

This adapter does not contact a Snowflake account during local tests. Supply
your own Snowflake package or stage configuration for deployment, and review
warehouse access and retention controls before processing clinical data.
