# PostgreSQL Redaction Adapter

OpenMed provides a local-first adapter for redacting explicitly selected text
columns in an existing PostgreSQL transaction. The adapter uses the Python
DB-API contract, so `psycopg` or another compatible driver remains an
application dependency and no PostgreSQL client is imported by OpenMed.

## Basic use

```python
import psycopg

from openmed.interop.postgres import redact_postgres_table

connection = psycopg.connect("postgresql://user:password@db.example/app")
try:
    result = redact_postgres_table(
        connection,
        table="clinical_notes",
        text_columns=["note", "summary"],
        key_column="record_id",
        batch_size=500,
        policy="hipaa_safe_harbor",
    )
    connection.commit()
finally:
    connection.close()

print(result.to_dict())
```

`connection` is supplied and owned by the caller. OpenMed does not connect,
close, or commit it. A successful run can therefore be committed together with
other application changes. If database or redaction processing fails, the
adapter rolls back the supplied connection and raises a generic
`PostgresRedactionError`.

## Privacy and transaction contract

- `text_columns` must be an explicit non-empty selection; unrelated columns
  are not read or written.
- The row key is ordered for deterministic bounded batches and must uniquely
  identify a row. Identifiers are safely quoted, while row values and batch
  bounds use PostgreSQL parameters.
- Updates use one parameterized `executemany` call for each changed batch.
- The result contains counts only: processed rows, updated rows, processed
  cells, redacted cells, redacted spans, and processed batches. It contains no
  source values, output text, row keys, or database error details.
- The module makes no mandatory network call. Supply a locally available
  deidentifier or ensure the OpenMed model is available through the normal
  local runtime before starting the transaction.

For an offline test, inject a deterministic callable:

```python
result = redact_postgres_table(
    connection,
    table="clinical_notes",
    text_columns=["note"],
    key_column="record_id",
    deidentifier=lambda text, **_: text.replace("synthetic", "[TOKEN]"),
)
```

The adapter is a redaction aid, not a compliance certification or a clinical
decision system. Review the selected columns, key constraints, transaction
isolation, and release policy before committing data.
