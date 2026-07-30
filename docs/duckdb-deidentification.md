# DuckDB De-identification UDFs

OpenMed can register local DuckDB scalar UDFs so analysts can redact free-text
columns directly in SQL over in-memory tables, CSV, Parquet, or DuckDB files.
The integration is optional: importing `openmed` or `openmed.interop` does not
import DuckDB.

Install the optional extra:

```bash
pip install "openmed[duckdb]"
```

Register the functions on an existing connection:

```python
import duckdb
from openmed.interop.duckdb_udf import register_openmed_udfs

con = duckdb.connect(":memory:")
register_openmed_udfs(con)

rows = con.sql(
    """
    SELECT openmed_deidentify(note, 'hipaa_safe_harbor') AS redacted_note
    FROM notes
    """
).fetchall()
```

`openmed_deidentify(text, policy)` returns redacted text using the named
OpenMed policy profile. For convenience, the DuckDB adapter also accepts
`safe_harbor` as an alias for `hipaa_safe_harbor`:

```sql
SELECT openmed_deidentify(note, 'safe_harbor') AS redacted_note
FROM notes;
```

The helper `openmed_pii_count(text)` returns the number of PII entities detected
under the default policy. The adapter reuses a cached OpenMed model loader
within the process and does not log raw cell text.
