# Executable UDF column redaction

OpenMed can run as a one-column `TabSeparated` transform for ClickHouse-style
executable user-defined functions. Each input row on stdin produces exactly one
redacted output row on stdout, in the same order. The adapter buffers rows and
reuses one model loader across batches; it never writes raw values to logs or
diagnostics.

Install OpenMed and cache the selected PII model before enabling the function.
The database host can then run with `OPENMED_OFFLINE=1` so query-time inference
does not require network access.

## Run the adapter directly

The package installs a dedicated console entrypoint:

```bash
printf 'Patient Jane Roe called 555-0100\n' \
  | OPENMED_OFFLINE=1 openmed-executable-udf --batch-size 32
```

Output is one escaped TSV string per row:

```text
Patient [NAME] called [PHONE]
```

Use `--policy`, `--method`, `--model-name`, `--lang`, and
`--confidence-threshold` to select the same de-identification behavior used by
the Python API. The default safety sweep stays enabled.

## Register the ClickHouse function

ClickHouse searches direct executable commands under its configured
`user_scripts_path`. Create `/var/lib/clickhouse/user_scripts/openmed-redact`
with an absolute path to the environment where OpenMed is installed:

```sh
#!/bin/sh
export OPENMED_OFFLINE=1
exec /opt/openmed/.venv/bin/openmed-executable-udf "$@"
```

Make the wrapper executable, readable only as appropriate for the ClickHouse
service account, and test it as that account. Then add
`/etc/clickhouse-server/openmed_redact_function.xml`:

```xml
<functions>
    <function>
        <type>executable</type>
        <name>openmed_redact</name>
        <return_type>String</return_type>
        <argument>
            <type>String</type>
            <name>text</name>
        </argument>
        <format>TabSeparated</format>
        <command>openmed-redact --batch-size 32</command>
        <execute_direct>true</execute_direct>
        <send_chunk_header>false</send_chunk_header>
        <command_read_timeout>120000</command_read_timeout>
        <command_write_timeout>120000</command_write_timeout>
        <stderr_reaction>throw</stderr_reaction>
        <check_exit_code>true</check_exit_code>
    </function>
</functions>
```

Ensure the server's `user_defined_executable_functions_config` pattern matches
that XML filename, reload the configuration, and confirm the function is
available before using it:

```sql
SELECT name, origin
FROM system.functions
WHERE name = 'openmed_redact';

SELECT openmed_redact(note_text)
FROM synthetic_notes;
```

The adapter expects exactly one non-nullable `String` argument serialized as
`TabSeparated`. It rejects extra TSV columns and fails the query when batch
redaction cannot return exactly one valid result for every input row.

See the [ClickHouse executable UDF documentation](https://clickhouse.com/docs/sql-reference/functions/udf#executable-user-defined-functions)
for server paths, configuration reload behavior, and executable settings.
