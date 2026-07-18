{#
    OpenMed de-identification macros for warehouse transformations.

    `redact(column, policy)` emits a call to the OpenMed warehouse
    de-identification UDF, `openmed_deidentify(text, policy)`. The UDF must be
    registered on the target warehouse/connection before the model runs (see the
    package README). These macros only generate SQL — they never see PHI
    themselves; redaction happens in-warehouse inside the UDF.

    `column` is rendered as a bare SQL identifier and `policy` as a single-quoted
    literal, so both are trusted, developer-controlled compile-time inputs: pass a
    real column name and a bundled OpenMed policy profile (e.g. hipaa_safe_harbor,
    gdpr_pseudonymization, strict_no_leak), never row data or untrusted strings.
#}

{% macro redact(column, policy='hipaa_safe_harbor') -%}
openmed_deidentify({{ column }}, '{{ policy }}')
{%- endmacro %}


{% macro redact_columns(columns, policy='hipaa_safe_harbor') -%}
{%- for column in columns -%}
{{ redact(column, policy) }} as {{ column }}{{ ", " if not loop.last }}
{%- endfor -%}
{%- endmacro %}
