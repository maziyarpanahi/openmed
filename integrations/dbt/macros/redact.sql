{#
    OpenMed privacy-safe text projection macros for dbt.

    These macros only compile SQL. The default adapter implementation calls the
    OpenMed warehouse UDF ``openmed_deidentify(text, policy)``; a project can
    provide an adapter-specific ``<adapter>__redact`` implementation through
    dbt dispatch when its warehouse exposes an equivalent parameterized call.

    Macro arguments are compile-time column identifiers and policy names. They
    must never be row values. Validation deliberately keeps compiler errors
    generic so an accidental value is not copied into logs or build artifacts.
#}

{% set OPENMED_REDACTION_POLICIES = (
    'hipaa_safe_harbor',
    'hipaa_expert_review_assist',
    'gdpr_pseudonymization',
    'gdpr_art9_health',
    'research_limited_dataset',
    'strict_no_leak',
    'clinical_minimal_redaction',
    'canada_pipeda',
    'uk_ico_anonymisation',
    'australia_privacy_act',
    'china_pipl',
    'india_dpdp_act',
    'africa_malabo_baseline',
    'za_popia',
    'ng_ndpa',
    'ke_dpa',
    'india_health_id',
    'eg_pdpl',
    'ma_law_09_08'
) %}


{% macro _openmed_validate_policy(policy) -%}
    {%- if policy is not string or policy not in OPENMED_REDACTION_POLICIES -%}
        {{ exceptions.raise_compiler_error(
            'OpenMed redact() requires a supported canonical policy name.'
        ) }}
    {%- endif -%}
{%- endmacro %}


{% macro _openmed_validate_column(column, require_bare=false) -%}
    {%- if column is not string -%}
        {{ exceptions.raise_compiler_error(
            'OpenMed redaction requires an explicit column identifier.'
        ) }}
    {%- endif -%}
    {%- set normalized_column = column | trim -%}
    {%- set identifier_start = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_' -%}
    {%- set identifier_body = identifier_start ~ '0123456789' -%}
    {%- set identifier_parts = normalized_column.split('.') -%}
    {%- set validation = namespace(valid=true) -%}
    {%- if not normalized_column or identifier_parts | length > 2 -%}
        {%- set validation.valid = false -%}
    {%- endif -%}
    {%- for part in identifier_parts -%}
        {%- if not part -%}
            {%- set validation.valid = false -%}
        {%- else -%}
            {%- if part[0] not in identifier_start -%}
                {%- set validation.valid = false -%}
            {%- endif -%}
            {%- for character in part -%}
                {%- if character not in identifier_body -%}
                    {%- set validation.valid = false -%}
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- endfor -%}
    {%- if not validation.valid -%}
        {{ exceptions.raise_compiler_error(
            'OpenMed redaction accepts one explicit column, not a wildcard or SQL fragment.'
        ) }}
    {%- endif -%}
    {%- if require_bare and identifier_parts | length != 1 -%}
        {{ exceptions.raise_compiler_error(
            'redact_columns() requires bare column names so each output is explicitly aliased.'
        ) }}
    {%- endif -%}
{%- endmacro %}


{% macro redact(column, policy='hipaa_safe_harbor') -%}
    {{- _openmed_validate_policy(policy) -}}
    {{- _openmed_validate_column(column) -}}
    {%- set normalized_column = column | trim -%}
    {%- if adapter is defined -%}
        {{- adapter.dispatch('redact')(normalized_column, policy) -}}
    {%- else -%}
        {{- default__redact(normalized_column, policy) -}}
    {%- endif -%}
{%- endmacro %}


{% macro default__redact(column, policy) -%}
openmed_deidentify({{ column }}, '{{ policy }}')
{%- endmacro %}


{% macro redact_columns(columns, policy='hipaa_safe_harbor') -%}
    {%- if columns is none or columns is string or columns is not sequence -%}
        {{ exceptions.raise_compiler_error(
            'redact_columns() requires a non-empty sequence of explicit column names; full-table projections are not allowed.'
        ) }}
    {%- endif -%}
    {{- _openmed_validate_policy(policy) -}}
    {%- set state = namespace(count=0) -%}
    {%- for column in columns -%}
        {%- set state.count = state.count + 1 -%}
        {{- _openmed_validate_column(column, require_bare=true) -}}
        {{- redact(column, policy) }} as {{ column | trim }}{{- ', ' if not loop.last -}}
    {%- endfor -%}
    {%- if state.count == 0 -%}
        {{ exceptions.raise_compiler_error(
            'redact_columns() requires at least one explicit column; full-table projections are not allowed.'
        ) }}
    {%- endif -%}
{%- endmacro %}
