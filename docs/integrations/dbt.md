# dbt Privacy-Safe Text Projection

OpenMed provides a standalone dbt macro for selecting which text columns are
redacted in a model. The macro compiles to SQL and keeps the selected policy
visible in the model definition; it does not inspect rows, call a service, or
make a network request.

Copy [`integrations/dbt/macros/redact.sql`](https://github.com/maziyarpanahi/openmed/blob/master/integrations/dbt/macros/redact.sql)
into a project's dbt `macro-paths`, or package it with the project's dbt
integration. The default implementation emits the parameterized warehouse UDF
call `openmed_deidentify(text, policy)`. Register or provision that equivalent
UDF in the target warehouse separately; the macro does not provision it.

## Explicit projection

Use `redact()` for one explicitly named column:

```sql
select
    encounter_id,
    {{ redact('clinical_note', policy='strict_no_leak') }} as clinical_note
from {{ ref('stg_encounters') }}
```

For more than one text column, use `redact_columns()` with a non-empty list of
bare column names:

```sql
select
    encounter_id,
    {{ redact_columns(
        ['clinical_note', 'free_text'],
        policy='hipaa_safe_harbor'
    ) }}
from {{ ref('stg_encounters') }}
```

The second example compiles to two explicitly aliased expressions. Structured
columns are not selected by the macro and the source relation is not rewritten.
Choose the model materialization and the non-sensitive columns in the model
itself.

## Policy validation

`redact()` and `redact_columns()` accept only canonical OpenMed policy names:

`hipaa_safe_harbor`, `hipaa_expert_review_assist`, `gdpr_pseudonymization`,
`gdpr_art9_health`, `research_limited_dataset`, `strict_no_leak`,
`clinical_minimal_redaction`, `canada_pipeda`, `uk_ico_anonymisation`,
`australia_privacy_act`, `china_pipl`, `india_dpdp_act`,
`africa_malabo_baseline`, `za_popia`, `ng_ndpa`, `ke_dpa`, `india_health_id`,
`eg_pdpl`, and `ma_law_09_08`.

An unknown policy fails at dbt compile time. Wildcards, SQL fragments, an empty
column list, and a string passed where an explicit list is required also fail
at compile time. Compiler errors do not echo the supplied argument, so an
accidental row value is not copied into logs or build artifacts.

The macro intentionally does not provide a `select *` or whole-relation rewrite
helper. Add every text column to the projection deliberately and keep raw text
out of seeds, fixtures, logs, and reports. This is a technical redaction
helper, not a compliance certification or clinical decision system.

## Adapter overrides

The public macro uses dbt adapter dispatch. A project can provide an adapter
implementation named `<adapter>__redact` when its warehouse uses a different
parameterized redaction function. The fallback remains:

```sql
openmed_deidentify(<column>, '<policy>')
```

Any override should preserve the same explicit-column and canonical-policy
validation contract and should keep row values inside the warehouse boundary.
