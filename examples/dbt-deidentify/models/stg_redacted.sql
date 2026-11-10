{{ config(materialized='view') }}

-- Staging model: apply OpenMed de-identification to the configured free-text
-- columns. Requires the `openmed_deidentify(text, policy)` UDF to be registered
-- on the warehouse connection (see the package README). Structured columns pass
-- through unchanged; only the listed text columns are redacted.

with source as (
    select * from {{ ref('synthetic_patients') }}
)

select
    patient_id,
    {{ redact('notes') }} as notes,
    {{ redact('reason_for_visit') }} as reason_for_visit
from source
