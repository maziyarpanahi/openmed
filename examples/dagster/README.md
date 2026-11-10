# Dagster de-identification asset

This example exposes a partitioned Dagster asset backed by a fully synthetic,
in-memory dataset. The asset reads the `source_dataset` resource, de-identifies
the configured text columns with `process_batch`, and records only row, cell,
span, and label counts as Dagster metadata.

Install the optional integration and start a local Dagster code location:

```bash
python -m pip install --upgrade "openmed[dagster]"
dagster dev -f examples/dagster/definitions.py
```

Materialize the `redacted_dataset` asset for partition `2026-01-01` with this
run configuration:

```yaml
ops:
  redacted_dataset:
    config:
      policy_profile: hipaa_safe_harbor
      text_columns:
        - note
```

The config accepts any canonical OpenMed policy profile. Unknown profiles are
rejected by Dagster before the asset runs. The source resource is intentionally
synthetic; replace it with an operator-managed resource in a separate code
location rather than placing source records in run configuration.
