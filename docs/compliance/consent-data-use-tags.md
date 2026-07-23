# Consent and data-use tags

OpenMed can attach consent and permitted-use constraints to a pipeline input,
check the intended use before processing, and propagate the decision to the
pipeline output and audit record. Enforcement is local and deterministic; it
does not call a consent registry or transmit document content.

This layer enforces constraints supplied by the caller. It does not capture
patient consent or decide which tags apply to a dataset.

## Tags and actions

`DataUseTag` provides the built-in constraints:

| Tag | Actions denied by the default policy |
| --- | --- |
| `research-only` | `clinical-care` |
| `no-secondary-use` | `secondary-use` |
| `no-export` | `export`, `fhir-export` |
| `no-retention` | `retention` |
| `no-surrogate-vault` | `surrogate-vault` |
| `embargoed` | `secondary-use`, `export`, `fhir-export` |
| `consent-withdrawn` | Every action |

The available `DataUseAction` values are `process`, `research`,
`clinical-care`, `secondary-use`, `export`, `fhir-export`, `retention`, and
`surrogate-vault`. Providing a surrogate vault to `Pipeline.run` automatically
adds `surrogate-vault` to the attempted actions, so a caller cannot bypass that
gate by labeling the overall operation as `process`.

## Enforce tags in the pipeline

Attach tags and state the intended downstream action when the input enters the
pipeline:

```python
from openmed.compliance import DataUseAction, DataUseTag
from openmed.core.pipeline import Pipeline

result = Pipeline().run(
    "Synthetic clinical note",
    data_use_tags=[DataUseTag.RESEARCH_ONLY, DataUseTag.NO_EXPORT],
    data_use_action=DataUseAction.RESEARCH,
)

assert result.data_use_tags == ("no-export", "research-only")
assert result.audit_record["data_use"]["decision"] == "allow"
```

Canonical strings are also accepted. Input is normalized to lower-case,
hyphenated values, deduplicated, and sorted before it reaches pipeline hooks,
outputs, or audit records.

The tags and allowed decision are available in three places:

- `PipelineContext.data_use_tags`, for registered pipeline hooks
- `PipelineResult.data_use_tags` and `PipelineResult.data_use_decision`
- `PipelineResult.audit_record["data_use"]` and the nested
  de-identification result's `metadata["data_use"]`

Untagged calls keep the existing audit and nested output metadata unchanged.

## Denied actions and violation reports

Enforcement happens before normalization, detection, redaction, or surrogate
vault access. A denied action raises `DataUsePolicyViolation`:

```python
from openmed.compliance import DataUsePolicyViolation

try:
    Pipeline().run(
        "Synthetic clinical note",
        data_use_tags=["no-export"],
        data_use_action="fhir-export",
    )
except DataUsePolicyViolation as error:
    print(error.report.to_dict())
```

The report contains only the normalized tag, attempted action, decision, and
the violating tag/action pairs. It never receives or serializes input text,
subject identifiers, or raw PHI. Unknown tags and actions raise a fail-closed
`ValueError` instead of being ignored.

Example report:

```json
{
  "tags": ["no-export"],
  "attempted_actions": ["fhir-export"],
  "decision": "deny",
  "violations": [
    {
      "tag": "no-export",
      "attempted_action": "fhir-export",
      "decision": "deny"
    }
  ]
}
```

## Customize the policy

Applications can replace the default denied-action map with an explicit local
policy and attach it to a pipeline:

```python
from openmed.compliance import DataUseAction, DataUsePolicy, DataUseTag

policy = DataUsePolicy(
    denied_actions={
        DataUseTag.RESEARCH_ONLY: [DataUseAction.RETENTION],
    }
)
pipeline = Pipeline(data_use_policy=policy)
```

An explicit map replaces the default map; include every rule the application
needs. Keep the policy decision close to input ingestion, and pass the correct
action again at each export, retention, or secondary-use boundary.
