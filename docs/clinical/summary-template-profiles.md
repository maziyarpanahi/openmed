# Versioned summary template profiles

OpenMed's local summary boundary can use a registered summary template profile
instead of an implicit or free-form system prompt. A profile is a deterministic
typed contract: it names the bounded fields a local generator may return, the
maximum size of each field, and the source sections that may support it.

Profiles contain no source text, extracted values, credentials, model settings,
or prompt text. They are output-shape metadata only. The result remains an
assistive artifact and requires qualified clinical review; a profile does not
make a clinical decision or certify a summary.

## Select a known version

The built-in catalog currently exposes version `1.0` for four bounded forms:

| Profile | Intended shape | Required fields |
| --- | --- | --- |
| `brief_hospital_course` (`bhc`) | Short hospital-course handoff | `admission_reason`, `discharge_diagnoses`, `hospital_course` |
| `discharge_summary` | Discharge-oriented record | `admission_reason`, `discharge_diagnoses`, `hospital_course` |
| `problem_oriented` | Assessment and plan | `active_problems`, `assessment`, `plan` |
| `clinical_handoff` | Situation-background-assessment-plan handoff | `current_situation`, `background`, `assessment`, `plan` |

```python
from openmed.clinical import load_summary_profile

profile = load_summary_profile("bhc", version="1.0")
assert profile.profile_name == "brief_hospital_course"
assert profile.profile_version == "1.0"
assert profile.format == "typed_json"
assert profile.requires_clinician_review is True
print(profile.digest)  # sha256:<64 lowercase hexadecimal characters>
```

The profile digest is the SHA-256 digest of `profile.canonical_json()`. It
covers the profile name and version, field order and types, cardinality and
character limits, allowed source sections, and review guardrails. It is stable
across processes and does not contain any summary value.

## Validate typed output locally

The profile can validate a generator's JSON-shaped mapping. Validation returns
only fixed field names, reason codes, and counts. It never stores or echoes an
invalid value, which keeps a report safe to retain for local review.

```python
from openmed.clinical import validate_summary_output

report = validate_summary_output(
    profile,
    {
        "admission_reason": "synthetic admission reason",
        "discharge_diagnoses": ["synthetic diagnosis"],
        "hospital_course": "synthetic course for an offline example",
    },
)
assert report.valid
assert report.to_dict()["profile_digest"] == profile.digest
```

List-valued fields have explicit item limits and every field has a character
limit. Unknown output fields make the report invalid and are counted without
being named. The profile itself exposes the normalized definitions through
`profile.fields` and `field.field_type`; supported types are `text`,
`text_list`, `problem_list`, `medication_list`, `procedure_list`, and
`follow_up_list`.

## Local loading and version rejection

`load_summary_profile` accepts a registered name, a mapping, JSON text, or a
local `Path`. A bare name selects the current registered version; serialized
profiles must include `version` (or the compatibility alias
`profile_version`). Local JSON is duplicate-key and non-finite-number checked
before validation. URLs and remote references are rejected, so profile loading
does not introduce a mandatory network call.

The catalog is closed-world. Unknown profile names, unknown versions, altered
field definitions, free-form `system_prompt` fields, and disabled review
guardrails are rejected at load time:

```python
from openmed.clinical import load_summary_profile
from openmed.clinical.summary_profiles import UnknownSummaryProfileVersionError

try:
    load_summary_profile("bhc", version="9.0")
except UnknownSummaryProfileVersionError:
    pass
```

To compare deployments, record the profile's `profile_id` and `digest` beside
the surrounding value-free provenance. Keep source documents and generated
summary text in the operator-controlled review boundary; neither belongs in a
profile, digest, exception, or report.
