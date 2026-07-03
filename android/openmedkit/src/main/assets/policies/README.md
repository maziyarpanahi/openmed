# OpenMedKit Policy Profiles

This directory packages the six OM-031a de-identification policy profiles for
the Kotlin OpenMedKit facade:

- `hipaa_safe_harbor`
- `hipaa_expert_review_assist`
- `gdpr_pseudonymization`
- `research_limited_dataset`
- `strict_no_leak`
- `clinical_minimal_redaction`

When `OpenMedKit.deidentify(...)` is called without an explicit policy, it uses
`hipaa_safe_harbor`.
