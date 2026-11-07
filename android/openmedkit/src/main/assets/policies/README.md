# OpenMedKit Policy Profiles

This directory holds the portable OpenMed de-identification policy resources
for the Kotlin OpenMedKit facade. They are generated from
`openmed/core/policies/` by `scripts/parity/generate_portable_policies.py` and
are byte-identical to the Swift and TypeScript copies. Do not edit them by
hand: change the Python source profiles and run
`python scripts/parity/generate_portable_policies.py --write`.

- `portable_policy_manifest.json` pins the digest of every profile and of the
  taxonomy, the profile aliases, and the Python-only profiles that are excluded.
- `portable_policy_taxonomy.json` is the shared label taxonomy and alias table.
- One file per portable profile: `australia_privacy_act`, `canada_pipeda`,
  `clinical_minimal_redaction`, `gdpr_art9_health`, `gdpr_pseudonymization`,
  `hipaa_expert_review_assist`, `hipaa_safe_harbor`,
  `research_limited_dataset`, and `strict_no_leak`.

When `OpenMedKit.deidentify(...)` is called without an explicit policy, it uses
`hipaa_safe_harbor`. See `docs/swift-kotlin-parity.md` for the portable policy
contract.
