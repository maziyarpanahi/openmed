# Synthetic prior-authorization packets

`openmed.training.synthetic.prior_auth` assembles deterministic synthetic notes,
orders and forms sharing one opaque case identity and controlled diagnosis and
treatment placeholders. It bundles no payer policy, insurer vocabulary, actual
records or coverage decision service.

Declare `AuthorizationCriterion(code, document_kinds, required=True)` and synthetic
`AuthorizationEvidence(document_kind, code, present, distractor=False)` items inside
an `AuthorizationCaseSpec`. A criterion only uses evidence from allowed document
kinds. Distractors remain labeled in document truth and never satisfy a criterion.
Missing evidence remains missing. Conflicting true/false observations produce
`conflicting`, never an arbitrary winner. Required missing/conflicting criteria
produce `abstain`; explicit false evidence yields `requirements_not_satisfied`;
complete positive evidence yields `ready_for_review`, not authorization approval.

`generate_authorization_packet(spec, seed=...)` produces byte-stable template text,
exact source spans, traceable evidence references and gold statuses. Seeded opaque
identities are stable for a declared recipe. `validate_authorization_packet`
independently checks shared identity and placeholders, gold spans, references and
review state. Inputs are bounded to 128 criteria and 1,024 evidence items; invalid
or duplicate declarations reject with controlled errors. All fixture text is
synthetically rendered from controlled codes; do not pass patient identifiers.
No network calls, patient data, restricted terminology or autonomous clinical /
payer decisions are involved. These templates test evidence-gap handling rather
than medical plausibility or real-world prior-authorization success rates.
