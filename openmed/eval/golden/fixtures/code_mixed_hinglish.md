# Synthetic Hinglish code-mixed routing fixture

`code_mixed_hinglish.jsonl` is synthetic-only and contains no patient data.
Token labels follow the LinCE-style routing subset used by OpenMed: `hi`, `en`,
`ne`, and `univ`; the runtime also supports `other` for unresolved tokens.

The release floor for this fixture is token-level accuracy of at least `0.80`.
The routed deterministic detector must also improve exact PHI recall over the
single-language English baseline, retain every `ne` token for downstream
Hindi and English processing, and leave zero gold entities unredacted.
