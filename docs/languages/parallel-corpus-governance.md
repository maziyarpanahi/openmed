# Parallel-corpus license and lineage gate

`ParallelCorpusShard` records content-free declarations for source provenance,
translation method, license, consent, permitted uses and derivative custody.
Call the explicit permission gate **before** the controlled training, evaluation
or export entry point. It never reads corpus text, downloads data or intercepts
unrelated training functions.

```python
from openmed.training.languages.corpus_license_gate import (
    ParallelCorpusShard, require_corpus_permission, render_corpus_use_report,
)

source, translated, method = ['sha256:' + c * 64 for c in 'abc']
shard = ParallelCorpusShard(
    shard_digest=translated, source_digest=source, license_id='MIT',
    translation_method='human', translation_evidence_digest=method,
    consent_category='synthetic', permitted_uses=('training', 'evaluation', 'export'),
    derivative_policy='redistributable', lineage_digests=(source, translated),
)
require_corpus_permission(shard, 'training')  # Raises before the caller's action.
print(render_corpus_use_report(shard, 'training'))
```

Lineage must start at the declared source and end at the shard. Human or local-
model translation needs a separate evidence digest and at least two lineage
references. Missing source, missing/unbound lineage, unknown translation,
missing license, missing explicit-consent evidence, denied/unknown consent,
unpermitted use or forbidden/unknown derivatives all deny the requested action.
Identifiers are SHA-256 references, not paths or endpoint URLs. Lineage is
ordered, unique and bounded to 32 references; permitted uses are a closed set.

License classification reuses OpenMed's existing reviewed SPDX normalization
allowlist. Unknown and non-permissive licenses stay restricted or unknown;
spelling, consent or translation never promotes their redistribution rights.
A restricted corpus needs an independent license-review digest, explicit user-
supplied custody and `user_supplied_only` derivative policy for declared training
or evaluation. Export always denies for that custody, even if `export` appears
in permitted uses. Restricted datasets are never bundled by this feature.

Reports expose only shard digest, requested use, license class, outcome and
controlled reasons. Source names, raw license labels, consent records, examples,
translation payloads, credentials, endpoints and paths are excluded. A failed
`require_corpus_permission` raises only `corpus_permission_denied`; callers can
inspect the separate content-free report for reasons.

These are declared permission and provenance metadata, not verified legal rights,
consent validity, digest/signature evidence or clinical qualification. Owners
must independently review the referenced evidence and enforce the gate at their
controlled action seams. No actual training, enrollment, networking, corpus
redistribution or autonomous clinical action is implemented.
