# Clinical language readiness

The clinical readiness registry is independent of installed locale preprocessing
packs. Being able to route a locale does not establish clinical capability.

`openmed.clinical.languages.readiness` provides immutable `LanguageEvidence`,
`ClinicalLanguageReadiness`, and a private `ClinicalLanguageReadinessRegistry`.
Evidence contains controlled kind/task/tier codes and SHA-256 references only.
Do not put reviewer names, patient examples, model paths or free text into it.

For each declared task, **supported** requires lexicon, fixture, evaluation and
native-review digests, an approved reviewer status, and device evidence for every
declared model tier. Script and task inventories must be nonempty. Missing
foundational evidence yields **experimental**. Foundational evidence with missing
device evidence, no tier inventory, or declared limitations yields **preview**.
Limitations always cap a complete entry at preview. Digests are references, not
proof that a study was scientifically valid. A supported declaration makes no
certification, diagnostic, or autonomous clinical decision claim.

```python
from openmed.clinical.languages.readiness import (
    ClinicalLanguageReadiness, ClinicalLanguageReadinessRegistry,
)

row = ClinicalLanguageReadiness(
    language="ne", version="1.0.0", scripts=("Devanagari",), tasks=("entities",),
)
assert row.state == "experimental"
report = ClinicalLanguageReadinessRegistry([row]).report()
assert report["languages"][0]["state"] == "experimental"
```

Language underscore aliases normalize and duplicate canonical entries reject.
Capabilities are sorted, evidence is bounded and task/tier bindings validate.
Reports are deterministic and require no model, network, weights or real text.
There is no process-global registration or implicit promotion of an installed
pack. Native reviewers should examine referenced evidence and known limitations;
this inventory does not replace that review.
