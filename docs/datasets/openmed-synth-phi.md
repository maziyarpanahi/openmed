# OpenMed synthetic clinical-PHI eval corpus

This card describes the public `openmed-synth` benchmark suite. The corpus is
generated locally by
[`scripts/eval/build_openmed_synth_corpus.py`](../../scripts/eval/build_openmed_synth_corpus.py)
and loaded through the OpenMed golden-fixture machinery. No network access or
credential is needed.

| Field | Value |
| --- | --- |
| Dataset | `openmed-synth` |
| Version | 1.0.0 |
| Record count | 14 |
| Labels | `DATE_OF_BIRTH`, `EMAIL`, `ID_NUM`, `LOCATION`, `PERSON`, `PHONE`, `STREET_ADDRESS`, `ZIPCODE` |
| Label distribution | 14 spans per label; 112 spans total |
| Languages | `de`, `en`, `es`, `fr`, `hi`, `pt`, `zh` |
| Generation method | Seeded Faker locales plus OpenMed `clinical_ids.py` providers; gold offsets and mask outputs are computed from the rendered segments |
| Default seed | 2352 |
| License | Apache-2.0 |
| Content hash | sha256:a41c4502a28029b717eb352804e7b69b80afa2ede0204d67b2797b5a66eb9b87 |

The content hash is the SHA-256 digest of the canonical JSONL emitted for the
default seed and size. Re-running the generator with the same inputs must
produce the same bytes and hash; changing the seed changes the corpus hash.
Each row contains source text, canonical gold spans, and an expected `mask`
post-action output. Spans are validated through `GoldenFixture` before the
suite exposes them to the benchmark harness.

All names, dates, contact details, addresses, locations, email addresses, and
medical-record identifiers are generated synthetic values. This corpus
contains no real PHI, production records, or DUA-gated data, and it is not
clinical ground truth. It is an assistive de-identification evaluation
fixture, not a clinical decision-making or patient-care tool.
