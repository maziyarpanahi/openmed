# OpenMed synthetic clinical-PHI eval corpus

This card describes the public `openmed-synth` benchmark suite. The corpus is
generated locally by
[`scripts/eval/build_openmed_synth_corpus.py`](https://github.com/maziyarpanahi/openmed/blob/master/scripts/eval/build_openmed_synth_corpus.py)
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
| Generation method | Seeded Faker locales plus OpenMed `clinical_ids.py` providers; dates and locale-shaped postcodes use stable seeded integer draws; gold offsets and mask outputs are computed from the rendered segments |
| Default seed | 2352 |
| License | Apache-2.0 |
| Content hash | sha256:56f7bab49f18143fb4acedaac701eab9b3d43856b79b634a0f4d330285d4fb54 |
| Faker version | 40.15.0 |
| Provider probe digest | sha256:f6aa2b0086133cad85c6939e0b66b18469be38524cafee59fe1755964fdaad75 |

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

## Smoke tests versus model evaluation

`openmed benchmark pii --suite openmed-synth` deliberately returns gold spans
as a fixture-integrity smoke test. Its report sets `uses_gold_reference: true`
and `evaluation_kind: fixture_smoke_only`. A perfect smoke score is not model
performance. The explicit `openmed-synth-reference` alias behaves identically.

Passing a different model name uses the regular local model runner, not gold
spans; cache the model assets before offline use. An injected Python runner is
labeled `caller_supplied_runner`, since the harness cannot verify whether it
actually executes a model. All modes set `clinical_validation: false`.

Generation is bounded to 1–10,000 records and a signed 64-bit seed. The pinned
hash above was verified with Faker 40.15.0 from `uv.lock`; reproducibility across
arbitrary Faker versions is not promised. Use the frozen environment when
reproducing this receipt. Faker values are invented, but accidental resemblance
to real people or addresses is possible; they are not anonymized patient data.

## Provider versions and drift

A fixed seed reproduces this corpus only while its surrogate providers stay the
same. The published provider receipt,
`openmed/eval/data/openmed_synth_provider_receipt.json`, records the corpus
identity and content hash together with:

- the Faker distribution version (`providers.faker`);
- the fixed provider probe set (`probe`: seed, draws, and method names) and
  its aggregate digest, shown in the table above;
- for each corpus locale, the Faker backend locale, one probe digest per
  provider component (`faker` for Faker's own providers,
  `openmed.clinical_ids` for repository clinical-ID providers), the digest of
  the repository-authored `openmed.synth_corpus` table (note template,
  date-of-birth range, and postcode rule), and the digest of that locale's
  generated rows.

Every entry is a version string, locale code, method name, or `sha256:`
digest. Probe and generated values are hashed and never stored.

`openmed_synth_suite_metadata()` and every `openmed-synth` benchmark report
carry the regenerated manifest as `surrogate_providers` and its comparison
with the receipt as `provider_drift`. When any version, probe, table, or
output digest differs, `provider_drift.status` is `drift`, `reasons` is
`["provider_version_drift"]`, and each finding names the component, locale,
and aspect that changed, such as `faker:version`,
`faker[fr_FR]:probe_digest`, or `corpus[fr_FR]:output_digest`. Output digests
and the content hash are compared only for the published seed and size; other
seeds still compare provider versions and probe digests. When only a provider
version changed, `content_hash_match` stays `true`: the corpus bytes still
match, but the environment is not the locked one.

`openmed benchmark pii --suite openmed-synth` keeps running on drift and
prints a `provider_version_drift` warning to stderr. Add `--strict` to exit
with status 1 and the `provider_version_drift` error code instead, before any
model runs:

```bash
openmed benchmark pii --suite openmed-synth --strict
```

`LocalePhiGenerator` and `BurnedInGenerator` examples record a compact
`surrogate_providers` stamp in their metadata: the schema version, Faker
version, requested and backend locales, and the aggregate probe digest. The
stamp holds no generated values. Probe digests are cached per process and Faker
version; call `clear_provider_probe_cache()` from
`openmed.training.synthetic.provider_versions` after patching providers at
runtime.

### Reproducing the published corpus

Install the locked dependency set and check the corpus and its providers
against the receipt:

```bash
uv sync --frozen --extra dev
uv run python scripts/eval/build_openmed_synth_corpus.py --hash-only
uv run openmed benchmark pii --suite openmed-synth --strict
```

The hash-only command must print the content hash above, and the strict
benchmark must exit 0. After an intentional Faker upgrade in `uv.lock` or a corpus
change, regenerate the receipt with
`python scripts/eval/build_openmed_synth_corpus.py --provider-manifest` and
update this card's content hash, Faker version, and probe digest in the same
change.
