# Indic NER checkpoint compatibility

OpenMed's optional Indic NER adapter accepts only a user-selected Hugging Face
token-classification checkpoint. It does not bundle weights, select a public
repository, or contact a model host when `OPENMED_INDIC_NER_MODEL` is unset.

Install the optional runtime before loading a checkpoint:

```bash
pip install "openmed[hf]" torch
```

Then use an existing local directory or an explicitly approved repository:

```python
from openmed.ner.families import load_indic_ner_adapter

local_adapter = load_indic_ner_adapter(
    "/srv/models/indic-ner",
    local_files_only=True,
)

remote_adapter = load_indic_ner_adapter(
    "your-org/approved-indic-ner",
    revision="reviewed-revision",
)
```

Existing paths are forced into local-only loading even if
`local_files_only=False`. Repository resolution is possible only when a caller
passes a repository identifier directly or sets `OPENMED_INDIC_NER_MODEL`.
Remote code remains disabled in both layouts.

## Compatibility matrix

| Checkpoint contract | Status | Notes |
| --- | --- | --- |
| `id2label` with integer keys | Supported | Keys must be contiguous from zero. |
| `id2label` with numeric string keys | Supported | Numeric keys are normalized to integers. |
| `label2id` with integer values | Supported | Used when a compatible `id2label` is absent. |
| Matching `id2label` and `label2id` | Supported | Conflicting compatible maps are rejected. |
| BIO, BIOES, or BILOU prefixes | Supported | `PER`/`PERSON`, `LOC`/`LOCATION`, and `ORG`/`ORGANIZATION` map to canonical OpenMed labels. |
| Unprefixed PER/LOC/ORG labels | Supported | Each token is treated as a singleton entity. |
| Fast-tokenizer subword offsets | Supported | Contiguous subwords merge according to their entity tags. |
| Local Hugging Face directory | Supported | Always loaded with local-files-only behavior. |
| Explicit Hugging Face repository | Supported, opt-in | May resolve remotely only after explicit configuration. Pin `revision` in governed deployments. |

A compatible map must contain an outside `O` label and semantic labels for all
three CoNLL entity classes: PER, LOC, and ORG. The map must match
`config.num_labels` when that value is present, and each model logit row must
have the same width.

## Unsupported contracts

The adapter fails closed for:

- slow tokenizers or tokenizers without exact character offsets;
- negative, reversed, out-of-bounds, overlapping, or backward offsets;
- generic `LABEL_0`-style maps without entity semantics;
- missing PER, LOC, or ORG labels, duplicate/non-contiguous indices, and
  conflicting label maps;
- CRF or custom heads that do not return token-aligned `logits`;
- checkpoints that require `trust_remote_code=True`;
- custom preprocessing whose offsets do not refer to the original Python
  string.

Compatibility exceptions contain only stable contract descriptions. They do
not include the input string, checkpoint error detail, token surfaces, or
credentials. Predictions serialize only offsets, canonical labels, and
confidence values.

## Synthetic robustness release gate

The committed Naamapadam-style evaluation uses synthetic records only. For
each of the 11 supported Indic languages, it retains one acceptance baseline
and adds deterministic cases for combining marks, punctuation-adjacent spans,
repeated entity surfaces, code mixing, and Latin/native script boundaries.
No Naamapadam corpus row or model weight is bundled.

A checkpoint is release-eligible only when all of these thresholds pass:

| Gate | Threshold |
| --- | ---: |
| Exact-span recall for every language, across the full synthetic suite | at least `0.80` |
| Exact-span recall for every language baseline | at least `0.80` |
| Exact-span recall for every language and robustness slice | at least `0.80` |
| Recall drop from a language baseline to each robustness slice | at most `0.10` |
| Surviving PER/LOC/ORG entity surfaces after masking | `0` |

In the aggregate report, `recall_delta` means baseline recall minus slice
recall, so a positive value is a regression. Failure summaries contain only
counts and SHA-256 fixture hashes. They never serialize fixture text, entity
surfaces, or model inputs. The suite remains offline when a predictor is
provided and returns a structured skip reason when optional weights are not
configured.

## Opt-in compatibility smoke tests

The real-checkpoint tests are skipped by default with the environment variable
needed to enable each layout:

```bash
OPENMED_INDIC_NER_COMPAT_LOCAL_MODEL=/srv/models/indic-ner \
  pytest tests/integration/test_indic_ner_checkpoint_compatibility.py::test_user_supplied_local_checkpoint_contract -q

OPENMED_INDIC_NER_COMPAT_REMOTE_MODEL=your-org/approved-indic-ner \
  pytest tests/integration/test_indic_ner_checkpoint_compatibility.py::test_explicit_remote_checkpoint_contract -q
```

An inaccessible configured checkpoint skips with an aggregate reason. A
checkpoint that loads but violates the label, offset, logits, or privacy
contract fails the test.
