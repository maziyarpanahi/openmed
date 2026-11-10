# Morphology-aware label projection

Labels projected from a high-resource language often land on a target word
whose surface was changed by case endings, clitics, agglutinative suffix
chains, noun-class prefixes or stem alternations. Finnish `diabetes` appears
as `Diabeteksen`, Turkish `kalp` as `kalbinde`, Swahili `ugonjwa` as
`magonjwa`, and Tamil `மருந்து` as `மருந்தை`. Plain string matching misses
these words, picks the wrong occurrence, or emits a span that cuts through an
affix or a grapheme cluster.

`openmed.training.languages.morphology_projection.project_labels` matches each
projected base form (lemma) against the target text through a caller-supplied
`MorphologyProfile`. It emits a training label only when exactly one
grapheme-safe span remains, and routes everything else to review with a
controlled reason code.

```python
from openmed.training.languages.morphology_projection import (
    LabelProjection,
    MorphologyProfile,
    project_labels,
)

profile = MorphologyProfile.from_dict(
    {
        "language": "fi",
        "affixes": [
            {"surface": "n", "position": "suffix", "kind": "inflection"},
            {"surface": "ssa", "position": "suffix", "kind": "inflection"},
            {"surface": "kin", "position": "suffix", "kind": "clitic"},
        ],
        "alternations": [{"lemma_edge": "s", "stem_edge": "kse"}],
    }
)
text = "Diabeteksen hoito jatkuu."
result = project_labels(text, [LabelProjection("DISEASE", "diabetes")], profile)
decision = result.decisions[0]
assert text[decision.start : decision.end] == "Diabeteksen"
```

## Profiles

A profile is configuration, not linguistic authority. List only the
inflectional affixes and clitics that a native reviewer approved for label
projection; derivational affixes (for example Turkish privative `-siz`,
"without") change meaning and must stay out so the projection is flagged.

| Field | Meaning |
| --- | --- |
| `language` | BCP 47 tag, validated offline. |
| `affixes` | `surface`, `position` (`prefix` or `suffix`) and `kind` (`inflection` or `clitic`). Clitics are always outermost. |
| `alternations` | `lemma_edge` rewritten to `stem_edge` at the `suffix` or `prefix` edge, such as Turkish `p` to `b` or Swahili prefix `u` to nothing. An alternation applies only when an affix was removed on that side. |
| `lexicalized_forms` | Whole words that an affix boundary may never split, such as Amharic `በሽታ` ("disease"), whose first syllable looks like the clitic `በ`. |
| `casing` | `unicode` full case folding, `turkic` dotted and dotless I folding, or `auto` (Turkic for `tr` and `az`). |
| `max_affixes` | Affixes removed per token, `0` to `8` (default `4`). |
| `min_stem_graphemes` | Graphemes that must remain after removing affixes (default `2`). |

Rules are deduplicated and sorted, so `profile.digest` is a stable SHA-256 of
the canonical configuration and can be recorded as provenance.

## Span policies

| Policy | Label covers |
| --- | --- |
| `token` | Whole matched tokens, affixes and clitics included. |
| `host` (default) | Tokens without outer clitics; inflection stays. |
| `stem` | Only the material left after removing every affix. |

Multi-word lemmas such as Amharic `ስኳር በሽታ` must match consecutive tokens
with the same spacing. Shared tokenization keeps apostrophe-separated Turkish
suffixes out of the label, so `EKG'de` labels `EKG`.

## Decisions and review reasons

| Status | Reason | Meaning |
| --- | --- | --- |
| `projected` | `exact_match`, `affix_match`, `alternation_match` | One unambiguous, grapheme-safe span. |
| `needs_review` | `multiple_matches` | Several locations match, such as Swahili `Mgonjwa` and `magonjwa` for `ugonjwa`. Pass the aligner's `window` to narrow them. |
| `needs_review` | `ambiguous_segmentation` | One location yields different spans under the chosen policy. |
| `needs_review` | `grapheme_bisection` | A label edge would cut a grapheme cluster or a case-folding expansion, such as the Tamil stem of `மருந்தை` under the `stem` policy. |
| `needs_review` | `unanalyzed_affix` | The lemma occurs only inside a token with unconfigured material, such as a compound or derivation. |
| `needs_review` | `overlapping_projection` | Projected spans overlap with a different extent or label. Identical duplicates stay projected. |
| `unmatched` | `no_match` | No candidate in the text or window. |

Review decisions list candidate code-point spans that never bisect a grapheme
cluster. Export only `result.projected` as training labels.

## Offsets

Projected labels follow the cross-runtime
[offset contract](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/decoding/OFFSET_CONTRACT.md):
`start` and `end` are half-open Unicode code-point offsets into the exact,
unnormalized source. Each decision also carries `grapheme_start` and
`grapheme_end` (extended-grapheme indices) and `byte_start` and `byte_end`
(UTF-8 byte offsets) for the same span. Matching runs on NFC-normalized,
case-folded text, and every boundary maps back through the shared offset
maps, so decomposed accents and multi-byte scripts keep exact offsets.

## Privacy and limits

Projection is deterministic, local and stdlib-only. It runs no models, makes
no network calls and never logs. `LabelProjection` hides its lemma from
`repr`; decisions and `result.to_dict()` carry offsets, label and reason
codes, counts and the profile digest only, never source text or lemmas.
Validation errors never echo input values.

The helper does not translate, align or certify labels, and it is not a
clinical decision aid. A batch accepts at most 10,000 projections; lemmas are
limited to 256 code points and 16 tokens. Configure each profile with a
native reviewer and keep review decisions out of training data until they are
resolved.

```sh
.venv/bin/python -m pytest tests/unit/training/languages/test_morphology_projection.py -q
```
