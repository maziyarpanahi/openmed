# Independent-source pseudo-label consensus

`openmed.training.languages.label_consensus.build_label_consensus` combines local
metadata votes from `native_model`, `translated_text`, `rule` and `terminology`
channels. Each `LabelVote` contains an opaque record digest, source code-point
span, controlled label, source category, producer digest and finite confidence.
Provide local pipeline outputs; this helper runs no models or translators.

`ConsensusRule` requires two to four independent source categories and a pinned
minimum confidence. A producer reused in several categories counts only once;
several producers in one category still count as one source. Exact duplicate
votes never add evidence. Maximum matching over the four categories determines
independent support. An exact confidence-threshold boundary is inclusive.

All overlapping non-identical projections, conflicting labels and low-confidence
votes require adjudication, regardless of a majority or high-scoring teacher.
Adjacent spans do not conflict. Accepted votes require exact span/label agreement
and the full independent-source quorum. Output order is stable across input
permutations. Decisions carry offsets, opaque references, controlled label/reason
codes and counts only; they never carry source snippets or model payloads.

The caller must supply controlled label vocabularies and producer provenance.
Digest syntax and self-reported provenance do not prove source independence or
PHI safety. Do not derive exported record references directly from patient IDs.
The helper does not certify labels, change training datasets, or replace native
review. It accepts at most 10,000 votes; matching is bounded by the four source
categories. Overlap checking can be quadratic in a pathological all-overlapping
batch, so keep document batches bounded. Records and model data remain local.

```sh
.venv/bin/python -m pytest tests/unit/training/languages/test_label_consensus.py -q
```
