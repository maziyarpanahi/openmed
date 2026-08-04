# Golden De-Identification Fixtures

This directory contains synthetic-only golden fixtures for the eval suites. The
fixtures contain no DUA data, no production data, and no real PHI.

Each JSON file has a top-level `fixtures` list; JSONL files use one fixture
object per line. Each fixture uses this shape:

```json
{
  "id": "golden-multilingual-en-ssn",
  "language": "en",
  "text": "Synthetic chart lists SSN 123-45-6789 for test patient.",
  "gold_spans": [
    {
      "start": 26,
      "end": 37,
      "label": "SSN",
      "text": "123-45-6789"
    }
  ],
  "metadata": {
    "category": "multilingual",
    "synthetic": true,
    "expected_output": {
      "method": "mask",
      "text": "Synthetic chart lists SSN [SSN] for test patient."
    }
  }
}
```

Required fields:

- `text`: source fixture text.
- `gold_spans`: canonical-label spans with character offsets into `text`.
  Synthetic hard-negative fixtures use an empty list because the confusable
  tokens are non-PHI and must remain unredacted.
- `metadata.category`: one of `nested_overlapping`, `chunk_boundary`,
  `multilingual`, `checksum_ids`, `financial_ids`, `date_arithmetic`,
  `policy_profile_actions`, `hard_negatives`, or `critical_findings`.
  The standalone `indic_name_variants` fixture uses its dedicated consistency
  suite schema because it groups multiple spellings under one synthetic
  identity rather than representing detector spans.
- `metadata.expected_output`: expected post-action output, including `method`
  and resulting `text`.
- `metadata.synthetic`: must be `true`.
- `metadata.hard_negative_candidates`: required only for `hard_negatives`;
  each candidate records canonical label, offsets, synthetic marker, and
  aggregate difficulty scores.
- `metadata.medical_device_disclaimer`: required only for `critical_findings`;
  it must note that the synthetic set is an assistive safety probe, not
  clinical ground truth.

The package loader validates unique fixture IDs, offsets, canonical labels,
synthetic markers, expected output, hard-negative candidate metadata,
critical-finding disclaimers, and language coverage. The JSON and JSONL files
are also compatible with `openmed.eval.harness.load_fixtures`; golden-specific
expected output remains available through each fixture's metadata.

## Temporal TLINK Fixtures

`fixtures/temporal_tlinks.jsonl` contains hand-authored, synthetic-only
discharge-summary-style notes for temporal graph evaluation. Each row declares
EVENT/TIMEX offsets, normalized time values, strict TLINK gold, its reduced
graph reference, scored decoder candidates, and contradiction-trap candidate
ids. DCT anchoring, EVENT-TIMEX links, EVENT-EVENT ordering, transitive
reduction, and weaker reverse-order traps are all represented.

Load and validate this specialized schema with
`openmed.eval.suites.temporal_tlinks.load_temporal_tlink_fixtures`. The loader
requires `metadata.synthetic=true` and `metadata.contains_real_phi=false`,
checks all offsets and references, and rejects inconsistent reduced gold. The
suite reports aggregate relation counts and PHI-safe reason codes only; no note
text or graph node ids appear in its gate artifacts.

## India Clinical De-Identification Corpus

`fixtures/i18n/india_clinical_manifest.json` and
`fixtures/i18n/india_clinical.jsonl` form a specialized synthetic-only corpus
for India clinical de-identification evaluation. The three code-mixed notes
cover Latin-script Hinglish, Devanagari, and Tamil; generated ABHA, Aadhaar,
PAN, Indian phone, address, and PIN values; AYUSH terminology; and one
fictional person represented by script-specific aliases across documents.

Load the corpus with
`openmed.eval.datasets.load_india_clinical_phi_corpus`. Its dedicated loader
validates the safety manifest, exact span offsets, canonical labels, generated
identifier shapes, address/PIN pairing, script coverage, and cross-document
identity metadata. The JSONL file is intentionally excluded from the generic
golden loader because its richer corpus schema is validated separately.

The corpus contains no real PHI, production data, restricted corpus material,
or DUA data. It is an assist-only, non-decisional evaluation fixture, not
clinical ground truth, and must not be used to make patient-care decisions.

## Joint Entity and Relation Fixtures

`fixtures/joint_entity_relation.jsonl` contains synthetic encoder states,
token-to-character offsets, entity spans, typed relations, and explicit
boundary, over-generation, and distractor traps for the backend-neutral joint
span-pair head. Every row sets `metadata.synthetic=true` and
`metadata.contains_real_phi=false`.

The specialized fixture is intentionally excluded from the generic
de-identification loader. Its states are small deterministic vectors used to
measure combined entity-and-relation micro-F1, endpoint-confidence suppression,
span-graph schema integrity, and false-positive relations on negative pairs.

## Relation Gold Fixtures

`fixtures/relation_gold.jsonl` contains synthetic-only relation extraction
fixtures for `openmed.eval.suites.relations`. Each JSONL row uses schema
version `1` and this shape:

```json
{
  "id": "relation-sentence-treatment",
  "schema_version": 1,
  "language": "en",
  "text": "Aspirin treats fever in note one.",
  "entities": [
    {
      "id": "e-medication",
      "start": 0,
      "end": 7,
      "label": "MEDICATION",
      "text": "Aspirin"
    }
  ],
  "relations": [
    {
      "id": "rel-aspirin-fever",
      "type": "treats",
      "head": "e-medication",
      "tail": "e-fever",
      "scope": "sentence"
    }
  ],
  "traps": [
    {
      "id": "trap-assertion-negation",
      "kind": "assertion",
      "relation_ids": ["rel-pneumonia-denied"],
      "zero_tolerance": true
    }
  ],
  "metadata": {
    "synthetic": true,
    "category": "relation_gold",
    "schema_version": 1
  }
}
```

Required fields:

- `entities`: canonical-label spans with unique ids and character offsets into
  `text`. Relation arguments reference these ids.
- `relations`: directed relation records with unique ids, `type`, `head`,
  `tail`, and a `scope` of `sentence` or `document`.
- `traps`: optional zero-tolerance assertion or temporal traps that reference
  relation ids and are carried into the score payload for release-gate wiring.
- `metadata.synthetic`: must be `true`; relation gold must not contain DUA,
  production, or real patient data.

The relation loader validates schema version, unique fixture/entity/relation
ids, argument references, offsets, canonical entity labels, relation scopes,
and trap metadata.

`fixtures/i18n/relations_zh.jsonl` and
`fixtures/i18n/relations_indic.jsonl` extend that schema with synthetic Chinese
and Hindi relation examples. They reuse canonical NER labels, carry registry
version `1`, and are scored separately through the relation metric's
`by_language`/`per_language` payloads. No CMeIE or other external corpus text is
bundled.

## Radiology Entity-and-Relation Fixtures

`fixtures/radiology_entity_relations.jsonl` contains hand-authored synthetic
radiology reports with `OBSERVATION` and `ANATOMY` spans, present/absent/uncertain
finding labels, and `LOCATED_AT`, `MODIFY`, and `SUGGESTIVE_OF` relations. The
dedicated loader rejects any committed row that is not explicitly marked
synthetic or lacks the required not-a-medical-device disclaimer. These rows are
offline evaluation probes for clinician-reviewed systems, not clinical ground
truth, patient-care guidance, RadGraph/MIMIC-CXR content, or other DUA data.
