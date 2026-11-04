# Finalized Transcript Language Routing

`TranscriptLanguageRouter` combines a fixed provider language hypothesis with
confident text-side token identification **before** any installed PHI detector
is called. Python and OpenMedKit expose the same decisions, reason codes,
confidence buckets and Unicode-scalar offsets. This is an independent local
language gate, not an ASR engine, translation service or language-pack installer.

## Inputs and decisions

Supply the actually installed `LanguagePack` metadata and one offline detector
per primary-language code. A pack listed in the catalog is not installed evidence.
Also supply an on-device `LanguageIdentifier` and its candidate languages,
including languages without installed PHI packs. No detector, language model or
network transport is loaded implicitly. Callbacks are trusted application code
and must run locally; do not wrap cloud services in them.

Python reuses the token boundaries from `lang_id_codemix.py`, ignoring its
Hinglish heuristic labels as release evidence. The injected identifier must
confidently identify **every letter-bearing token**, including names. Missing
evidence on even one token withholds the whole segment. There is no English or
pack-priority fallback. Token confidence and provider confidence must each meet
the configured threshold (default `0.8`, inclusive).

The provider hypothesis describes the segment's **dominant spoken language**.
Text dominance is the greatest count of letter-bearing tokens per primary
language. The provider must match a text-dominant language; equal-count ties
accept either tied language. Other confidently identified token languages form
mixed runs. A provider matching only a minority language is disagreement and
fails closed. This conservative policy is not a calibrated language classifier.

| Status | Behavior |
| --- | --- |
| `supported` | One confidently identified primary language has an installed detector. |
| `mixed` | Multiple confidently identified primary languages each have an installed detector. |
| `unsupported` | At least one identified run has no installed PHI pack; the entire segment is withheld. |
| `uncertain` | Missing/low evidence, disagreement, partial input or a failed detector; the entire segment is withheld. |

Tags use the bounded BCP 47 form `language[-Script][-REGION]`: a two-letter
language, optional four-letter script, and optional two-letter or three-digit
region. Case is canonicalized (`EN-us` → `en-US`); private-use extensions and
arbitrary labels are rejected with `invalid_language_tag`. Script subtags use
a fixed vocabulary of common Unicode scripts; arbitrary four-letter payloads
are rejected. Pack lookup and
provider/text agreement use the primary language. This gate does not qualify
script- or region-specific detector recall. Only supply a pack detector capable
of processing the scripts/regions your application enables.

## Routing and protected output

Adjacent same-tag lexical tokens form runs; language changes split at the next
lexical token's start. Leading punctuation/numbers stay with the first run.
Intervening neutral tokens, whitespace, numbers and trailing punctuation stay
with the preceding run. Thus runs tile **all** input characters without gaps,
normalization or offset changes. A detector receives the original substring and
returns local `TranscriptPHISpan` offsets; the router translates them back to
the original segment. Overlapping/adjacent PHI spans are merged and masked with
one `█` per source code point. Detector failure or invalid output withholds
everything, including results of any earlier successful detector call.

Detectors must handle all identifiers and neutral tokens in their run. This
routing gate alone does not establish direct-identifier recall, safe treatment
of cross-language identifiers or clinical validation. Unknown token languages
must abstain rather than guess. Existing consent, streaming withholding and
privacy qualification gates still apply at integration time.

Use `decision.reviewed_text(reviewer_confirmed=True)` for both release and
draft intake, after explicitly reviewing that segment's processed output.
Without confirmation it raises `review_required`; a withheld segment always
raises `segment_withheld`, even with confirmation. The decision binds the notice
`Non-diagnostic transcript; explicit reviewer confirmation required.`
No clinical action is triggered.

```python
from openmed.core.language_pack_catalog import LANGUAGE_PACK_ADAPTERS
from openmed.core.language_router import LanguagePrediction
from openmed.multimodal.transcript_language import (
    TranscriptLanguageRouter, TranscriptPHISpan, transcript_language_report,
)

# Synthetic fixed evidence only; these callbacks are not qualified models.
class SyntheticLocalIdentifier:
    name = "synthetic-local"

    def identify(self, text, candidates):
        labels = {"Patient": "en", "Ada": "en", "Paciente": "es", "Lucía": "es"}
        code = labels.get(text)
        return None if code is None else LanguagePrediction(code, 0.99)

def synthetic_detector(text):
    return [
        TranscriptPHISpan(text.index(name), text.index(name) + len(name))
        for name in ("Ada", "Lucía") if name in text
    ]

router = TranscriptLanguageRouter(
    installed_packs=[LANGUAGE_PACK_ADAPTERS.registry.get(code) for code in ("en", "es")],
    detectors={"en": synthetic_detector, "es": synthetic_detector},
    text_identifier=SyntheticLocalIdentifier(),
    candidate_languages=("en", "es", "fr"),
)
decision = router.route(
    segment_index=0, text="Patient Ada. Paciente Lucía.",
    provider_language=LanguagePrediction("en", 0.99), finalized=True,
)
assert decision.reviewed_text(reviewer_confirmed=True) == "Patient ███. Paciente █████."
print(transcript_language_report([decision]))  # counts, never transcript text
```

When the pending offline ASR contract is integrated, map its finalized segment
index/text and optional `language.language`/`language.confidence_ppm` to these
arguments, converting parts per million to `[0, 1]`. This adapter owns no timing,
revision, consent or buffer contract and does not copy the pending ASR types.

OpenMedKit provides `TranscriptLanguageHypothesis`, `TranscriptPHISpan`,
`TranscriptLanguageRouter(installedPackCodes:detectors:candidateLanguages:
confidenceThreshold:textIdentifier:)`, `route(segmentIndex:text:providerLanguage:
finalized:)`, and `reviewedText(reviewerConfirmed:)`. All detector offsets count
Unicode scalars, **not** UTF-16 code units or Swift `Character` graphemes.
Local NER adapters must convert their coordinates before returning spans.
`decision.audit` is the content-free Codable diagnostic surface;
`transcriptLanguageReport` supplies aggregate counts.

## Diagnostics and reason codes

`to_dict()` contains only the numeric segment index, status, controlled reason,
provider tag/bucket, run tags/buckets/offsets and PHI-span count. Aggregate reports
contain only `status_counts`, `reason_counts`, `tag_counts` (text runs),
`provider_tag_counts` and `confidence_bucket_counts` (provider hypotheses).
Buckets are `missing`, `low` (below threshold), `accepted` (passes threshold
but below `0.9`) and `high` (passes threshold and at least `0.9`).
Reports and ordinary decision descriptions exclude transcript text and identifier
values. Do not serialize Python dataclass internals, reflect private Swift
storage, or log protected output. Callback exception messages are discarded.

| Reason | Result |
| --- | --- |
| `language_routed` | Detector-processed output awaits explicit confirmation. |
| `segment_not_finalized` | Partial transcripts are withheld without LID/detector calls. |
| `provider_language_missing` | Provider evidence is absent. |
| `provider_confidence_low` | Provider confidence is below threshold. |
| `text_language_uncertain` | Text is empty/neutral-only or any lexical token lacks evidence. |
| `text_confidence_low` | A lexical token is below threshold. |
| `language_disagreement` | Provider and dominant text primary languages disagree. |
| `phi_pack_unavailable` | One or more text languages have no installed detector. |
| `text_identifier_failed` | Local LID failed or returned invalid evidence. |
| `detector_result_invalid` | A detector returned the wrong type or out-of-run offsets. |
| `detector_failed` | A local detector failed; no partial output is usable. |

Synthetic tests cover English, Spanish, code switches, non-BMP prefixes,
multilingual names/IDs, confidence boundaries, absent packs, disagreements,
backend failures and deterministic reports. They measure this contract only:
no model accuracy, provider qualification, translation or device benchmark claim.
