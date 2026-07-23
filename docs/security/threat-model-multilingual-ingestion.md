# Multilingual ingestion threat model

## Overview

OpenMed accepts operator-supplied dictionaries and multilingual byte streams at
the boundary between local files and PHI-sensitive text processing. Those inputs
are data, never code. A malformed source must fail closed before unbounded
decompression, parsing, codec dispatch, or regular-expression evaluation, and a
rejection must not copy dictionary content or a raw path into logs.

This model is deliberately narrower than the repository-wide
[redactor threat model](threat-model.md). It covers segmenter dictionaries, user
dictionaries, legacy-encoding conversion, Unicode script/confusable review, and
the locale format and PII-pattern tables in
[`locale_formats.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/locale_formats.py) and
[`pii_i18n.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/pii_i18n.py). Network services, model-file
integrity, and dependency/SBOM controls remain out of scope.

The hardening design consolidates untrusted dictionary handling in
[`tokenization.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/processing/tokenization.py) and strict byte
decoding in [`script_detect.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/script_detect.py). Later
segmenter and legacy-font adapters should call these boundaries instead of
opening files or resolving codecs directly.

## Threat Model, Trust Boundaries, and Assumptions

### Assets

| Asset | Security property |
| --- | --- |
| Segmenter and user-dictionary files | Entries remain literal, bounded data; a file cannot select executable regex behavior or exhaust memory/CPU. |
| Compressed dictionary members | Archive metadata is verified and decompression stays inside explicit byte and expansion budgets. |
| Legacy-encoded clinical text | Only an explicit, unambiguous codec can decode a bounded byte string; malformed sequences fail closed. |
| Locale format tables | Date/number interpretation and locale-to-PII pattern selection remain developer-controlled, deterministic, and ambiguity preserving. |
| PHI-bearing custom terms | Source text, entry values, and raw paths never enter logs or exceptions. |

### Actors and controls

- **Input author:** may control every byte in a dictionary or encoded document,
  including archive headers, line boundaries, encodings, Unicode controls, and
  mixed-script text. They cannot execute code in the OpenMed process.
- **Operator:** selects a local source path, an allow-listed codec, and optional
  lower resource limits. The operator cannot raise a built-in ceiling or
  disable a limit with a zero, negative, non-finite, or non-integral value.
- **Package developer:** controls the Python-defined locale and PII-pattern
  tables. Changing those tables is a code-review and package-integrity event,
  not an attacker-controlled runtime ingestion path.
- **Artifact reader:** may see operational logs. Logs are assumed less trusted
  than the process and must remain free of custom terms and raw paths.

### Trust boundaries and invariants

1. **Filesystem to archive preflight.** Before Python materializes the ZIP
   central directory, the bounded end record must declare exactly one entry on
   one disk. Before that regular member is opened, compressed size, encryption,
   compression method, declared uncompressed size, and expansion ratio are
   checked. A dictionary archive is never extracted to disk.
2. **Archive/plain stream to parser.** The parser reads fixed-size chunks,
   bounds decompressed bytes, physical records, accepted entries, and per-entry
   bytes, and stops at the first count beyond any configured ceiling. It never
   reads the entire source merely to decide that it is too large.
3. **Entry parser to consumer.** Strict UTF-8, term length, Unicode-control,
   literal-regex, frequency, and POS rules are enforced before an immutable
   `UserDictionaryEntry` reaches a segmenter.
4. **Encoded bytes to Unicode.** Codec selection passes through a static alias
   allow-list before Python codec lookup. Decoding is strict and size-bounded;
   ambiguous codecs such as `utf-16` and executable legacy codecs such as
   `utf-7` are not permitted.
5. **Unicode to downstream detection.** Script analysis reports only
   `mixed_script` and `confusable_characters` warning codes. The original text
   remains unchanged for offset correctness. Warning analysis is a single-pass,
   constant-space check; downstream PII detection can separately request the
   existing offset-preserving normalization when it needs alignment maps.
6. **Locale tables to parsers.** Locale selection uses fixed mappings and frozen
   value objects. Missing locale evidence preserves ambiguity instead of
   guessing; unsupported locale hints fail explicitly.
7. **Any rejection to logging.** A rejection log may contain only a SHA-256 path
   key, byte size, entry count, and machine-readable reason. It must not contain
   a path, an entry, decoded text, or a downstream exception message.

The security objective is availability and privacy, not authenticity of an
operator-approved dictionary. A malicious but syntactically valid literal term
can still influence segmentation quality. Operators remain responsible for the
provenance and clinical suitability of supplied vocabularies.

## Attack Surface, Mitigations, and Attacker Stories

### STRIDE-style analysis

| ID | STRIDE property | Asset and attacker story | Mitigation and enforcing code path | Executable evidence |
| --- | --- | --- | --- | --- |
| MI-01 | Spoofing | An input author uses mixed Greek/Cyrillic/Latin lookalikes so a decoded term appears benign while routing differently. | Strict decoding records content-free mixed-script/confusable warnings through [`decode_ingestion_bytes`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/script_detect.py); offset-preserving normalization remains available through `normalize_for_pii_detection`. | [`test_encoding_fuzz.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_encoding_fuzz.py), [`test_script_detect.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/core/test_script_detect.py) |
| MI-02 | Tampering | A dictionary entry embeds NUL, bidi/format controls, or line-shaping characters to alter downstream parsing. | Unicode category `C*` rejection and byte/character length rules in [`validate_user_dictionary_entry`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/processing/tokenization.py). | [`test_dictionary_hardening.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_dictionary_hardening.py) |
| MI-03 | Tampering / elevation of privilege | A user term such as a nested-quantifier expression becomes executable regex and creates catastrophic backtracking or changes match semantics. | Terms are literal-only; regex construct characters are rejected with the named `executable_regex_construct` rule before consumer access. | [`test_dictionary_hardening.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_dictionary_hardening.py) |
| MI-04 | Repudiation | An operator cannot distinguish a resource rejection from malformed content without exposing the offending term. | Typed exceptions expose stable classes/rule names, while logs carry only hashed-path and numeric metadata in [`tokenization.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/processing/tokenization.py) and [`script_detect.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/script_detect.py). | [`test_dictionary_hardening.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_dictionary_hardening.py), [`test_encoding_fuzz.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_encoding_fuzz.py) |
| MI-05 | Information disclosure | A PHI-like custom term, filename, undecodable byte sequence, or codec exception is copied into a warning or log. | Central rejection helpers never format raw paths, entries, decoded text, or caught exception messages. Confusable warnings contain only warning codes. | Log-capture and warning-capture cases in [`test_dictionary_hardening.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_dictionary_hardening.py) and [`test_encoding_fuzz.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_encoding_fuzz.py) |
| MI-06 | Denial of service | A highly compressed ZIP or a central directory with many metadata-only entries consumes memory or CPU before validation. | A bounded end-record check rejects any count other than one before `ZipFile` parses the central directory; declared decompressed size and expansion ratio `>= 100` are rejected before `ZipFile.open`; only stored and deflated members are accepted. | ZIP bomb and central-directory preflight cases in [`test_dictionary_hardening.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_dictionary_hardening.py) |
| MI-07 | Denial of service | A plain or compressed dictionary contains 10 million short, blank, or comment records. | Fixed-chunk streaming stops on accepted entry `100001` or physical record `200001`; decompressed bytes and individual lines have independent caps. | Entry-count and record-count cases in [`test_dictionary_hardening.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_dictionary_hardening.py) |
| MI-08 | Denial of service | A source contains one unterminated entry or a massive malformed UTF-8 sequence. | Pending line bytes are bounded before decode; decoding is strict UTF-8 and produces a typed content-free error. | Unit and property cases in [`test_dictionary_hardening.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_dictionary_hardening.py) |
| MI-09 | Spoofing / tampering | An input author chooses UTF-7, a BOM-dependent codec, a dynamically registered codec, or truncated multibyte data to change visible text. | A static encoding alias map is consulted before codec lookup; only explicit allow-listed codecs are decoded, with `errors="strict"` and a byte cap. | [`test_encoding_fuzz.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_encoding_fuzz.py) |
| MI-10 | Tampering | Locale-free dates/numbers or an unsupported locale are coerced into a convenient interpretation, changing which identifiers are detected. | Frozen locale values and conservative parsers in [`locale_formats.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/locale_formats.py) preserve ambiguous results; [`get_patterns_for_language`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/pii_i18n.py) rejects unsupported languages. | [`test_locale_formats.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/core/test_locale_formats.py), [`test_pii_i18n.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/test_pii_i18n.py) |
| MI-11 | Elevation of privilege | A ZIP uses encryption, multiple members, a non-file member layout, ZIP64/multi-disk metadata, or a high-complexity compression method to escape the intended parser boundary. | The archive boundary accepts one standard single-disk end record, exactly one unencrypted regular member, and an explicit compression-method allow-list; no member is extracted to the filesystem. | Archive cases and the bounded dictionary property target in [`test_dictionary_hardening.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_dictionary_hardening.py) |

### Hardening decision and tradeoffs

We keep enforcement in two reusable in-process boundaries instead of relying on
each future segmenter or converter to duplicate checks. This preserves the
local-first architecture and adds no network, subprocess, cache, or persistent
state. The main compatibility cost is intentional: dictionaries containing
regex metacharacters, Unicode controls, non-UTF-8 text, multiple ZIP members, or
more than 100,000 accepted entries or 200,000 physical records must be cleaned
or split before use. The main runtime cost is one bounded streaming validation
pass; memory is bounded by the entry objects plus a 64 KiB read chunk and one
bounded pending line.

We considered accepting arbitrary codecs through `codecs.lookup` and escaping
regex-like terms instead of rejecting them. Both options are more compatible,
but they move policy into downstream consumers: a later caller could forget to
escape a term or activate a registered codec with behavior outside this threat
model. The fail-closed boundary is preferable for a PHI-processing library.

Residual risks remain:

- Standard ZIP end-record parsing now bounds the entry count before central
  directory materialization; the compressed-source cap remains an independent
  defense against oversized metadata and comments.
- A syntactically valid dictionary can still poison segmentation quality or
  suppress PII recall. Provenance, review, and recall gates remain necessary.
- Confusable detection is a warning, not a rejection, because legitimate
  multilingual clinical text is often mixed-script.
- The fixed locale and PII tables depend on package and source integrity; this
  work does not replace supply-chain controls.

### Bounded and extended fuzzing

The regular suite runs each dictionary-parser, user-entry-validation, and
encoding-conversion property target for at least 500 examples. Hypothesis uses a
per-example deadline, and `pytest-timeout` imposes a 30-second whole-test limit.
The examples include invalid UTF-8, truncated multibyte sequences, oversized
records/byte strings, nested regex constructs, controls, and mixed-script
confusables.

Run the acceptance surface with the regular budget:

```bash
.venv/bin/python -m pytest \
  tests/unit/security/test_dictionary_hardening.py \
  tests/unit/security/test_encoding_fuzz.py -q
```

Increase every ingestion property target without allowing fewer than 500
examples:

```bash
OPENMED_INGESTION_FUZZ_EXAMPLES=5000 \
  .venv/bin/python -m pytest \
  tests/unit/security/test_dictionary_hardening.py \
  tests/unit/security/test_encoding_fuzz.py -q
```

## Severity Calibration (Critical, High, Medium, Low)

- **Critical:** a loader executes attacker-controlled dictionary syntax or a
  rejection path writes real PHI/PII to an artifact. Per `SECURITY.md`, direct
  PHI leakage is never treated as ordinary availability failure.
- **High:** crafted ingestion predictably suppresses PII detection across a
  supported profile, or causes secret/identifier content to enter logs without
  requiring code execution.
- **Medium:** a bounded local denial of service requires an operator to load an
  attacker-supplied file, or encoding confusion changes interpretation without
  demonstrated identifier leakage.
- **Low:** a hardening gap has no direct identifier exposure and remains inside
  documented operator-controlled inputs.

### Disclosure-policy review

This document and its tests use synthetic terms only and were checked against
[`disclosure-policy.md`](disclosure-policy.md) and
[`no-raw-phi-logging.md`](no-raw-phi-logging.md): suspected redaction bypasses
remain private-report issues; public examples do not contain real PHI, secrets,
or production artifacts; logs and exception tests assert content-free behavior;
and the threat model distinguishes implemented controls from residual risk.
