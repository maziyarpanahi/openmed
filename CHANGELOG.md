# Changelog

All notable changes to OpenMed will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Completed clinical temporal timeline composition with DCT/TIMEX anchors on
  every ordered event, transitively reduced public TLINK graphs, metric-ready
  edge keys, and retained/pruned privacy-safe decision provenance (#1253).
- Added closure-aware temporal TLINK F1, PHI-safe transitive-closure
  consistency scoring, a zero-violation blocking gate, and synthetic
  discharge-summary gold with DCT, EVENT-TIMEX, EVENT-EVENT, reduction, and
  contradiction-trap coverage (#1309).
- Added deterministic OncoTree tumor-type mapping
  (`openmed.clinical.load_oncotree`, `map_tumor_type`) against a
  caller-supplied local release snapshot (path / `OPENMED_ONCOTREE_PATH` and
  version / `OPENMED_ONCOTREE_VERSION`; nothing is bundled or downloaded). The
  snapshot must be a flat JSON list of tumor-type nodes; nested OncoTree tree
  dumps are unsupported. Exact and normalized name/code lookup supports an
  optional caller-supplied `synonyms` list and indexes history and revocation
  aliases with current codes winning collisions; unmatched or ambiguous
  mentions stay unmapped with a reason (no fuzzy/lexical fallback).
  Results are version-stamped `OncoTreeMapping` values. Includes synthetic
  golden fixtures and `oncotree_top1_accuracy` evaluation support.
- Added an experimental `yasbd` sentence-segmentation backend selectable via
  `segment_text(..., backend="yasbd")` and
  `analyze_text(..., sentence_backend="yasbd")`, backed by the optional
  `yasbd-lib` extra. The default routing and core dependency set remain
  unchanged; opt-in spans are normalized to OpenMed's exact contiguous-offset
  contract, with explicit errors for missing dependencies, unknown backends,
  and conflicting preconstructed segmenters (#1848).

### Fixed

- Fixed the PySpark batch de-identification adapter so
  `make_deidentify_udf()` supplies concrete pandas `Series` annotations during
  UDF construction instead of failing with an unsupported `Any` signature
  (#1942).
- Fixed `openmed risk discover`, `risk assess`, and `risk anonymize` handling
  of UTF-8 BOM-prefixed CSV and TSV schemas so the first column is classified
  consistently, and added bounded validation causes to structured-release CLI
  errors instead of replacing actionable `TypeError` and `ValueError` details
  with a generic schema mismatch.

## [2.0.0] - 2026-07-28

OpenMed 2.0 expands the local-first privacy, clinical extraction, evaluation,
and deployment stack across Python, Swift, Kotlin/Android, JavaScript, MCP, and
container environments. The release preserves the documented v1 Python entry
points and REST paths while establishing explicit multilingual, structured,
agent-tool, and release-evidence contracts for the v2 line.

### Added
- Added an offline nursing-care observation zero-shot domain with display
  label mappings (IntakeOutput, LineDrainTube, RiskScore, MobilityStatus,
  CareIntervention, PainScore, SkinAssessment), canonical policy label
  metadata, synthetic per-label fixture coverage for risk-score and
  line/drain/tube spans, and domain-coverage evaluation integration (#910).

- Expanded built-in PII routing to 34 language codes and added deterministic
  identifier, date, phone, address, and surrogate support across Chinese,
  Indic, African, Nordic, Central and Eastern European, Russian, Urdu,
  Vietnamese, and regional Arabic workflows.
- Added offset-preserving normalization and segmentation for full-width and
  half-width text, native digits, Simplified and Traditional Chinese,
  Chinese numerals and word boundaries, Indic graphemes and legacy encodings,
  cross-script transliteration, code-mixed Hinglish, and confusable text.
- Added China PIPL, India DPDP and ABDM, South Africa POPIA, Nigeria NDPA,
  Kenya DPA, Egypt PDPL, Morocco Law 09-08, pan-African Malabo, GDPR, EU AI
  Act, ISO 27701/27001, consent-tag, and data-residency evidence workflows.
- Added script-correct multilingual surrogates, Chinese address and Pinyin name
  handling, India identifier and transliterated-name consistency, African
  locale providers, and checksum-valid national-identifier replacements.
- Added deterministic SMS, low-resource CPU, air-gapped installation, model
  integrity verification, crash-safe batch resume, tamper-evident audit,
  Safe Harbor attestation, k-anonymity, adversarial-PHI, and leakage-dashboard
  paths.
- Added an offline structured-data release-risk workflow with advisory
  quasi-identifier discovery, explicit reviewer overrides, patient-level
  k-anonymity, distinct or entropy l-diversity, variational t-closeness,
  bounded hierarchy and suppression search, whole-privacy-unit suppression,
  materialized-output revalidation, and rollback-safe publication.
- Added exact offline reference-population assessment over row-level or keyed
  longitudinal profiles, with k-map, exact-linkage risk, delta-presence,
  conservative unmatched-profile handling, explicit model assumptions, and
  separate data, schema, policy, and integrity digests. Aggregate results do
  not serialize raw profiles or privacy-unit values.
- Added aggregate-only structured-risk dashboards, a strict `openmed risk gate`
  CI command, deterministic expert-review evidence, and expert-authored
  Ed25519 attestations that independently verify signature, evidence binding,
  conclusion, and freshness without claiming automated Expert Determination.
- Added radiology section and finding extraction, serial measurement trends,
  longitudinal document linking, clinical coreference, temporal normalization,
  abbreviation disambiguation, multilingual relations, flowsheets, lab panels,
  discharge summaries, procedures, pulmonology, and pediatrics coverage.
- Added local adapters and packaging for OpenMRS, DHIS2, OpenHIM, community
  health forms, WHO SMART profiles, PySpark, Prefect, scrubadub, LlamaIndex,
  scispaCy, QuickUMLS, and offline ICD-11 grounding.
- Added a typed MCP tool registry with annotations and structured output,
  an `openmed-mcp` entry point, an MCP-enabled container and Compose service,
  and portable repository skills plus agent-oriented documentation feeds.
- Added Chinese and Indic throughput gates, multilingual golden fixtures,
  annotation-agreement and corpus-quality evidence, parser fuzzing, regression
  tracking, model-size and ARM latency budgets, and a fail-closed signed
  release-readiness decision.
- Added watchOS and visionOS OpenMedKit targets, grapheme-safe Swift span
  parity, compact on-device segmenters, Android `EntityPrediction` metadata and
  `OpenMedSpan`, GGUF and WebGPU export guidance, and reproducible Nix
  development environments.
- Added deterministic, fully local longitudinal document linking with
  MinHash-based near-duplicate clustering, directional amendment edges,
  retained superseded documents, and non-text source/target provenance on
  every relationship (#1835).
- Added a PySpark `pandas_udf` adapter (`openmed.interop.spark_udf`) for
  redacting free-text Spark DataFrame columns at warehouse scale via
  `make_deidentify_udf()` and the `deidentify_columns()` convenience helper,
  with the OpenMed model loaded lazily and cached per executor worker
  process. `pyspark` is imported lazily and stays behind the existing `spark`
  extra; the adapter is registered as `spark` in `openmed.interop` (#1816).
- Added a procedures zero-shot domain for surgical and diagnostic procedures,
  devices, and surgical approach, with a new `DEVICE` canonical label,
  keyword routing metadata, and canonical label normalization (#313).
- Added deterministic serial measurement trends that group repeated entities,
  normalize compatible units, order points through the clinical timeline,
  preserve source spans and incomparable readings, and emit a clinician-review
  advisory with synthetic offline direction and grouping gates (#1831).
- Added word-aware Chinese Pinyin romanization with tone-mark, numeric-tone,
  and heteronym output, plus deterministic Han name surrogates and
  tone-insensitive Pinyin vault keys for consistent Chinese name matching.
- Added a fail-closed, signed release-readiness gate that verifies signed model
  gate evidence, release documentation, a machine-readable API compatibility
  report, the public clinical disclaimer, and workflow-produced golden-suite
  evidence before a release can proceed (#1814).
- Added a scrubadub adapter with `to_canonical()`/`from_canonical()` span
  conversion and canonical label mapping for scrubadub's core `Filth` types
  (including the `en_US`/`en_GB` locale detectors), splitting `credential`
  matches into separate `USERNAME`/`PASSWORD` entities using scrubadub's
  named regex groups, flattening scrubadub overlap wrappers without label loss,
  and losslessly recombining credential matches on the return trip. Scoreless
  scrubadub spans default to fallback priority during OpenMed arbitration.
  scrubadub stays an optional `scrubadub` extra, and the adapter is
  registered lazily as `scrubadub` in `openmed.interop` (#281).
- Added a full Russian (`ru`) PII language pack, including Cyrillic date,
  phone, postal-index, and street-address patterns, SNILS (insurance account
  number) and OMS (health-insurance policy number) validators with
  checksum-backed surrogates, and service/SDK wiring. Russian uses the
  documented multilingual default-model placeholder until dedicated weights
  ship; the release-wide supported PII language-code allow-list now covers
  34 codes (#1860).
- Added opt-in token-level language identification for Hinglish clinical text,
  with exact offset-only decisions, deterministic local fallback routing,
  optional caller-supplied model hooks, and synthetic token-accuracy and
  de-identification recall gates (#1490).
- Added an offline, native ARM64 SMS-scale INT8 latency benchmark with a
  committed synthetic corpus, exact model artifact provenance, aggregate
  p50/p95/throughput/peak-RSS reporting, a Raspberry Pi 5 target envelope, and
  a CI gate that fails regressions beyond the permitted 20% tolerance (#1456).
- Added opt-in Simplified/Traditional Chinese normalization through OpenCC,
  including Taiwan and Hong Kong conversion configs, mixed-variant detection,
  and offset-preserving span projection back to original text (#1467).
- Added source-aligned Chinese numeral parsing for everyday and financial
  forms, valid year/month/day normalization, and contextual Chinese date,
  medical-record identifier, and clinical-quantity PII patterns (#1469).
- Added a Swahili README and an African developer onboarding guide covering
  bandwidth-aware model sizing and offline setup, POPIA/NDPA policy pointers,
  OpenMRS FHIR and DHIS2 Tracker recipes, community links, and shared
  translation-drift enforcement (#1455).
- Added an offline-first `openmed models size` command with committed download,
  disk, and peak-RAM estimates, cache-aware remaining bytes, per-task bandwidth
  recommendations, JSON output, and explicitly opt-in remote refinement (#1453).
- Added an opt-in, offline FHIR R4 profile checker for locally supplied WHO
  SMART Guidelines implementation-guide packages, including cardinality,
  fixed-value, locally enumerable binding, identifier/category slice, and
  post-de-identification conformance checks (#1451).
- Added local, structure-preserving de-identification for ODK Central,
  CommCare HQ, and KoBoToolbox JSON/CSV form exports, including XForm path
  semantics, repeat fidelity, safe unknown-text handling, geopoint
  generalization, and value-free policy manifests (#1450).
- Added an opt-in OpenHIM de-identification mediator with authenticated
  registration and heartbeats, FHIR and text transformation envelopes,
  byte-preserving opaque pass-through, and an offline container smoke fixture
  for deployment inside an HIE trust boundary (#1448).
- Added data-driven African healthcare-context safety-sweep terms for named
  facilities, mobile-money references, and context-gated ethnic affiliations,
  with non-keep defaults across the initial African policy profiles and
  synthetic no-leak fixtures (#1447).
- Added PHI-free, hash-verifiable Africa data-residency attestations for audited
  local de-identification runs, with data-driven jurisdiction wording, exact
  policy and model provenance, conservative captured offline evidence, a public
  JSON Schema, and an offline deployment and review guide (#1446).
- Added an Igbo (`ig`) PII pack for Nigerian clinical text with NFC, NFD, and
  unmarked context support, native `ig_NG` surrogates, shared Nigerian NIN and
  phone patterns, and grapheme-safe replacement of dot-below names (#1441).
- Added a Yoruba (`yo`) PII pack with NFC, NFD, and unmarked context support,
  native `yo_NG` surrogates, and grapheme-safe normalized span remapping and
  replacement boundaries for stacked dot-below and tone marks (#1440).
- Added a deterministic Hausa (`ha`) PII pack for Boko and numeric Ajami text,
  including Nigerian NINs, Nigerian and Nigerien phone numbers, exact-offset
  native-digit matching, `ha_NG` surrogates, and Arabic-script arbitration
  without claiming Ajami lexical coverage (#1439).
- Added build-generated `llms.txt` and `llms-full.txt` documentation feeds with
  curated quickstart, API, de-identification, agent, MCP, and REST coverage,
  plus strict local and Pages build checks (#1787).
- Added character-offset Chinese and Hindi clinical relation extraction over
  existing multilingual NER spans, including the versioned 44-predicate CMeIE
  mapping, constrained graph decoding, assertion propagation, synthetic gold,
  and distinct per-language relation F1 reporting (#1205).
- Added a Prefect integration with a `deidentify_file_task` task and a
  `deidentify_dataset_flow` flow that fan the local dataset redaction runner
  over lists of files and return PHI-free count summaries. Prefect stays an
  optional `prefect` extra, and the adapter is registered lazily as
  `prefect` in `openmed.interop` (#471).
- Added a deterministic radiology report parser that separates findings,
  impression, and recommendation text with provenance spans, captures only
  explicitly stated BI-RADS or Lung-RADS categories, and includes synthetic
  offline accuracy gates (#1838).
- Added a deterministic radiology-finding extractor that binds laterality,
  measured size, and anatomic location with per-field provenance, supports
  caller-supplied offline RadLex JSON mappings, and includes a synthetic
  finding-tuple-F1 gate (#1837).
- Added a deterministic, offline ISO 15919 transliteration pivot for nine Indic
  scripts, ITRANS and Harvard-Kyoto parsing, offset-preserving romanization,
  and cross-script person-name linkage in surrogate vaults (#1483).
- Added an offline Vietnamese (`vi`) PII language pack with context-gated CCCD
  and legacy CMND detection, Vietnamese dates, phone numbers, addresses and
  five-digit postal codes, plus `vi_VN` surrogates and a synthetic golden
  fixture (#819).
- Added region-qualified Arabic Faker locales (`ar-SA`, `ar-AE`, `ar-JO`,
  `ar-PS`, and explicit `ar-EG`) so Gulf and Levant text receives in-region
  surrogates; bare `ar` still defaults to `ar_EG`. Locales missing from the
  installed Faker fall back to `ar_EG` with a one-time warning, and
  `list_regional_locales('ar')` enumerates the supported tags (#483).
- Added curated conceptual surrogate locales for Senegal, Côte d’Ivoire,
  Cameroon, Mozambique, and Angola (`fr_SN`, `fr_CI`, `fr_CM`, `pt_MZ`, and
  `pt_AO`), including in-country names, addresses, cities, phone formats, and
  context-only Senegal CNI and Angola BI detection. Arabic regional overrides
  now also document `ar-DZ` and `ar-MA`; unavailable Faker backends retain the
  existing one-time-warning fallback to `ar_EG` (#1443).
- Added conservative Egypt PDPL and Morocco Law 09-08 policy profiles with
  complete mask action maps, no reversible mappings, mandatory safety sweeps,
  declarative `ar_EG`/`ar_MA` clinical identifier formats, and a decision-support
  compliance checklist covering sensitive-data and transfer controls (#1444).
- Added a release evidence job that keylessly signs each wheel and source
  distribution with Sigstore and attaches the SLSA provenance bundle, the
  release artifact digest manifest, and the Sigstore bundles to the tagged
  GitHub release, so a release can be verified offline without the GitHub
  attestation API. Evidence generation stays best effort and cannot gate the
  PyPI upload, but evidence that is produced must verify against the signing
  workflow identity and the release commit before it is attached (#1540).
- Added a Hungarian (`hu`) national-ID-only PII pack with validator-backed TAJ
  detection, `hu_HU` locale-aware synthetic surrogates, Hungarian date, phone,
  address, and postcode patterns, and an offline synthetic golden fixture
  (#816).
- Added a Czech (`cs`) national-ID-only PII pack with validator-backed rodné
  číslo detection, Czech date, phone, address, and postcode cues, `cs_CZ`
  locale-aware synthetic surrogates, and an offline synthetic golden fixture
  ([#815](https://github.com/maziyarpanahi/openmed/issues/815)).
- Added a release run-ledger builder that records, per family, which candidate
  artifact was published under which gate decision, binding the artifact digest
  to a recomputed `GateReport` hash so a published artifact provably passed the
  gate it claims. Non-`RELEASABLE` families are quarantined with no publish
  target, the run outcome is reconstructable offline from
  `gates/release_runs.jsonl`, and the ledger carries only identifiers and
  hashes. Adds `compute_canonical_payload_hash()` to `openmed.core.repro_hash`
  as the generic counterpart to the training-shaped
  `compute_reproducibility_hash()` (#1805).

### Changed

- Made the language-pack catalog the shared source of truth for runtime,
  service, registry, fixture, documentation, and CLI capability reporting.
- Added uniform CLI JSON output and error envelopes and strengthened
  model-download, manifest-signature, provenance, SBOM, secret-scan, and
  dependency-policy enforcement.
- Added the `DEVICE` canonical clinical label. Consumers that treat canonical
  labels as a closed enum must add the new value before adopting v2.
- Added optional extras for Chinese, Indic, language identification, integrity
  verification, OpenMRS, scrubadub, Prefect, scispaCy, and QuickUMLS; raised
  the optional MCP runtime floor to the v1.27 API line.

### Compatibility

- Preserved all public Python symbols from `v1.9.1`; the static API comparison
  records 6,050 additions, no removals or narrowed signatures, and no newly
  deprecated symbols.
- Preserved the existing REST/OpenAPI path and schema set. Swift, Android, npm,
  service, configuration, and serialized evidence changes are additive except
  for closed-enum consumers that must recognize `DEVICE`.
- Preserved the documented root imports, including `OpenMedConfig`,
  `analyze_text`, `deidentify`, and `extract_pii`.

### Fixed

- Corrected compact Indic segmenter licensing to the valid SPDX `ICU`
  identifier, pinned the immutable ICU 57.1 source revision, retained the full
  source copyright and permission notice in Python distributions and standalone
  model bundles, and added fail-closed Python, Swift, and release-policy
  validation.
- Corrected PharmaDetect medication boundaries and optional precision
  filtering, contrastive-clause experiencer scope, radiology span and
  laterality binding, cross-platform path handling, and model-loader behavior
  across stale caches and low-memory environments.
- Allowed explicitly selected models to run with deterministic pattern-only
  language packs while keeping the no-default-model path fail closed.
- Restored direct Hub-ID loading for pre-manifest MLX artifacts that carry
  trusted converter markers, kept local PyTorch privacy-filter snapshots on
  Torch, and continued to reject unmarked bundles.
- Root-anchored Hatch build patterns so wheel and source distributions cannot
  absorb nested local worktrees or workspace-control files.
- Raised the optional spaCy integration to 3.8.9+, isolated its NumPy 1.x ABI
  route from incompatible ONNX combinations, and resolved factory type
  annotations before spaCy's Pydantic-backed config validation.
- Stabilized the long-input and arbitrary-text de-identification fuzz
  properties across slower CI platforms by keeping their deterministic example
  budgets and safety assertions while leaving performance enforcement to the
  dedicated latency and throughput gates.

### Security

- Added telemetry-off-by-default enforcement, mixed-script and confusable PII
  defenses, no-raw-PHI evidence formats, integrity-checked model downloads,
  tamper-evident audit chains, consent and data-use enforcement, and
  reproducible release evidence.
- Updated vulnerable locked dependencies and kept release images, Python and
  npm artifacts, Android coordinates, and signed evidence on independent,
  fail-closed validation paths.
- Required ONNX 1.21+ for ONNX-producing optional routes, refreshed the
  universal dependency lock to the fixed ONNX line, and excluded generated
  SBOM, audit, vulnerability, and JavaScript dependency artifacts from Docker
  build contexts so local evidence and development dependencies cannot
  contaminate service images or image scans.

### Complete commit and pull-request inventory

The audited `1ab2eca4cc89..6525adb5722c` range contains 539 commits and 197 merged pull requests. The ledger below is generated from exact Git ancestry; pull-request entries include every branch commit reachable through their merge commit, while direct and integration commits remain explicit.

The final changelog-only commit that records this ledger is represented by this inventory section itself; a commit cannot contain its own content-derived SHA.

<details>
<summary>Merged pull requests and their included commits</summary>

- [#1547](https://github.com/maziyarpanahi/openmed/pull/1547) feat: add full-width/half-width normalization with offset preservation (5 audited commits)
  - `4d2eb1d77321` feat: add full-width/half-width normalization with offset preservation
  - `81237954a3c7` fix: register cjk_width_convention in OpenMedConfig.from_dict
  - `8710f526dbfc` Merge current master into PR #1547
  - `a6663eff65db` fix: integrate CJK width normalization into PII detection
  - `413db39b2468` Merge pull request #1547 from pardeep-singh/pardeep/issue1468-zh-width-normalize
- [#1548](https://github.com/maziyarpanahi/openmed/pull/1548) feat: add Indic native-digit folding with offset preservation (6 audited commits)
  - `d9dde39fa33d` feat: add Indic native-digit folding with offset preservation
  - `f398af89c4d0` Merge current master into PR #1548
  - `081e817a84b1` Merge PR #1547 normalization integration into PR #1548
  - `211f93fa0642` fix: integrate Indic digit folding into PII detection
  - `240557c8cb74` Merge current master into PR #1548 after #1547
  - `56e05d7c0fb1` Merge pull request #1548 from pardeep-singh/pardeep/issue1485-indic-numerals
- [#1596](https://github.com/maziyarpanahi/openmed/pull/1596) feat: add Unified Social Credit Code recognizer and MOD-31-3 validator (5 audited commits)
  - `614bda8ec16c` feat: add Unified Social Credit Code recognizer and MOD-31-3 validator
  - `f27a75c169a3` Merge current master into PR #1596
  - `ba9ccb5769ea` fix: harden Unified Social Credit Code handling
  - `5ec3721d0e8f` Merge current master into PR #1596 after prior merges
  - `b7b0e0d4ff49` Merge pull request #1596 from pardeep-singh/pardeep/issue1476-uscc-recognizer
- [#1595](https://github.com/maziyarpanahi/openmed/pull/1595) feat(ner): add pulmonology domain with spirometry and respiratory labels (3 audited commits)
  - `cb2445b92b4b` feat(ner): add pulmonology domain with spirometry and respiratory labels
  - `3b3e90b61a63` Merge current master into PR #1595
  - `23c1e3c7dfe8` Merge pull request #1595 from PouyanJay/feat/pulmonology-domain-labels
- [#1592](https://github.com/maziyarpanahi/openmed/pull/1592) feat: add Estonian (et) PII language pack with isikukood validator (6 audited commits)
  - `a3506aa1e08a` feat: add Estonian (et) PII language pack with isikukood validator
  - `9606fa9518ad` Merge current master into PR #1592
  - `b0f2253a06c1` fix: align Estonian personal-code validation
  - `60615e797776` Merge remote-tracking branch 'origin/master' into review/pr-1592
  - `c7c9ccdebca7` fix: reject non-string Estonian codes
  - `5e66ceb95ec3` Merge pull request #1592 from PouyanJay/feat/estonian-isikukood-pii
- [#1593](https://github.com/maziyarpanahi/openmed/pull/1593) feat: add Hungarian TAJ language pack (9 audited commits)
  - `42d86399d753` feat: add Hungarian TAJ language pack
  - `705414f0449c` test: expose Hungarian TAJ generator
  - `4cfab5d63bb9` Merge current master into PR #1593
  - `022ef0574907` fix: harden Hungarian TAJ handling
  - `a4adcb8ce46f` Merge remote-tracking branch 'origin/master' into review/pr-1593
  - `8efee76aef71` Merge PR #1592 integration into PR #1593
  - `b36854fb939f` Merge branch 'review/pr-1592' into review/pr-1593
  - `97c8cf13ace1` Merge remote-tracking branch 'origin/master' into review/pr-1593
  - `abb671812926` Merge pull request #1593 from thangldw/feat/hungarian-taj-language-pack
- [#1597](https://github.com/maziyarpanahi/openmed/pull/1597) feat: add Serbian (sr) PII language pack with JMBG validator (8 audited commits)
  - `baf84a547829` feat: add Serbian (sr) PII language pack with JMBG validator
  - `729147ba9173` Merge current master into PR #1597
  - `e391fd08a907` fix: enforce strict Serbian JMBG shape
  - `519740766194` Merge PR #1593 integration into PR #1597
  - `e716e0b992e2` Merge branch 'review/pr-1593' into review/pr-1597
  - `dc068af35f67` Merge remote-tracking branch 'origin/master' into review/pr-1597
  - `875236ad3f74` Merge remote-tracking branch 'origin/master' into review/pr-1597
  - `ae188fe01921` Merge pull request #1597 from PouyanJay/feat/serbian-jmbg-pii
- [#1598](https://github.com/maziyarpanahi/openmed/pull/1598) feat: add Croatian (hr) PII language pack with OIB validator (8 audited commits)
  - `2c23eca49408` feat: add Croatian (hr) PII language pack with OIB validator
  - `217cd4779a6a` Merge current master into PR #1598
  - `0655fbf88e1a` fix: harden Croatian OIB handling
  - `eb38fc903554` Merge PR #1597 integration into PR #1598
  - `048249f82327` Merge remote-tracking branch 'origin/master' into review/pr-1598
  - `d7a8abb7a885` Merge remote-tracking branch 'origin/master' into review/pr-1598
  - `e4ec9ee962cf` Merge remote-tracking branch 'origin/master' into review/pr-1598
  - `4ce92e3e9e82` Merge pull request #1598 from PouyanJay/feat/croatian-oib-pii
- [#1599](https://github.com/maziyarpanahi/openmed/pull/1599) feat: add Bulgarian (bg) PII language pack with EGN validator (9 audited commits)
  - `a1b37c2c4088` feat: add Bulgarian (bg) PII language pack with EGN validator
  - `a7ef18421da3` Merge branch 'master' into review/pr-1599
  - `fb569c99a1f9` fix: enforce strict Bulgarian EGN shape
  - `137af1344e83` Merge PR #1598 integration into PR #1599
  - `854339246f93` Merge remote-tracking branch 'origin/master' into review/pr-1599
  - `82bbd143bee6` Merge remote-tracking branch 'origin/master' into review/pr-1599
  - `700b77f11b02` Merge remote-tracking branch 'origin/master' into review/pr-1599
  - `292bd9342382` Merge remote-tracking branch 'origin/master' into review/pr-1599
  - `e6919c6d1fd4` Merge pull request #1599 from PouyanJay/feat/bulgarian-egn-pii
- [#1600](https://github.com/maziyarpanahi/openmed/pull/1600) feat: add Finnish (fi) PII language pack with HETU validator (10 audited commits)
  - `2a758d6092e1` feat: add Finnish (fi) PII language pack with HETU validator
  - `3401354a6816` Merge branch 'master' into review/pr-1600
  - `979b924fe551` fix: harden Finnish HETU validation
  - `ae12d3b6441d` Merge PR #1599 integration into PR #1600
  - `76db11b978ae` Merge remote-tracking branch 'origin/master' into review/pr-1600
  - `9011c9cd8f93` Merge remote-tracking branch 'origin/master' into review/pr-1600
  - `16331a268132` Merge remote-tracking branch 'origin/master' into review/pr-1600
  - `defd1bd228eb` Merge remote-tracking branch 'origin/master' into review/pr-1600
  - `cda1e865f7db` Merge remote-tracking branch 'origin/master' into review/pr-1600
  - `ee9d5189f375` Merge pull request #1600 from PouyanJay/feat/finnish-hetu-pii
- [#1602](https://github.com/maziyarpanahi/openmed/pull/1602) feat: add Czech (cs) PII language pack with rodne cislo validator (5 audited commits)
  - `d2d9d8aad070` feat: add Czech (cs) PII language pack with rodne cislo validator
  - `d03663555e09` Merge master into Czech PII language pack
  - `a8539f11bb3a` feat(pii): expand Czech locale coverage and regression tests
  - `012021d20852` fix(pii): finalize Czech language pack validation
  - `291f40ec2562` Merge pull request #1602 from PouyanJay/feat/czech-rodne-cislo-pii
- [#1612](https://github.com/maziyarpanahi/openmed/pull/1612) Fix/social cards fix (16 audited commits)
  - `48cfe1c69677` Delete apple-touch-180.html
  - `d5e62412fd60` Delete avatar-circle-400.html
  - `11e53def2d66` Delete avatar-linkedin-300.html
  - `7790f60106fb` Delete avatar-square-512.html
  - `14c1cf13fbbe` Delete favicon-64.html
  - `ce6bf7369b72` Delete github-social.html
  - `ac891cf1a5f6` Delete hf-card.html
  - `e15591ccd4fc` Delete og.html
  - `a949b8a8d5dd` Delete readme-banner.html
  - `cd444f67a0b8` Delete x-header.html
  - `1cff6d19b7d1` Update hf-card.png
  - `0615a4e9cc38` Update og.png
  - `262dc2abe637` Update readme-banner.png
  - `5a02809a5eec` Update x-header.png
  - `f667dad4f3dc` fix: remove deleted social card test target
  - `c2b582f893ce` Merge pull request #1612 from maziyarpanahi/fix/social-cards-fix
- [#1617](https://github.com/maziyarpanahi/openmed/pull/1617) test(security): add telemetry-off-by-default enforcement guard (OM-099) (2 audited commits)
  - `4fe919994d4e` test(security): add telemetry-off-by-default enforcement guard
  - `981134bd9c76` Merge pull request #1617 from PouyanJay/feat/no-telemetry-guard
- [#1613](https://github.com/maziyarpanahi/openmed/pull/1613) feat(android): add EntityPrediction description and OpenMedSpan data model (3 audited commits)
  - `1a7d40213f98` feat(android): add EntityPrediction description and OpenMedSpan data model
  - `f9e0429bc4db` fix(android): match Swift half-even rounding in EntityPrediction.toString
  - `aac3019394ce` Merge pull request #1613 from PouyanJay/feat/android-entityprediction-openmedspan
- [#1618](https://github.com/maziyarpanahi/openmed/pull/1618) test: cover help for every CLI command (2 audited commits)
  - `8535b504b2d0` test: cover help for every CLI command
  - `12aae0364648` Merge pull request #1618 from ShiHuiwen-creat/test/485-cli-help-coverage
- [#1614](https://github.com/maziyarpanahi/openmed/pull/1614) fix(clinical): scope experiencer cues across contrastive clauses (#277) (4 audited commits)
  - `a7fc761dd641` fix(clinical): scope experiencer cues across contrastive clauses
  - `a556bf303aac` docs(clinical): note the contrastive-clause terminator set is intentional
  - `00eb7e0a92b9` fix: cover FHx experiencer cue
  - `25765c0019b5` Merge pull request #1614 from PouyanJay/feat/experiencer-context-axis
- [#1615](https://github.com/maziyarpanahi/openmed/pull/1615) docs: add per-language PII de-identification guide (3 audited commits)
  - `bd91e1fe1afe` docs: add per-language PII de-identification guide
  - `05a22712351a` style: format language docs coherence test
  - `384282bc6f2d` Merge pull request #1615 from cycsmail/docs-per-language-pii-guide-287
- [#1608](https://github.com/maziyarpanahi/openmed/pull/1608) docs: decide and document the on-device Android tokenization strategy (2 audited commits)
  - `719e0cfdaee7` docs: decide and document the on-device Android tokenization strategy
  - `e3644bdbead3` Merge pull request #1608 from PouyanJay/docs/android-tokenization
- [#1606](https://github.com/maziyarpanahi/openmed/pull/1606) docs: add Android quickstart covering setup, model load, and redaction (2 audited commits)
  - `132826d2d6c5` docs: add Android quickstart covering setup, model load, and redaction
  - `d45635306acb` Merge pull request #1606 from PouyanJay/docs/android-quickstart
- [#1603](https://github.com/maziyarpanahi/openmed/pull/1603) feat: add Greek (el) PII language pack with AMKA validator (2 audited commits)
  - `4c115aad70c4` feat: add Greek (el) PII language pack with AMKA validator
  - `079a8f8c2ea2` Merge pull request #1603 from PouyanJay/feat/greek-amka-pii
- [#1605](https://github.com/maziyarpanahi/openmed/pull/1605) feat: add Portuguese NIF validator distinct from Brazilian CPF (2 audited commits)
  - `1f7903005271` feat: add Portuguese NIF validator distinct from Brazilian CPF
  - `b71f382c0261` Merge pull request #1605 from PouyanJay/feat/portuguese-nif-validator
- [#1616](https://github.com/maziyarpanahi/openmed/pull/1616) test(eval): add per-language i18n golden fixtures for all wired languages (4 audited commits)
  - `bd65981f23b7` test(eval): add per-language i18n golden fixtures for wired languages
  - `107607449f4f` test: complete multilingual golden fixtures
  - `2256fa941576` Merge master into multilingual fixture update
  - `bea669d76c99` Merge pull request #1616 from PouyanJay/feat/i18n-golden-fixtures-supported
- [#1604](https://github.com/maziyarpanahi/openmed/pull/1604) feat: sign release distributions and attach verifiable evidence (4 audited commits)
  - `c31ba257ccbd` feat: sign release distributions and attach verifiable evidence
  - `7653158682b4` Merge remote-tracking branch 'origin/master' into maintainer/pr-1604-followup
  - `7ba52f9af461` fix: require complete release evidence
  - `abe12f6a2a7d` Merge pull request #1604 from DrVelvetFog/security/attach-release-provenance-and-verify-docs
- [#1549](https://github.com/maziyarpanahi/openmed/pull/1549) feat(locales): add regional Arabic Faker-locale overrides (OM-285) (6 audited commits)
  - `44c4c44adc51` feat(locales): add regional Arabic Faker-locale overrides (OM-285)
  - `eb2ceac94358` Merge branch 'master' into feat/ar-regional-locales-om285
  - `218bcb3467b6` Merge master into regional locale update
  - `8740c7687406` Merge remote-tracking branch 'origin/master' into maintainer/pr-1549-followup
  - `7de01dbe49fa` Merge remote-tracking branch 'origin/master' into maintainer/pr-1549-followup
  - `aba65fe93c2e` Merge pull request #1549 from chawki-nasrallah/feat/ar-regional-locales-om285
- [#1601](https://github.com/maziyarpanahi/openmed/pull/1601) feat: add Vietnamese PII language pack (5 audited commits)
  - `1f5edb88c5b8` feat: add Vietnamese PII language pack
  - `93d2b67ae140` fix: align Vietnamese CCCD validation with current law
  - `425b79d736c9` Merge master into Vietnamese PII update
  - `fe26421f80c0` Merge commit 'refs/pr-review/1549' into maintainer/pr-1601-followup
  - `0050e1410cdd` Merge pull request #1601 from thangldw/feat/vietnamese-pii-language-pack
- [#687](https://github.com/maziyarpanahi/openmed/pull/687) fix: raise TypeError for malformed policy argument types (5 audited commits)
  - `ced09f8839d8` fix: raise TypeError for malformed policy argument types
  - `550af1ae1492` Merge remote-tracking branch 'origin/master' into fix/policy-type-validation
  - `8344228d758d` test: cover policy profile error guidance
  - `9662c4acb2dd` Merge current master into policy type validation
  - `43c7e8e0a461` Merge pull request #687 from abdouloued/fix/policy-type-validation
- [#1611](https://github.com/maziyarpanahi/openmed/pull/1611) feat: add pediatrics growth-parameter domain to NER model and labels map (4 audited commits)
  - `1bf2b95ddd41` feat: pediatrics growth-parameter domain to NER model and labels map
  - `b5cca3a9de52` Merge remote-tracking branch 'origin/master' into review-1611-20260718
  - `1c96952ba827` fix: polish pediatrics growth metadata
  - `ba9407e8fb57` Merge pull request #1611 from mrfeathers/feature/om-896-pediatrics-labels
- [#1702](https://github.com/maziyarpanahi/openmed/pull/1702) docs/examples: add synthetic datasets walkthrough (2 audited commits)
  - `d7c49a4afb21` Add synthetic datasets walkthrough example
  - `30a375473013` Merge pull request #1702 from otmanm/codex/om-281-datasets-walkthrough
- [#1697](https://github.com/maziyarpanahi/openmed/pull/1697) feat: add inter-annotator agreement metrics for extraction gold sets (3 audited commits)
  - `0942b32b28a6` feat: add inter-annotator agreement metrics for extraction gold sets
  - `3ab7912b08b3` fix: complete span-overlap agreement coverage
  - `e3eb2801bd5c` Merge pull request #1697 from pardeep-singh/pardeep/issue1317-inter-annotator-agreement
- [#1698](https://github.com/maziyarpanahi/openmed/pull/1698) feat: add synthetic multi-annotator gold corpus and consensus loader (3 audited commits)
  - `0102bf1f3cac` feat: add synthetic multi-annotator gold corpus and consensus loader
  - `b9953896be6a` fix: preserve annotator relation evidence
  - `1e48420df8f1` Merge pull request #1698 from pardeep-singh/pardeep/issue1318-consensus-corpus
- [#1700](https://github.com/maziyarpanahi/openmed/pull/1700) feat: add gold-corpus quality report and evidence-bundle output (5 audited commits)
  - `a61e5bdb2625` Merge branch 'pardeep/issue1318-consensus-corpus' into pardeep/issue1321-quality-report
  - `25a60961b668` feat: add gold-corpus quality report and evidence-bundle output
  - `9ffa461aa779` Merge master into pardeep/issue1321-quality-report
  - `fb1c2d75b8cc` fix: report annotator relation agreement
  - `d3625ef85e90` Merge pull request #1700 from pardeep-singh/pardeep/issue1321-quality-report
- [#1716](https://github.com/maziyarpanahi/openmed/pull/1716) feat: add Prefect task and flow for batch de-identification (3 audited commits)
  - `195c09c9b267` feat: add Prefect task and flow for batch de-identification
  - `27a5ac11c19c` fix: harden Prefect batch integration
  - `6a6f3acca788` Merge pull request #1716 from RonitGandhi/fix/issue-471
- [#1746](https://github.com/maziyarpanahi/openmed/pull/1746) feat: add flowsheet and vitals time-series structurer (3 audited commits)
  - `29f991e742a9` feat: add flowsheet and vitals time-series structurer
  - `505fe34b5efe` fix: complete flowsheet continuation handling
  - `76268507702f` Merge pull request #1746 from pardeep-singh/pardeep/issue941-flowsheet
- [#1744](https://github.com/maziyarpanahi/openmed/pull/1744) feat: add lab-panel structurer mapping results into analyte rows (4 audited commits)
  - `ddf40bb96daf` feat: add lab-panel structurer mapping results into analyte rows
  - `dc24b8593de0` Merge master into pardeep/issue940-lab-panels
  - `34ce90783509` fix: complete lab panel report parsing
  - `9d9dc19912a1` Merge pull request #1744 from pardeep-singh/pardeep/issue940-lab-panels
- [#1743](https://github.com/maziyarpanahi/openmed/pull/1743) feat: add portable Agent Skills catalog for building with OpenMed (1 audited commit)
  - `1623bf04c1dd` feat: add portable Agent Skills catalog for building with OpenMed (#1743)
- [#1778](https://github.com/maziyarpanahi/openmed/pull/1778) test: locate py.typed through importlib.resources (2 audited commits)
  - `cdc195321c2e` test: locate py.typed as package resource
  - `d60a6f82ee44` Merge pull request #1778 from lntutor/feat/py-typed-254
- [#1840](https://github.com/maziyarpanahi/openmed/pull/1840) fix: format skills catalog test (2 audited commits)
  - `a38f7c8c78ce` fix: format skills catalog test
  - `2f0cbbaf22f8` Merge pull request #1840 from maziyarpanahi/fix/skills-catalog-format
- [#1755](https://github.com/maziyarpanahi/openmed/pull/1755) docs: add v1 to v2 migration guide (5 audited commits)
  - `ebb19df3676e` docs: add v1 to v2 migration guide
  - `e87bae08272f` Merge remote-tracking branch 'origin/master' into review/pr-1755
  - `ddb01ef4226f` docs: correct v2 migration history
  - `08f866c7965e` Merge remote-tracking branch 'origin/master' into review/pr-1755
  - `dedd914832a8` Merge pull request #1755 from lntutor/docs/v1-v2-migration-301
- [#1762](https://github.com/maziyarpanahi/openmed/pull/1762) docs: define detector plugin SDK stability (4 audited commits)
  - `19fbdfb6fb87` docs: define detector plugin SDK stability
  - `712385d3a948` Merge remote-tracking branch 'origin/master' into review/pr-1762
  - `85477c699ca6` docs: fix detector example span
  - `cde710f8e9e9` Merge pull request #1762 from lntutor/docs/plugin-sdk-stability-1327
- [#1586](https://github.com/maziyarpanahi/openmed/pull/1586) Add language pack registry foundation (6 audited commits)
  - `8b5a9ad6fdb6` Add language pack registry foundation
  - `c5fadb9512ca` fix: defer optional model imports
  - `027bd0ad4b0d` Merge remote-tracking branch 'origin/master' into review/pr-1586
  - `45f2b27c4285` fix: scope language pack foundation
  - `a89626b03300` Merge remote-tracking branch 'origin/master' into review/pr-1586
  - `6736c213b175` Merge pull request #1586 from maziyarpanahi/feature/om-678-language-pack-plugin-framework
- [#1761](https://github.com/maziyarpanahi/openmed/pull/1761) feat: add MCP console entry point (3 audited commits)
  - `950f977c4b9f` feat: add MCP console entry point
  - `afa170d9d3b5` Merge remote-tracking branch 'origin/master' into review/pr-1761
  - `4488346765e2` Merge pull request #1761 from lntutor/feat/mcp-console-clients-1739
- [#1844](https://github.com/maziyarpanahi/openmed/pull/1844) feat: make language packs source of truth (3 audited commits)
  - `94de85e2f278` feat: adapt language maps to registry
  - `ae813ab0c1c5` Merge remote-tracking branch 'origin/master' into feature/language-pack-adapters-1583
  - `e4922c631630` Merge pull request #1844 from maziyarpanahi/feature/language-pack-adapters-1583
- [#1780](https://github.com/maziyarpanahi/openmed/pull/1780) feat(cli): uniform --json output, error envelopes, and a tool-schema drift guard (6 audited commits)
  - `25f1722ab551` feat(cli): add uniform --json output, error envelopes, and a tool-schema drift guard
  - `6b9ebe6732af` Merge remote-tracking branch 'origin/master' into review/pr-1780
  - `aec3f342009f` fix: complete CLI JSON error handling
  - `754715ba9db0` Merge remote-tracking branch 'origin/master' into review/pr-1780
  - `b1c9ebf59836` Merge remote-tracking branch 'origin/master' into review/pr-1780
  - `f3c9745053b6` Merge pull request #1780 from PouyanJay/feat/cli-json-uniform
- [#1779](https://github.com/maziyarpanahi/openmed/pull/1779) feat(core): add language-pack coherence validation and capability coverage (6 audited commits)
  - `2239d923e62b` feat(core): add language-pack coherence validation and capability coverage
  - `52f21a10655b` Merge branch 'feature/language-pack-adapters-1583' into review/pr-1779
  - `8bee055e5a4d` fix: harden language pack coherence
  - `2da34471ec5c` Merge remote-tracking branch 'origin/master' into review/pr-1779
  - `83754a07dc18` Merge remote-tracking branch 'origin/master' into review/pr-1779
  - `142e99aac33d` Merge pull request #1779 from PouyanJay/feat/language-pack-coherence
- [#1845](https://github.com/maziyarpanahi/openmed/pull/1845) docs: add Windows uv installation steps (2 audited commits)
  - `fa9319010dff` docs: add Windows uv installation steps
  - `973dc3484c48` Merge pull request #1845 from maziyarpanahi/agent/windows-uv-install-docs
- [#1850](https://github.com/maziyarpanahi/openmed/pull/1850) build: consolidate dependency updates (2 audited commits)
  - `7631f08b23b6` build: consolidate dependency updates
  - `91a2fe0ae6a9` Merge pull request #1850 from maziyarpanahi/chore/dependency-refresh-july-2026
- [#1839](https://github.com/maziyarpanahi/openmed/pull/1839) feat: generate llms.txt and llms-full.txt (5 audited commits)
  - `0cef3e621c80` feat: generate llms.txt and llms-full.txt as requested in #1787
  - `62d60213b203` Merge origin/master into fix-llms-txt
  - `a07077d2d9cc` docs: generate LLM documentation feeds
  - `dad3f6f2837e` build: update Pillow security lock
  - `e380dcf7f355` Merge pull request #1839 from vamshiss/fix-llms-txt
- [#1566](https://github.com/maziyarpanahi/openmed/pull/1566) feat: add offset-safe Indic Unicode normalization (1 audited commit)
  - `8604ed52c49f` feat: add offset-safe Indic Unicode normalization (#1566)
- [#1573](https://github.com/maziyarpanahi/openmed/pull/1573) Add Indic Unicode script routing metadata (1 audited commit)
  - `84fc7fa537fa` feat: add Indic script routing metadata (#1573)
- [#1587](https://github.com/maziyarpanahi/openmed/pull/1587) Script-aware span decoder for no-whitespace CJK and grapheme-cluster Indic (1 audited commit)
  - `2f087f039e0c` feat: add script-aware grapheme span refinement (#1587)
- [#1588](https://github.com/maziyarpanahi/openmed/pull/1588) Non-Latin-script leakage evaluation harness with per-script recall floors (1 audited commit)
  - `4e3906199439` feat: add script-stratified leakage gates (#1588)
- [#1558](https://github.com/maziyarpanahi/openmed/pull/1558) Add pluggable Chinese word segmentation (1 audited commit)
  - `7277a1fe18b6` feat: add pluggable Chinese word segmentation (#1558)
- [#1669](https://github.com/maziyarpanahi/openmed/pull/1669) Defend de-identification against confusable and mixed-script evasion (1 audited commit)
  - `d472a38a8f35` fix: block confusable mixed-script PII evasion (#1669)
- [#1665](https://github.com/maziyarpanahi/openmed/pull/1665) Add token- and document-level language routing (1 audited commit)
  - `1a2ae86dad70` feat: add token- and document-level language routing (#1665)
- [#1563](https://github.com/maziyarpanahi/openmed/pull/1563) Resident ID (居民身份证) recognizer, MOD-11-2 validator, and locale-correct surrogate generator (1 audited commit)
  - `c0435ebf7f24` feat: add Chinese Resident ID protection (#1563)
- [#1564](https://github.com/maziyarpanahi/openmed/pull/1564) Add Chinese personal-name detection and locale-correct surrogates (1 audited commit)
  - `5858d32952b4` feat: add Chinese name detection and surrogates (#1564)
- [#1565](https://github.com/maziyarpanahi/openmed/pull/1565) China PIPL de-identification policy profile (1 audited commit)
  - `85fdacdd6740` feat: add China PIPL policy profile (#1565)
- [#1575](https://github.com/maziyarpanahi/openmed/pull/1575) Add India DPDP Act de-identification policy profile (1 audited commit)
  - `5ae1fa3a363e` feat: add India DPDP policy profile (#1575)
- [#1576](https://github.com/maziyarpanahi/openmed/pull/1576) India ABDM/ABHA-aware health-record de-identification mode (1 audited commit)
  - `93959a225c0b` feat: add India ABDM de-identification mode (#1576)
- [#1620](https://github.com/maziyarpanahi/openmed/pull/1620) Add Nigeria NIN and BVN recognizers with +234 mobile prefix validation and deterministic surrogates (1 audited commit)
  - `6905448c456a` feat: add Nigerian NIN and BVN recognition (#1620)
- [#1621](https://github.com/maziyarpanahi/openmed/pull/1621) Add Ghana Card and Kenya identity recognizers (1 audited commit)
  - `2e3029f93007` feat: add Ghana and Kenya identity recognizers (#1621)
- [#1623](https://github.com/maziyarpanahi/openmed/pull/1623) Add Swahili language pack with Sheng clinical note handling (1 audited commit)
  - `d53bd2f14dca` feat: add Swahili PII language pack (#1623)
- [#1625](https://github.com/maziyarpanahi/openmed/pull/1625) Add isiZulu and isiXhosa PII packs with South African ID validation (1 audited commit)
  - `c0b097b37020` feat: add isiZulu and isiXhosa PII packs (#1625)
- [#1626](https://github.com/maziyarpanahi/openmed/pull/1626) Add MasakhaNER African-language NER evaluation suite (1 audited commit)
  - `12a7d05c8aaf` feat: add MasakhaNER evaluation suite (#1626)
- [#1627](https://github.com/maziyarpanahi/openmed/pull/1627) Add South Africa POPIA policy profile (1 audited commit)
  - `e9118835e31f` feat: add South Africa POPIA policy profile (#1627)
- [#1628](https://github.com/maziyarpanahi/openmed/pull/1628) Add Nigeria NDPA 2023 policy profile (1 audited commit)
  - `cb7078075c05` feat: add Nigeria NDPA policy profile (#1628)
- [#1629](https://github.com/maziyarpanahi/openmed/pull/1629) Add Kenya Data Protection Act 2019 policy profile (ke_dpa) with health-data handling posture (1 audited commit)
  - `d4aac730334d` feat: add Kenya DPA policy profile (#1629)
- [#1631](https://github.com/maziyarpanahi/openmed/pull/1631) feat: add de-identified DHIS2 district exporter (1 audited commit)
  - `1b05eb865b21` feat: add privacy-safe DHIS2 exporter (#1631)
- [#1849](https://github.com/maziyarpanahi/openmed/pull/1849) feat(clinical): radiology report section parser with stated RADS capture (4 audited commits)
  - `f325d28b7d54` feat(clinical): add radiology report section parser with stated RADS capture
  - `f6810cd6e6ff` fix(clinical): harden radiology report parsing
  - `3fafae7ba562` fix: keep radiology changelog mergeable
  - `e4d57be15d82` Merge pull request #1849 from PouyanJay/feat/radiology-report-parser
- [#1632](https://github.com/maziyarpanahi/openmed/pull/1632) Add resilient integrity-checked model downloads (1 audited commit)
  - `eaadb395a24f` feat: add resilient integrity-checked model downloads (#1632)
- [#1852](https://github.com/maziyarpanahi/openmed/pull/1852) feat: add Urdu RTL PII language pack with Pakistani CNIC validation (6 audited commits)
  - `b4b30fe68ab9` feat: Add an Urdu (ur) RTL PII language pack with Pakistani CNIC validator
  - `cb0f276f6eaf` Merge branch 'master' into urdu_lang_pack
  - `c09ede04e1fc` Merge remote-tracking branch 'origin/master' into urdu_lang_pack
  - `1013c62da2f5` fix: complete Urdu CNIC language pack
  - `a6ae3df40b17` Merge remote-tracking branch 'origin/master' into urdu_lang_pack
  - `f9cead16b1f7` Merge pull request #1852 from AlyanPremani05/urdu_lang_pack
- [#1634](https://github.com/maziyarpanahi/openmed/pull/1634) Add CPU-only low-resource de-identification profile (1 audited commit)
  - `3d4273e604fb` feat: add CPU-only low-resource de-identification profile (#1634)
- [#1635](https://github.com/maziyarpanahi/openmed/pull/1635) Add crash-safe batch checkpoints and resume (1 audited commit)
  - `449ce08f97bf` feat: add crash-safe batch checkpoints and resume (#1635)
- [#1562](https://github.com/maziyarpanahi/openmed/pull/1562) Add Chinese clinical NER evaluation foundation (1 audited commit)
  - `e2658f963362` feat: add Chinese clinical NER evaluation foundation (#1562)
- [#1567](https://github.com/maziyarpanahi/openmed/pull/1567) Add Indic grapheme-safe span offsets (1 audited commit)
  - `05adab7cce8c` feat: add Indic grapheme-safe span offsets (#1567)
- [#1574](https://github.com/maziyarpanahi/openmed/pull/1574) Aadhaar recognizer hardening: Verhoeff validation, UIDAI masking rules, and checksum-valid surrogates (1 audited commit)
  - `f19754a1d37b` feat: harden Aadhaar de-identification (#1574)
- [#1581](https://github.com/maziyarpanahi/openmed/pull/1581) Add synthetic India code-mixed clinical de-identification corpus (1 audited commit)
  - `8bc252ad8e79` feat: add synthetic India clinical de-identification corpus (#1581)
- [#1589](https://github.com/maziyarpanahi/openmed/pull/1589) feat: add Chinese and Indic optional extras (1 audited commit)
  - `d6ecd85bf22e` feat: add Chinese and Indic optional extras (#1589)
- [#1590](https://github.com/maziyarpanahi/openmed/pull/1590) Audit PII tokenizer script coverage (1 audited commit)
  - `7268d6ac9939` feat: audit PII tokenizer script coverage (#1590)
- [#1591](https://github.com/maziyarpanahi/openmed/pull/1591) Verify cached model artifacts and signed manifests (1 audited commit)
  - `e0896dfe83c1` feat: verify cached model integrity (#1591)
- [#1619](https://github.com/maziyarpanahi/openmed/pull/1619) Add South African ID and mobile phone recognizers (1 audited commit)
  - `29c4398aa34b` feat: add South African ID and mobile phone recognizers (#1619)
- [#1622](https://github.com/maziyarpanahi/openmed/pull/1622) Add Egypt and Morocco identity recognizers (1 audited commit)
  - `d23616b474ad` feat: add Egypt and Morocco identity recognizers (#1622)
- [#1624](https://github.com/maziyarpanahi/openmed/pull/1624) Amharic language pack with Ethiopic script detection and grapheme-safe offsets (1 audited commit)
  - `bedef9103ff4` feat: add Amharic PII language pack (#1624)
- [#1630](https://github.com/maziyarpanahi/openmed/pull/1630) OpenMRS adapter: de-identify REST and FHIR2 handoffs locally (1 audited commit)
  - `675c2114fb67` feat: add local-first OpenMRS handoff adapter (#1630)
- [#1633](https://github.com/maziyarpanahi/openmed/pull/1633) Add offline installation kit builder (1 audited commit)
  - `37c9cd07f94c` feat: add air-gapped install kit builder (#1633)
- [#1841](https://github.com/maziyarpanahi/openmed/pull/1841) feat: add SMS short-text de-identification (1 audited commit)
  - `02eca8dde552` feat: add SMS short-text de-identification (#1841)
- [#1636](https://github.com/maziyarpanahi/openmed/pull/1636) Add clinical abbreviation and acronym sense disambiguation (1 audited commit)
  - `7796a8a405e9` feat: add clinical abbreviation and acronym sense disambiguation (#1636)
- [#1637](https://github.com/maziyarpanahi/openmed/pull/1637) Build a shared free-vocabulary lexical matcher engine and loader registry (1 audited commit)
  - `ffccf9559daa` feat: add shared free-vocabulary lexical matcher (#1637)
- [#1638](https://github.com/maziyarpanahi/openmed/pull/1638) Add tamper-evident audit chains for de-identification runs (1 audited commit)
  - `70e800c3a8a2` feat: add tamper-evident audit chains (#1638)
- [#1639](https://github.com/maziyarpanahi/openmed/pull/1639) Add a HIPAA Safe Harbor attestation report generator (1 audited commit)
  - `95376dc58e22` Add a HIPAA Safe Harbor attestation report generator (#1639)
- [#1640](https://github.com/maziyarpanahi/openmed/pull/1640) Add clinical coreference resolution linking entity mentions and pronouns (1 audited commit)
  - `49d3f56b657c` Add clinical coreference resolution linking entity mentions and pronouns (#1640)
- [#1641](https://github.com/maziyarpanahi/openmed/pull/1641) Normalize TIMEX3 temporal expressions to ISO values (1 audited commit)
  - `aecb78218c99` Normalize TIMEX3 temporal expressions to ISO values (#1641)
- [#1642](https://github.com/maziyarpanahi/openmed/pull/1642) Render a public auto-published benchmark leaderboard from archived eval reports (1 audited commit)
  - `16330e94924f` Render a public benchmark leaderboard from archived eval reports (#1642)
- [#1643](https://github.com/maziyarpanahi/openmed/pull/1643) Add a k-anonymity engine for tabular outputs (1 audited commit)
  - `65471ff0ffe8` Add a k-anonymity engine for tabular outputs (#1643)
- [#1644](https://github.com/maziyarpanahi/openmed/pull/1644) Add an eval-result provenance and reproducibility-hash ledger (1 audited commit)
  - `e5904a0969c3` Add an eval-result provenance and reproducibility-hash ledger (#1644)
- [#1647](https://github.com/maziyarpanahi/openmed/pull/1647) Add Chinese sentence segmentation honoring CJK punctuation (1 audited commit)
  - `6436a26e70cd` Add Chinese sentence segmentation honoring CJK punctuation (#1647)
- [#1648](https://github.com/maziyarpanahi/openmed/pull/1648) Add Chinese terminology grounding for user-supplied dictionaries (1 audited commit)
  - `de26a4cfc4b1` feat: add Chinese terminology grounding (#1648)
- [#1649](https://github.com/maziyarpanahi/openmed/pull/1649) Chinese address de-identification across province/city/district hierarchy with consistent surrogates (1 audited commit)
  - `cd8d5160e030` feat: add Chinese hierarchical address de-identification (#1649)
- [#1650](https://github.com/maziyarpanahi/openmed/pull/1650) Add Chinese mobile, bank-card, and travel-document recognizers (1 audited commit)
  - `5c854800f003` feat: add Chinese identifier recognizers (#1650)
- [#1651](https://github.com/maziyarpanahi/openmed/pull/1651) Add cross-script Indic transliteration with ISO 15919 (1 audited commit)
  - `8a7a68853a77` feat: add cross-script Indic transliteration (#1651)
- [#1652](https://github.com/maziyarpanahi/openmed/pull/1652) Add Indic danda-aware sentence and word tokenization (1 audited commit)
  - `675766269ee1` feat: add Indic sentence and word tokenization (#1652)
- [#1656](https://github.com/maziyarpanahi/openmed/pull/1656) Add optional Indic NER and 11-language evaluation (1 audited commit)
  - `779fdf684d0d` feat: add optional Indic NER evaluation (#1656)
- [#1658](https://github.com/maziyarpanahi/openmed/pull/1658) Add code-mixed Hinglish de-identification pipeline (1 audited commit)
  - `2cced3bab6a2` feat: add code-mixed Hinglish de-identification (#1658)
- [#1660](https://github.com/maziyarpanahi/openmed/pull/1660) feat: add India health-ID de-identification mode (1 audited commit)
  - `e1c8404aaf0f` feat: add India health-ID de-identification mode (#1660)
- [#1661](https://github.com/maziyarpanahi/openmed/pull/1661) Add India code-mixed clinical NER (1 audited commit)
  - `4e15672fffb1` feat: add India code-mixed clinical NER (#1661)
- [#1662](https://github.com/maziyarpanahi/openmed/pull/1662) India AYUSH and Indian drug terminology grounding (user-supplied, license-aware) (1 audited commit)
  - `0f6c2313b604` feat: add license-aware India terminology grounding (#1662)
- [#1663](https://github.com/maziyarpanahi/openmed/pull/1663) India locale-correct surrogate providers (1 audited commit)
  - `7e5132501661` feat: add India locale surrogate providers (#1663)
- [#1664](https://github.com/maziyarpanahi/openmed/pull/1664) Add consistent India transliterated-name surrogates (1 audited commit)
  - `60ab34c39cad` feat: add consistent India transliterated-name surrogates (#1664)
- [#1666](https://github.com/maziyarpanahi/openmed/pull/1666) Add license-aware CMeEE and Naamapadam eval suites (1 audited commit)
  - `adf50eefe325` feat: add license-aware CMeEE and Naamapadam eval suites (#1666)
- [#1667](https://github.com/maziyarpanahi/openmed/pull/1667) Multilingual surrogate framework with script-correct providers and cross-document consistency (1 audited commit)
  - `c2781792b683` feat: add script-correct multilingual surrogates (#1667)
- [#1668](https://github.com/maziyarpanahi/openmed/pull/1668) Package compact on-device segmenters for MLX, CoreML, and ONNX (1 audited commit)
  - `be19eada6064` feat: package compact on-device segmenters (#1668)
- [#1670](https://github.com/maziyarpanahi/openmed/pull/1670) Bring Simplified Chinese README to full parity and add translation drift check in CI (1 audited commit)
  - `d73cec9979b1` docs: enforce Chinese README parity (#1670)
- [#1671](https://github.com/maziyarpanahi/openmed/pull/1671) Bring Hindi README to parity with a synthetic Hinglish example (1 audited commit)
  - `5f117a09cfe5` docs: bring Hindi README to parity (#1671)
- [#1673](https://github.com/maziyarpanahi/openmed/pull/1673) docs: add China mirror and offline-cache onboarding (1 audited commit)
  - `f2135575ff80` docs: add China mirror and offline-cache onboarding (#1673)
- [#1674](https://github.com/maziyarpanahi/openmed/pull/1674) docs: add India DPDP onboarding guide (1 audited commit)
  - `7eb132d82053` docs: add India DPDP onboarding guide (#1674)
- [#1675](https://github.com/maziyarpanahi/openmed/pull/1675) Add Chinese and Hindi de-identification examples (1 audited commit)
  - `b9dcce966a3c` feat: add Chinese and Hindi de-identification examples (#1675)
- [#1676](https://github.com/maziyarpanahi/openmed/pull/1676) Add multilingual de-identification Space demo (1 audited commit)
  - `9351049ab9ec` feat: add multilingual de-identification Space demo (#1676)
- [#1678](https://github.com/maziyarpanahi/openmed/pull/1678) Add API-surface migration completeness gate (1 audited commit)
  - `dd9d40a5dd40` Add API-surface migration completeness gate (#1678)
- [#1679](https://github.com/maziyarpanahi/openmed/pull/1679) Add East African national ID recognizers (1 audited commit)
  - `4410103f5320` Add East African national ID recognizers (#1679)
- [#1680](https://github.com/maziyarpanahi/openmed/pull/1680) Add pan-African mobile phone patterns and prefix-preserving surrogates (1 audited commit)
  - `ce304e269592` Add pan-African mobile phone patterns and prefix-preserving surrogates (#1680)
- [#1681](https://github.com/maziyarpanahi/openmed/pull/1681) Add M-Pesa transaction code protection for Kenya and Tanzania (1 audited commit)
  - `9a31ba3399f2` Add M-Pesa transaction code protection for Kenya and Tanzania (#1681)
- [#1683](https://github.com/maziyarpanahi/openmed/pull/1683) Add mobile-money billing reference recognizers (1 audited commit)
  - `cf1fc8371f3d` feat: recognize mobile-money billing identifiers (#1683)
- [#1684](https://github.com/maziyarpanahi/openmed/pull/1684) Add Kenya KMHFL and Nigeria HFR health-facility code support (1 audited commit)
  - `a5350ff8a2da` feat: add African health facility code support (#1684)
- [#1685](https://github.com/maziyarpanahi/openmed/pull/1685) Add Hausa Boko and Ajami PII pack (1 audited commit)
  - `64758633f9fc` feat: add Hausa Boko and Ajami PII pack (#1685)
- [#1855](https://github.com/maziyarpanahi/openmed/pull/1855) Add deterministic radiology finding extractor (4 audited commits)
  - `3c67a01573c1` Add deterministic radiology finding extractor
  - `ea240d9e5dd0` Merge remote-tracking branch 'origin/master' into pr-1855
  - `0ffb352c081d` fix: complete radiology finding extraction
  - `241ed36bff90` Merge pull request #1855 from Udaytaneja/feature/radiology-finding-extractor
- [#1686](https://github.com/maziyarpanahi/openmed/pull/1686) Yoruba language pack with combining-diacritic-safe span offsets (1 audited commit)
  - `b1541bd11e30` feat: add Yoruba PII pack with grapheme-safe offsets (#1686)
- [#1687](https://github.com/maziyarpanahi/openmed/pull/1687) Add Igbo language pack for Nigerian clinical text (1 audited commit)
  - `78e45e6cc674` feat: add Igbo PII language pack (#1687)
- [#1688](https://github.com/maziyarpanahi/openmed/pull/1688) African French and Portuguese locale surrogate providers (1 audited commit)
  - `bde6c377fcf9` feat: add African French and Portuguese locale surrogates (#1688)
- [#1689](https://github.com/maziyarpanahi/openmed/pull/1689) Add Egypt PDPL and Morocco Law 09-08 profiles (1 audited commit)
  - `5cff74175c39` feat: add Egypt and Morocco privacy profiles (#1689)
- [#1690](https://github.com/maziyarpanahi/openmed/pull/1690) Add Africa data-residency deployment guide and attestation reports (1 audited commit)
  - `49bf7a4f2d63` feat: add Africa data-residency attestations (#1690)
- [#1691](https://github.com/maziyarpanahi/openmed/pull/1691) Add African healthcare-context safety-sweep terms (1 audited commit)
  - `2e8ecd28486e` feat: add African context safety-sweep terms (#1691)
- [#1692](https://github.com/maziyarpanahi/openmed/pull/1692) OpenHIE mediator packaging: run the de-identification service as an OpenHIM mediator (1 audited commit)
  - `bcc9d19c62fb` feat: add OpenHIM mediator packaging (#1692)
- [#1694](https://github.com/maziyarpanahi/openmed/pull/1694) feat: de-identify community health worker form exports (1 audited commit)
  - `b06833981cdf` feat: de-identify CHW form exports (#1694)
- [#1695](https://github.com/maziyarpanahi/openmed/pull/1695) feat: add WHO SMART Guidelines FHIR profile checks (1 audited commit)
  - `6e41f435b015` feat: check SMART FHIR profile conformance (#1695)
- [#1696](https://github.com/maziyarpanahi/openmed/pull/1696) Add model size budget command (1 audited commit)
  - `d0a8e7d5a524` feat: add model size budget command (#1696)
- [#1701](https://github.com/maziyarpanahi/openmed/pull/1701) Add Swahili README and African developer onboarding (1 audited commit)
  - `73fa605ba9c8` Add Swahili README and African developer onboarding (#1701)
- [#1703](https://github.com/maziyarpanahi/openmed/pull/1703) Add ARM SMS latency benchmark and budget gate (1 audited commit)
  - `f0c6f1f925f0` Add ARM SMS latency benchmark and budget gate (#1703)
- [#1645](https://github.com/maziyarpanahi/openmed/pull/1645) Add Simplified/Traditional Chinese conversion with offset-preserving alignment (1 audited commit)
  - `348014388991` Add Simplified/Traditional Chinese conversion with offset-preserving alignment (#1645)
- [#1646](https://github.com/maziyarpanahi/openmed/pull/1646) Add Chinese numeral normalization for dates, IDs, and quantities (1 audited commit)
  - `b2de38e0a720` Add Chinese numeral normalization for dates, IDs, and quantities (#1646)
- [#1657](https://github.com/maziyarpanahi/openmed/pull/1657) Add token-level Hinglish language routing for de-identification (1 audited commit)
  - `891d01c61c8d` Add token-level Hinglish language routing for de-identification (#1657)
- [#1659](https://github.com/maziyarpanahi/openmed/pull/1659) Add Indian multi-identifier recognizer pack (1 audited commit)
  - `1e4258d61211` Add Indian multi-identifier recognizer pack (#1659)
- [#1672](https://github.com/maziyarpanahi/openmed/pull/1672) docs: add Chinese and Hindi site locales (1 audited commit)
  - `81bbc7e7c1f5` docs: add Chinese and Hindi site locales (#1672)
- [#1677](https://github.com/maziyarpanahi/openmed/pull/1677) Harden multilingual ingestion boundaries (1 audited commit)
  - `898073dc2372` Harden multilingual ingestion boundaries (#1677)
- [#1693](https://github.com/maziyarpanahi/openmed/pull/1693) Add offline ICD-11 MMS snapshot grounding (1 audited commit)
  - `a8a5d5057bdf` Add offline ICD-11 MMS snapshot grounding (#1693)
- [#1699](https://github.com/maziyarpanahi/openmed/pull/1699) Add mirror and proxy installation guidance (1 audited commit)
  - `886184d5c6ca` Add mirror and proxy installation guidance (#1699)
- [#1842](https://github.com/maziyarpanahi/openmed/pull/1842) Build an adversarial-PHI red-team corpus and harness for the redactor (1 audited commit)
  - `6927964116e9` Build an adversarial-PHI red-team corpus and harness for the redactor (#1842)
- [#1858](https://github.com/maziyarpanahi/openmed/pull/1858) Fix PharmaDetect entity boundaries and medication filtering (1 audited commit)
  - `aab326d9c2c8` fix: improve PharmaDetect entity precision (#1858)
- [#1704](https://github.com/maziyarpanahi/openmed/pull/1704) Add Nordic PII language packs (Swedish, Danish, Norwegian) (1 audited commit)
  - `8d4af5c5315c` feat: add Nordic PII language packs (#1704)
- [#1705](https://github.com/maziyarpanahi/openmed/pull/1705) Add GDPR and EU AI Act compliance templates (1 audited commit)
  - `e32661ec8a8f` docs: add GDPR and EU AI Act templates (#1705)
- [#1707](https://github.com/maziyarpanahi/openmed/pull/1707) Add LlamaIndex node redaction postprocessor (1 audited commit)
  - `fb5bed7d6c92` feat: add node redaction postprocessor (#1707)
- [#1708](https://github.com/maziyarpanahi/openmed/pull/1708) Add a WASM/WebGPU browser inference demo and load-time benchmark page (1 audited commit)
  - `f69a51a49895` feat: add browser PII benchmark demo (#1708)
- [#1752](https://github.com/maziyarpanahi/openmed/pull/1752) Add a synthetic gold-corpus annotation toolkit with BRAT and CoNLL IO (1 audited commit)
  - `28e8f3f790a7` feat: add gold-corpus annotation toolkit (#1752)
- [#1753](https://github.com/maziyarpanahi/openmed/pull/1753) Add ISO 27701/27001 control-evidence pack generator (1 audited commit)
  - `64b9fa8d6119` feat: add ISO control evidence pack generator (#1753)
- [#1754](https://github.com/maziyarpanahi/openmed/pull/1754) Add consent and data-use tag enforcement (1 audited commit)
  - `bf9404852e41` feat: enforce consent data-use tags (#1754)
- [#1756](https://github.com/maziyarpanahi/openmed/pull/1756) Add an AWQ grounding embedder recall gate (1 audited commit)
  - `b789714781b4` feat: add AWQ grounding recall gate (#1756)
- [#1757](https://github.com/maziyarpanahi/openmed/pull/1757) Add a per-language leakage dashboard renderer over benchmark runs (1 audited commit)
  - `c1129ee870b1` Add a per-language leakage dashboard renderer over benchmark runs (#1757)
- [#1759](https://github.com/maziyarpanahi/openmed/pull/1759) Add a discharge-summary section structurer with typed slots (1 audited commit)
  - `1e88f6d0b970` feat: add discharge-summary section structurer (#1759)
- [#1760](https://github.com/maziyarpanahi/openmed/pull/1760) Add a synthetic tabular-data generator preserving column distributions (1 audited commit)
  - `80f98267fa4e` Add a synthetic tabular-data generator preserving column distributions (#1760)
- [#1763](https://github.com/maziyarpanahi/openmed/pull/1763) Add ISCII and legacy-font Devanagari conversion (1 audited commit)
  - `0904af6004f4` Add ISCII and legacy-font Devanagari conversion (#1763)
- [#1764](https://github.com/maziyarpanahi/openmed/pull/1764) Add conservative Indic morphology boundary refinement (1 audited commit)
  - `b943ee163335` Add conservative Indic morphology boundary refinement (#1764)
- [#1765](https://github.com/maziyarpanahi/openmed/pull/1765) Add transliteration-robust Indian name matching (1 audited commit)
  - `3898026e8fec` Add transliteration-robust Indian name matching (#1765)
- [#1860](https://github.com/maziyarpanahi/openmed/pull/1860) feat: Added Russian (ru) PII language package (5 audited commits)
  - `b82d51db0c9d` Added ru PII package
  - `763a673591b2` Updated CHANGELOG
  - `24e5f9b33ae1` Merge remote-tracking branch 'origin/master' into review/pr-1860
  - `0ba6ddac192f` fix: complete Russian PII language pack
  - `37937a07dbb6` Merge pull request #1860 from mrfeathers/featire/om-293-ru-pii-language
- [#1766](https://github.com/maziyarpanahi/openmed/pull/1766) Add optional MuRIL and IndicBERT encoder backbones (1 audited commit)
  - `f00b739a0efb` Add optional MuRIL and IndicBERT encoder backbones (#1766)
- [#1857](https://github.com/maziyarpanahi/openmed/pull/1857) feat: add scrubadub adapter to openmed.interop (5 audited commits)
  - `f43157e68795` feat: add scrubadub adapter to openmed.interop
  - `d148ce9d036d` Merge remote-tracking branch 'origin/master' into review/pr-1857
  - `27a083bc854e` fix: complete scrubadub adapter integration
  - `8ef1ebf7aa07` Merge remote-tracking branch 'origin/master' into review/pr-1857
  - `bc61cc6bbf4b` Merge pull request #1857 from affanhamid/feat/scrubadub-interop-adapter
- [#1859](https://github.com/maziyarpanahi/openmed/pull/1859) feat(eval): add release-readiness gate aggregating shippability checks (#1814) (6 audited commits)
  - `f08924a8d354` feat(eval): add release-readiness gate aggregating shippability checks (#1814)
  - `4811144c251f` Merge remote-tracking branch 'origin/master' into review/pr-1859
  - `077edcc33e11` fix: complete release readiness gate
  - `e7d7fab39f14` Merge remote-tracking branch 'origin/master' into review/pr-1859
  - `dfac7315bdb8` fix: use POSIX path separators in _display_path for cross-platform CI
  - `6a6f33808723` Merge pull request #1859 from JonthanaHanh/feat/release-readiness-gate-1814-v2
- [#1706](https://github.com/maziyarpanahi/openmed/pull/1706) Add scispaCy/QuickUMLS approximate-linker adapters (1 audited commit)
  - `8c0adbaaade3` feat: add UMLS linker adapters (#1706)
- [#1751](https://github.com/maziyarpanahi/openmed/pull/1751) Add a model-sharding and streaming weight loader for low-RAM devices (1 audited commit)
  - `98c032dad909` Add a model-sharding and streaming weight loader for low-RAM devices (#1751)
- [#1846](https://github.com/maziyarpanahi/openmed/pull/1846) Add coverage-guided fuzzing for document format parsers (1 audited commit)
  - `cfbcdc7a7e96` Add coverage-guided fuzzing for document format parsers (#1846)
- [#1758](https://github.com/maziyarpanahi/openmed/pull/1758) Add regression escape tracker dashboard (1 audited commit)
  - `9fb493e99483` Add regression escape tracker dashboard (#1758)
- [#1847](https://github.com/maziyarpanahi/openmed/pull/1847) Add multilingual clinical relation extraction (1 audited commit)
  - `6aa5df6b3cc3` feat: add multilingual relation extraction (#1847)
- [#1767](https://github.com/maziyarpanahi/openmed/pull/1767) Add Central and Eastern European PII language packs (1 audited commit)
  - `0189805b8145` feat: add Central and Eastern European PII packs (#1767)
- [#1768](https://github.com/maziyarpanahi/openmed/pull/1768) Add GGUF embedding-backbone export for grounding retrieval (1 audited commit)
  - `a1b21b3d7631` feat: add GGUF embedding backbone export (#1768)
- [#1769](https://github.com/maziyarpanahi/openmed/pull/1769) Add a Nix flake for reproducible builds and dev shells (1 audited commit)
  - `1adc8abdda22` Add a Nix flake for reproducible builds and dev shells (#1769)
- [#1777](https://github.com/maziyarpanahi/openmed/pull/1777) feat: add watchOS and visionOS OpenMedKit targets (1 audited commit)
  - `0d4b5f6081ff` feat: add watchOS and visionOS OpenMedKit targets (#1777)
- [#1865](https://github.com/maziyarpanahi/openmed/pull/1865) Build exact Chinese character-to-word offset mapping (1 audited commit)
  - `ad9c1b987026` feat: add Chinese character-word offset mapping (#1865)
- [#1854](https://github.com/maziyarpanahi/openmed/pull/1854) feat(clinical): add serial measurement and trend extractor (4 audited commits)
  - `ff25ee23685e` feat(clinical): add serial measurement and trend extractor
  - `cb133e1d07ca` Merge remote-tracking branch 'origin/master' into HEAD
  - `51698d4ff879` fix: complete measurement trend API and provenance
  - `a51e666086db` Merge pull request #1854 from PouyanJay/feat/serial-measurement-trend
- [#1885](https://github.com/maziyarpanahi/openmed/pull/1885) feat: add procedures zero-shot domain and DEVICE canonical label (4 audited commits)
  - `172e4b237e36` feat: add procedures zero-shot domain and DEVICE canonical label
  - `bfd493cc5235` fix: align clinical equipment with device taxonomy
  - `c6a26b8fd7f7` Merge remote-tracking branch 'origin/master' into HEAD
  - `f0266be4b69b` Merge pull request #1885 from RonitGandhi/fix/issue-313
- [#1892](https://github.com/maziyarpanahi/openmed/pull/1892) feat: add PySpark pandas_udf for batch de-identification (6 audited commits)
  - `e626cb6e39cf` feat: add PySpark pandas_udf for batch de-identification
  - `767fb47fd33e` fix: complete Spark UDF runtime extra
  - `9cbda1aee40a` fix: update GitPython past vulnerable releases
  - `3f3a245f596c` Merge remote-tracking branch 'origin/master' into HEAD
  - `d54d6758890e` Merge remote-tracking branch 'origin/master' into HEAD
  - `3e4f67b74615` Merge pull request #1892 from affanhamid/feat/spark-deidentify-udf
- [#1893](https://github.com/maziyarpanahi/openmed/pull/1893) Document ONNX and WebGPU export (2 audited commits)
  - `eb6ed0341a9c` Document ONNX and WebGPU export
  - `47d626010764` Merge pull request #1893 from alberthammerich/docs-onnx-webgpu-export
- [#1869](https://github.com/maziyarpanahi/openmed/pull/1869) Add grapheme-safe scalar span parity for Swift de-identification (1 audited commit)
  - `e21db5e29484` feat: add grapheme-safe scalar span parity (#1869)
- [#1873](https://github.com/maziyarpanahi/openmed/pull/1873) Add Marathi PII language pack (1 audited commit)
  - `2f6dcee26a93` feat: add Marathi PII language pack (#1873)
- [#1888](https://github.com/maziyarpanahi/openmed/pull/1888) Add a consumer agent-usage guide and ready-to-use repository skills (1 audited commit)
  - `b266684cf72d` feat: add agent usage guide and repository skills (#1888)
- [#1889](https://github.com/maziyarpanahi/openmed/pull/1889) Add Afrikaans PII language pack via Dutch pattern transfer (1 audited commit)
  - `86f3427a9f26` feat: add Afrikaans PII language pack (#1889)
- [#1867](https://github.com/maziyarpanahi/openmed/pull/1867) Add CJK-aware span decoding for Chinese text (1 audited commit)
  - `3cd0f0fb3a62` Add CJK-aware span decoding for Chinese text (#1867)
- [#1872](https://github.com/maziyarpanahi/openmed/pull/1872) Add Tamil PII language pack with native surrogates (1 audited commit)
  - `a5d42d46f5a0` Add Tamil PII language pack with native surrogates (#1872)
- [#1879](https://github.com/maziyarpanahi/openmed/pull/1879) Add path-filtered CJK and Indic fixture CI job (1 audited commit)
  - `92a85f1d3a3a` Add path-filtered CJK and Indic fixture CI job (#1879)
- [#1880](https://github.com/maziyarpanahi/openmed/pull/1880) Add Chinese and Indic throughput release gates (1 audited commit)
  - `79219f203ba8` Add Chinese and Indic throughput release gates (#1880)
- [#1882](https://github.com/maziyarpanahi/openmed/pull/1882) Publish Chinese and Indic PII registry metadata and model cards (1 audited commit)
  - `1220bf3dee02` Publish Chinese and Indic PII registry metadata and model cards (#1882)
- [#1887](https://github.com/maziyarpanahi/openmed/pull/1887) Ship an MCP-enabled container image and compose service (1 audited commit)
  - `c9aa1f5c784e` feat: add MCP container service (#1887)
- [#1890](https://github.com/maziyarpanahi/openmed/pull/1890) Add pan-African Malabo baseline and policy coverage eval (1 audited commit)
  - `90035ff6215a` Add pan-African Malabo baseline and policy coverage eval (#1890)
- [#1891](https://github.com/maziyarpanahi/openmed/pull/1891) African deployment reference: facility EMR to national HMIS synthetic demo (1 audited commit)
  - `8c5cf78ff975` feat: add African OpenMRS to DHIS2 reference (#1891)
- [#1883](https://github.com/maziyarpanahi/openmed/pull/1883) Add Pinyin romanization and deterministic Chinese name surrogates (1 audited commit)
  - `3e16ae7f2f24` Add Pinyin romanization and deterministic Chinese name surrogates (#1883)
- [#1884](https://github.com/maziyarpanahi/openmed/pull/1884) Add an Odia (or) PII language pack with native or_IN surrogates and Bengali-script confusion guards (1 audited commit)
  - `936fdd8dc3cf` feat: add Odia PII language pack (#1884)
- [#1886](https://github.com/maziyarpanahi/openmed/pull/1886) Add Assamese PII language pack with Bengali-script disambiguation (1 audited commit)
  - `04e0fd4c6e5d` feat: add Assamese PII language pack (#1886)
- [#1894](https://github.com/maziyarpanahi/openmed/pull/1894) feat(clinical): add longitudinal document near-duplicate hash and cop… (5 audited commits)
  - `c110ef63f4f0` feat(clinical): add longitudinal document near-duplicate hash and copy-forward linker
  - `abc2c93d52e1` fix: complete document linking provenance and safety
  - `75b8ef2f6ad3` Merge remote-tracking branch 'origin/master' into HEAD
  - `7f64df515df6` Merge remote-tracking branch 'origin/master' into HEAD
  - `f66f4f155c3a` Merge pull request #1894 from eslam-ahmed43/feat/document-linking-om-834
- [#1866](https://github.com/maziyarpanahi/openmed/pull/1866) docs: one-command multi-agent install and quickstart for the skills catalog (5 audited commits)
  - `023b206f8ad0` docs: one-command multi-agent install and quickstart for the skills catalog
  - `e44cd9aef37f` Merge remote-tracking branch 'origin/master' into feature/skills-multi-agent-readme
  - `f2010191d2ff` fix: harden multi-agent skills onboarding
  - `dfa492bfe183` test: make skills installer checks portable
  - `7d592dd86b3f` Merge pull request #1866 from maziyarpanahi/feature/skills-multi-agent-readme
- [#1900](https://github.com/maziyarpanahi/openmed/pull/1900) Render the MCP server from the tool registry with annotations and structured output (2 audited commits)
  - `ea654a7a24d0` feat: add structured registry tool metadata
  - `5c005e3c5aff` Merge pull request #1900 from maziyarpanahi/feature/om-394-mcp-server-registry-annotations-structured-outpu

</details>

<details>
<summary>Direct, integration, and release-preparation commits</summary>

- `364c4f3b116d` fix: preserve multi-arch release images
- `410da369ac43` Update README.md
- `70140723c86c` chore: set package version to 2.0.0
- `ada930a1642e` fix: harden v2 package build inputs
- `cb961e2644eb` chore: refresh the v2 dependency lock
- `17b42800aa2c` test: guard root-anchored package inputs
- `972573949933` fix: allow explicit models for pattern-only languages
- `58a4200b365e` test: cover explicit Afrikaans model routing
- `f5de56f4958f` fix: resolve spaCy factory annotations eagerly
- `9af355db3686` ci: align v2 release gates with v1.9.1
- `433868420bf4` docs: record the OpenMed 2.0.0 release
- `d0c057c54a92` docs: add the 1.9 to 2.0 migration guide
- `0fc824d4d175` docs: add OpenMed 2.0.0 release notes
- `765cf1cd820f` docs: add v2 release pages to navigation
- `e607c7618469` docs: make 2.0.0 the current documentation release
- `c65be2e2ff46` docs: update the Hindi landing page for v2
- `f4e7653be094` docs: update the Chinese landing page for v2
- `98e717b44e1b` docs: link v2 compatibility guidance from the feature map
- `92f81d23059f` docs: finalize the v2 migration contract
- `c57e34475a5f` docs: update example installation for v2
- `a23d7e91bf1e` docs: update REST health output for v2
- `848b309afcd2` docs: update Android quickstart for v2
- `c37a4ae2924d` docs: update Android export coordinates for v2
- `7593d831c158` docs: update OpenMedKit installation for v2
- `0eb3214bd05b` docs: update Helm deployment examples for v2
- `0177ba797889` docs: update provenance verification for v2
- `ef1037838ed8` docs: update the website release metadata to v2
- `5d6b8201d8ff` docs: regenerate the v2 OpenAPI artifact
- `9b4e46829092` docs: refresh the v2 benchmark leaderboard page
- `5c9a2fbead1a` docs: refresh the v2 benchmark leaderboard data
- `21f60ac9c52a` docs: update the main README for v2
- `f2ae630c6b26` docs: update the Arabic Swift release coordinate
- `af4db2d9c5e4` docs: update the German Swift release coordinate
- `3fd66cd8882d` docs: update the Spanish Swift release coordinate
- `b7c88fa21a58` docs: update the Persian Swift release coordinate
- `5cab138f6e75` docs: update the French Swift release coordinate
- `b8fa555db4e0` docs: update the Hindi release coordinates
- `06cf6ced22a6` docs: update the Italian Swift release coordinate
- `d888e107b7f5` docs: update the Japanese Swift release coordinate
- `55e9c5f510ca` docs: update the Dutch Swift release coordinate
- `3109d62e3ec8` docs: update the Portuguese Swift release coordinate
- `3493b3a93918` docs: update the Swahili release coordinates
- `f17acb2cdbb5` docs: update the Telugu Swift release coordinate
- `2b471993c8a5` docs: update the Turkish Swift release coordinate
- `41b09ae8eb26` docs: update the Chinese release coordinates
- `08a5ca4ad2fb` docs: refresh README translation hashes for v2
- `e40cfc5cf156` docs: update Android installation to v2
- `20cc658eb2fd` docs: update the Android library guide for v2
- `7577d55e362c` chore: set the Android library version to 2.0.0
- `eb80e34cbb7f` test: expect the Android 2.0.0 version
- `ac25b508a33b` chore: set the Helm app version to 2.0.0
- `eeb74a94ec4d` chore: set the default Helm image to 2.0.0
- `1dbd4a601883` test: use the v2 image in Helm CI values
- `03a039903df1` test: expect the v2 Helm image
- `c787310511ec` chore: set the web package version to 2.0.0
- `433a03a4ff77` chore: lock the web package at 2.0.0
- `17cae3705305` chore: set the OpenMed demo version to 2.0.0
- `60e04927df1f` chore: set the scan demo version to 2.0.0
- `1967dd381c4b` docs: update the OpenHIM mediator example to v2
- `2375f4c671b5` docs: update the de-identification demo dependency to v2
- `478fec399c06` fix: load converter-marked legacy MLX bundles
- `5c8cbc38f04a` test: cover legacy MLX Hub artifacts
- `ca5b7d3195a1` docs: record legacy MLX compatibility fix
- `252eae93c125` fix: route local privacy-filter artifacts by format
- `be1d3063d841` test: cover local privacy-filter backend routing
- `5dc58c3ae66e` docs: record privacy-filter routing fix
- `a0c1bdf15afc` Add structured quasi-identifier detection
- `e20f01631983` feat: automate structured release risk analysis
- `b3be34286aab` Update CHANGELOG.md
- `d75cfe82b6b9` Update examples.md
- `c5a78fe488cf` Update reidentification-risk.md
- `d3f4bae79571` Create structured_population_risk.py
- `322cc392db19` Update main.py
- `e1219a398b41` Update __init__.py
- `77fdab8eadf4` Create expert_attestation.py
- `6e74ff35eb40` Update expert_review.py
- `0857fea13d74` Update release_evidence.py
- `f32ac2c51608` Update release_gates.py
- `a2ba9755ffdf` Update __init__.py
- `216b6b207eb1` Update dashboard.py
- `1f20972a82fd` Update kanon.py
- `1c81b6460c8b` Create population.py
- `2e32512d8b7c` Update reid.py
- `2d6c4347e1a6` Update release.py
- `404da87e0287` Update qi_detect.py
- `0b08cd9e735a` Update test_risk_release_cli.py
- `e6aa77bca87d` Create test_expert_attestation.py
- `af37d2083b70` Update test_expert_review.py
- `c12b98e804e4` Update test_release_evidence.py
- `64fbaecb5046` Create test_unicode_attribute_names.py
- `da0277fabe57` Update test_audit_report.py
- `5a533ef31a7d` Update test_release_gates.py
- `953bfc97a12b` Create test_direct_identifier_names.py
- `92e124bab1f3` Update test_kanon_enforcement.py
- `c5d1a7af6475` Create test_population_risk.py
- `0e90cf6d2c1e` Update test_release.py
- `e6ed73d4d803` Update test_risk_dashboard.py
- `98b0695758c5` Create test_unicode_column_names.py
- `8756388a7246` Update test_qi_detect.py
- `b6c07862e5a7` Merge branch 'feature/automated-qi-risk-analysis' into release/openmed-200
- `bbfcad9f8467` fix: expose only implemented clinical benchmark tasks
- `cf4f5e914fed` test: cover clinical benchmark task choices
- `288dda9784e8` docs: finalize v2 release notes
- `767612e3cb0b` docs: refresh v2 migration inventory
- `458b7d9fd3c2` docs: synchronize Hindi language coverage
- `679ecb847f75` docs: synchronize website language coverage
- `98c54a5420b5` fix: require secure ONNX dependency routes
- `48eb9bc32b30` chore: refresh the secure dependency lock
- `4359141365d3` docs: record secure optional dependency routes
- `59b18a4b63b3` fix: exclude generated evidence from images
- `063dee8df7fa` test: protect image build contexts
- `f091ef635687` docs: complete the v2 release inventory
- `cd38b4647215` fix: exclude JavaScript dependencies from images
- `7c4e407a9548` test: keep JavaScript dependencies out of images
- `0402798c8713` docs: finalize the v2 release inventory
- `fa4c4ad2a190` test: remove flaky fuzz timing assertions
- `e83dd4766090` docs: record the Windows fuzz fix
- `28d5036c39de` legal: add the ICU license notice
- `271b9c1af457` legal: pin the ICU segmenter provenance
- `8807dace9552` fix: require ICU notices in segmenter bundles
- `cc7637a81a7a` fix: discover the ICU bundle notice
- `5a735b023b2b` fix(swift): validate ICU bundle attribution
- `00a192eb3e25` test(swift): cover the ICU bundle notice
- `f2bd824ec4a8` test: enforce ICU segmenter attribution
- `b8c1970aa88c` test(mlx): require the ICU bundle notice
- `4ef4a2c347de` test(coreml): require the ICU bundle notice
- `591d2cc56990` test(onnx): require the ICU bundle notice
- `c157e00a686b` test(web): require the ICU bundle notice
- `bf092e2605d3` fix: audit bundled license notices
- `333d8e929569` test: guard bundled ICU licensing
- `82d7882d45d7` docs: attribute the bundled ICU rules
- `1162b5009b28` docs: document ICU bundle attribution
- `677eaff81fbc` docs: note the ICU manifest correction
- `6525adb5722c` docs: record ICU attribution hardening

</details>

## [1.9.1] - 2026-07-14

This patch completes the `1.9` distribution rollout without changing the
public inference APIs introduced in `1.9.0`.

### Fixed

- Restored the documented root Swift Package Manager build by processing
  OpenMedKit policy resources in the root package, and moved Swift CI to build
  and test that public package entry point.
- Kept tag-driven Android validation green when the optional Maven Central
  signing credentials are absent while retaining the immutable JitPack release
  path and guarded manual Central uploads.
- Replaced placeholder model repository IDs in runtime and export documentation
  with tested public token-classification and causal-model examples.

### Security

- Updated the locked `setuptools` build dependency to a non-vulnerable release
  so the master and release `pip-audit` gates pass without a waiver.

## [1.9.0] - 2026-07-14

This release adds one model-repository contract for ONNX token-classification
inference across Python, browsers, Node.js, and Android, then extends the
clinical, multilingual privacy, evaluation, documentation, and developer
surfaces delivered after `v1.8.1`.

### Added

- Added concise cross-platform ONNX inference APIs: `OnnxModel` for Python CPU,
  `loadOnnxModel` for WebGPU/WebAssembly, and `OpenMedKit.fromDirectory` for
  Android with Hugging Face tokenizer offset parity. The same exported model
  repository can now serve every supported runtime without application-level
  tokenizer or tensor plumbing (#1550).
- Added a resumable Android ONNX batch rollout runner, Android/ORT model-card
  format metadata, immutable Git-tag installation through JitPack, and runnable
  MLX examples for token classification and GLiNER zero-shot NER (#1550).
- Added the public `openmed` npm package for browser and Node.js inference, with
  synchronized release versions, ESM/CommonJS exports, WebGPU/WebAssembly
  examples, package tests, npm audit enforcement, and provenance-backed tag
  publishing (#1550).
- Added Hugging Face Hub model-pull convenience helpers and artifact-backed
  model-card datasheets generated from provenance-hashed evaluation evidence
  (#1339, #1228).
- Added an offline immunization zero-shot domain with FHIR-aligned display
  labels, canonical policy metadata, synthetic per-label fixtures, and
  exporter-alignment documentation (#1159).
- Added an offline pediatrics-growth zero-shot domain with growth-parameter,
  percentile, z-score, developmental-milestone, feeding-history, and finding
  display labels, canonical policy metadata, and synthetic per-label fixture
  coverage (#1611).
- Added relation metrics, a synthetic gold loader, strict and relaxed clinical
  relation-extraction scoring, an RE release gate, and a dataframe API for
  clinical extraction results (#1211, #1212, #1224).
- Added a Go REST client, a Postman collection, copy-paste REST recipes, and a
  Jupyter/IPython rich display widget for de-identification results (#1379,
  #1385, #1387, #1382).
- Added full Korean (`ko`) and Romanian (`ro`) PII language packs, including
  native identifier validation, locale-aware surrogates, synthetic fixtures,
  and model/service wiring. The model-backed PII allow-list now covers 17
  language codes (#1544, #1389).
- Added Canadian SIN and provincial health-card validators, Australian Medicare
  and TFN validators, CJK family-name-first honorific stripping, RTL-aware
  redacted-output rendering, multilingual clinical section detection,
  translation augmentation for low-resource NER, and multilingual surrogate
  quality gates (#1340, #1342, #1346, #1384, #1226, #1221, #1225).
- Added critical-finding recall and leakage-under-extraction safety gates, a
  multilingual clinical NER benchmark aggregator, a false-negative explorer,
  throughput-versus-accuracy frontier reporting, inference memory profiling,
  property-based de-identification fuzzing, a burned-in-PHI DICOM benchmark,
  and a redactor threat model with leakage-bypass abuse cases (#1213, #1214,
  #1223, #1343, #1347, #1349, #1350, #1388, #1352).
- Added a public-API docstring coverage check plus PEP 561 `py.typed` packaging
  and scoped type-hint coverage for the expanded module surface (#1341, #1348).
- Added persona quickstarts, hardened offline model loading, and a dedicated
  troubleshooting and common-errors guide (#1386, #1380).

### Changed

- Consolidated OpenMed's brand and on-device clinical-AI messaging across the
  repository, documentation, and website (#1415).
- Bounded the ten-stage clinical pipeline on long notes and strengthened offline
  loading paths used by the new runtime quickstarts (#1383, #1386).

### Fixed

- Hardened Android ONNX export and publishing for large external-data graphs,
  Longformer tracing, fp16 metadata, dynamic INT8 graph ordering, optional ORT
  conversion failures, existing Hub repositories, and models that require
  zero-valued `token_type_ids` at runtime (#1550).
- Made Android artifact reuse fail closed when runtime files or ONNX external
  data are missing, required `tokenizer.json` before publication, and kept
  resumable batch cleanup from deleting valid artifacts (#1550).
- Made direct `OpenMed/...-mlx` repository IDs resolve as pre-converted MLX
  artifacts and kept optional tokenizer loading lazy, avoiding unintended
  PyTorch conversion and eager pandas imports (#1550).
- Aligned Python, npm, Swift, Android, Helm, OpenAPI, container, and demo release
  surfaces on `1.9.0` and immutable release coordinates (#1550).

### Security

- Replaced fixable vulnerability waivers with dependency and base-image
  upgrades, and made the vulnerability gate reject waivers when a fixed version
  is available (#1550).
- Added explicit release gates for critical-finding recall and leakage under
  clinical extraction, alongside the redactor threat model, de-identification
  fuzz harness, and synthetic burned-in-PHI DICOM benchmark (#1213, #1214,
  #1350, #1352, #1388).
## [1.8.1] - 2026-07-10

### Fixed

- Fixed automatic PyTorch attention selection so `auto` no longer forces SDPA onto Transformers architectures that do not support it, including `DebertaV2ForTokenClassification`; explicit `eager`, `sdpa`, and `flash_attention_2` selections remain available.
- Changed unavailable accelerated-attention fallbacks to use the architecture-independent eager implementation instead of selecting another accelerated backend from runtime capability alone.

## [1.8.0] - 2026-07-09

This release summarizes the cross-platform runtime, service hardening, multimodal privacy, clinical extraction, and release-evidence work merged after `v1.7.0`. The reviewed range is broad: 434 commits from `v1.7.0` through the final `release/openmed-180` branch tip prepared for the `v1.8.0` tag, covering Android, browser, and React Native runtimes, production service controls, structured health-data pipelines, and the privacy/evaluation gates that keep those surfaces aligned.

### Added

- Added the Android OpenMedKit surface: a Gradle project, Kotlin public API, token-classification decoder, ONNX and ORT Mobile paths, ML Kit OCR adapter, model catalog/download cache, document/image intake, Compose demo, scan demo, Python-to-Android span parity fixtures, Android CI, and guarded Maven Central publishing (#1114, #1115, #1116, #1117, #1118, #1119, #1120, #1121, #1122, #1123, #1124, #1146, #1148, #1149, #1150, #1155, #1156, #1161, #1162).
- Added browser, mobile JavaScript, and cross-platform client runtimes, including a typed OpenMedKit web package for Transformers.js/ONNX Runtime Web, a React Native bridge, Swift-Kotlin parity checks, public API parity coverage, and a typed TypeScript service client surface (#1132, #1177, #1178, #1123).
- Added production service and deployment capabilities: API-key/JWT auth, request correlation IDs, no-PHI JSON logging, OpenTelemetry tracing, gRPC, async jobs and webhooks, Helm deployment, multi-arch containers, circuit breakers, model-load retry/backoff, privacy-gateway redaction before external calls, SMART-on-FHIR bulk ingestion, object-storage batch runs, Spark/Dask/lakehouse/columnar redaction, DuckDB and pandas/polars accessors, agent/MCP tool orchestration, hardened distroless images, image signing, SLSA provenance, container SBOMs, and vulnerability scanning (#1080, #1081, #1082, #1084, #1109, #1110, #1126, #1127, #1129, #1130, #1131, #1133, #1136, #1138, #1139, #1140, #1141, #1143, #1144, #1152, #1153, #1154, #1175, #1176, #1179, #1180, #1185, #1189).
- Added deeper clinical extraction and interoperability: normalized clinical timelines, document assertion graphs, clinical event frames, medication relation decoding, concept normalization, UCUM units, free vocabulary grounding, RxNorm/ICD-10-CM/HPO linkers, CodeableConcept export, deterministic CDM extraction, OMOP CDM loader foundation, GDPR DSAR export, and severity/laterality, clinical-genomics, gastroenterology, endocrinology, nutrition/diet, and anesthesia domain coverage (#1019, #1022, #1025, #1026, #1027, #1079, #1086, #1105, #1134, #1135, #1137, #1160, #1164, #1165, #1166, #1167, #1170, #1182, #1183, #1184, #1187, #1219, #1292, #1299).
- Added multimodal and structured privacy coverage for DOCX offset extraction, plain-image redaction, DICOM header de-identification, burned-in DICOM pixel OCR redaction, redacted-PDF text-layer fidelity checks, EPUB extraction, vCard/iCalendar PHI redaction, UK health identifiers, IBAN/SWIFT/BIC identifiers, passport/MRZ validation, and additional validator-backed ID packs for Slovak, Latvian, Malay, Filipino, and Danish locales (#1093, #1098, #1106, #1107, #1108, #1112, #1128, #1142, #1163, #1171, #1173, #1186, #1188, #1406).
- Added evaluation, model, and release evidence infrastructure: streaming token classification, speculative MLX PII decoding, QLoRA smoke recipes, leakage-weighted distillation, Core ML and ONNX optimization/parity gates, OpenVINO export, paged KV-cache attention, memory-budgeted model scheduling, benchmark ledgers, active-learning gate queues, hard-negative mining, cross-lingual transfer evaluation, model-card/datasheet generation, flakiness quarantine, conformal calibration and abstention, mobile performance benchmarking, comparator matrices, load-test harnesses, and training provenance reproducibility gates (#1002, #1003, #1009, #1014, #1015, #1016, #1017, #1018, #1036, #1054, #1055, #1056, #1062, #1063, #1064, #1065, #1066, #1097, #1113, #1147, #1151, #1172, #1220).
- Added an endocrinology zero-shot domain for glycemic and thyroid-function
  measures, hormone levels, insulin regimens, metabolic findings, and endocrine
  glands, with canonical label normalization, keyword routing metadata, and
  synthetic fixture coverage (#895).
- Added `examples/gradio_deid_app.py`, an interactive Gradio demo that runs
  `deidentify` over synthetic text with a `mask`/`replace`/`hash` method
  selector and shows the redacted output alongside the detected PII entities.
  `gradio` stays an optional, example-local dependency with a graceful install
  hint, and the example is covered by import-safe smoke tests (#484).
- Added an `OPENMED_MLX_MMAP` toggle to `openmed.mlx.models.load_model`:
  safetensors weights load through MLX's memory-mapped, lazy path by default
  (keeping cold-start peak RSS low on the phone/laptop tiers), with
  `OPENMED_MLX_MMAP=0` forcing eager materialization as a documented fallback
  for debugging (#296).

### Changed

- Extended OpenMed from a Python/Swift-centered toolkit into a coordinated Python, Swift, Kotlin/Android, TypeScript, React Native, browser, REST, gRPC, and deployment release, with parity tests and shared fixtures keeping the platform surfaces aligned.
- Updated release engineering around guarded PyPI publishing, SLSA attestations, SBOMs, signed images, static OpenAPI regeneration, reproducible release metadata, baseline-aware secret scanning, and guarded mobile/container publishing so library, container, and mobile artifacts can be validated from the same source tree (#1104, #1144, #1153, #1154, #1405).

### Fixed

- Fixed optimizer-stripped assertions, explicit UTF-8 handling, JSON decoding failures, exception chaining, iOS MLX pinning, multilingual test span offsets, Pages deployment concurrency, HPO linker test adaptation, DSAR vault-key matching, and lint cleanup after the large v1.8 merge train (#1091, #1094, #1095, #1096, #1100, #1158, #1181, #1194, #1404).

### Security

- Added and strengthened no-raw-PHI logging, offline mode socket blocking, privacy-gateway redaction before external LLM calls, policy compiler coverage proofs, DP surrogate budgeting, k-anonymity/l-diversity/t-closeness enforcement, membership-inference defenses, adversarial de-identification robustness, federated leakage evaluation, secret scanning, pre-commit hook scanning, and vulnerability gates (#189, #190, #1034, #1035, #1037, #1043, #1047, #1082, #1127, #1141, #1405).

## [1.7.0] - 2026-07-01

This release summarizes 148 pull requests merged into
`release/openmed-170` after `v1.6.0`. The diff is additive overall: 483 files
changed, with no deleted or renamed files detected in the release range.

### Added

- Added lightweight multimodal document primitives, source spans, lazy handler
  registration, `redact_document`, image redaction, PDF span coordinate
  projection, Markdown/AsciiDoc offset-preserving extraction, audit-safe image,
  PDF, and DOCX metadata scrubbing, and JSONL chat-log de-identification with
  speaker pseudonymization (#555, #567, #726, #745, #755, #758).
- Added OCR engine coverage for Tesseract, PaddleOCR, EasyOCR, docTR, and test
  engines, including OCR language selection and available-engine discovery
  (#567, #717, #749, #558).
- Added CDA/C-CDA XML, HL7 v2, CSV/TSV, FHIR `$de-identify`, FHIR Bulk NDJSON,
  deterministic FHIR Bundle, FHIR `OperationOutcome`, FHIR `Provenance` /
  `AuditEvent`, deterministic `urn:uuid`, code-system provenance,
  CodeableConcept checks, and flat-table clinical entity export helpers (#566,
  #642, #631, #629, #626, #625, #553, #705, #737, #777, #784, #689, #690).
- Added clinical extraction and normalization helpers for labs, vital signs,
  medication sigs, problem lists, summary cards, microbiology labels,
  dermatology and ophthalmology domains, clinical concept labels, and clinical
  term protection, plus deterministic substance, employment, and living-status
  normalization (#552, #410, #560, #718, #683, #773, #684, #691, #698, #767).
- Added a nutrition and diet-order zero-shot domain, four canonical nutrition
  policy labels, policy-profile coverage, routing metadata, and synthetic
  fixture coverage for diet orders and feeding routes (#951).
- Added language and locale capabilities for Indonesian, Thai, Hebrew RTL,
  PESEL, Korean RRN, Unicode script detection, locale checksum registries,
  deterministic locale PHI generation, and locale-aware date/number
  normalization (#747, #746, #748, #709, #609, #610, #614, #766).
- Added de-identification runtime features: `DeidentificationResult.to_dataframe`,
  redaction preview diffs, cross-document surrogate vaults, patient-keyed date
  shifting, format-preserving identifier redaction, minimum-necessary strength
  selection, streaming incremental de-identification, typed analyze results,
  pipeline explain traces, section stamping, and per-document risk budgets
  (#706, #695, #729, #704, #778, #779, #731, #611, #727, #785, #733).
- Added CLI surfaces for policy-aware `openmed deid`, `openmed fhir bundle`,
  `openmed models recommend`, `openmed models diff`, `openmed policy diff`,
  `openmed doctor`, `openmed gates preview`, `openmed gates bundle`,
  `openmed audit`, `openmed risk`, and active-learning queue management (#741,
  #777, #721, #780, #771, #772, #775, #735, #787, #613).
- Added service features for model warm pools, dynamic batching, request
  coalescing, rate and concurrency limits, readiness/liveness endpoints,
  opt-in Prometheus metrics, and typed Python and TypeScript REST clients
  (#632, #630, #750, #742, #722, #788, #789, #756).
- Added an in-process ASGI load-test harness with configurable concurrency
  that reports requests per second, p50/p95/p99 latency, and error rate (#461).
- Added evaluation, release-gate, and risk tooling: DrugProt and public
  biomedical NER suites, i2b2 loader, multilingual golden fixtures, dataset
  cards, fixture coverage, per-section recall, result cache, leakage heatmaps,
  membership-inference
  probe, k-anonymity/l-diversity/t-closeness metrics, audit diffs, evidence
  bundles, scorecards, threshold sweeps, flaky-run detection, paired
  significance testing, calibration reliability data, utility-loss reports,
  policy-compliance suite, cross-release benchmark history diffs, nano-tier
  certification, and risk dashboard rendering (#617, #615, #743, #701, #688,
  #703, #702, #708, #725, #680, #724, #723, #740, #735, #681, #682, #752,
  #753, #754, #762, #765, #734, #764, #744, #786).
- Added model, backend, and training support for Laneformer MLX-LM, MLX INT4
  recall certification, Core ML INT8 palettized export, AWQ and GPTQ 4-bit
  quantization recipes, bitsandbytes 4-bit loading, FlashAttention/SDPA/eager
  attention selection, PyTorch MPS tuning, ONNX/WebGPU and Transformers.js
  exports, tokenizer caching, Mode-A distillation, DAPT corpus assembly, and
  ONNX/quantized artifact publishing metadata (#644, #620, #619, #627, #759,
  #760, #761, #719, #736, #790, #751, #622, #612).
- Added interop adapters for PHILTER, pyDeid, GLiNER-BioMed, LangChain, and the
  optional spaCy `openmed_deid` pipeline component (#372, #624).
- Added policy profiles and policy tooling for Australia Privacy Act, GDPR
  Article 9 health, UK ICO anonymisation, policy config diffing, and Swift
  OpenMedKit policy-driven de-identification (#769, #770, #768, #771, #685).
- Added Swift/OpenMedKit de-identification result JSON export and bundled
  policy resources for client-side policy workflows (#692, #685).
- Added examples and documentation for a first-five-minutes redaction/extraction
  to FHIR walkthrough, OpenAPI export, model manifest docs, REST clients, OCR,
  multimodal redaction, quantization exports, policy workflows, security, SBOM,
  reproducible dependencies, breach response, onboarding, community health, and
  release status contracts (#628, #694, #647, #716, #720, #1021, #409, #697).

### Changed

- `analyze_text(..., output_format="dict")` now returns a frozen
  `AnalyzeResult`; `to_dict()` and mapping access preserve the legacy dict shape
  (#611).
- PII extraction and the staged pipeline now apply clinical term protection by
  default, suppressing ambiguous PERSON/LOCATION/ORG matches that exactly match
  protected clinical vocabulary (#698).
- ConText temporality, uncertainty, and negation now use sentence/clause-bounded
  cue scope with section-aware priors and context offsets (#738, #739, #782).
- Pipeline span output can include populated `section` metadata after section
  stamping (#785).
- Lab reference-range parsing now accepts broader separators/operators and treats
  unknown explicit flags as `unknown` rather than deriving a normal/high/low
  result (#560).
- REST `/health` remains as a compatibility alias, while `/livez` and `/readyz`
  expose split liveness/readiness state and shutdown drains in-flight
  model-backed requests (#722).
- REST CORS and trusted-host handling is now deny-by-default except for exact
  configured origins and trusted hosts (#686).
- OCR auto-selection can now pick installed EasyOCR or docTR adapters in
  addition to Tesseract/PaddleOCR (#749, #558).
- Evaluation defaults now include DrugProt and biomedical NER suites, and
  leakage heatmaps now emit label-by-language matrices with totals and worst
  cells (#617, #743, #680).
- Model manifest rows now merge format lists for existing repositories and
  recognize ONNX/WebGPU and Transformers.js export formats (#736, #790).
- CI lint/test/security/build setup moved to `uv sync` / `uv run`, with GitHub
  Actions refs validated and Dependabot Actions updates limited to minor/patch
  bumps (#185, #700).
- PyTorch/HF backends can auto-select MPS on Apple Silicon when no device is set
  (#719).
- AWQ and GPTQ export paths now share synthetic quantization calibration
  metadata (#759).
- `shift_dates` documentation now describes patient-keyed stable date shifting;
  the legacy boolean remains accepted but deprecated in favor of
  `method="shift_dates"` (#704).

### Fixed

- Fixed nondeterministic audit span ordering so report serialization, hashes, and
  signatures are stable while preserving legacy verification (#645).
- Fixed date-shift parity between `python-dateutil` and fallback paths,
  including month-first English month-name dates, and aligned `uv.lock` with the
  dev extra dependency set (#616, #649).
- Fixed deterministic FHIR URN preservation during Bundle assembly (#553).
- Fixed JSON loading paths in core, eval, NER, and risk modules so corrupt JSON
  raises clearer errors or fails closed (#958).
- Fixed optional-extra diagnostics for missing `ftfy`, section detection, and
  date-shift capabilities (#781).
- Reduced numeric false positives in safety-sweep postcode-style matches by
  requiring stronger context (#783).
- Added explicit UTF-8 encodings for subprocess/file I/O paths and preserved
  exception chaining in model load failures (#1088).
- Added timeouts to `subprocess.run` calls in reproducibility hash and
  release-gate issue helpers (#1090).
- Replaced eager f-string logging with lazy logging interpolation across model,
  processing, batch, text, and utility modules (#1092).
- Fixed PII method quickstart docs for `mask`, `remove`, `replace`, `hash`,
  `shift_dates`, and `reidentify()` examples (#409).

### Security

- Added root `SECURITY.md`, private vulnerability disclosure guidance, security
  issue-template routing, security docs, and README links (#648).
- Added breach-notification runbook and breach report template with explicit
  no-raw-PHI/PII handling guidance (#1021).
- Added CycloneDX SBOM generation via `make sbom`, CI artifact upload, tagged
  release SBOM attachment, and supply-chain docs (#720).
- Added reproducible-lock GitHub Actions gate and contributing docs for pinned,
  hash-verified installs (#1083).
- Added lockfile drift, GitHub Actions ref, license-policy, and doctest-backed
  public-example gates (#693, #700, #763).
- Added PHI-safe defaults for progress callbacks, NDJSON error summaries,
  active-learning records, hashed examples, explain traces, dataset cards, and
  metadata scrubbing (#621, #737, #613, #765, #727, #701, #755).

### Dependencies

- Added optional extras and dependency policy entries for multimodal/OCR, spaCy,
  AWQ, GPTQ, MLX-LM, Kafka, PHILTER/pyDeid, TypeScript client support, and
  service clients (#555, #567, #624, #627, #644, #757, #759, #372, #756, #789).
- Updated GitHub Actions refs and maintenance dependencies, including checkout
  v7, setup-python v6, cache v6, upload-artifact v7, Ruff/pre-commit updates,
  and LangChain Core 1.x compatibility for the optional LangChain extra (#607,
  #710, #711, #712, #713, #714, #715).

### Removed

- No public files, modules, or APIs were removed in the reviewed release range.

### Upgrade Notes

- FHIR `OperationOutcome` output emits R4 `issue.expression`; legacy
  `issue.location` is accepted on input but is not emitted, and non-R4
  severities such as `info` are rejected (#566).
- `ServiceRuntime.get_loader()` returns the warm-pool proxy; use
  `get_model_loader()` when raw loader access is required (#632).
- Unsupported Core ML architectures now fail before model loading/tracing, and
  `--quantized-output` requires `--quantize int8` (#619).
- Custom OCR engines should tolerate the keyword-only `languages` parameter
  (#717).
- The canonical label set expanded with clinical concepts, which can affect
  callers enumerating exact label counts (#718).
- `format_preserve` expands the action enum/schema surface and updates schema
  fingerprints (#778).
- REST deployments using custom Host headers must configure
  `OPENMED_SERVICE_TRUSTED_HOSTS`; wildcard CORS/trusted-host settings are
  rejected (#686).
- OCR auto-selection order changed when optional EasyOCR or docTR engines are
  installed (#749, #558).

## [1.6.0] - 2026-06-22

### Added

- Added a policy-aware de-identification runtime with canonical `OpenMedSpan` schema contracts, a ten-stage `Pipeline`, detector arbitration/cascade routing, calibrated per-label/language/policy thresholds, deterministic safety sweep backstops, and six bundled policy profiles (`hipaa_safe_harbor`, `hipaa_expert_review_assist`, `gdpr_pseudonymization`, `research_limited_dataset`, `strict_no_leak`, `clinical_minimal_redaction`).
- Added signed, reproducible de-identification audit reports with span provenance, residual-risk metadata, reproducibility hashes, and optional HMAC signatures.
- Added re-identification risk reporting and adversarial re-identification benchmark support, including `openmed benchmark pii --attack reid`.
- Added a leakage-first evaluation harness with `BenchmarkReport`, synthetic golden de-identification fixtures, public/reference dataset adapters, DUA-gated corpus stubs, SHIELD comparison-suite support, weak labeling utilities, cold-start latency, and deterministic bootstrap confidence intervals.
- Added release-gate infrastructure for v1.6.0 model readiness: last-green baselines, calibration artifacts, G1a-G8 signed gate reports, quantization recall-delta checks, generated status/leaderboard pages, and a fail-closed release-gates workflow.
- Added clinical and interoperability utilities: ConText temporality and uncertainty axes, OHDSI Athena/Usagi ingestion, a Presidio adapter, and a deterministic FHIR R4 transaction/batch Bundle assembler.
- Added a cardiology zero-shot label-map domain (`CardiacFinding`, `ECGFinding`, `EjectionFraction`, `CardiacProcedure`, `CardiacDevice`, `Anatomy`) plus cardiology keyword routing metadata for future model registration. Public model suggestions continue to fall back to existing general medical models until a cardiology model is registered.
- Added a canonical `models.jsonl` manifest, manifest refresh workflow, manifest-driven Hugging Face model card generation, and HF publishing support for converted MLX/CoreML artifacts.
- Added a packaged `openmed` CLI surface with benchmark and calibration commands, plus a de-identification cookbook notebook and an offline clinical NER families example.
- Added governance, compliance, security, device-tier, FAQ, API reference, release-channel, status, leaderboard, and notebook documentation.

### Changed

- `deidentify()` now routes through the staged policy pipeline and accepts policy, calibration, threshold, and audit controls. When `audit=True`, it returns an audit report rather than the regular `DeidentificationResult`.
- `deidentify(..., keep_mapping=True)` now emits unique placeholders for repeated entities of the same type, such as `[NAME]` and `[NAME_2]`, so re-identification round trips can distinguish them.
- Label metadata now carries policy labels, HIPAA Safe Harbor mappings, risk levels, and ID-number subtype hints while keeping canonical labels stable.
- Benchmark steady-state latency now excludes cold start while preserving `latency.cold_start_ms` in reports.
- PyPI publishing now uses a single guarded tag/manual `publish.yml` workflow; the duplicate release workflow was removed.
- Release metadata now derives changelog sections and expected SemVer bumps from Conventional Commits.
- Python linting/formatting moved to Ruff and pre-commit, Swift formatting moved to checked-in `swift-format` scripts, and CI now enforces the updated repo policy, lint, tests, security, secret-scan, Swift-format, and release-gate jobs.
- Packaging now includes the model manifest, release-gate baseline, policy/schema JSON, `LICENSE`, and `NOTICE`.

### Fixed

- Fixed `method="shift_dates"` to recognize canonical date labels before redaction, so lowercase `date` output from the default English PII model and `date_of_birth` labels are shifted instead of masked; `keep_mapping` no longer treats shifted dates as mask placeholders.
- FHIR Bundle assembly now rejects duplicate `ResourceType/id` values instead of silently overwriting the earlier resource in the internal reference map. Duplicate resources raise a `ValueError` that names the colliding key, preventing downstream references from being rewritten to the wrong Bundle entry.
- REST/MCP request schemas now accept `ar`, `ja`, and `tr` for the `lang` field. These languages have published PII models and are listed in `SUPPORTED_LANGUAGES`, but the `lang` `Literal` in `openmed/service/schemas.py` was never updated, so the service rejected them with a 422 even though the Python API and the models worked. The four `lang` annotations now share a single `PIILanguage` alias kept in sync with `SUPPORTED_LANGUAGES` (guarded by a regression test).
- Fixed case-insensitive `trust_remote_code` allowlist matching for first-party and environment-configured privacy-filter repositories.
- Fixed Feb 29 date shifting when `keep_year=True` targets a non-leap year.
- Fixed REST oversized-text handling with `OPENMED_SERVICE_MAX_TEXT_LENGTH` (default `1_000_000` characters).
- Fixed `BatchProcessor.iter_process` so `batch_size` is honored while preserving output order.
- Fixed duplicate benchmark fixture IDs, duplicate benchmark CLI registration, release-gate behavior when no candidate report is present, and repo-policy ignored-file handling.
- Fixed user-controlled HTML formatter escaping and validation false positives for legitimate long non-ASCII/CJK clinical text.
- Fixed reversible `remove` mappings and repeated entity-type re-identification round trips when `keep_mapping=True`.

### Security

- Added a protected `hf-publish` environment and `HF_WRITE_TOKEN` policy for model publishing.
- Added dependency license policy, `pip-audit` security gate with time-boxed ignores, and gitleaks CI/pre-commit secret scanning with a canary fixture.
- Hardened de-identification audit report signing so `AuditReport.sign()` and `AuditReport.verify()` require a non-empty HMAC key. `None`, empty strings, and empty byte strings now raise `ValueError` instead of producing or accepting weak signatures.

### Tests

- Added FHIR Bundle regression coverage for empty resource lists across transaction, collection, and batch Bundles, and for dangling references that should remain unchanged when the referenced resource is absent from the Bundle.

### Notes

- `shift_dates` remains available as a compatibility alias; prefer `method="shift_dates"` in new code.
- REST clients sending more than `OPENMED_SERVICE_MAX_TEXT_LENGTH` characters now receive a 422 response unless the limit is raised.
- Full SHIELD/DUA datasets require approved or user-supplied access paths; restricted corpus rows are not vendored.
- Release-gate candidates for v1.6.0 need release metadata, calibration evidence for masking/replacement profiles, span fixtures for G8, and quantization evidence for quantized formats.

## [1.5.5] - 2026-06-08

### Added

- Added batch PII extraction and de-identification support through `BatchProcessor(operation="extract_pii")` and `BatchProcessor(operation="deidentify")`, including document-level `batch_size` chunking, shared loader/pipeline reuse, tests, docs, and a runnable example.
- Added REST service model lifecycle controls with `GET /models/loaded`, `POST /models/unload`, request-level `keep_alive`, `OPENMED_SERVICE_KEEP_ALIVE`, and model-loader cache release helpers.
- Added chunked Swift/OpenMedKit PII extraction for long OCR text and refreshed the OpenMed Scan Demo clinical document flow with updated sample text, a printable sample PDF, and a generator script.
- Added a project mascot, brand assets in `docs/brand/`, and an animated on-device PII de-identification demo (`docs/brand/openmed-pii-demo.gif`).
- Added README translations in 13 languages with a language switcher: zh-CN, es, fr, de, it, pt, nl, ar, hi, te, ja, tr, fa.

### Changed

- Batched privacy-filter inference now accepts list inputs across Torch and MLX paths and forwards batching controls to the underlying pipelines.
- The OpenMed Scan Demo now unloads inactive MLX runtime families when switching engines, sequences selected and secondary PII engine runs explicitly, improves OCR line ordering, and expands entity category mapping.
- README and service/model-loader documentation now cover batch PII operations and model unloading behavior.
- Overhauled the README with a visual hero, brand badges, Apple Silicon/Swift/iOS entry points, an OpenMed-vs-cloud comparison table, and a Mermaid flow diagram.

### Fixed

- Improved Swift structured PII recovery for clinical discharge summaries, including surname-first names, member and insurance IDs, account/encounter/document IDs, NPI values, PCP/signed-provider sections, and overlap deduplication.

## [1.5.2] - 2026-05-27

### Security

- Hardened the privacy-filter dispatcher to refuse `trust_remote_code=True` for model identifiers outside an explicit allowlist of first-party OpenAI/OpenMed privacy-filter family models (`openai/privacy-filter`, `OpenMed/privacy-filter-multilingual`, `OpenMed/privacy-filter-nemotron`). Previously, any HuggingFace repository whose name contained the substring `privacy-filter` would be loaded with custom-code execution enabled, allowing remote code execution by anyone able to control the `model_name` parameter on `/pii/extract` or `/pii/deidentify`. Operators with custom fine-tunes of the privacy-filter family can extend the allowlist via the `OPENMED_TRUSTED_REMOTE_CODE_MODELS` environment variable (comma-separated repo IDs).
- Changed `PrivacyFilterTorchPipeline`'s `trust_remote_code` default from `True` to `False`. The first-party dispatcher (`openmed.core.backends.create_privacy_filter_pipeline`) opts in explicitly only for allowlisted models.

### Changed

- README, docs, and website version surfaces now point at `1.5.2`.

### Fixed

- Fixed raw HuggingFace-to-MLX conversion for the OpenAI Privacy Filter family (`openai/privacy-filter`, `OpenMed/privacy-filter-nemotron`, and `OpenMed/privacy-filter-multilingual`) by casting BF16 tensors to float32 before NumPy conversion, remapping OPF/Nemotron checkpoints into the OpenMed MLX runtime layout, fusing Q/K/V projections, preserving classifier bias, and validating converted weight keys/shapes before artifact save.

### Tests

- Added `tests/unit/test_privacy_filter_security.py` covering the identifier matcher, allowlist gate, env-var override, local-artifact trust, and dispatcher opt-in.
- Added HTTP-level regression tests in `tests/unit/service/test_api.py` that POST the attacker-controlled `model_name` payload to `/pii/extract` and `/pii/deidentify` and verify the privacy-filter dispatcher is never reached.
- Added MLX converter regressions for BF16 NumPy conversion, OPF weight remapping, QKV fusion order, and partial-QKV rejection.

## [1.5.1] - 2026-05-21

### Changed

- README, docs, website, and Apple demo version surfaces now point at `1.5.1`.
- Prepared the patch release metadata for the tag-driven build and publish workflow.

## [1.5.0] - 2026-05-18

### Added

- Arabic (`ar`), Japanese (`ja`), and Turkish (`tr`) PII extraction support in the Python SDK, including language defaults, localized regex patterns, fake replacement data, and anonymizer locale routing.
- Registry entries for all API-visible Arabic, Japanese, and Turkish PII source checkpoints: 2 Arabic, 3 Japanese, and 32 Turkish models.
- Preconverted MLX routing for the 28 supported Arabic, Japanese, and Turkish PII `-mlx` repositories so `OpenMedConfig(backend="mlx")` can resolve uploaded artifacts directly.
- Turkish TCKN checksum validation plus context-aware Arabic and Japanese national ID patterns.

### Changed

- README, docs, website, and Apple demo version surfaces now point at `1.5.0`.
- Faker anonymization now falls back to `en_US` with a warning if a requested locale is unavailable at runtime.

### Fixed

- Turkish street-address matching now accepts both descriptor-first forms such as `Cadde İnönü 12` and common Turkish name-first forms such as `Atatürk Caddesi 12`.

### Tests

- Added language constant/default routing, model registry count, MLX mapping, anonymizer locale, and multilingual PII regression coverage for Arabic, Japanese, and Turkish.

## [1.4.1] - 2026-05-17

### Changed

- README, docs, website, and Apple demo version surfaces now point at `1.4.1`.

### Fixed

- `ModelLoader` now resolves existing filesystem paths before prepending the default Hugging Face org, so local model directories load correctly.
- Local model paths now set `local_files_only=True` across config, tokenizer, model, pipeline, and max-length probing to keep offline and air-gapped inference fully local.
- `analyze_text()` now accepts `model_id` as an alias for `model_name`, including local directory paths.

### Tests

- Added unit coverage for local path resolution, local-only loading, and `model_id` alias handling.

## [1.4.0] - 2026-05-04

### Added

- **OpenMed Multilingual Privacy Filter family**, registered across PyTorch and MLX:
  - `OpenMed/privacy-filter-multilingual` — PyTorch / Transformers (CPU + CUDA).
  - `OpenMed/privacy-filter-multilingual-mlx` — MLX full-precision (Apple Silicon).
  - `OpenMed/privacy-filter-multilingual-mlx-8bit` — MLX 8-bit quantized (Apple Silicon and OpenMedKit demos).
  These artifacts use the OpenAI Privacy Filter architecture and officially support 16 languages through the OpenMed multilingual PII corpus.
- **Python MLX routing for multilingual Privacy Filter artifacts**:
  - `_MLX_MODEL_MAP` entries for the full and 8-bit multilingual MLX repo IDs.
  - `privacy-filter-multilingual` and `multilingual-privacy-filter` MLX family aliases, both resolving to the existing OpenAI Privacy Filter model class and BIOES decoder.
  - Family-aware Torch fallback so multilingual MLX model names substitute `OpenMed/privacy-filter-multilingual` on non-MLX hosts instead of the OpenAI baseline.
- **Multilingual Privacy Filter Studio** in `examples/privacy_filter_multilingual_studio/`, a web demo comparing the OpenAI baseline, OpenAI Nemotron Privacy Filter, and OpenMed Multilingual Privacy Filter with English, French, and Arabic examples.
- **OpenMed Scan Demo multilingual mode** with `OpenMed/privacy-filter-multilingual-mlx-8bit`, a three-engine picker, EN/FR/AR sample buttons, and new French/Arabic scanned demo documents for screenshot-ready flows.
### Changed

- Privacy Filter docs and README now describe three Privacy Filter families and label the multilingual model as **OpenMed Multilingual Privacy Filter**.
- OpenMedKit and demo version surfaces now point at `1.4.0`.
- The scan demo clears previous annotation windows whenever the language/sample changes, avoiding stale entities from earlier model runs.
- The multilingual web studio scan animation now performs a single top-to-bottom pass while redacting line by line, matching the stronger visual rhythm of the original Privacy Filter Studio.

### Fixed

- Improved Swift model-download handling so stale cached 401/404 responses cannot masquerade as `openmed-mlx.json` manifests after a public model becomes available.
- Tightened stale-result invalidation in iOS and web demo flows so slower previous model runs cannot overwrite a newly selected language/sample.

### Tests

- Added Python unit coverage for multilingual MLX backend selection, family-aware Torch fallback, and MLX Privacy Filter family dispatch aliases.
- Rebuilt the OpenMed Scan Demo after the multilingual 8-bit integration.

## [1.3.0] - 2026-04-27

### Added

- **Faker-backed PII anonymization engine** (`openmed.core.anonymizer`):
  - `Anonymizer` class with cached per-locale Faker instances, deterministic seeding (`hashlib.blake2b`), and label-keyed generator dispatch.
  - `AnonymizerConfig` dataclass for advanced configuration.
  - Locale resolution map (`LANG_TO_LOCALE`) covering all nine OpenMed languages; Telugu falls back to `en_IN` with a one-time `UserWarning`.
  - Format-preserving helpers for phone numbers (digit-group lengths preserved), dates (separator/ordering preserved), emails (domain preserved), and generic IDs.
  - Custom Faker providers for clinical/national IDs where Faker's built-ins are missing or incorrect: `AadhaarProvider` (Verhoeff checksum), `GermanSteuerIdProvider`, `MedicalRecordNumberProvider`, `NPIProvider`. Faker's built-ins are reused for `pt_BR.cpf`/`cnpj`, `nl_NL.ssn` (BSN), `fr_FR.ssn` (NIR), `it_IT.ssn` (Codice Fiscale), and `es_ES.nie` after empirical verification against OpenMed's existing checksum validators.
  - `register_clinical_provider()` and `register_label_generator()` for extending coverage.
- **Canonical PII label taxonomy** (`openmed.core.labels`):
  - `CANONICAL_LABELS` set with 47 canonical labels in `UPPER_SNAKE_CASE`.
  - `normalize_label()` maps English lowercase, the 52 Portuguese UPPERCASE labels, BIOES-tagged variants (`B-NAME`, `I-DATE`), and arbitrary mixed-case forms to a single canonical form.
- **Unified privacy-filter dispatch** (`openmed.core.backends`):
  - `select_privacy_filter_backend()`, `resolve_privacy_filter_model()`, and `create_privacy_filter_pipeline()` route privacy-filter requests to MLX on Apple Silicon and PyTorch elsewhere with a one-time `UserWarning` when an MLX-only artifact name (`OpenMed/privacy-filter-mlx*`) is substituted with `openai/privacy-filter` on non-Mac hosts.
  - `extract_pii()` and `deidentify()` now route privacy-filter models through this dispatcher, skipping regex smart-merging since the model already does Viterbi-constrained BIOES decoding.
- **PyTorch privacy-filter wrapper** (`openmed.torch.PrivacyFilterTorchPipeline`):
  - Loads `openai/privacy-filter` (or any compatible HuggingFace fine-tune) via `transformers.AutoModelForTokenClassification` with auto device selection (CUDA → CPU).
  - Output entity-dict shape matches the MLX pipeline so the rest of OpenMed is backend-agnostic.
- **Shared decoding utilities** (`openmed.core.decoding`):
  - `TokenLabelInfo`, `build_label_info`, `viterbi_decode`, `labels_to_token_spans`, `zero_viterbi_biases`, `VITERBI_BIAS_KEYS` extracted from the MLX pipeline so the Torch wrapper reuses the same BIOES Viterbi decoder.
  - `trim_span_whitespace`, `refine_privacy_filter_span` for span post-processing across both backends.
- **`deidentify()` keyword arguments**: `consistent: bool`, `seed: Optional[int]`, `locale: Optional[str]` for deterministic, locale-overridable obfuscation. Passing `seed=` alone implies `consistent=True`.
- **Portuguese (`pt`) accepted by REST API schemas** in `openmed/service/schemas.py` (was previously library-only despite full core support).
- **Documentation**: new [Anonymization Guide](docs/anonymization.md) covering the Faker engine, locale table, determinism modes, format preservation, clinical-ID checksum sources, and the privacy-filter family.
- **Examples**:
  - `examples/obfuscation_demo.py` — random vs deterministic surrogates, locale walkthrough, format-preserving phone numbers, pt_BR CPF generation with checksum verification.
  - `examples/privacy_filter_unified.py` — same `extract_pii()` / `deidentify()` call works on Apple Silicon (MLX) and Linux (PyTorch); compares the OpenAI baseline against the Nemotron-PII fine-tune side-by-side.
  - `examples/privacy_filter_studio/` — interactive FastAPI + static web studio for two-pane PII masking/randomization with sample clinical notes, highlighted entities, backend/model status, and an explicit first-run download toggle.
- **Nemotron-PII fine-tune of the OpenAI Privacy Filter**, registered as three new model IDs that route through the existing privacy-filter pipeline (same architecture, different training data):
  - `OpenMed/privacy-filter-nemotron` — PyTorch / Transformers (CPU + CUDA).
  - `OpenMed/privacy-filter-nemotron-mlx` — MLX full-precision (Apple Silicon).
  - `OpenMed/privacy-filter-nemotron-mlx-8bit` — MLX 8-bit quantized (Apple Silicon).
  These checkpoints **are** the OpenAI Privacy Filter architecture (gpt-oss-style sparse-MoE transformer with local attention, sink tokens, RoPE+YaRN, tiktoken `o200k_base`) fine-tuned on the [Nemotron PII dataset](https://huggingface.co/datasets/nvidia/Nemotron-PII-v1). They reuse `OpenAIPrivacyFilterForTokenClassification` and `PrivacyFilterMLXPipeline` unchanged — no new architecture code needed.
- **`_MLX_MODEL_MAP` entries** for the two new Nemotron MLX repo IDs in `openmed.mlx.inference`.
- **Aliases for the new family in `_SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES`** (`privacy-filter-nemotron`, `nemotron-privacy-filter`) — both resolve to the existing `openai-privacy-filter` family so a Nemotron-fine-tune MLX artifact can ship with either family identifier in its manifest and still dispatch correctly.
- **Family-aware Torch fallback** in `openmed.core.backends`:
  - New `_TORCH_FALLBACK_BY_FAMILY` table and `_torch_fallback_for()` helper.
  - An MLX-only Nemotron request on a non-Apple-Silicon host now substitutes `OpenMed/privacy-filter-nemotron` instead of the unrelated default `openai/privacy-filter`, so the user gets the training distribution they asked for. A one-time `UserWarning` names the substitute.
  - Adding a future fine-tune that should fall back to its own PyTorch repo is a one-line addition to `_TORCH_FALLBACK_BY_FAMILY`.
- **Nemotron MLX classifier-head bias support**: `OpenAIPrivacyFilterForTokenClassification` now honors `classifier_bias` / `unembedding_bias` in artifact configs, while keeping the original OpenAI checkpoint bias-less by default.
- **Swift OpenMedKit privacy-filter classifier-head bias support**: the native MLX artifact loader now decodes `classifier_bias` / `unembedding_bias` and builds the Privacy Filter head with a learned bias when Nemotron-PII artifacts require it.

### Changed

- **`method="replace"` upgraded in place** to use the new Faker-backed `Anonymizer`. Surrogates are now locale-aware (e.g. German names for `lang="de"`, Portuguese phones for `lang="pt"`), format-preserving, and optionally deterministic. The previous tiny static `LANGUAGE_FAKE_DATA` lists are kept as a deprecated fallback used only when a Faker locale is unavailable.
- **Privacy filter book demo** (`examples/privacy_filter_book/app.py`) migrated to `PrivacyFilterTorchPipeline` for the CPU side, replacing the inline `AutoTokenizer`/`AutoModelForTokenClassification`/`pipeline` triple.
- **MLX inference module** trimmed: BIOES Viterbi (≈280 lines) and span-refinement helpers moved to `openmed.core.decoding`. Behavior unchanged.
- **Privacy Filter Studio** keeps model loading cache-only unless downloads are explicitly allowed, then restores the caller's Hugging Face offline environment after loading.
- **OpenMed Scan Demo privacy-filter option** now points at `OpenMed/privacy-filter-nemotron-mlx-8bit` and labels the engine as OpenAI Nemotron Privacy Filter throughout the picker, download events, and README.

### Breaking Changes

- **`faker>=22.0` is now a required core dependency**. Slim installs that skip the ML extras will still pull Faker (~3 MB).
- **`method="replace"` outputs no longer come from the prior hardcoded list** (`["Jane Smith", "John Doe", "Alex Johnson", "Sam Taylor"]`, etc.). Any test or downstream code asserting on those exact strings must either pass `consistent=True, seed=<value>` and update expected output, or assert non-equality with the original. All other methods (`mask`, `remove`, `hash`, `shift_dates`) are unchanged.
- **Privacy-filter routing through `extract_pii()`** skips regex smart-merging by design. Users who previously chained the low-level MLX pipeline with `merge_entities_with_semantic_units()` manually may see different entity counts; the new path produces cleaner spans because the model's Viterbi decoder already enforces BIOES validity.

### Tests

- New tests across `tests/unit/core/test_labels.py` (102), `tests/unit/core/test_anonymizer.py` (171, includes per-locale checksum validation across 100s of generated IDs), `tests/unit/test_privacy_filter_routing.py` (22 — backend selection, family-aware Torch fallback, dispatch, integration), Nemotron parametrisation of the existing privacy-filter MLX dispatch test (`tests/unit/mlx/test_privacy_filter_mlx.py::test_dispatches_privacy_filter_pipeline`), and Portuguese obfuscation regressions in `tests/unit/test_pii_multilingual_regression.py` (3).
- Swift OpenMedKit coverage for `classifier_bias` / `unembedding_bias` config decoding, Nemotron-biased Privacy Filter forward shape, and the baseline bias-less head.
- Focused privacy/anonymization suite: 458 passed, 6 skipped, 11 pre-existing span-validation warnings.

## [1.2.0] - 2026-04-24

### Added

- **Expanded Python MLX runtime support** for OpenMed MLX artifacts beyond classic token classification, including GLiNER span NER, GLiClass zero-shot classification, GLiNER-Relex relation extraction, and OpenAI Privacy Filter artifacts.
- **Native OpenAI Privacy Filter MLX pipeline** with tiktoken-compatible tokenization, byte-offset reconstruction, BIOES/Viterbi decoding, model-led span repair, and support for the public `OpenMed/privacy-filter-mlx` and `OpenMed/privacy-filter-mlx-8bit` artifacts.
- **Native Swift OpenMedKit GLiNER-family APIs**:
  - `OpenMedZeroShotNER`
  - `OpenMedZeroShotClassifier`
  - `OpenMedRelationExtractor`
- **Native Swift MLX DeBERTa-v2/v3 and Privacy Filter runtimes** for local inference on Apple Silicon macOS and physical iPhone/iPad devices.
- **Self-contained OpenMed MLX artifact handling** for `task`/`family` manifests, tokenizer assets, `weights.safetensors`, and `weights.npz` fallback paths.
- **OpenMed Scan Demo**: a guided iPhone workflow for document capture/sample loading, OCR review, PII de-identification, clinical extraction, summary review, model preparation, and PII engine comparison.
- **OpenMedDemo Privacy Filter option** so macOS/iOS users can test the public OpenAI Privacy Filter MLX artifact alongside OpenMed PII models.
- **App privacy readiness assets** for the scan demo, including a privacy manifest and camera usage copy for local document scanning.

### Changed

- Improved Apple model download/caching behavior so MLX artifacts are prepared once and reused offline from cache.
- Removed Hugging Face token UI and token persistence from demo flows now that release artifacts are public.
- Updated PII post-processing so Privacy Filter regex logic repairs model-predicted spans without inventing unsupported semantic labels.
- Refreshed OpenMedKit documentation and examples for native MLX artifacts, Swift package usage, and on-device Apple workflows.

### Fixed

- Reduced iOS memory pressure in the Privacy Filter MLX loader by tightening the Swift model loading path.
- Fixed local MLX artifact loading and model-store readiness checks for public Hub artifacts.
- Tightened PII entity merging and privacy-filtering tests around model/pattern span interactions.

### Tests

- Added Python unit coverage for MLX custom-task dispatch, Privacy Filter inference/decoding, artifact loading, and PII privacy-filter post-processing.
- Added Swift unit coverage for MLX artifact validation, DeBERTa/GLiNER-family runtime setup, Privacy Filter decoding, sample OCR assets, and post-processing behavior.

## [1.1.0] - 2026-04-20

### Added

- **Portuguese PII and de-identification support** via `lang="pt"`
  - Registered 31 API-visible Portuguese PII checkpoints from the OpenMed Hugging Face collection
  - Default Portuguese model: `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1`
  - Added Portuguese regex/semantic patterns for dates, phones, CPF, CNPJ, street addresses, and postcodes
  - Added CPF and CNPJ checksum validators, Portuguese fake replacement data, and localized date shifting
- **Portuguese docs and examples**
  - Updated multilingual PII documentation from 8 to 9 languages and from 179 to 210 PII models
  - Added a Portuguese model-card/README one-liner and smoke-example coverage

### Changed

- Expanded PII label normalization and replacement mapping for CPF/CNPJ and Portuguese model labels.

## [1.0.0] - 2026-04-03

### Added

- **Apple MLX inference backend** for hardware-accelerated NER on Apple Silicon
  - `openmed.mlx.models.bert_tc`: Pure MLX BERT implementation with token-classification head
  - `openmed.mlx.inference`: MLX NER pipeline producing HuggingFace-compatible output format
  - `openmed.mlx.convert`: CLI tool to convert HuggingFace token-classification models to MLX format with optional 4/8-bit quantization
  - Supports BIO tag decoding with `simple`, `first`, `average`, and `max` aggregation strategies
  - Auto-detection: prefers MLX on Apple Silicon when available, falls back to HuggingFace/PyTorch
- **CoreML export** for iOS and macOS deployment
  - `openmed.coreml.convert`: CLI tool to convert HuggingFace models to CoreML `.mlpackage` format
  - Supports flexible sequence lengths via `ct.RangeDim`, float16/float32 precision
  - Embeds `id2label` mapping in model metadata for self-contained deployment
- **Swift package: OpenMedKit** (`swift/OpenMedKit/`)
  - SPM package for iOS 16+ / macOS 13+ with CoreML-based NER inference
  - `NERPipeline`: CoreML inference with softmax → BIO decoding → entity extraction
  - `PostProcessing`: BIO tag grouping with first/average/max aggregation strategies
  - `EntityPrediction`: Swift equivalent of Python's EntityPrediction dataclass
  - Uses `swift-transformers` for HuggingFace-compatible tokenization
  - Includes unit tests for BIO decoding and aggregation strategies
- **Backend abstraction layer** (`openmed.core.backends`)
  - `InferenceBackend` protocol with `is_available()` and `create_pipeline()` interface
  - `HuggingFaceBackend` and `MLXBackend` implementations
  - `get_backend()` auto-detection with explicit override via `config.backend`
- **New optional dependency groups**: `pip install openmed[mlx]` and `pip install openmed[coreml]`
- **Pilot model**: `OpenMed-PII-SuperClinical-Small-44M-v1` as conversion and testing target
- **37 new tests** for backends, MLX conversion key remapping, MLX pipeline output format, CoreML module structure

### Changed

- Added `backend` field to `OpenMedConfig` (None/auto, "hf", "mlx")

### Documentation

- Updated README, CHANGELOG, and website for the `v1.0.0` release

## [0.6.4] - 2026-03-24

### Added

- **Aadhaar national ID support** for Hindi and Telugu PII detection
  - Added Verhoeff checksum validator (`validate_aadhaar`) for 12-digit Aadhaar numbers
  - Added Aadhaar patterns with context-aware scoring to Hindi and Telugu pattern libraries
- **PII accuracy test suite** (`tests/unit/test_pii_accuracy.py`)
  - Validation-failure confidence penalty tests
  - Pattern tightening regression tests (postal codes, phone numbers, Steuer-ID)
  - Confidence calibration verification
  - New normalize_label coverage tests

### Changed

- **`_fix_entity_spans` now Unicode-aware** — replaced `.isalnum()` with `unicodedata.category` check covering letters, combining marks, and numbers; capped forward extension at 10 characters; removed redundant `.strip()` that caused text-mismatch false positives
- **Quality gate text-mismatch relaxed** — whitespace-only differences (common after span normalization) are now downgraded to INFO level instead of WARNING
- **Failed pattern validation now penalized in merged confidence** — unvalidated patterns contribute only 10% weight (down from 40%) in the model/pattern confidence blend
- **`normalize_label` expanded** with `bsn`, `dni`, `nie`, `aadhaar` → `national_id`; `mrn` → `medical_record`; `account_number` → `account`; `credit_debit_card` → `payment_card`

### Fixed

- **French postal code pattern** tightened from bare `\d{5}` to range-constrained `01-95 + DOM-TOM 971-976` prefixes — reduces false positives from medical codes
- **German Steuer-ID pattern** tightened to reject leading-zero numbers (`[1-9]\d{10}`); base_score raised to 0.35
- **German postal code pattern** tightened to exclude `00xxx` range
- **German phone pattern** now requires at least 4 digits after area code, reducing short-number false positives
- **French NIR base_score** raised from 0.4 to 0.55 to reflect high structural specificity with validator

### Documentation

- Updated README, CHANGELOG, and website for the `v0.6.4` release

## [0.6.3] - 2026-03-19

### Added

- **Span-boundary quality gates** (`openmed.core.quality_gates`)
  - `validate_entity_spans()` checks start < end, in-bounds, text-match, and zero-length invariants for every entity after tokenizer repair and smart merging
  - `detect_overlapping_entities()` returns pairs of overlapping character spans for informational use
  - `SpanValidationWarning` emitted on violations — warn-only, never silently drops entities
  - Integrated into `OutputFormatter.format_predictions()` (after `_fix_entity_spans`) and `extract_pii()` (after smart merging)
- **Multilingual PII regression test suite** (`tests/unit/test_pii_multilingual_regression.py`)
  - Golden-input regression tests for all 8 supported languages (en, fr, de, it, es, nl, hi, te)
  - Validates entity type detection, span text matching, confidence thresholds, and smart merging boundaries
  - 31 deterministic test cases using mocked model output
- **Span-boundary guard tests** (`tests/unit/test_quality_gates.py`)
  - 19 tests covering valid entities, inverted/zero-length spans, out-of-bounds, text mismatch, overlap detection, and integration with `_fix_entity_spans`
- **Label-map consistency tests** (`tests/unit/ner/test_label_map_consistency.py`)
  - Validates `defaults.json` domain invariants (at least 1 label per domain, no case-insensitive duplicates, `generic` domain exists)
  - `normalize_label()` idempotency checks across all known label variants
  - Specificity hierarchy validation against `is_more_specific()`
  - All PII `entity_types` in `OPENMED_MODELS` recognized and idempotent under `normalize_label()`
  - At least one PII model per supported language in the registry

### Changed

- Updated website model count from 640+ to 750+

### Documentation

- Updated README, website copy, and CHANGELOG for the `v0.6.3` release

## [0.6.2] - 2026-03-10

### Added

- **Dutch, Hindi, and Telugu PII support**
  - `extract_pii()` and `deidentify()` now accept `lang="nl"`, `lang="hi"`, and `lang="te"`
  - Added sparse public registry entries for:
    - `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1`
    - `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1`
    - `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1`
  - Added locale-aware patterns for Dutch BSN, Dutch postcodes, India PIN codes, localized month names, and day-first date shifting
  - Added Dutch BSN checksum validation and locale-specific fake replacement data for `nl`, `hi`, and `te`
  - Added `examples/pii_multilingual_new_languages.py` for registry, regex, and live-model smoke coverage
- **REST service runtime hardening**
  - Added `openmed.service.runtime.ServiceRuntime` for shared per-process config and model-loader reuse
  - Added `OPENMED_SERVICE_PRELOAD_MODELS` to warm selected models at startup
  - Added structured validation/bad-request/timeout/internal-error JSON envelopes for non-2xx responses
  - Added request timeout enforcement around blocking inference work
- **Testing coverage**
  - Added regression tests for Dutch, Hindi, and Telugu routing, patterns, fake data, date handling, and entity merging
  - Added REST service tests for validation errors, timeout behavior, shared-loader reuse, preload parsing, and the new `lang` values

### Changed

- Expanded the multilingual PII catalog from 176 to 179 models across 8 languages
- `get_pii_models_by_language()` now returns sparse public releases for `nl`, `hi`, and `te` while keeping English filtering correct
- `ModelLoader.create_pipeline()` now caches created pipelines for repeated requests with identical parameters
- REST schemas now validate model names, confidence thresholds, extra fields, and the legacy `shift_dates` alias more strictly
- Updated multilingual examples, notebook guidance, website copy, and install snippets to reflect the 8-language / 179-model PII catalog and `uv pip install "openmed[hf]"`

### Fixed

- Smart semantic-merge resolution no longer lets weaker model labels overwrite stronger validated pattern labels
- Localized Dutch, Hindi, and Telugu month-name parsing now falls back correctly during date shifting instead of relying only on `dateutil`
- Dutch phone, BSN, and street-address patterns were tightened after live smoke review to reduce overlap and improve entity labeling

### Documentation

- Updated README, REST service docs, website copy, notebook index, and the multilingual PII notebook for the `v0.6.2` release

## [0.6.1] - 2026-03-01

### Added

- **Dockerized REST MVP** for OpenMed service use-cases
  - New FastAPI service module at `openmed.service`
  - `GET /health` endpoint for service status and active profile reporting
  - `POST /analyze` endpoint mapped to `analyze_text(..., output_format="dict")`
  - `POST /pii/extract` endpoint mapped to `extract_pii(...)`
  - `POST /pii/deidentify` endpoint mapped to `deidentify(...)`
- **Container runtime support**
  - New CPU-focused `Dockerfile` for service deployment
  - Added `.dockerignore` for smaller build contexts
- **Service validation tests**
  - New unit tests covering endpoint success/failure paths, schema validation, and profile selection

### Changed

- Added optional `service` dependency extra in `pyproject.toml` (`fastapi`, `uvicorn[standard]`)
- Expanded `dev` extra with API test dependencies (`fastapi`, `httpx`)

### Documentation

- Added REST service guide: `docs/rest-service.md`
- Added MkDocs navigation entry for REST service docs
- Updated README with REST API and Docker usage examples

## [0.6.0] - 2026-02-23

### Removed

- **CLI and TUI surfaces removed**: OpenMed is now a Python API-first package
  - Removed `openmed` console entrypoint from package metadata
  - Removed `openmed.cli` and `openmed.tui` modules
  - Removed zero-shot CLI modules under `openmed.zero_shot.cli`
  - Removed `cli_main` from the top-level `openmed` public API

### Changed

- Updated package metadata to remove CLI/TUI extras (`cli`, `tui`)
- Updated docs and website content to API-only guidance
- Consolidated PyPI publishing into a single tag-driven workflow (`publish.yml`)
- Updated release tooling to use `openmed/__about__.py` as the version source of truth

## [0.5.8] - 2026-02-19

### Fixed

- **PII replace label mapping coverage**:
  - Added robust normalization map so replacement data is generated for label variants (`first_name`, `last_name`, `dob`, `postal_code`, etc.)
  - Expanded locale fake-data dictionaries with `FIRST_NAME`, `LAST_NAME`, and `ZIPCODE` values across supported languages
- **Span alignment stability**:
  - `extract_pii()` and `deidentify()` now strip leading/trailing whitespace before inference so spans remain aligned with `analyze_text()` validation behavior
- **Spanish accent remapping robustness**:
  - Added regression coverage for off-by-one spans combined with accent restoration

## [0.5.7] - 2026-02-18

### Fixed

- **Entity span repair in output formatter**:
  - Added `_fix_entity_spans()` to correct tokenizer end-offset truncation and trim whitespace around predicted spans
  - Integrated span repair into output formatting before grouping
- **Regression coverage**:
  - Added dedicated tests for off-by-one span fixes, whitespace trimming, and boundary handling
- **Documentation notebook refresh**:
  - Updated multilingual PII notebook examples to reflect span-fix behavior

## [0.5.6] - 2026-02-18

### Added

- **Spanish PII Detection & De-identification**: Full Spanish language support for PII extraction
  - `extract_pii()` and `deidentify()` now accept `lang="es"` for Spanish clinical text
  - Automatic model selection for Spanish — correct language-specific model chosen when `lang="es"`
  - 7 new Spanish-specific regex patterns for dates, phone numbers, addresses, postal codes, and national IDs
  - Spanish date format support with unique "de" connector (e.g., "15 de enero de 2020")

- **Spanish National ID Validators**: DNI and NIE document validation with checksum verification
  - `validate_spanish_dni()` — Spanish DNI 8-digit + check letter (mod-23 lookup table)
  - `validate_spanish_nie()` — Spanish NIE with X/Y/Z prefix conversion and DNI algorithm

- **2 New English Base Model Architectures**: Expanded PII model coverage
  - `pii_biomed_bert_full` — BiomedBERTFull-Base-110M for comprehensive biomedical PII detection
  - `pii_lite_clinical_u` — LiteClinicalU-Small-66M for universal lightweight PII detection
  - Both architectures auto-generate variants for all 5 supported languages

- **Expanded Model Registry**: 35 Spanish PII models + 8 new models across existing languages
  - Total PII models expanded from 133 to 176+ (36 English + 35 x 4 languages)
  - `get_pii_models_by_language("es")` returns all 35 Spanish models
  - `get_default_pii_model("es")` returns the recommended Spanish default model

- **Accent Normalization**: Transparent accent stripping for models trained on accent-free text
  - `normalize_accents` parameter on `extract_pii()` and `deidentify()` (auto-enabled for Spanish)
  - Strips diacritical marks before model inference, maps entity positions back to original accented text
  - `_strip_accents()` helper preserves character count via NFC/NFD normalization
  - Can be explicitly enabled (`normalize_accents=True`) or disabled (`normalize_accents=False`) for any language

- **Spanish Locale Data**: Culturally appropriate synthetic data for the `replace` method
  - Spanish fake names, emails, phone numbers (+34), addresses, and IDs (DNI/NIE)
  - Spanish month names for date parsing and formatting
  - European DD/MM/YYYY date handling for Spanish

- **Testing**: Comprehensive Spanish PII test coverage
  - Spanish DNI validator tests (6 tests) and NIE validator tests (6 tests)
  - Spanish pattern matching tests for dates, phones, DNI, NIE
  - Spanish model registry tests: count, naming, mirror structure
  - Updated existing tests: fixed `"es"` to `"ja"` in unsupported language assertions

### Changed

- `_LANGUAGE_CONFIG` in model registry now includes `"es": {"name": "Spanish", "prefix": "Spanish-"}`
- French, German, and Italian model counts updated from 33 to 35 per language (2 new base architectures)
- `SUPPORTED_LANGUAGES` expanded to include `"es"`
- Date handling functions (`_shift_date`, `_shift_date_basic`, `_format_date_like_original`) now support Spanish

## [0.5.5] - 2026-02-11

### Added

- **Multilingual PII Detection & De-identification**: Language-aware PII extraction for clinical text
  - `extract_pii()` and `deidentify()` now accept a `lang` parameter (ISO 639-1: `en`, `fr`, `de`, `it`)
  - Automatic model selection — correct language-specific model chosen when `lang` is specified
  - Language-specific regex patterns for dates, phone numbers, addresses, postal codes, and national IDs
  - 18 new regex patterns (6 per language) for French, German, and Italian

- **National ID Validators**: Country-specific document validation with checksum verification
  - `validate_french_nir()` — French NIR/INSEE 15-digit social security numbers (mod-97 checksum)
  - `validate_german_steuer_id()` — German 11-digit tax identification numbers (digit-frequency rules)
  - `validate_italian_codice_fiscale()` — Italian 16-character alphanumeric fiscal codes

- **Locale-Aware Date Handling**: Language-appropriate date parsing and formatting
  - European day-first parsing for `fr`/`de`/`it` (DD/MM/YYYY, DD.MM.YYYY)
  - US month-first parsing for `en` (MM/DD/YYYY)
  - Localized month names preserved during date shifting

- **Culturally Appropriate De-identification**: Language-specific synthetic data for the `replace` method
  - Fake names, emails, phone numbers, addresses, and IDs per locale
  - `LANGUAGE_FAKE_DATA` dictionary for English, French, German, and Italian

- **Expanded Model Registry**: Multilingual model generation across all PII architectures
  - ~99 new multilingual PII models (33 architectures x 3 new languages)
  - Total PII models expanded from 33 to 132+
  - `get_pii_models_by_language()` — returns all PII models for a given language
  - `get_default_pii_model()` — returns the recommended default model for a language

- **New Module**: `openmed/core/pii_i18n.py` — Internationalization module
  - `SUPPORTED_LANGUAGES`, `DEFAULT_PII_MODELS`, `LANGUAGE_PII_PATTERNS` constants
  - `get_patterns_for_language()` — returns combined English + language-specific regex patterns
  - `LANGUAGE_MONTH_NAMES` dictionary with month names in all 4 languages

- **Documentation**
  - New [Multilingual PII Detection Guide](examples/notebooks/Multilingual_PII_Detection_Guide.ipynb) notebook
    - Cross-language comparison, batch processing, and custom model selection
    - Examples for French, German, and Italian clinical notes
    - All de-identification methods with multilingual fake data

- **Testing**
  - `test_pii_i18n.py` — unit tests for the i18n module (373 lines)
  - `test_model_registry_multilingual.py` — unit tests for multilingual model generation (202 lines)
  - Updated `test_pii.py` and `test_pii_entity_merger.py` with multilingual test cases

### Changed

- `_redact_entity()` and `_generate_fake_pii()` now propagate `lang` parameter for language-appropriate replacements
- `normalize_label()` handles national ID variants (`nir`, `insee`, `steuer_id`, `codice_fiscale`) and postal code variants (`postcode`, `zipcode`, `postal_code`)
- Label specificity hierarchy expanded with `national_id` sub-types for cross-language entity resolution
- `CATEGORIES["Privacy"]` dynamically includes all PII model keys (English + multilingual)
- Updated `__init__.py` exports with multilingual PII support functions

## [0.5.1] - 2026-01-14

### Added

- **Context-Aware PII Scoring**: Presidio-inspired confidence scoring system
  - `PIIPattern` dataclass extended with `base_score`, `context_words`, `context_boost`, and `validator` fields
  - Context detection via `find_context_words()` - boosts confidence when keywords like "SSN:", "DOB:", "NPI:" appear near detected entities
  - Checksum validation functions: `validate_ssn()`, `validate_luhn()` (credit cards), `validate_npi()`, `validate_phone_us()`
  - Invalid matches (e.g., SSN starting with 000 or 666) get reduced confidence scores
  - Combined model + pattern scoring (60/40 weighted average) for optimal accuracy
  - Low base scores prevent false positives; context words confirm true PHI

- **Website Updates**
  - New "Clinical Text De-Identification" section on landing page
  - Key stats row: 18+ PHI types, 100% local processing, $0 API fees, Apache-2.0
  - Six feature cards: Context-Aware Detection, Checksum Validation, Smart Merging, Zero Data Movement, Flexible Redaction, HIPAA Safe Harbor
  - Syntax-highlighted code example with correct API usage
  - CTA buttons linking to documentation and HuggingFace models

### Changed

- Updated default PII detection model name to `OpenMed-PII-SuperClinical-Small-44M-v1`
- `merge_entities_with_semantic_units()` now supports context-aware pattern scoring

### Fixed

- MkDocs navigation: Added `medical-tokenizer.md` and `pii-smart-merging.md` to nav structure
- Broken link in `cli.md` to PII notebook (now links to GitHub)
- Broken links in `pii-smart-merging.md` to non-existent documentation pages
- Website code example now uses correct API (`entity.text`, `entity.label`, `entity.confidence`)

## [0.5.0] - 2026-01-13

### Added

- **PII Detection & De-identification**: HIPAA-compliant PII extraction and de-identification
  - `extract_pii()` function for detecting PII entities in clinical text
  - `deidentify()` function with 5 de-identification methods:
    - `mask`: Replace with placeholders (`[NAME]`, `[DATE]`, etc.)
    - `remove`: Complete removal of PII entities
    - `replace`: Replace with synthetic data
    - `hash`: Cryptographic hashing for record linking
    - `shift_dates`: Shift dates while preserving temporal relationships
  - `reidentify()` function for reversing de-identification with stored mappings
  - Support for all 18 HIPAA Safe Harbor identifiers
  - Configurable confidence thresholds for precision/recall control
  - Batch processing support for PII extraction and de-identification
  - `PIIEntity` and `DeidentificationResult` dataclasses

- **Smart Entity Merging**: Advanced post-processing to fix tokenization fragmentation
  - Regex-based semantic unit detection with 20+ PII patterns
  - Automatic merging of fragmented entities (e.g., dates split as "01" + "/15/1970" → "01/15/1970")
  - Dominant label selection with confidence-based tie-breaking
  - Label specificity hierarchy (e.g., `date_of_birth` > `date`)
  - Support for dates (6 formats), SSN, phone numbers, emails, URLs, addresses, IP addresses, MAC addresses, ZIPs, credit cards
  - Custom pattern support via `PIIPattern` class
  - Enabled by default with `use_smart_merging=True` parameter
  - Public API exports: `merge_entities_with_semantic_units()`, `find_semantic_units()`, `calculate_dominant_label()`, `PII_PATTERNS`
  - Minimal performance overhead (~5-10%)

- **PII CLI Commands**: Comprehensive command-line interface for PII operations
  - `openmed pii extract`: Extract PII entities from text or files
  - `openmed pii deidentify`: De-identify text or files with method selection
  - `openmed pii batch-extract`: Batch PII extraction from directories
  - `openmed pii batch-deidentify`: Batch de-identification with method selection
  - All commands support confidence thresholds, smart merging, and output formatting
  - Date shifting parameter (`--date-shift-days`) for temporal preservation

- **PII TUI Mode**: Interactive PII detection in the terminal interface
  - Visual PII entity highlighting with color coding
  - Real-time de-identification preview
  - Model selection for PII detection models

- **PII Model Registry**: Added PII detection models
  - `pii_detection_superclinical` (434M parameters)
  - Covers 18+ PII entity types (names, dates, SSN, phone, email, addresses, medical records, etc.)

- **Comprehensive Documentation**
  - [PII Detection & Smart Merging Guide](docs/pii-smart-merging.md) (452 lines)
    - Algorithm explanation and implementation details
    - Complete API reference with examples
    - Supported PII patterns catalog
    - Performance characteristics
    - Troubleshooting guide
  - [Complete PII Jupyter Notebook](examples/notebooks/PII_Detection_Complete_Guide.ipynb) (48 cells)
    - Step-by-step tutorial covering all PII functionality
    - Before/after smart merging comparisons
    - All 5 de-identification methods demonstrated
    - Re-identification workflows
    - Batch processing examples
    - Confidence thresholding guidelines
    - Custom PII patterns
    - Clinical use cases (discharge summaries, research datasets, HIPAA compliance)
    - HTML visualization examples
    - CLI usage reference
    - Best practices and security considerations
  - [Notebooks README](examples/notebooks/README.md)
    - Navigation guide for all notebooks
    - Learning paths for different user types
    - Quick reference table
  - Updated README.md with PII capabilities
  - Updated CLI documentation with PII commands
  - Updated feature map and documentation index

- **Testing**
  - Comprehensive PII extraction and de-identification test suite
  - Smart entity merging validation tests
  - All 5 de-identification methods tested
  - Complex clinical note integration tests

### Changed

- Default PII extraction behavior now uses smart entity merging (`use_smart_merging=True`)
- Enhanced model registry with PII detection category

## [0.4.0] - 2025-12-29

### Added

- **Interactive TUI (Terminal User Interface)**: Full-featured terminal workbench for clinical NER analysis
  - Rich text input with multi-line support
  - Color-coded entity highlighting in annotated view
  - Entity table with confidence bars sorted by score
  - Model switcher modal (F2) for switching between models
  - Configuration panel (F3) for adjusting threshold and settings
  - Profile switcher (F4) for quick dev/prod/test/fast presets
  - Analysis history (F5) with recall and deletion
  - Export results (F6) to JSON, CSV, or clipboard
  - File navigation (Ctrl+O) for loading text files
  - Status bar showing model, profile, threshold, and inference time
  - CLI command: `openmed tui`

- **TUI Documentation**: Comprehensive guide at `docs/tui.md`
  - Interface overview with ASCII preview
  - Keyboard shortcuts reference
  - Profile presets documentation
  - Export format examples
  - Python API usage

- **Website Updates**
  - New Python Toolkit section showcasing TUI, CLI, batch processing, and profiles
  - Interactive TUI preview with color-coded entities
  - CLI and TUI tabs in hero code block
  - Updated software version metadata

### Changed

- Updated mkdocs navigation to include TUI documentation

## [0.3.0] - 2025-12-26

### Added

- **Batch Processing**: Process multiple texts or files in a single operation
  - `BatchProcessor` class for configurable batch operations
  - `BatchItem`, `BatchItemResult`, `BatchResult` dataclasses
  - `process_batch()` convenience function
  - File discovery with glob patterns and recursive search
  - Progress callbacks for monitoring long-running jobs
  - Configurable error handling (fail-fast or continue)
  - CLI `batch` command with full feature support

- **Configuration Profiles**: Named configuration presets for different environments
  - Built-in profiles: `dev`, `prod`, `test`, `fast`
  - `OpenMedConfig.from_profile()` and `with_profile()` methods
  - `list_profiles()`, `get_profile()`, `save_profile()`, `delete_profile()` functions
  - Custom profile persistence to disk
  - CLI commands: `config profiles`, `profile-show`, `profile-use`, `profile-save`, `profile-delete`
  - `--profile` flag for `config show` command

- **Performance Profiling**: Built-in timing and metrics utilities
  - `Timer` context manager for measuring code blocks
  - `Profiler` class for tracking metrics across multiple runs
  - `@profile` decorator for easy function profiling
  - `ProfilingMetrics` dataclass for structured timing data
  - Support for nested profiling and statistical aggregation

- **Documentation**
  - New [Batch Processing](./docs/batch-processing.md) guide
  - New [Configuration Profiles](./docs/profiles.md) guide
  - New [Performance Profiling](./docs/profiling.md) guide
  - Updated CLI documentation with new commands
  - Updated feature map and documentation index

- **Testing**
  - 89 new unit tests for batch, profiles, and profiling modules
  - Total test count: 218 passing tests

## [0.2.2] - 2024-12-20

### Added

- Medical-aware tokenizer with customizable exceptions
- CLI `--use-medical-tokenizer` and `--medical-tokenizer-exceptions` flags

### Fixed

- Token boundary issues with medical terminology

## [0.2.1] - 2024-12-18

### Added

- GLiNER2 support for zero-shot NER
- Enhanced model registry with GLiNER2 family

## [0.2.0] - 2024-12-15

### Added

- Typer-based CLI interface (`openmed` command)
- `analyze` command for single text analysis
- `models list` and `models info` commands
- `config show` and `config set` commands
- Rich terminal output formatting

### Changed

- Migrated CLI from argparse to Typer

## [0.1.10] - 2024-12-10

### Added

- Initial public release
- Core NER pipeline with HuggingFace integration
- Model registry with curated biomedical models
- `analyze_text()` one-call inference API
- Advanced NER post-processing (grouping, filtering)
- Multiple output formats (dict, JSON, HTML, CSV)
- YAML/ENV configuration via `OpenMedConfig`
- Zero-shot toolkit with GLiNER support

[Unreleased]: https://github.com/maziyarpanahi/openmed/compare/v1.9.1...HEAD
[1.9.1]: https://github.com/maziyarpanahi/openmed/compare/v1.9.0...v1.9.1
[1.9.0]: https://github.com/maziyarpanahi/openmed/compare/v1.8.1...v1.9.0
[1.8.1]: https://github.com/maziyarpanahi/openmed/compare/v1.8.0...v1.8.1
[1.8.0]: https://github.com/maziyarpanahi/openmed/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/maziyarpanahi/openmed/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/maziyarpanahi/openmed/compare/v1.5.5...v1.6.0
[0.6.1]: https://github.com/OpenMed/openmed/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/OpenMed/openmed/compare/v0.5.8...v0.6.0
[0.5.8]: https://github.com/OpenMed/openmed/compare/v0.5.7...v0.5.8
[0.5.7]: https://github.com/OpenMed/openmed/compare/v0.5.6...v0.5.7
[0.5.6]: https://github.com/OpenMed/openmed/compare/v0.5.5...v0.5.6
[0.5.5]: https://github.com/OpenMed/openmed/compare/v0.5.1...v0.5.5
[0.5.1]: https://github.com/OpenMed/openmed/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/OpenMed/openmed/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/OpenMed/openmed/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/OpenMed/openmed/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/OpenMed/openmed/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/OpenMed/openmed/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/OpenMed/openmed/compare/v0.1.10...v0.2.0
[0.1.10]: https://github.com/OpenMed/openmed/releases/tag/v0.1.10
