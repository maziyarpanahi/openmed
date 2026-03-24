# Changelog

All notable changes to OpenMed will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/OpenMed/openmed/compare/v0.6.1...HEAD
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
