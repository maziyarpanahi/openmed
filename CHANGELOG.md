# Changelog

All notable changes to OpenMed will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/OpenMed/openmed/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/OpenMed/openmed/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/OpenMed/openmed/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/OpenMed/openmed/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/OpenMed/openmed/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/OpenMed/openmed/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/OpenMed/openmed/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/OpenMed/openmed/compare/v0.1.10...v0.2.0
[0.1.10]: https://github.com/OpenMed/openmed/releases/tag/v0.1.10
