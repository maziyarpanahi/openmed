# Changelog

All notable changes to OpenMed will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2024-12-26

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

[Unreleased]: https://github.com/OpenMed/openmed/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/OpenMed/openmed/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/OpenMed/openmed/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/OpenMed/openmed/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/OpenMed/openmed/compare/v0.1.10...v0.2.0
[0.1.10]: https://github.com/OpenMed/openmed/releases/tag/v0.1.10
