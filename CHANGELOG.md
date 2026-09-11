# Changelog

All notable changes to OpenMed will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added deterministic counts-only trace privacy audit artifacts with canonical
  policy and file hashes, immutable category counts, value-free JSON and
  Markdown renderings, stable file fingerprinting, and private atomic writes
  (#2302).
- Added deterministic, PHI-free local release compute, cost, energy, and
  carbon tracking with orchestrator-linked stage timings, per-run and rolling
  budget verdicts, family/tier/workload breakdowns, optional advisory queue
  throttling, and hash-verified ledger replay (#1244).
- Added a bounded, deterministic nested-resource redaction contract with
  explicit scalar paths and actions, stable arrays and identifiers, closed
  policy validation, and raw-value-free reports and failures (#2413).
- Added declarative field-level FHIR and OMOP de-identification policies with
  fail-closed identifier handling, patient-consistent date shifting, schema
  linting, CSV/Parquet support, and resumable FHIR NDJSON integration (#2187).
