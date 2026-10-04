"""Offline locale-pack conformance checks for ``openmed i18n check``.

The command inspects one locale pack and reports one independent result per
component, so a missing registry entry, national-ID validator, surrogate locale,
synthetic fixture, span payload, or evidence reference can never mask another
gap. Checks run completely offline: no model weights, network access,
credentials, or real clinical text are required.

Every finding carries a stable reason code from :data:`REASON_CODES`; the same
report renders as deterministic JSON (``--json``) or as concise text. Component
inputs are opt-in flags -- ``--fixture-root``, ``--spans`` and ``--evidence`` --
and a component whose inputs were not supplied is reported as ``skipped``
instead of silently passing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..core.language_pack import (
    LANGUAGE_PACK_REGISTRY,
    LanguagePack,
    LanguagePackRegistry,
    get_language_pack,
)
from ..core.language_pack_catalog import LANG_TO_LOCALE, is_registered_segmenter
from ..core.locale_tag import normalize_locale_tag
from ._output import EXIT_ERROR, EXIT_OK, EXIT_USAGE, CliError, emit

I18N_CHECK_SCHEMA_VERSION: Final[str] = "openmed.i18n.locale_pack_conformance.v1"

CHECK_COMPONENTS: Final[tuple[str, ...]] = (
    "metadata",
    "registry",
    "validator",
    "surrogate",
    "fixtures",
    "span_integrity",
    "evidence",
)

CHECK_STATUSES: Final[tuple[str, ...]] = ("pass", "fail", "skipped")

PASS: Final[str] = "pass"
FAIL: Final[str] = "fail"
SKIPPED: Final[str] = "skipped"

REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        # metadata
        "pack_declared",
        "pack_not_registered",
        "metadata_language_mismatch",
        "metadata_scripts_missing",
        "metadata_default_model_missing",
        # registry
        "registry_wired",
        "registry_not_wired",
        "segmenter_not_registered",
        "recognizer_invalid",
        # validator
        "national_id_providers_resolved",
        "no_national_id_providers",
        "national_id_provider_invalid",
        "national_id_validator_missing",
        # surrogate
        "surrogate_locale_resolved",
        "surrogate_locale_missing",
        "surrogate_locale_invalid",
        # fixtures
        "fixtures_verified",
        "no_fixture_roots",
        "fixture_root_missing",
        "fixture_unreadable",
        "fixture_contains_text",
        "fixture_not_synthetic",
        "fixture_language_missing",
        # span integrity
        "spans_verified",
        "no_span_payloads",
        "span_payload_invalid",
        "span_integrity_failed",
        # evidence
        "evidence_verified",
        "no_evidence_references",
        "evidence_reference_missing",
        "evidence_reference_duplicate",
    }
)

_LANGUAGE_CODE = re.compile(r"^[a-z]{2}$")
_VERIFIED_SYNTHETIC: Final[str] = "verified_synthetic"
_TEXT_KEYS: Final[frozenset[str]] = frozenset(
    {"body", "content", "examples", "note", "notes", "raw_text", "text", "transcript"}
)
_FIXTURE_SUFFIXES: Final[frozenset[str]] = frozenset({".json", ".jsonl"})


@dataclass(frozen=True, slots=True)
class ConformanceFinding:
    """One component result with a stable, actionable reason code."""

    component: str
    status: str
    reason: str
    detail: str

    def __post_init__(self) -> None:
        """Reject components, statuses, and reasons outside the closed sets."""

        if self.component not in CHECK_COMPONENTS:
            raise ValueError(f"unknown conformance component {self.component!r}")
        if self.status not in CHECK_STATUSES:
            raise ValueError(f"unknown conformance status {self.status!r}")
        if self.reason not in REASON_CODES:
            raise ValueError(f"unknown conformance reason {self.reason!r}")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready finding mapping."""

        return {
            "component": self.component,
            "detail": self.detail,
            "reason": self.reason,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    """Deterministic conformance report for one locale pack."""

    language: str
    findings: tuple[ConformanceFinding, ...]
    schema_version: str = I18N_CHECK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize the finding tuple and reject empty schema versions."""

        object.__setattr__(self, "findings", tuple(self.findings))
        if not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")

    @property
    def ok(self) -> bool:
        """Return whether every checked component passed."""

        return all(finding.status != FAIL for finding in self.findings)

    def counts(self) -> dict[str, int]:
        """Return the number of findings per status."""

        counts = {status: 0 for status in CHECK_STATUSES}
        for finding in self.findings:
            counts[finding.status] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report payload with stable key ordering."""

        counts = self.counts()
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "language": self.language,
            "ok": self.ok,
            "schema_version": self.schema_version,
            "summary": {
                "components": len(CHECK_COMPONENTS),
                "fail": counts[FAIL],
                "pass": counts[PASS],
                "skipped": counts[SKIPPED],
            },
        }


def run_locale_pack_conformance(
    language: str,
    *,
    pack: LanguagePack | None = None,
    registry: LanguagePackRegistry | None = None,
    fixture_roots: Sequence[str | Path] = (),
    span_payloads: Sequence[str | Path] = (),
    evidence_paths: Sequence[str | Path] = (),
    repository_root: str | Path | None = None,
) -> ConformanceReport:
    """Check one locale pack component by component without touching the network.

    Args:
        language: Lowercase ISO 639-1 code of the pack to check.
        pack: Explicit pack to check; defaults to the registered built-in pack.
        registry: Registry snapshot used to verify wiring; defaults to the
            process-local :data:`LANGUAGE_PACK_REGISTRY`.
        fixture_roots: Directories scanned for metadata-only synthetic fixtures.
        span_payloads: JSON files holding ``{"text", "spans"}`` payloads whose
            offsets must stay aligned with their serialized text.
        evidence_paths: Evidence files that must exist and hash to a stable
            ``repository:sha256:...``/``external:sha256:...`` path digest.
        repository_root: Root used to derive repository-relative evidence
            references; defaults to the current working directory.

    Returns:
        A :class:`ConformanceReport` ordered by :data:`CHECK_COMPONENTS`.

    Raises:
        ValueError: If ``language`` is not a lowercase ISO 639-1 code.
    """

    normalized = _require_language(language)
    resolved = pack if pack is not None else get_language_pack(normalized)
    root = Path(repository_root) if repository_root is not None else Path.cwd()

    findings: list[ConformanceFinding] = [
        *_metadata_findings(normalized, resolved),
        *_registry_findings(resolved, registry),
        *_validator_findings(resolved),
        *_surrogate_findings(resolved),
        *_fixture_findings(normalized, fixture_roots, root),
        *_span_findings(span_payloads, root),
        *_evidence_findings(evidence_paths, root),
    ]
    ordered = sorted(findings, key=lambda item: CHECK_COMPONENTS.index(item.component))
    return ConformanceReport(
        language=normalized,
        findings=tuple(_dedupe(ordered)),
    )


def format_conformance_report(report: ConformanceReport) -> str:
    """Render a report as one deterministic line per finding plus a summary."""

    lines = [
        f"locale pack conformance: {report.language} [{report.schema_version}]",
    ]
    for finding in report.findings:
        lines.append(
            f"{finding.component} {finding.status} {finding.reason}: {finding.detail}"
        )
    counts = report.counts()
    lines.append(
        "summary "
        f"pass={counts[PASS]} fail={counts[FAIL]} skipped={counts[SKIPPED]} "
        f"ok={'true' if report.ok else 'false'}"
    )
    return "\n".join(lines)


def add_i18n_command(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the ``openmed i18n`` command group and its ``check`` leaf."""

    description = (
        "Inspect one locale pack offline and report each conformance component "
        "independently."
    )
    i18n_parser = subparsers.add_parser(
        "i18n",
        help="Check locale-pack conformance without models or network access.",
        description=description,
    )
    i18n_parser.set_defaults(handler=_help_handler(i18n_parser))

    i18n_subparsers = i18n_parser.add_subparsers(dest="i18n_command")
    check_parser = i18n_subparsers.add_parser(
        "check",
        help="Check registry, validator, surrogate, fixture, span, and evidence wiring.",
        description=description,
    )
    check_parser.add_argument(
        "language",
        help="Lowercase ISO 639-1 language code of the pack to check, e.g. 'zh'.",
    )
    check_parser.add_argument(
        "--fixture-root",
        action="append",
        default=[],
        metavar="PATH",
        help="Synthetic fixture root scanned for this language; repeatable.",
    )
    check_parser.add_argument(
        "--spans",
        action="append",
        default=[],
        metavar="PATH",
        help="JSON span payload whose offsets must match its text; repeatable.",
    )
    check_parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="PATH",
        help="Evidence path resolved to a stable path digest; repeatable.",
    )
    check_parser.set_defaults(handler=_handle_i18n_check)
    return i18n_parser


def _handle_i18n_check(args: argparse.Namespace) -> int:
    """Run the conformance check and emit its report."""

    try:
        report = run_locale_pack_conformance(
            args.language,
            fixture_roots=tuple(getattr(args, "fixture_root", ()) or ()),
            span_payloads=tuple(getattr(args, "spans", ()) or ()),
            evidence_paths=tuple(getattr(args, "evidence", ()) or ()),
        )
    except ValueError as exc:
        raise CliError(
            str(exc),
            code="invalid_language",
            exit_code=EXIT_USAGE,
        ) from exc

    emit(args, report.to_dict(), human=format_conformance_report(report))
    return EXIT_OK if report.ok else EXIT_ERROR


def _help_handler(
    parser: argparse.ArgumentParser,
) -> Callable[[argparse.Namespace], int]:
    """Return a handler that prints group help and reports a usage error."""

    def handler(args: argparse.Namespace) -> int:
        del args
        parser.print_help(sys.stderr)
        return EXIT_USAGE

    return handler


def _require_language(value: object) -> str:
    """Validate a lowercase ISO 639-1 language code."""

    if not isinstance(value, str) or not _LANGUAGE_CODE.fullmatch(value):
        raise ValueError(
            "language must be a lowercase ISO 639-1 code such as 'en' or 'zh'"
        )
    return value


def _finding(
    component: str,
    status: str,
    reason: str,
    detail: str,
) -> ConformanceFinding:
    """Build a finding, keeping construction sites terse."""

    return ConformanceFinding(
        component=component,
        status=status,
        reason=reason,
        detail=detail,
    )


def _unavailable(component: str) -> list[ConformanceFinding]:
    """Report a component that cannot be checked without a pack declaration."""

    return [
        _finding(
            component,
            FAIL,
            "pack_not_registered",
            "no locale pack is declared for this language",
        )
    ]


def _dedupe(findings: Sequence[ConformanceFinding]) -> list[ConformanceFinding]:
    """Drop duplicate findings while preserving order."""

    seen: set[tuple[str, str, str, str]] = set()
    unique: list[ConformanceFinding] = []
    for finding in findings:
        key = (finding.component, finding.status, finding.reason, finding.detail)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


def _metadata_findings(
    language: str,
    pack: LanguagePack | None,
) -> list[ConformanceFinding]:
    """Check that a pack declaration exists and is internally consistent."""

    if pack is None:
        return _unavailable("metadata")

    findings: list[ConformanceFinding] = []
    if pack.code != language:
        findings.append(
            _finding(
                "metadata",
                FAIL,
                "metadata_language_mismatch",
                f"pack code {pack.code!r} does not match requested {language!r}",
            )
        )
    if not pack.scripts:
        findings.append(
            _finding(
                "metadata",
                FAIL,
                "metadata_scripts_missing",
                "pack declares no Unicode scripts",
            )
        )
    if not pack.default_model.strip():
        findings.append(
            _finding(
                "metadata",
                FAIL,
                "metadata_default_model_missing",
                "pack declares no default model",
            )
        )
    if findings:
        return findings
    return [
        _finding(
            "metadata",
            PASS,
            "pack_declared",
            f"pack {pack.code!r} declares scripts {', '.join(pack.scripts)}",
        )
    ]


def _registry_findings(
    pack: LanguagePack | None,
    registry: LanguagePackRegistry | None,
) -> list[ConformanceFinding]:
    """Check registry wiring, segmenter registration, and recognizer ids."""

    if pack is None:
        return _unavailable("registry")

    snapshot = LANGUAGE_PACK_REGISTRY if registry is None else registry
    findings: list[ConformanceFinding] = []
    if snapshot.find(pack.code) is None:
        findings.append(
            _finding(
                "registry",
                FAIL,
                "registry_not_wired",
                f"pack {pack.code!r} is not registered in the locale-pack registry",
            )
        )
    if not is_registered_segmenter(pack.segmenter_id):
        findings.append(
            _finding(
                "registry",
                FAIL,
                "segmenter_not_registered",
                f"segmenter {pack.segmenter_id!r} is not a registered segmenter",
            )
        )
    malformed = [
        recognizer
        for recognizer in pack.recognizers
        if not isinstance(recognizer, str) or not recognizer.strip()
    ]
    if malformed:
        findings.append(
            _finding(
                "registry",
                FAIL,
                "recognizer_invalid",
                f"{len(malformed)} recognizer identifier(s) are not usable names",
            )
        )
    if findings:
        return findings
    return [
        _finding(
            "registry",
            PASS,
            "registry_wired",
            f"pack {pack.code!r} resolves segmenter {pack.segmenter_id!r}",
        )
    ]


def _validator_findings(pack: LanguagePack | None) -> list[ConformanceFinding]:
    """Check that every declared national-ID provider has a real validator."""

    if pack is None:
        return _unavailable("validator")

    providers = pack.national_id_providers
    if not providers:
        return [
            _finding(
                "validator",
                SKIPPED,
                "no_national_id_providers",
                "pack declares no national-ID providers",
            )
        ]

    # Imported lazily so the lightweight conformance path does not pull the
    # full PII pattern catalog unless a pack actually declares providers.
    from ..core import pii_i18n

    findings: list[ConformanceFinding] = []
    for provider, locale in providers.items():
        if (
            not isinstance(provider, str)
            or not provider.strip()
            or not isinstance(locale, str)
            or not locale.strip()
        ):
            findings.append(
                _finding(
                    "validator",
                    FAIL,
                    "national_id_provider_invalid",
                    f"national-ID provider entry {provider!r} is malformed",
                )
            )
            continue
        validator = getattr(pii_i18n, f"validate_{provider}", None)
        if not callable(validator):
            findings.append(
                _finding(
                    "validator",
                    FAIL,
                    "national_id_validator_missing",
                    f"pii_i18n has no validate_{provider} for locale {locale!r}",
                )
            )
    if findings:
        return findings
    return [
        _finding(
            "validator",
            PASS,
            "national_id_providers_resolved",
            f"resolved {len(providers)} national-ID validator(s)",
        )
    ]


def _surrogate_findings(pack: LanguagePack | None) -> list[ConformanceFinding]:
    """Check that the surrogate locale is a catalog locale or canonical tag."""

    if pack is None:
        return _unavailable("surrogate")

    locale = pack.surrogate_locale
    if not isinstance(locale, str) or not locale.strip():
        return [
            _finding(
                "surrogate",
                FAIL,
                "surrogate_locale_missing",
                "pack declares no surrogate locale",
            )
        ]
    catalog_locales = frozenset(LANG_TO_LOCALE.values())
    if locale in catalog_locales or _canonical_locale(locale) is not None:
        return [
            _finding(
                "surrogate",
                PASS,
                "surrogate_locale_resolved",
                f"surrogate locale {locale!r} resolves",
            )
        ]
    return [
        _finding(
            "surrogate",
            FAIL,
            "surrogate_locale_invalid",
            f"surrogate locale {locale!r} is neither a catalog nor a BCP 47 tag",
        )
    ]


def _canonical_locale(value: str) -> str | None:
    """Return the canonical form of a locale tag, or ``None`` when invalid."""

    try:
        return normalize_locale_tag(value)
    except ValueError:
        return None


def _fixture_findings(
    language: str,
    roots: Sequence[str | Path],
    repository_root: Path,
) -> list[ConformanceFinding]:
    """Check that verified synthetic fixtures exist for the pack's language."""

    if not roots:
        return [
            _finding(
                "fixtures",
                SKIPPED,
                "no_fixture_roots",
                "no --fixture-root was supplied",
            )
        ]

    findings: list[ConformanceFinding] = []
    verified = 0
    for raw_root in roots:
        root = Path(raw_root)
        display_root = _display_path(root, repository_root)
        if not root.is_dir():
            findings.append(
                _finding(
                    "fixtures",
                    FAIL,
                    "fixture_root_missing",
                    f"fixture root {display_root} is not a directory",
                )
            )
            continue
        for path in _fixture_files(root):
            display = _display_path(path, repository_root)
            try:
                records = tuple(_read_fixture_records(path))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                findings.append(
                    _finding(
                        "fixtures",
                        FAIL,
                        "fixture_unreadable",
                        f"fixture {display} could not be parsed",
                    )
                )
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    findings.append(
                        _finding(
                            "fixtures",
                            FAIL,
                            "fixture_unreadable",
                            f"fixture {display} holds a non-object record",
                        )
                    )
                    continue
                if _TEXT_KEYS & set(record):
                    findings.append(
                        _finding(
                            "fixtures",
                            FAIL,
                            "fixture_contains_text",
                            f"fixture {display} carries raw text keys",
                        )
                    )
                    continue
                if not _record_targets_language(record, language):
                    continue
                safety = record.get("safety")
                if safety != _VERIFIED_SYNTHETIC:
                    findings.append(
                        _finding(
                            "fixtures",
                            FAIL,
                            "fixture_not_synthetic",
                            f"fixture {display} declares safety {safety!r}",
                        )
                    )
                    continue
                verified += 1

    if findings:
        return findings
    if not verified:
        return [
            _finding(
                "fixtures",
                FAIL,
                "fixture_language_missing",
                f"no verified synthetic fixture declares language {language!r}",
            )
        ]
    return [
        _finding(
            "fixtures",
            PASS,
            "fixtures_verified",
            f"{verified} verified synthetic fixture record(s) for {language!r}",
        )
    ]


def _fixture_files(root: Path) -> tuple[Path, ...]:
    """Return the deterministic fixture file list below ``root``."""

    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in _FIXTURE_SUFFIXES
    )


def _read_fixture_records(path: Path) -> Iterator[Any]:
    """Yield every record in a JSON or JSONL fixture file."""

    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
        return
    payload = json.loads(text)
    if isinstance(payload, list):
        yield from payload
    else:
        yield payload


def _record_targets_language(record: Mapping[str, Any], language: str) -> bool:
    """Return whether a fixture record declares the requested language."""

    declared = record.get("language")
    if isinstance(declared, str) and declared == language:
        return True
    languages = record.get("languages")
    if isinstance(languages, str):
        return languages == language
    if isinstance(languages, Sequence) and not isinstance(languages, (str, bytes)):
        return language in languages
    return False


def _span_findings(
    payloads: Sequence[str | Path],
    repository_root: Path,
) -> list[ConformanceFinding]:
    """Check that supplied span payloads still point at their serialized text."""

    if not payloads:
        return [
            _finding(
                "span_integrity",
                SKIPPED,
                "no_span_payloads",
                "no --spans payload was supplied",
            )
        ]

    from ..training.synthetic.offset_projection import (
        SpanAnnotation,
        SpanProjectionError,
        validate_span_integrity,
    )

    findings: list[ConformanceFinding] = []
    verified = 0
    for raw_path in payloads:
        path = Path(raw_path)
        display = _display_path(path, repository_root)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            findings.append(
                _finding(
                    "span_integrity",
                    FAIL,
                    "span_payload_invalid",
                    f"span payload {display} could not be parsed",
                )
            )
            continue
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            try:
                text, spans = _coerce_span_payload(entry, SpanAnnotation)
            except ValueError as exc:
                findings.append(
                    _finding(
                        "span_integrity",
                        FAIL,
                        "span_payload_invalid",
                        f"span payload {display} is malformed: {exc}",
                    )
                )
                continue
            try:
                validate_span_integrity(text, spans)
            except SpanProjectionError as exc:
                findings.append(
                    _finding(
                        "span_integrity",
                        FAIL,
                        "span_integrity_failed",
                        f"span payload {display} failed integrity: {exc}",
                    )
                )
                continue
            verified += 1

    if findings:
        return findings
    if not verified:
        return [
            _finding(
                "span_integrity",
                FAIL,
                "span_payload_invalid",
                "span payload files held no annotated example",
            )
        ]
    return [
        _finding(
            "span_integrity",
            PASS,
            "spans_verified",
            f"{verified} span payload(s) keep offsets aligned",
        )
    ]


def _coerce_span_payload(
    entry: Any,
    annotation: Any,
) -> tuple[str, tuple[Any, ...]]:
    """Build ``(text, spans)`` from one JSON span payload entry."""

    if not isinstance(entry, Mapping):
        raise ValueError("payload must be a JSON object")
    text = entry.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("payload text must be a non-empty string")
    raw_spans = entry.get("spans")
    if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
        raise ValueError("payload spans must be a list")
    if not raw_spans:
        raise ValueError("payload declares no spans")

    spans: list[Any] = []
    for raw_span in raw_spans:
        if not isinstance(raw_span, Mapping):
            raise ValueError("each span must be a JSON object")
        start = raw_span.get("start")
        end = raw_span.get("end")
        label = raw_span.get("label")
        span_text = raw_span.get("text")
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("span start must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError("span end must be an integer")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("span label must be a non-empty string")
        if not isinstance(span_text, str):
            raise ValueError("span text must be a string")
        spans.append(annotation(start=start, end=end, label=label, text=span_text))
    return text, tuple(spans)


def _evidence_findings(
    paths: Sequence[str | Path],
    repository_root: Path,
) -> list[ConformanceFinding]:
    """Check that every evidence path exists and hashes to a unique reference."""

    if not paths:
        return [
            _finding(
                "evidence",
                SKIPPED,
                "no_evidence_references",
                "no --evidence path was supplied",
            )
        ]

    findings: list[ConformanceFinding] = []
    seen: dict[str, str] = {}
    verified = 0
    for raw_path in paths:
        path = Path(raw_path)
        display = _display_path(path, repository_root)
        if not path.is_file():
            findings.append(
                _finding(
                    "evidence",
                    FAIL,
                    "evidence_reference_missing",
                    f"evidence path {display} does not exist",
                )
            )
            continue
        relative, scope = _relative_reference(path, repository_root)
        digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        reference = f"{scope}:sha256:{digest}"
        if reference in seen:
            findings.append(
                _finding(
                    "evidence",
                    FAIL,
                    "evidence_reference_duplicate",
                    f"evidence {display} duplicates {seen[reference]}",
                )
            )
            continue
        seen[reference] = display
        verified += 1

    if findings:
        return findings
    return [
        _finding(
            "evidence",
            PASS,
            "evidence_verified",
            f"{verified} evidence reference(s) resolve",
        )
    ]


def _relative_reference(path: Path, repository_root: Path) -> tuple[str, str]:
    """Return ``(reference_path, scope)`` for a resolved evidence file."""

    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix(), (
            "repository"
        )
    except ValueError:
        return path.resolve().as_posix(), "external"


def _display_path(path: Path, repository_root: Path) -> str:
    """Return a stable, non-leaking display path for a finding detail."""

    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.name


__all__ = [
    "CHECK_COMPONENTS",
    "CHECK_STATUSES",
    "ConformanceFinding",
    "ConformanceReport",
    "FAIL",
    "I18N_CHECK_SCHEMA_VERSION",
    "PASS",
    "REASON_CODES",
    "SKIPPED",
    "add_i18n_command",
    "format_conformance_report",
    "run_locale_pack_conformance",
]
