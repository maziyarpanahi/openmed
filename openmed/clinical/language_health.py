"""Offline health checks for language routes, models, fixtures, and policies.

The matrix joins the process-local language-pack registry with the committed
model manifest, local synthetic fixtures, and bundled policy profiles.  It is
diagnostic metadata rather than a clinical qualification or a model-quality
claim.  Fixture contents are deliberately never copied into the report.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..core.language_pack import LANGUAGE_PACK_REGISTRY, LanguagePackRegistry
from ..core.language_pack_catalog import (
    DEFAULT_MODEL_PLACEHOLDER_LANGUAGES,
    LANG_TO_LOCALE,
    NATIONAL_ID_ONLY_LANGUAGES,
    SCRIPT_LANGUAGE_HINTS,
    SUPPORTED_LANGUAGES,
    USER_SUPPLIED_MODEL_LANGUAGES,
    is_registered_segmenter,
)
from ..core.manifest_schema import LANGUAGE_SCRIPT_TARGETS
from ..core.model_registry import load_manifest_rows
from ..core.policy import list_policies, load_policy
from ..core.thresholds import load_thresholds

SCHEMA_VERSION = 1
COMPONENTS: tuple[str, ...] = ("route", "model", "fixture", "policy")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIXTURE_ROOTS: tuple[Path, ...] = (
    _REPO_ROOT / "tests" / "fixtures" / "i18n",
    _REPO_ROOT / "openmed" / "eval" / "golden" / "fixtures" / "i18n",
)
_UNSAFE_FIXTURE_KEYS = frozenset(
    {
        "contains_dua_data",
        "contains_phi",
        "contains_real_phi",
        "contains_restricted_data",
        "real_phi",
    }
)
_SYNTHETIC_FIXTURE_KEYS = frozenset({"synthetic", "synthetic_only"})
_PII_FAMILY = "pii"


class LanguageHealthError(RuntimeError):
    """Raised when a language health report contains one or more findings."""

    def __init__(self, issue_count: int) -> None:
        self.issue_count = issue_count
        super().__init__(f"language health matrix contains {issue_count} issue(s)")


def _normalize_language(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("_", "-").casefold()
    if not normalized:
        return None
    return normalized.split("-", 1)[0]


def _languages_from_payload(payload: Mapping[str, Any]) -> set[str]:
    values: list[object] = []
    if "language" in payload:
        values.append(payload["language"])
    languages = payload.get("languages")
    if isinstance(languages, Sequence) and not isinstance(languages, (str, bytes)):
        values.extend(languages)
    return {
        language
        for value in values
        if (language := _normalize_language(value)) is not None
    }


def _safety_flags(value: object) -> tuple[bool, bool]:
    """Return ``(synthetic, unsafe)`` without inspecting text values."""

    synthetic = False
    unsafe = False
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key).casefold()
            if normalized_key in _SYNTHETIC_FIXTURE_KEYS and child is True:
                synthetic = True
            if normalized_key in _SYNTHETIC_FIXTURE_KEYS and child is False:
                unsafe = True
            if normalized_key in _UNSAFE_FIXTURE_KEYS and child is True:
                unsafe = True
            if isinstance(child, (Mapping, list, tuple)):
                child_synthetic, child_unsafe = _safety_flags(child)
                synthetic = synthetic or child_synthetic
                unsafe = unsafe or child_unsafe
    elif isinstance(value, (list, tuple)):
        for child in value:
            child_synthetic, child_unsafe = _safety_flags(child)
            synthetic = synthetic or child_synthetic
            unsafe = unsafe or child_unsafe
    return synthetic, unsafe


def _fixture_safety_status(payloads: Sequence[Mapping[str, Any]]) -> str:
    if not payloads:
        return "unverified"
    synthetic_flags = [_safety_flags(payload) for payload in payloads]
    if any(unsafe for _synthetic, unsafe in synthetic_flags):
        return "unsafe"
    if all(synthetic for synthetic, _unsafe in synthetic_flags):
        return "verified_synthetic"
    return "unverified"


def _display_fixture_path(path: Path) -> str:
    try:
        return path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _fixture_record(
    path: Path,
) -> tuple[set[str], dict[str, Any], list[Mapping[str, Any]]]:
    """Read one fixture file while retaining only non-text metadata."""

    payloads: list[Mapping[str, Any]] = []
    parse_errors = 0
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        parse_errors = 1
    else:
        if path.suffix.casefold() == ".jsonl":
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                if isinstance(payload, Mapping):
                    payloads.append(payload)
                else:
                    parse_errors += 1
        else:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                parse_errors = 1
            else:
                if isinstance(payload, Mapping):
                    payloads.append(payload)
                else:
                    parse_errors = 1

    languages: set[str] = set()
    for payload in payloads:
        languages.update(_languages_from_payload(payload))

    safety = _fixture_safety_status(payloads)
    if parse_errors:
        safety = "unverified" if safety == "verified_synthetic" else safety
    metadata = {
        "path": _display_fixture_path(path),
        "record_count": len(payloads),
        "languages": sorted(languages),
        "safety": safety,
        "parse_errors": parse_errors,
        "includes_text": False,
    }
    return languages, metadata, payloads


def _collect_fixture_evidence(
    fixture_roots: Iterable[str | Path],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_findings: list[dict[str, Any]] = []
    for root_value in fixture_roots:
        root = Path(root_value)
        if not root.is_dir():
            global_findings.append(
                {
                    "language": None,
                    "component": "fixture",
                    "message": f"fixture root {_display_fixture_path(root)} is missing",
                }
            )
            continue
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".json", ".jsonl"}
        )
        for path in paths:
            languages, metadata, _payloads = _fixture_record(path)
            if metadata["parse_errors"]:
                global_findings.append(
                    {
                        "language": None,
                        "component": "fixture",
                        "message": (
                            f"fixture {metadata['path']} contains invalid JSON records"
                        ),
                    }
                )
            if not languages:
                continue
            for language in sorted(languages):
                by_language[language].append(dict(metadata))
    return by_language, global_findings


def _is_pii_manifest_row(row: Mapping[str, Any]) -> bool:
    repo_id = str(row.get("repo_id") or "").casefold()
    family = str(row.get("family") or "").casefold()
    return family == _PII_FAMILY or "pii" in repo_id or "privacy-filter" in repo_id


def _manifest_language_set(row: Mapping[str, Any]) -> set[str]:
    values = row.get("languages")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return set()
    return {
        language
        for value in values
        if (language := _normalize_language(value)) is not None
    }


def _manifest_indexes(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, list[Mapping[str, Any]]]]:
    all_by_repo: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pii_by_repo: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        repo_id = row.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id.strip():
            continue
        all_by_repo[repo_id].append(row)
        if _is_pii_manifest_row(row):
            pii_by_repo[repo_id].append(row)
    return all_by_repo, pii_by_repo


def _script_verdicts(
    language: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    targets = LANGUAGE_SCRIPT_TARGETS.get(language, ())
    verdicts: dict[str, str] = {}
    for entry in entries:
        coverage = entry.get("script_coverage")
        if not isinstance(coverage, Mapping):
            continue
        for target in targets:
            value = coverage.get(target)
            if isinstance(value, Mapping) and isinstance(value.get("verdict"), str):
                verdicts[target] = str(value["verdict"])
    return dict(sorted(verdicts.items()))


def _route_component(
    language: str,
    pack: Any | None,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    if pack is None:
        if language in USER_SUPPLIED_MODEL_LANGUAGES:
            return (
                {
                    "status": "user_supplied",
                    "kind": "user_supplied_model",
                    "registered": False,
                    "model": "user-supplied",
                    "scripts": [],
                    "script_hints": {},
                    "segmenter": None,
                    "recognizers": [],
                    "locale": LANG_TO_LOCALE.get(language),
                },
                issues,
            )
        if language in NATIONAL_ID_ONLY_LANGUAGES:
            return (
                {
                    "status": "limited",
                    "kind": "national_id_only",
                    "registered": False,
                    "model": None,
                    "scripts": [],
                    "script_hints": {},
                    "segmenter": None,
                    "recognizers": [],
                    "locale": None,
                },
                issues,
            )
        issues.append("language has no registered language-pack route")
        return (
            {
                "status": "missing",
                "kind": "unregistered",
                "registered": False,
                "model": None,
                "scripts": [],
                "script_hints": {},
                "segmenter": None,
                "recognizers": [],
                "locale": None,
            },
            issues,
        )

    if not is_registered_segmenter(pack.segmenter_id):
        issues.append(f"segmenter {pack.segmenter_id!r} is not registered")
    if not pack.scripts:
        issues.append("route declares no scripts")
    if not pack.recognizers:
        issues.append("route declares no recognizers")
    component = {
        "status": "filled" if not issues else "contradictory",
        "kind": "language_pack",
        "registered": True,
        "model": pack.default_model,
        "scripts": list(pack.scripts),
        "script_hints": {
            script: list(SCRIPT_LANGUAGE_HINTS.get(script, ()))
            for script in pack.scripts
        },
        "segmenter": pack.segmenter_id,
        "recognizers": list(pack.recognizers),
        "locale": pack.surrogate_locale,
    }
    return component, issues


def _model_component(
    language: str,
    route: Mapping[str, Any],
    all_by_repo: Mapping[str, Sequence[Mapping[str, Any]]],
    pii_by_repo: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    default_model = route.get("model")
    candidate_languages = sorted(
        repo_id
        for repo_id, entries in pii_by_repo.items()
        if any(language in _manifest_language_set(entry) for entry in entries)
    )

    if route.get("kind") == "national_id_only":
        return (
            {
                "status": "not_applicable",
                "default_model": None,
                "manifest_claimed_languages": [],
                "candidate_model_count": len(candidate_languages),
                "script_verdicts": {},
            },
            issues,
        )
    if route.get("kind") == "user_supplied_model":
        return (
            {
                "status": "user_supplied",
                "default_model": "user-supplied",
                "manifest_claimed_languages": [],
                "candidate_model_count": len(candidate_languages),
                "script_verdicts": {},
            },
            issues,
        )
    if not isinstance(default_model, str) or not default_model:
        status = "missing" if route.get("kind") == "unregistered" else "missing"
        issues.append("route has no default model declaration")
        return (
            {
                "status": status,
                "default_model": None,
                "manifest_claimed_languages": [],
                "candidate_model_count": len(candidate_languages),
                "script_verdicts": {},
            },
            issues,
        )

    all_entries = list(all_by_repo.get(default_model, ()))
    pii_entries = list(pii_by_repo.get(default_model, ()))
    claimed_languages = sorted(
        {claimed for entry in pii_entries for claimed in _manifest_language_set(entry)}
    )
    verdicts = _script_verdicts(language, pii_entries)
    if not pii_entries:
        if all_entries:
            issues.append("default model is not a PII-family manifest entry")
        else:
            issues.append("default model is absent from the local manifest")
        status = "missing"
    elif language in claimed_languages:
        unsupported = [
            target for target, verdict in verdicts.items() if verdict == "unsupported"
        ]
        if unsupported:
            issues.append(
                "default model marks a claimed script unsupported: "
                + ", ".join(unsupported)
            )
            status = "contradictory"
        else:
            status = "filled"
    elif language in DEFAULT_MODEL_PLACEHOLDER_LANGUAGES:
        issues.append("default model is a named fallback without a language claim")
        status = "fallback"
    else:
        issues.append("default model does not claim the routed language")
        status = "contradictory"

    if len(all_entries) > 1:
        issues.append("manifest contains duplicate rows for the default model")
        status = "contradictory"
    return (
        {
            "status": status,
            "default_model": default_model,
            "manifest_claimed_languages": claimed_languages,
            "candidate_model_count": len(candidate_languages),
            "script_verdicts": verdicts,
        },
        issues,
    )


def _policy_catalog(
    policy_names: Iterable[str] | None,
) -> tuple[list[str], list[str]]:
    names = list(policy_names) if policy_names is not None else list(list_policies())
    normalized = sorted({str(name).strip() for name in names if str(name).strip()})
    invalid: list[str] = []
    for name in normalized:
        try:
            load_policy(name)
        except Exception:
            invalid.append(name)
    return normalized, invalid


def _policy_component(
    pack: Any | None,
    policy_names: Sequence[str],
    invalid_policy_names: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    if invalid_policy_names:
        issues.append("one or more bundled policy profiles could not be loaded")
    if not policy_names:
        issues.append("no bundled policy profiles are available")

    threshold_profile = None
    if pack is not None:
        threshold_profile = pack.policy_overrides.get("profile")
        if threshold_profile is not None:
            profiles = load_thresholds().get("profiles", {})
            if threshold_profile not in profiles:
                issues.append(
                    f"route policy profile {threshold_profile!r} is not in the threshold matrix"
                )
    status = "filled"
    if not policy_names:
        status = "missing"
    elif invalid_policy_names or (
        pack is not None
        and threshold_profile is not None
        and threshold_profile not in load_thresholds().get("profiles", {})
    ):
        status = "contradictory"
    return (
        {
            "status": status,
            "profiles": list(policy_names),
            "profile_count": len(policy_names),
            "threshold_profile": threshold_profile,
            "scope": "route_override" if threshold_profile else "global",
        },
        issues,
    )


def _fixture_component(
    language: str,
    fixture_evidence: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    records = list(fixture_evidence.get(language, ()))
    issues: list[str] = []
    if not records:
        issues.append("no local fixture evidence is registered for the language")
        return (
            {
                "status": "missing",
                "files": [],
                "file_count": 0,
                "record_count": 0,
                "synthetic_only": False,
                "includes_text": False,
            },
            issues,
        )

    safety_values = {str(record.get("safety")) for record in records}
    parse_errors = sum(int(record.get("parse_errors", 0)) for record in records)
    if "unsafe" in safety_values:
        status = "contradictory"
        issues.append("fixture metadata does not certify synthetic-only data")
    elif safety_values != {"verified_synthetic"}:
        status = "unverified"
        issues.append("fixture metadata does not certify every record as synthetic")
    else:
        status = "filled"
    if parse_errors:
        status = "contradictory"
        issues.append("fixture set contains invalid JSON records")

    files = sorted({str(record["path"]) for record in records})
    return (
        {
            "status": status,
            "files": files,
            "file_count": len(files),
            "record_count": sum(
                int(record.get("record_count", 0)) for record in records
            ),
            "synthetic_only": status == "filled",
            "includes_text": False,
        },
        issues,
    )


def _overall_status(statuses: Sequence[str]) -> str:
    if "contradictory" in statuses:
        return "contradictory"
    if "missing" in statuses:
        return "missing"
    if any(
        status in {"fallback", "limited", "unverified", "user_supplied"}
        for status in statuses
    ):
        return "degraded"
    return "healthy"


def _default_languages(
    registry: LanguagePackRegistry,
    manifest_rows: Sequence[Mapping[str, Any]],
    fixture_evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    include_catalog_languages: bool,
) -> set[str]:
    languages = set(registry.iter_codes())
    if include_catalog_languages:
        languages.update(SUPPORTED_LANGUAGES)
        languages.update(NATIONAL_ID_ONLY_LANGUAGES)
        languages.update(USER_SUPPLIED_MODEL_LANGUAGES)
    for row in manifest_rows:
        if _is_pii_manifest_row(row):
            languages.update(_manifest_language_set(row))
    languages.update(fixture_evidence)
    return languages


def build_language_health_matrix(
    *,
    registry: LanguagePackRegistry | None = None,
    manifest_rows: Iterable[Mapping[str, Any]] | None = None,
    manifest_path: str | Path | None = None,
    fixture_roots: Iterable[str | Path] | None = None,
    languages: Iterable[str] | None = None,
    policy_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, JSON-ready language-route health matrix.

    Args:
        registry: Optional isolated language-pack registry for testing or
            review. The process-local registry is used by default.
        manifest_rows: Optional already-loaded local manifest rows. When
            supplied, no manifest file is read.
        manifest_path: Optional local manifest path used when ``manifest_rows``
            is omitted. No network access is performed.
        fixture_roots: Local JSON/JSONL fixture roots. The repository's two
            synthetic i18n roots are used by default.
        languages: Optional language-code subset. Region and script suffixes
            are normalized to their primary language code.
        policy_names: Optional local policy profile names. The bundled policy
            catalog is used by default.

    Returns:
        A dictionary containing sorted language rows, component findings, and
        aggregate counts. Fixture text and prediction values are never
        included.
    """

    resolved_registry = registry or LANGUAGE_PACK_REGISTRY
    if manifest_rows is not None:
        resolved_manifest_rows = [dict(row) for row in manifest_rows]
    else:
        if manifest_path is None:
            resolved_manifest_rows = load_manifest_rows()
        else:
            resolved_manifest_rows = load_manifest_rows(Path(manifest_path))
    resolved_fixture_roots = (
        tuple(fixture_roots) if fixture_roots is not None else _DEFAULT_FIXTURE_ROOTS
    )
    fixture_evidence, global_findings = _collect_fixture_evidence(
        resolved_fixture_roots
    )
    all_by_repo, pii_by_repo = _manifest_indexes(resolved_manifest_rows)
    resolved_policy_names, invalid_policy_names = _policy_catalog(policy_names)

    if languages is None:
        selected_languages = _default_languages(
            resolved_registry,
            resolved_manifest_rows,
            fixture_evidence,
            include_catalog_languages=registry is None,
        )
    else:
        selected_languages = {
            language
            for value in languages
            if (language := _normalize_language(value)) is not None
        }

    rows: list[dict[str, Any]] = []
    findings = list(global_findings)
    for language in sorted(selected_languages):
        pack = resolved_registry.find(language)
        route, route_issues = _route_component(language, pack)
        model, model_issues = _model_component(
            language,
            route,
            all_by_repo,
            pii_by_repo,
        )
        fixture, fixture_issues = _fixture_component(language, fixture_evidence)
        policy, policy_issues = _policy_component(
            pack,
            resolved_policy_names,
            invalid_policy_names,
        )
        component_issues = {
            "route": route_issues,
            "model": model_issues,
            "fixture": fixture_issues,
            "policy": policy_issues,
        }
        row_issues = [
            f"{component}: {message}"
            for component in COMPONENTS
            for message in component_issues[component]
        ]
        statuses = [
            str(route["status"]),
            str(model["status"]),
            str(fixture["status"]),
            str(policy["status"]),
        ]
        row = {
            "language": language,
            "status": _overall_status(statuses),
            "route": route,
            "model": model,
            "fixture": fixture,
            "policy": policy,
            "issues": row_issues,
        }
        rows.append(row)
        for component in COMPONENTS:
            for message in component_issues[component]:
                findings.append(
                    {
                        "language": language,
                        "component": component,
                        "message": message,
                    }
                )

    findings.sort(
        key=lambda finding: (
            str(finding.get("language") or ""),
            str(finding.get("component") or ""),
            str(finding.get("message") or ""),
        )
    )
    status_counts = {
        status: sum(1 for row in rows if row["status"] == status)
        for status in ("healthy", "degraded", "missing", "contradictory")
    }
    component_counts = {
        component: {
            status: sum(1 for row in rows if row[component]["status"] == status)
            for status in sorted({str(row[component]["status"]) for row in rows})
        }
        for component in COMPONENTS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "components": list(COMPONENTS),
        "languages": rows,
        "rows": rows,
        "summary": {
            "language_count": len(rows),
            "issue_count": len(findings),
            "status_counts": status_counts,
            "component_counts": component_counts,
        },
        "issues": findings,
        "sources": {
            "manifest": _display_fixture_path(
                Path(manifest_path)
                if manifest_path is not None
                else _REPO_ROOT / "models.jsonl"
            ),
            "fixture_roots": [
                _display_fixture_path(Path(root)) for root in resolved_fixture_roots
            ],
            "policy_profiles": resolved_policy_names,
            "includes_fixture_text": False,
        },
    }


def language_health_report(**kwargs: Any) -> dict[str, Any]:
    """Return :func:`build_language_health_matrix` under report terminology."""

    return build_language_health_matrix(**kwargs)


def check_language_health(**kwargs: Any) -> int:
    """Return the number of deterministic findings in the health matrix."""

    return int(build_language_health_matrix(**kwargs)["summary"]["issue_count"])


def require_language_health(**kwargs: Any) -> None:
    """Raise :class:`LanguageHealthError` when the matrix has findings."""

    issue_count = check_language_health(**kwargs)
    if issue_count:
        raise LanguageHealthError(issue_count)


__all__ = [
    "COMPONENTS",
    "LanguageHealthError",
    "SCHEMA_VERSION",
    "build_language_health_matrix",
    "check_language_health",
    "language_health_report",
    "require_language_health",
]
