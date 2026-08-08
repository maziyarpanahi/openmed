#!/usr/bin/env python3
"""Scan explicitly selected files for likely sensitive values.

The scanner is deliberately self-contained so a composite GitHub Action can
run it without installing dependencies or making a network request.  It never
prints matched text.  Reports and annotations contain only counts, rule names,
and repository-relative file names.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

DEFAULT_OUTPUT = Path("privacy-scan-report.json")
_SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
_SAFE_LABEL_VALUES = frozenset(
    {
        "<none>",
        "<placeholder>",
        "<redacted>",
        "[masked]",
        "[redacted]",
        "masked",
        "n/a",
        "na",
        "none",
        "null",
        "placeholder",
        "redacted",
        "synthetic",
        "test",
        "unknown",
    }
)


class PrivacyScanError(Exception):
    """A safe, intentionally detail-free scanner error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


Validator = Callable[[str], bool]


@dataclass(frozen=True)
class Rule:
    """A named deterministic detector."""

    name: str
    pattern: re.Pattern[str]
    validator: Validator | None = None


@dataclass(frozen=True)
class Policy:
    """The rules enabled for one scan."""

    name: str
    rules: tuple[str, ...]


@dataclass(frozen=True)
class Match:
    """A match represented without retaining the matched value."""

    rule: str
    start: int
    end: int


@dataclass(frozen=True)
class FileSummary:
    """Counts for one scanned file."""

    path: str
    findings: int
    rules: Mapping[str, int]


@dataclass(frozen=True)
class ScanResult:
    """Aggregate, PHI-safe results from a scan."""

    policy: Policy
    scanned_files: int
    allowlisted_files: int
    files: tuple[FileSummary, ...]
    findings_by_rule: Mapping[str, int]

    @property
    def total_findings(self) -> int:
        """Return the total number of non-overlapping findings."""
        return sum(self.findings_by_rule.values())


def _valid_credit_card(candidate: str) -> bool:
    """Return whether a digit sequence passes the Luhn check."""
    digits = [character for character in candidate if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False

    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        value = int(digit)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


_RULES: tuple[Rule, ...] = (
    Rule(
        "credential",
        re.compile(
            r"(?<![A-Za-z0-9])(?:"
            r"(?:AKIA|ASIA)[0-9A-Z]{16}|"
            r"gh[pousr]_[A-Za-z0-9_]{20,}|"
            r"(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}|"
            r"xox[baprs]-[A-Za-z0-9-]{16,}|"
            r"npm_[A-Za-z0-9]{20,}|"
            r"pypi-[A-Za-z0-9_-]{20,}"
            r")(?![A-Za-z0-9])"
        ),
    ),
    Rule(
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    Rule(
        "database_url",
        re.compile(
            r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://"
            r"[^\s\"'<>]+"
        ),
    ),
    Rule(
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
            r"[A-Za-z0-9_-]{8,}\b"
        ),
    ),
    Rule(
        "email",
        re.compile(
            r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
            r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
            r"[A-Za-z]{2,63}(?![\w.-])"
        ),
    ),
    Rule(
        "ssn",
        re.compile(r"(?<!\d)(?:\d{3}-\d{2}-\d{4})(?!\d)"),
    ),
    Rule(
        "credit_card",
        re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
        validator=_valid_credit_card,
    ),
    Rule(
        "phone",
        re.compile(
            r"(?<!\d)(?:\+?[1-9]\d{9,14}|"
            r"(?:\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4})"
            r"(?!\d)"
        ),
    ),
    Rule(
        "ip_address",
        re.compile(
            r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)\."
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\."
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\."
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
        ),
    ),
    Rule(
        "labeled_sensitive",
        re.compile(
            r"(?ix)(?<![a-z0-9_])(?:"
            r"access[_-]?token|account[_-]?number|address|api[_-]?(?:key|token)|"
            r"authorization|birth[_-]?date|"
            r"client[_-]?id|date[_-]?of[_-]?birth|dob|email|"
            r"encounter[_-]?id|first[_-]?name|full[_-]?name|health[_-]?id|"
            r"last[_-]?name|member[_-]?id|medical[_-]?record[_-]?number|"
            r"mrn|name|password|passphrase|patient(?:[_-]?name|[_-]?id)?|"
            r"person[_-]?name|phone|postal[_-]?code|record[_-]?id|secret|"
            r"social[_-]?security[_-]?number|ssn|subject(?:[_-]?id)?|"
            r"telephone|token|user(?:[_-]?id|[_-]?name|name)?|username|zip"
            r")\s*[\"']?\s*[:=]\s*[\"']?"
            r"(?P<value>[^\"',;\r\n}{]+?)"
            r"(?=[\"']?(?:[,;}\r\n]|$))"
        ),
    ),
    Rule(
        "uuid",
        re.compile(
            r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
        ),
    ),
    Rule(
        "long_numeric_id",
        re.compile(r"(?<!\d)\d{8,}(?!\d)"),
    ),
)
_RULE_BY_NAME = {rule.name: rule for rule in _RULES}
_DEFAULT_RULES = tuple(rule.name for rule in _RULES[:-2])
_POLICY_PRESETS = {
    "credentials": (
        "credential",
        "private_key",
        "database_url",
        "jwt",
    ),
    "default": _DEFAULT_RULES,
    "minimal": (
        "credential",
        "private_key",
        "email",
        "ssn",
        "labeled_sensitive",
    ),
    "strict": _DEFAULT_RULES + ("uuid", "long_numeric_id"),
}


def _safe_policy_name(value: object) -> str:
    """Return a bounded policy name without echoing arbitrary configuration."""
    candidate = str(value or "").strip()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", candidate):
        return candidate
    return "custom"


def _normalize_rule_names(values: object) -> tuple[str, ...]:
    """Validate and normalize rule names from a policy document."""
    if isinstance(values, Mapping):
        names = [str(name) for name, enabled in values.items() if enabled]
    elif isinstance(values, str):
        names = [part.strip() for part in values.split(",") if part.strip()]
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        names = [str(value).strip() for value in values if str(value).strip()]
    else:
        raise PrivacyScanError("policy_rules_invalid")

    normalized: list[str] = []
    for name in names:
        if name not in _RULE_BY_NAME or name in normalized:
            raise PrivacyScanError("policy_rule_unknown")
        normalized.append(name)
    if not normalized:
        raise PrivacyScanError("policy_rules_empty")
    return tuple(normalized)


def load_policy(value: str | Path | Policy | None = None) -> Policy:
    """Load a built-in policy or a JSON policy document.

    Policy documents use ``{"name": "...", "rules": ["email", ...]}``.
    The parser intentionally accepts JSON only so the composite action has no
    third-party runtime dependency.
    """
    if isinstance(value, Policy):
        return value

    configured = str(value or "").strip()
    if not configured or configured.casefold() == "default":
        return Policy("default", _POLICY_PRESETS["default"])

    preset = _POLICY_PRESETS.get(configured.casefold())
    if preset is not None:
        return Policy(configured.casefold(), preset)

    try:
        if configured.startswith("{"):
            document = json.loads(configured)
        else:
            document = json.loads(Path(configured).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        del exc
        raise PrivacyScanError("policy_unreadable") from None

    if not isinstance(document, Mapping):
        raise PrivacyScanError("policy_document_invalid")
    values = document.get("rules", document.get("enabled_rules"))
    rules = _normalize_rule_names(values)
    return Policy(_safe_policy_name(document.get("name")), rules)


def _split_lines(value: str) -> list[str]:
    """Split newline-delimited input while ignoring blank and comment lines."""
    return [
        line.strip()
        for line in value.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _looks_like_path_pattern(value: str) -> bool:
    """Identify a likely path-list entry without inspecting its contents."""
    if any(character in value for character in ("@", "=", ":")):
        return False
    return "/" in value or "\\" in value or "*" in value or value.count(".")


def _read_allowlist_file(path: Path) -> list[str]:
    """Read a JSON or newline-delimited path allowlist."""
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        del exc
        raise PrivacyScanError("allowlist_unreadable") from None

    if path.suffix.casefold() == ".json":
        try:
            document = json.loads(content)
        except json.JSONDecodeError as exc:
            del exc
            raise PrivacyScanError("allowlist_invalid") from None
        if isinstance(document, Mapping):
            document = document.get("paths", document.get("patterns"))
        if not isinstance(document, Sequence) or isinstance(
            document, (str, bytes, bytearray)
        ):
            raise PrivacyScanError("allowlist_invalid")
        entries = [str(entry).strip() for entry in document if str(entry).strip()]
    else:
        entries = _split_lines(content)

    if not entries:
        raise PrivacyScanError("allowlist_empty")
    return entries


def load_allowlist(value: str | Path | None = None) -> tuple[str, ...]:
    """Load newline-delimited synthetic fixture paths or glob patterns.

    Prefix a path with ``@`` to explicitly load a path-list file.  An existing
    JSON/list-looking file is also accepted for convenience; a normal fixture
    path remains a direct pattern unless its contents look like path entries.
    """
    configured = str(value or "").strip()
    if not configured:
        return ()

    if configured.startswith("@"):
        return tuple(_read_allowlist_file(Path(configured[1:])))

    entries = _split_lines(configured)
    if len(entries) == 1:
        candidate = Path(entries[0])
        if candidate.is_file():
            if candidate.suffix.casefold() == ".json":
                return tuple(_read_allowlist_file(candidate))
            try:
                candidate_entries = _read_allowlist_file(candidate)
            except PrivacyScanError:
                candidate_entries = []
            if candidate_entries and all(
                _looks_like_path_pattern(entry) for entry in candidate_entries
            ):
                return tuple(candidate_entries)
    return tuple(entries)


def _flatten_input_values(values: str | Path | Iterable[str | Path]) -> list[str]:
    """Normalize newline-delimited path input into individual patterns."""
    if isinstance(values, (str, Path)):
        values = [values]
    flattened: list[str] = []
    for value in values:
        flattened.extend(_split_lines(os.fspath(value)))
    return flattened


def _expand_paths(values: Iterable[str], root: Path) -> list[Path]:
    """Expand explicit files, directories, and glob patterns deterministically."""

    def raise_walk_error(_error: OSError) -> None:
        raise PrivacyScanError("directory_unreadable")

    expanded: dict[str, Path] = {}
    for value in values:
        candidate = Path(value)
        pattern = str(candidate if candidate.is_absolute() else root / candidate)
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise PrivacyScanError("path_not_found")

        for match in matches:
            path = Path(match)
            if path.is_symlink():
                continue
            if path.is_file():
                resolved = path.resolve()
                expanded[os.path.normcase(str(resolved))] = resolved
                continue
            if not path.is_dir():
                continue

            for directory, directory_names, file_names in os.walk(
                path, onerror=raise_walk_error, followlinks=False
            ):
                directory_names[:] = sorted(
                    name
                    for name in directory_names
                    if name not in _SKIPPED_DIRECTORY_NAMES
                )
                for file_name in sorted(file_names):
                    file_path = Path(directory) / file_name
                    if file_path.is_symlink() or not file_path.is_file():
                        continue
                    resolved = file_path.resolve()
                    expanded[os.path.normcase(str(resolved))] = resolved

    return [expanded[key] for key in sorted(expanded)]


def _relative_path(path: Path, root: Path) -> str | None:
    """Return a stable repository-relative path when possible."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def _display_path(path: Path, root: Path) -> str:
    """Return a useful path that does not expose an external absolute path."""
    relative = _relative_path(path, root)
    if relative is not None:
        return relative or "."
    digest = hashlib.sha256(str(path).encode("utf-8", errors="replace")).hexdigest()
    return f"<external:{digest[:16]}>"


def _path_is_allowlisted(path: Path, root: Path, patterns: Iterable[str]) -> bool:
    """Return whether a repository-relative path matches an allowlist pattern."""
    relative = _relative_path(path, root)
    path_values = [_display_path(path, root), path.as_posix()]
    if relative is not None:
        path_values.append(relative)

    for raw_pattern in patterns:
        pattern = raw_pattern.replace("\\", "/")
        if pattern.startswith("./"):
            pattern = pattern[2:]
        if pattern.endswith("/"):
            pattern += "**"
        if any(fnmatch.fnmatchcase(path_value, pattern) for path_value in path_values):
            return True
        if relative is not None:
            try:
                if Path(relative).match(pattern):
                    return True
            except (IndexError, ValueError):
                continue
    return False


def _read_text(path: Path) -> str:
    """Read a file as UTF-8 without exposing decoder or path details."""
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError as exc:
        del exc
        raise PrivacyScanError("file_unreadable") from None


def _trimmed_match(match: re.Match[str]) -> tuple[int, int, str]:
    """Return match offsets and text only for local validation."""
    if "value" in match.re.groupindex:
        start, end = match.span("value")
    else:
        start, end = match.span()
    if start < 0 or end <= start:
        return start, end, ""

    value = match.string[start:end]
    leading = len(value) - len(value.lstrip())
    trailing = len(value) - len(value.rstrip())
    start += leading
    end -= trailing
    return start, end, match.string[start:end]


def _is_labeled_placeholder(value: str) -> bool:
    """Return whether a labeled value is an explicit non-sensitive marker."""
    return value.strip().casefold() in _SAFE_LABEL_VALUES


def _find_matches(text: str, policy: Policy) -> list[Match]:
    """Find non-overlapping matches without retaining their values."""
    candidates: list[tuple[int, int, int, Match]] = []
    for priority, rule_name in enumerate(policy.rules):
        rule = _RULE_BY_NAME[rule_name]
        for match in rule.pattern.finditer(text):
            start, end, value = _trimmed_match(match)
            if end <= start:
                continue
            if rule.validator is not None and not rule.validator(value):
                continue
            if rule.name == "labeled_sensitive" and _is_labeled_placeholder(value):
                continue
            candidates.append(
                (start, priority, -(end - start), Match(rule.name, start, end))
            )

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].end))
    accepted: list[Match] = []
    for _, _, _, candidate in candidates:
        if accepted and candidate.start < accepted[-1].end:
            continue
        accepted.append(candidate)
    return accepted


def scan_paths(
    paths: str | Path | Iterable[str | Path],
    policy: str | Path | Policy | None = None,
    synthetic_fixture_allowlist: str | Path | None = None,
    *,
    root: str | Path | None = None,
    excluded_paths: Iterable[str | Path] = (),
) -> ScanResult:
    """Scan explicit paths and return counts without matched values.

    ``paths`` may contain files, directories, or recursive glob patterns.  No
    default path is inferred.  Symlinks and common generated environment
    directories are skipped so the scan stays within explicitly selected local
    data.
    """
    root_path = Path(root or Path.cwd()).resolve()
    path_values = _flatten_input_values(paths)
    if not path_values:
        raise PrivacyScanError("paths_empty")

    configured_policy = load_policy(policy)
    allowlist = load_allowlist(synthetic_fixture_allowlist)
    files = _expand_paths(path_values, root_path)
    excluded = {Path(path).resolve() for path in _flatten_input_values(excluded_paths)}

    file_summaries: list[FileSummary] = []
    findings_by_rule: Counter[str] = Counter()
    allowlisted_files = 0
    for path in files:
        if path in excluded:
            continue
        display_path = _display_path(path, root_path)
        if _path_is_allowlisted(path, root_path, allowlist):
            allowlisted_files += 1
            continue

        matches = _find_matches(_read_text(path), configured_policy)
        counts = Counter(match.rule for match in matches)
        findings_by_rule.update(counts)
        file_summaries.append(
            FileSummary(
                path=display_path,
                findings=len(matches),
                rules=dict(sorted(counts.items())),
            )
        )

    file_summaries.sort(key=lambda summary: summary.path)
    return ScanResult(
        policy=configured_policy,
        scanned_files=len(file_summaries),
        allowlisted_files=allowlisted_files,
        files=tuple(file_summaries),
        findings_by_rule=dict(sorted(findings_by_rule.items())),
    )


def _result_payload(result: ScanResult) -> dict[str, object]:
    """Serialize a result using only aggregate, non-content fields."""
    total_findings = result.total_findings
    return {
        "allowlisted_files": result.allowlisted_files,
        "files": [
            {
                "findings": summary.findings,
                "path": summary.path,
                "rules": dict(summary.rules),
            }
            for summary in result.files
        ],
        "findings": total_findings,
        "findings_by_rule": dict(result.findings_by_rule),
        "policy": result.policy.name,
        "rules": list(result.policy.rules),
        "scanned_files": result.scanned_files,
        "schema_version": 1,
        "status": "failed" if total_findings else "passed",
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write deterministic JSON, translating all filesystem errors safely."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        del exc
        raise PrivacyScanError("report_unwritable") from None


def _write_error_report(path: Path, category: str) -> None:
    """Best-effort error artifact containing no exception details."""
    try:
        _write_json(
            path,
            {
                "error_category": category,
                "schema_version": 1,
                "status": "error",
            },
        )
    except PrivacyScanError:
        return


def _escape_command_field(value: str) -> str:
    """Escape a value for a GitHub Actions command without echoing content."""
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def _emit_annotations(result: ScanResult) -> None:
    """Emit one counts-only annotation per file with findings."""
    for summary in result.files:
        if not summary.findings:
            continue
        safe_path = _escape_command_field(summary.path)
        print(
            "::error "
            f"file={safe_path}::privacy scan found {summary.findings} "
            "potential sensitive value(s)"
        )


class _SafeArgumentParser(argparse.ArgumentParser):
    """Do not let argparse echo arbitrary argument values on failure."""

    def error(self, message: str) -> None:
        del message
        raise PrivacyScanError("invalid_arguments")


def _build_parser() -> argparse.ArgumentParser:
    """Build the scanner command-line parser."""
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Explicit files, directories, or newline-delimited glob patterns.",
    )
    parser.add_argument(
        "--path",
        action="append",
        dest="path_values",
        help="Additional explicit path; may be repeated.",
    )
    parser.add_argument(
        "--policy",
        default="default",
        help="Built-in policy name or a JSON policy document path.",
    )
    parser.add_argument(
        "--synthetic-fixture-allowlist",
        "--allowlist",
        default="",
        dest="synthetic_fixture_allowlist",
        help="Newline-delimited synthetic fixture paths/globs or @path-list.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Machine-readable JSON artifact path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the privacy scan and return a CI-friendly exit code."""
    output = DEFAULT_OUTPUT
    try:
        args = _build_parser().parse_args(argv)
        output = args.output
        paths = list(args.paths or []) + list(args.path_values or [])
        if not paths:
            raise PrivacyScanError("paths_empty")

        result = scan_paths(
            paths,
            policy=args.policy,
            synthetic_fixture_allowlist=args.synthetic_fixture_allowlist,
            excluded_paths=(output,),
        )
        _write_json(output, _result_payload(result))
        _emit_annotations(result)
        if result.total_findings:
            print(
                "privacy scan failed: "
                f"{result.total_findings} potential sensitive value(s) in "
                f"{sum(summary.findings > 0 for summary in result.files)} file(s)",
                file=sys.stderr,
            )
            return 1
        print(
            "privacy scan passed: "
            f"scanned {result.scanned_files} file(s), no potential sensitive "
            "values found"
        )
        return 0
    except PrivacyScanError:
        _write_error_report(output, "configuration")
        print("::error title=Privacy scan::privacy scan configuration error")
        print("privacy scan failed: configuration error", file=sys.stderr)
        return 2
    except Exception:
        _write_error_report(output, "internal")
        print("::error title=Privacy scan::privacy scan internal error")
        print("privacy scan failed: internal error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
