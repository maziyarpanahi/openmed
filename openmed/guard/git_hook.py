"""Deterministic, local-only privacy checks for pre-push candidate files.

The module deliberately has no runtime dependencies beyond the Python
standard library.  A pre-push hook supplies the refs that Git is about to
publish; the scanner resolves those refs locally, selects only added and
modified paths, and scans the corresponding committed blobs.  Reports contain
paths, categories, counts, and line numbers only.  Matched values are never
stored in a finding or included in an exception, report, or JSON payload.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, TextIO

ALLOWLIST_VERSION = 1
SCANNER_VERSION = "1"
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
ZERO_SHA = "0" * 40
DEFAULT_MAX_BYTES = 2_000_000


class PrivacyScanError(RuntimeError):
    """Raised when the scanner cannot safely inspect candidate files."""


@dataclass(frozen=True, order=True)
class Finding:
    """A value-free privacy finding.

    Attributes:
        path: Repository-relative candidate path.
        category: Stable high-level finding category.
        line: One-based source line, or ``0`` when no line is available.
    """

    path: str
    category: str
    line: int


@dataclass(frozen=True)
class AllowlistRule:
    """A narrowly scoped synthetic-fixture allowlist rule."""

    path: str
    category: str
    pattern: re.Pattern[str]
    reason: str


@dataclass(frozen=True)
class PrivacyAllowlist:
    """Versioned rules used to suppress documented synthetic values."""

    version: int
    rules: tuple[AllowlistRule, ...]


@dataclass(frozen=True)
class ScanResult:
    """Value-free aggregate result for one local scan."""

    findings: tuple[Finding, ...]
    scanned_files: tuple[str, ...]
    skipped_files: tuple[str, ...]
    allowlist_version: int = ALLOWLIST_VERSION

    @property
    def passed(self) -> bool:
        """Return whether no candidate file produced a blocking finding."""

        return not self.findings

    @property
    def categories(self) -> dict[str, int]:
        """Return deterministic category counts without matched values."""

        counts = Counter(finding.category for finding in self.findings)
        return dict(sorted(counts.items()))

    @property
    def files(self) -> dict[str, dict[str, int]]:
        """Return deterministic per-file category counts."""

        grouped: dict[str, Counter[str]] = {}
        for finding in self.findings:
            grouped.setdefault(finding.path, Counter())[finding.category] += 1
        return {
            path: dict(sorted(counts.items()))
            for path, counts in sorted(grouped.items())
        }


@dataclass(frozen=True)
class PushUpdate:
    """One update record from Git's pre-push protocol."""

    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str


@dataclass(frozen=True)
class _Detector:
    category: str
    pattern: re.Pattern[str]
    validator: Callable[[re.Match[str]], bool] | None = None


def _compile(
    pattern: str, flags: int = re.IGNORECASE | re.MULTILINE
) -> re.Pattern[str]:
    return re.compile(pattern, flags)


_EMAIL_PATTERN = _compile(
    r"(?<![\w.+-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}(?![\w.-])"
)
_PHONE_PATTERN = _compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-])?(?:\(\d{3}\)|\d{3})"
    r"[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
)
_SSN_PATTERN = _compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_CARD_PATTERN = _compile(r"(?<!\d)(?:\d{4}[ -]?){3,4}\d{1,4}(?!\d)")
_IP_PATTERN = _compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_PRIVATE_KEY_PATTERN = _compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_CLOUD_ACCESS_KEY_PATTERN = _compile(r"\bAKIA[0-9A-Z]{16}\b")
_GITHUB_TOKEN_PATTERN = _compile(r"\b(?:gh[pousr])_[A-Za-z0-9]{20,}\b")
_OPENAI_TOKEN_PATTERN = _compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
_BEARER_TOKEN_PATTERN = _compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE)
_JWT_PATTERN = _compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_URL_CREDENTIAL_PATTERN = _compile(r"\bhttps?://[^\s/@:]+:[^\s/@]+@[^\s/]+")
_SECRET_ASSIGNMENT_PATTERN = _compile(
    r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"client[_-]?secret|password|passwd|secret)\b\s*[:=]\s*"
    r"(?P<quote>[\"']?)(?P<value>[^\s,\"'\}\]]+)"
)
_SENSITIVE_FIELD_PATTERN = _compile(
    r"(?P<key>[\"']?(?:patient[_ -]?(?:name|id|text)|full[_ -]?name|"
    r"medical[_ -]?record[_ -]?number|\bmrn\b|encounter[_ -]?id|"
    r"date[_ -]?of[_ -]?birth|\bdob\b|email|phone|address|"
    r"raw[_ -]?text|source[_ -]?text|document[_ -]?text|"
    r"prompt|completion)[\"']?)\s*[:=]\s*"
    r"(?P<quote>[\"'])(?P<value>[^\"'\r\n]{2,})(?P=quote)"
)


def _is_valid_ip(match: re.Match[str]) -> bool:
    try:
        address = ipaddress.ip_address(match.group(0))
    except ValueError:
        return False
    return isinstance(address, ipaddress.IPv4Address)


def _passes_luhn(match: re.Match[str]) -> bool:
    digits = re.sub(r"\D", "", match.group(0))
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        value = int(digit)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().casefold()
    if not normalized:
        return True
    if (normalized.startswith("<") and normalized.endswith(">")) or (
        normalized.startswith("[") and normalized.endswith("]")
    ):
        return True
    if normalized in {
        "changeme",
        "change_me",
        "dummy",
        "example",
        "fake",
        "fixture",
        "null",
        "none",
        "placeholder",
        "redacted",
        "sample",
        "synthetic",
        "test",
        "todo",
    }:
        return True
    return normalized.startswith(("example-", "example_", "synthetic-", "synthetic_"))


_DETECTORS = (
    _Detector("email", _EMAIL_PATTERN),
    _Detector("phone", _PHONE_PATTERN),
    _Detector("government_id", _SSN_PATTERN),
    _Detector("payment_card", _CARD_PATTERN, _passes_luhn),
    _Detector("ip_address", _IP_PATTERN, _is_valid_ip),
    _Detector("secret", _PRIVATE_KEY_PATTERN),
    _Detector("secret", _CLOUD_ACCESS_KEY_PATTERN),
    _Detector("secret", _GITHUB_TOKEN_PATTERN),
    _Detector("secret", _OPENAI_TOKEN_PATTERN),
    _Detector("secret", _BEARER_TOKEN_PATTERN),
    _Detector("secret", _JWT_PATTERN),
    _Detector("secret", _URL_CREDENTIAL_PATTERN),
    _Detector(
        "secret",
        _SECRET_ASSIGNMENT_PATTERN,
        lambda match: (
            not _is_placeholder(match.group("value")) and len(match.group("value")) >= 8
        ),
    ),
)


def _rule(
    path: str,
    category: str,
    pattern: str,
    reason: str,
) -> AllowlistRule:
    return AllowlistRule(path, category, _compile(pattern), reason)


DEFAULT_ALLOWLIST = PrivacyAllowlist(
    version=ALLOWLIST_VERSION,
    rules=(
        _rule(
            "*",
            "email",
            r"(?<![\w.+-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
            r"(?:example\.(?:com|net|org|test)|invalid|localhost)"
            r"(?![\w.-])",
            "RFC-reserved documentation mailbox",
        ),
        _rule(
            "*",
            "phone",
            r"(?<!\d)(?:\+?1[\s.-]?)?555[\s.-]01\d{2}(?!\d)",
            "reserved 555-01xx test telephone number",
        ),
        _rule(
            "*",
            "ip_address",
            r"(?<![\d.])(?:10|127)\.(?:\d{1,3}\.){2}\d{1,3}"
            r"|(?<![\d.])172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}"
            r"|(?<![\d.])192\.168(?:\.\d{1,3}){2}"
            r"|(?<![\d.])(?:192\.0\.2|198\.51\.100|203\.0\.113)"
            r"\.\d{1,3}(?![\d.])",
            "private or RFC-reserved documentation address",
        ),
        _rule(
            "*",
            "government_id",
            r"(?<!\d)000-00-0000(?!\d)",
            "reserved synthetic identifier used in documentation",
        ),
        _rule(
            "tests/fixtures/secret_scan_canary.txt",
            "secret",
            r"OPENMED_SECRET_SCAN_CANARY_[A-Z0-9]{32}",
            "committed scanner canary; CI copies it to an unallowlisted path",
        ),
    ),
)


def load_allowlist(path: str | Path | None = None) -> PrivacyAllowlist:
    """Load a versioned JSON extension to the built-in allowlist.

    The built-in rules remain active.  An extension must contain an integer
    ``version`` matching :data:`ALLOWLIST_VERSION` and an ``entries`` list.
    Each entry requires a path glob, category, regular expression, and reason;
    broad path-only exemptions are intentionally rejected.
    """

    if path is None:
        return DEFAULT_ALLOWLIST

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrivacyScanError("privacy allowlist could not be loaded") from exc

    if not isinstance(payload, Mapping):
        raise PrivacyScanError("privacy allowlist must be an object")
    version = payload.get("version")
    if isinstance(version, bool) or version != ALLOWLIST_VERSION:
        raise PrivacyScanError("privacy allowlist version is unsupported")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise PrivacyScanError("privacy allowlist entries must be a list")

    extension: list[AllowlistRule] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            raise PrivacyScanError(f"privacy allowlist entry {index} is invalid")
        path_glob = entry.get("path")
        category = entry.get("category")
        pattern = entry.get("pattern")
        reason = entry.get("reason")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (path_glob, category, pattern, reason)
        ):
            raise PrivacyScanError(f"privacy allowlist entry {index} is incomplete")
        try:
            compiled = _compile(pattern)
        except re.error as exc:
            raise PrivacyScanError(
                f"privacy allowlist entry {index} has invalid pattern"
            ) from exc
        extension.append(
            AllowlistRule(path_glob.strip(), category.strip(), compiled, reason.strip())
        )

    return PrivacyAllowlist(
        version=ALLOWLIST_VERSION,
        rules=DEFAULT_ALLOWLIST.rules + tuple(extension),
    )


def _path_matches(path: str, rule_path: str) -> bool:
    normalized = path.replace("\\", "/")
    return fnmatchcase(normalized, rule_path.replace("\\", "/"))


def _is_allowlisted(
    path: str,
    category: str,
    text: str,
    start: int,
    end: int,
    allowlist: PrivacyAllowlist,
) -> bool:
    for rule in allowlist.rules:
        if rule.category not in {"*", category} or not _path_matches(path, rule.path):
            continue
        if any(
            match.start() <= start and end <= match.end()
            for match in rule.pattern.finditer(text)
        ):
            return True
    return False


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _context_category(key: str) -> str:
    normalized = re.sub(r"[^a-z]", "", key.casefold())
    if normalized in {
        "rawtext",
        "sourcetext",
        "documenttext",
        "patienttext",
        "prompt",
        "completion",
    }:
        return "raw_text"
    if normalized in {"patientname", "fullname"}:
        return "name"
    if normalized in {"dateofbirth", "dob"}:
        return "date_of_birth"
    if normalized in {"email"}:
        return "email"
    if normalized in {"phone"}:
        return "phone"
    if normalized in {"address"}:
        return "address"
    if normalized in {"patientid", "medicalrecordnumber", "mrn", "encounterid"}:
        return "record_identifier"
    return "sensitive_field"


def scan_text(
    text: str,
    *,
    path: str = "<text>",
    allowlist: PrivacyAllowlist | None = None,
) -> tuple[Finding, ...]:
    """Scan UTF-8 text and return value-free findings in stable order.

    Args:
        text: Candidate file content held in memory.
        path: Repository-relative display path used in findings.
        allowlist: Optional built-in-plus-extension allowlist.

    Returns:
        One value-free finding per category and source occurrence.
    """

    if not isinstance(text, str):
        raise TypeError("privacy scanner input must be text")
    active_allowlist = allowlist or DEFAULT_ALLOWLIST
    findings: set[Finding] = set()

    for detector in _DETECTORS:
        for match in detector.pattern.finditer(text):
            if detector.validator is not None and not detector.validator(match):
                continue
            start, end = match.span()
            if _is_allowlisted(
                path, detector.category, text, start, end, active_allowlist
            ):
                continue
            findings.add(Finding(path, detector.category, _line_number(text, start)))

    for match in _SENSITIVE_FIELD_PATTERN.finditer(text):
        value = match.group("value")
        if _is_placeholder(value):
            continue
        category = _context_category(match.group("key"))
        start, end = match.span("value")
        if _is_allowlisted(path, category, text, start, end, active_allowlist):
            continue
        findings.add(Finding(path, category, _line_number(text, start)))

    return tuple(sorted(findings))


def _display_path(path: Path, repo_root: Path | None) -> str:
    candidate = path
    if repo_root is not None:
        root = repo_root.resolve()
        try:
            candidate = path.resolve().relative_to(root)
        except ValueError:
            candidate = path
    return candidate.as_posix()


def scan_paths(
    paths: Iterable[str | Path],
    *,
    repo_root: str | Path | None = None,
    allowlist: PrivacyAllowlist | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ScanResult:
    """Scan explicitly selected repository paths without network access.

    Binary files are recorded as skipped because this text scanner cannot
    safely interpret their content.  Oversized and unreadable candidate files
    are blocking findings so a hook cannot silently bypass inspection.
    """

    root = Path(repo_root).resolve() if repo_root is not None else None
    active_allowlist = allowlist or DEFAULT_ALLOWLIST
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    unique_paths: dict[str, Path] = {}
    for raw_path in paths:
        source = Path(raw_path)
        candidate = source if source.is_absolute() or root is None else root / source
        display = _display_path(candidate, root)
        unique_paths.setdefault(display, candidate)

    findings: list[Finding] = []
    scanned: list[str] = []
    skipped: list[str] = []
    for display, candidate in sorted(unique_paths.items()):
        if root is not None:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                findings.append(Finding(display, "unsafe_path", 0))
                continue
        if candidate.is_symlink():
            findings.append(Finding(display, "unsafe_path", 0))
            continue
        try:
            data = candidate.read_bytes()
        except OSError as exc:
            del exc
            findings.append(Finding(display, "unreadable_file", 0))
            continue
        file_findings, was_scanned, was_skipped = _scan_bytes(
            display,
            data,
            allowlist=active_allowlist,
            max_bytes=max_bytes,
        )
        findings.extend(file_findings)
        if was_scanned:
            scanned.append(display)
        if was_skipped:
            skipped.append(display)

    return ScanResult(
        findings=tuple(sorted(set(findings))),
        scanned_files=tuple(scanned),
        skipped_files=tuple(skipped),
        allowlist_version=active_allowlist.version,
    )


def _scan_bytes(
    display: str,
    data: bytes,
    *,
    allowlist: PrivacyAllowlist,
    max_bytes: int,
) -> tuple[tuple[Finding, ...], bool, bool]:
    """Scan one byte payload and return findings, scanned, and skipped flags."""

    if len(data) > max_bytes:
        return (Finding(display, "file_too_large", 0),), False, False
    if b"\x00" in data[:8192]:
        return (), False, True
    text = data.decode("utf-8", errors="replace")
    return scan_text(text, path=display, allowlist=allowlist), True, False


def _scan_commit_blobs(
    repo_root: Path,
    heads_and_paths: Iterable[tuple[str, str]],
    *,
    allowlist: PrivacyAllowlist,
    max_bytes: int,
) -> ScanResult:
    """Scan committed blobs selected from local Git objects."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    findings: list[Finding] = []
    scanned: set[str] = set()
    skipped: set[str] = set()
    for head_sha, path in sorted(set(heads_and_paths)):
        try:
            data = _git_output(repo_root, ["show", f"{head_sha}:{path}"])
        except PrivacyScanError:
            findings.append(Finding(path, "unreadable_file", 0))
            continue
        file_findings, was_scanned, was_skipped = _scan_bytes(
            path,
            data,
            allowlist=allowlist,
            max_bytes=max_bytes,
        )
        findings.extend(file_findings)
        if was_scanned:
            scanned.add(path)
        if was_skipped:
            skipped.add(path)

    return ScanResult(
        findings=tuple(sorted(set(findings))),
        scanned_files=tuple(sorted(scanned)),
        skipped_files=tuple(sorted(skipped)),
        allowlist_version=allowlist.version,
    )


def _git_output(repo_root: Path, args: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise PrivacyScanError("local git command could not be started") from exc
    if completed.returncode != 0:
        raise PrivacyScanError("local git candidate selection failed")
    return completed.stdout


def changed_paths(
    repo_root: str | Path,
    base_sha: str,
    head_sha: str,
) -> tuple[str, ...]:
    """Return added and modified paths between two locally available commits."""

    root = Path(repo_root).resolve()
    if _is_zero_object_id(head_sha) or not head_sha:
        return ()
    base = EMPTY_TREE_SHA if _is_zero_object_id(base_sha) or not base_sha else base_sha
    output = _git_output(
        root,
        [
            "diff",
            "--name-status",
            "--diff-filter=AM",
            "--no-renames",
            "-z",
            base,
            head_sha,
            "--",
        ],
    )
    parts = output.split(b"\x00")
    paths: set[str] = set()
    index = 0
    while index + 1 < len(parts):
        status = parts[index].decode("ascii", errors="replace")
        raw_path = parts[index + 1]
        index += 2
        if not raw_path or status not in {"A", "M"}:
            continue
        paths.add(raw_path.decode("utf-8", errors="surrogateescape"))
    return tuple(sorted(paths))


def parse_pre_push_updates(stream: Iterable[str]) -> tuple[PushUpdate, ...]:
    """Parse Git's four-column pre-push update stream."""

    updates: list[PushUpdate] = []
    for line in stream:
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 4:
            raise PrivacyScanError("pre-push input is malformed")
        local_ref, local_sha, remote_ref, remote_sha = fields
        if not _valid_object_id(local_sha) or not _valid_object_id(remote_sha):
            raise PrivacyScanError("pre-push object id is malformed")
        updates.append(PushUpdate(local_ref, local_sha, remote_ref, remote_sha))
    return tuple(updates)


def _valid_object_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{40,64}", value))


def _is_zero_object_id(value: str) -> bool:
    return bool(value) and not value.strip("0")


def scan_pushed_updates(
    repo_root: str | Path,
    updates: Iterable[PushUpdate],
    *,
    allowlist: PrivacyAllowlist | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ScanResult:
    """Scan the union of added/modified files in pre-push updates."""

    root = Path(repo_root).resolve()
    active_allowlist = allowlist or DEFAULT_ALLOWLIST
    heads_and_paths: list[tuple[str, str]] = []
    for update in sorted(updates, key=lambda item: (item.local_ref, item.local_sha)):
        if _is_zero_object_id(update.local_sha):
            continue
        for path in changed_paths(root, update.remote_sha, update.local_sha):
            heads_and_paths.append((update.local_sha, path))
    return _scan_commit_blobs(
        root,
        heads_and_paths,
        allowlist=active_allowlist,
        max_bytes=max_bytes,
    )


def scan_commit_ranges(
    repo_root: str | Path,
    ranges: Iterable[str],
    *,
    allowlist: PrivacyAllowlist | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ScanResult:
    """Scan the union of added/modified files in ``base..head`` ranges."""

    root = Path(repo_root).resolve()
    active_allowlist = allowlist or DEFAULT_ALLOWLIST
    heads_and_paths: list[tuple[str, str]] = []
    for value in ranges:
        base, separator, head = value.partition("..")
        if not separator or not base or not head or ".." in head:
            raise PrivacyScanError("commit range is malformed")
        for path in changed_paths(root, base, head):
            heads_and_paths.append((head, path))
    return _scan_commit_blobs(
        root,
        heads_and_paths,
        allowlist=active_allowlist,
        max_bytes=max_bytes,
    )


def format_report(result: ScanResult) -> str:
    """Render a deterministic report containing no matched values."""

    if result.passed:
        status = "passed"
    else:
        status = "failed"
    lines = [
        f"privacy scan {status}: {len(result.scanned_files)} file(s) scanned, "
        f"{len(result.findings)} finding(s)"
    ]
    for path, categories in result.files.items():
        category_summary = ", ".join(
            f"{category} ({count})" for category, count in categories.items()
        )
        lines.append(f"- {path}: {category_summary}")
    if result.skipped_files:
        lines.append(f"skipped {len(result.skipped_files)} binary file(s)")
    return "\n".join(lines)


def _json_report(result: ScanResult) -> str:
    payload = {
        "allowlist_version": result.allowlist_version,
        "categories": result.categories,
        "findings": [
            {"category": finding.category, "line": finding.line, "path": finding.path}
            for finding in result.findings
        ],
        "passed": result.passed,
        "scanned_files": list(result.scanned_files),
        "skipped_files": list(result.skipped_files),
        "version": SCANNER_VERSION,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_parser() -> argparse.ArgumentParser:
    """Build the local scanner command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="repository root containing the candidate files",
    )
    parser.add_argument(
        "--range",
        dest="ranges",
        action="append",
        help="local commit range BASE..HEAD; repeat for multiple pushes",
    )
    parser.add_argument(
        "--path",
        dest="paths",
        action="append",
        help="explicit repository-relative candidate path; repeat as needed",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        help="optional versioned JSON allowlist extension",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="maximum text candidate size (default: 2000000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a stable JSON report without matched values",
    )
    parser.add_argument(
        "hook_args",
        nargs="*",
        help="remote name and URL supplied by Git's pre-push hook",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, stdin: TextIO | None = None) -> int:
    """Run an explicit-path, range, or Git pre-push privacy scan."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_bytes <= 0:
        parser.error("--max-bytes must be positive")
    root = args.repo.resolve()
    try:
        active_allowlist = load_allowlist(args.allowlist)
        if args.paths:
            result = scan_paths(
                args.paths,
                repo_root=root,
                allowlist=active_allowlist,
                max_bytes=args.max_bytes,
            )
        elif args.ranges:
            result = scan_commit_ranges(
                root,
                args.ranges,
                allowlist=active_allowlist,
                max_bytes=args.max_bytes,
            )
        else:
            input_stream = stdin if stdin is not None else sys.stdin
            updates = parse_pre_push_updates(input_stream)
            result = scan_pushed_updates(
                root,
                updates,
                allowlist=active_allowlist,
                max_bytes=args.max_bytes,
            )
    except (OSError, PrivacyScanError, ValueError) as exc:
        del exc
        print(
            "privacy scan failed: local candidate inspection unavailable",
            file=sys.stderr,
        )
        return 1

    report = _json_report(result) if args.json else format_report(result)
    print(report, file=sys.stderr if not result.passed else sys.stdout)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
