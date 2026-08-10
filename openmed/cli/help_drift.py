"""Deterministic, value-free comparisons for CLI help surfaces.

The checker consumes synthetic command records rather than invoking commands or
contacting a service.  Normalization keeps only the parts of a help surface
that describe its shape: command paths, option flags, requiredness, value
arity, and repeatability.  Defaults, help text, choices, metavars, and other
option values are deliberately discarded before a signature or report is
created.

The module also provides a small JSON-file CLI for release automation::

    python -m openmed.cli.help_drift baseline.json candidate.json

The input format and exit categories are documented in
``docs/cli/help-drift.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "openmed.cli.help_drift.v1"


class DriftCategory(str, Enum):
    """Stable categories returned by a help-surface comparison."""

    CLEAN = "clean"
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    MIXED = "mixed"


# Small, distinct exit values make the result useful to CI while retaining a
# zero exit for an equivalent surface.  Invalid input is separate from drift.
EXIT_CLEAN = 0
EXIT_ADDED = 1
EXIT_REMOVED = 2
EXIT_CHANGED = 3
EXIT_MIXED = 4
EXIT_INVALID = 5
EXIT_OK = EXIT_CLEAN

EXIT_CODES: Mapping[DriftCategory, int] = {
    DriftCategory.CLEAN: EXIT_CLEAN,
    DriftCategory.ADDED: EXIT_ADDED,
    DriftCategory.REMOVED: EXIT_REMOVED,
    DriftCategory.CHANGED: EXIT_CHANGED,
    DriftCategory.MIXED: EXIT_MIXED,
}

# These aliases make the public terminology explicit for callers that refer to
# the result as an exit category rather than a drift category.
ExitCategory = DriftCategory
HelpDriftCategory = DriftCategory


class HelpDriftError(ValueError):
    """Raised when a synthetic help record cannot be normalized safely."""


@dataclass(frozen=True)
class OptionSignature:
    """The value-free shape of one CLI option.

    ``flags`` are sorted aliases.  ``arity`` is one of ``none``, ``one``,
    ``optional``, ``zero_or_more``, ``one_or_more``, or ``fixed:N``.  The
    signature never stores defaults, choices, help text, or user-provided
    option values.
    """

    flags: tuple[str, ...]
    required: bool
    arity: str
    repeatable: bool

    @property
    def identifier(self) -> str:
        """Return the stable comparison key for this option."""

        long_flags = tuple(flag for flag in self.flags if flag.startswith("--"))
        return long_flags[0] if long_flags else self.flags[0]

    @property
    def takes_value(self) -> bool:
        """Return whether the option consumes a value."""

        return self.arity != "none"

    def to_dict(self) -> dict[str, Any]:
        """Return the privacy-safe JSON representation."""

        return {
            "flags": list(self.flags),
            "required": self.required,
            "arity": self.arity,
            "repeatable": self.repeatable,
        }


@dataclass(frozen=True)
class CommandSignature:
    """The normalized help surface for one command path."""

    command: tuple[str, ...]
    options: tuple[OptionSignature, ...]

    @property
    def path(self) -> tuple[str, ...]:
        """Alias for the command path."""

        return self.command

    def to_dict(self) -> dict[str, Any]:
        """Return the privacy-safe JSON representation."""

        return {
            "command": list(self.command),
            "options": [option.to_dict() for option in self.options],
        }


@dataclass(frozen=True)
class HelpSurfaceSignature:
    """A deterministic, serializable signature for a set of help records."""

    commands: tuple[CommandSignature, ...]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready signature."""

        return {
            "schema_version": self.schema_version,
            "commands": [command.to_dict() for command in self.commands],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the signature with stable key and record ordering."""

        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True
        )

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical signature JSON."""

        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OptionChange:
    """One added, removed, or changed option in a command path."""

    command: tuple[str, ...]
    option: str
    before: OptionSignature | None
    after: OptionSignature | None

    @property
    def category(self) -> DriftCategory:
        """Return the deterministic category for this change."""

        if self.before is None:
            return DriftCategory.ADDED
        if self.after is None:
            return DriftCategory.REMOVED
        return DriftCategory.CHANGED

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free JSON representation of the change."""

        return {
            "command": list(self.command),
            "option": self.option,
            "category": self.category.value,
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
        }


@dataclass(frozen=True)
class HelpDriftReport:
    """Structured comparison output suitable for deterministic CI checks."""

    baseline: HelpSurfaceSignature
    candidate: HelpSurfaceSignature
    added: tuple[OptionChange, ...]
    removed: tuple[OptionChange, ...]
    changed: tuple[OptionChange, ...]
    added_commands: tuple[tuple[str, ...], ...]
    removed_commands: tuple[tuple[str, ...], ...]
    category: DriftCategory
    exit_code: int

    @property
    def exit_category(self) -> DriftCategory:
        """Return the category represented by :attr:`exit_code`."""

        return self.category

    @property
    def is_clean(self) -> bool:
        """Return whether the candidate surface matches the baseline."""

        return self.category is DriftCategory.CLEAN

    @property
    def has_drift(self) -> bool:
        """Return whether any command or option changed."""

        return not self.is_clean

    @property
    def added_options(self) -> tuple[OptionChange, ...]:
        """Return added options (an explicit alias for :attr:`added`)."""

        return self.added

    @property
    def removed_options(self) -> tuple[OptionChange, ...]:
        """Return removed options (an explicit alias for :attr:`removed`)."""

        return self.removed

    @property
    def changed_options(self) -> tuple[OptionChange, ...]:
        """Return changed options (an explicit alias for :attr:`changed`)."""

        return self.changed

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report without raw input values."""

        return {
            "schema_version": SCHEMA_VERSION,
            "category": self.category.value,
            "exit_category": self.category.value,
            "exit_code": self.exit_code,
            "baseline_digest": self.baseline.digest,
            "candidate_digest": self.candidate.digest,
            "added_commands": [list(command) for command in self.added_commands],
            "removed_commands": [list(command) for command in self.removed_commands],
            "added": [change.to_dict() for change in self.added],
            "removed": [change.to_dict() for change in self.removed],
            "changed": [change.to_dict() for change in self.changed],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report with stable ordering."""

        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True
        )


def normalize_help_records(
    records: Any,
) -> HelpSurfaceSignature:
    """Normalize synthetic command help records into a stable signature.

    ``records`` may be a sequence of command mappings, a mapping containing a
    ``commands`` or ``records`` sequence, or one command mapping.  Each command
    mapping accepts ``command`` (or ``command_path``/``path``/``name``) and
    ``options``.  An option accepts ``flags`` (or ``option_strings``/``names``) plus optional
    ``required``, ``nargs``, ``action``, and ``repeatable`` fields.

    Values that describe examples or runtime data are ignored.  Invalid shape
    errors intentionally contain only structural context and never echo input
    values.
    """

    if isinstance(records, HelpSurfaceSignature):
        return records

    raw_records = _record_sequence(records)
    commands: list[CommandSignature] = []
    seen_commands: set[tuple[str, ...]] = set()
    for raw_record in raw_records:
        command = _normalize_command(raw_record)
        if command.command in seen_commands:
            raise HelpDriftError("duplicate command record")
        seen_commands.add(command.command)
        commands.append(command)

    commands.sort(key=lambda item: item.command)
    return HelpSurfaceSignature(commands=tuple(commands))


def build_surface_signature(records: Any) -> dict[str, Any]:
    """Return the normalized signature as a JSON-ready dictionary."""

    return normalize_help_records(records).to_dict()


def surface_digest(records: Any) -> str:
    """Return the canonical digest for synthetic help records."""

    if isinstance(records, HelpSurfaceSignature):
        return records.digest
    return normalize_help_records(records).digest


def compare_help_surfaces(baseline: Any, candidate: Any) -> HelpDriftReport:
    """Classify deterministic command and option drift between two surfaces.

    Option identity prefers the long flag (for example ``--format``) and falls
    back to the first short flag.  An alias change for the same long flag is
    therefore a changed option; a rename is an added option plus a removed
    option.  All output collections are sorted by command path and option key.
    """

    baseline_signature = _as_signature(baseline)
    candidate_signature = _as_signature(candidate)
    baseline_commands = {
        command.command: command for command in baseline_signature.commands
    }
    candidate_commands = {
        command.command: command for command in candidate_signature.commands
    }

    added_commands = tuple(sorted(set(candidate_commands) - set(baseline_commands)))
    removed_commands = tuple(sorted(set(baseline_commands) - set(candidate_commands)))

    added: list[OptionChange] = []
    removed: list[OptionChange] = []
    changed: list[OptionChange] = []

    for command_path in added_commands:
        for option in candidate_commands[command_path].options:
            added.append(
                OptionChange(
                    command=command_path,
                    option=option.identifier,
                    before=None,
                    after=option,
                )
            )

    for command_path in removed_commands:
        for option in baseline_commands[command_path].options:
            removed.append(
                OptionChange(
                    command=command_path,
                    option=option.identifier,
                    before=option,
                    after=None,
                )
            )

    for command_path in sorted(set(baseline_commands) & set(candidate_commands)):
        baseline_options = {
            option.identifier: option
            for option in baseline_commands[command_path].options
        }
        candidate_options = {
            option.identifier: option
            for option in candidate_commands[command_path].options
        }

        for option_name in sorted(set(candidate_options) - set(baseline_options)):
            added.append(
                OptionChange(
                    command=command_path,
                    option=option_name,
                    before=None,
                    after=candidate_options[option_name],
                )
            )
        for option_name in sorted(set(baseline_options) - set(candidate_options)):
            removed.append(
                OptionChange(
                    command=command_path,
                    option=option_name,
                    before=baseline_options[option_name],
                    after=None,
                )
            )
        for option_name in sorted(set(baseline_options) & set(candidate_options)):
            before = baseline_options[option_name]
            after = candidate_options[option_name]
            if before != after:
                changed.append(
                    OptionChange(
                        command=command_path,
                        option=option_name,
                        before=before,
                        after=after,
                    )
                )

    added.sort(key=_change_sort_key)
    removed.sort(key=_change_sort_key)
    changed.sort(key=_change_sort_key)
    category = _classify(
        has_added=bool(added or added_commands),
        has_removed=bool(removed or removed_commands),
        has_changed=bool(changed),
    )
    return HelpDriftReport(
        baseline=baseline_signature,
        candidate=candidate_signature,
        added=tuple(added),
        removed=tuple(removed),
        changed=tuple(changed),
        added_commands=added_commands,
        removed_commands=removed_commands,
        category=category,
        exit_code=EXIT_CODES[category],
    )


def classify_help_drift(baseline: Any, candidate: Any) -> HelpDriftReport:
    """Alias for :func:`compare_help_surfaces` used by CI integrations."""

    return compare_help_surfaces(baseline, candidate)


def diff_help_surfaces(baseline: Any, candidate: Any) -> HelpDriftReport:
    """Alias for :func:`compare_help_surfaces`."""

    return compare_help_surfaces(baseline, candidate)


def _as_signature(value: Any) -> HelpSurfaceSignature:
    if isinstance(value, HelpSurfaceSignature):
        return value
    return normalize_help_records(value)


def _record_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        if "commands" in value:
            value = value["commands"]
        elif "records" in value:
            value = value["records"]
        elif any(key in value for key in ("command", "command_path", "path", "name")):
            value = (value,)
        else:
            raise HelpDriftError("help surface must contain command records")

    if isinstance(value, (str, bytes, bytearray)):
        raise HelpDriftError("help surface records must be a sequence")
    if not isinstance(value, Iterable):
        raise HelpDriftError("help surface records must be a sequence")

    normalized: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise HelpDriftError("each help record must be an object")
        normalized.append(item)
    return tuple(normalized)


def _normalize_command(record: Mapping[str, Any]) -> CommandSignature:
    raw_command = next(
        (
            record[key]
            for key in ("command", "command_path", "path", "name")
            if key in record
        ),
        None,
    )
    command = _normalize_command_path(raw_command)
    raw_options = record.get("options", record.get("arguments", ()))
    options: list[OptionSignature] = []
    seen_options: set[str] = set()
    for raw_option in _option_sequence(raw_options):
        option = _normalize_option(raw_option)
        if option.identifier in seen_options:
            raise HelpDriftError("duplicate option in command record")
        seen_options.add(option.identifier)
        options.append(option)
    options.sort(key=lambda item: (item.identifier, item.flags))
    return CommandSignature(command=command, options=tuple(options))


def _normalize_command_path(raw_command: Any) -> tuple[str, ...]:
    if isinstance(raw_command, str):
        parts = tuple(raw_command.split())
    elif isinstance(raw_command, Sequence) and not isinstance(
        raw_command, (bytes, bytearray)
    ):
        parts = tuple(raw_command)
    else:
        raise HelpDriftError("command path must be text or a sequence")

    if not parts or any(not isinstance(part, str) for part in parts):
        raise HelpDriftError("command path must contain one or more names")
    cleaned = tuple(part.strip() for part in parts)
    if any(
        not part or any(character.isspace() for character in part) for part in cleaned
    ):
        raise HelpDriftError("command path contains an empty or invalid name")
    return cleaned


def _option_sequence(raw_options: Any) -> tuple[Any, ...]:
    if raw_options is None:
        return ()
    if isinstance(raw_options, Mapping):
        items: list[Any] = []
        for flag in sorted(raw_options, key=_stable_text):
            metadata = raw_options[flag]
            if isinstance(metadata, Mapping):
                option = dict(metadata)
                if not any(
                    key in option
                    for key in (
                        "flags",
                        "option_strings",
                        "names",
                        "flag",
                        "option",
                        "name",
                    )
                ):
                    option["flags"] = flag
                items.append(option)
            else:
                items.append({"flags": flag})
        return tuple(items)
    if isinstance(raw_options, (str, bytes, bytearray)):
        raise HelpDriftError("options must be a sequence or mapping")
    if not isinstance(raw_options, Iterable):
        raise HelpDriftError("options must be a sequence or mapping")
    return tuple(raw_options)


def _normalize_option(raw_option: Any) -> OptionSignature:
    if isinstance(raw_option, str):
        option: Mapping[str, Any] = {"flags": raw_option}
    elif isinstance(raw_option, Mapping):
        option = raw_option
    else:
        raise HelpDriftError("each option must be text or an object")

    raw_flags = next(
        (option[key] for key in ("flags", "option_strings", "names") if key in option),
        None,
    )
    if raw_flags is None:
        raw_flags = next(
            (
                option[key]
                for key in ("flag", "option", "option_string", "name")
                if key in option
            ),
            None,
        )
    flags = _normalize_flags(raw_flags)

    required = _boolean_field(option, "required", default=False)
    repeatable = _boolean_field(option, "repeatable", default=False)
    action = option.get("action")
    if action is not None and not isinstance(action, str):
        raise HelpDriftError("option field action must be text")
    if not repeatable and isinstance(action, str):
        repeatable = action in {"append", "extend", "count"}
    arity = _normalize_arity(option, action=action)
    return OptionSignature(
        flags=flags,
        required=required,
        arity=arity,
        repeatable=repeatable,
    )


def _normalize_flags(raw_flags: Any) -> tuple[str, ...]:
    if isinstance(raw_flags, str):
        values = (raw_flags,)
    elif isinstance(raw_flags, Sequence) and not isinstance(
        raw_flags, (bytes, bytearray)
    ):
        values = tuple(raw_flags)
    else:
        raise HelpDriftError("option flags must be text or a sequence")

    normalized: set[str] = set()
    for raw_flag in values:
        if not isinstance(raw_flag, str):
            raise HelpDriftError("option flags must contain text")
        flag = raw_flag.strip().split("=", 1)[0]
        if (
            len(flag) < 2
            or not flag.startswith("-")
            or flag == "--"
            or any(character.isspace() for character in flag)
            or any(ord(character) < 32 for character in flag)
        ):
            raise HelpDriftError("option flags must use option syntax")
        normalized.add(flag)
    if not normalized:
        raise HelpDriftError("each option must define one or more flags")
    return tuple(sorted(normalized, key=_stable_text))


def _boolean_field(option: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = option.get(key, default)
    if not isinstance(value, bool):
        raise HelpDriftError(f"option field {key!r} must be boolean")
    return value


def _normalize_arity(option: Mapping[str, Any], *, action: Any) -> str:
    raw_nargs = option.get("nargs")
    if raw_nargs is None:
        takes_value = option.get("takes_value")
        if takes_value is not None and not isinstance(takes_value, bool):
            raise HelpDriftError("option field takes_value must be boolean")
        if takes_value is False or (
            isinstance(action, str) and action in {"store_true", "store_false", "count"}
        ):
            return "none"
        return "one"

    if isinstance(raw_nargs, bool):
        raise HelpDriftError("option field nargs has an invalid shape")
    if isinstance(raw_nargs, int):
        if raw_nargs < 0:
            raise HelpDriftError("option field nargs has an invalid shape")
        return "none" if raw_nargs == 0 else _fixed_arity(raw_nargs)
    if not isinstance(raw_nargs, str):
        raise HelpDriftError("option field nargs has an invalid shape")

    normalized = raw_nargs.strip().casefold()
    aliases = {
        "0": "none",
        "1": "one",
        "?": "optional",
        "*": "zero_or_more",
        "+": "one_or_more",
        "none": "none",
        "one": "one",
        "optional": "optional",
        "zero_or_more": "zero_or_more",
        "one_or_more": "one_or_more",
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized.isdigit():
        return _fixed_arity(int(normalized))
    raise HelpDriftError("option field nargs has an invalid shape")


def _fixed_arity(count: int) -> str:
    if count <= 0:
        return "none"
    return f"fixed:{count}"


def _stable_text(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        return (value.casefold(), value)
    return (type(value).__name__, type(value).__name__)


def _change_sort_key(change: OptionChange) -> tuple[tuple[str, ...], str]:
    return (change.command, change.option)


def _classify(
    *, has_added: bool, has_removed: bool, has_changed: bool
) -> DriftCategory:
    categories = sum((has_added, has_removed, has_changed))
    if categories == 0:
        return DriftCategory.CLEAN
    if categories > 1:
        return DriftCategory.MIXED
    if has_added:
        return DriftCategory.ADDED
    if has_removed:
        return DriftCategory.REMOVED
    return DriftCategory.CHANGED


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HelpDriftError("could not read help surface input") from exc


def _render_text(report: HelpDriftReport) -> str:
    lines = [
        f"category: {report.category.value}",
        f"exit_code: {report.exit_code}",
        f"added: {len(report.added) + len(report.added_commands)}",
        f"removed: {len(report.removed) + len(report.removed_commands)}",
        f"changed: {len(report.changed)}",
    ]
    for command in report.added_commands:
        lines.append(f"added command: {' '.join(command)}")
    for command in report.removed_commands:
        lines.append(f"removed command: {' '.join(command)}")
    for change in (*report.added, *report.removed, *report.changed):
        lines.append(
            f"{change.category.value} option: {' '.join(change.command)} "
            f"{change.option}"
        )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the local JSON comparison CLI parser."""

    parser = argparse.ArgumentParser(
        description="Compare two deterministic CLI help-surface records."
    )
    parser.add_argument("baseline", type=Path, help="Baseline JSON records.")
    parser.add_argument("candidate", type=Path, help="Candidate JSON records.")
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Report format (default: json).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Compare two local JSON help surfaces and return a CI exit category."""

    args = build_argument_parser().parse_args(argv)
    try:
        report = compare_help_surfaces(
            _load_json(args.baseline),
            _load_json(args.candidate),
        )
    except HelpDriftError:
        print("help surface input is invalid", file=sys.stderr)
        return EXIT_INVALID

    if args.format == "json":
        print(report.to_json())
    else:
        print(_render_text(report))
    return report.exit_code


# Readable aliases for callers using the more explicit normalization wording.
normalize_help_surface = normalize_help_records
canonicalize_help_records = normalize_help_records
signature_digest = surface_digest


__all__ = [
    "SCHEMA_VERSION",
    "DriftCategory",
    "ExitCategory",
    "HelpDriftCategory",
    "EXIT_CLEAN",
    "EXIT_ADDED",
    "EXIT_REMOVED",
    "EXIT_CHANGED",
    "EXIT_MIXED",
    "EXIT_INVALID",
    "EXIT_OK",
    "EXIT_CODES",
    "HelpDriftError",
    "OptionSignature",
    "CommandSignature",
    "HelpSurfaceSignature",
    "OptionChange",
    "HelpDriftReport",
    "normalize_help_records",
    "normalize_help_surface",
    "canonicalize_help_records",
    "build_surface_signature",
    "surface_digest",
    "signature_digest",
    "compare_help_surfaces",
    "classify_help_drift",
    "diff_help_surfaces",
    "build_argument_parser",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
