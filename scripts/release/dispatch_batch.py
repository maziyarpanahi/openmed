#!/usr/bin/env python3
"""Plan and execute queued OpenMed model release batches."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - covered by CLI error path
    yaml = None  # type: ignore[assignment]


DEFAULT_QUEUE = Path("recipes/queue.yaml")
DEFAULT_OUTPUT_ROOT = Path("release-artifacts")
DEFAULT_MANIFEST = Path("models.jsonl")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")
QUANTIZED_EDGE_FORMATS = {"mlx-4bit", "mlx-8bit", "coreml"}
SUPPORTED_FORMATS = {"mlx-fp", *QUANTIZED_EDGE_FORMATS}
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class BatchDispatchError(RuntimeError):
    """Raised when the release queue cannot be planned or executed."""


@dataclass(frozen=True)
class QueueItem:
    """One scheduled model release unit."""

    id: str
    model_id: str
    weekday: str
    theme: str
    formats: tuple[str, ...]
    publish: bool = True
    depends_on_green_parent: tuple[str, ...] = field(default_factory=tuple)
    gate_command: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QueueItem":
        item_id = _required_str(value, "id")
        if not _SAFE_ID_RE.fullmatch(item_id):
            raise BatchDispatchError(
                f"queue item id {item_id!r} must use lowercase letters, digits, '.', '_', or '-'"
            )
        formats = tuple(
            _normalize_format(format_name)
            for format_name in _required_list(value, "formats")
        )
        if not formats:
            raise BatchDispatchError(
                f"queue item {item_id!r} must list at least one format"
            )

        return cls(
            id=item_id,
            model_id=_required_str(value, "model_id"),
            weekday=_normalize_weekday(_required_str(value, "weekday")),
            theme=_required_str(value, "theme"),
            formats=formats,
            publish=_optional_bool(value, "publish", default=True),
            depends_on_green_parent=_optional_str_list(
                value,
                "depends_on_green_parent",
            ),
            gate_command=_optional_str_list(
                value,
                "gate_command",
            ),
        )

    def matrix_entry(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "weekday": self.weekday,
            "theme": self.theme,
            "formats": list(self.formats),
            "publish": self.publish,
            "depends_on_green_parent": list(self.depends_on_green_parent),
            "gate_command": list(self.gate_command),
        }


@dataclass(frozen=True)
class CommandRecord:
    """A command planned or executed for a queue item."""

    item_id: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class ItemResult:
    """Execution result for one queue item."""

    item_id: str
    ok: bool
    commands: tuple[CommandRecord, ...]
    error: str | None = None


CommandRunner = Callable[[list[str], dict[str, str] | None], None]


def load_queue(path: str | Path = DEFAULT_QUEUE) -> list[QueueItem]:
    """Load and validate release queue items from YAML."""

    if yaml is None:
        raise BatchDispatchError("PyYAML is required to read the release queue")

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise BatchDispatchError(f"could not read release queue {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise BatchDispatchError(
            f"invalid release queue YAML in {path}: {exc}"
        ) from exc

    if not isinstance(data, Mapping):
        raise BatchDispatchError("release queue must be a YAML mapping")
    if data.get("version") != 1:
        raise BatchDispatchError("release queue version must be 1")

    weekly_themes = data.get("weekly_themes")
    if not isinstance(weekly_themes, Mapping):
        raise BatchDispatchError("release queue weekly_themes must be a mapping")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise BatchDispatchError("release queue items must be a non-empty list")

    items: list[QueueItem] = []
    for position, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            raise BatchDispatchError(
                f"release queue item at index {position} must be a mapping"
            )
        items.append(QueueItem.from_mapping(raw_item))
    validate_queue(items, weekly_themes=weekly_themes)
    return items


def validate_queue(
    items: Iterable[QueueItem],
    *,
    weekly_themes: Mapping[str, Any] | None = None,
) -> None:
    """Validate uniqueness and parent ordering in the release queue."""

    by_id: dict[str, QueueItem] = {}
    for item in items:
        if item.id in by_id:
            raise BatchDispatchError(f"duplicate queue item id: {item.id}")
        if not item.formats or set(item.formats) - SUPPORTED_FORMATS:
            raise BatchDispatchError(
                f"queue item {item.id!r} contains unsupported release formats"
            )
        by_id[item.id] = item

    if weekly_themes is not None:
        normalized_themes: dict[str, str] = {}
        for day, theme in weekly_themes.items():
            if not isinstance(day, str) or not isinstance(theme, str) or not theme:
                raise BatchDispatchError(
                    "release queue weekly_themes keys and values must be non-empty strings"
                )
            normalized_themes[_normalize_weekday(day)] = theme
        missing_days = sorted(set(WEEKDAYS) - set(normalized_themes))
        if missing_days:
            raise BatchDispatchError(
                "release queue weekly_themes missing weekdays: "
                + ", ".join(missing_days)
            )
        for item in by_id.values():
            expected_theme = normalized_themes[item.weekday]
            if item.theme != expected_theme:
                raise BatchDispatchError(
                    f"queue item {item.id!r} theme {item.theme!r} does not match "
                    f"{item.weekday!r} theme {expected_theme!r}"
                )

    for item in by_id.values():
        is_edge_item = bool(set(item.formats) & QUANTIZED_EDGE_FORMATS)
        if is_edge_item and not item.depends_on_green_parent:
            raise BatchDispatchError(
                f"quantized edge item {item.id!r} must declare depends_on_green_parent"
            )
        item_day = WEEKDAYS.index(item.weekday)
        for parent_id in item.depends_on_green_parent:
            parent = by_id.get(parent_id)
            if parent is None:
                raise BatchDispatchError(
                    f"queue item {item.id!r} depends on missing parent {parent_id!r}"
                )
            parent_day = WEEKDAYS.index(parent.weekday)
            if parent_day >= item_day:
                raise BatchDispatchError(
                    f"queue item {item.id!r} must trail parent {parent_id!r} by at least one day"
                )
            if not parent.publish:
                raise BatchDispatchError(
                    f"queue item {item.id!r} depends on unpublished parent {parent_id!r}"
                )
            if parent.model_id != item.model_id:
                raise BatchDispatchError(
                    f"queue item {item.id!r} and parent {parent_id!r} must use the same model_id"
                )
            if set(parent.formats) & QUANTIZED_EDGE_FORMATS:
                raise BatchDispatchError(
                    f"queue item {item.id!r} parent {parent_id!r} must be a non-edge artifact"
                )


def select_items(
    items: Iterable[QueueItem],
    *,
    weekday: str | None = None,
    theme: str | None = None,
    include_all: bool = False,
) -> list[QueueItem]:
    """Select the queue slice for a scheduled run."""

    selected = list(items)
    if include_all:
        return selected

    if weekday is None and theme is None:
        weekday = datetime.now(timezone.utc).strftime("%A").lower()

    if weekday is not None:
        normalized_weekday = _normalize_weekday(weekday)
        selected = [item for item in selected if item.weekday == normalized_weekday]

    if theme is not None:
        selected = [item for item in selected if item.theme == theme]

    return selected


def build_matrix(items: Iterable[QueueItem]) -> dict[str, list[dict[str, Any]]]:
    """Return a GitHub Actions matrix object for queue items."""

    return {"include": [item.matrix_entry() for item in items]}


def run_batch(
    items: Iterable[QueueItem],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    python_executable: str = sys.executable,
    runner: CommandRunner | None = None,
    dry_run: bool = False,
) -> list[ItemResult]:
    """Run each item and continue collecting per-model failures."""

    results: list[ItemResult] = []
    for item in items:
        results.append(
            run_item(
                item,
                output_root=output_root,
                manifest_path=manifest_path,
                python_executable=python_executable,
                runner=runner,
                dry_run=dry_run,
            )
        )
    return results


def run_item(
    item: QueueItem,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    python_executable: str = sys.executable,
    runner: CommandRunner | None = None,
    dry_run: bool = False,
) -> ItemResult:
    """Run convert, optional gate, and publish commands for one queue item."""

    runner = runner or _run_command
    commands: list[CommandRecord] = []
    output_root = Path(output_root)
    manifest_path = Path(manifest_path)

    try:
        for format_name in item.formats:
            artifact_dir = output_root / item.id / format_name
            artifact_dir.mkdir(parents=True, exist_ok=True)
            convert_command, publish_artifact_dir = _convert_command(
                item,
                format_name=format_name,
                artifact_dir=artifact_dir,
                python_executable=python_executable,
            )
            _record_and_run(
                item.id,
                convert_command,
                commands,
                runner,
                dry_run=dry_run,
                expose_token=False,
            )

            if item.gate_command:
                _record_and_run(
                    item.id,
                    list(item.gate_command),
                    commands,
                    runner,
                    dry_run=dry_run,
                    expose_token=False,
                )

            if item.publish:
                publish_command = [
                    python_executable,
                    "-m",
                    "openmed.core.hf_publish",
                    "--model",
                    item.model_id,
                    "--artifact-dir",
                    str(publish_artifact_dir),
                    "--format",
                    format_name,
                    "--manifest",
                    str(manifest_path),
                ]
                _record_and_run(
                    item.id,
                    publish_command,
                    commands,
                    runner,
                    dry_run=dry_run,
                    expose_token=True,
                )
    except Exception as exc:
        return ItemResult(
            item_id=item.id,
            ok=False,
            commands=tuple(commands),
            error=str(exc),
        )

    return ItemResult(item_id=item.id, ok=True, commands=tuple(commands))


def parse_item_json(value: str) -> QueueItem:
    """Parse one matrix item JSON value from the workflow environment."""

    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise BatchDispatchError(f"invalid queue item JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise BatchDispatchError("queue item JSON must be an object")
    if "formats" not in data and "format" in data:
        data["formats"] = [data["format"]]
    return QueueItem.from_mapping(data)


def write_github_outputs(path: str | Path, outputs: dict[str, str]) -> None:
    """Append step outputs for GitHub Actions."""

    with Path(path).open("a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            handle.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.command == "plan":
        items = select_items(
            load_queue(args.queue),
            weekday=args.weekday,
            theme=args.theme,
            include_all=args.all,
        )
        matrix = json.dumps(build_matrix(items), separators=(",", ":"))
        outputs = {
            "matrix": matrix,
            "count": str(len(items)),
            "ids": ",".join(item.id for item in items),
        }
        if args.github_output:
            write_github_outputs(args.github_output, outputs)
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return 0

    if args.command == "validate":
        items = load_queue(args.queue)
        print(f"validated {len(items)} queue items")
        return 0

    if args.command == "run-item":
        item_json = args.item_json or os.environ.get("OPENMED_BATCH_ITEM")
        if not item_json:
            raise BatchDispatchError(
                "run-item requires --item-json or OPENMED_BATCH_ITEM"
            )
        item = parse_item_json(item_json)
        result = run_item(
            item,
            output_root=args.output_root,
            manifest_path=args.manifest,
            python_executable=args.python,
            dry_run=args.dry_run,
        )
        _print_results([result])
        return 0 if result.ok else 1

    if args.command == "run-batch":
        items = select_items(
            load_queue(args.queue),
            weekday=args.weekday,
            theme=args.theme,
            include_all=args.all,
        )
        results = run_batch(
            items,
            output_root=args.output_root,
            manifest_path=args.manifest,
            python_executable=args.python,
            dry_run=args.dry_run,
        )
        _print_results(results)
        return 0 if all(result.ok for result in results) else 1

    raise BatchDispatchError(f"unknown command: {args.command}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Build a matrix from the release queue")
    _add_queue_selection_args(plan)
    plan.add_argument(
        "--github-output",
        default=None,
        help="Optional GitHub Actions output file",
    )

    validate = subparsers.add_parser("validate", help="Validate the release queue")
    validate.add_argument("--queue", default=str(DEFAULT_QUEUE))

    run_item_parser = subparsers.add_parser("run-item", help="Run one matrix item")
    run_item_parser.add_argument("--item-json", default=None)
    _add_run_args(run_item_parser)

    run_batch_parser = subparsers.add_parser(
        "run-batch", help="Run selected queue items"
    )
    _add_queue_selection_args(run_batch_parser)
    _add_run_args(run_batch_parser)

    return parser.parse_args(argv)


def _add_queue_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--weekday", default=None, help="Weekday queue slice to select")
    parser.add_argument("--theme", default=None, help="Theme queue slice to select")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Select every queue item instead of the current weekday",
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and record commands without executing them",
    )


def _convert_command(
    item: QueueItem,
    *,
    format_name: str,
    artifact_dir: Path,
    python_executable: str,
) -> tuple[list[str], Path]:
    if format_name in {"mlx-fp", "mlx-4bit", "mlx-8bit"}:
        command = [
            python_executable,
            "-m",
            "openmed.mlx.convert",
            "--model",
            item.model_id,
            "--output",
            str(artifact_dir),
        ]
        if format_name == "mlx-4bit":
            command.extend(["--quantize", "4"])
        if format_name == "mlx-8bit":
            command.extend(["--quantize", "8"])
        return command, artifact_dir

    if format_name == "coreml":
        package_path = artifact_dir / "OpenMedModel.mlpackage"
        command = [
            python_executable,
            "-m",
            "openmed.coreml.convert",
            "--model",
            item.model_id,
            "--output",
            str(package_path),
        ]
        return command, artifact_dir

    raise BatchDispatchError(f"unsupported format {format_name!r} for {item.id!r}")


def _record_and_run(
    item_id: str,
    command: list[str],
    commands: list[CommandRecord],
    runner: CommandRunner,
    *,
    dry_run: bool,
    expose_token: bool,
) -> None:
    commands.append(CommandRecord(item_id=item_id, command=tuple(command)))
    print("+ " + " ".join(command))
    if dry_run:
        return

    env = os.environ.copy()
    if not expose_token:
        env.pop("HF_WRITE_TOKEN", None)
    runner(command, env)


def _run_command(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def _print_results(results: Iterable[ItemResult]) -> None:
    failures = []
    for result in results:
        status = "ok" if result.ok else "failed"
        print(f"{result.item_id}: {status}")
        if result.error:
            print(f"{result.item_id}: {result.error}")
        if not result.ok:
            failures.append(result.item_id)
    if failures:
        print("failed queue items: " + ", ".join(failures), file=sys.stderr)


def _required_str(value: Mapping[str, Any], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw or raw != raw.strip() or "\n" in raw:
        raise BatchDispatchError(f"queue item field {key!r} must be a non-empty string")
    return raw


def _required_list(value: Mapping[str, Any], key: str) -> list[Any]:
    raw = value.get(key)
    if not isinstance(raw, list):
        raise BatchDispatchError(f"queue item field {key!r} must be a list")
    return raw


def _optional_bool(
    value: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    raw = value.get(key, default)
    if not isinstance(raw, bool):
        raise BatchDispatchError(f"queue item field {key!r} must be a boolean")
    return raw


def _optional_str_list(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = value.get(key, [])
    if raw is None:
        raw = []
    if not isinstance(raw, list) or any(
        not isinstance(part, str) or not part or part != part.strip() for part in raw
    ):
        raise BatchDispatchError(
            f"queue item field {key!r} must be a list of non-empty strings"
        )
    return tuple(raw)


def _normalize_weekday(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in WEEKDAYS:
        raise BatchDispatchError(f"unsupported release weekday: {value!r}")
    return normalized


def _normalize_format(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BatchDispatchError("queue item formats must contain non-empty strings")
    normalized = value.lower().replace("_", "-")
    aliases = {
        "mlx": "mlx-fp",
        "mlx-float": "mlx-fp",
        "mlx-float16": "mlx-fp",
        "mlx-int8": "mlx-8bit",
        "mlx-int4": "mlx-4bit",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_FORMATS:
        raise BatchDispatchError(f"unsupported release format: {value!r}")
    return normalized


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatchDispatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
