"""Freshness metrics for the canonical OpenMed model manifest."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.model_registry import MANIFEST_PATH, load_manifest_rows


MEDIAN_AGE_TARGET_DAYS = 30
TIMESTAMP_FIELDS = (
    "released",
    "updated",
    "updated_at",
    "last_modified",
    "lastModified",
    "created_at",
    "createdAt",
)


@dataclass(frozen=True)
class FleetCheckpoint:
    """A dated model manifest entry used for freshness calculations."""

    repo_id: str
    released_on: date
    languages: tuple[str, ...]

    @property
    def released(self) -> str:
        return self.released_on.isoformat()

    def age_days(self, as_of: date) -> int:
        """Return non-negative age in days as of *as_of*."""
        return max((as_of - self.released_on).days, 0)

    def to_dict(self, *, as_of: date) -> dict[str, Any]:
        """Return a JSON-serializable checkpoint summary."""
        return {
            "repo_id": self.repo_id,
            "released": self.released,
            "languages": list(self.languages),
            "age_days": self.age_days(as_of),
        }


@dataclass(frozen=True)
class FleetFreshnessMetrics:
    """Serializable fleet freshness metrics for badge or status surfaces."""

    as_of: date
    total_model_count: int
    dated_model_count: int
    undated_model_count: int
    median_age_days: float | None
    median_age_target_days: int
    days_since_last_released_checkpoint: int | None
    days_since_last_released_language: int | None
    days_since_last_released_language_by_code: dict[str, int]
    last_released_checkpoint: FleetCheckpoint | None
    last_released_language: tuple[str, FleetCheckpoint] | None
    last_released_language_by_code: dict[str, FleetCheckpoint]

    @property
    def median_age_target_met(self) -> bool | None:
        """Whether median age is below the reference target, if known."""
        if self.median_age_days is None:
            return None
        return self.median_age_days < self.median_age_target_days

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable metrics payload."""
        return {
            "as_of": self.as_of.isoformat(),
            "total_model_count": self.total_model_count,
            "dated_model_count": self.dated_model_count,
            "undated_model_count": self.undated_model_count,
            "median_age_days": self.median_age_days,
            "median_age_target": {
                "operator": "<",
                "days": self.median_age_target_days,
                "met": self.median_age_target_met,
                "gating": False,
            },
            "days_since_last_released_checkpoint": self.days_since_last_released_checkpoint,
            "last_released_checkpoint": (
                self.last_released_checkpoint.to_dict(as_of=self.as_of)
                if self.last_released_checkpoint is not None
                else None
            ),
            "days_since_last_released_language": self.days_since_last_released_language,
            "last_released_language": _language_summary(
                self.last_released_language,
                as_of=self.as_of,
            ),
            "days_since_last_released_language_by_code": dict(
                sorted(self.days_since_last_released_language_by_code.items())
            ),
            "last_released_language_by_code": {
                language: checkpoint.to_dict(as_of=self.as_of)
                for language, checkpoint in sorted(
                    self.last_released_language_by_code.items()
                )
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize metrics to deterministic JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write metrics JSON to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Serialize metrics to a small Markdown status artifact."""
        checkpoint = self.last_released_checkpoint
        language = self.last_released_language
        lines = [
            "# Model Fleet Freshness",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| As of | `{self.as_of.isoformat()}` |",
            f"| Manifest rows | {self.total_model_count} |",
            f"| Dated rows | {self.dated_model_count} |",
            f"| Undated rows | {self.undated_model_count} |",
            f"| Fleet median age | {_format_days(self.median_age_days)} |",
            (
                "| Median age target | "
                f"< {self.median_age_target_days} days "
                f"(reference only; met: {_format_bool(self.median_age_target_met)}) |"
            ),
            (
                "| Days since last released checkpoint | "
                f"{_format_days(self.days_since_last_released_checkpoint)} |"
            ),
            (
                "| Days since last released language | "
                f"{_format_days(self.days_since_last_released_language)} |"
            ),
        ]

        if checkpoint is not None:
            lines.extend(
                [
                    "",
                    "## Latest Checkpoint",
                    "",
                    "| Field | Value |",
                    "|---|---|",
                    f"| Repository | `{checkpoint.repo_id}` |",
                    f"| Released | `{checkpoint.released}` |",
                    f"| Languages | `{', '.join(checkpoint.languages) or 'none'}` |",
                ]
            )

        if language is not None:
            language_code, language_checkpoint = language
            lines.extend(
                [
                    "",
                    "## Latest Language Release",
                    "",
                    "| Field | Value |",
                    "|---|---|",
                    f"| Language | `{language_code}` |",
                    f"| Repository | `{language_checkpoint.repo_id}` |",
                    f"| Released | `{language_checkpoint.released}` |",
                ]
            )

        if self.last_released_language_by_code:
            lines.extend(
                [
                    "",
                    "## Language Freshness",
                    "",
                    "| Language | Days Since Release | Repository | Released |",
                    "|---|---:|---|---|",
                ]
            )
            for language_code, language_checkpoint in sorted(
                self.last_released_language_by_code.items()
            ):
                lines.append(
                    "| "
                    f"`{language_code}` | "
                    f"{language_checkpoint.age_days(self.as_of)} | "
                    f"`{language_checkpoint.repo_id}` | "
                    f"`{language_checkpoint.released}` |"
                )

        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write metrics Markdown to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def compute_fleet_freshness(
    rows: Iterable[Mapping[str, Any]],
    *,
    as_of: date | datetime | str | None = None,
    median_age_target_days: int = MEDIAN_AGE_TARGET_DAYS,
) -> FleetFreshnessMetrics:
    """Compute model fleet freshness metrics from manifest rows."""
    as_of_date = _parse_as_of(as_of)
    materialized_rows = list(rows)
    checkpoints = _dated_checkpoints(materialized_rows)
    ages = [checkpoint.age_days(as_of_date) for checkpoint in checkpoints]
    median_age = float(statistics.median(ages)) if ages else None
    latest_checkpoint = _latest_checkpoint(checkpoints)
    latest_by_language = _latest_checkpoint_by_language(checkpoints)
    latest_language = _latest_language_release(latest_by_language)

    return FleetFreshnessMetrics(
        as_of=as_of_date,
        total_model_count=len(materialized_rows),
        dated_model_count=len(checkpoints),
        undated_model_count=len(materialized_rows) - len(checkpoints),
        median_age_days=median_age,
        median_age_target_days=median_age_target_days,
        days_since_last_released_checkpoint=(
            latest_checkpoint.age_days(as_of_date)
            if latest_checkpoint is not None
            else None
        ),
        days_since_last_released_language=(
            latest_language[1].age_days(as_of_date)
            if latest_language is not None
            else None
        ),
        days_since_last_released_language_by_code={
            language: checkpoint.age_days(as_of_date)
            for language, checkpoint in latest_by_language.items()
        },
        last_released_checkpoint=latest_checkpoint,
        last_released_language=latest_language,
        last_released_language_by_code=latest_by_language,
    )


def compute_fleet_freshness_from_manifest(
    manifest_path: str | Path = MANIFEST_PATH,
    *,
    as_of: date | datetime | str | None = None,
    median_age_target_days: int = MEDIAN_AGE_TARGET_DAYS,
) -> FleetFreshnessMetrics:
    """Load a JSONL manifest and compute fleet freshness metrics."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Model manifest not found: {path}")
    return compute_fleet_freshness(
        load_manifest_rows(path),
        as_of=as_of,
        median_age_target_days=median_age_target_days,
    )


def write_fleet_freshness_artifact(
    metrics: FleetFreshnessMetrics,
    output_path: str | Path,
    *,
    output_format: str = "json",
) -> Path:
    """Write metrics as either ``json`` or ``markdown``."""
    if output_format == "json":
        return metrics.write_json(output_path)
    if output_format == "markdown":
        return metrics.write_markdown(output_path)
    raise ValueError("output_format must be 'json' or 'markdown'")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the standalone fleet freshness CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute freshness metrics from the canonical model manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to the manifest JSONL file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional path to write the metrics artifact.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Artifact format to print or write.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Reference date in YYYY-MM-DD format. Defaults to today in UTC.",
    )
    parser.add_argument(
        "--target-days",
        type=int,
        default=MEDIAN_AGE_TARGET_DAYS,
        help="Reference median-age target in days.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Compute and print or write fleet freshness metrics."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    metrics = compute_fleet_freshness_from_manifest(
        args.manifest,
        as_of=args.as_of,
        median_age_target_days=args.target_days,
    )
    if args.output:
        write_fleet_freshness_artifact(
            metrics,
            args.output,
            output_format=args.format,
        )
        print(f"Wrote fleet freshness metrics to {args.output}")
    else:
        if args.format == "json":
            print(metrics.to_json())
        else:
            print(metrics.to_markdown(), end="")
    return 0


def _dated_checkpoints(rows: Iterable[Mapping[str, Any]]) -> list[FleetCheckpoint]:
    checkpoints: list[FleetCheckpoint] = []
    for row in rows:
        released_on = _row_timestamp(row)
        if released_on is None:
            continue
        checkpoints.append(
            FleetCheckpoint(
                repo_id=str(row.get("repo_id") or "unknown"),
                released_on=released_on,
                languages=_normalise_languages(row.get("languages")),
            )
        )
    return checkpoints


def _latest_checkpoint(
    checkpoints: Iterable[FleetCheckpoint],
) -> FleetCheckpoint | None:
    return max(
        checkpoints,
        key=lambda checkpoint: (checkpoint.released_on, checkpoint.repo_id),
        default=None,
    )


def _latest_checkpoint_by_language(
    checkpoints: Iterable[FleetCheckpoint],
) -> dict[str, FleetCheckpoint]:
    latest: dict[str, FleetCheckpoint] = {}
    for checkpoint in checkpoints:
        for language in checkpoint.languages:
            current = latest.get(language)
            if current is None or (
                checkpoint.released_on,
                checkpoint.repo_id,
            ) > (
                current.released_on,
                current.repo_id,
            ):
                latest[language] = checkpoint
    return latest


def _latest_language_release(
    latest_by_language: Mapping[str, FleetCheckpoint],
) -> tuple[str, FleetCheckpoint] | None:
    if not latest_by_language:
        return None
    return max(
        latest_by_language.items(),
        key=lambda item: (item[1].released_on, item[1].repo_id, item[0]),
    )


def _row_timestamp(row: Mapping[str, Any]) -> date | None:
    for field in TIMESTAMP_FIELDS:
        parsed = _parse_date(row.get(field))
        if parsed is not None:
            return parsed
    return None


def _parse_as_of(value: date | datetime | str | None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    parsed = _parse_date(value)
    if parsed is None:
        raise ValueError(f"Invalid as_of date: {value!r}")
    return parsed


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).date()
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    if len(text) >= 10:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    return None


def _normalise_languages(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_languages = value.replace(";", ",").split(",")
    else:
        try:
            raw_languages = list(value)
        except TypeError:
            raw_languages = [value]
    return tuple(
        sorted(
            {
                str(language).strip().lower()
                for language in raw_languages
                if str(language).strip()
            }
        )
    )


def _language_summary(
    value: tuple[str, FleetCheckpoint] | None,
    *,
    as_of: date,
) -> dict[str, Any] | None:
    if value is None:
        return None
    language, checkpoint = value
    payload = checkpoint.to_dict(as_of=as_of)
    payload["language"] = language
    return payload


def _format_days(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value)} days"
    return f"{value:g} days"


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


__all__ = [
    "FleetCheckpoint",
    "FleetFreshnessMetrics",
    "MEDIAN_AGE_TARGET_DAYS",
    "TIMESTAMP_FIELDS",
    "build_arg_parser",
    "compute_fleet_freshness",
    "compute_fleet_freshness_from_manifest",
    "main",
    "write_fleet_freshness_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
