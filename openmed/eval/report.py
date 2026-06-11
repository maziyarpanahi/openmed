"""Benchmark report serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BenchmarkReport:
    """Serializable benchmark report emitted by the eval harness."""

    suite: str
    model_name: str
    device: str
    fixture_count: int
    metrics: Mapping[str, Any]
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report dictionary with stable keys."""
        result: dict[str, Any] = {
            "device": self.device,
            "fixture_count": self.fixture_count,
            "generated_at": self.generated_at,
            "metadata": _plain(self.metadata),
            "metrics": _plain(self.metrics),
            "model_name": self.model_name,
            "suite": self.suite,
        }
        return result

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""
        output_path = Path(path)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Serialize the report to deterministic Markdown."""
        lines = [
            f"# Benchmark Report: {self.suite}",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Suite | `{self.suite}` |",
            f"| Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| Fixtures | {self.fixture_count} |",
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | `{self.generated_at}` |")

        lines.extend(["", "## Metrics", "", "| Metric | Value |", "|---|---:|"])
        for key, value in _flatten(_plain(self.metrics)):
            lines.append(f"| `{key}` | {_format_value(value)} |")

        if self.metadata:
            lines.extend(["", "## Metadata", "", "| Key | Value |", "|---|---|"])
            for key, value in _flatten(_plain(self.metadata)):
                lines.append(f"| `{key}` | {_format_value(value)} |")

        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""
        output_path = Path(path)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key in sorted(value, key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(value[key], child_prefix))
        return rows
    return [(prefix, value)]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


__all__ = ["BenchmarkReport"]
