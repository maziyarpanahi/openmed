"""Cost-vs-cloud benchmark calculations over local performance reports.

The module deliberately consumes a committed price table rather than querying
provider APIs.  Prices are normalized to a per-character rate, while local
cost is derived from the measured character throughput and an explicit
hardware-cost model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

MILLION_CHARS = 1_000_000
SECONDS_PER_HOUR = 3_600
COST_REPORT_SCHEMA_VERSION = 1
DEFAULT_CLOUD_PRICES_PATH = Path(__file__).with_name("data") / "cloud_prices.json"

DEFAULT_HARDWARE_COST_MODEL: Mapping[str, float] = {
    "purchase_price_usd": 1_200.0,
    "amortization_hours": 8_760.0,
    "operating_cost_usd_per_hour": 0.03,
}


class PriceCitationError(ValueError):
    """Raised when a cloud-price row is not traceable to a dated source."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        message = "cloud price table citation validation failed: " + "; ".join(
            self.issues
        )
        super().__init__(message)


@dataclass(frozen=True)
class HardwareCostModel:
    """Amortized hardware and variable operating-cost assumptions."""

    purchase_price_usd: float = 0.0
    amortization_hours: float = 8_760.0
    operating_cost_usd_per_hour: float = 0.0
    fixed_cost_usd: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HardwareCostModel":
        """Build a model from common hardware-cost field spellings."""
        purchase_price = _number(
            value,
            "purchase_price_usd",
            "hardware_cost_usd",
            "acquisition_cost_usd",
            default=0.0,
        )
        fixed_cost = _number(value, "fixed_cost_usd", "upfront_cost_usd", default=0.0)
        amortization_hours = _number(
            value,
            "amortization_hours",
            "useful_life_hours",
            "lifetime_hours",
            default=8_760.0,
        )
        operating_cost = _number(
            value,
            "operating_cost_usd_per_hour",
            "hourly_operating_cost_usd",
            "cost_per_hour_usd",
            "hourly_cost_usd",
            "cost_per_hour",
            default=0.0,
        )

        if purchase_price < 0 or fixed_cost < 0 or operating_cost < 0:
            raise ValueError("hardware cost values must be non-negative")
        if amortization_hours <= 0:
            raise ValueError("hardware amortization_hours must be positive")

        return cls(
            purchase_price_usd=purchase_price,
            amortization_hours=amortization_hours,
            operating_cost_usd_per_hour=operating_cost,
            fixed_cost_usd=fixed_cost,
        )

    @property
    def capital_cost_usd(self) -> float:
        """Return the upfront capital cost used for break-even math."""
        return self.purchase_price_usd + self.fixed_cost_usd

    @property
    def amortized_capital_cost_usd_per_hour(self) -> float:
        """Return capital cost spread across the useful life."""
        return self.capital_cost_usd / self.amortization_hours

    @property
    def hourly_cost_usd(self) -> float:
        """Return amortized capital plus variable operating cost per hour."""
        return (
            self.amortized_capital_cost_usd_per_hour + self.operating_cost_usd_per_hour
        )

    def to_dict(self) -> dict[str, float]:
        """Return stable, JSON-compatible hardware assumptions."""
        return {
            "amortization_hours": self.amortization_hours,
            "amortized_capital_cost_usd_per_hour": (
                self.amortized_capital_cost_usd_per_hour
            ),
            "capital_cost_usd": self.capital_cost_usd,
            "fixed_cost_usd": self.fixed_cost_usd,
            "hourly_cost_usd": self.hourly_cost_usd,
            "operating_cost_usd_per_hour": self.operating_cost_usd_per_hour,
            "purchase_price_usd": self.purchase_price_usd,
        }


@dataclass(frozen=True)
class CloudPrice:
    """One cited cloud billing unit normalized to character counts."""

    price_id: str
    provider: str
    service: str
    price_usd_per_unit: float
    unit_characters: int
    source_url: str
    capture_date: str
    verify: Any
    billing_unit: str = "characters"

    @property
    def price_usd_per_character(self) -> float:
        """Return the normalized price for one input character."""
        return self.price_usd_per_unit / self.unit_characters

    @property
    def price_usd_per_1k_chars(self) -> float:
        """Return the normalized price for one thousand characters."""
        return self.price_usd_per_character * 1_000

    def to_dict(self) -> dict[str, Any]:
        """Return the cited row plus normalized unit prices."""
        return {
            "billing_unit": self.billing_unit,
            "capture_date": self.capture_date,
            "price_id": self.price_id,
            "price_usd_per_1k_chars": self.price_usd_per_1k_chars,
            "price_usd_per_character": self.price_usd_per_character,
            "price_usd_per_unit": self.price_usd_per_unit,
            "provider": self.provider,
            "service": self.service,
            "source_url": self.source_url,
            "unit_characters": self.unit_characters,
            "verify": self.verify,
        }


@dataclass(frozen=True)
class CloudCostComparison:
    """Local-versus-cloud cost comparison for one provider row."""

    price: CloudPrice
    cloud_cost_per_million_chars_usd: float
    local_cost_per_million_chars_usd: float
    local_variable_cost_per_million_chars_usd: float
    breakeven_volume_chars: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable comparison payload."""
        return {
            "break_even_volume_chars": self.breakeven_volume_chars,
            "breakeven_volume_chars": self.breakeven_volume_chars,
            "breakeven_volume_million_chars": (
                None
                if self.breakeven_volume_chars is None
                else self.breakeven_volume_chars / MILLION_CHARS
            ),
            "cloud_cost_per_million_chars_usd": (self.cloud_cost_per_million_chars_usd),
            "cloud_cost_per_million_chars": self.cloud_cost_per_million_chars_usd,
            "local_cost_per_million_chars_usd": self.local_cost_per_million_chars_usd,
            "local_cost_per_million_chars": self.local_cost_per_million_chars_usd,
            "local_variable_cost_per_million_chars_usd": (
                self.local_variable_cost_per_million_chars_usd
            ),
            "price": self.price.to_dict(),
            "price_id": self.price.price_id,
            "provider": self.price.provider,
            "service": self.price.service,
            "source_url": self.price.source_url,
            "capture_date": self.price.capture_date,
            "verify": self.price.verify,
        }


@dataclass(frozen=True)
class CostVsCloudReport:
    """Serializable cost comparison report for one local perf report."""

    perf: Mapping[str, Any]
    chars_per_doc: float
    chars_per_second: float
    hardware_cost_model: HardwareCostModel
    comparisons: tuple[CloudCostComparison, ...]
    citation_issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible cost report."""
        comparisons = [comparison.to_dict() for comparison in self.comparisons]
        cloud_costs = {
            comparison.price.price_id: comparison.cloud_cost_per_million_chars_usd
            for comparison in self.comparisons
        }
        breakeven = {
            comparison.price.price_id: comparison.breakeven_volume_chars
            for comparison in self.comparisons
        }
        return {
            "benchmark": "cost_vs_cloud",
            "chars_per_doc": self.chars_per_doc,
            "chars_per_second": self.chars_per_second,
            "citation_issues": list(self.citation_issues),
            "cloud_cost_per_million_chars_usd": cloud_costs,
            "cloud_cost_per_million_chars": cloud_costs,
            "comparisons": comparisons,
            "hardware_cost_model": self.hardware_cost_model.to_dict(),
            "local": {
                "cost_per_million_chars_usd": (self._local_cost_per_million_chars_usd),
                "hourly_cost_usd": self.hardware_cost_model.hourly_cost_usd,
                "variable_cost_per_million_chars_usd": (
                    self._local_variable_cost_per_million_chars_usd
                ),
            },
            "local_cost_per_million_chars_usd": self._local_cost_per_million_chars_usd,
            "local_cost_per_million_chars": self._local_cost_per_million_chars_usd,
            "break_even_volume_chars": breakeven,
            "breakeven_volume_chars": breakeven,
            "perf": dict(self.perf),
            "schema_version": COST_REPORT_SCHEMA_VERSION,
        }

    @property
    def _local_cost_per_million_chars_usd(self) -> float:
        """Return amortized local cost for one million characters."""
        return _hourly_to_million_chars(
            self.hardware_cost_model.hourly_cost_usd,
            self.chars_per_second,
        )

    @property
    def _local_variable_cost_per_million_chars_usd(self) -> float:
        """Return variable-only local cost for one million characters."""
        return _hourly_to_million_chars(
            self.hardware_cost_model.operating_cost_usd_per_hour,
            self.chars_per_second,
        )

    def __getitem__(self, key: str) -> Any:
        """Allow callers to inspect the report like a JSON mapping."""
        return self.to_dict()[key]

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Serialize the report to a concise comparison table."""
        lines = [
            "# Cost-vs-cloud Benchmark Report",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Characters per document | {self.chars_per_doc} |",
            f"| Characters per second | {self.chars_per_second} |",
            (
                "| Local amortized cost per million characters | "
                f"${self._local_cost_per_million_chars_usd:.6f} |"
            ),
            "",
            "## Cloud comparison",
            "",
            "| Provider | Service | Cloud / 1M chars | Local / 1M chars | "
            "Break-even chars |",
            "|---|---|---:|---:|---:|",
        ]
        for comparison in self.comparisons:
            breakeven = comparison.breakeven_volume_chars
            breakeven_text = "none" if breakeven is None else f"{breakeven:.2f}"
            lines.append(
                f"| {comparison.price.provider} | {comparison.price.service} | "
                f"${comparison.cloud_cost_per_million_chars_usd:.6f} | "
                f"${comparison.local_cost_per_million_chars_usd:.6f} | "
                f"{breakeven_text} |"
            )

        lines.extend(["", "## Price citations", ""])
        for comparison in self.comparisons:
            price = comparison.price
            lines.append(
                f"- `{price.price_id}`: {price.source_url} "
                f"(captured {price.capture_date}; verify={price.verify})"
            )
        if self.citation_issues:
            lines.extend(["", "## Citation issues", ""])
            lines.extend(f"- {issue}" for issue in self.citation_issues)
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def load_cloud_prices(path: str | Path) -> Mapping[str, Any] | list[Any]:
    """Load the committed or user-supplied cloud price table from JSON."""
    price_path = Path(path)
    try:
        payload = json.loads(price_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"unable to read cloud price table {price_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"cloud price table is not valid JSON: {price_path}") from exc
    if not isinstance(payload, (MappingABC, list)):
        raise ValueError("cloud price table must be a JSON object or array")
    return payload


def cost_vs_cloud_report(
    perf_report: Any,
    cloud_prices: Any,
    hardware_cost_model: HardwareCostModel | Mapping[str, Any] | None = None,
) -> CostVsCloudReport:
    """Compare local amortized cost with each cited cloud price row.

    ``perf_report`` may be a :class:`PerfReport`, a JSON-compatible mapping, or
    a path to a JSON report. It must provide ``docs_per_second`` and
    ``chars_per_doc`` (or an equivalent character-throughput field). The price
    table may be a list of rows or an object containing ``prices`` and an
    optional ``hardware_cost_model``.
    """
    perf = _coerce_mapping(perf_report, "performance report")
    price_table = (
        load_cloud_prices(cloud_prices)
        if isinstance(cloud_prices, (str, Path))
        else cloud_prices
    )
    table_rows = _price_rows(price_table)
    prices = _parse_prices(table_rows)
    if not prices:
        raise ValueError("cloud price table must contain at least one price row")

    resolved_hardware = _resolve_hardware_cost_model(
        hardware_cost_model,
        price_table,
        perf,
    )
    docs_per_second = _performance_value(
        perf,
        "docs_per_second",
        "documents_per_second",
    )
    if docs_per_second <= 0:
        raise ValueError("performance report docs_per_second must be positive")
    chars_per_doc = _chars_per_doc(perf, docs_per_second)
    chars_per_second = docs_per_second * chars_per_doc
    local_cost = _hourly_to_million_chars(
        resolved_hardware.hourly_cost_usd,
        chars_per_second,
    )
    local_variable_cost = _hourly_to_million_chars(
        resolved_hardware.operating_cost_usd_per_hour,
        chars_per_second,
    )
    local_variable_cost_per_char = resolved_hardware.operating_cost_usd_per_hour / (
        chars_per_second * SECONDS_PER_HOUR
    )

    comparisons = tuple(
        CloudCostComparison(
            price=price,
            cloud_cost_per_million_chars_usd=price.price_usd_per_unit
            * MILLION_CHARS
            / price.unit_characters,
            local_cost_per_million_chars_usd=local_cost,
            local_variable_cost_per_million_chars_usd=local_variable_cost,
            breakeven_volume_chars=_breakeven_volume(
                fixed_cost_usd=resolved_hardware.capital_cost_usd,
                cloud_cost_per_char=price.price_usd_per_character,
                local_variable_cost_per_char=local_variable_cost_per_char,
            ),
        )
        for price in prices
    )
    return CostVsCloudReport(
        perf=_perf_summary(perf, docs_per_second, chars_per_doc),
        chars_per_doc=chars_per_doc,
        chars_per_second=chars_per_second,
        hardware_cost_model=resolved_hardware,
        comparisons=comparisons,
    )


def _price_rows(cloud_prices: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(cloud_prices, (str, Path)):
        cloud_prices = load_cloud_prices(cloud_prices)
    if isinstance(cloud_prices, MappingABC):
        rows = cloud_prices.get("prices", cloud_prices.get("entries"))
        if rows is None and "provider" in cloud_prices:
            rows = [cloud_prices]
    else:
        rows = cloud_prices
    if isinstance(rows, (str, bytes)) or not isinstance(rows, SequenceABC):
        raise ValueError("cloud price table must contain a prices array")
    if not all(isinstance(row, MappingABC) for row in rows):
        raise ValueError("cloud price table rows must be JSON objects")
    return rows


def _parse_prices(rows: Sequence[Mapping[str, Any]]) -> tuple[CloudPrice, ...]:
    prices: list[CloudPrice] = []
    issues: list[str] = []
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or row.get("price_id") or f"row-{index}")
        source_url = _first(row, "source_url", "source", "citation_url")
        capture_date = _first(row, "capture_date", "captured_at", "date")
        verify = _first(row, "verify", "verification", default=None)
        if not isinstance(source_url, str) or not source_url.strip():
            issues.append(f"{row_id}: missing source_url citation")
        elif not source_url.strip().startswith(("http://", "https://")):
            issues.append(f"{row_id}: source_url must be an http(s) URL")
        if not isinstance(capture_date, str) or not capture_date.strip():
            issues.append(f"{row_id}: missing capture_date")
        if verify is None:
            issues.append(f"{row_id}: missing verify marker")
        elif verify is False or verify == "":
            issues.append(f"{row_id}: verify marker is not affirmative")

        try:
            price, unit_characters = _price_and_unit(row)
        except ValueError as exc:
            issues.append(f"{row_id}: {exc}")
            continue
        if not isinstance(source_url, str) or not source_url.strip():
            continue
        if not isinstance(capture_date, str) or not capture_date.strip():
            continue
        if verify is None or verify is False or verify == "":
            continue
        prices.append(
            CloudPrice(
                price_id=row_id,
                provider=str(row.get("provider") or "unknown"),
                service=str(row.get("service") or row.get("product") or row_id),
                price_usd_per_unit=price,
                unit_characters=unit_characters,
                source_url=source_url.strip(),
                capture_date=capture_date.strip(),
                verify=verify,
                billing_unit=str(row.get("billing_unit") or "characters"),
            )
        )
    if issues:
        raise PriceCitationError(issues)
    return tuple(prices)


def _price_and_unit(row: Mapping[str, Any]) -> tuple[float, int]:
    per_1k = _first(row, "price_usd_per_1k_chars", "price_per_1k_chars")
    if per_1k is not None:
        return _positive_number(per_1k, "price_usd_per_1k_chars"), 1_000

    per_char = _first(row, "price_usd_per_character", "price_per_character")
    if per_char is not None:
        return _positive_number(per_char, "price_usd_per_character"), 1

    price = _first(row, "price_usd_per_unit", "price_usd", "price")
    if price is None:
        raise ValueError("missing price")
    unit_characters = _first(
        row,
        "unit_characters",
        "characters_per_unit",
        "chars_per_unit",
    )
    if unit_characters is None:
        raise ValueError("missing unit_characters")
    return _positive_number(price, "price_usd_per_unit"), _positive_int(
        unit_characters,
        "unit_characters",
    )


def _resolve_hardware_cost_model(
    explicit: HardwareCostModel | Mapping[str, Any] | None,
    cloud_prices: Any,
    perf: Mapping[str, Any],
) -> HardwareCostModel:
    value: Any = explicit
    if value is None and isinstance(cloud_prices, MappingABC):
        value = cloud_prices.get("hardware_cost_model")
    if value is None:
        value = perf.get("hardware_cost_model")
    if value is None:
        value = DEFAULT_HARDWARE_COST_MODEL
    if isinstance(value, HardwareCostModel):
        return value
    if not isinstance(value, MappingABC):
        raise ValueError("hardware_cost_model must be a mapping")
    return HardwareCostModel.from_mapping(value)


def _chars_per_doc(perf: Mapping[str, Any], docs_per_second: float) -> float:
    value = _performance_value(
        perf,
        "chars_per_doc",
        "characters_per_doc",
        "average_chars_per_doc",
        "avg_chars_per_doc",
        required=False,
    )
    if value <= 0:
        chars_per_second = _performance_value(
            perf,
            "throughput_chars_per_second",
            "chars_per_second",
            required=False,
        )
        if chars_per_second > 0:
            value = chars_per_second / docs_per_second
    if value <= 0:
        raise ValueError(
            "performance report must provide positive chars_per_doc or "
            "throughput_chars_per_second"
        )
    return value


def _performance_value(
    perf: Mapping[str, Any],
    *keys: str,
    required: bool = True,
) -> float:
    value = _first_nested(perf, keys)
    if value is None:
        if required:
            raise ValueError(f"performance report is missing {keys[0]}")
        return 0.0
    return _positive_number(value, keys[0])


def _first_nested(perf: Mapping[str, Any], keys: Sequence[str]) -> Any:
    nested = [perf.get("metadata"), perf.get("throughput"), perf.get("workload")]
    for mapping in (perf, *nested):
        if isinstance(mapping, MappingABC):
            value = _first(mapping, *keys, default=None)
            if value is not None:
                return value
    return None


def _perf_summary(
    perf: Mapping[str, Any],
    docs_per_second: float,
    chars_per_doc: float,
) -> dict[str, Any]:
    fields = ("model_name", "device", "tier", "canonical_tier", "generated_at")
    summary = {key: perf[key] for key in fields if key in perf}
    summary.update(
        {
            "chars_per_doc": chars_per_doc,
            "docs_per_second": docs_per_second,
        }
    )
    return summary


def _breakeven_volume(
    *,
    fixed_cost_usd: float,
    cloud_cost_per_char: float,
    local_variable_cost_per_char: float,
) -> float | None:
    margin = cloud_cost_per_char - local_variable_cost_per_char
    if margin <= 0:
        return None
    if fixed_cost_usd <= 0:
        return 0.0
    return fixed_cost_usd / margin


def _hourly_to_million_chars(hourly_cost_usd: float, chars_per_second: float) -> float:
    return hourly_cost_usd * MILLION_CHARS / (chars_per_second * SECONDS_PER_HOUR)


def _coerce_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        try:
            payload = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to read {label}: {value}") from exc
        value = payload
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, MappingABC):
        raise ValueError(f"{label} must be a JSON object or mapping")
    return value


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _number(
    mapping: Mapping[str, Any],
    *keys: str,
    default: float,
) -> float:
    value = _first(mapping, *keys, default=default)
    return _finite_float(value, keys[0])


def _positive_number(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _positive_number(value, name)
    if not result.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(result)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "COST_REPORT_SCHEMA_VERSION",
    "DEFAULT_CLOUD_PRICES_PATH",
    "DEFAULT_HARDWARE_COST_MODEL",
    "CloudCostComparison",
    "CloudPrice",
    "CostVsCloudReport",
    "HardwareCostModel",
    "PriceCitationError",
    "cost_vs_cloud_report",
    "load_cloud_prices",
]
