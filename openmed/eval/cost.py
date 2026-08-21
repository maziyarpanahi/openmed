"""Deterministic cost comparison for local and hosted clinical text processing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

COST_REPORT_SCHEMA_VERSION = 1
CLOUD_PRICE_SCHEMA_VERSION = 1
DEFAULT_CLOUD_PRICE_RESOURCE = "data/cloud_prices.json"
MILLION_CHARACTERS = 1_000_000


@dataclass(frozen=True)
class CloudCostComparison:
    """One normalized cloud-price tier compared with local execution."""

    provider: str
    service: str
    region: str
    tier: str
    minimum_monthly_characters: int
    maximum_monthly_characters: int | None
    price_per_1000_characters_usd: float
    cloud_cost_per_million_characters_usd: float
    local_cost_per_million_characters_usd: float
    savings_per_million_characters_usd: float
    breakeven_characters: int | None
    source_url: str
    captured_at: str
    source_effective_at: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible row."""

        return {
            "breakeven_characters": self.breakeven_characters,
            "captured_at": self.captured_at,
            "cloud_cost_per_million_characters_usd": _rounded(
                self.cloud_cost_per_million_characters_usd
            ),
            "local_cost_per_million_characters_usd": _rounded(
                self.local_cost_per_million_characters_usd
            ),
            "maximum_monthly_characters": self.maximum_monthly_characters,
            "minimum_monthly_characters": self.minimum_monthly_characters,
            "price_per_1000_characters_usd": _rounded(
                self.price_per_1000_characters_usd
            ),
            "provider": self.provider,
            "region": self.region,
            "savings_per_million_characters_usd": _rounded(
                self.savings_per_million_characters_usd
            ),
            "service": self.service,
            "source_effective_at": self.source_effective_at,
            "source_url": self.source_url,
            "tier": self.tier,
        }


@dataclass(frozen=True)
class CostVsCloudReport:
    """Aggregate-only local cost model and cited cloud comparisons."""

    model_name: str
    device: str
    docs_per_second: float
    chars_per_document: float
    chars_per_second: float
    hardware_purchase_price_usd: float
    hardware_useful_life_hours: float
    power_watts: float
    electricity_usd_per_kwh: float
    amortized_local_cost_per_hour_usd: float
    local_cost_per_million_characters_usd: float
    comparisons: tuple[CloudCostComparison, ...]
    input_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible report."""

        return {
            "amortized_local_cost_per_hour_usd": _rounded(
                self.amortized_local_cost_per_hour_usd
            ),
            "chars_per_document": _rounded(self.chars_per_document),
            "chars_per_second": _rounded(self.chars_per_second),
            "comparisons": [row.to_dict() for row in self.comparisons],
            "device": self.device,
            "docs_per_second": _rounded(self.docs_per_second),
            "electricity_usd_per_kwh": _rounded(self.electricity_usd_per_kwh),
            "hardware_purchase_price_usd": _rounded(self.hardware_purchase_price_usd),
            "hardware_useful_life_hours": _rounded(self.hardware_useful_life_hours),
            "input_fingerprint": self.input_fingerprint,
            "local_cost_per_million_characters_usd": _rounded(
                self.local_cost_per_million_characters_usd
            ),
            "model_name": self.model_name,
            "power_watts": _rounded(self.power_watts),
            "schema_version": COST_REPORT_SCHEMA_VERSION,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        """Render an aggregate-only Markdown comparison."""

        lines = [
            "# Cost vs cloud benchmark",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| Documents per second | {_rounded(self.docs_per_second)} |",
            f"| Characters per document | {_rounded(self.chars_per_document)} |",
            f"| Characters per second | {_rounded(self.chars_per_second)} |",
            (
                "| Amortized local USD / million characters | "
                f"{_rounded(self.local_cost_per_million_characters_usd)} |"
            ),
            "",
            "## Paid cloud tiers",
            "",
            (
                "| Provider | Service | Tier | USD / 1M chars | Local USD / 1M "
                "chars | Savings / 1M chars | Breakeven chars | Source |"
            ),
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        for row in self.comparisons:
            breakeven = (
                "never"
                if row.breakeven_characters is None
                else f"{row.breakeven_characters:,}"
            )
            lines.append(
                f"| {row.provider} | {row.service} | {row.tier} | "
                f"{_rounded(row.cloud_cost_per_million_characters_usd)} | "
                f"{_rounded(row.local_cost_per_million_characters_usd)} | "
                f"{_rounded(row.savings_per_million_characters_usd)} | "
                f"{breakeven} | [captured {row.captured_at}]({row.source_url}) |"
            )
        lines.extend(
            [
                "",
                "Prices are dated snapshots, not quotes. Re-verify every row "
                "against its source URL before publication or purchasing decisions.",
                "",
            ]
        )
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> Path:
        """Write deterministic JSON to path."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(), encoding="utf-8")
        return output

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to path."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_markdown(), encoding="utf-8")
        return output


def cost_vs_cloud_report(
    perf_report: Mapping[str, Any] | Any,
    cloud_prices: Mapping[str, Any],
    hardware_cost_model: Mapping[str, Any],
) -> CostVsCloudReport:
    """Compare measured local throughput with cited paid cloud-price tiers.

    Args:
        perf_report: Perf report mapping or object exposing to_dict(). It must
            provide docs_per_second plus chars_per_document either at the
            top level or in metadata.
        cloud_prices: Versioned table containing fully cited and verified
            normalized paid tiers.
        hardware_cost_model: Purchase price, useful life, power draw, and
            electricity price used for the local amortization model.

    Returns:
        Deterministic aggregate cost comparison.

    Raises:
        TypeError: If an input has the wrong shape.
        ValueError: If a required value or price citation is missing.
    """

    perf = _coerce_mapping(perf_report, "perf_report")
    prices = _coerce_mapping(cloud_prices, "cloud_prices")
    hardware = _coerce_mapping(hardware_cost_model, "hardware_cost_model")

    docs_per_second = _positive_number(perf.get("docs_per_second"), "docs_per_second")
    metadata = perf.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise TypeError("perf_report.metadata must be a mapping")
    chars_per_document = _positive_number(
        perf.get("chars_per_document")
        or perf.get("chars_per_doc")
        or metadata.get("chars_per_document")
        or metadata.get("chars_per_doc"),
        "chars_per_document",
    )
    chars_per_second = docs_per_second * chars_per_document

    purchase_price = _non_negative_number(
        hardware.get("purchase_price_usd"), "purchase_price_usd"
    )
    useful_life_hours = _positive_number(
        hardware.get("useful_life_hours"), "useful_life_hours"
    )
    power_watts = _non_negative_number(hardware.get("power_watts"), "power_watts")
    electricity_rate = _non_negative_number(
        hardware.get("electricity_usd_per_kwh"), "electricity_usd_per_kwh"
    )
    energy_cost_per_hour = power_watts / 1000.0 * electricity_rate
    amortized_cost_per_hour = purchase_price / useful_life_hours + energy_cost_per_hour
    chars_per_hour = chars_per_second * 3600.0
    local_cost_per_million = (
        amortized_cost_per_hour / chars_per_hour * MILLION_CHARACTERS
    )
    energy_cost_per_character = energy_cost_per_hour / chars_per_hour

    price_rows = _validated_price_rows(prices)
    comparisons: list[CloudCostComparison] = []
    for row in price_rows:
        cloud_cost_per_million = row["price_per_1000_characters_usd"] * 1000.0
        cloud_cost_per_character = cloud_cost_per_million / MILLION_CHARACTERS
        contribution_margin = cloud_cost_per_character - energy_cost_per_character
        breakeven = (
            None
            if contribution_margin <= 0.0
            else math.ceil(purchase_price / contribution_margin)
        )
        comparisons.append(
            CloudCostComparison(
                provider=row["provider"],
                service=row["service"],
                region=row["region"],
                tier=row["tier"],
                minimum_monthly_characters=row["minimum_monthly_characters"],
                maximum_monthly_characters=row["maximum_monthly_characters"],
                price_per_1000_characters_usd=row["price_per_1000_characters_usd"],
                cloud_cost_per_million_characters_usd=cloud_cost_per_million,
                local_cost_per_million_characters_usd=local_cost_per_million,
                savings_per_million_characters_usd=(
                    cloud_cost_per_million - local_cost_per_million
                ),
                breakeven_characters=breakeven,
                source_url=row["source_url"],
                captured_at=row["captured_at"],
                source_effective_at=row["source_effective_at"],
            )
        )

    fingerprint_payload = {
        "chars_per_document": chars_per_document,
        "cloud_prices": prices,
        "docs_per_second": docs_per_second,
        "hardware_cost_model": hardware,
    }
    fingerprint = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )

    return CostVsCloudReport(
        model_name=str(perf.get("model_name") or "unknown"),
        device=str(perf.get("device") or "unknown"),
        docs_per_second=docs_per_second,
        chars_per_document=chars_per_document,
        chars_per_second=chars_per_second,
        hardware_purchase_price_usd=purchase_price,
        hardware_useful_life_hours=useful_life_hours,
        power_watts=power_watts,
        electricity_usd_per_kwh=electricity_rate,
        amortized_local_cost_per_hour_usd=amortized_cost_per_hour,
        local_cost_per_million_characters_usd=local_cost_per_million,
        comparisons=tuple(comparisons),
        input_fingerprint=fingerprint,
    )


def load_cloud_prices(path: str | Path | None = None) -> dict[str, Any]:
    """Load the bundled or caller-selected cloud-price table."""

    if path is None:
        resource = resources.files("openmed.eval").joinpath(
            DEFAULT_CLOUD_PRICE_RESOURCE
        )
        with resource.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("cloud price table must be a JSON object")
    return payload


def load_cost_input(path: str | Path, *, name: str) -> dict[str, Any]:
    """Load one JSON object used by the cost benchmark."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a JSON object")
    return payload


def _validated_price_rows(prices: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if prices.get("schema_version") != CLOUD_PRICE_SCHEMA_VERSION:
        raise ValueError(
            f"cloud price table schema_version must be {CLOUD_PRICE_SCHEMA_VERSION}"
        )
    if prices.get("currency") != "USD":
        raise ValueError("cloud price table currency must be USD")
    raw_rows = prices.get("prices")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise TypeError("cloud price table prices must be a sequence")
    if not raw_rows:
        raise ValueError("cloud price table must contain at least one paid tier")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"cloud price row {index} must be an object")
        if raw.get("verify") is not True:
            raise ValueError(f"cloud price row {index} must set verify to true")
        source_url = _required_text(
            raw.get("source_url"), f"prices[{index}].source_url"
        )
        parsed_url = urlparse(source_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError(f"cloud price row {index} requires an HTTPS source URL")
        captured_at = _iso_date(raw.get("captured_at"), f"prices[{index}].captured_at")
        source_effective = raw.get("source_effective_at")
        if source_effective is not None:
            source_effective = _iso_date(
                source_effective,
                f"prices[{index}].source_effective_at",
            )
        minimum = _non_negative_integer(
            raw.get("minimum_monthly_characters"),
            f"prices[{index}].minimum_monthly_characters",
        )
        maximum_raw = raw.get("maximum_monthly_characters")
        maximum = (
            None
            if maximum_raw is None
            else _positive_integer(
                maximum_raw,
                f"prices[{index}].maximum_monthly_characters",
            )
        )
        if maximum is not None and maximum <= minimum:
            raise ValueError(f"cloud price row {index} maximum must exceed its minimum")
        row: dict[str, Any] = {
            "provider": _required_text(
                raw.get("provider"), f"prices[{index}].provider"
            ),
            "service": _required_text(raw.get("service"), f"prices[{index}].service"),
            "region": _required_text(raw.get("region"), f"prices[{index}].region"),
            "tier": _required_text(raw.get("tier"), f"prices[{index}].tier"),
            "minimum_monthly_characters": minimum,
            "maximum_monthly_characters": maximum,
            "price_per_1000_characters_usd": _positive_number(
                raw.get("price_per_1000_characters_usd"),
                f"prices[{index}].price_per_1000_characters_usd",
            ),
            "source_url": source_url,
            "captured_at": captured_at,
            "source_effective_at": source_effective,
        }
        key = (row["provider"], row["service"], row["tier"], minimum)
        if key in seen:
            raise ValueError(f"duplicate cloud price tier: {key!r}")
        seen.add(key)
        normalized.append(row)

    return tuple(
        sorted(
            normalized,
            key=lambda row: (
                row["provider"],
                row["service"],
                row["minimum_monthly_characters"],
                row["tier"],
            ),
        )
    )


def _coerce_mapping(value: Mapping[str, Any] | Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError(f"{name} must be a mapping or expose to_dict()")


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _iso_date(value: Any, name: str) -> str:
    text = _required_text(value, name)
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date") from exc
    return text


def _positive_number(value: Any, name: str) -> float:
    parsed = _finite_number(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _non_negative_number(value: Any, name: str) -> float:
    parsed = _finite_number(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _positive_integer(value: Any, name: str) -> int:
    parsed = _non_negative_integer(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _rounded(value: float) -> float:
    return round(float(value), 12)


__all__ = [
    "CLOUD_PRICE_SCHEMA_VERSION",
    "COST_REPORT_SCHEMA_VERSION",
    "CloudCostComparison",
    "CostVsCloudReport",
    "cost_vs_cloud_report",
    "load_cloud_prices",
    "load_cost_input",
]
