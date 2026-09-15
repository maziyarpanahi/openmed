"""Deterministic cost comparison for local and hosted clinical text processing."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import tempfile
import unicodedata
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
_MAX_HARDWARE_COST_USD = 1_000_000_000.0
_MAX_INPUT_BYTES = 4 * 1024 * 1024
_MAX_MAPPING_FIELDS = 128
_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_MAX_PRICE_BYTES = 1024 * 1024
_MAX_PRICE_PER_1000_USD = 1_000_000.0
_MAX_PRICE_ROWS = 256
_MAX_SOURCE_URL_CHARS = 4_096
_MAX_TEXT_CHARS = 512
_MAX_THROUGHPUT_DOCS_PER_SECOND = 1_000_000_000.0
_MAX_CHARS_PER_DOCUMENT = 100_000_000.0
_MAX_USEFUL_LIFE_HOURS = 10_000_000.0
_MAX_POWER_WATTS = 10_000_000.0
_MAX_ELECTRICITY_USD_PER_KWH = 100_000.0
_MAX_MONTHLY_CHARACTERS = 10**18
_SAFE_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@+-]*\Z")


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

        normalized_indent = _normalize_indent(indent)
        payload = (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=False,
                indent=normalized_indent,
                sort_keys=True,
            )
            + "\n"
        )
        if len(payload.encode("utf-8")) > _MAX_OUTPUT_BYTES:
            raise ValueError("cost report exceeds the output size limit")
        return payload

    def to_markdown(self) -> str:
        """Render an aggregate-only Markdown comparison."""

        lines = [
            "# Cost vs cloud benchmark",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| Model | `{_markdown_cell(self.model_name)}` |",
            f"| Device | `{_markdown_cell(self.device)}` |",
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
                "| Provider | Service | Tier | Monthly character band | "
                "Marginal USD / 1M chars | Local USD / 1M chars | Savings / 1M "
                "chars | Breakeven chars at marginal rate | Source |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in self.comparisons:
            breakeven = (
                "never"
                if row.breakeven_characters is None
                else f"{row.breakeven_characters:,}"
            )
            lines.append(
                f"| {_markdown_cell(row.provider)} | "
                f"{_markdown_cell(row.service)} | {_markdown_cell(row.tier)} | "
                f"{_character_band(row)} | "
                f"{_rounded(row.cloud_cost_per_million_characters_usd)} | "
                f"{_rounded(row.local_cost_per_million_characters_usd)} | "
                f"{_rounded(row.savings_per_million_characters_usd)} | "
                f"{breakeven} | [captured {row.captured_at}]"
                f"(<{_markdown_url(row.source_url)}>) |"
            )
        lines.extend(
            [
                "",
                "Prices are dated snapshots, not quotes. Re-verify every row "
                "against its source URL before publication or purchasing decisions.",
                "Tier prices are marginal monthly bands. The report does not model "
                "a progressive monthly invoice; each breakeven is a sensitivity "
                "calculation at that row's marginal rate.",
                "",
            ]
        )
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> Path:
        """Write deterministic JSON to path."""

        output = Path(path)
        _atomic_write_text(output, self.to_json())
        return output

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to path."""

        output = Path(path)
        content = self.to_markdown()
        if len(content.encode("utf-8")) > _MAX_OUTPUT_BYTES:
            raise ValueError("cost report exceeds the output size limit")
        _atomic_write_text(output, content)
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

    docs_per_second = _positive_number(
        perf.get("docs_per_second"),
        "docs_per_second",
        maximum=_MAX_THROUGHPUT_DOCS_PER_SECOND,
    )
    metadata = perf.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("perf_report.metadata must be a mapping")
    metadata = _snapshot_mapping(metadata, "perf_report.metadata")
    chars_value = _first_present(
        perf,
        metadata,
        names=("chars_per_document", "chars_per_doc"),
    )
    chars_per_document = _positive_number(
        chars_value,
        "chars_per_document",
        maximum=_MAX_CHARS_PER_DOCUMENT,
    )
    chars_per_second = _positive_derived(
        docs_per_second * chars_per_document,
        "chars_per_second",
    )

    purchase_price = _non_negative_number(
        hardware.get("purchase_price_usd"),
        "purchase_price_usd",
        maximum=_MAX_HARDWARE_COST_USD,
    )
    useful_life_hours = _positive_number(
        hardware.get("useful_life_hours"),
        "useful_life_hours",
        maximum=_MAX_USEFUL_LIFE_HOURS,
    )
    power_watts = _non_negative_number(
        hardware.get("power_watts"),
        "power_watts",
        maximum=_MAX_POWER_WATTS,
    )
    electricity_rate = _non_negative_number(
        hardware.get("electricity_usd_per_kwh"),
        "electricity_usd_per_kwh",
        maximum=_MAX_ELECTRICITY_USD_PER_KWH,
    )
    energy_cost_per_hour = _finite_derived(
        power_watts / 1000.0 * electricity_rate,
        "energy_cost_per_hour",
    )
    amortized_cost_per_hour = _finite_derived(
        purchase_price / useful_life_hours + energy_cost_per_hour,
        "amortized_cost_per_hour",
    )
    chars_per_hour = _positive_derived(chars_per_second * 3600.0, "chars_per_hour")
    lifetime_character_capacity = _positive_derived(
        chars_per_hour * useful_life_hours,
        "lifetime_character_capacity",
    )
    local_cost_per_million = _finite_derived(
        amortized_cost_per_hour / chars_per_hour * MILLION_CHARACTERS,
        "local_cost_per_million_characters_usd",
    )
    energy_cost_per_character = _finite_derived(
        energy_cost_per_hour / chars_per_hour,
        "energy_cost_per_character",
    )

    price_rows = _validated_price_rows(prices)
    comparisons: list[CloudCostComparison] = []
    for row in price_rows:
        cloud_cost_per_million = _positive_derived(
            row["price_per_1000_characters_usd"] * 1000.0,
            "cloud_cost_per_million_characters_usd",
        )
        cloud_cost_per_character = _positive_derived(
            cloud_cost_per_million / MILLION_CHARACTERS,
            "cloud_cost_per_character",
        )
        contribution_margin = cloud_cost_per_character - energy_cost_per_character
        breakeven = None
        if contribution_margin > 0.0:
            candidate = purchase_price / contribution_margin
            if math.isfinite(candidate) and candidate <= lifetime_character_capacity:
                breakeven = math.ceil(candidate)
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

    model_name = _model_identifier(perf.get("model_name"))
    device = _safe_identifier(perf.get("device"), "device", default="unknown")
    normalized_hardware = {
        "electricity_usd_per_kwh": electricity_rate,
        "power_watts": power_watts,
        "purchase_price_usd": purchase_price,
        "useful_life_hours": useful_life_hours,
    }
    fingerprint_payload = {
        "chars_per_document": chars_per_document,
        "cloud_prices": {
            "currency": "USD",
            "prices": price_rows,
            "schema_version": CLOUD_PRICE_SCHEMA_VERSION,
        },
        "device": device,
        "docs_per_second": docs_per_second,
        "hardware_cost_model": normalized_hardware,
        "model_name": model_name,
    }
    fingerprint = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )

    return CostVsCloudReport(
        model_name=model_name,
        device=device,
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
        with resource.open("rb") as handle:
            encoded = handle.read(_MAX_PRICE_BYTES + 1)
    else:
        encoded = _read_bounded_bytes(
            Path(path),
            maximum_bytes=_MAX_PRICE_BYTES,
            name="cloud price table",
        )
    if len(encoded) > _MAX_PRICE_BYTES:
        raise ValueError("cloud price table exceeds the size limit")
    return _decode_json_object(encoded, "cloud price table")


def load_cost_input(path: str | Path, *, name: str) -> dict[str, Any]:
    """Load one JSON object used by the cost benchmark."""

    encoded = _read_bounded_bytes(
        Path(path),
        maximum_bytes=_MAX_INPUT_BYTES,
        name=name,
    )
    return _decode_json_object(encoded, name)


def _validated_price_rows(prices: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if prices.get("schema_version") != CLOUD_PRICE_SCHEMA_VERSION:
        raise ValueError(
            f"cloud price table schema_version must be {CLOUD_PRICE_SCHEMA_VERSION}"
        )
    if prices.get("currency") != "USD":
        raise ValueError("cloud price table currency must be USD")
    table_captured_at = prices.get("captured_at")
    if table_captured_at is not None:
        table_captured_at = _iso_date(
            table_captured_at,
            "cloud price table captured_at",
        )
    raw_rows = prices.get("prices")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise TypeError("cloud price table prices must be a sequence")
    if not raw_rows:
        raise ValueError("cloud price table must contain at least one paid tier")
    if len(raw_rows) > _MAX_PRICE_ROWS:
        raise ValueError("cloud price table contains too many tiers")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    for index, raw in enumerate(raw_rows):
        if index >= _MAX_PRICE_ROWS:
            raise ValueError("cloud price table contains too many tiers")
        if not isinstance(raw, Mapping):
            raise TypeError(f"cloud price row {index} must be an object")
        raw = _snapshot_mapping(raw, f"cloud price row {index}")
        if set(raw) != {
            "captured_at",
            "maximum_monthly_characters",
            "minimum_monthly_characters",
            "price_per_1000_characters_usd",
            "provider",
            "region",
            "service",
            "source_effective_at",
            "source_url",
            "tier",
            "verify",
        }:
            raise ValueError(f"cloud price row {index} has an invalid schema")
        if raw.get("verify") is not True:
            raise ValueError(f"cloud price row {index} must set verify to true")
        source_url = _source_url(
            raw.get("source_url"),
            f"prices[{index}].source_url",
        )
        captured_at = _iso_date(raw.get("captured_at"), f"prices[{index}].captured_at")
        if table_captured_at is not None and captured_at != table_captured_at:
            raise ValueError(
                f"cloud price row {index} capture date differs from the table"
            )
        source_effective = raw.get("source_effective_at")
        if source_effective is not None:
            source_effective = _iso_date(
                source_effective,
                f"prices[{index}].source_effective_at",
            )
            if source_effective > captured_at:
                raise ValueError(
                    f"cloud price row {index} effective date follows capture date"
                )
        minimum = _non_negative_integer(
            raw.get("minimum_monthly_characters"),
            f"prices[{index}].minimum_monthly_characters",
            maximum=_MAX_MONTHLY_CHARACTERS,
        )
        maximum_raw = raw.get("maximum_monthly_characters")
        maximum = (
            None
            if maximum_raw is None
            else _positive_integer(
                maximum_raw,
                f"prices[{index}].maximum_monthly_characters",
                maximum=_MAX_MONTHLY_CHARACTERS,
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
                maximum=_MAX_PRICE_PER_1000_USD,
            ),
            "source_url": source_url,
            "captured_at": captured_at,
            "source_effective_at": source_effective,
            "verify": True,
        }
        key = (
            row["provider"],
            row["service"],
            row["region"],
            row["tier"],
            minimum,
        )
        if key in seen:
            raise ValueError("cloud price table contains a duplicate tier")
        seen.add(key)
        normalized.append(row)

    ordered = sorted(
        normalized,
        key=lambda row: (
            row["provider"],
            row["service"],
            row["region"],
            row["minimum_monthly_characters"],
            row["tier"],
        ),
    )
    previous_by_meter: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in ordered:
        meter = (row["provider"], row["service"], row["region"])
        previous = previous_by_meter.get(meter)
        if previous is not None:
            previous_maximum = previous["maximum_monthly_characters"]
            if previous_maximum is None or row["minimum_monthly_characters"] < (
                previous_maximum
            ):
                raise ValueError("cloud price table contains overlapping tiers")
        previous_by_meter[meter] = row
    return tuple(ordered)


def _coerce_mapping(value: Mapping[str, Any] | Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _snapshot_mapping(value, name)
    try:
        to_dict = getattr(value, "to_dict", None)
    except Exception as exc:
        raise TypeError(f"{name} must expose an ordinary to_dict() method") from exc
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return _snapshot_mapping(payload, name)
    raise TypeError(f"{name} must be a mapping or expose to_dict()")


def _snapshot_mapping(value: Mapping[Any, Any], name: str) -> dict[str, Any]:
    if len(value) > _MAX_MAPPING_FIELDS:
        raise ValueError(f"{name} contains too many fields")
    result: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= _MAX_MAPPING_FIELDS:
            raise ValueError(f"{name} contains too many fields")
        if (
            not isinstance(key, str)
            or not key
            or len(key) > _MAX_TEXT_CHARS
            or _has_control_character(key)
        ):
            raise ValueError(f"{name} contains an invalid field name")
        result[key] = item
    return result


def _first_present(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    *,
    names: Sequence[str],
) -> Any:
    for mapping in (primary, secondary):
        for name in names:
            if name in mapping and mapping[name] is not None:
                return mapping[name]
    return None


def _required_text(
    value: Any,
    name: str,
    *,
    maximum_chars: int = _MAX_TEXT_CHARS,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    if len(normalized) > maximum_chars:
        raise ValueError(f"{name} exceeds the size limit")
    if _has_control_character(normalized):
        raise ValueError(f"{name} must not contain control characters")
    return normalized


def _source_url(value: Any, name: str) -> str:
    source_url = _required_text(
        value,
        name,
        maximum_chars=_MAX_SOURCE_URL_CHARS,
    )
    if any(character.isspace() or character in "<>\\" for character in source_url):
        raise ValueError(f"{name} must be a safe HTTPS URL")
    parsed_url = urlparse(source_url)
    try:
        port = parsed_url.port
    except ValueError as exc:
        raise ValueError(f"{name} must be a safe HTTPS URL") from exc
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname is None
        or parsed_url.username is not None
        or parsed_url.password is not None
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise ValueError(f"{name} must be a safe HTTPS URL")
    return source_url


def _iso_date(value: Any, name: str) -> str:
    text = _required_text(value, name)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date") from exc
    if parsed > date.today():
        raise ValueError(f"{name} must not be in the future")
    return text


def _positive_number(value: Any, name: str, *, maximum: float) -> float:
    parsed = _finite_number(value, name)
    if not 0 < parsed <= maximum:
        raise ValueError(f"{name} must be greater than zero and within the limit")
    return parsed


def _non_negative_number(value: Any, name: str, *, maximum: float) -> float:
    parsed = _finite_number(value, name)
    if not 0 <= parsed <= maximum:
        raise ValueError(f"{name} must be non-negative and within the limit")
    return parsed


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _finite_derived(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"derived {name} is outside the supported range")
    return value


def _positive_derived(value: float, name: str) -> float:
    parsed = _finite_derived(value, name)
    if parsed <= 0:
        raise ValueError(f"derived {name} must be greater than zero")
    return parsed


def _positive_integer(value: Any, name: str, *, maximum: int) -> int:
    parsed = _non_negative_integer(value, name, maximum=maximum)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _non_negative_integer(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be non-negative and within the limit")
    return value


def _normalize_indent(indent: int) -> int:
    if isinstance(indent, bool) or not isinstance(indent, int) or not 0 <= indent <= 8:
        raise ValueError("indent must be an integer between 0 and 8")
    return indent


def _read_bounded_bytes(path: Path, *, maximum_bytes: int, name: str) -> bytes:
    with path.open("rb") as handle:
        encoded = handle.read(maximum_bytes + 1)
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{name} exceeds the size limit")
    return encoded


def _decode_json_object(encoded: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a JSON object")
    return payload


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    descriptor_open = True
    try:
        set_descriptor_mode = getattr(os, "fchmod", None)
        if set_descriptor_mode is not None:
            set_descriptor_mode(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor_open = False
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def _safe_identifier(value: Any, name: str, *, default: str) -> str:
    if value is None:
        return default
    text = _required_text(value, name)
    if _SAFE_IDENTIFIER_RE.fullmatch(text) is None or "//" in text:
        raise ValueError(f"{name} must be a safe identifier")
    return text


def _model_identifier(value: Any) -> str:
    if value is None:
        return "unknown"
    text = _required_text(value, "model_name", maximum_chars=4_096)
    if (
        Path(text).is_absolute()
        or text.startswith(("./", "../"))
        or "\\" in text
        or _SAFE_IDENTIFIER_RE.fullmatch(text) is None
    ):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"local-sha256-{digest}"
    return text


def _markdown_cell(value: str) -> str:
    return html.escape(value, quote=False).replace("|", "\\|")


def _markdown_url(value: str) -> str:
    return (
        value.replace("(", "%28")
        .replace(")", "%29")
        .replace('"', "%22")
        .replace("'", "%27")
    )


def _has_control_character(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


def _character_band(row: CloudCostComparison) -> str:
    maximum = (
        "unbounded"
        if row.maximum_monthly_characters is None
        else f"{row.maximum_monthly_characters:,}"
    )
    return f"{row.minimum_monthly_characters:,}–{maximum}"


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
