"""Fail-closed install-size and peak-RSS budgets for edge SBC reports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_EDGE_FOOTPRINT_BUDGETS = Path(__file__).with_name("budgets") / "edge_sbc.json"


@dataclass(frozen=True)
class FootprintBudget:
    """Maximum installed and resident bytes for one edge device profile."""

    profile: str
    device: str
    tier: str
    install_size_bytes_max: int
    peak_rss_bytes_max: int

    def to_dict(self) -> dict[str, Any]:
        """Return stable machine-readable budget metadata."""

        return {
            "device": self.device,
            "install_size_bytes_max": self.install_size_bytes_max,
            "peak_rss_bytes_max": self.peak_rss_bytes_max,
            "profile": self.profile,
            "tier": self.tier,
        }


@dataclass(frozen=True)
class FootprintCheck:
    """One inclusive upper-bound comparison."""

    name: str
    measured_bytes: int | None
    maximum_bytes: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a stable check record with byte and MiB values."""

        return {
            "maximum_bytes": self.maximum_bytes,
            "maximum_mib": round(self.maximum_bytes / (1024.0 * 1024.0), 6),
            "measured_bytes": self.measured_bytes,
            "measured_mib": (
                None
                if self.measured_bytes is None
                else round(self.measured_bytes / (1024.0 * 1024.0), 6)
            ),
            "name": self.name,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class FootprintGateResult:
    """Combined edge footprint verdict and failure evidence."""

    profile: str
    budget: FootprintBudget
    checks: Mapping[str, FootprintCheck]
    errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Return true only when every required measurement is valid and fits."""

        return not self.errors and all(check.passed for check in self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic aggregate gate evidence."""

        return {
            "budget": self.budget.to_dict(),
            "checks": {name: check.to_dict() for name, check in self.checks.items()},
            "errors": list(self.errors),
            "gate": "edge_sbc_footprint",
            "passed": self.passed,
            "profile": self.profile,
            "schema_version": 1,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the gate result as deterministic JSON."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the deterministic gate result to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path


def load_footprint_budgets(
    path: str | Path = DEFAULT_EDGE_FOOTPRINT_BUDGETS,
) -> dict[str, FootprintBudget]:
    """Load and validate every committed edge footprint profile."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("edge footprint budget requires schema_version 1")
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("edge footprint budget requires a non-empty profiles object")

    budgets: dict[str, FootprintBudget] = {}
    for profile, raw_budget in profiles.items():
        if not isinstance(profile, str) or not profile:
            raise ValueError("edge footprint profile names must be non-empty strings")
        if not isinstance(raw_budget, Mapping):
            raise ValueError("edge footprint profile entries must be objects")
        budget = FootprintBudget(
            profile=profile,
            device=str(raw_budget.get("device") or ""),
            tier=str(raw_budget.get("tier") or ""),
            install_size_bytes_max=_positive_int(
                raw_budget.get("install_size_bytes_max"),
                "install_size_bytes_max",
            ),
            peak_rss_bytes_max=_positive_int(
                raw_budget.get("peak_rss_bytes_max"),
                "peak_rss_bytes_max",
            ),
        )
        if not budget.device or not budget.tier:
            raise ValueError("edge footprint profiles require device and tier")
        budgets[profile] = budget
    return budgets


def evaluate_footprint(
    report: Mapping[str, Any] | Any,
    budget: FootprintBudget,
) -> FootprintGateResult:
    """Apply *budget* to an edge benchmark report and fail closed on omissions."""

    payload = report.to_dict() if hasattr(report, "to_dict") else report
    if not isinstance(payload, Mapping):
        raise TypeError("edge footprint gate requires a report object")

    errors: list[str] = []
    if payload.get("benchmark") != "edge_sbc":
        errors.append("report benchmark must be edge_sbc")
    if payload.get("schema_version") != 1:
        errors.append("report requires schema_version 1")
    if payload.get("offline") is not True:
        errors.append("report must assert offline execution")
    if payload.get("network_guard") != "socket-blocked":
        errors.append("report must assert the socket-blocked network guard")
    report_profile = payload.get("profile")
    if report_profile != budget.profile:
        errors.append("report profile does not match the selected budget")

    install_size = _measurement(payload.get("install_size_bytes"))
    peak_rss = _measurement(payload.get("peak_rss_bytes"))
    if install_size is None:
        errors.append("report is missing a valid install_size_bytes measurement")
    if peak_rss is None:
        errors.append("report is missing a valid peak_rss_bytes measurement")

    checks = {
        "install_size_bytes": FootprintCheck(
            name="install_size_bytes",
            measured_bytes=install_size,
            maximum_bytes=budget.install_size_bytes_max,
            passed=(
                install_size is not None
                and install_size <= budget.install_size_bytes_max
            ),
        ),
        "peak_rss_bytes": FootprintCheck(
            name="peak_rss_bytes",
            measured_bytes=peak_rss,
            maximum_bytes=budget.peak_rss_bytes_max,
            passed=peak_rss is not None and peak_rss <= budget.peak_rss_bytes_max,
        ),
    }
    return FootprintGateResult(
        profile=budget.profile,
        budget=budget,
        checks=checks,
        errors=tuple(errors),
    )


def gate_footprint(
    report: Mapping[str, Any] | Any,
    *,
    profile: str,
    budget_path: str | Path = DEFAULT_EDGE_FOOTPRINT_BUDGETS,
) -> FootprintGateResult:
    """Load *profile* and return its footprint verdict for *report*."""

    budgets = load_footprint_budgets(budget_path)
    try:
        budget = budgets[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(budgets))
        raise ValueError(
            f"unknown edge footprint profile; expected: {choices}"
        ) from exc
    return evaluate_footprint(report, budget)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _measurement(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Gate one edge result record and emit machine-readable evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--budget",
        type=Path,
        default=DEFAULT_EDGE_FOOTPRINT_BUDGETS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("edge-footprint-gate.json"),
    )
    args = parser.parse_args(argv)

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        result = gate_footprint(
            report,
            profile=args.profile,
            budget_path=args.budget,
        )
        result.write_json(args.output)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"Edge footprint gate could not run: {exc}", file=sys.stderr)
        return 2

    print(result.to_json())
    if result.passed:
        return 0
    print(
        "Edge footprint budget exceeded or required evidence is missing",
        file=sys.stderr,
    )
    return 1


__all__ = [
    "DEFAULT_EDGE_FOOTPRINT_BUDGETS",
    "FootprintBudget",
    "FootprintCheck",
    "FootprintGateResult",
    "evaluate_footprint",
    "gate_footprint",
    "load_footprint_budgets",
]


if __name__ == "__main__":
    raise SystemExit(main())
