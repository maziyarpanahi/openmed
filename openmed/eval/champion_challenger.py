"""Champion-challenger promotion framework gated on safety wins.

A daily re-release replaces a family's shipping model. This module treats the
incumbent shipping model as the *champion* and the freshly gated candidate as
the *challenger*, aligns their signed :class:`~openmed.eval.release_gates.GateReport`
evidence per gate, and emits a single, auditable :class:`PromotionVerdict`
(``PROMOTE`` / ``HOLD`` / ``REJECT``).

Design contract
---------------
The whole decision is *pure and offline*: it is derived from the two committed
gate reports with **no live model call**. A challenger is promotable only when
it wins on the safety-critical axes -- per-label recall, residual leakage,
quantization recall delta, and tier-fit latency -- and never on aggregate F1.
Aggregate F1 is deliberately excluded from every promotion reason; it is
captured in ``ignored_f1`` purely for transparency and is never read by the
verdict logic.

The verdict requires the challenger to meet every G1a-G8 floor (its own gate
decision is ``RELEASABLE``) *and* to not regress any critical-label recall
versus the champion beyond the shared G7 tolerance
(:data:`~openmed.eval.release_gates.G7_RECALL_DROP_LIMIT`). The per-gate
win/loss/tie accounting is encoded in a signed comparison record whose ledger
reproduces byte-identically from the two reports (a deterministic sha256 of the
canonical JSON is embedded as the signature).

On ``PROMOTE`` the caller writes the new champion pointer and the comparison
record. On ``HOLD`` / ``REJECT`` the champion pointer is left intact and an
offline issue draft is emitted -- this module never touches the network.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.audit import stable_hash
from openmed.core.baseline import baseline_key
from openmed.eval.release_gates import (
    _CRITICAL_LABELS,
    G7_RECALL_DROP_LIMIT,
    RELEASABLE,
    GateReport,
)

#: Labels whose recall is treated as safety-critical for promotion. Sourced from
#: the release-gate definition so the two never drift.
CRITICAL_RECALL_LABELS = _CRITICAL_LABELS

#: Promotion verdicts.
PROMOTE = "PROMOTE"
HOLD = "HOLD"
REJECT = "REJECT"

#: Per-axis ledger outcomes.
WIN = "win"
TIE = "tie"
REGRESSION = "regression"

#: Ledger axis categories.
CATEGORY_RECALL = "recall"
CATEGORY_LEAKAGE = "leakage"
CATEGORY_QUANT = "quant"
CATEGORY_LATENCY = "latency"
CATEGORY_GATE = "gate"

HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"

COMPARISON_RECORD_ARTIFACT = "openmed.eval.champion_challenger_comparison"
COMPARISON_RECORD_SCHEMA_VERSION = 1
CHAMPION_POINTER_SCHEMA_VERSION = 1

_REPO_ROOT = Path(__file__).resolve().parents[2]
#: Committed champion pointer store, persisted alongside ``gates/baseline.json``.
CHAMPION_POINTER_PATH = _REPO_ROOT / "gates" / "champion.json"

#: Aggregate-F1 metric markers. These are captured for transparency but are
#: never used as a promotion reason.
_AGGREGATE_F1_MARKERS = (
    "aggregate_f1",
    "f1",
    "f1_score",
    "macro_f1",
    "micro_f1",
    "relaxed_f1",
    "strict_f1",
)

_ABS_TOL = 1e-12

ReportLike = GateReport | Mapping[str, Any] | str | Path


@dataclass(frozen=True)
class GateLedgerEntry:
    """One aligned champion-vs-challenger axis in the promotion ledger."""

    axis: str
    category: str
    direction: str
    champion: float
    challenger: float
    delta: float
    outcome: str
    tolerance: float = 0.0

    @property
    def blocking(self) -> bool:
        """Whether a regression on this axis must force ``REJECT``.

        Critical-label recall regressions beyond tolerance, a gate the champion
        passed that the challenger now fails, and any increase in the critical
        leakage count are hard safety failures. Leakage-rate, quant-delta, and
        tier-fit regressions are soft and never block on their own.
        """

        if self.outcome != REGRESSION:
            return False
        if self.category == CATEGORY_RECALL:
            return True
        if self.category == CATEGORY_GATE:
            return True
        return self.axis == "critical_leakage_count"

    @property
    def soft_safety_regression(self) -> bool:
        """Whether a non-blocking safety regression downgrades ``PROMOTE`` to ``HOLD``."""

        if self.outcome != REGRESSION or self.blocking:
            return False
        return self.category in (CATEGORY_LEAKAGE, CATEGORY_QUANT)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "axis": self.axis,
            "blocking": self.blocking,
            "category": self.category,
            "challenger": self.challenger,
            "champion": self.champion,
            "delta": self.delta,
            "direction": self.direction,
            "outcome": self.outcome,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class PromotionVerdict:
    """Signed, reproducible champion-challenger promotion decision."""

    verdict: str
    key: str
    family: str
    tier: str
    format: str
    champion_repo_id: str
    challenger_repo_id: str
    champion_repro_hash: str
    challenger_repro_hash: str
    ledger: tuple[GateLedgerEntry, ...] = ()
    reasons: tuple[str, ...] = ()
    ignored_f1: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = COMPARISON_RECORD_SCHEMA_VERSION
    record_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ledger",
            tuple(sorted(self.ledger, key=lambda entry: entry.axis)),
        )
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "ignored_f1", dict(self.ignored_f1))
        if not self.record_hash:
            object.__setattr__(self, "record_hash", self.recompute_record_hash())

    @property
    def promoted(self) -> bool:
        """Whether the verdict authorizes advancing the champion pointer."""

        return self.verdict == PROMOTE

    @property
    def wins(self) -> tuple[str, ...]:
        """Axes where the challenger strictly beat the champion."""

        return tuple(entry.axis for entry in self.ledger if entry.outcome == WIN)

    @property
    def ties(self) -> tuple[str, ...]:
        """Axes where the challenger neither beat nor regressed the champion."""

        return tuple(entry.axis for entry in self.ledger if entry.outcome == TIE)

    @property
    def regressions(self) -> tuple[str, ...]:
        """Axes where the challenger regressed versus the champion."""

        return tuple(entry.axis for entry in self.ledger if entry.outcome == REGRESSION)

    def _payload(self, *, include_record_hash: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact": COMPARISON_RECORD_ARTIFACT,
            "challenger": {
                "repo_id": self.challenger_repo_id,
                "repro_hash": self.challenger_repro_hash,
            },
            "champion": {
                "repo_id": self.champion_repo_id,
                "repro_hash": self.champion_repro_hash,
            },
            "family": self.family,
            "format": self.format,
            "ignored_f1": _json_plain(self.ignored_f1),
            "key": self.key,
            "ledger": [entry.to_dict() for entry in self.ledger],
            "reasons": list(self.reasons),
            "regressions": list(self.regressions),
            "schema_version": self.schema_version,
            "tier": self.tier,
            "ties": list(self.ties),
            "verdict": self.verdict,
            "wins": list(self.wins),
        }
        if include_record_hash:
            payload["record_hash"] = self.record_hash
        return payload

    def recompute_record_hash(self) -> str:
        """Recompute the content hash without trusting the stored value."""

        return stable_hash(self._payload(include_record_hash=False))

    def verify(self) -> bool:
        """Return whether the stored record hash matches the canonical payload."""

        return (
            bool(self.record_hash) and self.record_hash == self.recompute_record_hash()
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the signed, JSON-ready comparison record."""

        return self._payload(include_record_hash=True)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the comparison record to byte-stable JSON."""

        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromotionVerdict":
        """Rebuild a verdict from a persisted comparison record."""

        champion = _mapping(data.get("champion"))
        challenger = _mapping(data.get("challenger"))
        ledger = tuple(
            GateLedgerEntry(
                axis=str(item.get("axis", "")),
                category=str(item.get("category", "")),
                direction=str(item.get("direction", HIGHER_IS_BETTER)),
                champion=float(item.get("champion", 0.0)),
                challenger=float(item.get("challenger", 0.0)),
                delta=float(item.get("delta", 0.0)),
                outcome=str(item.get("outcome", TIE)),
                tolerance=float(item.get("tolerance", 0.0)),
            )
            for item in data.get("ledger", [])
            if isinstance(item, Mapping)
        )
        return cls(
            verdict=str(data.get("verdict", HOLD)),
            key=str(data.get("key", "")),
            family=str(data.get("family", "")),
            tier=str(data.get("tier", "")),
            format=str(data.get("format", "")),
            champion_repo_id=str(champion.get("repo_id", "")),
            challenger_repo_id=str(challenger.get("repo_id", "")),
            champion_repro_hash=str(champion.get("repro_hash", "")),
            challenger_repro_hash=str(challenger.get("repro_hash", "")),
            ledger=ledger,
            reasons=tuple(str(reason) for reason in data.get("reasons", [])),
            ignored_f1=_mapping(data.get("ignored_f1")),
            schema_version=int(
                data.get("schema_version", COMPARISON_RECORD_SCHEMA_VERSION)
            ),
            record_hash=str(data.get("record_hash", "")),
        )


@dataclass(frozen=True)
class ChampionPointerEntry:
    """One committed champion pointer for a ``(family, tier, format)`` key."""

    family: str
    tier: str
    format: str
    champion: str
    record_hash: str = ""
    previous_champion: str | None = None

    @property
    def key(self) -> str:
        """Return the canonical ``family::tier::format`` store key."""

        return baseline_key(self.family, self.tier, self.format)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "champion": self.champion,
            "family": self.family,
            "format": self.format,
            "key": self.key,
            "previous_champion": self.previous_champion,
            "record_hash": self.record_hash,
            "tier": self.tier,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChampionPointerEntry":
        """Restore a pointer entry from its JSON payload."""

        return cls(
            family=str(data.get("family", "")),
            tier=str(data.get("tier", "")),
            format=str(data.get("format", "")),
            champion=str(data.get("champion", "")),
            record_hash=str(data.get("record_hash", "")),
            previous_champion=(
                None
                if data.get("previous_champion") is None
                else str(data["previous_champion"])
            ),
        )


@dataclass
class ChampionPointer:
    """Committed champion pointer store keyed by rollout coordinates."""

    entries: dict[str, ChampionPointerEntry] = field(default_factory=dict)
    schema_version: int = CHAMPION_POINTER_SCHEMA_VERSION

    def get(self, family: str, tier: str, format: str) -> ChampionPointerEntry | None:
        """Return the champion pointer for a key, or ``None``."""

        return self.entries.get(baseline_key(family, tier, format))

    def set(self, entry: ChampionPointerEntry) -> "ChampionPointer":
        """Store *entry*, replacing any existing pointer for its key."""

        self.entries[entry.key] = entry
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible document for the whole store."""

        return {
            "entries": {
                key: self.entries[key].to_dict() for key in sorted(self.entries)
            },
            "schema_version": self.schema_version,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return the store's canonical, sorted JSON document."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChampionPointer":
        """Restore a pointer store from its JSON document."""

        version = data.get("schema_version", CHAMPION_POINTER_SCHEMA_VERSION)
        if int(version) != CHAMPION_POINTER_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported champion pointer schema_version: {version!r}"
            )
        raw_entries = data.get("entries") or {}
        if not isinstance(raw_entries, Mapping):
            raise ValueError("champion pointer 'entries' must be an object")
        entries: dict[str, ChampionPointerEntry] = {}
        for stored_key, payload in raw_entries.items():
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"champion pointer entry {stored_key!r} must be an object"
                )
            entry = ChampionPointerEntry.from_dict(payload)
            entries[entry.key] = entry
        return cls(entries=entries, schema_version=CHAMPION_POINTER_SCHEMA_VERSION)

    @classmethod
    def load(cls, path: str | Path = CHAMPION_POINTER_PATH) -> "ChampionPointer":
        """Load a committed champion pointer store, or return an empty store."""

        pointer_path = Path(path)
        if not pointer_path.is_file():
            return cls()
        with pointer_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{pointer_path} must contain a JSON object")
        return cls.from_dict(payload)

    def save(self, path: str | Path = CHAMPION_POINTER_PATH) -> Path:
        """Write the store to *path* with deterministic formatting."""

        pointer_path = Path(path)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(self.to_json(), encoding="utf-8")
        return pointer_path


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of applying a :class:`PromotionVerdict` to the pointer store."""

    verdict: PromotionVerdict
    promoted: bool
    pointer_entry: ChampionPointerEntry | None = None
    record_path: Path | None = None
    issue_draft: Mapping[str, Any] | None = None


def compare_champion_challenger(
    champion: ReportLike | None,
    challenger: ReportLike,
) -> PromotionVerdict:
    """Return the promotion verdict for a champion-vs-challenger pair.

    Both arguments may be a :class:`GateReport`, a gate-report mapping, or a
    path to a serialized gate report. The decision is pure and derived only from
    the two reports; it never calls a live model. ``champion`` may be ``None`` to
    bootstrap a first release, in which case a ``RELEASABLE`` challenger is
    promoted and any other challenger is rejected.
    """

    challenger_payload = _payload(challenger)
    challenger_report = _gate_report(challenger_payload)
    challenger_f1 = _aggregate_f1(challenger_payload)
    key = baseline_key(
        challenger_report.family,
        challenger_report.tier,
        challenger_report.format,
    )

    if champion is None:
        releasable = challenger_report.decision == RELEASABLE
        verdict = PROMOTE if releasable else REJECT
        reason = (
            "no incumbent champion; releasable challenger bootstraps the pointer"
            if releasable
            else f"challenger gate decision is {challenger_report.decision!r}, "
            f"expected {RELEASABLE!r}"
        )
        return PromotionVerdict(
            verdict=verdict,
            key=key,
            family=challenger_report.family,
            tier=challenger_report.tier,
            format=challenger_report.format,
            champion_repo_id="",
            challenger_repo_id=challenger_report.repo_id,
            champion_repro_hash="",
            challenger_repro_hash=challenger_report.repro_hash,
            ledger=(),
            reasons=(reason,),
            ignored_f1=_ignored_f1_payload(None, challenger_f1),
        )

    champion_payload = _payload(champion)
    champion_report = _gate_report(champion_payload)
    champion_f1 = _aggregate_f1(champion_payload)

    ledger = _build_ledger(champion_report, challenger_report)
    verdict_name, reasons = _decide(challenger_report, ledger)

    return PromotionVerdict(
        verdict=verdict_name,
        key=key,
        family=challenger_report.family,
        tier=challenger_report.tier,
        format=challenger_report.format,
        champion_repo_id=champion_report.repo_id,
        challenger_repo_id=challenger_report.repo_id,
        champion_repro_hash=champion_report.repro_hash,
        challenger_repro_hash=challenger_report.repro_hash,
        ledger=ledger,
        reasons=reasons,
        ignored_f1=_ignored_f1_payload(champion_f1, challenger_f1),
    )


def write_comparison_record(
    verdict: PromotionVerdict,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Write the signed comparison record to *path* using byte-stable JSON."""

    record_path = Path(path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(verdict.to_json(indent=indent) + "\n", encoding="utf-8")
    return record_path


def build_promotion_issue_draft(verdict: PromotionVerdict) -> dict[str, Any]:
    """Return a deterministic, offline issue draft for a non-promotion verdict.

    "Opening an issue" on ``HOLD`` / ``REJECT`` is represented as a structured
    artifact rather than a network call, so the whole flow stays offline and
    reproducible. Callers may render or file it however they wish.
    """

    title = f"Champion-challenger {verdict.verdict}: {verdict.challenger_repo_id or verdict.key}"
    body_lines = [
        f"Verdict: {verdict.verdict}",
        f"Key: {verdict.key}",
        f"Champion: {verdict.champion_repo_id or '(none)'}",
        f"Challenger: {verdict.challenger_repo_id or '(none)'}",
        "",
        "Reasons:",
        *[f"- {reason}" for reason in verdict.reasons],
        "",
        f"Regressions: {', '.join(verdict.regressions) or '(none)'}",
        f"Wins: {', '.join(verdict.wins) or '(none)'}",
    ]
    return {
        "artifact": "openmed.eval.champion_challenger_issue_draft",
        "body": "\n".join(body_lines),
        "labels": ["release-gate", "champion-challenger", verdict.verdict.lower()],
        "record_hash": verdict.record_hash,
        "title": title,
        "verdict": verdict.verdict,
    }


def apply_promotion(
    verdict: PromotionVerdict,
    *,
    pointer_path: str | Path = CHAMPION_POINTER_PATH,
    record_path: str | Path | None = None,
    challenger_version: str | None = None,
) -> PromotionResult:
    """Advance the champion pointer only on ``PROMOTE``; otherwise leave it intact.

    On ``PROMOTE`` this writes the comparison record (when *record_path* is
    given) and advances the committed champion pointer to the challenger. On
    ``HOLD`` / ``REJECT`` the pointer store is not read or written at all -- the
    champion is left byte-for-byte unchanged -- and an offline issue draft is
    returned instead.
    """

    if not verdict.promoted:
        return PromotionResult(
            verdict=verdict,
            promoted=False,
            issue_draft=build_promotion_issue_draft(verdict),
        )

    written_record: Path | None = None
    if record_path is not None:
        written_record = write_comparison_record(verdict, record_path)

    pointer = ChampionPointer.load(pointer_path)
    previous = pointer.get(verdict.family, verdict.tier, verdict.format)
    entry = ChampionPointerEntry(
        family=verdict.family,
        tier=verdict.tier,
        format=verdict.format,
        champion=(
            challenger_version
            if challenger_version is not None
            else _version_hint(verdict.challenger_repo_id)
        ),
        record_hash=verdict.record_hash,
        previous_champion=previous.champion if previous is not None else None,
    )
    pointer.set(entry)
    pointer.save(pointer_path)
    return PromotionResult(
        verdict=verdict,
        promoted=True,
        pointer_entry=entry,
        record_path=written_record,
    )


def _build_ledger(
    champion: GateReport,
    challenger: GateReport,
) -> tuple[GateLedgerEntry, ...]:
    entries: list[GateLedgerEntry] = []

    # Critical-label recall, gated on the shared G7 tolerance.
    champion_recall = _float_map(champion.per_label_recall)
    challenger_recall = _float_map(challenger.per_label_recall)
    for label in sorted(CRITICAL_RECALL_LABELS):
        if label not in champion_recall or label not in challenger_recall:
            continue
        entries.append(
            _numeric_entry(
                axis=f"per_label_recall.{label}",
                category=CATEGORY_RECALL,
                direction=HIGHER_IS_BETTER,
                champion=champion_recall[label],
                challenger=challenger_recall[label],
                tolerance=G7_RECALL_DROP_LIMIT,
            )
        )

    # Residual leakage and the zero-tolerance critical leakage count.
    entries.append(
        _numeric_entry(
            axis="residual_leakage_rate",
            category=CATEGORY_LEAKAGE,
            direction=LOWER_IS_BETTER,
            champion=float(champion.residual_leakage_rate),
            challenger=float(challenger.residual_leakage_rate),
        )
    )
    entries.append(
        _numeric_entry(
            axis="critical_leakage_count",
            category=CATEGORY_LEAKAGE,
            direction=LOWER_IS_BETTER,
            champion=float(champion.critical_leakage_count),
            challenger=float(challenger.critical_leakage_count),
        )
    )

    # Quantization recall delta (recall fidelity under quantization).
    quant_entry = _optional_numeric_entry(
        axis="quant_recall_delta",
        category=CATEGORY_QUANT,
        direction=LOWER_IS_BETTER,
        champion=champion.quant_recall_delta,
        challenger=challenger.quant_recall_delta,
    )
    if quant_entry is not None:
        entries.append(quant_entry)

    # Tier-fit latency and memory.
    for axis, champion_value, challenger_value in (
        ("p50_ms", champion.p50_ms, challenger.p50_ms),
        ("p95_ms", champion.p95_ms, challenger.p95_ms),
        ("ram_mb", champion.ram_mb, challenger.ram_mb),
    ):
        latency_entry = _optional_numeric_entry(
            axis=axis,
            category=CATEGORY_LATENCY,
            direction=LOWER_IS_BETTER,
            champion=champion_value,
            challenger=challenger_value,
        )
        if latency_entry is not None:
            entries.append(latency_entry)

    # Named gate pass/fail transitions. A gate the champion passed that the
    # challenger now fails is a blocking regression; the reverse is a win.
    champion_gates = {check.gate: bool(check.passed) for check in champion.gate_results}
    challenger_gates = {
        check.gate: bool(check.passed) for check in challenger.gate_results
    }
    for gate in sorted(set(champion_gates) & set(challenger_gates)):
        entries.append(
            _numeric_entry(
                axis=f"gate.{gate}",
                category=CATEGORY_GATE,
                direction=HIGHER_IS_BETTER,
                champion=1.0 if champion_gates[gate] else 0.0,
                challenger=1.0 if challenger_gates[gate] else 0.0,
            )
        )

    return tuple(entries)


def _decide(
    challenger: GateReport,
    ledger: Sequence[GateLedgerEntry],
) -> tuple[str, tuple[str, ...]]:
    releasable = challenger.decision == RELEASABLE
    blocking = [entry for entry in ledger if entry.blocking]
    wins = [entry for entry in ledger if entry.outcome == WIN]
    soft = [entry for entry in ledger if entry.soft_safety_regression]

    reasons: list[str] = []
    if not releasable:
        reasons.append(
            f"challenger gate decision is {challenger.decision!r}, "
            f"expected {RELEASABLE!r} (fails a G1a-G8 floor)"
        )
    for entry in blocking:
        reasons.append(
            f"blocking safety regression on {entry.axis}: "
            f"champion={entry.champion} challenger={entry.challenger}"
        )

    if not releasable or blocking:
        return REJECT, tuple(sorted(reasons))

    if not wins:
        reasons.append(
            "no safety-critical improvement over champion; aggregate F1 is "
            "ignored as a promotion reason"
        )
        return HOLD, tuple(sorted(reasons))

    if soft:
        for entry in soft:
            reasons.append(
                f"safety-critical win offset by regression on {entry.axis}: "
                f"champion={entry.champion} challenger={entry.challenger}"
            )
        return HOLD, tuple(sorted(reasons))

    for entry in wins:
        reasons.append(
            f"safety-critical win on {entry.axis}: "
            f"champion={entry.champion} challenger={entry.challenger}"
        )
    return PROMOTE, tuple(sorted(reasons))


def _numeric_entry(
    *,
    axis: str,
    category: str,
    direction: str,
    champion: float,
    challenger: float,
    tolerance: float = 0.0,
) -> GateLedgerEntry:
    delta = challenger - champion
    effect = delta if direction == HIGHER_IS_BETTER else -delta
    if math.isclose(champion, challenger, abs_tol=_ABS_TOL) or abs(effect) <= tolerance:
        outcome = TIE
    else:
        outcome = WIN if effect > 0 else REGRESSION
    return GateLedgerEntry(
        axis=axis,
        category=category,
        direction=direction,
        champion=champion,
        challenger=challenger,
        delta=delta,
        outcome=outcome,
        tolerance=tolerance,
    )


def _optional_numeric_entry(
    *,
    axis: str,
    category: str,
    direction: str,
    champion: float | None,
    challenger: float | None,
    tolerance: float = 0.0,
) -> GateLedgerEntry | None:
    if champion is None or challenger is None:
        return None
    return _numeric_entry(
        axis=axis,
        category=category,
        direction=direction,
        champion=float(champion),
        challenger=float(challenger),
        tolerance=tolerance,
    )


def _aggregate_f1(payload: Mapping[str, Any]) -> dict[str, float]:
    """Extract aggregate-F1 metrics for transparency; never used in the verdict."""

    metrics = payload.get("metrics")
    found: dict[str, float] = {}
    if isinstance(metrics, Mapping):
        for name, value in metrics.items():
            normalized = str(name).lower().replace("-", "_")
            if normalized in _AGGREGATE_F1_MARKERS and _is_number(value):
                found[str(name)] = float(value)
    return found


def _ignored_f1_payload(
    champion_f1: Mapping[str, float] | None,
    challenger_f1: Mapping[str, float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "challenger": {name: challenger_f1[name] for name in sorted(challenger_f1)},
        "note": "aggregate F1 is captured for transparency and never used as a promotion reason",
    }
    if champion_f1 is not None:
        payload["champion"] = {name: champion_f1[name] for name in sorted(champion_f1)}
    return payload


def _version_hint(repo_id: str) -> str:
    for token in reversed(repo_id.replace("/", "-").split("-")):
        if token and token[0] in "vV" and token[1:].isdigit():
            return token.lower()
    return repo_id


def _payload(report: ReportLike) -> dict[str, Any]:
    if isinstance(report, GateReport):
        return report.to_dict()
    if isinstance(report, (str, Path)):
        path = Path(report)
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"champion-challenger report must be an object: {path}")
        return dict(loaded)
    if isinstance(report, Mapping):
        return dict(report)
    raise TypeError(f"unsupported champion-challenger report: {type(report).__name__}")


def _gate_report(payload: Mapping[str, Any]) -> GateReport:
    return GateReport.from_dict(payload)


def _float_map(value: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(value[key]) for key in value if _is_number(value[key])}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _json_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_plain(item) for item in value]
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the champion-challenger promotion command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Decide whether a challenger gate report may promote past the "
            "incumbent champion. The verdict is derived from the two committed "
            "gate reports with no live model call."
        )
    )
    parser.add_argument(
        "--champion",
        help="Path to the champion (-vN) signed GateReport JSON payload.",
    )
    parser.add_argument(
        "--challenger",
        required=True,
        help="Path to the challenger candidate signed GateReport JSON payload.",
    )
    parser.add_argument(
        "--record",
        help="Path to write the signed comparison record JSON (written on PROMOTE).",
    )
    parser.add_argument(
        "--pointer",
        default=str(CHAMPION_POINTER_PATH),
        help="Committed champion pointer store to advance on PROMOTE.",
    )
    parser.add_argument(
        "--challenger-version",
        help="Opaque champion pointer version recorded on PROMOTE (e.g. v3).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Advance the champion pointer on PROMOTE (default: dry run).",
    )
    parser.add_argument(
        "--issue-output",
        help="Optional path to write the offline issue draft on HOLD/REJECT.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the champion-challenger promotion CLI over committed inputs only."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        verdict = compare_champion_challenger(args.champion, args.challenger)
    except (OSError, TypeError, ValueError) as exc:
        print(f"champion-challenger comparison failed: {exc}", file=sys.stderr)
        return 2

    print(verdict.to_json())

    if verdict.promoted:
        if args.apply:
            apply_promotion(
                verdict,
                pointer_path=args.pointer,
                record_path=args.record,
                challenger_version=args.challenger_version,
            )
        elif args.record is not None:
            write_comparison_record(verdict, args.record)
        return 0

    if args.issue_output is not None:
        draft = build_promotion_issue_draft(verdict)
        Path(args.issue_output).write_text(
            json.dumps(draft, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 1


__all__ = [
    "CATEGORY_GATE",
    "CATEGORY_LATENCY",
    "CATEGORY_LEAKAGE",
    "CATEGORY_QUANT",
    "CATEGORY_RECALL",
    "CHAMPION_POINTER_PATH",
    "CHAMPION_POINTER_SCHEMA_VERSION",
    "COMPARISON_RECORD_ARTIFACT",
    "COMPARISON_RECORD_SCHEMA_VERSION",
    "CRITICAL_RECALL_LABELS",
    "HIGHER_IS_BETTER",
    "HOLD",
    "LOWER_IS_BETTER",
    "PROMOTE",
    "REGRESSION",
    "REJECT",
    "TIE",
    "WIN",
    "ChampionPointer",
    "ChampionPointerEntry",
    "GateLedgerEntry",
    "PromotionResult",
    "PromotionVerdict",
    "apply_promotion",
    "build_arg_parser",
    "build_promotion_issue_draft",
    "compare_champion_challenger",
    "main",
    "write_comparison_record",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
