"""Offline SDOH extraction scoring with aggregate-only reports.

Synthetic gold is repository authored. SHAC is read only from a credentialed
local path, held in memory, and never copied into benchmark artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from openmed.clinical.sdoh import SDOHFinding, extract_sdoh
from openmed.core.offline import network_blocked_if_offline
from openmed.eval.datasets.dua_stubs import DUACredentialRequired
from openmed.eval.datasets.shac import SHAC_PATH_ENV, load_shac
from openmed.eval.report import BenchmarkReport

SDOH_EXTRACTION = "sdoh-extraction"
SHAC_SDOH_EXTRACTION = "shac-sdoh-extraction"
SDOH_GOLD_PATH = (
    Path(__file__).parents[1] / "fixtures" / "sdoh_extraction_synthetic.json"
)
SDOH_REPORT_PATH = (
    Path(__file__).parents[1] / "fixtures" / "sdoh_extraction_synthetic_report.json"
)
SDOH_DETERMINANTS = (
    "alcohol",
    "drug",
    "employment",
    "food_insecurity",
    "living_status",
    "tobacco",
)
_STATUSES = {
    "alcohol": frozenset({"current", "past", "none", "unknown"}),
    "drug": frozenset({"current", "past", "none", "unknown"}),
    "tobacco": frozenset({"current", "past", "none", "unknown"}),
    "employment": frozenset(
        {
            "employed",
            "unemployed",
            "retired",
            "disabled",
            "student",
            "homemaker",
            "former",
            "never",
            "unknown",
        }
    ),
    "living_status": frozenset(
        {
            "lives_alone",
            "lives_with_family",
            "lives_with_others",
            "homeless",
            "housed",
            "assisted_living",
            "former",
            "never",
            "future",
            "unknown",
        }
    ),
    "food_insecurity": frozenset({"current", "past", "none", "unknown"}),
}
SDOHExtractor = Callable[[str], Iterable[SDOHFinding]]


class SDOHBenchmarkError(ValueError):
    """Fixed-code benchmark refusal without source content or exception context."""


class SDOHUnavailableReason(str, Enum):
    """Controlled reasons why a SHAC lane has no score."""

    NOT_CONFIGURED = "shac_not_configured"
    PATH_UNAVAILABLE = "shac_path_unavailable"
    GOLD_INVALID = "shac_gold_invalid"
    GOLD_UNSUPPORTED = "shac_gold_unsupported"


@dataclass(frozen=True)
class SDOHBenchmarkUnavailable:
    """An unavailable lane, with no metrics or implicit zero score."""

    reason: SDOHUnavailableReason

    def __post_init__(self) -> None:
        if not isinstance(self.reason, SDOHUnavailableReason):
            raise SDOHBenchmarkError("sdoh_unavailable_reason_invalid")

    def to_dict(self) -> dict[str, str]:
        """Return controlled availability codes only."""
        return {
            "suite": SHAC_SDOH_EXTRACTION,
            "availability": "unavailable",
            "reason": self.reason.value,
        }


@dataclass(frozen=True, order=True)
class SDOHGoldFinding:
    """One controlled determinant/status label and half-open trigger span."""

    category: str
    status: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.category, str)
            or not isinstance(self.status, str)
            or self.category not in _STATUSES
            or self.status not in _STATUSES[self.category]
        ):
            raise SDOHBenchmarkError("sdoh_label_invalid")
        if (
            any(
                isinstance(x, bool) or not isinstance(x, int)
                for x in (self.start, self.end)
            )
            or not 0 <= self.start < self.end
        ):
            raise SDOHBenchmarkError("sdoh_span_invalid")


@dataclass(frozen=True)
class SDOHBenchmarkCase:
    """An in-memory social-history window; source values are excluded from repr.

    Args:
        text: An already selected social-history window. No section detection
            or gold spans are passed to the extractor.
        gold: Independently authored trigger, determinant and status labels.
        synthetic: Whether this case is repository-authored synthetic data.
    """

    text: str = field(repr=False)
    gold: tuple[SDOHGoldFinding, ...]
    synthetic: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str)
            or not self.text
            or type(self.synthetic) is not bool
        ):
            raise SDOHBenchmarkError("sdoh_case_invalid")
        object.__setattr__(self, "gold", tuple(self.gold))
        if any(
            not isinstance(x, SDOHGoldFinding) or x.end > len(self.text)
            for x in self.gold
        ):
            raise SDOHBenchmarkError("sdoh_gold_invalid")


def _digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def load_sdoh_extraction_fixtures(
    path: str | Path | None = None,
) -> list[SDOHBenchmarkCase]:
    """Load explicitly synthetic gold, refusing malformed or restricted data.

    Args:
        path: Optional synthetic manifest path; defaults to the packaged gold.

    Returns:
        Validated synthetic cases with independently authored labels.

    Raises:
        SDOHBenchmarkError: Fixed-code refusal for an unreadable or invalid gold.
    """
    failed = False
    cases: list[SDOHBenchmarkCase] = []
    try:
        payload = json.loads(
            Path(path or SDOH_GOLD_PATH).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"schema_version", "synthetic", "contains_real_phi", "cases"}
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1
            or payload.get("synthetic") is not True
            or payload.get("contains_real_phi") is not False
        ):
            raise ValueError
        rows = payload["cases"]
        if not isinstance(rows, list) or not rows:
            raise ValueError
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"text", "gold"}
                or not isinstance(row["gold"], list)
            ):
                raise ValueError
            gold = []
            for item in row["gold"]:
                if not isinstance(item, dict) or set(item) != {
                    "category",
                    "status",
                    "start",
                    "end",
                }:
                    raise ValueError
                gold.append(SDOHGoldFinding(**item))
            cases.append(SDOHBenchmarkCase(row["text"], tuple(gold)))
    except Exception:
        failed = True
    if failed:
        raise SDOHBenchmarkError("sdoh_synthetic_gold_invalid")
    return cases


def _local_extractor(text: str) -> Iterable[SDOHFinding]:
    # Corpus records are already selected social-history windows. Supplying
    # gold candidate spans would give the extractor annotation information.
    return extract_sdoh(text, ())


def _pairs(
    gold: Sequence[SDOHGoldFinding],
    predicted: Sequence[SDOHGoldFinding],
    *,
    exact: bool = False,
    status: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Deterministic maximum-cardinality one-to-one matching, never greedy."""
    edges = []
    for item in gold:
        candidates = [
            i
            for i, other in enumerate(predicted)
            if item.category == other.category
            and (not status or item.status == other.status)
            and (
                (item.start, item.end) == (other.start, other.end)
                if exact
                else max(item.start, other.start) < min(item.end, other.end)
            )
        ]
        candidates.sort(
            key=lambda i: (
                (item.start, item.end) != (predicted[i].start, predicted[i].end),
                -(
                    min(item.end, predicted[i].end)
                    - max(item.start, predicted[i].start)
                ),
                predicted[i],
                i,
            )
        )
        edges.append(candidates)
    assigned: dict[int, int] = {}

    def augment(index: int, seen: set[int]) -> bool:
        for target in edges[index]:
            if target in seen:
                continue
            seen.add(target)
            if target not in assigned or augment(assigned[target], seen):
                assigned[target] = index
                return True
        return False

    for i in range(len(gold)):
        augment(i, set())
    return tuple(sorted((i, j) for j, i in assigned.items()))


def _counts() -> dict[str, int]:
    return {
        key: 0
        for key in (
            "gold_count",
            "prediction_count",
            "overlap_matches",
            "status_matches",
            "exact_matches",
        )
    }


def _rates(counts: Mapping[str, int]) -> dict[str, int | float | None]:
    gold, predicted = counts["gold_count"], counts["prediction_count"]
    overlap = counts["overlap_matches"]
    return {
        **counts,
        "false_negatives": gold - overlap,
        "false_positives": predicted - overlap,
        "precision": overlap / predicted if predicted else None,
        "recall": overlap / gold if gold else None,
        "status_accuracy": counts["status_matches"] / gold if gold else None,
        "exact_offset_match": counts["exact_matches"] / gold if gold else None,
    }


def sdoh_report_digest(report: BenchmarkReport) -> str:
    """Digest the canonical aggregate payload excluding its own digest field."""
    payload = report.to_dict()
    payload["metadata"].pop("report_digest", None)
    return _digest(payload)


def _score(
    cases: Sequence[SDOHBenchmarkCase],
    extractor: SDOHExtractor,
    suite: str,
    *,
    injected: bool = False,
) -> BenchmarkReport:
    if not cases:
        raise SDOHBenchmarkError("sdoh_cases_empty")
    overall = _counts()
    by_category = {key: _counts() for key in SDOH_DETERMINANTS}
    by_status = {
        key: {status: _counts() for status in sorted(_STATUSES[key])}
        for key in SDOH_DETERMINANTS
    }
    fixture_digests = []
    prediction_digests = []
    for case in cases:
        gold = sorted(case.gold)
        predicted: list[SDOHGoldFinding] = []
        failed = False
        try:
            with network_blocked_if_offline(local_only=True):
                for item in extractor(case.text):
                    if not isinstance(item, SDOHFinding):
                        raise ValueError
                    if not isinstance(item.status, str):
                        raise ValueError
                    projected = SDOHGoldFinding(
                        item.category, item.status, item.span[0], item.span[1]
                    )
                    if projected.end > len(case.text):
                        raise ValueError
                    predicted.append(projected)
        except Exception:
            failed = True
        if failed:
            raise SDOHBenchmarkError("sdoh_extractor_contract_failed")
        predicted.sort()
        overlap = _pairs(gold, predicted)
        status = _pairs(gold, predicted, status=True)
        exact = _pairs(gold, predicted, exact=True)
        overall["gold_count"] += len(gold)
        overall["prediction_count"] += len(predicted)
        overall["overlap_matches"] += len(overlap)
        overall["status_matches"] += len(status)
        overall["exact_matches"] += len(exact)
        for category in SDOH_DETERMINANTS:
            counts = by_category[category]
            counts["gold_count"] += sum(x.category == category for x in gold)
            counts["prediction_count"] += sum(x.category == category for x in predicted)
            for key, matches in (
                ("overlap_matches", overlap),
                ("status_matches", status),
                ("exact_matches", exact),
            ):
                counts[key] += sum(gold[i].category == category for i, _ in matches)
            for label, counts in by_status[category].items():
                counts["gold_count"] += sum(
                    x.category == category and x.status == label for x in gold
                )
                counts["prediction_count"] += sum(
                    x.category == category and x.status == label for x in predicted
                )
                status_count = sum(
                    gold[i].category == category and gold[i].status == label
                    for i, _ in status
                )
                counts["overlap_matches"] += status_count
                counts["status_matches"] += status_count
                counts["exact_matches"] += len(
                    _pairs(
                        [
                            x
                            for x in gold
                            if x.category == category and x.status == label
                        ],
                        [
                            x
                            for x in predicted
                            if x.category == category and x.status == label
                        ],
                        exact=True,
                    )
                )
        fixture_digest = _digest(
            {
                "text_digest": hashlib.sha256(case.text.encode("utf-8")).hexdigest(),
                "gold": [[x.category, x.status, x.start, x.end] for x in gold],
            }
        )
        fixture_digests.append(fixture_digest)
        prediction_digests.append(
            _digest(
                {
                    "fixture_digest": fixture_digest,
                    "prediction": [
                        [x.category, x.status, x.start, x.end] for x in predicted
                    ],
                }
            )
        )
    report = BenchmarkReport(
        suite=suite,
        model_name="injected-sdoh-extractor" if injected else "extract_sdoh",
        device="cpu",
        fixture_count=len(cases),
        metrics={
            "overall": _rates(overall),
            "by_determinant": {
                key: _rates(value) for key, value in by_category.items()
            },
            "by_status": {
                key: {label: _rates(value) for label, value in values.items()}
                for key, values in by_status.items()
            },
        },
        metadata={
            "schema_version": 1,
            "fixture_digest": _digest(sorted(fixture_digests)),
            "prediction_digest": _digest(sorted(prediction_digests)),
        },
    )
    return replace(
        report,
        metadata={**report.metadata, "report_digest": sdoh_report_digest(report)},
    )


def run_sdoh_extraction_benchmark(
    fixtures: Sequence[SDOHBenchmarkCase] | None = None,
    *,
    extractor: SDOHExtractor | None = None,
) -> BenchmarkReport:
    """Score synthetic SDOH gold without changing extraction or filtering misses.

    Args:
        fixtures: Optional synthetic cases; defaults to the packaged gold.
        extractor: Optional trusted callable for deterministic comparisons.

    Returns:
        A counts/rates/digests-only report with fixed suite/model/device headers.

    Raises:
        SDOHBenchmarkError: Content-free refusal on invalid inputs or predictions.
    """
    cases = list(load_sdoh_extraction_fixtures() if fixtures is None else fixtures)
    if any(
        not isinstance(case, SDOHBenchmarkCase) or not case.synthetic for case in cases
    ):
        raise SDOHBenchmarkError("sdoh_synthetic_cases_required")
    return _score(
        cases,
        _local_extractor if extractor is None else extractor,
        SDOH_EXTRACTION,
        injected=extractor is not None,
    )


def _shac_cases(path: str | Path | None) -> list[SDOHBenchmarkCase]:
    cases = []
    for fixture in load_shac(path):
        if fixture.language != "en":
            raise SDOHBenchmarkError("shac_gold_unsupported")
        # Keep distinct event IDs even when native events share a trigger. In
        # relation-only exports, the trigger span is the event grouping key.
        events: dict[tuple[Any, ...], list[Any]] = {}
        trigger_spans = set()
        for entity in fixture.entities.values():
            label = (
                str(entity.metadata.get("source_label", "")).casefold().replace("_", "")
            )
            if label in {
                "alcohol",
                "drug",
                "tobacco",
                "employment",
                "livingstatus",
                "living",
            }:
                trigger_spans.add((entity.start, entity.end))
        for relation in fixture.gold_relations:
            key = (
                relation.head.start,
                relation.head.end,
                relation.metadata.get("source_event_id", ""),
            )
            events.setdefault(key, []).append(relation)
        gold = []
        observed = set()
        for key, relations in events.items():
            head = relations[0].head
            label = (
                str(head.metadata.get("source_label", "")).casefold().replace("_", "")
            )
            category = "living_status" if label in {"livingstatus", "living"} else label
            if category not in set(SDOH_DETERMINANTS) - {
                "food_insecurity"
            } or head.metadata.get("discontinuous"):
                raise SDOHBenchmarkError("shac_gold_unsupported")
            status_arguments = [
                x.tail
                for x in relations
                if x.metadata.get("source_relation_type", "").casefold() == "status"
            ]
            type_arguments = [
                x.tail
                for x in relations
                if x.metadata.get("source_relation_type", "").casefold() == "type"
            ]
            if len(status_arguments) != 1:
                raise SDOHBenchmarkError("shac_gold_unsupported")
            argument = status_arguments[0]
            native = argument.metadata.get("source_subtype")
            expected_type = "statusemploy" if category == "employment" else "statustime"
            if (
                str(argument.metadata.get("source_label", "")).casefold()
                != expected_type
                or native is None
            ):
                raise SDOHBenchmarkError("shac_gold_unsupported")
            if category == "living_status":
                if (
                    len(type_arguments) != 1
                    or str(
                        type_arguments[0].metadata.get("source_label", "")
                    ).casefold()
                    != "typeliving"
                    or native not in {"current", "past", "future"}
                ):
                    raise SDOHBenchmarkError("shac_gold_unsupported")
                living = type_arguments[0].metadata.get("source_subtype")
                if living not in {"alone", "with_family", "with_others", "homeless"}:
                    raise SDOHBenchmarkError("shac_gold_unsupported")
                status = (
                    {
                        "alone": "lives_alone",
                        "with_family": "lives_with_family",
                        "with_others": "lives_with_others",
                        "homeless": "homeless",
                    }[living]
                    if native == "current"
                    else {"past": "former", "future": "future"}[native]
                )
            elif category == "employment":
                status = "disabled" if native == "on_disability" else native
            else:
                if native not in {"current", "past", "none"}:
                    raise SDOHBenchmarkError("shac_gold_unsupported")
                status = native
            gold.append(SDOHGoldFinding(category, status, head.start, head.end))
            observed.add((head.start, head.end))
        if trigger_spans != observed:
            raise SDOHBenchmarkError("shac_gold_unsupported")
        cases.append(SDOHBenchmarkCase(fixture.text, tuple(gold), synthetic=False))
    return cases


def run_shac_sdoh_benchmark(
    path: str | Path | None = None, *, extractor: SDOHExtractor | None = None
) -> BenchmarkReport | SDOHBenchmarkUnavailable:
    """Score native SHAC gold from an explicit credentialed local path.

    Args:
        path: User-supplied path, or ``OPENMED_SHAC_PATH`` when omitted.
        extractor: Optional trusted extractor; defaults to ``extract_sdoh``.

    Returns:
        Aggregate report, or a typed unavailable result with no metrics. Native
        status attributes are required; labels are never inferred from note text.
    """
    if path is None and not os.environ.get(SHAC_PATH_ENV, "").strip():
        return SDOHBenchmarkUnavailable(SDOHUnavailableReason.NOT_CONFIGURED)
    reason = None
    cases = []
    try:
        cases = _shac_cases(path)
    except DUACredentialRequired:
        reason = SDOHUnavailableReason.PATH_UNAVAILABLE
    except OSError:
        reason = SDOHUnavailableReason.PATH_UNAVAILABLE
    except SDOHBenchmarkError:
        reason = SDOHUnavailableReason.GOLD_UNSUPPORTED
    except Exception:
        reason = SDOHUnavailableReason.GOLD_INVALID
    if reason is not None:
        return SDOHBenchmarkUnavailable(reason)
    return _score(
        cases,
        _local_extractor if extractor is None else extractor,
        SHAC_SDOH_EXTRACTION,
        injected=extractor is not None,
    )


def sdoh_extraction_metadata() -> dict[str, Any]:
    """Return fixed discovery metadata without touching SHAC or loading models."""
    return {
        "suite": SDOH_EXTRACTION,
        "task": "sdoh_extraction",
        "synthetic": True,
        "network_fetch": False,
        "shac_eval_only": True,
        "clinical_validation": False,
    }


__all__ = [
    "SDOH_EXTRACTION",
    "SHAC_SDOH_EXTRACTION",
    "SDOH_GOLD_PATH",
    "SDOH_REPORT_PATH",
    "SDOH_DETERMINANTS",
    "SDOHBenchmarkError",
    "SDOHUnavailableReason",
    "SDOHBenchmarkUnavailable",
    "SDOHGoldFinding",
    "SDOHBenchmarkCase",
    "load_sdoh_extraction_fixtures",
    "run_sdoh_extraction_benchmark",
    "run_shac_sdoh_benchmark",
    "sdoh_report_digest",
    "sdoh_extraction_metadata",
]
