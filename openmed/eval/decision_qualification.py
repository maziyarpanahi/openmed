"""Offline qualification of caller-selected fixed-option decision backends.

Thresholds are fit per decision mode on a caller-supplied development split
only, then frozen before a separate evaluation split is scored. Reports hold
counts, metrics, controlled reason codes and SHA-256 digests; they never hold
input text, option values, record identifiers or file paths. A qualified report
attests only to the supplied files and policy. It is not clinical validation,
and emitted calibration profiles are never registered as defaults.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from openmed.structured.decision import (
    DECISION_SCHEMA_VERSION,
    DecisionBackend,
    DecisionBackendIdentity,
    DecisionCalibrationExample,
    DecisionCalibrationProfile,
    DecisionConflictError,
    DecisionError,
    DecisionMode,
    DecisionRequest,
    DecisionResult,
    DecisionState,
    DecisionUnsupportedError,
    decide,
    evaluate_decision_calibration,
)

DECISION_QUALIFICATION_SCHEMA_VERSION: Final = "1.0.0"
DECISION_QUALIFICATION_METHOD: Final = "development_max_coverage"
QUALIFIABLE_DECISION_MODES: Final = (
    DecisionMode.FIXED_CHOICE,
    DecisionMode.BOOLEAN_CHOICE,
    DecisionMode.ORDERED_PREFERENCE,
    DecisionMode.MULTI_LABEL,
)
MAX_SPLIT_BYTES: Final = 64 * 1024 * 1024
MAX_SPLIT_RECORDS: Final = 50_000
_MAX_CANDIDATES: Final = 256
_REQUIRED_FIELDS: Final = frozenset(
    {"mode", "input_text", "options", "expected_choices"}
)
_RECORD_FIELDS: Final = _REQUIRED_FIELDS | {"id"}
# Leave room for the longest ``.<mode>`` suffix inside the 128-character limit.
_CALIBRATION_ID_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,107}$")
_VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_DATA_REASON_MARKERS: Final = (
    "split_overlap",
    "_missing_",
    "_insufficient_option_support_",
)


class DecisionQualificationError(DecisionError):
    """Value-free qualification failure carrying a controlled reason code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class DecisionQualificationPolicy:
    """Caller-selected support and operating floors, not clinical norms.

    Attributes:
        min_examples_per_option: Minimum evaluation examples whose expected
            choices include each declared option, counted per decision mode.
            Below it, no profile is emitted.
        accuracy_floor: Minimum selective accuracy on covered examples. The
            development split selects thresholds against it and the
            evaluation split must meet it at the frozen thresholds.
        coverage_floor: Minimum evaluation coverage at the frozen thresholds.
    """

    min_examples_per_option: int = 20
    accuracy_floor: float = 0.9
    coverage_floor: float = 0.5

    def __post_init__(self) -> None:
        if (
            type(self.min_examples_per_option) is not int
            or self.min_examples_per_option < 1
        ):
            raise DecisionQualificationError("invalid_policy")
        for value in (self.accuracy_floor, self.coverage_floor):
            if type(value) not in (float, int) or not 0 <= value <= 1:
                raise DecisionQualificationError("invalid_policy")


@dataclass(frozen=True, slots=True)
class DecisionQualificationReport:
    """Immutable value-free report; profiles are present only when qualified."""

    _payload: str = field(repr=False)

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest bound to the whole report payload."""
        return "sha256:" + hashlib.sha256(self._payload.encode("utf-8")).hexdigest()

    @property
    def status(self) -> str:
        """Return ``qualified``, ``insufficient`` or ``not_qualified``."""
        return str(json.loads(self._payload)["status"])

    @property
    def qualified(self) -> bool:
        """Return whether every policy check passed on the supplied data."""
        return bool(json.loads(self._payload)["qualified"])

    def profiles(self) -> dict[str, DecisionCalibrationProfile]:
        """Return emitted profiles keyed by calibration ID for ``decide``.

        Returns:
            One profile per qualified mode, or an empty mapping when the report
            did not qualify.
        """
        return {
            key: DecisionCalibrationProfile(**value)
            for key, value in json.loads(self._payload)["profiles"].items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive copy including the report digest."""
        return {**json.loads(self._payload), "report_digest": self.digest}

    def to_json(self) -> str:
        """Serialize counts, metrics, codes and digests as canonical JSON."""
        return _canonical(self.to_dict())


@dataclass(frozen=True)
class _Split:
    digest: str
    examples: tuple[DecisionCalibrationExample, ...] = field(repr=False)
    keys: frozenset[str] = field(repr=False)
    ids: frozenset[str] = field(repr=False)


class _ReplayBackend:
    """Replay recorded outcomes so each record reaches the backend once."""

    def __init__(
        self, identity: DecisionBackendIdentity, outcomes: Mapping[Any, Any]
    ) -> None:
        self.identity = identity
        self._outcomes = outcomes

    def score(self, request: DecisionRequest) -> Any:
        outcome = self._outcomes[request]
        if isinstance(outcome, type) and issubclass(outcome, BaseException):
            raise outcome("recorded backend outcome")
        return outcome


def fit_decision_thresholds(
    examples: Sequence[DecisionCalibrationExample],
    *,
    backend: DecisionBackend,
    policy: DecisionQualificationPolicy = DecisionQualificationPolicy(),
) -> dict[str, dict[str, Any]]:
    """Fit per-mode abstention thresholds on development examples only.

    Each example is scored once. Candidate thresholds come only from the
    supplied examples' confidences, margins and option scores (plus zero). The
    selected point maximizes coverage subject to ``policy.accuracy_floor``;
    ties prefer higher accuracy, then stricter thresholds.

    Args:
        examples: Development examples. All requests of one mode must share
            the calibration ID that mode's profile will use.
        backend: Explicit local decision backend; it is never stored.
        policy: Accuracy floor used for selection.

    Returns:
        Mapping from mode to the selected thresholds (``None`` when no
        candidate meets the floor), the number of candidates evaluated, the
        backend-error count and development metrics at the selection.

    Raises:
        DecisionQualificationError: If the backend, policy or examples are
            invalid.
    """
    _require_backend(backend)
    if not isinstance(policy, DecisionQualificationPolicy):
        raise DecisionQualificationError("invalid_policy")
    groups = _by_mode(examples)
    if not groups:
        raise DecisionQualificationError("empty_split")
    digest = _examples_digest(examples)
    replay = _ReplayBackend(backend.identity, _record(backend, examples))
    fits: dict[str, dict[str, Any]] = {}
    for mode, group in groups.items():
        calibration_id = group[0].request.calibration_id
        probe = DecisionCalibrationProfile(
            calibration_id=calibration_id,
            method="development_probe",
            minimum_confidence=0.0,
            minimum_margin=0.0,
            multi_label_threshold=0.0,
            in_domain_digest=digest,
            out_of_domain_digest=digest,
        )
        # Zero thresholds expose every raw score through the shared contract.
        results = [
            decide(
                example.request,
                backend=replay,
                calibration_profiles={calibration_id: probe},
            )
            for example in group
        ]
        if mode is DecisionMode.MULTI_LABEL:
            thresholds, candidates = _fit_multi_label(group, results, policy)
        else:
            thresholds, candidates = _fit_single_choice(group, results, policy)
        development = None
        if thresholds is not None:
            profile = _profile(
                calibration_id,
                "1.0.0",
                thresholds,
                in_domain=digest,
                out_of_domain=digest,
            )
            development, _ = _assess(group, replay, profile)
        fits[mode.value] = {
            "thresholds": thresholds,
            "candidates": candidates,
            "backend_errors": sum(
                result.state is not DecisionState.SUCCESS for result in results
            ),
            "development": development,
        }
    return fits


def qualify_decision_backend(
    backend: DecisionBackend,
    *,
    development: str | Path,
    evaluation: str | Path,
    calibration_id: str,
    calibration_version: str = "1.0.0",
    policy: DecisionQualificationPolicy = DecisionQualificationPolicy(),
) -> DecisionQualificationReport:
    """Fit thresholds on development data and report held-out evaluation.

    Each JSONL line holds ``mode``, ``input_text``, ``options`` and
    ``expected_choices`` plus an optional opaque ``id``. Thresholds are frozen
    before any evaluation record is scored. Profiles are emitted only when
    every policy check passes, and are returned to the caller, never
    registered.

    Args:
        backend: Explicit local decision backend. There is no default.
        development: Caller-owned JSONL used only for threshold fitting.
        evaluation: Separate caller-owned JSONL used only for the report.
        calibration_id: Controlled prefix; each mode's profile is named
            ``<calibration_id>.<mode>``. The ``openmed`` namespace is reserved.
        calibration_version: Semantic version recorded in emitted profiles.
        policy: Support, accuracy and coverage floors.

    Returns:
        Value-free report with input digests, per-mode metrics, per-option
        support and recall keyed by option digest, and controlled reasons.

    Raises:
        DecisionQualificationError: If inputs are unreadable or malformed. The
            error carries only a controlled code.
    """
    _require_backend(backend)
    if not isinstance(policy, DecisionQualificationPolicy):
        raise DecisionQualificationError("invalid_policy")
    prefix = _calibration_prefix(calibration_id)
    if (
        not isinstance(calibration_version, str)
        or _VERSION_RE.fullmatch(calibration_version) is None
    ):
        raise DecisionQualificationError("invalid_calibration_version")
    dev = _read_split(development, prefix)
    held_out = _read_split(evaluation, prefix)
    dev_groups, eval_groups = _by_mode(dev.examples), _by_mode(held_out.examples)

    reasons: list[str] = []
    if dev.keys & held_out.keys or dev.ids & held_out.ids:
        reasons.append("split_overlap")
    for mode in QUALIFIABLE_DECISION_MODES:
        if mode in dev_groups and mode not in eval_groups:
            reasons.append(f"evaluation_missing_{mode.value}")
        if mode in eval_groups and mode not in dev_groups:
            reasons.append(f"development_missing_{mode.value}")

    fits = fit_decision_thresholds(dev.examples, backend=backend, policy=policy)
    # Thresholds are fixed here; evaluation records are scored only afterwards.
    frozen = {mode: fit["thresholds"] for mode, fit in fits.items()}
    replay = _ReplayBackend(
        backend.identity,
        _record(
            backend,
            [
                example
                for example in held_out.examples
                if frozen.get(example.request.mode.value) is not None
            ],
        ),
    )

    modes: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    for mode in QUALIFIABLE_DECISION_MODES:
        dev_group = dev_groups.get(mode, ())
        eval_group = eval_groups.get(mode, ())
        if not dev_group and not eval_group:
            continue
        fit = fits.get(mode.value)
        thresholds = frozen.get(mode.value)
        if fit is not None and fit["backend_errors"]:
            reasons.append(f"development_backend_errors_{mode.value}")
        if fit is not None and thresholds is None:
            reasons.append(f"development_accuracy_floor_unmet_{mode.value}")
        evaluated = None
        results: tuple[DecisionResult, ...] = ()
        if thresholds is not None and eval_group:
            profile = _profile(
                f"{prefix}.{mode.value}",
                calibration_version,
                thresholds,
                in_domain=dev.digest,
                out_of_domain=held_out.digest,
            )
            evaluated, results = _assess(eval_group, replay, profile)
            profiles[profile.calibration_id] = profile.to_dict()
        options = _option_report(dev_group, eval_group, results, policy)
        if eval_group and not all(item["sufficient"] for item in options):
            reasons.append(f"evaluation_insufficient_option_support_{mode.value}")
        if evaluated is not None:
            if set(evaluated["states"]) - {"success", "abstained"}:
                reasons.append(f"evaluation_backend_errors_{mode.value}")
            if (
                not evaluated["covered"]
                or evaluated["accuracy"] < policy.accuracy_floor
            ):
                reasons.append(f"evaluation_accuracy_floor_unmet_{mode.value}")
            if evaluated["coverage"] < policy.coverage_floor:
                reasons.append(f"evaluation_coverage_floor_unmet_{mode.value}")
        modes[mode.value] = {
            "development": fit["development"] if fit is not None else None,
            "evaluation": evaluated,
            "options": options,
        }

    qualified = not reasons
    if qualified:
        status = "qualified"
    elif any(marker in reason for reason in reasons for marker in _DATA_REASON_MARKERS):
        status = "insufficient"
    else:
        status = "not_qualified"
    payload = {
        "schema_version": DECISION_QUALIFICATION_SCHEMA_VERSION,
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "status": status,
        "qualified": qualified,
        "reasons": reasons,
        "human_review_required": True,
        "autonomous_action": False,
        "contains_source_text": False,
        "backend": backend.identity.to_dict(),
        "policy": asdict(policy),
        "inputs": {
            "development": {"digest": dev.digest, "records": len(dev.examples)},
            "evaluation": {
                "digest": held_out.digest,
                "records": len(held_out.examples),
            },
        },
        "fit": {
            "split": "development",
            "input_digest": dev.digest,
            "method": DECISION_QUALIFICATION_METHOD,
            "modes": {
                mode: {
                    key: fit[key]
                    for key in ("thresholds", "candidates", "backend_errors")
                }
                for mode, fit in fits.items()
            },
        },
        "modes": modes,
        "profiles": profiles if qualified else {},
    }
    return DecisionQualificationReport(_canonical(payload))


def _require_backend(backend: Any) -> None:
    if not isinstance(backend, DecisionBackend) or not isinstance(
        getattr(backend, "identity", None), DecisionBackendIdentity
    ):
        raise DecisionQualificationError("backend_invalid")


def _calibration_prefix(value: Any) -> str:
    if not isinstance(value, str) or _CALIBRATION_ID_RE.fullmatch(value) is None:
        raise DecisionQualificationError("invalid_calibration_id")
    if re.split(r"[.:/]", value, maxsplit=1)[0] == "openmed":
        # Bundled profiles own this namespace; caller fits never shadow them.
        raise DecisionQualificationError("reserved_calibration_id")
    return value


def _read_split(path: str | Path, prefix: str) -> _Split:
    try:
        with Path(path).open("rb") as handle:
            data = handle.read(MAX_SPLIT_BYTES + 1)
    except (OSError, TypeError, ValueError):
        raise DecisionQualificationError("input_unavailable") from None
    if len(data) > MAX_SPLIT_BYTES:
        raise DecisionQualificationError("split_too_large")
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise DecisionQualificationError("invalid_record") from None
    examples: list[DecisionCalibrationExample] = []
    keys: set[str] = set()
    ids: set[str] = set()
    for line in lines:
        if not line.strip():
            continue
        if len(examples) >= MAX_SPLIT_RECORDS:
            raise DecisionQualificationError("split_too_large")
        example, identifier = _parse_record(line, prefix)
        if identifier is not None:
            if identifier in ids:
                raise DecisionQualificationError("duplicate_id")
            ids.add(identifier)
        request = example.request
        keys.add(
            _digest(
                _canonical(
                    [
                        request.mode.value,
                        _normalize(request.input_text),
                        sorted(_normalize(option) for option in request.options),
                    ]
                )
            )
        )
        examples.append(example)
    if not examples:
        raise DecisionQualificationError("empty_split")
    return _Split(
        digest=_digest(data),
        examples=tuple(examples),
        keys=frozenset(keys),
        ids=frozenset(ids),
    )


def _parse_record(
    line: str, prefix: str
) -> tuple[DecisionCalibrationExample, str | None]:
    try:
        record = json.loads(line)
        if (
            type(record) is not dict
            or not _REQUIRED_FIELDS <= set(record)
            or not set(record) <= _RECORD_FIELDS
        ):
            raise ValueError
        mode = DecisionMode(record["mode"])
    except (TypeError, ValueError):
        raise DecisionQualificationError("invalid_record") from None
    if mode not in QUALIFIABLE_DECISION_MODES:
        raise DecisionQualificationError("unsupported_mode")
    try:
        options, expected = record["options"], record["expected_choices"]
        if type(options) is not list or type(expected) is not list:
            raise TypeError
        if not all(isinstance(item, str) for item in expected):
            raise TypeError
        choices = tuple(item.strip() for item in expected)
        if len(set(choices)) != len(choices) or (
            mode is not DecisionMode.MULTI_LABEL and len(choices) != 1
        ):
            raise ValueError
        identifier = record.get("id")
        if identifier is not None and (
            not isinstance(identifier, str) or not identifier.strip()
        ):
            raise ValueError
        example = DecisionCalibrationExample(
            slice_name=mode.value,
            request=DecisionRequest(
                mode=mode,
                input_text=record["input_text"],
                options=tuple(options),
                calibration_id=f"{prefix}.{mode.value}",
            ),
            expected_choices=choices,
        )
    except (DecisionError, TypeError, ValueError):
        raise DecisionQualificationError("invalid_record") from None
    return example, identifier


def _by_mode(
    examples: Sequence[DecisionCalibrationExample],
) -> dict[DecisionMode, tuple[DecisionCalibrationExample, ...]]:
    groups: dict[DecisionMode, list[DecisionCalibrationExample]] = {}
    for example in examples:
        if not isinstance(example, DecisionCalibrationExample):
            raise DecisionQualificationError("invalid_record")
        if example.request.mode not in QUALIFIABLE_DECISION_MODES:
            raise DecisionQualificationError("unsupported_mode")
        groups.setdefault(example.request.mode, []).append(example)
    for group in groups.values():
        if len({example.request.calibration_id for example in group}) != 1:
            raise DecisionQualificationError("calibration_id_mismatch")
    return {
        mode: tuple(groups[mode])
        for mode in QUALIFIABLE_DECISION_MODES
        if mode in groups
    }


def _record(
    backend: DecisionBackend, examples: Sequence[DecisionCalibrationExample]
) -> dict[DecisionRequest, Any]:
    outcomes: dict[DecisionRequest, Any] = {}
    for example in examples:
        request = example.request
        if request in outcomes:
            continue
        started = time.monotonic()
        try:
            outcome = backend.score(request)
        except DecisionUnsupportedError:
            outcome = DecisionUnsupportedError
        except DecisionConflictError:
            outcome = DecisionConflictError
        except TimeoutError:
            outcome = TimeoutError
        except Exception:
            # Keep only the failure class; backend messages may echo inputs.
            outcome = RuntimeError
        else:
            if (time.monotonic() - started) * 1000.0 > request.timeout_ms:
                outcome = TimeoutError
        outcomes[request] = outcome
    return outcomes


def _fit_single_choice(
    group: Sequence[DecisionCalibrationExample],
    results: Sequence[DecisionResult],
    policy: DecisionQualificationPolicy,
) -> tuple[dict[str, float] | None, int]:
    # Correctness matches evaluate_decision_calibration: the selected set must
    # equal the expected set.
    rows = [
        (
            float(result.confidence),
            float(result.margin),
            {result.choice} == set(example.expected_choices),
        )
        for example, result in zip(group, results, strict=True)
        if result.state is DecisionState.SUCCESS
    ]
    confidences = _candidates([0.0, *(row[0] for row in rows)])
    margins = _candidates([0.0, *(row[1] for row in rows)])
    best: tuple[int, float, float, float] | None = None
    evaluated = 0
    for confidence in confidences:
        eligible = sorted(
            (margin, correct) for value, margin, correct in rows if value >= confidence
        )
        keys = [margin for margin, _ in eligible]
        correct_from = [0] * (len(eligible) + 1)
        for index in range(len(eligible) - 1, -1, -1):
            correct_from[index] = correct_from[index + 1] + eligible[index][1]
        for margin in margins:
            if margin > confidence:
                break
            evaluated += 1
            start = bisect.bisect_left(keys, margin)
            covered = len(eligible) - start
            if not covered:
                continue
            accuracy = correct_from[start] / covered
            point = (covered, accuracy, confidence, margin)
            if accuracy >= policy.accuracy_floor and (best is None or point > best):
                best = point
    if best is None:
        return None, evaluated
    return {"minimum_confidence": best[2], "minimum_margin": best[3]}, evaluated


def _fit_multi_label(
    group: Sequence[DecisionCalibrationExample],
    results: Sequence[DecisionResult],
    policy: DecisionQualificationPolicy,
) -> tuple[dict[str, float] | None, int]:
    rows = []
    observed = [0.0]
    for example, result in zip(group, results, strict=True):
        if result.state is not DecisionState.SUCCESS:
            continue
        expected = set(example.expected_choices)
        scores = [
            (item["option"], float(item["score"])) for item in result.option_scores
        ]
        observed.extend(score for _, score in scores)
        # A threshold t selects exactly the expected set iff lowest < t <= highest.
        lowest = max(
            (score for option, score in scores if option not in expected), default=-1.0
        )
        highest = min(score for option, score in scores if option in expected)
        rows.append((max(score for _, score in scores), lowest, highest))
    candidates = _candidates(observed)
    best: tuple[int, float, float] | None = None
    for threshold in candidates:
        covered = sum(top >= threshold for top, _, _ in rows)
        if not covered:
            continue
        correct = sum(low < threshold <= high for _, low, high in rows)
        accuracy = correct / covered
        point = (covered, accuracy, threshold)
        if accuracy >= policy.accuracy_floor and (best is None or point > best):
            best = point
    if best is None:
        return None, len(candidates)
    return {"multi_label_threshold": best[2]}, len(candidates)


def _candidates(values: Sequence[float]) -> list[float]:
    unique = sorted(set(values))
    if len(unique) <= _MAX_CANDIDATES:
        return unique
    last = len(unique) - 1
    return sorted(
        {
            unique[round(index * last / (_MAX_CANDIDATES - 1))]
            for index in range(_MAX_CANDIDATES)
        }
    )


def _assess(
    group: Sequence[DecisionCalibrationExample],
    replay: _ReplayBackend,
    profile: DecisionCalibrationProfile,
) -> tuple[dict[str, Any], tuple[DecisionResult, ...]]:
    mode = group[0].request.mode.value
    examples = [
        DecisionCalibrationExample(
            slice_name=mode,
            request=example.request,
            expected_choices=example.expected_choices,
        )
        for example in group
    ]
    (summary,) = evaluate_decision_calibration(
        examples, backend=replay, profile=profile
    )["slices"]
    results = tuple(
        decide(
            example.request,
            backend=replay,
            calibration_profiles={profile.calibration_id: profile},
        )
        for example in examples
    )
    states = Counter(result.state.value for result in results)
    return {
        "total": summary["total"],
        "covered": summary["covered"],
        "coverage": summary["coverage"],
        "accuracy": summary["accuracy"],
        "expected_calibration_error": summary["expected_calibration_error"],
        "abstention_rate": states[DecisionState.ABSTAINED.value] / summary["total"],
        "states": dict(sorted(states.items())),
    }, results


def _option_report(
    dev_group: Sequence[DecisionCalibrationExample],
    eval_group: Sequence[DecisionCalibrationExample],
    results: Sequence[DecisionResult],
    policy: DecisionQualificationPolicy,
) -> list[dict[str, Any]]:
    declared = {
        _normalize(option)
        for example in (*dev_group, *eval_group)
        for option in example.request.options
    }
    dev_support = Counter(
        _normalize(choice)
        for example in dev_group
        for choice in example.expected_choices
    )
    eval_support = Counter(
        _normalize(choice)
        for example in eval_group
        for choice in example.expected_choices
    )
    recalled: Counter[str] = Counter()
    for example, result in zip(eval_group, results):
        selected = (
            {_normalize(choice) for choice in result.choices}
            if result.state is DecisionState.SUCCESS
            else set()
        )
        recalled.update(
            _normalize(choice)
            for choice in example.expected_choices
            if _normalize(choice) in selected
        )
    entries = [
        {
            "option_digest": _digest(option),
            "development_examples": dev_support[option],
            "evaluation_examples": eval_support[option],
            "evaluation_recall": (
                recalled[option] / eval_support[option]
                if results and eval_support[option]
                else None
            ),
            "sufficient": eval_support[option] >= policy.min_examples_per_option,
        }
        for option in declared
    ]
    return sorted(entries, key=lambda entry: entry["option_digest"])


def _profile(
    calibration_id: str,
    version: str,
    thresholds: Mapping[str, float],
    *,
    in_domain: str,
    out_of_domain: str,
) -> DecisionCalibrationProfile:
    # Thresholds for other modes are 1.0 so reuse in another mode fails closed.
    return DecisionCalibrationProfile(
        calibration_id=calibration_id,
        version=version,
        method=DECISION_QUALIFICATION_METHOD,
        minimum_confidence=thresholds.get("minimum_confidence", 1.0),
        minimum_margin=thresholds.get("minimum_margin", 1.0),
        multi_label_threshold=thresholds.get("multi_label_threshold", 1.0),
        in_domain_digest=in_domain,
        out_of_domain_digest=out_of_domain,
    )


def _examples_digest(examples: Sequence[DecisionCalibrationExample]) -> str:
    return _digest(
        _canonical(
            [
                [example.request.to_dict(), list(example.expected_choices)]
                for example in examples
            ]
        )
    )


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(data).hexdigest()


__all__ = [
    "DECISION_QUALIFICATION_METHOD",
    "DECISION_QUALIFICATION_SCHEMA_VERSION",
    "MAX_SPLIT_BYTES",
    "MAX_SPLIT_RECORDS",
    "QUALIFIABLE_DECISION_MODES",
    "DecisionQualificationError",
    "DecisionQualificationPolicy",
    "DecisionQualificationReport",
    "fit_decision_thresholds",
    "qualify_decision_backend",
]
