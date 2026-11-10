"""Offline adversarial-PHI corpus runner and bypass-rate gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REDTEAM_REPORT_SCHEMA_VERSION = 1
REDTEAM_THRESHOLD_ENV_VAR = "OPENMED_REDTEAM_MAX_BYPASS_RATE"
DEFAULT_REDTEAM_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
MAX_REDTEAM_CORPUS_BYTES = 5 * 1024 * 1024
MAX_REDTEAM_CASES = 1_000
MAX_REDTEAM_CASE_TEXT_CHARS = 100_000
MAX_REDTEAM_ASSERTIONS_PER_CASE = 128
MAX_REDTEAM_OUTPUT_CHARS = 1_000_000
_SOURCE_REDTEAM_CORPUS = (
    Path(__file__).resolve().parents[2]
    / "eval"
    / "redteam"
    / "corpus"
    / "adversarial_phi.jsonl"
)
_PACKAGED_REDTEAM_CORPUS = (
    Path(__file__).resolve().parent / "data" / "adversarial_phi.jsonl"
)
DEFAULT_REDTEAM_CORPUS = (
    _PACKAGED_REDTEAM_CORPUS
    if _PACKAGED_REDTEAM_CORPUS.is_file()
    else _SOURCE_REDTEAM_CORPUS
)

_ABUSE_CASE_ID = re.compile(r"^AC-\d{2}$")
_REPORT_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_LANGUAGE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
_MATCH_MODES = frozenset({"exact", "casefold", "normalized", "alnum"})
_ASSERTION_FIELDS = frozenset({"label", "value", "match"})
_CASE_FIELDS = frozenset(
    {
        "id",
        "abuse_case_id",
        "attack_type",
        "text",
        "expected_protected",
        "language",
        "synthetic",
    }
)


class RedTeamCorpusError(ValueError):
    """Raised when an adversarial corpus violates its fail-closed schema."""


@dataclass(frozen=True)
class ProtectedAssertion:
    """A synthetic surface that must not survive de-identification."""

    label: str
    value: str
    match: str = "normalized"

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        case_id: str,
        index: int,
    ) -> ProtectedAssertion:
        """Validate and build one expected-protected assertion."""

        if set(data) - _ASSERTION_FIELDS:
            raise RedTeamCorpusError(
                f"case {case_id!r} assertion {index} contains unsupported fields"
            )

        label = data.get("label")
        value = data.get("value")
        match = data.get("match", "normalized")
        prefix = f"case {case_id!r} assertion {index}"
        if not isinstance(label, str) or not label.strip():
            raise RedTeamCorpusError(f"{prefix} requires a non-empty label")
        if not isinstance(value, str) or not value:
            raise RedTeamCorpusError(f"{prefix} requires a non-empty value")
        if len(value) > MAX_REDTEAM_CASE_TEXT_CHARS:
            raise RedTeamCorpusError(f"{prefix} value is too long")
        if not isinstance(match, str) or match not in _MATCH_MODES:
            allowed = ", ".join(sorted(_MATCH_MODES))
            raise RedTeamCorpusError(f"{prefix} match must be one of: {allowed}")
        if match in {"normalized", "alnum"}:
            comparison_value = _normalized_surface(value)
            if match == "alnum":
                comparison_value = "".join(
                    character for character in comparison_value if character.isalnum()
                )
            if not comparison_value:
                raise RedTeamCorpusError(
                    f"{prefix} becomes empty in {match} matching mode"
                )
        return cls(label=label.strip(), value=value, match=match)

    @property
    def value_hash(self) -> str:
        """Return a stable hash for PHI-free reporting."""

        digest = hashlib.sha256(self.value.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


@dataclass(frozen=True)
class RedTeamCase:
    """One synthetic attack case loaded from the JSONL corpus."""

    case_id: str
    abuse_case_id: str
    attack_type: str
    text: str
    expected_protected: tuple[ProtectedAssertion, ...]
    language: str = "en"
    synthetic: bool = True

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        line_number: int,
    ) -> RedTeamCase:
        """Validate and build one corpus case."""

        if set(data) - _CASE_FIELDS:
            raise RedTeamCorpusError(
                f"corpus line {line_number} contains unsupported fields"
            )

        case_id = data.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise RedTeamCorpusError(
                f"corpus line {line_number} requires a non-empty id"
            )
        case_id = case_id.strip()
        if not _REPORT_SAFE_ID.fullmatch(case_id):
            raise RedTeamCorpusError(
                f"corpus line {line_number} id must be a lowercase safe slug"
            )

        abuse_case_id = data.get("abuse_case_id")
        if not isinstance(abuse_case_id, str) or not _ABUSE_CASE_ID.fullmatch(
            abuse_case_id
        ):
            raise RedTeamCorpusError(
                f"case {case_id!r} requires an abuse_case_id like AC-01"
            )

        attack_type = data.get("attack_type")
        if not isinstance(attack_type, str) or not _REPORT_SAFE_ID.fullmatch(
            attack_type
        ):
            raise RedTeamCorpusError(
                f"case {case_id!r} requires a lowercase slug attack_type"
            )

        text = data.get("text")
        if not isinstance(text, str) or not text:
            raise RedTeamCorpusError(f"case {case_id!r} requires non-empty text")
        if len(text) > MAX_REDTEAM_CASE_TEXT_CHARS:
            raise RedTeamCorpusError(f"case {case_id!r} text is too long")

        if data.get("synthetic") is not True:
            raise RedTeamCorpusError(
                f"case {case_id!r} must explicitly set synthetic to true"
            )

        language = data.get("language", "en")
        if not isinstance(language, str) or not _LANGUAGE_ID.fullmatch(language):
            raise RedTeamCorpusError(
                f"case {case_id!r} requires a safe language identifier"
            )

        raw_assertions = data.get("expected_protected")
        if not isinstance(raw_assertions, list) or not raw_assertions:
            raise RedTeamCorpusError(
                f"case {case_id!r} requires expected_protected assertions"
            )
        if len(raw_assertions) > MAX_REDTEAM_ASSERTIONS_PER_CASE:
            raise RedTeamCorpusError(
                f"case {case_id!r} has too many expected_protected assertions"
            )
        assertions: list[ProtectedAssertion] = []
        seen_assertions: set[tuple[str, str]] = set()
        for index, assertion in enumerate(raw_assertions, start=1):
            if not isinstance(assertion, Mapping):
                raise RedTeamCorpusError(
                    f"case {case_id!r} assertion {index} must be an object"
                )
            parsed = ProtectedAssertion.from_mapping(
                assertion,
                case_id=case_id,
                index=index,
            )
            if parsed.value not in text:
                raise RedTeamCorpusError(
                    f"case {case_id!r} assertion {index} is absent from its text"
                )
            assertion_key = (parsed.value, parsed.match)
            if assertion_key in seen_assertions:
                raise RedTeamCorpusError(
                    f"case {case_id!r} repeats an expected_protected assertion"
                )
            seen_assertions.add(assertion_key)
            assertions.append(parsed)

        return cls(
            case_id=case_id,
            abuse_case_id=abuse_case_id,
            attack_type=attack_type.strip(),
            text=text,
            expected_protected=tuple(assertions),
            language=language,
        )


@dataclass(frozen=True)
class RedTeamCaseResult:
    """PHI-free outcome for one corpus case."""

    case_id: str
    abuse_case_id: str
    attack_type: str
    assertion_count: int
    failed_assertion_count: int
    bypassed: bool
    leaked_assertion_hashes: tuple[str, ...] = ()
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result without source or output text."""

        return {
            "abuse_case_id": self.abuse_case_id,
            "assertion_count": self.assertion_count,
            "attack_type": self.attack_type,
            "bypassed": self.bypassed,
            "case_id": self.case_id,
            "error_type": self.error_type,
            "failed_assertion_count": self.failed_assertion_count,
            "leaked_assertion_hashes": list(self.leaked_assertion_hashes),
        }


@dataclass(frozen=True)
class AttackBypassReport:
    """Aggregate bypass rate for one attack type."""

    attack_type: str
    abuse_case_ids: tuple[str, ...]
    case_count: int
    bypassed_cases: int
    bypass_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready per-attack report."""

        return {
            "abuse_case_ids": list(self.abuse_case_ids),
            "attack_type": self.attack_type,
            "bypass_rate": self.bypass_rate,
            "bypassed_cases": self.bypassed_cases,
            "case_count": self.case_count,
        }


@dataclass(frozen=True)
class AbuseCaseBypassReport:
    """Aggregate bypass rate for one threat-model abuse case."""

    abuse_case_id: str
    attack_types: tuple[str, ...]
    case_count: int
    bypassed_cases: int
    bypass_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready per-abuse-case report."""

        return {
            "abuse_case_id": self.abuse_case_id,
            "attack_types": list(self.attack_types),
            "bypass_rate": self.bypass_rate,
            "bypassed_cases": self.bypassed_cases,
            "case_count": self.case_count,
        }


@dataclass(frozen=True)
class RedTeamReport:
    """Adversarial-PHI bypass report and optional threshold decision."""

    corpus_sha256: str
    case_results: tuple[RedTeamCaseResult, ...]
    attack_reports: tuple[AttackBypassReport, ...]
    abuse_case_reports: tuple[AbuseCaseBypassReport, ...]
    bypassed_cases: int
    bypass_rate: float
    max_bypass_rate: float | None
    gate_passed: bool
    schema_version: int = REDTEAM_REPORT_SCHEMA_VERSION

    @property
    def case_count(self) -> int:
        """Return the number of evaluated corpus cases."""

        return len(self.case_results)

    @property
    def gate_configured(self) -> bool:
        """Return whether a maximum bypass rate was configured."""

        return self.max_bypass_rate is not None

    @property
    def decision(self) -> str:
        """Return a stable CI decision label."""

        if not self.gate_configured:
            return "MEASURED"
        return "PASSED" if self.gate_passed else "FAILED"

    def to_dict(self) -> dict[str, Any]:
        """Return the complete PHI-free report payload."""

        return {
            "abuse_case_reports": [
                report.to_dict() for report in self.abuse_case_reports
            ],
            "attack_reports": [report.to_dict() for report in self.attack_reports],
            "bypass_rate": self.bypass_rate,
            "bypassed_cases": self.bypassed_cases,
            "case_count": self.case_count,
            "case_results": [result.to_dict() for result in self.case_results],
            "corpus_sha256": self.corpus_sha256,
            "decision": self.decision,
            "gate_configured": self.gate_configured,
            "gate_passed": self.gate_passed,
            "max_bypass_rate": self.max_bypass_rate,
            "schema_version": self.schema_version,
        }

    def write_json(self, path: str | Path) -> Path:
        """Atomically write a deterministic JSON report and return its path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.is_symlink():
            raise ValueError("report output path must not be a symbolic link")
        if output_path.exists() and not output_path.is_file():
            raise ValueError("report output path must be a regular file")

        payload = (json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output_file:
                output_file.write(payload)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return output_path


RedTeamDeidentifier = Callable[[RedTeamCase], Any]


class _LocalPipelineRunner:
    """Reuse one local-only model loader across all default-runner cases."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._config: Any = None
        self._loader: Any = None
        self._initialization_error: Exception | None = None

    def __call__(self, case: RedTeamCase) -> Any:
        if self._initialization_error is not None:
            raise self._initialization_error
        if self._loader is None:
            try:
                from openmed.core.config import OpenMedConfig
                from openmed.core.models import ModelLoader

                self._config = OpenMedConfig(local_only=True)
                self._loader = ModelLoader(self._config)
            except Exception as exc:
                self._initialization_error = exc
                raise
        return _pipeline_deidentify(
            case,
            model_name=self.model_name,
            config=self._config,
            loader=self._loader,
        )


def load_redteam_corpus(
    path: str | Path = DEFAULT_REDTEAM_CORPUS,
) -> tuple[RedTeamCase, ...]:
    """Load and fail-closed validate a synthetic adversarial JSONL corpus."""

    _, cases = _read_redteam_corpus(Path(path))
    return cases


def _read_redteam_corpus(
    corpus_path: Path,
) -> tuple[bytes, tuple[RedTeamCase, ...]]:
    if not corpus_path.is_file():
        raise RedTeamCorpusError(f"red-team corpus not found: {corpus_path}")
    try:
        corpus_bytes = corpus_path.read_bytes()
    except OSError as exc:
        raise RedTeamCorpusError(
            f"red-team corpus could not be read: {corpus_path}"
        ) from exc
    if len(corpus_bytes) > MAX_REDTEAM_CORPUS_BYTES:
        raise RedTeamCorpusError("red-team corpus exceeds the maximum byte size")
    try:
        corpus_text = corpus_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RedTeamCorpusError("red-team corpus must be valid UTF-8") from exc

    cases: list[RedTeamCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        corpus_text.splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        if len(cases) >= MAX_REDTEAM_CASES:
            raise RedTeamCorpusError("red-team corpus contains too many cases")
        try:
            payload = json.loads(
                raw_line,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise RedTeamCorpusError(
                f"invalid JSON on corpus line {line_number}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RedTeamCorpusError(
                f"corpus line {line_number} must contain a JSON object"
            )
        case = RedTeamCase.from_mapping(payload, line_number=line_number)
        if case.case_id in seen_ids:
            raise RedTeamCorpusError(f"duplicate corpus case id: {case.case_id}")
        seen_ids.add(case.case_id)
        cases.append(case)

    if not cases:
        raise RedTeamCorpusError("red-team corpus must contain at least one case")
    return corpus_bytes, tuple(cases)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise RedTeamCorpusError("corpus JSON objects must not repeat keys")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> Any:
    raise RedTeamCorpusError(
        f"corpus JSON must not contain the non-finite number {value}"
    )


def run_redteam(
    corpus_path: str | Path = DEFAULT_REDTEAM_CORPUS,
    *,
    deidentifier: RedTeamDeidentifier | None = None,
    model_name: str = DEFAULT_REDTEAM_MODEL,
    max_bypass_rate: float | None = None,
) -> RedTeamReport:
    """Run every corpus case and compute leakage-first bypass rates.

    The default runner calls :func:`openmed.deidentify` with local-only mode
    enabled. A processing error is scored as a bypass so a broken model or
    pipeline cannot make a configured CI gate pass by skipping cases.

    Args:
        corpus_path: JSONL adversarial corpus to evaluate.
        deidentifier: Optional callable accepting a :class:`RedTeamCase` and
            returning text or an object with ``deidentified_text``.
        model_name: Cached model id or local model path for the default runner.
        max_bypass_rate: Optional inclusive pass threshold in ``[0, 1]``.

    Returns:
        A PHI-free :class:`RedTeamReport` with case and per-attack rates.

    Raises:
        RedTeamCorpusError: If the corpus violates the required schema.
        ValueError: If ``max_bypass_rate`` is outside ``[0, 1]``.
    """

    _validate_threshold(max_bypass_rate)
    corpus_path = Path(corpus_path)
    corpus_bytes, cases = _read_redteam_corpus(corpus_path)
    runner = deidentifier or _LocalPipelineRunner(model_name)

    results = tuple(_run_case(case, runner) for case in cases)
    bypassed_cases = sum(result.bypassed for result in results)
    bypass_rate = bypassed_cases / len(results)
    attack_reports = _aggregate_attacks(results)
    abuse_case_reports = _aggregate_abuse_cases(results)
    evaluated_rates = (
        bypass_rate,
        *(report.bypass_rate for report in attack_reports),
        *(report.bypass_rate for report in abuse_case_reports),
    )
    gate_passed = max_bypass_rate is None or all(
        rate <= max_bypass_rate for rate in evaluated_rates
    )
    corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()
    return RedTeamReport(
        corpus_sha256=f"sha256:{corpus_hash}",
        case_results=results,
        attack_reports=attack_reports,
        abuse_case_reports=abuse_case_reports,
        bypassed_cases=bypassed_cases,
        bypass_rate=bypass_rate,
        max_bypass_rate=max_bypass_rate,
        gate_passed=gate_passed,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the adversarial-PHI harness command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the offline adversarial-PHI corpus and report redaction bypasses."
        )
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_REDTEAM_CORPUS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_REDTEAM_MODEL)
    parser.add_argument(
        "--max-bypass-rate",
        type=float,
        default=None,
        help=(
            "Fail with exit 1 when bypass rate exceeds this value; falls back "
            f"to {REDTEAM_THRESHOLD_ENV_VAR} when omitted."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the harness, emit its report, and return a CI-friendly exit code."""

    args = build_arg_parser().parse_args(argv)
    try:
        threshold = _configured_threshold(args.max_bypass_rate)
        report = run_redteam(
            args.corpus,
            model_name=args.model_name,
            max_bypass_rate=threshold,
        )
    except (RedTeamCorpusError, ValueError) as exc:
        print(f"red-team harness configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.output is None:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            report.write_json(args.output)
    except (OSError, ValueError):
        print("red-team report could not be written safely", file=sys.stderr)
        return 2
    return 0 if report.gate_passed else 1


def _pipeline_deidentify(
    case: RedTeamCase,
    *,
    model_name: str,
    config: Any = None,
    loader: Any = None,
) -> Any:
    from openmed import deidentify
    from openmed.core.config import OpenMedConfig

    config = config or OpenMedConfig(local_only=True)
    return deidentify(
        case.text,
        method="mask",
        model_name=model_name,
        lang=case.language,
        config=config,
        use_safety_sweep=True,
        loader=loader,
    )


def _run_case(
    case: RedTeamCase,
    runner: RedTeamDeidentifier,
) -> RedTeamCaseResult:
    try:
        output = _coerce_deidentified_text(runner(case))
        leaked = _leaked_assertion_hashes(case.expected_protected, output)
    except Exception as exc:
        return RedTeamCaseResult(
            case_id=case.case_id,
            abuse_case_id=case.abuse_case_id,
            attack_type=case.attack_type,
            assertion_count=len(case.expected_protected),
            failed_assertion_count=len(case.expected_protected),
            bypassed=True,
            error_type=type(exc).__name__,
        )

    return RedTeamCaseResult(
        case_id=case.case_id,
        abuse_case_id=case.abuse_case_id,
        attack_type=case.attack_type,
        assertion_count=len(case.expected_protected),
        failed_assertion_count=len(leaked),
        bypassed=bool(leaked),
        leaked_assertion_hashes=leaked,
    )


def _coerce_deidentified_text(value: Any) -> str:
    if isinstance(value, str):
        output = value
    elif isinstance(value, Mapping):
        output = value.get("deidentified_text")
    else:
        output = getattr(value, "deidentified_text", None)
    if not isinstance(output, str):
        raise TypeError(
            "deidentifier must return text or an object with deidentified_text"
        )
    if len(output) > MAX_REDTEAM_OUTPUT_CHARS:
        raise ValueError("deidentified output exceeds the red-team size limit")
    return output


def _leaked_assertion_hashes(
    assertions: Sequence[ProtectedAssertion],
    output: str,
) -> tuple[str, ...]:
    casefold_output: str | None = None
    normalized_output: str | None = None
    alnum_output: str | None = None
    leaked_hashes: list[str] = []

    for assertion in assertions:
        if assertion.match == "exact":
            survives = assertion.value in output
        elif assertion.match == "casefold":
            if casefold_output is None:
                casefold_output = output.casefold()
            survives = assertion.value.casefold() in casefold_output
        else:
            if normalized_output is None:
                normalized_output = _normalized_surface(output)
            expected = _normalized_surface(assertion.value)
            observed = normalized_output
            if assertion.match == "alnum":
                expected = "".join(
                    character for character in expected if character.isalnum()
                )
                if alnum_output is None:
                    alnum_output = "".join(
                        character
                        for character in normalized_output
                        if character.isalnum()
                    )
                observed = alnum_output
            survives = expected in observed
        if survives:
            leaked_hashes.append(assertion.value_hash)
    return tuple(leaked_hashes)


def _normalized_surface(value: str) -> str:
    from openmed.core.script_detect import normalize_for_pii_detection

    return normalize_for_pii_detection(value).text.casefold()


def _aggregate_attacks(
    results: Sequence[RedTeamCaseResult],
) -> tuple[AttackBypassReport, ...]:
    grouped: defaultdict[str, list[RedTeamCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.attack_type].append(result)

    reports: list[AttackBypassReport] = []
    for attack_type in sorted(grouped):
        attack_results = grouped[attack_type]
        bypassed_cases = sum(result.bypassed for result in attack_results)
        reports.append(
            AttackBypassReport(
                attack_type=attack_type,
                abuse_case_ids=tuple(
                    sorted({result.abuse_case_id for result in attack_results})
                ),
                case_count=len(attack_results),
                bypassed_cases=bypassed_cases,
                bypass_rate=bypassed_cases / len(attack_results),
            )
        )
    return tuple(reports)


def _aggregate_abuse_cases(
    results: Sequence[RedTeamCaseResult],
) -> tuple[AbuseCaseBypassReport, ...]:
    grouped: defaultdict[str, list[RedTeamCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.abuse_case_id].append(result)

    reports: list[AbuseCaseBypassReport] = []
    for abuse_case_id in sorted(grouped):
        abuse_results = grouped[abuse_case_id]
        bypassed_cases = sum(result.bypassed for result in abuse_results)
        reports.append(
            AbuseCaseBypassReport(
                abuse_case_id=abuse_case_id,
                attack_types=tuple(
                    sorted({result.attack_type for result in abuse_results})
                ),
                case_count=len(abuse_results),
                bypassed_cases=bypassed_cases,
                bypass_rate=bypassed_cases / len(abuse_results),
            )
        )
    return tuple(reports)


def _validate_threshold(value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("max_bypass_rate must be a number between 0 and 1")
    if not 0.0 <= value <= 1.0:
        raise ValueError("max_bypass_rate must be between 0 and 1")


def _configured_threshold(cli_value: float | None) -> float | None:
    if cli_value is not None:
        _validate_threshold(cli_value)
        return cli_value
    raw_value = os.getenv(REDTEAM_THRESHOLD_ENV_VAR)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{REDTEAM_THRESHOLD_ENV_VAR} must contain a number") from exc
    _validate_threshold(value)
    return value


__all__ = [
    "AbuseCaseBypassReport",
    "AttackBypassReport",
    "DEFAULT_REDTEAM_CORPUS",
    "DEFAULT_REDTEAM_MODEL",
    "ProtectedAssertion",
    "REDTEAM_REPORT_SCHEMA_VERSION",
    "REDTEAM_THRESHOLD_ENV_VAR",
    "RedTeamCase",
    "RedTeamCaseResult",
    "RedTeamCorpusError",
    "RedTeamReport",
    "build_arg_parser",
    "load_redteam_corpus",
    "main",
    "run_redteam",
]


if __name__ == "__main__":
    raise SystemExit(main())
