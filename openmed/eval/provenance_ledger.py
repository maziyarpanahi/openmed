"""Hash-chained provenance ledger for evaluation results.

The ledger binds aggregate, PHI-safe evaluation results to the exact inputs
that produced them. Each record commits to the preceding record, allowing the
verifier to detect partial mutation, insertion, deletion, and reordering. A
trusted head hash additionally detects a complete chain rewrite. Result metadata
is restricted to schema-defined keys and aggregate values so raw clinical text
cannot be persisted accidentally.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.repro_hash import compute_reproducibility_hash

LEDGER_SCHEMA_VERSION = "openmed.eval_provenance_ledger.v1"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,255}$")
_METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ALLOWED_RESULT_METADATA_KEYS = frozenset(
    {
        "accuracy",
        "auroc",
        "average_precision",
        "count",
        "document_count",
        "duration_ms",
        "error_count",
        "f1",
        "failed_count",
        "false_negative_count",
        "false_positive_count",
        "fixture_count",
        "latency_ms",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "max",
        "mean",
        "median",
        "metrics",
        "micro_f1",
        "micro_precision",
        "micro_recall",
        "min",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "passed",
        "passed_count",
        "precision",
        "rate",
        "recall",
        "result_digest",
        "score",
        "specificity",
        "standard_deviation",
        "success_count",
        "text_length",
        "true_negative_count",
        "true_positive_count",
        "weighted_f1",
        "weighted_precision",
        "weighted_recall",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_json(value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


GENESIS_HASH = _hash_json({"schema_version": LEDGER_SCHEMA_VERSION, "sequence": -1})


class ProvenanceLedgerError(ValueError):
    """Base error for malformed or unverifiable provenance ledgers."""


class UnsafeResultMetadataError(ProvenanceLedgerError):
    """Raised when result metadata could persist raw or identifying data."""


class LedgerVerificationError(ProvenanceLedgerError):
    """Raised when an evaluation provenance ledger fails verification."""


@dataclass(frozen=True)
class EvalProvenanceRecord:
    """One PHI-safe evaluation result and its reproducibility provenance."""

    sequence: int
    model_digest: str
    suite_id: str
    suite_version: str
    code_hash: str
    seed: int
    config_digest: str
    env_fingerprint: str
    result_metadata: Mapping[str, Any]
    result_hash: str
    reproducibility_hash: str
    previous_hash: str
    record_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequence", _non_negative_int(self.sequence, "sequence")
        )
        object.__setattr__(
            self,
            "model_digest",
            _required_digest(self.model_digest, "model_digest"),
        )
        object.__setattr__(self, "suite_id", _identifier(self.suite_id, "suite_id"))
        object.__setattr__(
            self,
            "suite_version",
            _identifier(self.suite_version, "suite_version"),
        )
        object.__setattr__(self, "code_hash", _identifier(self.code_hash, "code_hash"))
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(
            self,
            "config_digest",
            _required_digest(self.config_digest, "config_digest"),
        )
        object.__setattr__(
            self,
            "env_fingerprint",
            _required_digest(self.env_fingerprint, "env_fingerprint"),
        )
        object.__setattr__(
            self,
            "result_metadata",
            _normalise_result_metadata(self.result_metadata),
        )
        for field_name in (
            "result_hash",
            "reproducibility_hash",
            "previous_hash",
            "record_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_digest(getattr(self, field_name), field_name),
            )

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        model_digest: str,
        suite_id: str,
        suite_version: str,
        code_hash: str,
        seed: int,
        config_digest: str,
        env_fingerprint: str,
        result_metadata: Mapping[str, Any],
        previous_hash: str,
    ) -> "EvalProvenanceRecord":
        """Build a record and derive all reproducibility and chain hashes."""

        normalised_result = _normalise_result_metadata(result_metadata)
        payload = {
            "sequence": _non_negative_int(sequence, "sequence"),
            "model_digest": _required_digest(model_digest, "model_digest"),
            "suite_id": _identifier(suite_id, "suite_id"),
            "suite_version": _identifier(suite_version, "suite_version"),
            "code_hash": _identifier(code_hash, "code_hash"),
            "seed": _integer(seed, "seed"),
            "config_digest": _required_digest(config_digest, "config_digest"),
            "env_fingerprint": _required_digest(
                env_fingerprint,
                "env_fingerprint",
            ),
            "result_metadata": normalised_result,
            "result_hash": compute_result_hash(normalised_result),
            "previous_hash": _required_digest(previous_hash, "previous_hash"),
        }
        payload["reproducibility_hash"] = compute_eval_reproducibility_hash(
            model_digest=payload["model_digest"],
            suite_id=payload["suite_id"],
            suite_version=payload["suite_version"],
            code_hash=payload["code_hash"],
            seed=payload["seed"],
            config_digest=payload["config_digest"],
            env_fingerprint=payload["env_fingerprint"],
        )
        record_hash = _hash_json(payload)
        return cls(**payload, record_hash=record_hash)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvalProvenanceRecord":
        """Load one record from its strict JSON representation."""

        required = {
            "sequence",
            "model_digest",
            "suite_id",
            "suite_version",
            "code_hash",
            "seed",
            "config_digest",
            "env_fingerprint",
            "result_metadata",
            "result_hash",
            "reproducibility_hash",
            "previous_hash",
            "record_hash",
        }
        _require_exact_fields(payload, required, "provenance record")
        result_metadata = payload["result_metadata"]
        if not isinstance(result_metadata, Mapping):
            raise ProvenanceLedgerError("result_metadata must be an object")
        return cls(
            sequence=payload["sequence"],
            model_digest=payload["model_digest"],
            suite_id=payload["suite_id"],
            suite_version=payload["suite_version"],
            code_hash=payload["code_hash"],
            seed=payload["seed"],
            config_digest=payload["config_digest"],
            env_fingerprint=payload["env_fingerprint"],
            result_metadata=result_metadata,
            result_hash=payload["result_hash"],
            reproducibility_hash=payload["reproducibility_hash"],
            previous_hash=payload["previous_hash"],
            record_hash=payload["record_hash"],
        )

    def hash_material(self) -> dict[str, Any]:
        """Return the canonical fields committed by ``record_hash``."""

        payload = self.to_dict()
        payload.pop("record_hash")
        return payload

    def compute_record_hash(self) -> str:
        """Recompute this record's expected chain hash."""

        return _hash_json(self.hash_material())

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible record."""

        return {
            "code_hash": self.code_hash,
            "config_digest": self.config_digest,
            "env_fingerprint": self.env_fingerprint,
            "model_digest": self.model_digest,
            "previous_hash": self.previous_hash,
            "record_hash": self.record_hash,
            "reproducibility_hash": self.reproducibility_hash,
            "result_hash": self.result_hash,
            "result_metadata": _normalise_result_metadata(self.result_metadata),
            "seed": self.seed,
            "sequence": self.sequence,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
        }


@dataclass(frozen=True)
class EvalProvenanceLedger:
    """An append-only sequence of hash-linked evaluation records."""

    records: tuple[EvalProvenanceRecord, ...] = ()
    schema_version: str = LEDGER_SCHEMA_VERSION
    genesis_hash: str = GENESIS_HASH
    record_count: int = 0
    head_hash: str = GENESIS_HASH

    def __post_init__(self) -> None:
        if self.schema_version != LEDGER_SCHEMA_VERSION:
            raise ProvenanceLedgerError(
                f"unsupported ledger schema_version: {self.schema_version!r}"
            )
        if self.genesis_hash != GENESIS_HASH:
            raise ProvenanceLedgerError("ledger genesis_hash does not match the schema")
        object.__setattr__(
            self, "record_count", _non_negative_int(self.record_count, "record_count")
        )
        object.__setattr__(
            self, "head_hash", _required_digest(self.head_hash, "head_hash")
        )
        normalised_records = tuple(
            record
            if isinstance(record, EvalProvenanceRecord)
            else EvalProvenanceRecord.from_mapping(record)
            for record in self.records
        )
        object.__setattr__(self, "records", normalised_records)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvalProvenanceLedger":
        """Load a ledger from its strict JSON representation."""

        required = {
            "schema_version",
            "genesis_hash",
            "record_count",
            "head_hash",
            "records",
        }
        _require_exact_fields(payload, required, "provenance ledger")
        records = payload["records"]
        if not isinstance(records, list):
            raise ProvenanceLedgerError("ledger records must be a list")
        return cls(
            schema_version=payload["schema_version"],
            genesis_hash=payload["genesis_hash"],
            record_count=payload["record_count"],
            head_hash=payload["head_hash"],
            records=tuple(EvalProvenanceRecord.from_mapping(row) for row in records),
        )

    def append(self, record: EvalProvenanceRecord) -> "EvalProvenanceLedger":
        """Return a ledger with one already-built record appended."""

        self.validate()
        if record.sequence != self.record_count:
            raise ProvenanceLedgerError(
                "record sequence does not match the next ledger position"
            )
        if record.previous_hash != self.head_hash:
            raise ProvenanceLedgerError("record does not link to the ledger head")
        _validate_record(record, self.record_count, self.head_hash)
        return EvalProvenanceLedger(
            records=(*self.records, record),
            record_count=self.record_count + 1,
            head_hash=record.record_hash,
        )

    def record_result(
        self,
        *,
        model_digest: str,
        suite_id: str,
        suite_version: str,
        code_hash: str,
        seed: int,
        config_digest: str,
        env_fingerprint: str,
        result_metadata: Mapping[str, Any],
    ) -> tuple["EvalProvenanceLedger", EvalProvenanceRecord]:
        """Create and append the provenance record for one completed eval run."""

        self.validate()
        record = EvalProvenanceRecord.create(
            sequence=self.record_count,
            model_digest=model_digest,
            suite_id=suite_id,
            suite_version=suite_version,
            code_hash=code_hash,
            seed=seed,
            config_digest=config_digest,
            env_fingerprint=env_fingerprint,
            result_metadata=result_metadata,
            previous_hash=self.head_hash,
        )
        return self.append(record), record

    def validate(self) -> None:
        """Raise ``LedgerVerificationError`` unless the full ledger is intact."""

        if self.record_count != len(self.records):
            raise LedgerVerificationError(
                "ledger record_count does not match the number of records"
            )

        previous_hash = self.genesis_hash
        for sequence, record in enumerate(self.records):
            _validate_record(record, sequence, previous_hash)
            previous_hash = record.record_hash

        if not hmac.compare_digest(self.head_hash, previous_hash):
            raise LedgerVerificationError("ledger head_hash does not match the chain")

    def verify(self) -> bool:
        """Return whether every ledger anchor, link, and derived hash is valid."""

        try:
            self.validate()
        except ProvenanceLedgerError:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-compatible ledger payload."""

        return {
            "genesis_hash": self.genesis_hash,
            "head_hash": self.head_hash,
            "record_count": self.record_count,
            "records": [record.to_dict() for record in self.records],
            "schema_version": self.schema_version,
        }


def compute_result_hash(result_metadata: Mapping[str, Any]) -> str:
    """Hash a PHI-safe aggregate result summary deterministically."""

    return _hash_json(_normalise_result_metadata(result_metadata))


def compute_eval_reproducibility_hash(
    *,
    model_digest: str,
    suite_id: str,
    suite_version: str,
    code_hash: str,
    seed: int,
    config_digest: str,
    env_fingerprint: str,
) -> str:
    """Derive the OM-023 reproducibility hash for one evaluation input set."""

    model = _required_digest(model_digest, "model_digest")
    suite = _identifier(suite_id, "suite_id")
    version = _identifier(suite_version, "suite_version")
    code = _identifier(code_hash, "code_hash")
    eval_seed = _integer(seed, "seed")
    config = _required_digest(config_digest, "config_digest")
    environment = _required_digest(env_fingerprint, "env_fingerprint")
    return compute_reproducibility_hash(
        recipe={"config_digest": config, "seed": eval_seed},
        data_manifest={
            "environment_fingerprint": environment,
            "suite_id": suite,
            "suite_version": version,
        },
        base_model={"digest": model},
        git_sha=code,
    )


def append_eval_result(
    path: str | Path,
    *,
    model_digest: str,
    suite_id: str,
    suite_version: str,
    code_hash: str,
    seed: int,
    config_digest: str,
    env_fingerprint: str,
    result_metadata: Mapping[str, Any],
) -> EvalProvenanceRecord:
    """Append one completed eval run to *path* and return its record."""

    ledger_path = Path(path)
    ledger = (
        load_ledger(ledger_path) if ledger_path.exists() else EvalProvenanceLedger()
    )
    ledger.validate()
    updated, record = ledger.record_result(
        model_digest=model_digest,
        suite_id=suite_id,
        suite_version=suite_version,
        code_hash=code_hash,
        seed=seed,
        config_digest=config_digest,
        env_fingerprint=env_fingerprint,
        result_metadata=result_metadata,
    )
    write_ledger(updated, ledger_path)
    return record


def load_ledger(path: str | Path) -> EvalProvenanceLedger:
    """Load an evaluation provenance ledger without hiding verification errors."""

    ledger_path = Path(path)
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerVerificationError(
            f"cannot read ledger {ledger_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise LedgerVerificationError("provenance ledger must be a JSON object")
    try:
        return EvalProvenanceLedger.from_mapping(payload)
    except ProvenanceLedgerError:
        raise
    except (TypeError, ValueError) as exc:
        raise LedgerVerificationError(f"malformed provenance ledger: {exc}") from exc


def write_ledger(
    ledger: EvalProvenanceLedger,
    path: str | Path,
) -> Path:
    """Atomically persist a verified ledger using deterministic JSON."""

    ledger.validate()
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            ledger.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    descriptor, temporary_name = tempfile.mkstemp(
        dir=ledger_path.parent,
        prefix=f".{ledger_path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, ledger_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return ledger_path


def verify_ledger(
    ledger: EvalProvenanceLedger | str | Path,
    *,
    expected_head_hash: str | None = None,
    raise_on_error: bool = False,
) -> bool:
    """Verify the chain, derived hashes, and an optional trusted head anchor."""

    try:
        resolved = load_ledger(ledger) if isinstance(ledger, (str, Path)) else ledger
        if not isinstance(resolved, EvalProvenanceLedger):
            raise LedgerVerificationError("expected a provenance ledger or path")
        resolved.validate()
        if expected_head_hash is not None:
            expected_head = _required_digest(
                expected_head_hash,
                "expected_head_hash",
            )
            if not hmac.compare_digest(resolved.head_hash, expected_head):
                raise LedgerVerificationError(
                    "ledger head_hash does not match the trusted expected head"
                )
    except (OSError, json.JSONDecodeError, ProvenanceLedgerError) as exc:
        if raise_on_error:
            if isinstance(exc, LedgerVerificationError):
                raise
            raise LedgerVerificationError(str(exc)) from exc
        return False
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the standalone provenance-ledger command parser."""

    parser = argparse.ArgumentParser(
        prog="python -m openmed.eval.provenance_ledger",
        description=(
            "Verify OpenMed evaluation provenance ledgers, optionally against "
            "a trusted head hash."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser(
        "verify",
        help="Re-check the chain, derived hashes, and an optional trusted head.",
    )
    verify_parser.add_argument("ledger", type=Path, help="Ledger JSON file to verify.")
    verify_parser.add_argument(
        "--expected-head",
        metavar="sha256:...",
        help="Trusted ledger head hash used to detect a complete chain rewrite.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the provenance-ledger verification command."""

    args = build_arg_parser().parse_args(argv)
    if args.command != "verify":  # pragma: no cover - argparse requires a command.
        return 2
    try:
        ledger = load_ledger(args.ledger)
        verify_ledger(
            ledger,
            expected_head_hash=args.expected_head,
            raise_on_error=True,
        )
    except ProvenanceLedgerError as exc:
        print(f"Ledger verification failed: {exc}", file=sys.stderr)
        return 1
    anchor = " against the trusted head" if args.expected_head is not None else ""
    print(
        f"Verified {ledger.record_count} eval provenance record(s){anchor}: "
        f"{args.ledger}"
    )
    return 0


def _validate_record(
    record: EvalProvenanceRecord,
    sequence: int,
    previous_hash: str,
) -> None:
    if record.sequence != sequence:
        raise LedgerVerificationError(
            f"record {sequence} has unexpected sequence {record.sequence}"
        )
    if not hmac.compare_digest(record.previous_hash, previous_hash):
        raise LedgerVerificationError(
            f"record {sequence} does not link to the previous record"
        )

    expected_result = compute_result_hash(record.result_metadata)
    if not hmac.compare_digest(record.result_hash, expected_result):
        raise LedgerVerificationError(f"record {sequence} has an invalid result_hash")

    expected_reproducibility = compute_eval_reproducibility_hash(
        model_digest=record.model_digest,
        suite_id=record.suite_id,
        suite_version=record.suite_version,
        code_hash=record.code_hash,
        seed=record.seed,
        config_digest=record.config_digest,
        env_fingerprint=record.env_fingerprint,
    )
    if not hmac.compare_digest(
        record.reproducibility_hash,
        expected_reproducibility,
    ):
        raise LedgerVerificationError(
            f"record {sequence} has an invalid reproducibility_hash"
        )

    expected_record = record.compute_record_hash()
    if not hmac.compare_digest(record.record_hash, expected_record):
        raise LedgerVerificationError(f"record {sequence} has an invalid record_hash")


def _normalise_result_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise UnsafeResultMetadataError("result_metadata must be an object")
    if not value:
        raise UnsafeResultMetadataError("result_metadata must not be empty")
    return {
        _metadata_key(key): _normalise_metadata_value(
            item,
            path=f"result_metadata.{key}",
        )
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }


def _normalise_metadata_value(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        return {
            _metadata_key(key): _normalise_metadata_value(
                item,
                path=f"{path}.{key}",
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalise_metadata_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsafeResultMetadataError(f"{path} must be finite")
        return value
    if isinstance(value, str) and _DIGEST_RE.fullmatch(value):
        return value
    raise UnsafeResultMetadataError(
        f"{path} must be numeric, boolean, null, nested metadata, or a sha256 digest"
    )


def _metadata_key(value: Any) -> str:
    if not isinstance(value, str) or not _METADATA_KEY_RE.fullmatch(value):
        raise UnsafeResultMetadataError(f"unsafe result metadata key: {value!r}")
    if value not in ALLOWED_RESULT_METADATA_KEYS:
        raise UnsafeResultMetadataError(
            f"result metadata key is not permitted by the v1 schema: {value!r}"
        )
    return value


def _required_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ProvenanceLedgerError(
            f"{field_name} must be a lowercase sha256:<64 hex> digest"
        )
    return value


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ProvenanceLedgerError(f"{field_name} must be a non-empty identifier")
    return value


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProvenanceLedgerError(f"{field_name} must be an integer")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    integer = _integer(value, field_name)
    if integer < 0:
        raise ProvenanceLedgerError(f"{field_name} must be non-negative")
    return integer


def _require_exact_fields(
    payload: Mapping[str, Any],
    required: set[str],
    context: str,
) -> None:
    fields = set(payload)
    missing = sorted(required - fields)
    if missing:
        raise ProvenanceLedgerError(f"{context} missing fields: {missing}")
    unknown = sorted(str(field) for field in fields - required)
    if unknown:
        raise ProvenanceLedgerError(f"{context} has unknown fields: {unknown}")


__all__ = [
    "ALLOWED_RESULT_METADATA_KEYS",
    "GENESIS_HASH",
    "LEDGER_SCHEMA_VERSION",
    "EvalProvenanceLedger",
    "EvalProvenanceRecord",
    "LedgerVerificationError",
    "ProvenanceLedgerError",
    "UnsafeResultMetadataError",
    "append_eval_result",
    "build_arg_parser",
    "compute_eval_reproducibility_hash",
    "compute_result_hash",
    "load_ledger",
    "main",
    "verify_ledger",
    "write_ledger",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
