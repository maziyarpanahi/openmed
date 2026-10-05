"""Offline qualification of caller-owned artifacts against an explicit policy.

Receipts attest only to the supplied data and policy, not clinical approval or
provenance authenticity. No default checkpoint or evaluation data is supplied.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openmed.clinical.nli_backends import EncoderNLIBackend, LocalNLIError
from openmed.clinical.nli_gate import NLIThresholds
from openmed.clinical.nli_labels import normalize_nli_label
from openmed.eval.nli_calibration import calibrate_nli_thresholds
from openmed.eval.nli_error_slices import CLINICAL_NLI_PHENOMENA

_LABELS = ("entailment", "contradiction", "neutral")
_KINDS = ("synthetic", "caller_supplied", "restricted")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


class NLIQualificationError(LocalNLIError):
    """Value-free receipt admission or input failure."""


@dataclass(frozen=True)
class NLIQualificationPolicy:
    """Caller-selected support and operating requirements, not clinical norms.

    Support requires every gold class in both splits and each required slice.
    Precision, recall and FPR constraints apply to both decisive classes.
    """

    min_per_class: int = 20
    precision_floor: float = 0.95
    recall_floor: float = 0.8
    false_positive_rate_ceiling: float = 0.05
    margin: float = 0.05
    required_slices: tuple[str, ...] = CLINICAL_NLI_PHENOMENA

    def __post_init__(self) -> None:
        if type(self.min_per_class) is not int or self.min_per_class < 1:
            raise NLIQualificationError("invalid_policy")
        for value in (
            self.precision_floor,
            self.recall_floor,
            self.false_positive_rate_ceiling,
            self.margin,
        ):
            if type(value) not in (float, int) or not 0 <= value <= 1:
                raise NLIQualificationError("invalid_policy")
        if (
            not isinstance(self.required_slices, tuple)
            or len(set(self.required_slices)) != len(self.required_slices)
            or not set(self.required_slices) <= set(CLINICAL_NLI_PHENOMENA)
        ):
            raise NLIQualificationError("invalid_policy")


@dataclass(frozen=True)
class NLIQualificationReceipt:
    """Immutable aggregate receipt; no paths, identifiers or source text."""

    _payload: str = field(repr=False)

    @property
    def digest(self) -> str:
        """Return the calibration identifier bound to the entire receipt."""
        return hashlib.sha256(self._payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive copy with its content digest."""
        return {**json.loads(self._payload), "receipt_digest": self.digest}

    def to_json(self) -> str:
        """Serialize only controlled metadata, digests and aggregate metrics."""
        return _json(self.to_dict())


def _artifact_digest(path: Path) -> str:
    try:
        if not path.is_dir() or path.is_symlink():
            raise ValueError
        entries = []
        for item in sorted(path.rglob("*")):
            if item.is_symlink():
                raise ValueError
            if item.is_file():
                digest = hashlib.sha256()
                with item.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                entries.append((item.relative_to(path).as_posix(), digest.hexdigest()))
        if not entries:
            raise ValueError
        return _digest(entries)
    except (OSError, ValueError):
        raise NLIQualificationError("artifact_unavailable") from None


def _mapping(value: Mapping[str, str]) -> dict[str, str]:
    try:
        mapped = {str(k): normalize_nli_label(v).value for k, v in value.items()}
        if set(mapped) != {"0", "1", "2"} or set(mapped.values()) != set(_LABELS):
            raise ValueError
        return mapped
    except (AttributeError, TypeError, ValueError):
        raise NLIQualificationError("unknown_label") from None


def _split(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        provenance = value["provenance"]
        kind = provenance["kind"]
        reference = provenance["reference"]
        if (
            kind not in _KINDS
            or not isinstance(reference, str)
            or not reference.strip()
        ):
            raise ValueError
        records = []
        for record in value["records"]:
            try:
                label = normalize_nli_label(record["gold_label"]).value
            except (TypeError, ValueError):
                raise NLIQualificationError("unknown_label") from None
            if label not in _LABELS:
                raise NLIQualificationError("unknown_label")
            slices = tuple(sorted(set(record.get("slices", ()))))
            if not set(slices) <= set(CLINICAL_NLI_PHENOMENA):
                raise ValueError
            strings = [record[k] for k in ("id", "group_id", "premise", "hypothesis")]
            if any(not isinstance(s, str) or not s.strip() for s in strings):
                raise ValueError
            records.append(
                dict(
                    zip(("id", "group_id", "premise", "hypothesis"), strings),
                    gold_label=label,
                    slices=slices,
                )
            )
        if len({r["id"] for r in records}) != len(records):
            raise ValueError
        return {
            "provenance": {"kind": kind, "reference_digest": _digest(reference)},
            "records": records,
        }
    except NLIQualificationError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError):
        raise NLIQualificationError("invalid_dataset") from None


def _binding(path, mapping, development, evaluation, policy, runtime):
    if runtime not in ("torch", "onnx"):
        raise NLIQualificationError("invalid_runtime")
    return {
        "artifact_digest": _artifact_digest(Path(path)),
        "label_mapping_digest": _digest(mapping),
        "development_digest": _digest(development),
        "evaluation_digest": _digest(evaluation),
        "policy_digest": _digest(asdict(policy)),
        "runtime": runtime,
    }


def _support(records, minimum):
    counts = Counter(r["gold_label"] for r in records)
    return all(counts[label] >= minimum for label in _LABELS)


def _scored(backend, records, margin):
    rows = []
    for record in records:
        scores = backend.predict_scores(record["premise"], record["hypothesis"])
        row = dict(record)
        # Match the deployed gate: a tied, non-winning or low-margin class
        # cannot become an accepted positive in the calibration report.
        row["scores"] = {
            label: scores[label]
            if scores[label] > max(scores[k] for k in _LABELS if k != label)
            and scores[label] - max(scores[k] for k in _LABELS if k != label) >= margin
            else 0.0
            for label in ("entailment", "contradiction")
        }
        rows.append(row)
    return rows


def _report(rows, label, policy, threshold=None):
    return calibrate_nli_thresholds(
        [
            {
                "id": r["id"],
                "premise": r["premise"],
                "hypothesis": r["hypothesis"],
                "gold_label": "entailment"
                if r["gold_label"] == label
                else "not_entailment",
                "score": r["scores"][label],
            }
            for r in rows
        ],
        model_id="caller-supplied-local-artifact",
        thresholds=(threshold,)
        if threshold is not None
        else tuple(
            sorted({1.0, *(r["scores"][label] for r in rows if r["scores"][label] > 0)})
        ),
        precision_floor=policy.precision_floor,
        recall_floor=policy.recall_floor,
        false_positive_rate_ceiling=policy.false_positive_rate_ceiling,
    )


def _meets(report, policy):
    point = report.recommended_point
    return (
        point.confusion.true_positives > 0
        and point.precision >= policy.precision_floor
        and point.recall >= policy.recall_floor
        and point.false_positive_rate <= policy.false_positive_rate_ceiling
    )


def qualify_local_nli(
    model_path: str | Path,
    *,
    label_mapping: Mapping[str, str],
    development: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    policy: NLIQualificationPolicy = NLIQualificationPolicy(),
    runtime: str = "torch",
    loader: Any | None = None,
) -> NLIQualificationReceipt:
    """Load, calibrate and evaluate an existing local artifact without downloads.

    Args:
        model_path: Caller-owned local directory containing the complete artifact.
        label_mapping: Explicit index-to-three-class mapping.
        development: Provenance and source-grouped records for threshold selection.
        evaluation: Separate held-out records, never used to select thresholds.
        policy: Explicit operating constraints and minimum per-class support.
        runtime: Existing local torch or ONNX runtime.
        loader: Optional injected local loader for offline adapter tests.

    Returns:
        Digest-bound receipt with unavailable, insufficient, synthetic-only or
        qualified status. Qualified means only the declared supplied-policy test.
        Missing data never falls back to bundled synthetic examples.
    """
    reasons = []
    payload = {
        "schema_version": 1,
        "status": "unavailable",
        "qualified": False,
        "human_review_required": True,
        "reasons": reasons,
        "binding": None,
        "provenance": {},
        "split_counts": {},
        "supported_slices": [],
        "reports": {},
        "thresholds": None,
    }
    try:
        mapping = _mapping(label_mapping)
        dev, ev = _split(development), _split(evaluation)
        binding = _binding(model_path, mapping, dev, ev, policy, runtime)
        payload["binding"] = binding
        for name, split in (("development", dev), ("evaluation", ev)):
            if split is None:
                reasons.append(f"{name}_unavailable")
            else:
                payload["provenance"][name] = split["provenance"]
                payload["split_counts"][name] = len(split["records"])
        if dev is None or ev is None:
            return NLIQualificationReceipt(_json(payload))
        drows, erows = dev["records"], ev["records"]
        groups = {r["group_id"] for r in drows} & {r["group_id"] for r in erows}
        pairs = {(r["premise"], r["hypothesis"]) for r in drows} & {
            (r["premise"], r["hypothesis"]) for r in erows
        }
        identifiers = {r["id"] for r in drows} & {r["id"] for r in erows}
        if groups or pairs or identifiers:
            reasons.append("split_overlap")
        for name, rows in (("development", drows), ("evaluation", erows)):
            if not _support(rows, policy.min_per_class):
                reasons.append(f"{name}_insufficient_class_support")
        for slice_name in CLINICAL_NLI_PHENOMENA:
            if all(
                _support(
                    [r for r in rows if slice_name in r["slices"]], policy.min_per_class
                )
                for rows in (drows, erows)
            ):
                payload["supported_slices"].append(slice_name)
            elif slice_name in policy.required_slices:
                reasons.append(f"insufficient_slice_{slice_name}")
        payload["status"] = "insufficient"
        if reasons:
            return NLIQualificationReceipt(_json(payload))
        backend = EncoderNLIBackend(
            model_path,
            label_mapping=mapping,
            thresholds=NLIThresholds(),
            runtime=runtime,
            loader=loader,
        )
        dscores = _scored(backend, drows, policy.margin)
        escores = _scored(backend, erows, policy.margin)
        thresholds = {}
        for label in ("entailment", "contradiction"):
            report = _report(dscores, label, policy)
            threshold = report.recommended_threshold
            thresholds[label] = threshold
            heldout = _report(escores, label, policy, threshold)
            payload["reports"][label] = {
                "development": report.to_dict(),
                "evaluation": heldout.to_dict(),
                "slices": {},
            }
            if not _meets(report, policy):
                reasons.append(f"development_constraints_{label}")
            if not _meets(heldout, policy):
                reasons.append(f"evaluation_constraints_{label}")
            for slice_name in payload["supported_slices"]:
                sliced = _report(
                    [r for r in escores if slice_name in r["slices"]],
                    label,
                    policy,
                    threshold,
                )
                payload["reports"][label]["slices"][slice_name] = sliced.to_dict()
                if slice_name in policy.required_slices and not _meets(sliced, policy):
                    reasons.append(f"slice_constraints_{slice_name}_{label}")
        payload["thresholds"] = {**thresholds, "margin": policy.margin}
        if _binding(model_path, mapping, dev, ev, policy, runtime) != binding:
            raise NLIQualificationError("receipt_mismatch")
        if any(s["provenance"]["kind"] == "synthetic" for s in (dev, ev)):
            reasons.append("synthetic_not_clinical_evidence")
            payload["status"] = (
                "synthetic_only" if len(reasons) == 1 else "insufficient"
            )
        elif not reasons:
            payload["status"] = "qualified"
            payload["qualified"] = True
    except LocalNLIError as exc:
        # Never serialize backend exceptions or arbitrary provider payloads.
        reasons.append(
            str(exc)
            if isinstance(exc, NLIQualificationError)
            else "local_inference_unavailable"
        )
        payload["status"] = "unavailable"
    return NLIQualificationReceipt(_json(payload))


class QualifiedNLIBackend:
    """Receipt-bound local score callback for BriefContext, with drift checks."""

    def __init__(self, backend, receipt, check):
        self._backend = backend
        self._check = check
        self.thresholds = NLIThresholds(
            **receipt.to_dict()["thresholds"], calibration_id=receipt.digest
        )

    def __call__(self, premise: str, hypothesis: str) -> dict[str, Any]:
        """Return receipt-bound scores or refuse when any binding has changed."""
        self._check()
        scores = self._backend.predict_scores(premise, hypothesis)
        self._check()
        return {
            **scores,
            "calibration_id": self.thresholds.calibration_id,
            "calibrated": True,
        }


def bind_qualified_nli(
    receipt: NLIQualificationReceipt,
    model_path: str | Path,
    *,
    label_mapping: Mapping[str, str],
    development: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
    policy: NLIQualificationPolicy = NLIQualificationPolicy(),
    runtime: str = "torch",
    loader: Any | None = None,
) -> QualifiedNLIBackend:
    """Bind a successful receipt to BriefContext's explicit callback contract.

    Recheck artifacts and calibration inputs before and after each inference.
    Synthetic receipts cannot enter this path. Receipts are local evidence,
    not signed third-party attestations or approval to use restricted data.
    """
    payload = receipt.to_dict()
    if (
        not payload["qualified"]
        or payload["status"] != "qualified"
        or payload["reasons"]
    ):
        raise NLIQualificationError("uncalibrated")
    mapping = _mapping(label_mapping)

    def check():
        actual = _binding(
            model_path,
            _mapping(label_mapping),
            _split(development),
            _split(evaluation),
            policy,
            runtime,
        )
        if actual != payload["binding"]:
            raise NLIQualificationError("receipt_mismatch")

    check()
    backend = EncoderNLIBackend(
        model_path,
        label_mapping=mapping,
        thresholds=NLIThresholds(
            **payload["thresholds"], calibration_id=receipt.digest
        ),
        runtime=runtime,
        loader=loader,
    )
    backend._load()
    check()
    return QualifiedNLIBackend(backend, receipt, check)
