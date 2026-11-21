"""Build unsigned capability evidence from explicitly indexed local reports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from openmed import __version__
from openmed.core.intended_use import load_intended_use_registry
from openmed.eval.evidence_pack import EvidencePackError, ValidationEvidencePack
from openmed.eval.evidence_pack.lanes import (
    CLINICAL_CREDENTIALED,
    CLINICAL_SYNTHETIC,
    clinical_lane,
    human_factors_lane,
    membership_lane,
    required_lanes,
    site_validation_lane,
)
from openmed.eval.journey_release import (
    _reject_duplicate_keys,
    _reject_non_finite,
)


def _read_report_file(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise EvidencePackError("invalid_report_file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    with os.fdopen(os.open(path, flags), "rb") as file:
        if not os.path.samestat(info, os.fstat(file.fileno())):
            raise EvidencePackError("invalid_report_file")
        return file.read()


def _parse(raw: bytes):
    return json.loads(
        raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_non_finite
    )


def _project_gate(report: dict, lane: str) -> tuple[str, dict]:
    fields = {
        "g1a": "direct_identifier_recall",
        "g2": "recall",
        "g3": "critical_leakage_count",
        "g4": "quantized_recall_delta",
    }
    checks = [
        check for check in report["gate_results"] if check["gate"].lower() == lane
    ]
    if lane not in fields or len(checks) != 1 or type(checks[0]["passed"]) is not bool:
        raise EvidencePackError("invalid_gate_report")
    check = checks[0]
    if lane in {"g1a", "g2"}:
        labels = check["details"].get("applicable_labels", [])
        value = min(
            (report["per_label_recall"][label] for label in labels), default=None
        )
    else:
        key = "quant_recall_delta" if lane == "g4" else "critical_leakage_count"
        value = report.get(key)
    return ("success" if check["passed"] else "failure"), {fields[lane]: value}


def _project(report, kind, lane):
    if kind == "gate_report":
        return _project_gate(report, lane)
    if kind == "benchmark_report":
        if type(report["fixture_count"]) is not int or not report["suite"]:
            raise EvidencePackError("invalid_benchmark_report")
        return ("success" if report["fixture_count"] > 0 else "unknown"), report[
            "metrics"
        ]
    if kind == "review_metrics" and report["lane"] == lane:
        return report["state"], report["metrics"]
    raise EvidencePackError("invalid_report_type")


def build_evidence_pack(
    capability: str,
    inputs: Path,
    *,
    generated_at: datetime,
    version: str = __version__,
    site_validation_key: bytes | None = None,
) -> ValidationEvidencePack:
    """Assemble only indexed report files, retaining numeric metrics and digests."""
    profile = required_lanes(capability)
    optional, optional_metrics = CLINICAL_CREDENTIALED.get(capability, (None, ()))
    if optional:
        profile[optional] = optional_metrics
    declaration = load_intended_use_registry().declarations[capability]
    lanes = {
        name: {
            "id": name,
            "mandatory": name != optional,
            "state": "unknown",
            "report_digest": None,
            "required_metrics": list(metrics),
            "metrics": {},
            "claims": [],
        }
        for name, metrics in profile.items()
    }
    site_inputs = []
    human_inputs = []
    seen = set()
    try:
        if generated_at.utcoffset() is None:
            raise EvidencePackError("invalid_generation_time")
        index = _parse(_read_report_file(inputs / "reports.json"))
        if not isinstance(index, list):
            raise EvidencePackError("invalid_report_index")
        for entry in index:
            name = entry["lane"]
            digest = entry["sha256"]
            kind = entry["type"]
            clinical = name in CLINICAL_SYNTHETIC.get(capability, {}) or (
                optional and name == optional
            )
            if clinical and kind != "clinical_report":
                raise EvidencePackError("clinical_report_required")
            human = (
                name == "human_factors"
                and kind == "human_factors_report"
                and name in profile
            )
            if (
                name == "human_factors" or kind == "human_factors_report"
            ) and not human:
                raise EvidencePackError("human_factors_sources_required")
            membership = (
                name == "membership_inference" and kind == "membership_self_test"
            )
            if (
                name == "membership_inference" or kind == "membership_self_test"
            ) and not membership:
                raise EvidencePackError("membership_self_test_required")
            site = name == "site_validation" and kind == "validation_report"
            if (
                set(entry) != {"lane", "type", "sha256"}
                or (not site and name not in profile)
                or name in seen
                or (
                    not site
                    and not clinical
                    and not membership
                    and not human
                    and kind
                    not in {"gate_report", "benchmark_report", "review_metrics"}
                )
                or not re.fullmatch("[0-9a-f]{64}", digest)
            ):
                raise EvidencePackError("invalid_report_index")
            seen.add(name)
            try:
                raw = _read_report_file(inputs / (digest + ".json"))
            except FileNotFoundError:
                if site:
                    lanes[name], _ = site_validation_lane({}, site_validation_key)
                    lanes[name].update(
                        state="unknown", reason_code="site_validation_missing"
                    )
                continue
            if site:
                if hashlib.sha256(raw).hexdigest() != digest:
                    lanes[name], _ = site_validation_lane({}, site_validation_key)
                    lanes[name]["reason_code"] = "site_validation_unverified"
                else:
                    try:
                        study = _parse(raw)
                    except (ValueError, UnicodeError):
                        study = {}
                    lanes[name], site_inputs = site_validation_lane(
                        study, site_validation_key
                    )
                continue
            if hashlib.sha256(raw).hexdigest() != digest:
                raise EvidencePackError("report_digest_mismatch")
            if clinical:
                state, metrics, reason = clinical_lane(
                    name, _parse(raw), credentialed=name == optional
                )
                if reason:
                    lanes[name]["reason_code"] = reason
                if state == "success" and name == optional:
                    lanes[name]["claims"] = ["credentialed_corpus_measured"]
            elif membership:
                state, metrics = membership_lane(_parse(raw))
            elif human:
                state, metrics, human_inputs = human_factors_lane(
                    capability, _parse(raw)
                )
            else:
                state, metrics = _project(_parse(raw), kind, name)
            if not isinstance(metrics, dict):
                raise EvidencePackError("invalid_report_metrics")
            observed = {
                key: metrics[key]
                for key in profile[name]
                if metrics.get(key) is not None
            }
            if state == "success" and len(observed) != len(profile[name]):
                state = "partial"
            lanes[name].update(
                state=state, metrics=observed, report_digest="sha256:" + digest
            )
        return ValidationEvidencePack(
            {
                "schema_version": "1.0.0",
                "version": version,
                "capability": capability,
                "regulatory_tier": declaration.regulatory_tier.value,
                "intended_use_digest": declaration.digest(),
                "generated_at": generated_at.astimezone(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "lanes": list(lanes.values()),
                "claims": sorted(
                    {claim for lane in lanes.values() for claim in lane["claims"]}
                ),
                "exceptions": [],
                "licenses": [],
                "frozen_inputs": [
                    {
                        "id": lane["id"],
                        "origin": "credentialed"
                        if lane["id"] == optional and lane["claims"]
                        else "aggregate",
                        "digest": lane["report_digest"],
                    }
                    for lane in lanes.values()
                    if lane["report_digest"]
                ]
                + site_inputs
                + human_inputs,
                "limitations": ["not_clinically_validated", "not_a_medical_device"],
                "signature": None,
            }
        )
    except EvidencePackError:
        raise
    except (
        OSError,
        AttributeError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
    ):
        raise EvidencePackError("invalid_report_input") from None


def write_evidence_pack(
    pack: ValidationEvidencePack, output: Path, *, overwrite: bool = False
) -> Path:
    """Publish one complete JSON file, refusing existing destinations by default."""
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
            staged = Path(scratch) / "pack.json"
            with staged.open("w", encoding="utf-8") as handle:
                handle.write(pack.to_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if overwrite:
                os.replace(staged, output)
            else:
                os.link(staged, output)
        return output
    except FileExistsError:
        raise EvidencePackError("output_exists") from None
    except (OSError, ValueError):
        raise EvidencePackError("pack_write_failed") from None
