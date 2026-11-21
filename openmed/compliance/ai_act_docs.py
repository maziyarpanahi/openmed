"""Render the reviewed model-card field specification as a local deployment draft."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from openmed.compliance.corrective_action import CorrectiveActionLedger
from openmed.core.audit import AuditReport
from openmed.core.intended_use import load_intended_use_registry
from openmed.core.labels import CANONICAL_LABELS
from openmed.core.manifest import load_manifest_rows
from openmed.core.manifest_schema import validate_manifest_row
from openmed.eval.evidence_pack import ValidationEvidencePack
from openmed.eval.evidence_pack.verify import sign_evidence_pack
from openmed.eval.release_gates import GateReport

NOTICE = "Template — requires legal review. Not legal advice."
MISSING = "Not provided"
ACCOUNTABLE = "Requires an accountable person"
# Field order and ownership follow docs/compliance/model-card-eu-ai-act-fields.md.
FIELDS = {
    "System and regulatory context": (
        ("Model and version", False),
        ("Artifact formats and hashes", False),
        ("Provider/deployer roles", True),
        ("AI Act classification", True),
        ("GDPR processing role", True),
        ("Technical-documentation owner", True),
        ("Review date and status", True),
    ),
    "Intended purpose": (
        ("Model task and family", False),
        ("Intended clinical or operational purpose", True),
        ("Intended users and affected persons", True),
        ("Intended languages and environments", False),
        ("Integration and deployment mode", True),
        ("Permitted decisions", True),
        ("Prohibited uses", True),
    ),
    "Known limitations and reasonably foreseeable misuse": (
        ("Unsupported populations, languages, or inputs", True),
        ("Data and evaluation limitations", True),
        ("Known failure modes", True),
        ("Release-gate failures or blocked formats", True),
        ("Foreseeable misuse", True),
        ("Required local validation", True),
    ),
    "Accuracy and leakage metrics": (
        ("Exact-span F1", False),
        ("Relaxed-span F1", False),
        ("Character recall", False),
        ("Direct-identifier recall by label/language", True),
        ("Character leakage", True),
        ("Critical leakage count", False),
        ("Residual leakage rate and target", True),
        ("Quantization recall delta", True),
    ),
    "Human oversight": (
        ("Oversight owner and qualifications", True),
        ("Review point", True),
        ("Information shown to reviewer", True),
        ("Escalation and abstention triggers", True),
        ("Override and stop authority", True),
        ("Automation restriction", True),
        ("Oversight effectiveness review", True),
    ),
    "Robustness, cybersecurity, and monitoring evidence": (
        ("Robustness suites", True),
        ("Robustness results and failures", True),
        ("Evaluation and fixture hashes", True),
        ("Device/format performance envelope", True),
        ("Integrity and provenance", True),
        ("Cybersecurity and local-processing controls", True),
        ("Drift and incident monitoring", True),
    ),
    "GDPR and de-identification evidence": (
        ("Processing purpose and lawful basis", True),
        ("OpenMed policy profile", True),
        ("Audit residual-risk score", False),
        ("Other audit risk indicators", False),
        ("Pseudonymization and key custody", True),
        ("DPIA and DPA records", True),
        ("Retention and deletion", True),
        ("Data-subject rights process", True),
    ),
}


def _number(value):
    if value is None:
        return MISSING
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid_aggregate_metric")
    return value


def _code(value):
    if value is None or value == "":
        return MISSING
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,95}", value
    ):
        raise ValueError("invalid_evidence_code")
    return value


def build_draft(
    capability: str,
    *,
    model=None,
    manifest=None,
    benchmark=None,
    gate=None,
    evidence_pack=None,
    capa=None,
    audit=None,
    pack_key=None,
    gate_key=None,
    audit_key=None,
):
    """Project reviewed declarations and caller-selected local artifacts onto every template field.

    File arguments identify local evidence. Signed sources require their matching
    key file. Source digests bind the unchanged input bytes; private paths and
    unselected source properties are omitted from both output formats.
    """
    registry = load_intended_use_registry()
    declaration = registry.declarations[capability]
    sources = {"intended_use": declaration.digest()}
    values = {
        "Intended clinical or operational purpose": declaration.purpose,
        "Intended users and affected persons": list(declaration.intended_users),
        "Intended languages and environments": {
            "registry_languages": list(declaration.languages)
        },
        "Prohibited uses": list(declaration.excluded_uses),
        "Review point": declaration.human_oversight,
        "Automation restriction": {
            "autonomous_clinical_action": declaration.autonomous_clinical_action
        },
    }

    def read(name, path):
        raw = Path(path).read_bytes()
        sources[name] = "sha256:" + hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    if manifest is not None and model is None:
        raise ValueError("model_required")
    if model is not None:
        rows = load_manifest_rows(manifest) if manifest else load_manifest_rows()
        matches = [row for row in rows if row.get("repo_id") == model]
        if len(matches) != 1 or validate_manifest_row(matches[0], 1):
            raise ValueError("invalid_manifest_selection")
        row = matches[0]
        from openmed.core.audit import stable_hash

        sources["manifest_row"] = stable_hash(row)
        values.update(
            {
                "Model and version": {
                    "repo_id": model,
                    "released": row["released"] or MISSING,
                },
                "Artifact formats and hashes": {
                    "formats": row["formats"],
                    "reproducibility_hash": row["reproducibility_hash"] or MISSING,
                },
                "Model task and family": {"task": row["task"], "family": row["family"]},
                "Integrity and provenance": {
                    "license": row["license"] or MISSING,
                    "reproducibility_hash": row["reproducibility_hash"] or MISSING,
                },
            }
        )
        values["Intended languages and environments"].update(
            languages=row["languages"], tier=row["tier"] or MISSING
        )
    if benchmark:
        b = read("benchmark", benchmark)
        if not model or b.get("model_name") != model:
            raise ValueError("benchmark_model_mismatch")
        m = b["metrics"]
        context = {
            "suite": _code(b["suite"]),
            "device": _code(b["device"]),
            "timestamp": _code(b.get("generated_at")),
            "fixture_count": _number(b["fixture_count"]),
            "source_digest": sources["benchmark"],
        }
        values["Evaluation and fixture hashes"] = context
        values["Intended languages and environments"]["device"] = context["device"]
        for field, group, key in [
            ("Exact-span F1", "exact_span_f1", "f1"),
            ("Relaxed-span F1", "relaxed_span_f1", "f1"),
            ("Character recall", "character_recall", "rate"),
            ("Character leakage", "leakage", "overall"),
        ]:
            values[field] = _number(m.get(group, {}).get(key))
        slices = m.get("recall_slices", {})
        by_label = {
            k: _number(v)
            for k, v in slices.get("by_label", {}).items()
            if k in CANONICAL_LABELS
        }
        from openmed.core.pii_i18n import SUPPORTED_LANGUAGES

        by_language = {
            k: _number(v)
            for k, v in slices.get("by_language", {}).items()
            if k in SUPPORTED_LANGUAGES
        }
        if by_label or by_language:
            values["Direct-identifier recall by label/language"] = {
                "by_label": by_label,
                "by_language": by_language,
            }
    if gate:
        data = read("gate_report", gate)
        report = GateReport.from_dict(data)
        if (
            not model
            or report.repo_id != model
            or not gate_key
            or not report.verify(Path(gate_key).read_bytes())
        ):
            raise ValueError("invalid_gate_report")
        from openmed.eval.model_card_builder import _validate_gate_manifest_identity

        _validate_gate_manifest_identity(row, report.to_dict())
        values.update(
            {
                "Critical leakage count": _number(report.critical_leakage_count),
                "Residual leakage rate and target": {
                    "rate": _number(report.residual_leakage_rate),
                    "target": _number(report.target_leakage_rate),
                },
                "Release-gate failures or blocked formats": {
                    "decision": _code(report.decision),
                    "blocked_formats": [_code(v) for v in report.blocked_formats],
                    "checks": [
                        {"gate": _code(c.gate), "passed": c.passed}
                        for c in report.gate_results
                    ],
                },
                "Quantization recall delta": _number(report.quant_recall_delta),
                "Device/format performance envelope": {
                    "format": _code(report.format),
                    "p50_ms": _number(report.p50_ms),
                    "p95_ms": _number(report.p95_ms),
                    "ram_mb": _number(report.ram_mb),
                },
            }
        )
        values.setdefault("Evaluation and fixture hashes", {}).update(
            leakage_fixture_hash=_code(report.leakage_fixture_hash),
            eval_set_hash=_code(report.eval_set_hash),
            policy=_code(report.policy),
            threshold_profile=_code(report.threshold_profile),
        )
    if evidence_pack:
        data = read("evidence_pack", evidence_pack)
        pack = ValidationEvidencePack(data)
        sig = data["signature"]
        if not (
            pack_key
            and sig
            and data["capability"] == capability
            and data["intended_use_digest"] == declaration.digest()
        ):
            raise ValueError("invalid_capability_pack")
        expected = sign_evidence_pack(
            pack, key_id=sig["key_id"], key=Path(pack_key).read_bytes()
        ).to_dict()["signature"]
        if expected != sig:
            raise ValueError("invalid_pack_signature")
        values["Robustness results and failures"] = [
            {
                "lane": v["id"],
                "state": v["state"],
                "metrics": v["metrics"],
                "report_digest": v["report_digest"] or MISSING,
            }
            for v in data["lanes"]
        ]
        values["Data and evaluation limitations"] = data["limitations"]
    if capa:
        ledger = CorrectiveActionLedger.from_dict(read("corrective_actions", capa))
        values["Drift and incident monitoring"] = [
            {
                "state": r.state,
                "source": r.source,
                "severity": r.severity,
                "artifact_digests": list(r.artifact_digests),
            }
            for r in ledger.records()
            if capability in r.capabilities
        ] or MISSING
    if audit:
        report = AuditReport.from_dict(read("audit_report", audit))
        if not audit_key or not report.verify(Path(audit_key).read_bytes()):
            raise ValueError("invalid_audit_report")
        values["OpenMed policy profile"] = _code(report.policy)
        values["Audit residual-risk score"] = _number(
            report.residual_risk.get("risk_report_record_score")
        )
        summary = report.residual_risk.get("risk_report", {})
        values["Other audit risk indicators"] = {
            "projected_leakage": _number(report.residual_risk.get("projected_leakage")),
            **{
                k: _number(summary[k])
                for k in ("leakage_rate", "reid_rate", "k_min")
                if k in summary
            },
        }
    return {
        "notice": NOTICE,
        "capability": capability,
        "status": "draft",
        "source_digests": sources,
        "sections": {
            section: [
                {
                    "field": field,
                    "value": values.get(field, MISSING),
                    **({"owner": ACCOUNTABLE} if owned else {}),
                }
                for field, owned in rows
            ]
            for section, rows in FIELDS.items()
        },
    }


def render_markdown(draft: dict) -> str:
    """Render the same fields, source bindings and accountability notes as JSON."""
    lines = [
        f"> **{NOTICE}**",
        "",
        f"# Technical documentation draft for {draft['capability']}",
    ]
    for section, rows in draft["sections"].items():
        lines.extend(
            [
                "",
                f"## {section}",
                "",
                "| Field | Evidence | Deployment review |",
                "|---|---|---|",
            ]
        )
        for row in rows:
            value = row["value"]
            if not isinstance(value, str):
                value = json.dumps(
                    value, ensure_ascii=False, sort_keys=True, allow_nan=False
                )
            value = value.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['field']} | {value} | {row.get('owner', '')} |")
    lines.extend(["", "## Source digests", ""])
    lines.extend(
        f"- {name} `{digest}`" for name, digest in draft["source_digests"].items()
    )
    return "\n".join(lines) + "\n"
