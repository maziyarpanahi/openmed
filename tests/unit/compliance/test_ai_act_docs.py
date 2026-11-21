"""Verify the generator against the reviewed Markdown field specification."""

import hashlib
import json
from pathlib import Path

from openmed.cli.main import main
from openmed.compliance.ai_act_docs import (
    ACCOUNTABLE,
    MISSING,
    NOTICE,
    build_draft,
    render_markdown,
)


def fields(draft):
    return {r["field"]: r for rows in draft["sections"].values() for r in rows}


def test_all_template_rows_and_human_owned_gaps(capsys):
    assert (
        main(
            [
                "compliance",
                "ai-act-docs",
                "--capability",
                "deidentify",
                "--format",
                "json",
            ]
        )
        == 0
    )
    draft = json.loads(capsys.readouterr().out)
    template = (
        (
            Path(__file__).resolve().parents[3]
            / "docs/compliance/model-card-eu-ai-act-fields.md"
        )
        .read_text()
        .split("```markdown")[1]
        .split("```")[0]
    )
    expected = [
        r.split("|")[1].strip()
        for r in template.splitlines()
        if r.startswith("| ") and not r.startswith(("| Field |", "| Metric |"))
    ]
    rows = fields(draft)
    assert list(rows) == expected
    for line in template.splitlines():
        if line.startswith("| ") and "[" in line:
            assert rows[line.split("|")[1].strip()]["owner"] == ACCOUNTABLE
    assert rows["AI Act classification"]["value"] == MISSING
    assert rows["Exact-span F1"]["value"] == MISSING
    assert rows["Prohibited uses"]["value"]
    assert render_markdown(draft).startswith(f"> **{NOTICE}**")


def test_benchmark_is_bound_and_copied_without_text(tmp_path):
    row = {
        "repo_id": "OpenMed/synthetic-pii",
        "family": "PII",
        "task": "token-classification",
        "languages": ["en"],
        "tier": None,
        "param_count": 100,
        "architecture": "bert",
        "base_model": None,
        "formats": ["pytorch"],
        "canonical_labels": ["PERSON"],
        "benchmark": {"dataset": None, "micro_f1": None, "recall": None},
        "arxiv": None,
        "license": "mit",
        "reproducibility_hash": "sha256:" + "a" * 64,
        "released": "2026-01-01",
    }
    from openmed.core.manifest_schema import SCRIPT_COVERAGE_TARGETS

    row["script_coverage"] = {
        script: {
            "unk_rate": 0.0,
            "byte_fallback_rate": 0.0,
            "tokens_per_grapheme": 1.0,
            "verdict": "unclaimed",
        }
        for script in SCRIPT_COVERAGE_TARGETS
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row))
    bench = tmp_path / "benchmark.json"
    bench.write_text(
        json.dumps(
            {
                "model_name": row["repo_id"],
                "suite": "synthetic",
                "device": "cpu",
                "fixture_count": 6,
                "generated_at": "2026-01-01T00:00:00Z",
                "metrics": {
                    "exact_span_f1": {"f1": 0.812345678901},
                    "relaxed_span_f1": {"f1": 0},
                    "recall_slices": {
                        "by_label": {"PERSON": 0.82, "PRIVATE_SENTINEL": 1}
                    },
                },
                "metadata": {"text": "PRIVATE_SENTINEL"},
            }
        )
    )
    draft = build_draft(
        "deidentify", model=row["repo_id"], manifest=manifest, benchmark=bench
    )
    rows = fields(draft)
    assert rows["Exact-span F1"]["value"] == 0.812345678901
    assert rows["Relaxed-span F1"]["value"] == 0
    assert (
        draft["source_digests"]["benchmark"]
        == "sha256:" + hashlib.sha256(bench.read_bytes()).hexdigest()
    )
    assert rows["Integrity and provenance"]["value"]["license"] == row["license"]
    assert "PRIVATE_SENTINEL" not in render_markdown(draft)
    assert str(tmp_path) not in json.dumps(draft)


def test_wrong_pack_and_unusable_private_artifacts_fail_value_free(tmp_path, capsys):
    secret = tmp_path / "SECRET_SOURCE"
    secret.write_text('{"private":"SECRET_INPUT"}')
    assert (
        main(
            [
                "compliance",
                "ai-act-docs",
                "--capability",
                "deidentify",
                "--evidence-pack",
                str(secret),
            ]
        )
        == 2
    )
    output = capsys.readouterr()
    assert not output.out and "SECRET" not in output.err


def test_signed_pack_and_verified_capa_are_projected_without_local_notes(tmp_path):
    import pytest

    from openmed.compliance.corrective_action import (
        CorrectiveAction,
        CorrectiveActionLedger,
    )
    from openmed.core.intended_use import load_intended_use_registry
    from openmed.eval.evidence_pack import ValidationEvidencePack
    from openmed.eval.evidence_pack.verify import sign_evidence_pack

    source = (
        Path(__file__).resolve().parents[2]
        / "fixtures/evidence_pack/deidentification_synthetic_v1/pack.json"
    )
    payload = json.loads(source.read_bytes())
    declaration = load_intended_use_registry().declarations["deidentify"]
    payload["intended_use_digest"] = declaration.digest()
    key = tmp_path / "key"
    key.write_bytes(b"s" * 32)
    pack = tmp_path / "evidence.json"
    pack.write_text(
        sign_evidence_pack(
            ValidationEvidencePack(payload), key_id="synthetic", key=key.read_bytes()
        ).to_json()
    )
    ledger = CorrectiveActionLedger()
    ledger.open(
        CorrectiveAction(
            record_id="CAPA-2026-0001",
            source="release_gate",
            severity="high",
            capabilities=("deidentify",),
            versions=("4.3.0",),
            local_note="PRIVATE_NOTE",
        )
    )
    capa = tmp_path / "actions.json"
    capa.write_text(json.dumps(ledger.to_dict()))
    draft = build_draft("deidentify", evidence_pack=pack, pack_key=key, capa=capa)
    rows = fields(draft)
    assert rows["Drift and incident monitoring"]["value"][0]["state"] == "open"
    assert (
        rows["Robustness results and failures"]["value"][0]["metrics"]
        == payload["lanes"][0]["metrics"]
    )
    assert "PRIVATE_NOTE" not in render_markdown(draft)
    key.write_bytes(b"x" * 32)
    with pytest.raises(ValueError, match="invalid_pack_signature"):
        build_draft("deidentify", evidence_pack=pack, pack_key=key)
