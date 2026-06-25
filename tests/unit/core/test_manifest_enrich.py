"""Tests for manifest enrichment from benchmark and latency results."""

from __future__ import annotations

import json

import pytest

from scripts.manifest import enrich_manifest


def test_enrich_manifest_merges_measurements_by_repo_id_and_preserves_unmatched():
    matched = _manifest_row("OpenMed/OpenMed-PII-Fixture-Tiny-65M")
    unmatched = _manifest_row("OpenMed/OpenMed-NER-Unmatched-Base-184M")
    unmatched_line = json.dumps(unmatched, indent=2)
    lines = [
        json.dumps(matched, separators=(",", ":")) + "\n",
        unmatched_line + "\n",
    ]
    measurements = {
        "OpenMed/OpenMed-PII-Fixture-Tiny-65M": {
            "latency_ms": {"iphone_15_pro": 18.4, "m2_air": 7.0},
            "peak_ram_mb": {"iphone_15_pro": 512, "m2_air": 384},
            "recommended_tier": "phone",
            "benchmark": [
                {
                    "suite": "shield",
                    "dataset": "openmed-golden-pii",
                    "micro_f1": 0.9823,
                    "recall": 0.991,
                    "leakage": 0.0,
                }
            ],
        }
    }

    output, updated = enrich_manifest.enrich_manifest_lines(lines, measurements)

    assert updated == 1
    enriched = json.loads(output[0])
    assert enriched["latency_ms"] == {"iphone_15_pro": 18.4, "m2_air": 7.0}
    assert enriched["peak_ram_mb"] == {"iphone_15_pro": 512, "m2_air": 384}
    assert enriched["recommended_tier"] == "phone"
    assert enriched["benchmark"][0]["suite"] == "shield"
    assert output[1] == unmatched_line + "\n"


def test_enrich_manifest_file_accepts_results_list(tmp_path):
    manifest = tmp_path / "models.jsonl"
    results = tmp_path / "results.json"
    output = tmp_path / "enriched.jsonl"
    row = _manifest_row("OpenMed/OpenMed-PII-Fixture-Tiny-65M")
    manifest.write_text(json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8")
    results.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "repo_id": row["repo_id"],
                        "latency_ms": {"m2_air": 7.0},
                        "peak_ram_mb": {"m2_air": 384},
                        "recommended_tier": "laptop",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    updated = enrich_manifest.enrich_manifest_file(manifest, results, output)

    assert updated == 1
    enriched = json.loads(output.read_text(encoding="utf-8"))
    assert enriched["latency_ms"] == {"m2_air": 7.0}
    assert enriched["peak_ram_mb"] == {"m2_air": 384}
    assert enriched["recommended_tier"] == "laptop"


def test_enrich_manifest_rejects_invalid_enriched_rows():
    row = _manifest_row("OpenMed/OpenMed-PII-Fixture-Tiny-65M")

    with pytest.raises(ValueError, match="recommended_tier"):
        enrich_manifest.enrich_manifest_lines(
            [json.dumps(row, separators=(",", ":")) + "\n"],
            {row["repo_id"]: {"recommended_tier": "watch"}},
        )


def _manifest_row(repo_id: str) -> dict[str, object]:
    return {
        "repo_id": repo_id,
        "family": "PII" if "PII" in repo_id else "NER",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Tiny",
        "param_count": 65_000_000,
        "architecture": "distilbert",
        "base_model": None,
        "formats": ["pytorch"],
        "canonical_labels": ["PERSON", "DATE"],
        "benchmark": {"dataset": None, "micro_f1": None, "recall": None},
        "arxiv": None,
        "license": "apache-2.0",
        "reproducibility_hash": (
            "sha256:1111111111111111111111111111111111111111111111111111111111111111"
        ),
        "released": "2026-06-24",
    }
