"""Synthetic privacy-disclosure inputs for model-card tests.

Every value is a controlled code, a count, a band or a synthetic digest. No
site identity, per-site metric or source payload appears in these fixtures.
"""

from __future__ import annotations

import copy
from typing import Any

from openmed.eval.release_gates import RELEASABLE, GateCheck, GateReport

SIGNING_KEY = "unit-model-card-privacy-key"
REPO_ID = "OpenMed/OpenMed-PII-Tiny-44M-v1-mlx"
PRIVACY_DISCLOSURE_SCHEMA = "openmed.model_card.privacy_disclosure.v1"
LINEAGE_DIGEST = "sha256:" + "a" * 64

# Sentinel values that must never reach a rendered card or an error message.
SITE_SENTINEL = "Synthetic Riverside Site 07"
SITE_METRIC_SENTINEL = "0.987654321"


def dp_federated_disclosure() -> dict[str, Any]:
    """Return a complete synthetic federated DP disclosure."""

    return {
        "schema_version": PRIVACY_DISCLOSURE_SCHEMA,
        "training_setting": "federated",
        "mechanism": "dp_fedavg_gaussian",
        "privacy_unit": "patient",
        "adjacency": "add_remove",
        "epsilon": 3.0,
        "delta": 1e-06,
        "accountant": "rdp",
        "conversion": "rdp_to_approx_dp",
        "iteration_unit": "rounds",
        "iteration_count": 200,
        "sampling": "poisson",
        "amplification_claimed": True,
        "clipping": "per_unit_l2",
        "secure_aggregation": "cryptographic",
        "participant_band": "fourfold_or_more",
        "tuning_cost_included": True,
        "limitations": [
            "public_pretraining_data_not_covered",
            "group_privacy_not_covered",
        ],
    }


def dp_centralized_disclosure() -> dict[str, Any]:
    """Return a complete synthetic centralized DP-SGD disclosure."""

    return {
        "schema_version": PRIVACY_DISCLOSURE_SCHEMA,
        "training_setting": "centralized",
        "mechanism": "dp_sgd_gaussian",
        "privacy_unit": "note",
        "adjacency": "replace_one",
        "epsilon": 8,
        "delta": 1e-05,
        "accountant": "prv",
        "conversion": "direct",
        "iteration_unit": "steps",
        "iteration_count": 1200,
        "sampling": "shuffled",
        "amplification_claimed": False,
        "clipping": "per_example_l2",
        "secure_aggregation": "not_applicable",
        "participant_band": "not_applicable",
        "tuning_cost_included": False,
        "limitations": ["inference_inputs_not_covered"],
    }


def federated_without_dp_disclosure() -> dict[str, Any]:
    """Return a synthetic federated disclosure with no DP mechanism."""

    return {
        "schema_version": PRIVACY_DISCLOSURE_SCHEMA,
        "training_setting": "federated",
        "mechanism": "none",
        "iteration_unit": "rounds",
        "iteration_count": 50,
        "clipping": "per_client_update_l2",
        "secure_aggregation": "cryptographic",
        "participant_band": "double_to_under_fourfold",
        "limitations": ["secure_aggregation_not_dp"],
    }


def promotion_evidence(statement_digest: str) -> dict[str, str]:
    """Return injected promotion-evidence digests for one disclosure."""

    return {
        "accountant_statement_digest": statement_digest,
        "lineage_digest": LINEAGE_DIGEST,
    }


def manifest_row() -> dict[str, Any]:
    """Return a synthetic manifest row with no provenance path."""

    return {
        "repo_id": REPO_ID,
        "family": "PII",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Tiny",
        "param_count": 44_000_000,
        "architecture": "deberta-v2",
        "base_model": "OpenMed/OpenMed-PII-Tiny-44M",
        "formats": ["mlx-fp"],
        "canonical_labels": ["PERSON", "DATE"],
        "benchmark": {
            "dataset": "synthetic-card-fixture",
            "micro_f1": 0.99,
            "recall": 0.995,
        },
        "arxiv": "2508.01630",
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:" + "1" * 64,
        "released": "2026-07-05",
    }


def training_provenance() -> dict[str, Any]:
    """Return synthetic in-memory training provenance."""

    return {
        "base_model": "OpenMed/OpenMed-PII-Tiny-44M",
        "base_model_revision": "main",
        "data_manifest_hash": "sha256:" + "2" * 64,
        "recipe_config_hash": "sha256:" + "3" * 64,
        "env_lock_digest": "sha256:" + "4" * 64,
        "git_sha": "abc123",
        "rng_seeds": {"python": 123, "numpy": 456},
        "license_tags": ["synthetic-openmed"],
    }


def gate_report() -> GateReport:
    """Return a signed synthetic release-gate report."""

    return GateReport(
        repo_id=REPO_ID,
        family="PII",
        tier="Tiny",
        param_count=44_000_000,
        format="mlx-fp",
        per_label_recall={"PERSON": 0.995, "DATE": 0.992},
        per_label_precision={"PERSON": 0.994, "DATE": 0.991},
        critical_leakage_count=0,
        residual_leakage_rate=0.0,
        quant_recall_delta=0.0,
        p50_ms=32.0,
        p95_ms=81.0,
        ram_mb=128.0,
        eval_set_hash="sha256:eval",
        leakage_fixture_hash="sha256:leakage",
        decision=RELEASABLE,
        gate_results=(
            GateCheck(
                "G5",
                True,
                reason="ok",
                details={
                    "tier": "tiny",
                    "budget": {"ram_mb": 350.0, "p50_ms": 60.0, "p95_ms": 150.0},
                    "violations": {},
                },
            ),
        ),
        policy="hipaa_safe_harbor",
        threshold_profile="strict",
        target_leakage_rate=0.005,
    ).sign(SIGNING_KEY)


def without(payload: dict[str, Any], *fields: str) -> dict[str, Any]:
    """Return a deep copy of *payload* without *fields*."""

    result = copy.deepcopy(payload)
    for field in fields:
        result.pop(field, None)
    return result
