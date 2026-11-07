"""Regenerate the cross-kit bundle-cache state-machine and recovery vectors.

The vectors drive the generation caches in OpenMedKit for Android
(``BundleGenerationCache``), OpenMedKit for Swift (``OpenMedBundleCache``), and
the ``openmed`` web package (``BundleGenerationCache``). Each scenario is a
script of cache operations, crash checkpoints, restarts, and tampering, with the
exact value-free result every implementation must produce. The contract lives
in ``docs/models/bundle-cache.md``.

TEST-ONLY KEYS. The bundles are signed with the TEST-ONLY keys of
``tests/fixtures/portable_bundle/generate_vectors.py``. Never trust them in
production.

Usage::

    python tests/fixtures/bundle_cache/generate_vectors.py          # write
    python tests/fixtures/bundle_cache/generate_vectors.py --check  # verify

All bundle files are tiny synthetic text placeholders. The vectors hold
states, counts, sizes, digests, and reason codes only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.core import portable_bundle as pb  # noqa: E402

VECTORS_PATH = Path(__file__).with_name("vectors.json")
PORTABLE_GENERATOR = (
    ROOT / "tests" / "fixtures" / "portable_bundle" / "generate_vectors.py"
)

CHUNK_BYTES = 16
"""Resume checkpoint size used by every scenario (production default: 4 MiB)."""

QUOTA_UNLIMITED = 1 << 40
LEASE_TTL_MS = 60_000
STAGING_MAX_AGE_MS = 86_400_000
CLOCK = {"start_ms": 1_000_000, "step_ms": 1_000}

CHECKPOINTS = (
    "staging_record_written",
    "chunk_committed",
    "verifying_record_written",
    "generation_committed",
    "pointer_committed",
    "eviction_committed",
)
"""Named crash points, in the order an acquisition passes them."""

STATES = (
    "absent",
    "staging",
    "resumable",
    "verifying",
    "ready",
    "active",
    "superseded",
    "evictable",
    "quarantined",
    "evicted",
)

REASONS = (
    "bundle_rejected",
    "lock_timeout",
    "quota_exceeded",
    "storage_pressure",
    "transfer_interrupted",
    "cancelled",
    "range_mismatch",
    "verification_failed",
    "storage_io",
    "generation_not_ready",
    "generation_corrupt",
    "no_active_generation",
    "rollback_unavailable",
    "lease_unknown",
)

RESUME_REASONS = (
    "source_changed",
    "trust_policy_changed",
    "chunk_size_changed",
    "partial_digest_mismatch",
    "range_not_honored",
)

QUARANTINE_REASONS = (
    "file_missing",
    "file_size_mismatch",
    "file_digest_mismatch",
    "file_undeclared",
    "payload_mismatch",
    "rolled_back",
)
"""Reasons recorded with a quarantined generation."""

RECOVERY_ACTIONS = (
    "stale_lock_cleared",
    "trash_purged",
    "quarantine_purged",
    "staging_orphaned",
    "staging_expired",
    "staging_resumable",
    "generation_quarantined",
    "pointer_repaired",
    "lease_expired",
)


def _portable() -> Any:
    spec = importlib.util.spec_from_file_location(
        "portable_bundle_vector_generator", PORTABLE_GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PB = _portable()


# ---------------------------------------------------------------------------
# Identities shared by every kit.


def sha256_text(data: bytes) -> str:
    """Return ``sha256:<hex>`` for ``data``."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def bundle_key(bundle_id: str) -> str:
    """Return the opaque directory key of a bundle id."""

    return "b-" + hashlib.sha256(bundle_id.encode("utf-8")).hexdigest()[:32]


def generation_id(content_id: str) -> str:
    """Return the opaque generation id of a manifest content identity."""

    return "g-" + content_id[len("sha256:") : len("sha256:") + 32]


def trust_digest(policy: dict) -> str:
    """Return the trust-policy binding digest recorded with resumable bytes."""

    keys = sorted(policy["keys"], key=lambda item: item["kid"].encode("utf-8"))
    value = {
        "keys": [
            {
                "kid": key["kid"],
                "not_after": key["not_after"],
                "not_before": key["not_before"],
                "revoked": key["status"] == "revoked",
                "x": key["x"],
            }
            for key in keys
        ],
        "revoked_content_ids": sorted(policy["revoked_content_ids"]),
        "threshold": policy["threshold"],
    }
    return sha256_text(pb.canonical_json_bytes(value))


# ---------------------------------------------------------------------------
# Synthetic signed bundles.


def _bundle_value(version: str, sequence: int, changes: dict[str, str]) -> tuple:
    tree = dict(PB.GOLDEN_TREE)
    tree.update(changes)
    value = PB.golden_manifest()
    value["bundle"]["version"] = version
    value["bundle"]["sequence"] = sequence
    value["files"] = [
        PB.file_entry(path, tree[path])
        for path in sorted(tree, key=lambda item: item.encode("utf-8"))
    ]
    return value, tree


def _bundle_entry(value: dict, tree: dict[str, str], kids: list[str]) -> dict:
    manifest = PB.canonical_text(value)
    envelope = PB.sign(manifest, kids)
    content = pb.content_id(manifest.encode("utf-8"))
    files = [
        {"path": path, "text": tree[path]}
        for path in sorted(tree, key=lambda item: item.encode("utf-8"))
    ]
    return {
        "bundle_id": value["bundle"]["id"],
        "content_id": content,
        "envelope": envelope,
        "files": files,
        "generation": generation_id(content),
        "manifest": manifest,
        "total_bytes": sum(len(item["text"].encode("utf-8")) for item in files),
    }


def build_bundles() -> dict[str, dict]:
    """Return the synthetic bundles keyed by alias."""

    root = ["openmed-test-root-a"]
    model = "onnx/model.onnx"
    int8 = "onnx/model_int8.onnx"
    v2_change = {model: "synthetic fp32 onnx graph placeholder, generation two\n"}
    bundles = {
        "v1": _bundle_entry(*_bundle_value("1.2.0", 7, {}), root),
        "v2": _bundle_entry(*_bundle_value("1.3.0", 8, v2_change), root),
        "v3": _bundle_entry(
            *_bundle_value(
                "1.4.0",
                9,
                {int8: "synthetic int8 onnx graph placeholder, generation three\n"},
            ),
            root,
        ),
        "v4": _bundle_entry(
            *_bundle_value(
                "1.5.0",
                10,
                {
                    model: (
                        "synthetic fp32 onnx graph placeholder, generation four, "
                        "the largest\n"
                    )
                },
            ),
            root,
        ),
        "v2_untrusted": _bundle_entry(
            *_bundle_value("1.3.0", 8, v2_change), ["openmed-test-rogue"]
        ),
    }
    schema_value, schema_tree = _bundle_value("1.3.0", 8, v2_change)
    schema_value["schema_version"] = 2
    bundles["v2_schema"] = _bundle_entry(schema_value, schema_tree, root)
    sizes = [bundles[name]["total_bytes"] for name in ("v1", "v2", "v3", "v4")]
    if sizes != sorted(sizes) or len(set(sizes)) != len(sizes):
        raise SystemExit("bundle sizes must strictly increase from v1 to v4")
    return bundles


BUNDLES = build_bundles()


# ---------------------------------------------------------------------------
# Byte arithmetic for the expectations (chunk commits in manifest order).


def file_sizes(alias: str) -> list[tuple[str, int]]:
    """Return ``(path, size)`` pairs in manifest order."""

    return [
        (item["path"], len(item["text"].encode("utf-8")))
        for item in BUNDLES[alias]["files"]
    ]


def chunk_plan(alias: str) -> list[int]:
    """Return every chunk length an acquisition from zero commits, in order."""

    plan = []
    for _, size in file_sizes(alias):
        offset = 0
        while offset < size:
            length = min(CHUNK_BYTES, size - offset)
            plan.append(length)
            offset += length
    return plan


def total(alias: str) -> int:
    """Return the declared byte total of a bundle."""

    return BUNDLES[alias]["total_bytes"]


def kept_after_failure(alias: str, delivered: int) -> int:
    """Bytes committed when the source fails after ``delivered`` bytes."""

    kept = 0
    for length in chunk_plan(alias):
        if kept + length > delivered:
            break
        kept += length
    return kept


def kept_after_crash(alias: str, occurrence: int) -> int:
    """Bytes committed when the process dies at the n-th chunk commit."""

    return sum(chunk_plan(alias)[:occurrence])


def kept_after_cancel(alias: str, delivered: int) -> int:
    """Bytes committed when cancellation is requested after ``delivered`` bytes."""

    kept = 0
    for length in chunk_plan(alias):
        kept += length
        if kept >= delivered:
            break
    return kept


def file_offset(alias: str, path: str) -> int:
    """Return the offset of ``path`` in the concatenated manifest order."""

    offset = 0
    for name, size in file_sizes(alias):
        if name == path:
            return offset
        offset += size
    raise KeyError(path)


def file_size(alias: str, path: str) -> int:
    """Return one declared file size."""

    return dict(file_sizes(alias))[path]


# ---------------------------------------------------------------------------
# Scenario helpers.

FILES = len(BUNDLES["v1"]["files"])
MODEL = "onnx/model.onnx"
CONFIG = "config.json"


def source(alias: str, identity: str = "source-a", **extra: Any) -> dict:
    """Return a fake range-source description."""

    value = {
        "cancel_after_bytes": None,
        "corrupt": None,
        "extra_bytes": None,
        "fail_after_bytes": None,
        "identity": identity,
        "ignore_range": False,
        "serve": alias,
    }
    value.update(extra)
    return value


def acquire(alias: str, expect: dict, **extra: Any) -> dict:
    """Return an acquire step."""

    step = {
        "activate": True,
        "bundle": alias,
        "confirm": True,
        "crash_at": None,
        "expect": expect,
        "op": "acquire",
        "policy": "default",
        "source": source(alias),
        "storage_available_bytes": None,
    }
    step.update(extra)
    return step


def acquired(
    alias: str,
    *,
    state: str = "active",
    transferred: int | None = None,
    resumed: bool = False,
    resume_reason: str | None = None,
    from_cache: bool = False,
    evicted: tuple[str, ...] = (),
) -> dict:
    """Return the expectation of a successful acquisition."""

    return {
        "bytes_transferred": total(alias) if transferred is None else transferred,
        "evicted": list(evicted),
        "files_verified": FILES,
        "from_cache": from_cache,
        "generation": alias,
        "ok": True,
        "resume_reason": resume_reason,
        "resumed": resumed,
        "state": state,
        "verified_bytes": total(alias),
    }


def failed(reason: str, bundle_reason: str | None = None) -> dict:
    """Return the expectation of a refused operation."""

    return {"bundle_reason": bundle_reason, "ok": False, "reason": reason}


OK = {"ok": True}
CRASHED = {"crashed": True}


def crash(checkpoint: str, occurrence: int = 1) -> dict:
    """Return a crash point."""

    if checkpoint not in CHECKPOINTS:
        raise SystemExit(f"unknown checkpoint {checkpoint}")
    return {"checkpoint": checkpoint, "occurrence": occurrence}


def entry(
    alias: str,
    state: str,
    *,
    protections: tuple[str, ...] = (),
    verified: int | None = None,
) -> dict:
    """Return one expected snapshot generation."""

    return {
        "generation": alias,
        "protections": list(protections),
        "state": state,
        "total_bytes": total(alias),
        "verified_bytes": total(alias) if verified is None else verified,
    }


def snapshot(
    *,
    active: str | None = None,
    last_green: str | None = None,
    rollback: str | None = None,
    generations: tuple[dict, ...] = (),
    quarantined: tuple[tuple[str, str], ...] = (),
) -> dict:
    """Return a snapshot step with its expectation."""

    used = sum(
        item["total_bytes"]
        for item in generations
        if item["state"] in ("ready", "active", "superseded", "evictable")
    )
    ordered = sorted(
        generations, key=lambda item: BUNDLES[item["generation"]]["generation"]
    )
    return {
        "expect": {
            "generations": ordered,
            "pointer": {
                "active": active,
                "last_green": last_green,
                "rollback": rollback,
            },
            "quarantined": [
                {"generation": alias, "reason": reason}
                for alias, reason in sorted(
                    quarantined, key=lambda item: BUNDLES[item[0]]["generation"]
                )
            ],
            "used_bytes": used,
        },
        "op": "snapshot",
    }


def recover(*actions: tuple[str, str | None]) -> dict:
    """Return a recover step with its expected actions in report order."""

    return {
        "expect": {
            "actions": [
                {"action": action, "generation": alias} for action, alias in actions
            ]
        },
        "op": "recover",
    }


def restart(advance_ms: int = 0) -> dict:
    """Return a restart step (new instance, new owner, same storage)."""

    return {"advance_ms": advance_ms, "op": "restart"}


def scenario(sid: str, description: str, steps: list[dict], **options: Any) -> dict:
    """Return one scenario."""

    value = {
        "chunk_bytes": CHUNK_BYTES,
        "lease_ttl_ms": LEASE_TTL_MS,
        "quota_bytes": QUOTA_UNLIMITED,
        "staging_max_age_ms": STAGING_MAX_AGE_MS,
    }
    value.update(options)
    return {"description": description, "id": sid, "options": value, "steps": steps}


A = ("active",)
AG = ("active", "last_green")
R = ("rollback",)
G = ("last_green",)


def build_scenarios() -> list[dict]:
    """Return every scenario in a stable order."""

    v1_t = total("v1")
    half = file_offset("v1", MODEL) + 5
    kept_half = kept_after_failure("v1", half)
    model_offset = file_offset("v1", MODEL)
    partial = model_offset + CHUNK_BYTES + 4
    scenarios = [
        scenario(
            "fresh_acquire_activates",
            "A verified acquisition commits one generation and activates it.",
            [
                snapshot(),
                acquire("v1", acquired("v1")),
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(entry("v1", "active", protections=AG),),
                ),
            ],
        ),
        scenario(
            "reacquire_is_idempotent",
            "A ready generation is served from the cache without transfer.",
            [
                acquire("v1", acquired("v1")),
                acquire("v1", acquired("v1", transferred=0, from_cache=True)),
                restart(),
                recover(),
                acquire("v1", acquired("v1", transferred=0, from_cache=True)),
            ],
        ),
        scenario(
            "interrupted_transfer_resumes",
            "An interrupted transfer keeps only committed chunks and resumes from them.",
            [
                acquire(
                    "v1",
                    failed("transfer_interrupted"),
                    source=source("v1", fail_after_bytes=half),
                ),
                snapshot(generations=(entry("v1", "resumable", verified=kept_half),)),
                acquire(
                    "v1",
                    acquired("v1", transferred=v1_t - kept_half, resumed=True),
                ),
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(entry("v1", "active", protections=AG),),
                ),
            ],
        ),
        scenario(
            "changed_source_discards_partial_bytes",
            "Resume from a different source identity refetches every byte.",
            [
                acquire(
                    "v1",
                    failed("transfer_interrupted"),
                    source=source("v1", fail_after_bytes=half),
                ),
                acquire(
                    "v1",
                    acquired("v1", resume_reason="source_changed"),
                    source=source("v1", identity="source-b"),
                ),
            ],
        ),
        scenario(
            "changed_trust_policy_discards_partial_bytes",
            "Resume under a different trust policy refetches every byte.",
            [
                acquire(
                    "v1",
                    failed("transfer_interrupted"),
                    source=source("v1", fail_after_bytes=half),
                ),
                acquire(
                    "v1",
                    acquired("v1", resume_reason="trust_policy_changed"),
                    policy="rotation",
                ),
            ],
        ),
        scenario(
            "tampered_partial_bytes_are_truncated",
            "A staged chunk that no longer matches its digest is dropped with the rest of its file.",
            [
                acquire(
                    "v1",
                    failed("transfer_interrupted"),
                    source=source("v1", fail_after_bytes=half),
                ),
                {
                    "area": "staging",
                    "generation": "v1",
                    "mode": "flip",
                    "offset": 20,
                    "op": "tamper",
                    "path": CONFIG,
                },
                acquire(
                    "v1",
                    acquired(
                        "v1",
                        transferred=v1_t
                        - (kept_half - file_size("v1", CONFIG) + CHUNK_BYTES),
                        resumed=True,
                        resume_reason="partial_digest_mismatch",
                    ),
                ),
            ],
        ),
        scenario(
            "range_not_honored_restarts_file",
            "A source that ignores the requested offset restarts only the partial file.",
            [
                acquire(
                    "v1",
                    failed("transfer_interrupted"),
                    source=source("v1", fail_after_bytes=partial),
                ),
                snapshot(
                    generations=(
                        entry("v1", "resumable", verified=model_offset + CHUNK_BYTES),
                    )
                ),
                acquire(
                    "v1",
                    acquired(
                        "v1",
                        transferred=v1_t - model_offset,
                        resumed=True,
                        resume_reason="range_not_honored",
                    ),
                    source=source("v1", ignore_range=True),
                ),
            ],
        ),
        scenario(
            "crash_mid_transfer_recovers_resumable",
            "A crash after a chunk commit leaves a stale lock and resumable staging.",
            [
                acquire("v1", CRASHED, crash_at=crash("chunk_committed", 5)),
                restart(),
                recover(("stale_lock_cleared", None), ("staging_resumable", "v1")),
                snapshot(
                    generations=(
                        entry("v1", "resumable", verified=kept_after_crash("v1", 5)),
                    )
                ),
                acquire(
                    "v1",
                    acquired(
                        "v1",
                        transferred=v1_t - kept_after_crash("v1", 5),
                        resumed=True,
                    ),
                ),
            ],
        ),
        scenario(
            "crash_before_first_byte_recovers_empty_staging",
            "A crash right after the staging record keeps an empty resumable entry.",
            [
                acquire("v1", CRASHED, crash_at=crash("staging_record_written")),
                restart(),
                recover(("stale_lock_cleared", None), ("staging_resumable", "v1")),
                snapshot(generations=(entry("v1", "resumable", verified=0),)),
                acquire("v1", acquired("v1")),
            ],
        ),
        scenario(
            "crash_while_verifying_reverifies",
            "A crash during verification never promotes; the next attempt re-verifies without transfer.",
            [
                acquire("v1", CRASHED, crash_at=crash("verifying_record_written")),
                restart(),
                recover(("stale_lock_cleared", None), ("staging_resumable", "v1")),
                acquire("v1", acquired("v1", transferred=0, resumed=True)),
            ],
        ),
        scenario(
            "crash_after_commit_keeps_previous_active",
            "A crash between commit and activation leaves the old generation active.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", CRASHED, crash_at=crash("generation_committed")),
                restart(),
                recover(("stale_lock_cleared", None)),
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(
                        entry("v1", "active", protections=AG),
                        entry("v2", "evictable"),
                    ),
                ),
                {"confirm": True, "expect": OK, "generation": "v2", "op": "activate"},
                snapshot(
                    active="v2",
                    last_green="v2",
                    rollback="v1",
                    generations=(
                        entry("v1", "superseded", protections=R),
                        entry("v2", "active", protections=AG),
                    ),
                ),
            ],
        ),
        scenario(
            "crash_after_pointer_switch_is_durable",
            "A crash after the pointer write keeps the new activation.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", CRASHED, crash_at=crash("pointer_committed")),
                restart(),
                recover(("stale_lock_cleared", None)),
                snapshot(
                    active="v2",
                    last_green="v2",
                    rollback="v1",
                    generations=(
                        entry("v1", "superseded", protections=R),
                        entry("v2", "active", protections=AG),
                    ),
                ),
            ],
        ),
        scenario(
            "digest_mismatch_quarantines_and_keeps_active",
            "Corrupted bytes fail verification, are quarantined, and leave the active generation intact.",
            [
                acquire("v1", acquired("v1")),
                acquire(
                    "v2",
                    failed("verification_failed", "file_digest_mismatch"),
                    source=source("v2", corrupt={"offset": 3, "path": MODEL}),
                ),
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(entry("v1", "active", protections=AG),),
                    quarantined=(("v2", "file_digest_mismatch"),),
                ),
                restart(),
                recover(("quarantine_purged", "v2")),
                acquire("v2", acquired("v2")),
            ],
        ),
        scenario(
            "oversized_file_quarantines",
            "A source that sends more bytes than declared is refused and quarantined.",
            [
                acquire("v1", acquired("v1")),
                acquire(
                    "v2",
                    failed("verification_failed", "file_size_mismatch"),
                    source=source("v2", extra_bytes={"count": 3, "path": MODEL}),
                ),
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(entry("v1", "active", protections=AG),),
                    quarantined=(("v2", "file_size_mismatch"),),
                ),
            ],
        ),
        scenario(
            "untrusted_signature_is_refused_before_staging",
            "A bundle signed by an untrusted key is refused before any byte is staged.",
            [
                acquire("v1", acquired("v1")),
                acquire(
                    "v2_untrusted",
                    failed("bundle_rejected", "untrusted_key"),
                ),
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(entry("v1", "active", protections=AG),),
                ),
            ],
        ),
        scenario(
            "unsupported_schema_is_refused_before_staging",
            "A manifest with an unsupported schema version is refused before staging.",
            [
                acquire("v1", acquired("v1")),
                acquire(
                    "v2_schema",
                    failed("bundle_rejected", "schema_version_unsupported"),
                ),
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(entry("v1", "active", protections=AG),),
                ),
            ],
        ),
        scenario(
            "cancellation_keeps_resumable_and_active",
            "Cancellation stops after the current chunk and keeps the previous active generation.",
            [
                acquire("v1", acquired("v1")),
                acquire(
                    "v2",
                    failed("cancelled"),
                    source=source("v2", cancel_after_bytes=40),
                ),
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(
                        entry("v1", "active", protections=AG),
                        entry("v2", "resumable", verified=kept_after_cancel("v2", 40)),
                    ),
                ),
                acquire(
                    "v2",
                    acquired(
                        "v2",
                        transferred=total("v2") - kept_after_cancel("v2", 40),
                        resumed=True,
                    ),
                ),
            ],
        ),
        scenario(
            "storage_pressure_is_refused_before_staging",
            "Insufficient free storage is refused before staging and keeps the active generation.",
            [
                acquire("v1", acquired("v1")),
                acquire(
                    "v2",
                    failed("storage_pressure"),
                    storage_available_bytes=total("v2") - 1,
                ),
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(entry("v1", "active", protections=AG),),
                ),
            ],
        ),
        scenario(
            "quota_never_evicts_protected_generations",
            "When only protected generations hold the quota, acquisition is refused.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", failed("quota_exceeded")),
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(entry("v1", "active", protections=AG),),
                ),
            ],
            quota_bytes=total("v1") + total("v2") - 1,
        ),
        scenario(
            "lru_eviction_is_deterministic",
            "Quota eviction removes the least recently used unprotected generation only.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", acquired("v2")),
                acquire("v3", acquired("v3")),
                snapshot(
                    active="v3",
                    last_green="v3",
                    rollback="v2",
                    generations=(
                        entry("v1", "evictable"),
                        entry("v2", "superseded", protections=R),
                        entry("v3", "active", protections=AG),
                    ),
                ),
                acquire("v4", acquired("v4", evicted=("v1",))),
                snapshot(
                    active="v4",
                    last_green="v4",
                    rollback="v3",
                    generations=(
                        entry("v2", "evictable"),
                        entry("v3", "superseded", protections=R),
                        entry("v4", "active", protections=AG),
                    ),
                ),
            ],
            quota_bytes=total("v2") + total("v3") + total("v4"),
        ),
        scenario(
            "pins_and_leases_block_eviction",
            "Pinned and leased generations are never evicted; releasing them makes them evictable.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", acquired("v2")),
                acquire("v3", acquired("v3")),
                {"expect": OK, "generation": "v1", "op": "pin"},
                {
                    "additional_bytes": 1,
                    "crash_at": None,
                    "expect": {
                        "evicted": [],
                        "satisfied": False,
                        "used_bytes": total("v1") + total("v2") + total("v3"),
                    },
                    "op": "evict",
                },
                {"expect": OK, "generation": "v1", "op": "unpin"},
                {
                    "expect": {"generation": "v1", "ok": True},
                    "generation": "v1",
                    "lease": "L1",
                    "op": "lease",
                },
                snapshot(
                    active="v3",
                    last_green="v3",
                    rollback="v2",
                    generations=(
                        entry("v1", "superseded", protections=("leased",)),
                        entry("v2", "superseded", protections=R),
                        entry("v3", "active", protections=AG),
                    ),
                ),
                {
                    "additional_bytes": 1,
                    "crash_at": None,
                    "expect": {
                        "evicted": [],
                        "satisfied": False,
                        "used_bytes": total("v1") + total("v2") + total("v3"),
                    },
                    "op": "evict",
                },
                {"expect": OK, "lease": "L1", "op": "release"},
                {
                    "additional_bytes": 1,
                    "crash_at": None,
                    "expect": {
                        "evicted": ["v1"],
                        "satisfied": True,
                        "used_bytes": total("v2") + total("v3"),
                    },
                    "op": "evict",
                },
            ],
            quota_bytes=total("v1") + total("v2") + total("v3"),
        ),
        scenario(
            "lease_survives_restart_until_ttl",
            "A lease from a dead process protects its generation until the lease TTL passes.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", acquired("v2")),
                acquire("v3", acquired("v3")),
                {
                    "expect": {"generation": "v1", "ok": True},
                    "generation": "v1",
                    "lease": "L1",
                    "op": "lease",
                },
                restart(),
                recover(),
                {
                    "additional_bytes": 1,
                    "crash_at": None,
                    "expect": {
                        "evicted": [],
                        "satisfied": False,
                        "used_bytes": total("v1") + total("v2") + total("v3"),
                    },
                    "op": "evict",
                },
                {"ms": LEASE_TTL_MS, "op": "advance"},
                recover(("lease_expired", "v1")),
                {
                    "additional_bytes": 1,
                    "crash_at": None,
                    "expect": {
                        "evicted": ["v1"],
                        "satisfied": True,
                        "used_bytes": total("v2") + total("v3"),
                    },
                    "op": "evict",
                },
            ],
            quota_bytes=total("v1") + total("v2") + total("v3"),
        ),
        scenario(
            "crash_during_eviction_reclaims_bytes",
            "A crash after an eviction commit leaves only trash that recovery purges.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", acquired("v2")),
                acquire("v3", acquired("v3")),
                {
                    "additional_bytes": 1,
                    "crash_at": crash("eviction_committed"),
                    "expect": CRASHED,
                    "op": "evict",
                },
                restart(),
                recover(("stale_lock_cleared", None), ("trash_purged", "v1")),
                snapshot(
                    active="v3",
                    last_green="v3",
                    rollback="v2",
                    generations=(
                        entry("v2", "superseded", protections=R),
                        entry("v3", "active", protections=AG),
                    ),
                ),
            ],
            quota_bytes=total("v1") + total("v2") + total("v3"),
        ),
        scenario(
            "rollback_restores_previous_generation",
            "Rollback reactivates the retained previous generation without network access.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", acquired("v2")),
                {"expect": OK, "op": "rollback", "quarantine": False},
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(
                        entry("v1", "active", protections=AG),
                        entry("v2", "evictable"),
                    ),
                ),
                {
                    "expect": failed("rollback_unavailable"),
                    "op": "rollback",
                    "quarantine": False,
                },
            ],
        ),
        scenario(
            "rollback_can_quarantine_the_abandoned_generation",
            "Rollback with quarantine removes the abandoned generation from use.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", acquired("v2")),
                {"expect": OK, "op": "rollback", "quarantine": True},
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(entry("v1", "active", protections=AG),),
                    quarantined=(("v2", "rolled_back"),),
                ),
            ],
        ),
        scenario(
            "probation_keeps_last_green_until_confirmed",
            "An unconfirmed activation keeps the previous generation as last green.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", acquired("v2"), confirm=False),
                snapshot(
                    active="v2",
                    last_green="v1",
                    rollback="v1",
                    generations=(
                        entry(
                            "v1", "superseded", protections=("last_green", "rollback")
                        ),
                        entry("v2", "active", protections=A),
                    ),
                ),
                {"expect": OK, "op": "confirm"},
                snapshot(
                    active="v2",
                    last_green="v2",
                    rollback="v1",
                    generations=(
                        entry("v1", "superseded", protections=R),
                        entry("v2", "active", protections=AG),
                    ),
                ),
            ],
        ),
        scenario(
            "prepared_without_activation_is_evictable",
            "A generation prepared without activation is ready but unprotected until pinned or activated.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", acquired("v2", state="evictable"), activate=False),
                {"expect": OK, "generation": "v2", "op": "pin"},
                snapshot(
                    active="v1",
                    last_green="v1",
                    generations=(
                        entry("v1", "active", protections=AG),
                        entry("v2", "ready", protections=("pinned",)),
                    ),
                ),
                {
                    "confirm": True,
                    "expect": failed("generation_not_ready"),
                    "generation": "v3",
                    "op": "activate",
                },
            ],
        ),
        scenario(
            "corrupt_active_detected_on_recover",
            "Recovery quarantines an active generation whose files changed size and falls back.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", acquired("v2")),
                {
                    "area": "ready",
                    "generation": "v2",
                    "mode": "truncate",
                    "offset": 4,
                    "op": "tamper",
                    "path": MODEL,
                },
                restart(),
                recover(("generation_quarantined", "v2"), ("pointer_repaired", None)),
                snapshot(
                    active="v1",
                    generations=(entry("v1", "active", protections=A),),
                    quarantined=(("v2", "payload_mismatch"),),
                ),
            ],
        ),
        scenario(
            "corrupt_active_detected_on_full_open",
            "A full open re-hashes every file, quarantines a corrupt active generation, and falls back.",
            [
                acquire("v1", acquired("v1")),
                acquire("v2", acquired("v2")),
                {
                    "area": "ready",
                    "generation": "v2",
                    "mode": "flip",
                    "offset": 4,
                    "op": "tamper",
                    "path": MODEL,
                },
                {
                    "expect": {"files_checked": FILES, "generation": "v2", "ok": True},
                    "lease": "M1",
                    "op": "open",
                    "verification": "metadata",
                },
                {"expect": OK, "lease": "M1", "op": "release"},
                {
                    "expect": failed("generation_corrupt", "file_digest_mismatch"),
                    "lease": "F1",
                    "op": "open",
                    "verification": "full",
                },
                {
                    "expect": {"files_checked": FILES, "generation": "v1", "ok": True},
                    "lease": "F2",
                    "op": "open",
                    "verification": "full",
                },
                snapshot(
                    active="v1",
                    generations=(
                        entry("v1", "active", protections=("active", "leased")),
                    ),
                    quarantined=(("v2", "file_digest_mismatch"),),
                ),
            ],
        ),
        scenario(
            "staging_expires_after_bounded_age",
            "Orphan staging older than the bounded age is removed on recovery.",
            [
                acquire("v1", CRASHED, crash_at=crash("chunk_committed", 3)),
                restart(advance_ms=STAGING_MAX_AGE_MS),
                recover(("stale_lock_cleared", None), ("staging_expired", "v1")),
                snapshot(),
            ],
        ),
        scenario(
            "new_manifest_supersedes_partial_staging",
            "Starting a different generation discards partial bytes of the old one.",
            [
                acquire(
                    "v1",
                    failed("transfer_interrupted"),
                    source=source("v1", fail_after_bytes=half),
                ),
                acquire("v2", acquired("v2")),
                snapshot(
                    active="v2",
                    last_green="v2",
                    generations=(entry("v2", "active", protections=AG),),
                ),
            ],
        ),
        scenario(
            "offline_open_of_prepared_bundle",
            "A prepared bundle opens and activates after restart with no source at all.",
            [
                acquire("v1", acquired("v1", state="evictable"), activate=False),
                restart(),
                recover(),
                {
                    "expect": failed("no_active_generation"),
                    "lease": "X0",
                    "op": "open",
                    "verification": "full",
                },
                {"confirm": True, "expect": OK, "generation": "v1", "op": "activate"},
                {
                    "expect": {"files_checked": FILES, "generation": "v1", "ok": True},
                    "lease": "X1",
                    "op": "open",
                    "verification": "full",
                },
            ],
        ),
    ]
    for item in scenarios:
        for step in item["steps"]:
            expect = step.get("expect", {})
            if step["op"] == "acquire" and expect.get("ok") is False:
                if expect["reason"] not in REASONS:
                    raise SystemExit(f"{item['id']}: unknown reason {expect['reason']}")
    return scenarios


def build_vectors() -> dict:
    """Return the complete vector document."""

    policies = PB.trust_policies()
    chosen = {name: policies[name] for name in ("default", "rotation")}
    bundles = copy.deepcopy(BUNDLES)
    return {
        "bundles": bundles,
        "checkpoints": list(CHECKPOINTS),
        "clock": CLOCK,
        "context": {"now": PB.NOW, "target": None},
        "identities": {
            "bundle_id": BUNDLES["v1"]["bundle_id"],
            "bundle_key": bundle_key(BUNDLES["v1"]["bundle_id"]),
            "source_digests": {
                name: sha256_text(name.encode("utf-8"))
                for name in ("source-a", "source-b")
            },
            "trust_digests": {
                name: trust_digest(value) for name, value in chosen.items()
            },
        },
        "notice": (
            "TEST-ONLY vectors. Bundles are signed with the TEST-ONLY keys of "
            "tests/fixtures/portable_bundle; never trust them in production."
        ),
        "quarantine_reasons": list(QUARANTINE_REASONS),
        "reason_codes": list(REASONS),
        "recovery_actions": list(RECOVERY_ACTIONS),
        "resume_reasons": list(RESUME_REASONS),
        "scenarios": build_scenarios(),
        "schema": "openmed.bundle_cache.vectors",
        "states": list(STATES),
        "synthetic_only": True,
        "test_only_keys": True,
        "trust_policies": chosen,
        "version": 1,
    }


def render() -> str:
    """Return the serialized vector document."""

    return json.dumps(build_vectors(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write or check ``vectors.json``."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        if (
            not VECTORS_PATH.exists()
            or VECTORS_PATH.read_text(encoding="utf-8") != text
        ):
            print("bundle cache vectors are stale")
            return 1
        return 0
    VECTORS_PATH.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
