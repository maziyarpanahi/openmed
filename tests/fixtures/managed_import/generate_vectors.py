"""Regenerate the cross-kit managed bundle import vectors.

The vectors drive the managed importers in OpenMedKit for Android
(``ManagedBundleImporter``) and OpenMedKit for Swift
(``OpenMedManagedBundleImporter``). Each scenario is a script of imports through
fake managed-configuration and local-file handoffs, with the exact value-free
receipt both kits must produce. The contract lives in
``docs/mobile/managed-model-bundles.md``.

TEST-ONLY KEYS. The bundles are signed with the TEST-ONLY keys of
``tests/fixtures/portable_bundle/generate_vectors.py``. Never trust them in
production.

Archives are written by a small deterministic ZIP writer. Deflated entries use
fixed Huffman codes with run-length back references, so the bytes never depend
on the local zlib build.

Usage::

    python tests/fixtures/managed_import/generate_vectors.py          # write
    python tests/fixtures/managed_import/generate_vectors.py --check  # verify

All bundle files are tiny synthetic text placeholders. The vectors hold
outcomes, states, counts, digests, and reason codes only.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.core import portable_bundle as pb  # noqa: E402

VECTORS_PATH = Path(__file__).with_name("vectors.json")
FIXTURES = ROOT / "tests" / "fixtures"

CHUNK_BYTES = 16
"""Cache resume checkpoint size used by every scenario (production: 4 MiB)."""

CLOCK = {"start_ms": 1_000_000, "step_ms": 1_000}

CONFIG_VERSION = 1

OUTCOMES = (
    "imported",
    "already_present",
    "activated",
    "deferred",
    "rolled_back",
    "unsupported",
    "rejected",
)

REASONS = (
    "managed_config_absent",
    "handoff_pending",
    "config_version_unsupported",
    "handoff_unsupported",
    "archive_format_unsupported",
    "config_invalid",
    "trust_policy_missing",
    "trust_policy_invalid",
    "trust_anchor_unpinned",
    "location_unapproved",
    "handoff_unreadable",
    "budget_archive_bytes",
    "budget_file_count",
    "budget_total_bytes",
    "budget_compression_ratio",
    "archive_malformed",
    "entry_unsafe",
    "entry_duplicate",
    "bundle_rejected",
    "bundle_not_expected",
    "verification_failed",
    "storage_pressure",
    "quota_exceeded",
    "transfer_interrupted",
    "cancelled",
    "lock_timeout",
    "storage_io",
    "activation_check_failed",
)
"""Managed-import reason codes (``reason`` in a receipt)."""

OUTCOME_OF_REASON = {
    "managed_config_absent": "deferred",
    "handoff_pending": "deferred",
    "config_version_unsupported": "unsupported",
    "handoff_unsupported": "unsupported",
    "archive_format_unsupported": "unsupported",
}
"""Reasons whose outcome is not ``rejected`` (``rolled_back`` aside)."""

CHECKPOINTS = (
    "archive_spooled",
    "staging_record_written",
    "chunk_committed",
    "verifying_record_written",
    "generation_committed",
    "pointer_committed",
)
"""Crash and cancellation points: the importer's own, then the cache's."""

DEFAULT_LIMITS = {
    "max_archive_bytes": 4_294_967_295,
    "max_compression_ratio": 200,
    "max_entries": 4_096,
    "max_total_bytes": 4_294_967_295,
}

KEYS = {
    "activation": "openmed_bundle_activation",
    "config_version": "openmed_bundle_config_version",
    "expected_content_id": "openmed_bundle_expected_content_id",
    "expected_id": "openmed_bundle_expected_id",
    "key_pins": "openmed_bundle_key_pins",
    "location": "openmed_bundle_location",
    "min_sequence": "openmed_bundle_min_sequence",
    "trust_policy": "openmed_bundle_trust_policy",
}
"""Managed configuration keys shared by Apple and Android."""


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PB = _load(
    "portable_bundle_vector_generator",
    FIXTURES / "portable_bundle" / "generate_vectors.py",
)
BC = _load(
    "bundle_cache_vector_generator",
    FIXTURES / "bundle_cache" / "generate_vectors.py",
)


# ---------------------------------------------------------------------------
# Synthetic signed bundles: one ONNX runtime for Android and iOS, four files.

ROOT_A = "openmed-test-root-a"
ROOT_B = "openmed-test-root-b"
ROGUE = "openmed-test-rogue"
MODEL = "onnx/model.onnx"
BASE_TREE = {
    path: PB.GOLDEN_TREE[path]
    for path in ("LICENSE", "config.json", MODEL, "vocab.txt")
}
CHANGES = {
    "v1": ("1.2.0", 7, {}),
    "v2": (
        "1.3.0",
        8,
        {MODEL: "synthetic fp32 onnx graph placeholder, generation two\n"},
    ),
    "v3": ("1.4.0", 9, {MODEL: "synthetic fp32 onnx graph placeholder, three\n"}),
}


def manifest_value(version: str, sequence: int, tree: dict[str, str]) -> dict:
    """Return a small manifest value with every optional section empty."""

    value = PB.golden_manifest()
    value["bundle"]["version"] = version
    value["bundle"]["sequence"] = sequence
    value["calibration"] = None
    value["evidence"] = []
    value["extensions"] = {}
    value["labels"] = None
    value["policy"]["path"] = None
    value["provenance"]["export_record"] = None
    value["segmenter"] = None
    value["compatibility"]["sdks"] = [
        item
        for item in value["compatibility"]["sdks"]
        if item["name"] in ("openmedkit-android", "openmedkit-swift")
    ]
    value["tokenizer"]["files"] = ["vocab.txt"]
    value["runtimes"] = [
        {
            "entrypoint": MODEL,
            "fallback": None,
            "files": ["config.json", MODEL],
            "format": "onnx",
            "id": "onnx-fp32",
            "min_memory_bytes": 268435456,
            "min_storage_bytes": 2097152,
            "platforms": [
                {"max_os_exclusive": None, "min_os": "26", "platform": "android"},
                {"max_os_exclusive": None, "min_os": "17.0", "platform": "ios"},
            ],
            "precision": "fp32",
        }
    ]
    value["files"] = [
        PB.file_entry(path, tree[path])
        for path in sorted(tree, key=lambda item: item.encode("utf-8"))
    ]
    return value


def bundle_entry(alias: str, kids: list[str]) -> dict:
    """Return one signed synthetic bundle."""

    version, sequence, changes = CHANGES[alias]
    tree = dict(BASE_TREE)
    tree.update(changes)
    manifest = PB.canonical_text(manifest_value(version, sequence, tree))
    content = pb.content_id(manifest.encode("utf-8"))
    files = [
        {"path": path, "text": tree[path]}
        for path in sorted(tree, key=lambda item: item.encode("utf-8"))
    ]
    return {
        "bundle_id": PB.golden_manifest()["bundle"]["id"],
        "content_id": content,
        "envelope": PB.sign(manifest, kids),
        "files": files,
        "generation": BC.generation_id(content),
        "manifest": manifest,
        "sequence": sequence,
        "total_bytes": sum(len(item["text"].encode("utf-8")) for item in files),
    }


def build_bundles() -> dict[str, dict]:
    """Return the synthetic bundles keyed by alias."""

    return {
        "v1": bundle_entry("v1", [ROOT_A]),
        "v2": bundle_entry("v2", [ROOT_A]),
        "v2_ab": bundle_entry("v2", [ROOT_A, ROOT_B]),
        "v2_rogue": bundle_entry("v2", [ROGUE]),
        "v3": bundle_entry("v3", [ROOT_A]),
    }


BUNDLES = build_bundles()
GENERATION_ALIAS = {"v1": "v1", "v2": "v2", "v2_ab": "v2", "v2_rogue": "v2", "v3": "v3"}
"""Bundles that share a manifest share a generation."""


def tree(alias: str) -> list[tuple[str, bytes]]:
    """Return the files of a bundle with its manifest and envelope first."""

    bundle = BUNDLES[alias]
    files = [
        (pb.MANIFEST_FILENAME, bundle["manifest"].encode("utf-8")),
        (pb.ENVELOPE_FILENAME, bundle["envelope"].encode("utf-8")),
    ]
    files += [(item["path"], item["text"].encode("utf-8")) for item in bundle["files"]]
    return files


# ---------------------------------------------------------------------------
# Deterministic raw DEFLATE (RFC 1951) with fixed Huffman codes.

_LENGTH_BASES = (
    (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 1),
    (13, 1), (15, 1), (17, 1), (19, 2), (23, 2), (27, 2), (31, 2), (35, 3),
    (43, 3), (51, 3), (59, 3), (67, 4), (83, 4), (99, 4), (115, 4), (131, 5),
    (163, 5), (195, 5), (227, 5), (258, 0),
)  # fmt: skip


class _Bits:
    def __init__(self) -> None:
        self.value = 0
        self.count = 0
        self.out = bytearray()

    def put(self, value: int, width: int) -> None:
        self.value |= value << self.count
        self.count += width
        while self.count >= 8:
            self.out.append(self.value & 0xFF)
            self.value >>= 8
            self.count -= 8

    def code(self, code: int, width: int) -> None:
        reversed_code = int(format(code, f"0{width}b")[::-1], 2)
        self.put(reversed_code, width)

    def finish(self) -> bytes:
        if self.count:
            self.out.append(self.value & 0xFF)
        return bytes(self.out)


def _symbol(bits: _Bits, symbol: int) -> None:
    if symbol < 144:
        bits.code(0x30 + symbol, 8)
    elif symbol < 256:
        bits.code(0x190 + symbol - 144, 9)
    elif symbol < 280:
        bits.code(symbol - 256, 7)
    else:
        bits.code(0xC0 + symbol - 280, 8)


def _match(bits: _Bits, length: int) -> None:
    """Write one ``<length, distance 1>`` back reference (3 <= length <= 258)."""

    if length == 258:
        index = len(_LENGTH_BASES) - 1
    else:
        index = max(
            position
            for position, (base, _) in enumerate(_LENGTH_BASES[:-1])
            if base <= length
        )
    base, extra = _LENGTH_BASES[index]
    _symbol(bits, 257 + index)
    if extra:
        bits.put(length - base, extra)
    bits.code(0, 5)  # distance code 0: distance 1, no extra bits


def deflate(data: bytes) -> bytes:
    """Return a raw DEFLATE stream of ``data`` (one fixed-Huffman block)."""

    bits = _Bits()
    bits.put(1, 1)  # BFINAL
    bits.put(1, 2)  # BTYPE = 01 (fixed Huffman)
    index = 0
    while index < len(data):
        run = 1
        while index + run < len(data) and data[index + run] == data[index]:
            run += 1
        _symbol(bits, data[index])
        remaining = run - 1
        while remaining >= 3:
            length = min(remaining, 258)
            _match(bits, length)
            remaining -= length
        for _ in range(remaining):
            _symbol(bits, data[index])
        index += run
    _symbol(bits, 256)
    encoded = bits.finish()
    if zlib.decompress(encoded, -15) != data:
        raise SystemExit("deterministic deflate failed its round trip")
    return encoded


# ---------------------------------------------------------------------------
# Deterministic ZIP writer with hooks for hostile archives.

DOS_DATE = ((2026 - 1980) << 9) | (1 << 5) | 1
FILE_MODE = 0o100644
DIRECTORY_MODE = 0o040755
SYMLINK_MODE = 0o120777


def member(name: str, data: bytes = b"", **options: Any) -> dict:
    """Return one archive member description."""

    value = {
        "cd_csize": None,
        "cd_name": None,
        "data": data,
        "flags": 0,
        "local_name": None,
        "method": 0,
        "mode": DIRECTORY_MODE if name.endswith("/") else FILE_MODE,
        "name": name,
        "payload": None,
    }
    value.update(options)
    return value


def write_zip(members: list[dict], *, cd_offset_delta: int = 0) -> bytes:
    """Return ZIP bytes for ``members`` in order."""

    body = bytearray()
    central = bytearray()
    for item in members:
        data = item["data"]
        payload = deflate(data) if item["method"] == 8 else data
        if item["payload"] is not None:
            payload = item["payload"]
        crc = zlib.crc32(data) & 0xFFFFFFFF
        local_name = (item["local_name"] or item["name"]).encode("utf-8")
        cd_name = (item["cd_name"] or item["name"]).encode("utf-8")
        offset = len(body)
        body += struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,
            20,
            item["flags"],
            item["method"],
            0,
            DOS_DATE,
            crc,
            len(payload),
            len(data),
            len(local_name),
            0,
        )
        body += local_name + payload
        external = item["mode"] << 16
        if item["mode"] & 0o170000 == 0o040000:
            external |= 0x10
        csize = item["cd_csize"] if item["cd_csize"] is not None else len(payload)
        central += struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,
            (3 << 8) | 20,
            20,
            item["flags"],
            item["method"],
            0,
            DOS_DATE,
            crc,
            csize,
            len(data),
            len(cd_name),
            0,
            0,
            0,
            0,
            external,
            offset,
        )
        central += cd_name
    eocd = struct.pack(
        "<IHHHHIIH",
        0x06054B50,
        0,
        0,
        len(members),
        len(members),
        len(central),
        len(body) + cd_offset_delta,
        0,
    )
    return bytes(body + central + eocd)


def bundle_members(alias: str, *, method: int = 0, prefix: str = "") -> list[dict]:
    """Return the members of a bundle archive."""

    members = []
    if prefix:
        directories = {prefix}
        for path, _ in tree(alias):
            parts = path.split("/")[:-1]
            for depth in range(1, len(parts) + 1):
                directories.add(prefix + "/".join(parts[:depth]) + "/")
        members += [member(name) for name in sorted(directories)]
    members += [
        member(prefix + path, data, method=method) for path, data in tree(alias)
    ]
    return members


def _replace(members: list[dict], name: str, **options: Any) -> list[dict]:
    result = []
    for item in members:
        if item["name"] == name:
            item = dict(item)
            item.update(options)
        result.append(item)
    return result


def _without(members: list[dict], name: str) -> list[dict]:
    return [item for item in members if item["name"] != name]


def _tamper(text: bytes) -> bytes:
    return text[:-2] + bytes([text[-2] ^ 0x01]) + text[-1:]


VOCAB = "vocab.txt"
CONFIG = "config.json"


def build_archives() -> dict[str, dict]:
    """Return every archive keyed by alias."""

    v2 = bundle_members("v2")
    vocab = dict(tree("v2"))[VOCAB]
    specs: dict[str, tuple[str | None, str, bytes]] = {
        "v1": ("v1", "stored, root level", write_zip(bundle_members("v1"))),
        "v2": (
            "v2",
            "deflated, nested under one top-level directory with directory entries",
            write_zip(bundle_members("v2", method=8, prefix="openmed-bundle-v2/")),
        ),
        "v2_ab": (
            "v2_ab",
            "stored; signed by root-a and root-b",
            write_zip(bundle_members("v2_ab")),
        ),
        "v2_rogue": (
            "v2_rogue",
            "stored; signed by an untrusted key",
            write_zip(bundle_members("v2_rogue")),
        ),
        "v3": ("v3", "stored, root level", write_zip(bundle_members("v3"))),
        "traversal": (
            "v2",
            "adds a ../ entry",
            write_zip(v2 + [member("../escape.txt", b"synthetic\n")]),
        ),
        "duplicate": (
            "v2",
            "repeats config.json",
            write_zip(v2 + [member(CONFIG, b"{}\n")]),
        ),
        "case_duplicate": (
            "v2",
            "adds Config.json",
            write_zip(v2 + [member("Config.json", b"{}\n")]),
        ),
        "symlink": (
            "v2",
            "vocab.txt is a symbolic link",
            write_zip(_replace(v2, VOCAB, mode=SYMLINK_MODE)),
        ),
        "encrypted": (
            "v2",
            "vocab.txt has the encryption flag",
            write_zip(_replace(v2, VOCAB, flags=1)),
        ),
        "zip64": (
            "v2",
            "vocab.txt has a ZIP64 size marker",
            write_zip(_replace(v2, VOCAB, cd_csize=0xFFFFFFFF)),
        ),
        "unsupported_method": (
            "v2",
            "vocab.txt uses bzip2",
            write_zip(_replace(v2, VOCAB, method=12)),
        ),
        "not_zip": (
            None,
            "plain text, no end of central directory",
            b"synthetic text, not an archive\n" * 2,
        ),
        "truncated": (
            "v2",
            "central directory offset is off by one",
            write_zip(v2, cd_offset_delta=1),
        ),
        "local_mismatch": (
            "v2",
            "local header name differs",
            write_zip(_replace(v2, VOCAB, local_name="vocab.txx")),
        ),
        "undeclared": (
            "v2",
            "adds an undeclared file",
            write_zip(v2 + [member("extra/notes.txt", b"synthetic\n")]),
        ),
        "missing": ("v2", "omits vocab.txt", write_zip(_without(v2, VOCAB))),
        "size_mismatch": (
            "v2",
            "vocab.txt has one extra byte",
            write_zip(_replace(v2, VOCAB, data=vocab + b"x")),
        ),
        "tampered": (
            "v2",
            "vocab.txt has one flipped bit",
            write_zip(_replace(v2, VOCAB, data=_tamper(vocab))),
        ),
        "corrupt_deflate": (
            "v2",
            "config.json carries a reserved DEFLATE block type",
            write_zip(
                _replace(
                    bundle_members("v2", method=8), CONFIG, payload=b"\xff\xff\xff\xff"
                )
            ),
        ),
        "no_manifest": (
            "v2",
            "omits the manifest",
            write_zip(_without(v2, pb.MANIFEST_FILENAME)),
        ),
        "bomb": (
            "v2",
            "adds 256 KiB of zeros deflated about 158:1",
            write_zip(v2 + [member("padding.bin", bytes(1 << 18), method=8)]),
        ),
    }
    archives = {}
    for alias, (bundle, description, data) in specs.items():
        archives[alias] = {
            "base64": base64.b64encode(data).decode("ascii"),
            "bundle": bundle,
            "description": description,
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    return archives


ARCHIVES = build_archives()


# ---------------------------------------------------------------------------
# Trust policies, pins, managed configurations, and targets.


def fingerprint(kid: str) -> str:
    """Return the pin of a TEST-ONLY key: SHA-256 of the raw public key."""

    return "sha256:" + hashlib.sha256(pb.ed25519_public_key(PB.seed(kid))).hexdigest()


def trust_policies() -> dict[str, dict]:
    """Return the trust policies the managed configurations deliver."""

    base = PB.trust_policies()
    policies = {name: base[name] for name in ("default", "rotation", "key_revoked")}
    window = ("2026-01-01T00:00:00Z", "2027-07-01T00:00:00Z")
    policies["revoked_v2"] = PB.policy(
        [PB.jwk(ROOT_A, *window)], revoked=[BUNDLES["v2"]["content_id"]]
    )
    return policies


POLICIES = trust_policies()


def config(policy: str | None = "default", **values: Any) -> dict:
    """Return a managed configuration dictionary."""

    value: dict[str, Any] = {KEYS["config_version"]: CONFIG_VERSION}
    if policy is not None:
        value[KEYS["trust_policy"]] = PB.canonical_text(POLICIES[policy])
    for name, item in values.items():
        value[KEYS[name]] = item
    return value


def build_configs() -> dict[str, dict]:
    """Return managed configurations keyed by alias."""

    return {
        "default": config(),
        "stage": config(activation="stage"),
        "probation": config(activation="probation"),
        "rotation": config("rotation"),
        "key_revoked": config("key_revoked"),
        "revoked_v2": config("revoked_v2"),
        "pinned": config(key_pins=fingerprint(ROOT_A) + " " + fingerprint(ROOT_B)),
        "pinned_other": config(key_pins=fingerprint(ROOT_B)),
        "expect_v1": config(
            expected_id=BUNDLES["v1"]["bundle_id"],
            expected_content_id=BUNDLES["v1"]["content_id"],
        ),
        "min_sequence_9": config(min_sequence=9),
        "version_2": {**config(), KEYS["config_version"]: 2},
        "no_policy": config(None),
        "bad_policy": {**config(), KEYS["trust_policy"]: "not a trust policy"},
        "unknown_key": {**config(), "openmed_bundle_activaton": "stage"},
        "bad_pin": config(key_pins="sha256:not-hex"),
        "foreign_keys_only": {**config(), "com.example.unrelated": "kept by the app"},
    }


CONFIGS = build_configs()

TARGETS = {
    "android": {
        "default": PB.ANDROID,
        "no_runtime": {**PB.ANDROID, "runtimes": ["ort"]},
    },
    "ios": {"default": PB.IOS, "no_runtime": {**PB.IOS, "runtimes": ["mlx"]}},
}
"""Each kit verifies against its own platform; both must agree."""


def _context(target: dict, now: str, floors: dict) -> pb.VerificationContext:
    return pb.VerificationContext(
        now=now,
        target=pb.VerificationTarget(
            platform=target["platform"],
            sdk_name=target["sdk_name"],
            sdk_version=target["sdk_version"],
            runtimes=tuple(target["runtimes"]),
            os_version=target["os_version"],
            policy_version=target["policy_version"],
            memory_bytes=target["memory_bytes"],
        ),
        sequence_floors=floors,
    )


def verifier_reason(
    bundle: str, policy: str, variant: str, now: str, floor: int = 0
) -> str | None:
    """Return the Python verifier reason, which both platform targets must share."""

    reasons = set()
    for platform in TARGETS:
        result = pb.verify_portable_bundle(
            BUNDLES[bundle]["manifest"].encode("utf-8"),
            BUNDLES[bundle]["envelope"].encode("utf-8"),
            trust_policy=pb.TrustPolicy.from_mapping(POLICIES[policy]),
            context=_context(
                TARGETS[platform][variant], now, {BUNDLES[bundle]["bundle_id"]: floor}
            ),
        )
        reasons.add(None if result.ok else result.reason)
    if len(reasons) != 1:
        raise SystemExit(f"platform targets disagree for {bundle}: {reasons}")
    return reasons.pop()


# ---------------------------------------------------------------------------
# Scenarios.

NOW = PB.NOW
FILES = len(BUNDLES["v1"]["files"])


def imp(
    expect: dict,
    archive: str | None = None,
    *,
    handoff: str = "file",
    bundle: str | None = None,
    configuration: str | None = "default",
    now: str = NOW,
    health: str | None = None,
    cancel_at: dict | None = None,
    crash_at: dict | None = None,
    storage_available_bytes: int | None = None,
) -> dict:
    """Return an import step.

    ``handoff`` is ``file`` (an archive file), ``stream`` (archive bytes from a
    stream), or ``directory`` (the extracted files of ``bundle``).
    """

    if handoff == "directory":
        assert bundle is not None and archive is None
    else:
        assert archive is not None and bundle is None
    return {
        "archive": archive,
        "bundle": bundle,
        "cancel_at": cancel_at,
        "config": configuration,
        "crash_at": crash_at,
        "expect": expect,
        "handoff": handoff,
        "health": health,
        "now": now,
        "op": "import",
        "storage_available_bytes": storage_available_bytes,
    }


def receipt(
    outcome: str,
    *,
    reason: str | None = None,
    bundle_reason: str | None = None,
    generation: str | None = None,
    previous: str | None = None,
    active: str | None = None,
    last_green: str | None = None,
    resumed: bool = False,
    files_verified: int = 0,
) -> dict:
    """Return the expected receipt fields."""

    return {
        "active": active,
        "bundle_reason": bundle_reason,
        "files_verified": files_verified,
        "generation": generation,
        "last_green": last_green,
        "outcome": outcome,
        "previous_active": previous,
        "reason": reason,
        "resumed": resumed,
    }


def activated(
    gen: str,
    *,
    previous: str | None = None,
    resumed: bool = False,
    confirmed: bool = True,
) -> dict:
    """Return the receipt of an activation."""

    return receipt(
        "activated",
        generation=gen,
        previous=previous,
        active=gen,
        last_green=gen if confirmed else previous,
        resumed=resumed,
        files_verified=FILES,
    )


def present(gen: str, *, last_green: str | None = None) -> dict:
    """Return the receipt of a duplicate delivery."""

    return receipt(
        "already_present",
        generation=gen,
        previous=gen,
        active=gen,
        last_green=last_green or gen,
        files_verified=FILES,
    )


def refused(
    reason: str,
    *,
    bundle_reason: str | None = None,
    known: str | None = None,
    active: str | None = None,
) -> dict:
    """Return the receipt of a refusal.

    ``known`` names the generation when the manifest authenticated; identities
    and pointers are reported only then.
    """

    outcome = OUTCOME_OF_REASON.get(reason, "rejected")
    if known is None:
        return receipt(outcome, reason=reason, bundle_reason=bundle_reason)
    return receipt(
        outcome,
        reason=reason,
        bundle_reason=bundle_reason,
        generation=known,
        previous=active,
        active=active,
        last_green=active,
    )


def at(checkpoint: str, occurrence: int = 1) -> dict:
    """Return a crash or cancellation point."""

    if checkpoint not in CHECKPOINTS:
        raise SystemExit(f"unknown checkpoint {checkpoint}")
    return {"checkpoint": checkpoint, "occurrence": occurrence}


def state(
    *,
    active: str | None = None,
    last_green: str | None = None,
    rollback: str | None = None,
    quarantined: tuple[tuple[str, str], ...] = (),
    floor: int | None = None,
    receipts: int | None = None,
) -> dict:
    """Return a state step: cache pointer, quarantine, ledger floor, receipts."""

    return {
        "expect": {
            "active": active,
            "last_green": last_green,
            "quarantined": [
                {"generation": alias, "reason": reason}
                for alias, reason in sorted(
                    quarantined, key=lambda item: BUNDLES[item[0]]["generation"]
                )
            ],
            "receipts": receipts,
            "rollback": rollback,
            "sequence_floor": floor,
        },
        "op": "state",
    }


def restart() -> dict:
    """Return a restart step (new cache and importer on the same storage)."""

    return {"op": "restart"}


def recover(spool_purged: int) -> dict:
    """Return a recover step."""

    return {"expect": {"spool_purged": spool_purged}, "op": "recover"}


def confirm() -> dict:
    """Return a confirm step for the active probation generation."""

    return {"op": "confirm"}


def scenario(sid: str, description: str, steps: list[dict], **options: Any) -> dict:
    """Return one scenario."""

    limits = dict(DEFAULT_LIMITS)
    limits.update(options.pop("limits", {}))
    value = {
        "description": description,
        "id": sid,
        "limits": limits,
        "steps": steps,
        "target": options.pop("target", "default"),
    }
    if options:
        raise SystemExit(f"unknown scenario options {sorted(options)}")
    return value


def build_scenarios() -> list[dict]:
    """Return every scenario in a stable order."""

    rotation_now = "2026-12-15T00:00:00Z"
    expired_now = "2027-06-01T00:00:00Z"
    v1_active = [imp(activated("v1"), "v1")]
    scenarios = [
        scenario(
            "import_activates_and_is_idempotent",
            "A stored archive activates; the same delivery again changes nothing.",
            [
                state(),
                imp(activated("v1"), "v1"),
                imp(present("v1"), "v1"),
                imp(present("v1"), handoff="directory", bundle="v1"),
                imp(present("v1"), "v1", handoff="stream"),
                state(active="v1", last_green="v1", floor=7, receipts=4),
            ],
        ),
        scenario(
            "deflated_nested_archive_upgrades",
            "A deflated archive under one top-level directory supersedes v1.",
            [
                *v1_active,
                imp(activated("v2", previous="v1"), "v2"),
                state(active="v2", last_green="v2", rollback="v1", floor=8, receipts=2),
            ],
        ),
        scenario(
            "stream_and_directory_handoffs",
            "Stream and directory handoffs import the same generations.",
            [
                imp(activated("v1"), "v1", handoff="stream"),
                imp(activated("v2", previous="v1"), handoff="directory", bundle="v2"),
                state(active="v2", last_green="v2", rollback="v1", floor=8),
            ],
        ),
        scenario(
            "stage_then_activate",
            "A stage-only policy commits without activating; a later delivery activates.",
            [
                *v1_active,
                imp(
                    receipt(
                        "imported",
                        generation="v2",
                        previous="v1",
                        active="v1",
                        last_green="v1",
                        files_verified=FILES,
                    ),
                    "v2",
                    configuration="stage",
                ),
                state(active="v1", last_green="v1", floor=7),
                imp(
                    receipt(
                        "already_present",
                        generation="v2",
                        previous="v1",
                        active="v1",
                        last_green="v1",
                        files_verified=FILES,
                    ),
                    "v2",
                    configuration="stage",
                ),
                imp(activated("v2", previous="v1"), "v2"),
                state(active="v2", last_green="v2", rollback="v1", floor=8),
            ],
        ),
        scenario(
            "probation_until_confirmed",
            "A probation activation keeps v1 as last green until the host confirms.",
            [
                *v1_active,
                imp(
                    activated("v2", previous="v1", confirmed=False),
                    "v2",
                    configuration="probation",
                ),
                state(active="v2", last_green="v1", rollback="v1", floor=7),
                imp(present("v2", last_green="v1"), "v2", configuration="probation"),
                confirm(),
                state(active="v2", last_green="v2", rollback="v1", floor=8),
            ],
        ),
        scenario(
            "failed_health_check_rolls_back",
            "A failed activation check restores last green and quarantines the new generation.",
            [
                *v1_active,
                imp(
                    receipt(
                        "rolled_back",
                        reason="activation_check_failed",
                        generation="v2",
                        previous="v1",
                        active="v1",
                        last_green="v1",
                        files_verified=FILES,
                    ),
                    "v2",
                    health="fail",
                ),
                state(
                    active="v1",
                    last_green="v1",
                    quarantined=(("v2", "rolled_back"),),
                    floor=7,
                    receipts=2,
                ),
                imp(activated("v2", previous="v1"), "v2", health="pass"),
                state(active="v2", last_green="v2", rollback="v1", floor=8),
            ],
        ),
        scenario(
            "failed_first_health_check_leaves_nothing_active",
            "Without a prior generation a failed check is a rejection.",
            [
                imp(
                    receipt(
                        "rejected",
                        reason="activation_check_failed",
                        generation="v1",
                        files_verified=FILES,
                    ),
                    "v1",
                    health="fail",
                ),
                state(quarantined=(("v1", "rolled_back"),)),
            ],
        ),
        scenario(
            "signature_and_revocation_refusals",
            "Untrusted, revoked, and expired bundles are refused before staging.",
            [
                *v1_active,
                imp(
                    refused(
                        "bundle_rejected",
                        bundle_reason=verifier_reason(
                            "v2_rogue", "default", "default", NOW
                        ),
                    ),
                    "v2_rogue",
                ),
                imp(
                    refused(
                        "bundle_rejected",
                        bundle_reason=verifier_reason(
                            "v2", "key_revoked", "default", NOW
                        ),
                    ),
                    "v2",
                    configuration="key_revoked",
                ),
                imp(
                    refused(
                        "bundle_rejected",
                        bundle_reason=verifier_reason(
                            "v2", "revoked_v2", "default", NOW
                        ),
                    ),
                    "v2",
                    configuration="revoked_v2",
                ),
                imp(
                    refused(
                        "bundle_rejected",
                        bundle_reason=verifier_reason(
                            "v2", "default", "default", expired_now
                        ),
                    ),
                    "v2",
                    now=expired_now,
                ),
                state(active="v1", last_green="v1", floor=7, receipts=5),
            ],
        ),
        scenario(
            "trust_root_rotation",
            "After rotation only bundles also signed by the new key verify.",
            [
                imp(
                    refused(
                        "bundle_rejected",
                        bundle_reason=verifier_reason(
                            "v2", "rotation", "default", rotation_now
                        ),
                    ),
                    "v2",
                    configuration="rotation",
                    now=rotation_now,
                ),
                imp(
                    activated("v2"), "v2_ab", configuration="rotation", now=rotation_now
                ),
                imp(present("v2"), "v2_ab", configuration="rotation", now=rotation_now),
            ],
        ),
        scenario(
            "downgrade_is_refused_and_persists",
            "The ledger floor refuses older sequences, also after a restart.",
            [
                imp(activated("v2"), "v2"),
                imp(
                    refused(
                        "bundle_rejected",
                        bundle_reason=verifier_reason(
                            "v1", "default", "default", NOW, floor=8
                        ),
                    ),
                    "v1",
                ),
                restart(),
                recover(0),
                imp(
                    refused(
                        "bundle_rejected",
                        bundle_reason=verifier_reason(
                            "v1", "default", "default", NOW, floor=8
                        ),
                    ),
                    "v1",
                ),
                imp(
                    refused(
                        "bundle_rejected",
                        bundle_reason=verifier_reason(
                            "v2", "default", "default", NOW, floor=9
                        ),
                    ),
                    "v2",
                    configuration="min_sequence_9",
                ),
                imp(
                    activated("v3", previous="v2"), "v3", configuration="min_sequence_9"
                ),
                state(active="v3", last_green="v3", rollback="v2", floor=9),
            ],
        ),
        scenario(
            "key_pins_and_expectations",
            "Pins constrain the delivered trust policy; expectations pin the bundle.",
            [
                imp(activated("v1"), "v1", configuration="pinned"),
                imp(
                    refused("trust_anchor_unpinned"), "v2", configuration="pinned_other"
                ),
                imp(
                    refused("bundle_not_expected", known="v2", active="v1"),
                    "v2",
                    configuration="expect_v1",
                ),
                imp(present("v1"), "v1", configuration="expect_v1"),
            ],
        ),
        scenario(
            "managed_configuration_errors",
            "Missing, unsupported, and malformed managed configurations change nothing.",
            [
                *v1_active,
                imp(refused("managed_config_absent"), "v2", configuration=None),
                imp(
                    refused("config_version_unsupported"),
                    "v2",
                    configuration="version_2",
                ),
                imp(refused("trust_policy_missing"), "v2", configuration="no_policy"),
                imp(refused("trust_policy_invalid"), "v2", configuration="bad_policy"),
                imp(refused("config_invalid"), "v2", configuration="unknown_key"),
                imp(refused("config_invalid"), "v2", configuration="bad_pin"),
                imp(
                    activated("v2", previous="v1"),
                    "v2",
                    configuration="foreign_keys_only",
                ),
            ],
        ),
        scenario(
            "hostile_archives_are_refused",
            "Unsafe, duplicate, unsupported, and malformed archives never stage.",
            [
                *v1_active,
                imp(refused("entry_unsafe"), "traversal"),
                imp(refused("entry_duplicate"), "duplicate"),
                imp(refused("entry_duplicate"), "case_duplicate"),
                imp(refused("entry_unsafe"), "symlink"),
                imp(refused("archive_format_unsupported"), "encrypted"),
                imp(refused("archive_format_unsupported"), "zip64"),
                imp(refused("archive_format_unsupported"), "unsupported_method"),
                imp(refused("archive_format_unsupported"), "not_zip"),
                imp(refused("archive_malformed"), "truncated"),
                imp(refused("archive_malformed"), "local_mismatch"),
                imp(
                    refused("bundle_rejected", bundle_reason="file_missing"),
                    "no_manifest",
                ),
                imp(refused("archive_format_unsupported"), "not_zip", handoff="stream"),
                state(active="v1", last_green="v1", floor=7, receipts=13),
            ],
        ),
        scenario(
            "inventory_and_digest_failures_happen_before_activation",
            "Undeclared, missing, resized, and tampered files never activate.",
            [
                *v1_active,
                imp(
                    refused(
                        "verification_failed",
                        bundle_reason="file_undeclared",
                        known="v2",
                        active="v1",
                    ),
                    "undeclared",
                ),
                imp(
                    refused(
                        "verification_failed",
                        bundle_reason="file_missing",
                        known="v2",
                        active="v1",
                    ),
                    "missing",
                ),
                imp(
                    refused(
                        "verification_failed",
                        bundle_reason="file_size_mismatch",
                        known="v2",
                        active="v1",
                    ),
                    "size_mismatch",
                ),
                imp(
                    refused("archive_malformed", known="v2", active="v1"),
                    "corrupt_deflate",
                ),
                imp(
                    refused(
                        "verification_failed",
                        bundle_reason="file_digest_mismatch",
                        known="v2",
                        active="v1",
                    ),
                    "tampered",
                ),
                state(
                    active="v1",
                    last_green="v1",
                    quarantined=(("v2", "file_digest_mismatch"),),
                    floor=7,
                ),
                imp(activated("v2", previous="v1"), "v2"),
            ],
        ),
        scenario(
            "compression_ratio_budget",
            "An entry that inflates beyond the ratio budget is refused before staging.",
            [
                imp(refused("budget_compression_ratio"), "bomb"),
                imp(activated("v2"), "v2"),
            ],
            limits={"max_compression_ratio": 100},
        ),
        scenario(
            "entry_count_budget",
            "An archive with more entries than the budget is refused before staging.",
            [
                imp(refused("budget_file_count"), "v1"),
                imp(refused("budget_file_count"), "v1", handoff="stream"),
            ],
            limits={"max_entries": 4},
        ),
        scenario(
            "archive_byte_budget",
            "A file or stream larger than the archive budget is refused.",
            [
                imp(refused("budget_archive_bytes"), "v1"),
                imp(refused("budget_archive_bytes"), "v1", handoff="stream"),
                imp(refused("budget_archive_bytes"), "v2", handoff="stream"),
            ],
            limits={"max_archive_bytes": 512},
        ),
        scenario(
            "uncompressed_byte_budget",
            "Declared uncompressed bytes above the budget are refused.",
            [imp(refused("budget_total_bytes"), "v1")],
            limits={"max_total_bytes": 256},
        ),
        scenario(
            "incompatible_platform",
            "A bundle without a runtime for the device is refused.",
            [
                imp(
                    refused(
                        "bundle_rejected",
                        bundle_reason=verifier_reason(
                            "v1", "default", "no_runtime", NOW
                        ),
                    ),
                    "v1",
                )
            ],
            target="no_runtime",
        ),
        scenario(
            "storage_pressure_keeps_last_green",
            "Low storage refuses the import; later the same delivery resumes cleanly.",
            [
                *v1_active,
                imp(
                    refused("storage_pressure", known="v2", active="v1"),
                    "v2",
                    storage_available_bytes=64,
                ),
                state(active="v1", last_green="v1", floor=7),
                imp(activated("v2", previous="v1"), "v2"),
            ],
        ),
        scenario(
            "cancellation_resumes",
            "Cancellation keeps verified chunks; the next delivery resumes and activates.",
            [
                *v1_active,
                imp(
                    refused("cancelled", known="v2", active="v1"),
                    "v2",
                    cancel_at=at("chunk_committed", 3),
                ),
                state(active="v1", last_green="v1", floor=7),
                imp(activated("v2", previous="v1", resumed=True), "v2"),
            ],
        ),
        scenario(
            "process_death_while_staging",
            "A crash mid-transfer leaves v1 usable; recovery and re-delivery resume.",
            [
                *v1_active,
                imp({"crashed": True}, "v2", crash_at=at("chunk_committed", 4)),
                restart(),
                recover(0),
                state(active="v1", last_green="v1", floor=7),
                imp(
                    activated("v2", previous="v1", resumed=True), "v2", handoff="stream"
                ),
            ],
        ),
        scenario(
            "process_death_after_commit",
            "A crash after commit leaves a ready generation that a re-delivery activates.",
            [
                *v1_active,
                imp({"crashed": True}, "v2", crash_at=at("generation_committed")),
                restart(),
                recover(0),
                state(active="v1", last_green="v1", floor=7),
                imp(
                    receipt(
                        "activated",
                        generation="v2",
                        previous="v1",
                        active="v2",
                        last_green="v2",
                        files_verified=FILES,
                    ),
                    "v2",
                ),
            ],
        ),
        scenario(
            "process_death_after_spooling",
            "A crash after spooling a stream leaves a spool file that recovery purges.",
            [
                *v1_active,
                imp(
                    {"crashed": True},
                    "v2",
                    handoff="stream",
                    crash_at=at("archive_spooled"),
                ),
                restart(),
                recover(1),
                recover(0),
                imp(activated("v2", previous="v1"), "v2", handoff="stream"),
            ],
        ),
        scenario(
            "process_death_at_activation",
            "A crash while switching the pointer leaves either generation complete.",
            [
                *v1_active,
                imp({"crashed": True}, "v2", crash_at=at("pointer_committed")),
                restart(),
                recover(0),
                imp(
                    receipt(
                        "already_present",
                        generation="v2",
                        previous="v2",
                        active="v2",
                        last_green="v2",
                        files_verified=FILES,
                    ),
                    "v2",
                ),
                state(active="v2", last_green="v2", rollback="v1", floor=8),
            ],
        ),
    ]
    for item in scenarios:
        for step in item["steps"]:
            expect = step.get("expect", {})
            if step["op"] == "import" and "outcome" in expect:
                if expect["outcome"] not in OUTCOMES:
                    raise SystemExit(f"{item['id']}: unknown outcome")
                if expect["reason"] is not None and expect["reason"] not in REASONS:
                    raise SystemExit(f"{item['id']}: unknown reason {expect['reason']}")
    return scenarios


def build_vectors() -> dict:
    """Return the complete vector document."""

    return {
        "archives": ARCHIVES,
        "bundles": copy.deepcopy(BUNDLES),
        "checkpoints": list(CHECKPOINTS),
        "chunk_bytes": CHUNK_BYTES,
        "clock": CLOCK,
        "config_keys": KEYS,
        "config_version": CONFIG_VERSION,
        "configs": CONFIGS,
        "default_limits": DEFAULT_LIMITS,
        "generations": GENERATION_ALIAS,
        "identities": {
            "bundle_id": BUNDLES["v1"]["bundle_id"],
            "bundle_key": BC.bundle_key(BUNDLES["v1"]["bundle_id"]),
            "key_pins": {kid: fingerprint(kid) for kid in (ROOT_A, ROOT_B, ROGUE)},
            "trust_digests": {
                name: BC.trust_digest(value) for name, value in POLICIES.items()
            },
        },
        "notice": (
            "TEST-ONLY vectors. Bundles are signed with the TEST-ONLY keys of "
            "tests/fixtures/portable_bundle; never trust them in production."
        ),
        "outcomes": list(OUTCOMES),
        "reason_codes": list(REASONS),
        "scenarios": build_scenarios(),
        "schema": "openmed.managed_import.vectors",
        "synthetic_only": True,
        "targets": TARGETS,
        "test_only_keys": True,
        "trust_policies": POLICIES,
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
            print("managed import vectors are stale")
            return 1
        return 0
    VECTORS_PATH.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
