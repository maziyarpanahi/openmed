"""Versioned, value-free projections shared with OpenMedKit.

These protected wire packets preserve source coordinates and review holds. They
are not clinical approvals, custody attestations or operational log records.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol

from .asset_manifest import AssetManifest
from .evidence.fused_graph import EvidenceNode, NodeKind, PageGeometry, SpatialUnit
from .page_rotation import PageTransform
from .provenance.clock_alignment import (
    CLOCK_VERSION,
    AlignmentResult,
    AlignmentState,
    ClockReview,
)
from .provider_result import ProviderResultEnvelope
from .streaming.transcript.revision_ledger import (
    REVISION_SCHEMA_VERSION,
    RevisionLedger,
    SpanInvalidation,
    SpanReference,
)

SCHEMA = "openmed.multimodal.apple.v1"
MAX_WIRE_BYTES = 65536
MAX_INTEGER = (1 << 63) - 1


class ContractRefusal(str, Enum):
    """Closed projection failures; never interpolate rejected source values."""

    INVALID = "invalid_multimodal_contract"
    SCHEMA = "unsupported_multimodal_schema"
    CAPABILITY = "unsupported_multimodal_capability"
    TRANSFORM = "invalid_multimodal_transform"
    PROVENANCE = "incomplete_multimodal_provenance"


class AppleContractError(ValueError):
    """Controlled Python/Swift interoperability error."""

    def __init__(self, reason: ContractRefusal = ContractRefusal.INVALID) -> None:
        self.reason = reason
        super().__init__(reason.value)


class ProjectionKind(str, Enum):
    """Supported metadata projections, independent of inference engines."""

    ASSET = "asset"
    VISUAL = "visual"
    TRANSCRIPT = "transcript"
    PROVIDER = "provider"
    CLOCK = "clock"


def _check(condition: bool, reason: ContractRefusal = ContractRefusal.INVALID) -> None:
    if not condition:
        raise AppleContractError(reason)


def _keys(value: object, names: str) -> dict[str, Any]:
    _check(type(value) is dict and set(value) == set(names.split()))
    assert isinstance(value, dict)
    return value


def _int(value: object, minimum: int = 0, maximum: int = MAX_INTEGER) -> int:
    _check(type(value) is int and minimum <= value <= maximum)
    assert isinstance(value, int)
    return value


def _hex(value: object, width: int = 64) -> None:
    _check(
        type(value) is str
        and re.fullmatch(r"[0-9a-f]{" + str(width) + "}", value) is not None,
        ContractRefusal.PROVENANCE,
    )


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _bits(value: float | int) -> str:
    _check(type(value) in (float, int) and math.isfinite(value))
    return struct.pack(">d", 0.0 if value == 0 else float(value)).hex()


def _number(value: object, minimum: float = 0, maximum: float = 2**31 - 1) -> float:
    _check(type(value) is str and re.fullmatch(r"[0-9a-f]{16}", value) is not None)
    assert isinstance(value, str)
    number = struct.unpack(">d", bytes.fromhex(value))[0]
    _check(math.isfinite(number) and minimum <= number <= maximum)
    _check(_bits(number) == value)
    return number


def _pair(value: object, maximum: int, *, allow_empty: bool = False) -> list[int]:
    _check(type(value) is list and len(value) == 2)
    assert isinstance(value, list)
    first, last = (_int(x, maximum=maximum) for x in value)
    _check(first <= last if allow_empty else first < last)
    return [first, last]


def _asset(payload: dict[str, Any]) -> None:
    _keys(
        payload,
        "source_schema manifest_digest asset_ref media_type sha256 byte_size pages width height frames duration_f64",
    )
    _check(payload["source_schema"] == 1, ContractRefusal.SCHEMA)
    _hex(payload["manifest_digest"])
    _hex(payload["asset_ref"], 32)
    _hex(payload["sha256"])
    duration = (
        None if payload["duration_f64"] is None else _number(payload["duration_f64"])
    )
    AssetManifest(
        asset_id=payload["asset_ref"],
        media_type=payload["media_type"],
        sha256=payload["sha256"],
        byte_size=payload["byte_size"],
        pages=payload["pages"],
        width=payload["width"],
        height=payload["height"],
        frames=payload["frames"],
        duration_seconds=duration,
    )
    _int(payload["source_schema"], 1, 1)


def _visual(payload: dict[str, Any]) -> None:
    _keys(
        payload,
        "source_schema node_ref source_digest kind page width_f64 height_f64 unit box_f64 span rotation display_width_f64 display_height_f64 display_box_f64",
    )
    _check(
        payload["source_schema"] == "openmed.multimodal.fused-evidence.v1",
        ContractRefusal.SCHEMA,
    )
    _hex(payload["node_ref"], 32)
    _hex(payload["source_digest"])
    _check(
        payload["kind"] in {"page_geometry", "pixels", "ocr_span"},
        ContractRefusal.CAPABILITY,
    )
    width, height = _number(payload["width_f64"]), _number(payload["height_f64"])
    region = payload["box_f64"]
    if region is not None:
        _check(type(region) is list and len(region) == 4)
        region = tuple(_number(x) for x in region)
    span = payload["span"]
    if span is not None:
        span = tuple(_pair(span, 2**31 - 1))
    EvidenceNode(
        node_id=payload["node_ref"],
        source_digest=payload["source_digest"],
        kind=NodeKind(payload["kind"]),
        geometry=PageGeometry(
            page=payload["page"],
            width=width,
            height=height,
            unit=SpatialUnit(payload["unit"]),
        ),
        region=region,
        span=span,
    )
    rotation = payload["rotation"]
    _check(
        type(rotation) is int and rotation in (0, 90, 180, 270),
        ContractRefusal.TRANSFORM,
    )
    transform = PageTransform((width, height), rotation)
    expected = None
    if region is not None:
        transformed = transform.bbox(region)
        _check(
            transformed[0] < transformed[2] and transformed[1] < transformed[3],
            ContractRefusal.TRANSFORM,
        )
        expected = [_bits(x) for x in transformed]
    _check(
        payload["display_width_f64"] == _bits(transform.target_size.width)
        and payload["display_height_f64"] == _bits(transform.target_size.height)
        and payload["display_box_f64"] == expected,
        ContractRefusal.TRANSFORM,
    )


def _transcript(payload: dict[str, Any]) -> None:
    _keys(
        payload,
        "source_schema snapshot_digest session_ref segment_ref span_ref original_revision revision original_span token_refs length finalized invalidation current_span time_ms sample_rate_hz sample_bounds",
    )
    _check(payload["source_schema"] == REVISION_SCHEMA_VERSION, ContractRefusal.SCHEMA)
    _hex(payload["snapshot_digest"])
    for name in ("session_ref", "segment_ref", "span_ref"):
        _hex(payload[name], 32)
    revision = _int(payload["revision"], maximum=4096)
    _int(payload["original_revision"], maximum=revision)
    _pair(payload["original_span"], 65536)
    refs = payload["token_refs"]
    _check(type(refs) is list and 0 < len(refs) <= 4096)
    for ref in refs:
        _hex(ref, 32)
    _check(len(set(refs)) == len(refs))
    length = _int(payload["length"], maximum=65536)
    _check(type(payload["finalized"]) is bool)
    rate = _int(payload["sample_rate_hz"], 1, 384000)
    if payload["invalidation"] is not None:
        _check(payload["invalidation"] in {item.value for item in SpanInvalidation})
        _check(
            all(
                payload[k] is None for k in ("current_span", "time_ms", "sample_bounds")
            )
        )
    else:
        _pair(payload["current_span"], length)
        first, last = _pair(payload["time_ms"], 86400000, allow_empty=True)
        # A declared target sample grid, not a new acoustic alignment claim.
        _check(
            payload["sample_bounds"]
            == [first * rate // 1000, (last * rate + 999) // 1000]
        )
        _pair(payload["sample_bounds"], 86400000 * 384000 // 1000, allow_empty=True)


def _provider(payload: dict[str, Any]) -> None:
    _keys(
        payload,
        "schema_version provider_id model_id input_digest output_digest outcome abstention_code duration_f64 count_metadata",
    )
    data = dict(payload)
    data["duration_ms"] = _number(data.pop("duration_f64"), maximum=86400000)
    _check(
        data["schema_version"] == "openmed.multimodal.provider_result.v1",
        ContractRefusal.SCHEMA,
    )
    for name in ("input_digest", "output_digest"):
        if name == "input_digest" or data[name] is not None:
            _hex(data[name])
    ProviderResultEnvelope.from_dict(data)


def _clock(payload: dict[str, Any]) -> None:
    _keys(
        payload,
        "source_schema source_result_digest state reasons source_reference_digest target_reference_digest source_clock_digest target_clock_digest policy_digest target_start_ns target_end_ns guaranteed_overlap_ns uncertainty_ns",
    )
    _check(payload["source_schema"] == CLOCK_VERSION, ContractRefusal.SCHEMA)
    for key in (
        "source_result_digest",
        "source_reference_digest",
        "target_reference_digest",
        "source_clock_digest",
        "target_clock_digest",
        "policy_digest",
    ):
        _hex(payload[key])
    reasons = payload["reasons"]
    _check(type(reasons) is list and len(reasons) <= len(ClockReview))
    _check(
        all(
            type(code) is str and code in {x.value for x in ClockReview}
            for code in reasons
        )
    )
    _check(len(set(reasons)) == len(reasons))
    _check(payload["state"] in {x.value for x in AlignmentState})
    timing = (
        "target_start_ns",
        "target_end_ns",
        "guaranteed_overlap_ns",
        "uncertainty_ns",
    )
    if payload["state"] == AlignmentState.REVIEW.value:
        _check(bool(reasons) and all(payload[k] is None for k in timing))
    else:
        _check(not reasons)
        bounds = []
        for name in ("target_start_ns", "target_end_ns"):
            bound = _keys(payload[name], "lower upper")
            lower = _int(
                bound["lower"], -9_000_000_000_000_000_000, 9_000_000_000_000_000_000
            )
            upper = _int(bound["upper"], lower, 9_000_000_000_000_000_000)
            bounds.append((lower, upper))
        overlap = payload["guaranteed_overlap_ns"]
        _check(type(overlap) is list and len(overlap) == 2)
        first = _int(overlap[0], bounds[0][1], bounds[1][0])
        last = _int(overlap[1], first + 1, bounds[1][0])
        _check(first < last)
        _int(
            payload["uncertainty_ns"],
            max(hi - lo for lo, hi in bounds),
            9_000_000_000_000_000_000,
        )
    original = {
        k: v
        for k, v in payload.items()
        if k not in {"source_schema", "source_result_digest"}
    }
    _check(
        _hash({"algorithm": CLOCK_VERSION, "result": original})
        == payload["source_result_digest"],
        ContractRefusal.PROVENANCE,
    )


_VALIDATORS = {
    ProjectionKind.ASSET: _asset,
    ProjectionKind.VISUAL: _visual,
    ProjectionKind.TRANSCRIPT: _transcript,
    ProjectionKind.PROVIDER: _provider,
    ProjectionKind.CLOCK: _clock,
}


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    _check(len(result) == len(pairs))
    return result


def _reject_constant(value: str) -> None:
    raise AppleContractError()


def _decode(wire: bytes) -> dict[str, Any]:
    _check(type(wire) is bytes and 0 < len(wire) <= MAX_WIRE_BYTES)
    try:
        data = _keys(
            json.loads(
                wire, object_pairs_hook=_object, parse_constant=_reject_constant
            ),
            "schema_version kind payload evidence_digest",
        )
        _check(data["schema_version"] == SCHEMA, ContractRefusal.SCHEMA)
        _check(
            type(data["kind"]) is str
            and data["kind"] in {k.value for k in ProjectionKind},
            ContractRefusal.CAPABILITY,
        )
        _hex(data["evidence_digest"])
        payload = data["payload"]
        _check(type(payload) is dict)
        _VALIDATORS[ProjectionKind(data["kind"])](payload)
        identity = {k: v for k, v in data.items() if k != "evidence_digest"}
        _check(_hash(identity) == data["evidence_digest"], ContractRefusal.PROVENANCE)
        _check(_json(data) == wire)
        return data
    except AppleContractError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
        UnicodeError,
    ):
        raise AppleContractError() from None


@dataclass(frozen=True, slots=True)
class MultimodalPacket:
    """Validated canonical wire metadata; every construction rechecks the packet.

    Args:
        wire: Strict canonical ASCII JSON, including its evidence digest.
    """

    wire: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _decode(self.wire)

    @property
    def kind(self) -> ProjectionKind:
        """Return the validated projection kind."""
        return ProjectionKind(json.loads(self.wire)["kind"])

    @property
    def evidence_digest(self) -> str:
        """Return the shared SHA-256 identity, without a language-specific prefix."""
        return json.loads(self.wire)["evidence_digest"]

    def payload(self) -> dict[str, Any]:
        """Return a fresh payload copy; mutation cannot change the validated packet."""
        return json.loads(self.wire)["payload"]


def _packet(kind: ProjectionKind, payload: dict[str, Any]) -> MultimodalPacket:
    body = {"schema_version": SCHEMA, "kind": kind.value, "payload": payload}
    return MultimodalPacket(_json({**body, "evidence_digest": _hash(body)}))


def project_asset(asset: AssetManifest) -> MultimodalPacket:
    """Project a validated manifest, replacing its identifier with an opaque hash."""
    _check(type(asset) is AssetManifest)
    AssetManifest.from_dict(asset.to_dict())
    return _packet(
        ProjectionKind.ASSET,
        {
            "source_schema": asset.version,
            "manifest_digest": hashlib.sha256(asset.to_json().encode()).hexdigest(),
            "asset_ref": hashlib.sha256(asset.asset_id.encode()).hexdigest()[:32],
            "media_type": asset.media_type,
            "sha256": asset.sha256,
            "byte_size": asset.byte_size,
            "pages": asset.pages,
            "width": asset.width,
            "height": asset.height,
            "frames": asset.frames,
            "duration_f64": None
            if asset.duration_seconds is None
            else _bits(asset.duration_seconds),
        },
    )


def project_visual(node: EvidenceNode, *, rotation: int = 0) -> MultimodalPacket:
    """Preserve source evidence and derive a checked quarter-turn display mapping."""
    _check(type(node) is EvidenceNode)
    _check(
        node.kind in (NodeKind.PAGE, NodeKind.PIXELS, NodeKind.OCR),
        ContractRefusal.CAPABILITY,
    )
    assert node.geometry is not None
    _check(
        type(rotation) is int and rotation in (0, 90, 180, 270),
        ContractRefusal.TRANSFORM,
    )
    geometry = node.geometry
    transform = PageTransform((geometry.width, geometry.height), rotation)
    return _packet(
        ProjectionKind.VISUAL,
        {
            "source_schema": "openmed.multimodal.fused-evidence.v1",
            "node_ref": node.node_id,
            "source_digest": node.source_digest,
            "kind": node.kind.value,
            "page": geometry.page,
            "width_f64": _bits(geometry.width),
            "height_f64": _bits(geometry.height),
            "unit": geometry.unit.value,
            "box_f64": None if node.region is None else [_bits(x) for x in node.region],
            "span": None if node.span is None else list(node.span),
            "rotation": rotation,
            "display_width_f64": _bits(transform.target_size.width),
            "display_height_f64": _bits(transform.target_size.height),
            "display_box_f64": None
            if node.region is None
            else [_bits(x) for x in transform.bbox(node.region)],
        },
    )


def project_transcript(
    ledger: RevisionLedger, reference: SpanReference, *, sample_rate_hz: int
) -> MultimodalPacket:
    """Project a registered citation with its snapshot and declared target sample grid.

    Args:
        ledger: Current source ledger, accessed serially by the caller.
        reference: Exact citation previously registered in that ledger.
        sample_rate_hz: Target grid; conversion does not assert acoustic alignment.
    """
    _check(type(ledger) is RevisionLedger and type(reference) is SpanReference)
    _int(sample_rate_hz, 1, 384000)
    try:
        snapshot = ledger.to_json()
        mapped = ledger.remap_span(reference)
        _check(ledger.to_json() == snapshot, ContractRefusal.PROVENANCE)
    except ValueError:
        raise AppleContractError(ContractRefusal.PROVENANCE) from None
    times = None
    samples = None
    if mapped.valid:
        assert mapped.start_ms is not None and mapped.end_ms is not None
        times = [mapped.start_ms, mapped.end_ms]
        samples = [
            mapped.start_ms * sample_rate_hz // 1000,
            (mapped.end_ms * sample_rate_hz + 999) // 1000,
        ]
    return _packet(
        ProjectionKind.TRANSCRIPT,
        {
            "source_schema": REVISION_SCHEMA_VERSION,
            "snapshot_digest": hashlib.sha256(snapshot.encode()).hexdigest(),
            "session_ref": reference.session_ref,
            "segment_ref": reference.segment_ref,
            "span_ref": reference.span_ref,
            "original_revision": reference.revision,
            "revision": mapped.revision,
            "original_span": [reference.start, reference.end],
            "token_refs": list(reference.token_refs),
            "length": json.loads(snapshot)["length"],
            "finalized": mapped.finalized,
            "invalidation": mapped.invalidation.value if mapped.invalidation else None,
            "current_span": [mapped.start, mapped.end] if mapped.valid else None,
            "time_ms": times,
            "sample_rate_hz": sample_rate_hz,
            "sample_bounds": samples,
        },
    )


def project_provider(result: ProviderResultEnvelope) -> MultimodalPacket:
    """Preserve the original provider outcome, abstention vocabulary and digests."""
    _check(type(result) is ProviderResultEnvelope)
    payload = result.to_dict()
    payload["duration_f64"] = _bits(payload.pop("duration_ms"))
    return _packet(ProjectionKind.PROVIDER, payload)


def project_clock(result: AlignmentResult) -> MultimodalPacket:
    """Project protected alignment bounds without running or approving a clock fit."""
    _check(type(result) is AlignmentResult)
    payload = asdict(result)
    return _packet(
        ProjectionKind.CLOCK,
        {
            "source_schema": CLOCK_VERSION,
            "source_result_digest": result.digest,
            **payload,
        },
    )


class LocalMultimodalAdapter(Protocol):
    """Injected local execution; declarations are not a network sandbox."""

    local_only: bool
    capabilities: frozenset[ProjectionKind]

    def run(self, request: MultimodalPacket) -> MultimodalPacket:
        """Return a provider outcome bound to the exact request evidence identity."""
        ...


def run_local_adapter(
    adapter: LocalMultimodalAdapter, request: MultimodalPacket
) -> MultimodalPacket:
    """Refuse unsupported capabilities before invocation and reject unbound results."""
    _check(type(request) is MultimodalPacket)
    try:
        capabilities = adapter.capabilities
        _check(
            type(capabilities) is frozenset
            and all(type(kind) is ProjectionKind for kind in capabilities)
        )
        _check(
            adapter.local_only is True and request.kind in capabilities,
            ContractRefusal.CAPABILITY,
        )
    except AppleContractError:
        raise
    except Exception:
        raise AppleContractError() from None
    try:
        response = adapter.run(request)
    except Exception:
        raise AppleContractError() from None
    _check(
        type(response) is MultimodalPacket and response.kind is ProjectionKind.PROVIDER
    )
    _check(
        response.payload()["input_digest"] == request.evidence_digest,
        ContractRefusal.PROVENANCE,
    )
    return response
