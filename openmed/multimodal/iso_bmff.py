"""Bounded HEIF item and M4A track preflight without media decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .media_type import detect_media_type

__all__ = ["BmffPreflightError", "HeifPreflight", "M4aPreflight", "preflight_iso_bmff"]

_UINT64_MAX = (1 << 64) - 1
_IMAGE_ITEMS = {b"hvc1", b"av01", b"jpeg", b"grid", b"iovl", b"iden", b"unci"}


class BmffPreflightError(ValueError):
    """A stable, value-free refusal; source strings are never interpolated."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class HeifPreflight:
    """Declared primary geometry and unique item-role counts, without payloads."""

    width: int
    height: int
    item_count: int
    thumbnail_count: int
    auxiliary_count: int
    burst_count: int
    has_exif: bool
    has_xmp: bool
    status: Literal["decoder_required"] = "decoder_required"


@dataclass(frozen=True, slots=True)
class M4aPreflight:
    """Declared audio metadata; duration is media time, not edited playback."""

    codec_category: str
    channels: int
    sample_rate_hz: float
    duration_seconds: float
    status: Literal["decoder_required"] = "decoder_required"
    notice: Literal["non_diagnostic"] = "non_diagnostic"
    reviewer_confirmation_required: bool = True


@dataclass(frozen=True, slots=True)
class _Box:
    kind: bytes
    start: int
    end: int
    depth: int


class _Walker:
    def __init__(self, data: bytes, limit: int, depth: int, count: int, size: int):
        self.data = memoryview(data)
        self.limit = limit
        self.max_depth = depth
        self.max_count = count
        self.max_size = size
        self.count = 0

    def read(self, offset: int, size: int, end: int) -> bytes:
        if size > end - offset or offset < 0:
            raise BmffPreflightError("bmff_truncated")
        if size > self.limit - offset:
            raise BmffPreflightError("bmff_metadata_out_of_bounds")
        return bytes(self.data[offset : offset + size])

    def boxes(self, start: int, end: int, depth: int = 1) -> list[_Box]:
        if depth > self.max_depth:
            raise BmffPreflightError("bmff_depth_limit")
        result = []
        while start < end:
            self.count += 1
            if self.count > self.max_count:
                raise BmffPreflightError("bmff_box_count_limit")
            raw = self.read(start, 8, end)
            size = int.from_bytes(raw[:4], "big")
            header = 8
            if size == 1:
                size = int.from_bytes(self.read(start + 8, 8, end), "big")
                header = 16
            elif size == 0:
                size = end - start
            if size < header:
                raise BmffPreflightError("bmff_box_size_invalid")
            if size > _UINT64_MAX - start:
                raise BmffPreflightError("bmff_size_overflow")
            if size > self.max_size:
                raise BmffPreflightError("bmff_declared_size_limit")
            if size > end - start:
                raise BmffPreflightError("bmff_truncated")
            result.append(_Box(raw[4:8], start + header, start + size, depth))
            start += size
        return result

    def children(self, box: _Box, skip: int = 0) -> list[_Box]:
        self.read(box.start, skip, box.end)
        return self.boxes(box.start + skip, box.end, box.depth + 1)

    def payload(self, box: _Box) -> bytes:
        return self.read(box.start, box.end - box.start, box.end)


def _one(boxes: list[_Box], kind: bytes) -> _Box:
    matches = [box for box in boxes if box.kind == kind]
    if len(matches) != 1:
        raise BmffPreflightError("bmff_metadata_missing_or_duplicate")
    return matches[0]


class _Fields:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def take(self, size: int) -> bytes:
        if size > len(self.data) - self.offset:
            raise BmffPreflightError("bmff_truncated")
        result = self.data[self.offset : self.offset + size]
        self.offset += size
        return result

    def uint(self, size: int) -> int:
        return int.from_bytes(self.take(size), "big")

    def cstring(self) -> bytes:
        end = self.data.find(b"\0", self.offset)
        if end < 0:
            raise BmffPreflightError("bmff_truncated")
        return self.take(end - self.offset + 1)[:-1]

    def full(self, versions: tuple[int, ...], flags: int = 0) -> tuple[int, int]:
        version, actual_flags = self.uint(1), self.uint(3)
        if version not in versions or actual_flags & ~flags:
            raise BmffPreflightError("bmff_layout_unsupported")
        return version, actual_flags

    def done(self) -> None:
        if self.offset != len(self.data):
            raise BmffPreflightError("bmff_metadata_invalid")


def preflight_iso_bmff(
    source: bytes,
    *,
    max_header_bytes: int = 1024 * 1024,
    max_depth: int = 16,
    max_boxes: int = 4096,
    max_declared_box_bytes: int = 4 * 1024**3,
    max_pixels: int = 100_000_000,
) -> HeifPreflight | M4aPreflight:
    """Read bounded item/track headers from caller-owned in-memory bytes.

    Args:
        source: Complete local container bytes; never logged or retained.
        max_header_bytes: Absolute offset bound, including skipped payloads.
        max_depth: Maximum nesting depth (at most 64).
        max_boxes: Maximum number of parsed boxes.
        max_declared_box_bytes: Maximum declared size, including media boxes.
        max_pixels: Maximum primary-image area.

    Returns:
        Numeric/categorical metadata with ``decoder_required`` status.

    Raises:
        BmffPreflightError: Malformed, ambiguous, unsupported or over-limit input.
        ValueError: Invalid limits.
        TypeError: Source is not bytes.

    Media and unknown box payloads are skipped using offsets, never decoded.
    Headers after the byte bound are refused even if the source contains them.
    This is not validation of payloads, EXIF scrubbing or clinical evidence.
    """
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    limits = (
        max_header_bytes,
        max_depth,
        max_boxes,
        max_declared_box_bytes,
        max_pixels,
    )
    if any(type(value) is not int or value <= 0 for value in limits) or max_depth > 64:
        raise ValueError("limits must be positive integers; depth must not exceed 64")
    walker = _Walker(
        source, max_header_bytes, max_depth, max_boxes, max_declared_box_bytes
    )
    top = walker.boxes(0, len(source))
    ftyp = _one(top, b"ftyp")
    if not top or top[0] != ftyp:
        raise BmffPreflightError("bmff_signature_invalid")
    # Detection is intentionally prefix-only; do not expand its search bound.
    media_type = detect_media_type(walker.read(0, min(ftyp.end, 132), len(source)))
    if media_type is None:
        raise BmffPreflightError("bmff_signature_invalid")
    if any(box.kind == b"moof" for box in top):
        raise BmffPreflightError("bmff_layout_unsupported")
    if media_type.startswith("image/"):
        if any(box.kind == b"moov" for box in top):
            raise BmffPreflightError("bmff_layout_unsupported")
        return _heif(walker, _one(top, b"meta"), max_pixels)
    return _m4a(walker, _one(top, b"moov"))


def _heif(w: _Walker, meta: _Box, max_pixels: int) -> HeifPreflight:
    _Fields(w.read(meta.start, 4, meta.end)).full((0,))
    children = w.children(meta, 4)
    pitm = _Fields(w.payload(_one(children, b"pitm")))
    version, _ = pitm.full((0, 1))
    primary = pitm.uint(2 if version == 0 else 4)
    pitm.done()

    iinf = _one(children, b"iinf")
    fields = _Fields(w.read(iinf.start, min(8, iinf.end - iinf.start), iinf.end))
    version, _ = fields.full((0, 1))
    count = fields.uint(2 if version == 0 else 4)
    entries = w.children(iinf, 6 if version == 0 else 8)
    if count != len(entries):
        raise BmffPreflightError("bmff_metadata_invalid")
    items: dict[int, bytes] = {}
    exif = xmp = False
    for entry in entries:
        if entry.kind != b"infe":
            raise BmffPreflightError("bmff_layout_unsupported")
        f = _Fields(w.payload(entry))
        version, _ = f.full((2, 3), 1)
        item_id = f.uint(2 if version == 2 else 4)
        if item_id == 0 or item_id in items or f.uint(2) != 0:
            raise BmffPreflightError("bmff_metadata_invalid")
        kind = f.take(4)
        f.cstring()  # Item names are neither decoded nor returned.
        items[item_id] = kind
        exif |= kind == b"Exif"
        if kind == b"mime":
            xmp |= f.cstring() == b"application/rdf+xml"
            if f.offset < len(f.data):
                f.cstring()  # optional content encoding
        elif kind == b"uri ":
            f.cstring()
        f.done()
    if items.get(primary) not in _IMAGE_ITEMS:
        raise BmffPreflightError("bmff_layout_unsupported")

    iprp = w.children(_one(children, b"iprp"))
    properties = w.children(_one(iprp, b"ipco"))
    associations: dict[int, list[int]] = {}
    for association in [box for box in iprp if box.kind == b"ipma"]:
        f = _Fields(w.payload(association))
        version, flags = f.full((0, 1), 1)
        for _ in range(f.uint(4)):
            item_id = f.uint(2 if version == 0 else 4)
            if item_id not in items or item_id in associations:
                raise BmffPreflightError("bmff_metadata_invalid")
            indices = []
            for _ in range(f.uint(1)):
                index = f.uint(2 if flags & 1 else 1) & (0x7FFF if flags & 1 else 0x7F)
                if index > len(properties):
                    raise BmffPreflightError("bmff_metadata_invalid")
                if index:
                    indices.append(index)
            associations[item_id] = indices
        f.done()
    sizes = []
    for index in associations.get(primary, []):
        prop = properties[index - 1]
        if prop.kind == b"ispe":
            f = _Fields(w.payload(prop))
            f.full((0,))
            sizes.append((f.uint(4), f.uint(4)))
            f.done()
    if len(sizes) != 1:
        raise BmffPreflightError("bmff_metadata_missing_or_duplicate")
    width, height = sizes[0]
    if not width or not height:
        raise BmffPreflightError("bmff_metadata_invalid")
    if width > max_pixels // height:
        raise BmffPreflightError("bmff_pixel_limit")

    roles: dict[bytes, set[int]] = {b"thmb": set(), b"auxl": set()}
    graph: dict[int, set[int]] = {item: set() for item in items}
    refs = [box for box in children if box.kind == b"iref"]
    if len(refs) > 1:
        raise BmffPreflightError("bmff_metadata_missing_or_duplicate")
    for ref in refs:
        f = _Fields(w.read(ref.start, 4, ref.end))
        version, _ = f.full((0, 1))
        for box in w.children(ref, 4):
            f = _Fields(w.payload(box))
            origin = f.uint(2 if version == 0 else 4)
            targets = {f.uint(2 if version == 0 else 4) for _ in range(f.uint(2))}
            f.done()
            if origin not in items or not targets or not targets <= items.keys():
                raise BmffPreflightError("bmff_metadata_invalid")
            graph[origin].update(targets)
            if box.kind in roles:
                if items[origin] not in _IMAGE_ITEMS:
                    raise BmffPreflightError("bmff_metadata_invalid")
                roles[box.kind].add(origin)
    _acyclic(graph)
    burst: set[int] = set()
    for group in [box for box in children if box.kind == b"grpl"]:
        for box in w.children(group):
            if box.kind != b"brst":
                continue
            f = _Fields(w.payload(box))
            f.full((0,))
            f.uint(4)  # opaque group identifier
            entities = {f.uint(4) for _ in range(f.uint(4))}
            if not entities or any(
                items.get(item) not in _IMAGE_ITEMS for item in entities
            ):
                raise BmffPreflightError("bmff_layout_unsupported")
            # brst may carry group-specific trailing fields; they are not exposed.
            burst.update(entities)
    return HeifPreflight(
        width,
        height,
        len(items),
        len(roles[b"thmb"]),
        len(roles[b"auxl"]),
        len(burst),
        exif,
        xmp,
    )


def _acyclic(graph: dict[int, set[int]]) -> None:
    incoming = dict.fromkeys(graph, 0)
    for targets in graph.values():
        for target in targets:
            incoming[target] += 1
    queue = [node for node, degree in incoming.items() if not degree]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for target in graph[node]:
            incoming[target] -= 1
            if not incoming[target]:
                queue.append(target)
    if visited != len(graph):
        raise BmffPreflightError("bmff_item_cycle")


def _m4a(w: _Walker, moov: _Box) -> M4aPreflight:
    tracks = []
    for track in w.children(moov):
        if track.kind != b"trak":
            continue
        children = w.children(track)
        if any(box.kind == b"edts" for box in children):
            raise BmffPreflightError("bmff_layout_unsupported")
        mdia = w.children(_one(children, b"mdia"))
        hdlr = _Fields(w.payload(_one(mdia, b"hdlr")))
        hdlr.full((0,))
        hdlr.take(4)
        if hdlr.take(4) != b"soun":
            continue
        f = _Fields(w.payload(_one(mdia, b"mdhd")))
        version, _ = f.full((0, 1))
        f.take(8 if version == 0 else 16)
        timescale = f.uint(4)
        duration = f.uint(4 if version == 0 else 8)
        f.take(4)  # language and predefined
        f.done()
        if not timescale or duration == (1 << (32 if version == 0 else 64)) - 1:
            raise BmffPreflightError("bmff_metadata_invalid")
        minf = w.children(_one(mdia, b"minf"))
        stbl = w.children(_one(minf, b"stbl"))
        stsd = _one(stbl, b"stsd")
        f = _Fields(w.read(stsd.start, 8, stsd.end))
        f.full((0,))
        if f.uint(4) != 1:
            raise BmffPreflightError("bmff_layout_unsupported")
        entries = w.children(stsd, 8)
        if len(entries) != 1:
            raise BmffPreflightError("bmff_metadata_invalid")
        entry = entries[0]
        codec = {b"mp4a": "mpeg4_audio", b"alac": "apple_lossless"}.get(entry.kind)
        if codec is None:
            raise BmffPreflightError("bmff_codec_unsupported")
        f = _Fields(w.read(entry.start, 28, entry.end))
        f.take(8)
        if f.uint(2) != 0:
            raise BmffPreflightError("bmff_layout_unsupported")
        f.take(6)
        channels = f.uint(2)
        f.take(6)
        rate = f.uint(4) / 65536
        if not channels or not rate:
            raise BmffPreflightError("bmff_metadata_invalid")
        # Validate box envelopes of codec configuration, without parsing payloads.
        w.children(entry, 28)
        tracks.append(M4aPreflight(codec, channels, rate, duration / timescale))
    if len(tracks) != 1:
        raise BmffPreflightError("bmff_audio_track_ambiguous")
    return tracks[0]
