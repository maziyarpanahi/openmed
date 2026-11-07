"""Hand-built synthetic TIFF/BigTIFF slides; no real media or identifiers."""

import struct

SENTINELS = (
    b"SYNTHETIC_NAME_ALICE",
    b"SYNTHETIC_ACCESSION_123",
    "SYNTHETIC_患者".encode(),
    b"SYNTHETIC_OPERATOR_DATE_20000101",
)


def slide(
    *,
    big=False,
    endian="<",
    roles=("level", "thumbnail", "level", "label", "macro"),
    vendor=True,
    extra_tags=None,
    multiple_tiles=False,
    deflate=False,
):
    """Build independently encoded directories with known payloads."""
    word, count_size, entry_size = (8, 8, 20) if big else (4, 2, 12)
    wf, cf = ("Q", "Q") if big else ("I", "H")

    def pack(value, fmt=wf):
        return struct.pack(endian + fmt, value)

    output = bytearray(b"II" if endian == "<" else b"MM")
    output += pack(43 if big else 42, "H")
    output += (pack(8, "H") + pack(0, "H")) if big else b""
    output += pack(0)

    def append(raw):
        output.extend(b"\0" * (-len(output) % (8 if big else 2)))
        offset = len(output)
        output.extend(raw)
        return offset

    previous_pointer = 8 if big else 4
    level_index = 0
    locations = []
    for index, role in enumerate(roles):
        tiled = role == "level"
        size = (32 if multiple_tiles else 16) >> level_index if tiled else 4
        subtype = {
            "level": 0 if level_index == 0 else 1,
            "thumbnail": 0,
            "label": 1,
            "macro": 9,
            "unknown": 4,
        }[role]
        if tiled:
            level_index += 1
        if tiled:
            count = ((size + 15) // 16) ** 2
            payloads = [bytes([30 + index + tile]) * 256 for tile in range(count)]
            if deflate:
                import zlib

                payloads = [zlib.compress(payload) for payload in payloads]
        else:
            payloads = [SENTINELS[index % len(SENTINELS)]]
        offsets = [append(payload) for payload in payloads]
        payload_at = offsets[0]
        desc = (
            (b"Aperio synthetic" if vendor else b"Synthetic pyramid")
            + b"|AppMag=40|MPP=0.25|Operator="
            + SENTINELS[-1]
            + b"\0"
        )
        entries = {
            254: (4, 1, pack(subtype, "I")),
            256: (4, 1, pack(size if tiled else len(payloads[0]), "I")),
            257: (4, 1, pack(size if tiled else 1, "I")),
            258: (3, 1, pack(8, "H")),
            259: (3, 1, pack(8 if tiled and deflate else 1, "H")),
            262: (3, 1, pack(1, "H")),
            270: (2, len(desc), desc),
            277: (3, 1, pack(1, "H")),
            324 if tiled else 273: (
                16 if big else 4,
                len(offsets),
                b"".join(pack(o) for o in offsets),
            ),
            325 if tiled else 279: (
                16 if big else 4,
                len(payloads),
                b"".join(pack(len(b)) for b in payloads),
            ),
        }
        if tiled:
            entries.update({322: (4, 1, pack(16, "I")), 323: (4, 1, pack(16, "I"))})
        else:
            entries[278] = (4, 1, pack(1, "I"))
        entries.update(extra_tags or {})
        encoded = bytearray()
        for tag, (kind, n, raw) in sorted(entries.items()):
            value = pack(append(raw)) if len(raw) > word else raw.ljust(word, b"\0")
            encoded += pack(tag, "H") + pack(kind, "H") + pack(n) + value
        offset = append(pack(len(entries), cf) + encoded + pack(0))
        output[previous_pointer : previous_pointer + word] = pack(offset)
        previous_pointer = offset + count_size + len(entries) * entry_size
        locations.append(
            {
                "ifd": offset,
                "next": previous_pointer,
                "payload": payload_at,
                "entries": {
                    tag: offset + count_size + i * entry_size
                    for i, tag in enumerate(sorted(entries))
                },
            }
        )
    output.extend(b"SYNTHETIC_ORPHAN_IDENTIFIER")
    return bytes(output), locations
