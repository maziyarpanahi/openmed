"""Small synthetic BigTIFF pyramids; tile bytes are deliberately not images."""

import struct


def pyramid(*, order="<", subifds=False, dimensions=((8, 8), (4, 4)), compression=1):
    """Build aligned directories, tile tables and four-byte synthetic tiles."""
    directories = [16 + 256 * i for i in range(len(dimensions))]
    payload = bytearray(16 + 256 * len(dimensions))
    payload[:16] = (b"II" if order == "<" else b"MM") + struct.pack(
        order + "HHHQ", 43, 8, 0, directories[0]
    )
    tile_entries = []
    for index, (width, height) in enumerate(dimensions):
        count = ((width + 3) // 4) * ((height + 3) // 4)
        tags = {
            254: (4, (0 if index == 0 else 1,)),
            256: (4, (width,)),
            257: (4, (height,)),
            259: (3, (compression,)),
            277: (3, (1,)),
            284: (3, (1,)),
            322: (4, (4,)),
            323: (4, (4,)),
            324: (16, (0,) * count),
            325: (16, (4,) * count),
        }
        if subifds and index == 0 and len(directories) > 1:
            tags[330] = (18, tuple(directories[1:]))
        base = directories[index]
        struct.pack_into(order + "Q", payload, base, len(tags))
        for ordinal, (tag, (kind, values)) in enumerate(sorted(tags.items())):
            start = base + 8 + ordinal * 20
            fmt = {3: "H", 4: "I", 16: "Q", 18: "Q"}[kind]
            blob = struct.pack(order + str(len(values)) + fmt, *values)
            struct.pack_into(order + "HHQ", payload, start, tag, kind, len(values))
            if len(blob) <= 8:
                location = start + 12
                payload[location : location + 8] = blob.ljust(8, b"\x00")
            else:
                payload.extend(b"\x00" * (-len(payload) % 8))
                location = len(payload)
                struct.pack_into(order + "Q", payload, start + 12, location)
                payload.extend(blob)
            if tag == 324:
                tile_entries.append((location, len(values)))
        next_ifd = (
            0 if subifds or index + 1 == len(directories) else directories[index + 1]
        )
        struct.pack_into(order + "Q", payload, base + 8 + len(tags) * 20, next_ifd)
    header_end = len(payload)
    for location, count in tile_entries:
        for i in range(count):
            struct.pack_into(order + "Q", payload, location + 8 * i, len(payload))
            payload.extend(b"TEST")
    return bytes(payload), header_end


def entry_offset(data, tag, *, level=0, order="<"):
    """Locate a synthetic directory entry for deterministic negative controls."""
    base = 16 + 256 * level
    count = struct.unpack_from(order + "Q", data, base)[0]
    for i in range(count):
        start = base + 8 + i * 20
        if struct.unpack_from(order + "H", data, start)[0] == tag:
            return start
    raise AssertionError("synthetic tag missing")


def changed_entry(data, tag, *, level=0, kind=None, count=None, value=None):
    """Replace a little-endian synthetic entry without allocating its claims."""
    payload = bytearray(data)
    start = entry_offset(data, tag, level=level)
    if kind is not None:
        struct.pack_into("<H", payload, start + 2, kind)
    if count is not None:
        struct.pack_into("<Q", payload, start + 4, count)
    if value is not None:
        struct.pack_into("<Q", payload, start + 12, value)
    return bytes(payload)
