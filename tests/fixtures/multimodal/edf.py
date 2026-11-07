"""In-memory, hand-checkable EDF fixtures; never real recordings."""

import struct


def synthetic_edf(
    *,
    kind="EDF",
    onsets=("+0", "+1"),
    duration="1",
    patient="SYNTHETIC_PATIENT",
    recording="SYNTHETIC_RECORDING",
    label="ECG II",
    annotation_text="SYNTHETIC_ANNOTATION",
    extra_annotation_channel=False,
    declared=None,
    samples=((-2, -1, 0, 2), (2, 0, -1, -2)),
    other_samples=None,
    annotation_only=False,
):
    """Generate field-major headers and little-endian records without disk I/O."""

    def field(value, width):
        encoded = str(value).encode("ascii")
        assert len(encoded) <= width
        return encoded.ljust(width, b" ")

    plus = kind != "EDF"
    ns = (
        int(not annotation_only)
        + int(other_samples is not None)
        + int(plus)
        + int(extra_annotation_channel)
    )
    fixed = b"".join(
        field(value, width)
        for value, width in zip(
            (
                "0",
                patient,
                recording,
                "09.10.26",
                "12.34.56",
                256 * (ns + 1),
                kind if plus else "",
                len(onsets) if declared is None else declared,
                duration,
                ns,
            ),
            (8, 80, 80, 8, 8, 8, 44, 8, 8, 4),
            strict=True,
        )
    )
    channels = [(label, "", "mV", -1, 1, -2, 2, "", len(samples[0]), "")]
    if annotation_only:
        channels = []
    if other_samples is not None:
        channels.append(
            ("EMG", "", "uV", -10, 10, -2, 2, "", len(other_samples[0]), "")
        )
    channels += [("EDF Annotations", "", "", -1, 1, -32768, 32767, "", 128, "")] * (
        int(plus) + int(extra_annotation_channel)
    )
    header = fixed + b"".join(
        field(channel[col], width)
        for col, width in enumerate((16, 80, 8, 8, 8, 8, 8, 80, 8, 32))
        for channel in channels
    )
    records = []
    for index, onset in enumerate(onsets):
        payload = (
            b""
            if annotation_only
            else struct.pack("<" + "h" * len(samples[index]), *samples[index])
        )
        if other_samples is not None:
            payload += struct.pack(
                "<" + "h" * len(other_samples[index]), *other_samples[index]
            )
        if plus:
            tal = (
                onset.encode()
                + b"\x14\x14\x00"
                + onset.encode()
                + b"\x15"
                + b"0.5\x14"
                + annotation_text.encode("utf-8")
                + b"\x14other\x14\x00"
            )
            assert len(tal) <= 256
            payload += tal.ljust(256, b"\x00")
        if extra_annotation_channel:
            payload += (onset.encode() + b"\x14extra\x14\x00").ljust(256, b"\x00")
        records.append(payload)
    return header + b"".join(records)


def replace_field(payload, start, size, text):
    """Replace one fixed-width synthetic header field."""
    raw = text if isinstance(text, bytes) else str(text).encode("ascii")
    assert len(raw) <= size
    return payload[:start] + raw.ljust(size, b" ") + payload[start + size :]
