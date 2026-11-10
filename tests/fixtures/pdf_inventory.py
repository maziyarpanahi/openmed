"""Hand-built synthetic PDFs; no patient or licensed source assets."""

import zlib

_SYNTHETIC_PAGE_TEXT = "Synthetic Patient Jane Roe MRN 000123"

SENTINEL = "SYNTHETIC-PHI-3824-Jane-Roe-000123"


def _classic_pdf(
    objects: dict[int, str], *, version: str = "1.7", trailer: str = ""
) -> bytes:
    """Build a synthetic PDF with a classic cross-reference table."""
    output = bytearray(f"%PDF-{version}\n%\xe2\xe3\xcf\xd3\n".encode("latin-1"))
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode("latin-1")
    size = max(objects) + 1
    xref = len(output)
    output += f"xref\n0 {size}\n0000000000 65535 f \n".encode("ascii")
    for number in range(1, size):
        if number in offsets:
            output += f"{offsets[number]:010d} 00000 n \n".encode("ascii")
        else:
            output += b"0000000000 65535 f \n"
    output += (
        f"trailer\n<< /Size {size} /Root 1 0 R {trailer}>>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(output)


def _stream_object(dictionary: str, payload: bytes) -> bytes:
    return (
        f"<< {dictionary} /Length {len(payload)} >>\nstream\n".encode("latin-1")
        + payload
        + b"\nendstream"
    )


def _object_stream_pdf(
    compressed: dict[int, str],
    direct: dict[int, str],
    *,
    filter_name: str | None = "FlateDecode",
    corrupt: bool = False,
) -> bytes:
    """Build a synthetic PDF 1.5 file with an object stream and an xref stream."""
    numbers = sorted(compressed)
    bodies = [compressed[number].encode("latin-1") for number in numbers]
    offsets: list[int] = []
    position = 0
    for body in bodies:
        offsets.append(position)
        position += len(body) + 1
    index = " ".join(f"{number} {offset}" for number, offset in zip(numbers, offsets))
    header = (index + "\n").encode("ascii")
    payload = header + b"\n".join(bodies) + b"\n"
    encoded = zlib.compress(payload) if filter_name == "FlateDecode" else payload
    if corrupt:
        encoded = b"\x00not-zlib" + encoded[8:]
    filter_entry = f"/Filter /{filter_name}" if filter_name else ""
    stream_number = max([*compressed, *direct]) + 1
    output = bytearray(b"%PDF-1.5\n")
    for number in sorted(direct):
        output += f"{number} 0 obj\n{direct[number]}\nendobj\n".encode("latin-1")
    output += f"{stream_number} 0 obj\n".encode("ascii")
    output += _stream_object(
        f"/Type /ObjStm /N {len(numbers)} /First {len(header)} {filter_entry}",
        encoded,
    )
    output += b"\nendobj\n"
    xref_number = stream_number + 1
    xref_offset = len(output)
    output += f"{xref_number} 0 obj\n".encode("ascii")
    output += _stream_object(
        f"/Type /XRef /Size {xref_number + 1} /Root 1 0 R /W [1 4 2]",
        b"",
    )
    output += f"\nendobj\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    return bytes(output)


def _single_page(**page_entries: str) -> dict[int, str]:
    entries = " ".join(f"/{key} {value}" for key, value in page_entries.items())
    return {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: f"<< /Type /Page /Parent 2 0 R {entries} /Contents 4 0 R >>",
        4: _stream_object(
            "", f"BT ({_SYNTHETIC_PAGE_TEXT}) Tj ET".encode("latin-1")
        ).decode("latin-1"),
    }


def plain_pdf(**entries: str) -> bytes:
    """Return one text-only page with optional page dictionary entries."""
    return _classic_pdf(_single_page(MediaBox="[0 0 612 792]", **entries))


def category_pdf(category: str, *, compressed: bool = False) -> bytes:
    """Return one category with sentinels in every private payload."""
    objects = _single_page(MediaBox="[0 0 612 792]")
    catalog = "/Type /Catalog /Pages 2 0 R"
    if category == "annotations":
        objects[3] = objects[3].replace("/Contents", "/Annots [5 0 R 6 0 R] /Contents")
        objects[5] = f"<< /Subtype /Text /Rect [1 2 3 4] /Contents ({SENTINEL}) >>"
        objects[6] = f"<< /Type /Annot /Subtype /{SENTINEL} /Rect [1 2 3 4] >>"
    elif category == "forms":
        catalog += " /AcroForm 5 0 R"
        objects[5] = "<< /Fields [6 0 R] >>"
        objects[6] = f"<< /FT /Tx /T ({SENTINEL}) /V ({SENTINEL}) >>"
    elif category == "attachments":
        catalog += " /Names << /EmbeddedFiles 5 0 R >>"
        objects[5] = f"<< /Names [({SENTINEL}.txt) 6 0 R] >>"
        objects[6] = f"<< /Type /Filespec /F ({SENTINEL}.txt) /EF << /F 7 0 R >> >>"
        objects[7] = _stream_object("/Type /EmbeddedFile", SENTINEL.encode()).decode()
    elif category == "xfa":
        catalog += (
            " /AcroForm << /Fields [] /XFA [(template) 5 0 R (datasets) 6 0 R] >>"
        )
        objects[5] = objects[6] = _stream_object("", SENTINEL.encode()).decode()
    elif category == "optional_content":
        catalog += " /OCProperties << /OCGs [5 0 R] >>"
        objects[5] = f"<< /Type /OCG /Name ({SENTINEL}) >>"
    elif category == "javascript":
        catalog += " /Names << /JavaScript << /Names [(private) 5 0 R] >> >>"
        objects[5] = f"<< /S /JavaScript /JS ({SENTINEL}) >>"
    elif category == "open_action":
        catalog += " /OpenAction 5 0 R"
        objects[5] = "<< /S /GoTo /D [3 0 R /Fit] >>"
    elif category == "launch_action":
        objects[3] = objects[3].replace("/Contents", "/AA << /O 5 0 R >> /Contents")
        objects[5] = f"<< /S /Launch /F ({SENTINEL}) >>"
    else:
        raise ValueError("synthetic_category_invalid")
    objects[1] = f"<< {catalog} >>"
    if compressed:
        # Stream objects stay direct, as required by the PDF object stream format.
        direct = {n: v for n, v in objects.items() if "\nstream\n" in v}
        embedded = {n: v for n, v in objects.items() if n not in direct}
        return _object_stream_pdf(embedded, direct)
    return _classic_pdf(objects)


def incremental_pdf(*, remove_field: bool = False) -> bytes:
    """Append a real update with Prev and a changed/deleted field declaration."""
    original = category_pdf("forms")
    previous = int(original.rsplit(b"startxref\n", 1)[1].splitlines()[0])
    position = len(original)
    field = (
        "null"
        if remove_field
        else f"<< /FT /Tx /T ({SENTINEL}) /V (changed-{SENTINEL}) >>"
    )
    update = f"6 0 obj\n{field}\nendobj\n".encode()
    xref = position + len(update)
    update += (
        f"xref\n6 1\n{position:010d} 00000 n \ntrailer\n"
        f"<< /Size 7 /Root 1 0 R /Prev {previous} >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return original + update
