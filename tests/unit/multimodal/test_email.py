"""Synthetic offline tests for EML and optional MSG ingestion."""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from email import policy as email_policy
from email.message import EmailMessage
from email.parser import BytesParser
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import openmed.multimodal.base as base
import openmed.multimodal.email as email_module
from openmed.multimodal import (
    ExtractedDocument,
    extract_email,
    redact_document,
    redact_email,
)
from openmed.multimodal.exceptions import (
    MissingDependencyError,
    UnsupportedDocumentError,
)
from openmed.multimodal.ocr import OcrResult, OcrWord

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_phi.eml"
RAW_IDENTIFIERS = (
    "Alice Patient",
    "alice.patient@example.test",
    "alice.reply@example.test",
    "bob.care@clinic.test",
    "12345678",
    "415-555-1212",
    "999-99-9999",
)


class NamedBytesIO(BytesIO):
    """Bytes buffer exposing a safe filename for extension dispatch."""

    def __init__(self, payload: bytes, name: str) -> None:
        super().__init__(payload)
        self.name = name


def synthetic_detector(text: str, *, lang: str | None = None):
    """Detect synthetic names while the production safety sweep catches IDs."""
    assert lang in {None, "en"}
    entities = []
    patterns = (
        (r"Alice(?: Patient)?", "PERSON"),
        (r"Bob Care", "PERSON"),
        (r"Records Team", "ORGANIZATION"),
        (r"Hidden Reviewer", "PERSON"),
        (r"\b\d{8}\b", "MEDICAL_RECORD_NUMBER"),
    )
    for pattern, label in patterns:
        entities.extend(
            {
                "start": match.start(),
                "end": match.end(),
                "label": label,
                "confidence": 1.0,
            }
            for match in re.finditer(pattern, text)
        )
    return {"entities": entities}


def _parsed(payload: bytes):
    return BytesParser(policy=email_policy.default).parsebytes(payload)


def _text_parts(message) -> list[str]:
    return [
        str(part.get_content())
        for part in message.walk()
        if not part.is_multipart()
        and part.get_content_disposition() != "attachment"
        and part.get_content_type() in {"text/plain", "text/html"}
    ]


def _attachment(message):
    return next(
        part
        for part in message.walk()
        if part.get_content_disposition() == "attachment"
    )


def _message_with_attachment(
    payload: bytes,
    *,
    filename: str,
    maintype: str,
    subtype: str,
) -> bytes:
    message = EmailMessage()
    message["From"] = "sender@example.test"
    message["To"] = "recipient@example.test"
    message["Subject"] = "Synthetic attachment"
    message.set_content("Synthetic attachment body")
    message.add_attachment(
        payload,
        maintype=maintype,
        subtype=subtype,
        filename=filename,
    )
    return message.as_bytes()


def test_extract_email_decodes_headers_bodies_and_html_offset_map():
    document = extract_email(FIXTURE)

    assert "From: Alice Patient <alice.patient@example.test>" in document.text
    assert "Subject: Alice Patient MRI report - MRN 12345678" in document.text
    assert "Hello Alice Patient" in document.text
    assert "<strong>" not in document.text
    assert "999-99-9999" not in document.text
    assert document.metadata["format"] == "eml"
    assert document.metadata["body_part_count"] == 2
    assert document.metadata["attachment_count"] == 1

    html_spans = [
        span
        for span in document.spans
        if span.metadata.get("content_type") == "text/html"
    ]
    assert html_spans
    assert all(span.start < span.end <= len(document.text) for span in document.spans)
    assert all(
        span.metadata["html_source_start"] < span.metadata["html_source_end"]
        for span in html_spans
    )
    assert document.location_at(document.text.index("Alice Patient")) is not None


def test_malformed_address_header_is_extracted_and_redacted_without_crashing(
    monkeypatch,
):
    payload = b"From: Synthetic Clinic <clinic@"

    document = extract_email(payload)
    assert document.text == "From: Synthetic Clinic <clinic@"
    assert document.metadata["header_count"] == 1

    monkeypatch.setattr(
        email_module._TextProcessor,
        "redact",
        lambda _self, text: SimpleNamespace(text=text),
    )
    result = redact_email(payload, models=lambda text, **_: text)
    message = _parsed(result.email_bytes)

    assert str(message["From"]) == "redacted-address@openmed.invalid"
    assert result.header_redaction_count == 1
    assert b"Synthetic Clinic <clinic@" not in result.email_bytes


def test_redact_email_redacts_headers_plain_html_and_attachment_metadata(
    tmp_path: Path,
    monkeypatch,
):
    clean_pdf = b"%PDF-1.4\n% clean synthetic replacement\n%%EOF\n"
    dispatched: list[str] = []

    def fake_redact_document(source, *, policy=None, models=None, lang=None):
        dispatched.append(source.name)
        assert source.name == "attachment-0001.pdf"
        assert source.read(5) == b"%PDF-"
        assert callable(models["detector"])
        assert lang == "en"
        policy["output_path"].write(clean_pdf)
        return ExtractedDocument(
            text="Patient Alice Patient MRN 12345678",
            metadata={"format": "pdf", "detected_span_count": 2},
        )

    monkeypatch.setattr(email_module, "redact_document", fake_redact_document)
    output = tmp_path / "clean.eml"
    result = redact_email(
        FIXTURE,
        output_path=output,
        models={"detector": synthetic_detector},
        lang="en",
    )

    assert output.read_bytes() == result.email_bytes
    assert dispatched == ["attachment-0001.pdf"]
    assert result.header_redaction_count >= 8
    assert result.body_redaction_count == 2
    assert len(result.attachments) == 1
    assert result.attachments[0].handler_format == "pdf"
    assert result.attachments[0].detected_span_count == 2

    message = _parsed(result.email_bytes)
    serialized = result.email_bytes.decode("utf-8", errors="replace")
    visible_text = "\n".join(_text_parts(message))
    for identifier in RAW_IDENTIFIERS:
        assert identifier not in serialized
        assert identifier not in result.document.text
    assert "Authentication-Results" not in message
    assert "X-Alice-Patient-ID" not in message
    assert "Date" not in message
    assert "synthetic-outer-boundary" not in serialized
    assert "synthetic-alternative-boundary" not in serialized
    assert message.preamble is None
    assert message.epilogue in {None, ""}
    assert "const ssn" not in visible_text
    assert "mailto:alice.patient@example.test" not in visible_text
    assert "<strong>[PERSON]</strong>" in visible_text
    assert 'src="cid:attachment-0001@openmed.invalid"' in visible_text
    assert "<script></script>" in visible_text
    assert "<!DOCTYPE html><html>" in visible_text
    assert "<alice-patient" not in visible_text
    assert "data-private" not in visible_text
    assert "javascript:" not in visible_text
    assert "tracker.example.test" not in visible_text

    attachment = _attachment(message)
    assert attachment.get_filename() == "attachment-0001.pdf"
    assert attachment["Content-ID"] == "<attachment-0001@openmed.invalid>"
    assert attachment.get("Content-Description") is None
    assert attachment.get("X-Attachment-Patient") is None
    assert attachment.get_payload(decode=True) == clean_pdf


def test_attached_pdf_is_redacted_through_real_pdf_handler(monkeypatch):
    pytest.importorskip("pdfplumber")
    pytest.importorskip("PIL.Image")
    monkeypatch.setattr(base, "_missing_multimodal_dependencies", lambda: [])

    result = redact_email(
        FIXTURE,
        models={"detector": synthetic_detector},
        lang="en",
    )

    attachment = _attachment(_parsed(result.email_bytes))
    redacted_pdf = attachment.get_payload(decode=True)
    assert redacted_pdf.startswith(b"%PDF")
    assert result.attachments[0].handler_format == "pdf"
    assert result.attachments[0].detected_span_count >= 2

    pdfplumber = pytest.importorskip("pdfplumber")
    with pdfplumber.open(BytesIO(redacted_pdf)) as pdf:
        assert len(pdf.pages) == 1
        assert not (pdf.pages[0].extract_text() or "").strip()


def test_attached_docx_is_redacted_and_metadata_scrubbed(monkeypatch):
    docx = pytest.importorskip("docx")
    monkeypatch.setattr(base, "_missing_multimodal_dependencies", lambda: [])
    source = BytesIO()
    document = docx.Document()
    document.core_properties.author = "Alice Patient"
    document.core_properties.title = "Alice Patient private referral"
    document.add_paragraph("Patient Alice Patient")
    document.save(source)
    eml = _message_with_attachment(
        source.getvalue(),
        filename="Alice Patient referral.docx",
        maintype="application",
        subtype=("vnd.openxmlformats-officedocument.wordprocessingml.document"),
    )

    result = redact_email(
        eml,
        models={"detector": synthetic_detector},
        lang="en",
    )

    attachment = _attachment(_parsed(result.email_bytes))
    clean_docx = attachment.get_payload(decode=True)
    assert attachment.get_filename() == "attachment-0001.docx"
    redacted = docx.Document(BytesIO(clean_docx))
    assert "[PERSON]" in "\n".join(item.text for item in redacted.paragraphs)
    assert "Alice Patient" not in "\n".join(item.text for item in redacted.paragraphs)
    with zipfile.ZipFile(BytesIO(clean_docx)) as archive:
        assert not [
            member
            for member in archive.namelist()
            if b"Alice Patient" in archive.read(member)
        ]


def test_attached_image_is_pixel_redacted_and_metadata_free(monkeypatch):
    image_module = pytest.importorskip("PIL.Image")
    image_draw = pytest.importorskip("PIL.ImageDraw")
    png_plugin = pytest.importorskip("PIL.PngImagePlugin")
    monkeypatch.setattr(base, "_missing_multimodal_dependencies", lambda: [])

    image = image_module.new("RGB", (180, 48), "white")
    image_draw.Draw(image).text((8, 14), "Alice Patient", fill="black")
    metadata = png_plugin.PngInfo()
    metadata.add_text("comment", "Alice Patient")
    source = BytesIO()
    image.save(source, format="PNG", pnginfo=metadata)

    class SyntheticOcr:
        name = "synthetic-email"

        def recognize(self, image, *, languages=None):
            return OcrResult(
                words=(
                    OcrWord(
                        "Alice",
                        (8.0, 12.0, 48.0, 32.0),
                        1.0,
                        page=0,
                    ),
                    OcrWord(
                        "Patient",
                        (50.0, 12.0, 106.0, 32.0),
                        1.0,
                        page=0,
                    ),
                ),
                metadata={"engine": self.name},
            )

    eml = _message_with_attachment(
        source.getvalue(),
        filename="Alice Patient scan.png",
        maintype="image",
        subtype="png",
    )
    result = redact_email(
        eml,
        models={
            "detector": synthetic_detector,
            "ocr_engine": SyntheticOcr(),
            "verify": False,
        },
        lang="en",
    )

    attachment = _attachment(_parsed(result.email_bytes))
    clean_png = attachment.get_payload(decode=True)
    assert attachment.get_filename() == "attachment-0001.png"
    with image_module.open(BytesIO(clean_png)) as redacted:
        assert "comment" not in redacted.info
        assert redacted.convert("RGB").getpixel((20, 20)) == (0, 0, 0)


def test_attached_eml_remains_a_readable_redacted_message():
    nested = EmailMessage()
    nested["From"] = "Alice Patient <alice.patient@example.test>"
    nested["To"] = "recipient@example.test"
    nested["Subject"] = "Alice Patient MRN 12345678"
    nested.set_content("Alice Patient MRN 12345678")
    outer = EmailMessage()
    outer["From"] = "sender@example.test"
    outer["To"] = "recipient@example.test"
    outer.set_content("Synthetic outer message")
    outer.add_attachment(nested)

    result = redact_email(
        outer.as_bytes(),
        models={"detector": synthetic_detector},
        lang="en",
    )

    attachment = _attachment(_parsed(result.email_bytes))
    assert attachment.get_filename() == "attachment-0001.eml"
    nested_messages = attachment.get_payload()
    assert isinstance(nested_messages, list)
    assert len(nested_messages) == 1
    nested_clean = nested_messages[0].as_bytes(policy=email_policy.SMTP)
    assert b"Alice Patient" not in nested_clean
    assert b"12345678" not in nested_clean
    assert b"[PERSON]" in nested_clean
    assert b"[MEDICAL_RECORD_NUMBER]" in nested_clean


def test_supplied_detector_still_gets_deterministic_safety_sweep():
    message = EmailMessage()
    message["From"] = "sender@example.test"
    message["To"] = "recipient@example.test"
    message["Subject"] = "MRN 12345678"
    message.set_content("Call 415-555-1212 about MRN 12345678")

    result = redact_email(
        message.as_bytes(),
        models={"detector": lambda text, **kwargs: {"entities": []}},
    )

    serialized = result.email_bytes.decode("utf-8", errors="replace")
    assert "12345678" not in serialized
    assert "415-555-1212" not in serialized
    assert "[MEDICAL_RECORD_NUMBER]" in serialized
    assert "[PHONE_NUMBER]" in serialized


def test_writable_output_stream_is_truncated_before_clean_eml_is_written():
    message = EmailMessage()
    message["From"] = "Alice Patient <alice.patient@example.test>"
    message["To"] = "recipient@example.test"
    message.set_content("Alice Patient MRN 12345678")
    output = NamedBytesIO(b"Alice Patient stale bytes" * 1000, "clean.eml")

    result = redact_email(
        message.as_bytes(),
        output_path=output,
        models={"detector": synthetic_detector},
    )

    assert output.getvalue() == result.email_bytes
    assert b"stale bytes" not in output.getvalue()


def test_unsupported_attachment_fails_closed_without_echoing_filename():
    message = EmailMessage()
    message["From"] = "sender@example.test"
    message["To"] = "recipient@example.test"
    message.set_content("Synthetic note")
    message.add_attachment(
        b"raw payload",
        maintype="text",
        subtype="csv",
        filename="Alice Patient private.csv",
    )

    with pytest.raises(UnsupportedDocumentError) as excinfo:
        redact_email(
            message.as_bytes(),
            models={"detector": synthetic_detector},
        )

    assert "no supported clean-output handler" in str(excinfo.value)
    assert "Alice Patient" not in str(excinfo.value)


def test_named_memory_stream_dispatches_eml_without_multimodal_extra(monkeypatch):
    monkeypatch.setattr(
        base,
        "_missing_multimodal_dependencies",
        lambda: ["pdfplumber"],
    )
    source = NamedBytesIO(FIXTURE.read_bytes(), "synthetic.eml")

    document = redact_document(source)

    assert document.metadata["format"] == "eml"
    assert "Alice Patient" in document.text


def test_missing_extract_msg_dependency_has_actionable_extra(monkeypatch):
    monkeypatch.setattr(email_module.importlib.util, "find_spec", lambda name: None)
    source = NamedBytesIO(b"synthetic-msg", "message.msg")

    with pytest.raises(MissingDependencyError) as excinfo:
        extract_email(source)

    message = str(excinfo.value)
    assert "extract-msg" in message
    assert "openmed[email-msg-gpl]" in message


def test_msg_bridge_is_isolated_and_normalizes_to_eml(monkeypatch):
    captured: dict[str, object] = {}
    eml = EmailMessage()
    eml["From"] = "sender@example.test"
    eml["To"] = "recipient@example.test"
    eml["Subject"] = "Synthetic MSG"
    eml.set_content("offline body")

    monkeypatch.setattr(
        email_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(name=name),
    )

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=eml.as_bytes(), stderr=b"")

    monkeypatch.setattr(email_module.subprocess, "run", fake_run)
    source = NamedBytesIO(b"synthetic-msg-bytes", "message.msg")

    document = extract_email(source)

    args = captured["args"]
    assert args[:3] == [sys.executable, "-I", "-c"]
    compile(args[3], "<openmed-msg-bridge>", "exec")
    assert "BytesIO" in args[3]
    assert captured["input"] == b"synthetic-msg-bytes"
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.PIPE
    assert captured["timeout"] == email_module._MSG_BRIDGE_TIMEOUT_SECONDS
    assert b"synthetic-msg-bytes" not in " ".join(args).encode()
    assert document.metadata["format"] == "msg"
    assert "offline body" in document.text
