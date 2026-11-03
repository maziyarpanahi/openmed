"""Small valid synthetic assets and fixed local providers for session controls."""

import hashlib
import io
import struct
import wave
import zlib

from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.base import ExtractedDocument, SourceSpan
from openmed.multimodal.runtime_session import AssetHandle, ProviderLease
from openmed.multimodal.streaming.speech.providers import (
    FinalizedSegment,
    PartialHypothesis,
    ProviderProvenance,
    SpeechProviderDeclaration,
)
from openmed.multimodal.vision.providers.base import (
    LocalProviderMetadata,
    RuntimeBackend,
    RuntimeMetadata,
    VLMResult,
)

PRIVATE = "SYNTHETIC-PRIVATE-DO-NOT-LOG"


def png():
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0\x80\x80\x80\x80" * 4))
        + chunk(b"IEND", b"")
    )


def pdf():
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] >>",
    ]
    data = bytearray(b"%PDF-1.7\n")
    offsets = []
    for index, body in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n{body}\nendobj\n".encode())
    xref = len(data)
    data.extend(b"xref\n0 4\n0000000000 65535 f \n")
    for offset in offsets:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(data)


def wav():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0\0" * 1600)
    return buffer.getvalue()


class CountedStream(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.close_calls = 0
        self.read_calls = 0

    def close(self):
        self.close_calls += 1
        super().close()

    def read(self, size=-1):
        self.read_calls += 1
        return super().read(size)


def asset(kind="image", *, data=None, owned=True, source=None, **overrides):
    payload = (
        data
        if data is not None
        else {"image": png, "document": pdf, "audio": wav}[kind]()
    )
    media, fields = {
        "image": ("image/png", {"width": 4, "height": 4}),
        "document": ("application/pdf", {"pages": 1}),
        "audio": ("audio/wav", {"duration_seconds": 0.1}),
    }[kind]
    values = dict(
        asset_id="synthetic-asset",
        media_type=media,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        **fields,
    )
    values.update(overrides)
    return AssetHandle(
        manifest=AssetManifest(**values),
        source=source if source is not None else CountedStream(payload),
        owned=owned,
    )


class FixedProvider:
    metadata = LocalProviderMetadata(
        backend=RuntimeBackend.SYNTHETIC,
        model_digest="a" * 64,
        preprocessing_digest="b" * 64,
    )
    declaration = SpeechProviderDeclaration(
        provenance=ProviderProvenance(
            provider_id="synthetic",
            provider_revision="1",
            model_fingerprint="sha256:" + "a" * 64,
        ),
        languages=("en",),
        sample_rates_hz=(16000,),
    )

    def __init__(self):
        self.calls = []
        self.closed = 0
        self.cancelled = 0
        self.aborted = 0
        self.document = ExtractedDocument(
            PRIVATE, (SourceSpan(0, len(PRIVATE)),), {"format": "pdf"}
        )

    def infer(self, request):
        self.calls.append("image")
        return VLMResult(
            request_digest=request.digest,
            runtime=RuntimeMetadata(
                provider=self.metadata, duration_ms=0, output_tokens=1
            ),
            text=PRIVATE,
        )

    def extract(self, data, *, limits, cancellation):
        self.calls.append("document")
        self.document_limits = limits
        return self.document

    def push(self, chunk, *, cancellation):
        self.calls.append("audio")
        return (PartialHypothesis(revision=chunk.sequence, text=PRIVATE),)

    def finish(self, *, cancellation):
        self.calls.append("finish_audio")
        return (
            FinalizedSegment(segment_index=0, text=PRIVATE, start_ms=0, end_ms=100),
        )

    def close(self):
        self.closed += 1

    def cancel(self):
        self.cancelled += 1

    def abort(self):
        self.aborted += 1

    def lease(self, *, owned=True):
        return ProviderLease(
            provider=self,
            owned=owned,
            release=self.close if owned else None,
            cancel=self.cancel,
        )
