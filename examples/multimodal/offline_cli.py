"""Synthetic injected session example; no inference, capture or model qualification."""

import hashlib
import json

from openmed.cli.multimodal_inference import (
    LocalSessionRequest,
    LocalSessionResult,
    LocalTask,
    SessionRefusal,
)
from openmed.multimodal.provider_result import (
    ProviderResultEnvelope,
    ProviderResultOutcome,
)


class SyntheticSession:
    """Return fixed synthetic content and erase owned references on cleanup."""

    def __init__(self, request: LocalSessionRequest) -> None:
        self.request: LocalSessionRequest | None = request

    def run(self) -> LocalSessionResult:
        """Refuse non-synthetic input and unrecognized synthetic consent."""
        request = self.request
        if request is None:
            raise SessionRefusal("session_failed")
        if request.media != b"synthetic media\n":
            raise SessionRefusal("malformed_media")
        if (
            request.task is not LocalTask.PROCESS_DOCUMENT
            and request.consent_evidence != b"synthetic consent\n"
        ):
            raise SessionRefusal("consent_required")
        content = json.dumps(
            {"text": "Synthetic draft only", "source_offsets": [0, 16]}
        ).encode()
        return LocalSessionResult(
            content,
            ProviderResultEnvelope(
                provider_id="synthetic-cli",
                model_id="synthetic-fixture",
                input_digest=hashlib.sha256(request.media).hexdigest(),
                output_digest=hashlib.sha256(content).hexdigest(),
                outcome=ProviderResultOutcome.SUCCESS,
                duration_ms=0,
                count_metadata={"input_items": 1, "output_items": 1},
            ),
        )

    def cancel(self) -> None:
        """Invalidate owned source references; no work is retained."""
        self.request = None

    def close(self) -> None:
        """Release owned references; this example allocates no provider buffers."""
        self.request = None


class SyntheticService:
    """Example of the command-owned protocol, not an ambient session library."""

    local_only = True
    tasks = frozenset(LocalTask)

    def open(self, request: LocalSessionRequest) -> SyntheticSession:
        """Open synthetic file processing with no clinical side effects."""
        return SyntheticSession(request)


def create_service() -> SyntheticService:
    """Create a fake service for the documented offline CLI walkthrough."""
    return SyntheticService()
