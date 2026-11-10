"""Fixed synthetic provider outputs and local screening; no ASR or patient audio."""

import hashlib

from openmed.multimodal.audio_windows import plan_audio_windows
from openmed.multimodal.streaming.privacy import (
    PhiRedactionSpan,
    PhiResolution,
    PhiScreenDeclaration,
    PhiScreenResult,
    SegmentPhiDecision,
)
from openmed.multimodal.streaming.speech.providers import (
    PartialHypothesis,
    ProviderProvenance,
    TokenTime,
)
from openmed.multimodal.streaming.transcript.window_stitching import (
    AlignedWindow,
    reconcile_windows,
)

SESSION = "1" * 32
SOURCE = hashlib.sha256(
    b"synthetic fixed ASR window source; no clinical recording"
).hexdigest()
PRIVATE = "SYNTHETIC PATIENT"
PROVIDER = ProviderProvenance("synthetic-overlap", "v1", "sha256:" + "a" * 64)
PLAN = plan_audio_windows(3000, window_ms=2000, overlap_ms=1000)


def window(index, words, *, final=True, complete=True):
    """Build explicitly timed word/character alignments from fixed synthetic tokens."""
    text = " ".join(word for word, _, _ in words)
    offsets = []
    position = 0
    for word, _, _ in words:
        offsets.append((position, position + len(word)))
        position += len(word) + 1
    return AlignedWindow(
        window_index=index,
        source_digest=SOURCE,
        provider=PROVIDER,
        hypothesis=PartialHypothesis(
            revision=0,
            text=text,
            is_final=final,
            token_times=tuple(
                TokenTime(
                    token_index=i,
                    start_ms=start,
                    end_ms=end,
                    confidence_ppm=900000,
                    token=word,
                )
                for i, (word, start, end) in enumerate(words)
            ),
        ),
        character_offsets=tuple(offsets),
        coverage_complete=complete,
    )


def negation_windows():
    """A negation before the seam and a repeated medication inside the overlap."""
    return (
        window(
            0,
            (
                ("No", 100, 300),
                ("allergy", 1100, 1250),
                ("to", 1300, 1350),
                ("penicillin", 1500, 1800),
            ),
        ),
        window(
            1,
            (
                ("allergy", 1100, 1250),
                ("to", 1300, 1350),
                ("penicillin", 1500, 1800),
                ("reported", 2200, 2500),
            ),
        ),
    )


def reconcile(windows=None, *, plan=PLAN):
    """Use fixed synthetic source and provider metadata for replay controls."""
    return reconcile_windows(
        plan,
        negation_windows() if windows is None else windows,
        session_ref=SESSION,
        source_digest=SOURCE,
        provider=PROVIDER,
    )


class SyntheticScreener:
    """Exact known marker control; it is not a clinical identifier detector."""

    declaration = PhiScreenDeclaration(artifact_digest="b" * 64)

    def __init__(self, resolution=PhiResolution.RESOLVED):
        self.resolution = resolution
        self.calls = 0

    def screen(self, request):
        """Resolve only a known synthetic marker against the bound whole transcript."""
        self.calls += 1
        decisions = []
        for segment in request.segments:
            start = segment.text.find(PRIVATE)
            spans = (
                (PhiRedactionSpan(start, start + len(PRIVATE)),) if start >= 0 else ()
            )
            decisions.append(
                SegmentPhiDecision(
                    segment_ref=segment.segment_ref,
                    revision=segment.revision,
                    resolution=self.resolution,
                    redactions=spans,
                )
            )
        return PhiScreenResult(
            request_digest=request.request_digest, decisions=tuple(decisions)
        )
