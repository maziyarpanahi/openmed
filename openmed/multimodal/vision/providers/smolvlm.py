"""Bounded offline SmolVLM provider with mandatory input and output screening."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from typing import TYPE_CHECKING

from openmed.multimodal.abstention import (
    AbstentionReason,
    AbstentionRecord,
    AbstentionStage,
)
from openmed.multimodal.capability_matrix import (
    ModalityCapability,
    ProviderCapabilityEnvelope,
)
from openmed.multimodal.generated_text_guard import (
    AssetIdentifierDigest,
    GeneratedTextGuard,
)
from openmed.multimodal.vision.safety.quality_gate import (
    QualityPolicy,
    _decoder_diagnostics,
    assess_image_quality,
)

from .base import (
    LocalProviderMetadata,
    RuntimeBackend,
    RuntimeMetadata,
    VLMContractError,
    VLMRequest,
    VLMResult,
)
from .local_assets import LocalVLMAssets

if TYPE_CHECKING:
    from openmed.mlx.models.smolvlm import SmolVLMRuntime


def _versions() -> dict[str, str]:
    versions = {}
    for name in ("mlx", "mlx-vlm", "transformers", "tokenizers", "Pillow", "numpy"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


# Shared MLX seed/stream and processor mutation must not overlap across instances.
_RUNTIME_LOCK = threading.Lock()


class SmolVLMProvider:
    """One-image MLX adapter; inputs and generated drafts stay in memory.

    Args:
        assets: Local checkpoint and processor digest manifest.
        quality_policy: Explicit task-specific image admission thresholds.
        guard: Qualified local output detector. Missing detectors abstain.

    No default clinical privacy detector or diagnostic qualification is implied.
    Region, video, DICOM and PDF requests require separate reviewed pipelines.
    Runtime failures return a value-free abstention with exact model provenance.
    """

    def __init__(
        self,
        assets: LocalVLMAssets,
        *,
        quality_policy: QualityPolicy,
        guard: GeneratedTextGuard,
    ) -> None:
        if (
            type(assets) is not LocalVLMAssets
            or type(quality_policy) is not QualityPolicy
            or type(guard) is not GeneratedTextGuard
        ):
            raise VLMContractError("local VLM configuration rejected")
        quality_policy.__post_init__()
        self._assets, self._quality, self._guard = assets, quality_policy, guard
        self._runtime: SmolVLMRuntime | None = None
        self._versions = _versions()
        preprocessing_digest = hashlib.sha256(
            json.dumps(
                [assets.preprocessing_digest, self._versions],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self._metadata = LocalProviderMetadata(
            backend=RuntimeBackend.MLX,
            model_digest=assets.model_digest,
            preprocessing_digest=preprocessing_digest,
        )

    @property
    def metadata(self) -> LocalProviderMetadata:
        """Return pinned model and processor identity without model loading."""
        return self._metadata

    @property
    def capabilities(self) -> ProviderCapabilityEnvelope:
        """Declare the supported encoded-image path, independently of qualification."""
        return ProviderCapabilityEnvelope(
            "smolvlm-mlx",
            "1.0.0",
            (
                ModalityCapability(
                    "image", ("single_image", "opaque_rgb", "greedy_text")
                ),
            ),
        )

    def provenance_json(self) -> str:
        """Report model/processor/license digests and actual loaded runtime versions."""
        data = json.loads(self._assets.to_json())
        data.update(
            preprocessing_digest=self.metadata.preprocessing_digest,
            runtime_versions=dict(self._versions),
            quality_policy_digest=self._quality.digest,
            guard_policy_digest=self._guard.policy.digest,
            review_required=True,
            qualification="not_implied_by_loading",
        )
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def infer(
        self,
        request: VLMRequest,
        *,
        identifiers: tuple[AssetIdentifierDigest, ...] = (),
    ) -> VLMResult:
        """Admit geometry/quality, generate locally, and screen before returning text.

        Args:
            request: One bounded image and prompt; no region-selection support.
            identifiers: Optional same-image identifier commitments. Invalid or
                cross-asset evidence abstains without running the model.
        """
        if type(request) is not VLMRequest:
            raise VLMContractError("local VLM request rejected")
        start = time.perf_counter()

        def abstain(reason: AbstentionReason) -> VLMResult:
            return VLMResult(
                request_digest=request.digest,
                runtime=RuntimeMetadata(
                    provider=self.metadata,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    output_tokens=0,
                ),
                abstention=AbstentionRecord(AbstentionStage.INFERENCE, reason),
            )

        if (
            type(identifiers) is not tuple
            or len(identifiers) > 128
            or any(
                type(x) is not AssetIdentifierDigest
                or x.asset_digest != request.image.digest
                for x in identifiers
            )
        ):
            return abstain(AbstentionReason.PHI_UNCERTAINTY)

        if (
            request.regions
            or request.image.width > 1536
            or request.image.height > 1536
            or request.image.width * request.image.height > 1536**2
            or len(request.prompt.text.encode()) > 4096
            or request.prompt.max_output_tokens > 512
        ):
            return abstain(AbstentionReason.RESOURCE_LIMIT)
        if not assess_image_quality(request.image, self._quality).accepted:
            return abstain(AbstentionReason.LOW_QUALITY)
        if not _RUNTIME_LOCK.acquire(blocking=False):
            return abstain(AbstentionReason.PROVIDER_UNAVAILABLE)
        try:
            from PIL import Image

            from openmed.mlx.models.smolvlm import SmolVLMRuntime

            with (
                _decoder_diagnostics(),
                Image.open(BytesIO(request.image.data)) as decoded,
            ):
                # No implicit orientation change or source-coordinate ambiguity.
                if decoded.getexif().get(274, 1) != 1:
                    return abstain(AbstentionReason.LOW_QUALITY)
                with decoded.convert("RGB") as rgb:
                    rgb.info.clear()
                    if self._runtime is None:
                        self._runtime = SmolVLMRuntime(self._assets)
                        if self._runtime.versions != self._versions:
                            self._runtime = None
                            return abstain(AbstentionReason.PROVIDER_UNAVAILABLE)
                    generated = self._runtime.generate(
                        rgb,
                        request.prompt.text,
                        request.prompt.max_output_tokens,
                        request.prompt.seed,
                    )
            if (
                type(generated.text) is not str
                or not generated.text
                or len(generated.text.encode()) > request.prompt.max_output_bytes
                or type(generated.output_tokens) is not int
                or not 1 <= generated.output_tokens <= request.prompt.max_output_tokens
            ):
                return abstain(AbstentionReason.RESOURCE_LIMIT)
            guarded = self._guard.release(
                generated.text,
                asset_digest=request.image.digest,
                identifiers=identifiers,
                provider_id="smolvlm-mlx",
                model_id="smolvlm",
            )
            if guarded.text is None or not guarded.text:
                return abstain(AbstentionReason.PHI_UNCERTAINTY)
            if len(guarded.text.encode()) > request.prompt.max_output_bytes:
                return abstain(AbstentionReason.RESOURCE_LIMIT)
            return VLMResult(
                request_digest=request.digest,
                runtime=RuntimeMetadata(
                    provider=self.metadata,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    output_tokens=generated.output_tokens,
                ),
                text=guarded.text,
            )
        except Exception:
            return abstain(AbstentionReason.PROVIDER_UNAVAILABLE)
        finally:
            _RUNTIME_LOCK.release()
