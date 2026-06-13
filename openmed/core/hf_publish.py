"""Publish a generated model card to its HuggingFace repo.

Renders the card with :func:`openmed.core.model_card.render_model_card` and uploads it
as ``README.md``. The ``api`` parameter is injectable so the upload path is unit
testable without a network call.
"""

from __future__ import annotations

from typing import Any

from .model_card import render_model_card

try:  # pragma: no cover - import guard mirrors scripts/manifest/generate_manifest.py
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover
    HfApi = None  # type: ignore[assignment]

_DEFAULT_COMMIT_MESSAGE = "Update auto-generated model card"


def publish_model_card(
    row: dict[str, Any],
    *,
    token: str | None = None,
    api: Any = None,
    commit_message: str | None = None,
) -> Any:
    """Render the card for ``row`` and upload it as ``README.md`` to its HF repo.

    Returns whatever ``HfApi.upload_file`` returns (a commit URL / ``CommitInfo``).
    """
    card = render_model_card(row)
    if api is None:
        if HfApi is None:
            raise RuntimeError(
                "huggingface_hub is required to publish model cards; install it or "
                "pass an explicit `api`."
            )
        api = HfApi(token=token)
    return api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=row["repo_id"],
        repo_type="model",
        commit_message=commit_message or _DEFAULT_COMMIT_MESSAGE,
    )
