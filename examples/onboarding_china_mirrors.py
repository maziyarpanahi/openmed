#!/usr/bin/env python3
"""Use a Hugging Face mirror to warm a cache, then resolve it offline.

The example is cache-only by default. Set
``OPENMED_EXAMPLE_ALLOW_DOWNLOAD=1`` on a connected machine to pre-download
the model through the configured mirror. A normal run sets the Hugging Face
offline flags and never attempts a network request.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
DEFAULT_MODEL_ID = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
ALLOW_DOWNLOAD_ENV = "OPENMED_EXAMPLE_ALLOW_DOWNLOAD"

SnapshotLoader = Callable[..., str]


@contextmanager
def hub_environment(*, offline: bool) -> Iterator[None]:
    """Temporarily configure the mirror and cache-only environment flags."""
    updates: Mapping[str, str | None] = {
        "HF_ENDPOINT": HF_MIRROR_ENDPOINT,
        "HF_HUB_OFFLINE": "1" if offline else None,
        "TRANSFORMERS_OFFLINE": "1" if offline else None,
    }
    previous = {name: os.environ.get(name) for name in updates}
    for name, value in updates.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _snapshot_loader() -> SnapshotLoader:
    """Import the Hub client only after its environment is configured."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            'Install the Hub client with: pip install "openmed[hf]"'
        ) from exc
    return snapshot_download


def predownload_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    loader: SnapshotLoader | None = None,
) -> Path:
    """Download *model_id* through the mirror into the standard local cache."""
    with hub_environment(offline=False):
        snapshot_path = (loader or _snapshot_loader())(
            repo_id=model_id,
            repo_type="model",
            local_files_only=False,
        )
    return Path(snapshot_path)


def load_cached_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    loader: SnapshotLoader | None = None,
) -> Path:
    """Resolve *model_id* from the local cache without any network access."""
    with hub_environment(offline=True):
        snapshot_path = (loader or _snapshot_loader())(
            repo_id=model_id,
            repo_type="model",
            local_files_only=True,
        )
    return Path(snapshot_path)


def run(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    loader: SnapshotLoader | None = None,
) -> Path | None:
    """Warm the cache when opted in; otherwise perform cache-only resolution."""
    if os.getenv(ALLOW_DOWNLOAD_ENV) == "1":
        path = predownload_model(model_id, loader=loader)
        print(f"Cached {model_id} at {path}")
        return path

    print("Offline cache-only mode (HF_HUB_OFFLINE=1)")
    try:
        path = load_cached_model(model_id, loader=loader)
    except Exception as exc:
        print(f"Model is not available in the local cache: {exc}")
        print(
            "On a connected machine, set "
            f"{ALLOW_DOWNLOAD_ENV}=1 and run this example once."
        )
        return None

    print(f"Resolved cached model at {path}")
    return path


def main() -> None:
    """Run the mirror-to-offline-cache example."""
    run()


if __name__ == "__main__":
    main()
