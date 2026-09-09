"""Download the pinned PHI model while building the agent container."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from openmed.integrations.health_universe_phi import MODEL_ID, MODEL_REVISION


def main() -> int:
    """Download the model snapshot into a plain local directory."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/app/model"))
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()
    snapshot_download(
        repo_id=MODEL_ID,
        revision=args.revision,
        local_dir=str(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
