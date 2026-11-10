"""De-identify a local ODK, CommCare, or KoBoToolbox form export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from openmed.multimodal import redact_chw_form


def build_parser() -> argparse.ArgumentParser:
    """Build the CHW form example command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local JSON, CSV, or TSV export")
    parser.add_argument("--platform", choices=("odk", "commcare", "kobo"))
    parser.add_argument("--output", type=Path, help="Write the redacted export here")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Write the PHI-free policy/count manifest as JSON",
    )
    parser.add_argument(
        "--drop-metadata",
        action="store_true",
        help="Drop platform metadata instead of hashing it",
    )
    parser.add_argument(
        "--drop-geopoints",
        action="store_true",
        help="Drop geopoint/geotrace fields instead of generalizing them",
    )
    return parser


def run(
    source: Path,
    *,
    platform: str | None = None,
    drop_metadata: bool = False,
    drop_geopoints: bool = False,
) -> tuple[str, str]:
    """Return the redacted export and serialized PHI-free manifest."""
    policy = {
        "metadata_action": "drop" if drop_metadata else "hash",
        "geopoint_action": "drop" if drop_geopoints else "generalize_geo",
    }
    result = redact_chw_form(source, platform=platform, policy=policy)
    manifest = json.dumps(result.manifest, indent=2, sort_keys=True) + "\n"
    return result.text, manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local-file example without any platform API calls."""
    args = build_parser().parse_args(argv)
    redacted, manifest = run(
        args.source,
        platform=args.platform,
        drop_metadata=args.drop_metadata,
        drop_geopoints=args.drop_geopoints,
    )

    if args.output:
        args.output.write_text(redacted, encoding="utf-8")
    else:
        print(redacted, end="")
    if args.manifest:
        args.manifest.write_text(manifest, encoding="utf-8")
    else:
        print(manifest, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
