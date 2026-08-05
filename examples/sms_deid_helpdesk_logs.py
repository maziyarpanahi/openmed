#!/usr/bin/env python3
"""De-identify a RapidPro-style JSON or generic CSV message export."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openmed.multimodal import redact_sms_export

HASH_KEY_ENV = "OPENMED_SMS_CONTACT_HASH_KEY"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redact an SMS helpdesk export with the short_text preset."
    )
    parser.add_argument("input", type=Path, help="Source .json or .csv export")
    parser.add_argument("output", type=Path, help="Destination redacted export")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Maximum source records retained per redaction batch",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = redact_sms_export(
        args.input,
        args.output,
        batch_size=args.batch_size,
        contact_hash_key=os.environ.get(HASH_KEY_ENV),
    )
    print(json.dumps(result.summary.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
