#!/usr/bin/env python3
"""Decide champion-vs-challenger promotion from committed gate reports.

This thin CLI wraps :func:`openmed.eval.champion_challenger.main` so a release
operator can run the promotion decision that gates the STABLE transition from
committed inputs only -- the champion ``-vN`` gate report and the challenger
candidate gate report. It performs no live model or Hugging Face call: the
``PROMOTE`` / ``HOLD`` / ``REJECT`` verdict is derived purely from the two
reports, and only ``PROMOTE`` advances the committed champion pointer.
"""

from __future__ import annotations

from openmed.eval.champion_challenger import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
