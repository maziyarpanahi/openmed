"""On-device vision contracts for multimodal workflows.

This package groups the vision-specific contracts that build on the shared
:mod:`openmed.multimodal` ingestion surface. The sub-packages stay stdlib-only,
so importing :mod:`openmed.multimodal.vision` never pulls an optional extra.
"""

from __future__ import annotations

from . import runtime

__all__ = ["runtime"]
