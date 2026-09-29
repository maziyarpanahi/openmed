"""Streaming modality surfaces for local pipelines.

The package groups bounded streaming contracts by modality. Every subpackage is
import-safe: importing it declares types and helper functions without loading a
model, opening a socket, or reading audio.
"""

from __future__ import annotations

from . import speech

__all__ = ["speech"]
