"""Opt-in, in-memory shadow evaluation of local PII extraction or redaction."""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from openmed.core.errors import BudgetExceededError
from openmed.core.labels import normalize_label
from openmed.eval.shadow_compare import compare_shadow_spans


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int
    canonical_label: str
    # Both sets are constructed inside a single call on the same document.
    doc_id: str = "current"


def _spans(result, operation) -> tuple[_Span, ...]:
    entities = result.entities if operation == "extract_pii" else result.pii_entities
    spans = []
    for entity in entities:
        if (
            type(entity.start) is not int
            or type(entity.end) is not int
            or not 0 <= entity.start < entity.end
        ):
            raise ValueError("invalid_shadow_offsets")
        label = getattr(entity, "canonical_label", None) or entity.label
        spans.append(_Span(entity.start, entity.end, normalize_label(label)))
    return tuple(spans)


def _pipeline(provider, operation):
    return getattr(provider, operation, None) or provider


class ShadowRunner:
    """Compare local candidates and return active pipeline results unchanged.

    Args:
        active: Bound active callable or provider exposing the two operations.
        candidate: Local candidate callable or provider accepting only the input.
            Bind candidate configuration separately from active options.
        timeout: Maximum seconds to wait for a candidate comparison.
    """

    def __init__(self, active, candidate, *, timeout: float = 5.0):
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
            or timeout > threading.TIMEOUT_MAX
        ):
            raise ValueError("invalid_shadow_timeout")
        self._active = active
        self._candidate = candidate
        self._timeout = timeout
        self._slot = threading.BoundedSemaphore(1)
        self._lock = threading.Lock()
        self._events = dict.fromkeys(
            (
                "compared",
                "candidate_errors",
                "timeouts",
                "busy",
                "missed_direct_identifiers",
            ),
            0,
        )
        self._counts = {
            "by_canonical_label": defaultdict(Counter),
            "by_policy_label": defaultdict(Counter),
        }

    def extract_pii(self, text, *args, **kwargs):
        """Run the active extractor and compare its spans with the candidate."""
        return self._run("extract_pii", text, args, kwargs)

    def deidentify(self, text, *args, **kwargs):
        """Return the active redaction result, including its original metadata."""
        return self._run("deidentify", text, args, kwargs)

    def _run(self, operation, text, args, kwargs):
        result = _pipeline(self._active, operation)(text, *args, **kwargs)
        if not self._slot.acquire(blocking=False):
            self._increment("busy")
            return result
        future = Future()
        try:
            active_spans = _spans(result, operation)
            source = bytes(text) if isinstance(text, (bytearray, memoryview)) else text
            worker = threading.Thread(
                target=self._compare,
                args=(future, operation, source, active_spans),
                daemon=True,
            )
            worker.start()
        except Exception:
            self._slot.release()
            self._increment("candidate_errors")
            return result

        try:
            event, comparison = future.result(timeout=self._timeout)
        except TimeoutError:
            self._increment("timeouts")
            return result
        with self._lock:
            self._events[event] += 1
            if comparison is not None:
                self._events["missed_direct_identifiers"] += comparison[
                    "missed_direct_identifiers"
                ]
                for key, groups in self._counts.items():
                    for label, counts in comparison[key].items():
                        groups[label].update(counts)
        return result

    def _compare(self, future, operation, text, active_spans):
        try:
            result = _pipeline(self._candidate, operation)(text)
            candidate_spans = _spans(result, operation)
            del result
            comparison = compare_shadow_spans(active_spans, candidate_spans)
            outcome = ("compared", comparison)
        except (BudgetExceededError, TimeoutError):
            outcome = ("timeouts", None)
        except BaseException:
            # A candidate terminating its thread must not expose its exception
            # via threading.excepthook or change the caller's active response.
            outcome = ("candidate_errors", None)
        finally:
            self._slot.release()
        future.set_result(outcome)

    def _increment(self, name):
        with self._lock:
            self._events[name] += 1

    def snapshot(self) -> dict[str, Any]:
        """Return detached cumulative counts with no row, text or offset values."""
        with self._lock:
            return {
                **self._events,
                **{
                    key: {label: dict(counts) for label, counts in groups.items()}
                    for key, groups in self._counts.items()
                },
            }
