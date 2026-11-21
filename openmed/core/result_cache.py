"""
In-process result cache for OpenMed calls.

This cache is opt-in (not included by default), and only saves data to memory. A customized implementation of LRU Cache is used.
It never writes keys or results to disk due to PII/PHI sensitivity.
"""

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from uuid import uuid4

RESULT_CACHE = None
_CACHE_LOCK = RLock()
_MODEL_GENERATION = 0


@dataclass(frozen=True)
class _SubjectKey:
    subject: str
    key: object


def _scoped_key(key, fingerprint):
    if fingerprint is None:
        return key
    if not isinstance(fingerprint, str) or not re.fullmatch(
        r"(?:hmac-)?sha256:[0-9a-f]{64}", fingerprint
    ):
        raise ValueError("invalid subject scope fingerprint")
    return _SubjectKey(fingerprint, key)


class ResultCache:
    """In-memory LRU and optional participant in coordinated subject withdrawal."""

    def __init__(self, max_entries):
        self.max_entries = max_entries
        self.cache = OrderedDict()
        self._lock = RLock()
        self._instance = uuid4().hex
        self._revision = 0
        self._revisions = {}
        self._withdrawal_epoch = 0

    def __len__(self):
        with self._lock:
            return len(self.cache)

    @property
    def withdrawal_epoch(self):
        """Capture the write fence before starting a cacheable inference."""
        with self._lock:
            return self._withdrawal_epoch

    def get(self, key, *, subject_scope_fingerprint=None):
        key = _scoped_key(key, subject_scope_fingerprint)
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def clear(self):
        with self._lock:
            self.cache.clear()
            self._revisions.clear()
            self._withdrawal_epoch += 1

    def set(self, key, result, *, subject_scope_fingerprint=None, expected_epoch=None):
        key = _scoped_key(key, subject_scope_fingerprint)
        if self.max_entries <= 0:
            return None

        with self._lock:
            if expected_epoch is not None and self._withdrawal_epoch != expected_epoch:
                return None
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = result
            if subject_scope_fingerprint is not None:
                self._revision += 1
                self._revisions[key] = self._revision
            if len(self.cache) > self.max_entries:
                evicted, _ = self.cache.popitem(last=False)
                self._revisions.pop(evicted, None)

    def plan(self, subject_ref_digest):
        """Bind the selected entry revisions and count unattributable entries."""
        from openmed.compliance.withdrawal import ParticipantPlan
        from openmed.core.audit import stable_hash

        _scoped_key(None, subject_ref_digest)
        if subject_ref_digest is None:
            raise ValueError("missing subject scope fingerprint")
        with self._lock:
            revisions = sorted(
                revision
                for key, revision in self._revisions.items()
                if key.subject == subject_ref_digest
            )
            return ParticipantPlan(
                count=len(revisions),
                selection_digest=stable_hash(
                    [self._instance, subject_ref_digest, revisions]
                ),
                unattributable_count=len(self.cache) - len(self._revisions),
            )

    def apply(self, subject_ref_digest, plan):
        """Remove the confirmed selection under the cache lock."""
        with self._lock:
            if self.plan(subject_ref_digest) != plan:
                raise ValueError("result cache selection changed")
            for key in [
                key for key in self._revisions if key.subject == subject_ref_digest
            ]:
                del self.cache[key]
                del self._revisions[key]
            self._withdrawal_epoch += 1
            return plan.count

    def verify(self, subject_ref_digest):
        """Re-select the subject independently of the applied count."""
        return self.plan(subject_ref_digest).count == 0


def get_result_cache(max_entries=128):
    """Return the shared cache selected for this call's requested capacity."""
    global RESULT_CACHE

    with _CACHE_LOCK:
        if RESULT_CACHE is None or max_entries != RESULT_CACHE.max_entries:
            RESULT_CACHE = ResultCache(max_entries)
        # Evaluate the return value before another caller can replace the global
        # cache with one configured for a different capacity.
        return RESULT_CACHE


def current_model_generation():
    """Return the counter that changes whenever served model weights may change."""
    with _CACHE_LOCK:
        return _MODEL_GENERATION


def notify_model_generation_change():
    """Invalidate cached results after a model activation, promotion or rollback.

    A model can keep its name while its weights change. Bumping the generation
    makes every later cache key differ from keys built before the change, and the
    shared cache is cleared as a second guard.

    Returns:
        The new model generation.
    """
    global _MODEL_GENERATION

    with _CACHE_LOCK:
        _MODEL_GENERATION += 1
        if RESULT_CACHE is not None:
            RESULT_CACHE.clear()
        return _MODEL_GENERATION


def _resolve_model_revision(params):
    revision = params.get("revision")
    if revision is None:
        pipeline_kwargs = params.get("pipeline_kwargs")
        if isinstance(pipeline_kwargs, dict):
            revision = pipeline_kwargs.get("revision")
    return revision


def freeze_value(value):
    if isinstance(value, dict):
        return (
            "__dict__",
            tuple(sorted((str(k), freeze_value(v)) for k, v in value.items())),
        )

    elif isinstance(value, list):
        return ("__list__", tuple(freeze_value(v) for v in value))

    elif isinstance(value, tuple):
        return ("__tuple__", tuple(freeze_value(v) for v in value))

    elif isinstance(value, set):
        return ("__set__", tuple(sorted(repr(freeze_value(v)) for v in value)))

    elif isinstance(value, (str, int, float, bool, type(None))):
        return value

    return ("__repr__", repr(value))


def make_cache_key(inquiry_type, params):
    normalized = dict(params)
    normalized["text"] = normalized.get(
        "validated_text", normalized.get("text", "")
    ).strip()
    normalized["model_name"] = normalized.get(
        "validated_model",
        normalized.get("model_name", normalized.get("model_id", "")),
    )
    normalized.pop("model_id", None)
    normalized["model_revision"] = _resolve_model_revision(normalized)
    normalized["model_generation"] = current_model_generation()

    for name in (
        "config",
        "loader",
        "sentence_segmenter",
        "cache_results",
        "max_cache_entries",
        "subject_scope_fingerprint",
        "selected_model",
        "validated_text",
        "validated_model",
    ):
        normalized.pop(name, None)

    payload = json.dumps(
        freeze_value(normalized), sort_keys=True, separators=(",", ":")
    )
    return f"{inquiry_type}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
