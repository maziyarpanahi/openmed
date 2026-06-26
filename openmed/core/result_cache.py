"""
In-process result cache for OpenMed calls.

This cache is opt-in (not included by default), and only saves data to memory. A customized implementation of LRU Cache is used.
It never writes keys or results to disk due to PII/PHI sensitivity.
"""

import hashlib
import json
from collections import OrderedDict
from threading import RLock

RESULT_CACHE = None
_CACHE_LOCK = RLock()


class ResultCache:
    def __init__(self, max_entries):
        self.max_entries = max_entries
        self.cache = OrderedDict()
        self._lock = RLock()

    def get(self, key):
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key, result):
        if self.max_entries <= 0:
            return None

        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = result
            if len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)


def get_result_cache(max_entries=128):
    global RESULT_CACHE

    with _CACHE_LOCK:
        if RESULT_CACHE is None or max_entries != RESULT_CACHE.max_entries:
            RESULT_CACHE = ResultCache(max_entries)

    return RESULT_CACHE


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

    for name in (
        "config",
        "loader",
        "sentence_segmenter",
        "cache_results",
        "max_cache_entries",
        "selected_model",
        "validated_text",
        "validated_model",
    ):
        normalized.pop(name, None)

    payload = json.dumps(
        freeze_value(normalized), sort_keys=True, separators=(",", ":")
    )
    return f"{inquiry_type}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
