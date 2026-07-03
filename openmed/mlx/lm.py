"""MLX language-model support for OpenMed.

The PII/NER MLX backend uses OpenMed's artifact contract. Causal language
models use the MLX-LM artifact contract so OpenMed can load custom `model_file`
implementations such as Laneformer without bundling model weights.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

LANEFORMER_SOURCE_MODEL = "kogai/laneformer-2b-it"
LANEFORMER_MLX_MODEL = "OpenMed/laneformer-2b-it-q4-mlx"
LANEFORMER_DRAFT_MLX_MODEL = "OpenMed/laneformer-pii-draft-350m-q4-mlx"
DEFAULT_SPECULATIVE_TOKENS = 4

_MLX_LANGUAGE_MODEL_MAP: dict[str, str] = {
    "laneformer-2b-it": LANEFORMER_MLX_MODEL,
    LANEFORMER_SOURCE_MODEL: LANEFORMER_MLX_MODEL,
    LANEFORMER_MLX_MODEL: LANEFORMER_MLX_MODEL,
    "laneformer-pii-draft": LANEFORMER_DRAFT_MLX_MODEL,
    "laneformer-2b-it-draft": LANEFORMER_DRAFT_MLX_MODEL,
    LANEFORMER_DRAFT_MLX_MODEL: LANEFORMER_DRAFT_MLX_MODEL,
}

_MLX_LM_ALLOW_PATTERNS = [
    "README.md",
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "*.py",
]


@dataclass(frozen=True)
class PagedKVCacheStats:
    """Aggregate, PHI-free paged KV-cache accounting."""

    total_pages: int
    resident_pages: int
    free_pages: int
    peak_resident_pages: int
    evictions: int
    page_size_tokens: int
    memory_budget_bytes: int
    bytes_per_page: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable stats payload."""
        return {
            "total_pages": self.total_pages,
            "resident_pages": self.resident_pages,
            "free_pages": self.free_pages,
            "peak_resident_pages": self.peak_resident_pages,
            "evictions": self.evictions,
            "page_size_tokens": self.page_size_tokens,
            "memory_budget_bytes": self.memory_budget_bytes,
            "bytes_per_page": self.bytes_per_page,
        }


@dataclass(frozen=True)
class TokenRange:
    """Half-open token span used for chunked long-note prefill."""

    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the token span length."""
        return max(self.end - self.start, 0)


@dataclass(frozen=True)
class PagedKVCachePlan:
    """Execution plan for a prompt under a fixed paged KV-cache budget."""

    total_tokens: int
    page_size_tokens: int
    total_pages: int
    chunk_size_tokens: int
    resident_window_tokens: int
    memory_budget_bytes: int
    bytes_per_page: int
    chunk_ranges: tuple[TokenRange, ...]
    evictions: int = 0
    recompute_tokens: int = 0
    budget_exceeded: bool = False

    @property
    def peak_resident_pages(self) -> int:
        """Return the maximum page occupancy for this plan."""
        pages_needed = math.ceil(self.total_tokens / self.page_size_tokens)
        return min(max(pages_needed, 0), self.total_pages)

    @property
    def resident_pages(self) -> int:
        """Return resident pages at the end of prefill."""
        return self.peak_resident_pages

    @property
    def exact(self) -> bool:
        """Return whether the whole prompt fits without eviction/recompute."""
        return not self.budget_exceeded and self.recompute_tokens == 0

    @property
    def prefill_step_size(self) -> int:
        """Return the step size to use for chunked prompt prefill."""
        return self.chunk_size_tokens

    def stats(self) -> PagedKVCacheStats:
        """Return aggregate paged-cache stats for metrics."""
        resident_pages = self.resident_pages
        return PagedKVCacheStats(
            total_pages=self.total_pages,
            resident_pages=resident_pages,
            free_pages=max(self.total_pages - resident_pages, 0),
            peak_resident_pages=self.peak_resident_pages,
            evictions=self.evictions,
            page_size_tokens=self.page_size_tokens,
            memory_budget_bytes=self.memory_budget_bytes,
            bytes_per_page=self.bytes_per_page,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plan payload."""
        return {
            "total_tokens": self.total_tokens,
            "page_size_tokens": self.page_size_tokens,
            "total_pages": self.total_pages,
            "chunk_size_tokens": self.chunk_size_tokens,
            "resident_window_tokens": self.resident_window_tokens,
            "memory_budget_bytes": self.memory_budget_bytes,
            "bytes_per_page": self.bytes_per_page,
            "chunk_ranges": [
                {"start": chunk.start, "end": chunk.end} for chunk in self.chunk_ranges
            ],
            "evictions": self.evictions,
            "recompute_tokens": self.recompute_tokens,
            "budget_exceeded": self.budget_exceeded,
            "exact": self.exact,
        }


@dataclass(frozen=True)
class PagedKVCacheConfig:
    """Configuration for block-paged MLX-LM KV-cache planning."""

    memory_budget_bytes: int
    page_size_tokens: int = 128
    chunk_size_tokens: int = 512
    window_size_tokens: int | None = None
    bytes_per_token: int = 65_536

    def __post_init__(self) -> None:
        """Validate page-pool sizing inputs."""
        if self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        if self.page_size_tokens <= 0:
            raise ValueError("page_size_tokens must be positive")
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if self.bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")
        if self.window_size_tokens is not None and self.window_size_tokens <= 0:
            raise ValueError("window_size_tokens must be positive when set")
        if self.memory_budget_bytes < self.bytes_per_page:
            raise ValueError("memory_budget_bytes must hold at least one KV-cache page")

    @property
    def bytes_per_page(self) -> int:
        """Return the memory footprint of one KV-cache page."""
        return self.page_size_tokens * self.bytes_per_token

    @property
    def page_count(self) -> int:
        """Return the fixed page-pool size."""
        return self.memory_budget_bytes // self.bytes_per_page

    @property
    def exact_context_tokens(self) -> int:
        """Return the maximum prompt length that fits without eviction."""
        resident_capacity = self.page_count * self.page_size_tokens
        if self.window_size_tokens is None:
            return resident_capacity
        return min(self.window_size_tokens, resident_capacity)

    def plan(self, total_tokens: int) -> PagedKVCachePlan:
        """Build a chunked prefill and eviction plan for ``total_tokens``."""
        token_count = max(int(total_tokens), 0)
        resident_capacity = self.page_count * self.page_size_tokens
        resident_window = min(
            resident_capacity,
            self.window_size_tokens or resident_capacity,
        )
        pages_needed = math.ceil(token_count / self.page_size_tokens)
        evictions = max(pages_needed - self.page_count, 0)
        budget_exceeded = token_count > resident_window
        recompute_tokens = max(token_count - resident_window, 0)
        return PagedKVCachePlan(
            total_tokens=token_count,
            page_size_tokens=self.page_size_tokens,
            total_pages=self.page_count,
            chunk_size_tokens=self.chunk_size_tokens,
            resident_window_tokens=resident_window,
            memory_budget_bytes=self.memory_budget_bytes,
            bytes_per_page=self.bytes_per_page,
            chunk_ranges=_chunk_token_ranges(token_count, self.chunk_size_tokens),
            evictions=evictions,
            recompute_tokens=recompute_tokens,
            budget_exceeded=budget_exceeded,
        )


class OpenMedPagedKVCache:
    """Pure-Python page table for deterministic KV-cache accounting."""

    def __init__(self, config: PagedKVCacheConfig) -> None:
        """Create a fixed-size page pool from ``config``."""
        self.config = config
        self.page_size_tokens = config.page_size_tokens
        self.total_pages = config.page_count
        self._free_pages = list(range(self.total_pages))
        self._page_table: dict[str, dict[int, int]] = {}
        self._reverse_page_table: dict[int, tuple[str, int]] = {}
        self._pages: dict[int, list[Any | None]] = {
            page_id: [None] * self.page_size_tokens
            for page_id in range(self.total_pages)
        }
        self._lru_pages: OrderedDict[int, None] = OrderedDict()
        self._next_positions: dict[str, int] = {}
        self._evictions = 0
        self._peak_resident_pages = 0

    def append(self, sequence_id: str, key: Any, value: Any) -> int:
        """Append one token KV payload and return its token position."""
        position = self._next_positions.get(str(sequence_id), 0)
        self.store(sequence_id, position, key, value)
        return position

    def store(self, sequence_id: str, position: int, key: Any, value: Any) -> None:
        """Store a token KV payload at ``sequence_id``/``position``."""
        if position < 0:
            raise ValueError("position must be non-negative")

        sequence_key = str(sequence_id)
        virtual_page = position // self.page_size_tokens
        offset = position % self.page_size_tokens
        table = self._page_table.setdefault(sequence_key, {})
        physical_page = table.get(virtual_page)
        if physical_page is None:
            protected_min = max(
                position - self.config.exact_context_tokens + 1,
                0,
            )
            physical_page = self._allocate_page(
                sequence_key,
                virtual_page,
                protected_min_position=protected_min,
            )

        self._pages[physical_page][offset] = (key, value)
        self._next_positions[sequence_key] = max(
            self._next_positions.get(sequence_key, 0),
            position + 1,
        )
        self._touch(physical_page)

    def get(self, sequence_id: str, position: int) -> tuple[Any, Any]:
        """Return a stored token KV payload."""
        if position < 0:
            raise ValueError("position must be non-negative")

        sequence_key = str(sequence_id)
        virtual_page = position // self.page_size_tokens
        offset = position % self.page_size_tokens
        try:
            physical_page = self._page_table[sequence_key][virtual_page]
        except KeyError as exc:
            raise KeyError(
                f"KV-cache page for sequence {sequence_key!r} position "
                f"{position} is not resident"
            ) from exc

        item = self._pages[physical_page][offset]
        if item is None:
            raise KeyError(
                f"KV-cache entry for sequence {sequence_key!r} position "
                f"{position} is not resident"
            )
        self._touch(physical_page)
        return item

    def page_table(self, sequence_id: str) -> dict[int, int]:
        """Return a copy of the virtual-to-physical page table."""
        return dict(self._page_table.get(str(sequence_id), {}))

    def resident_token_positions(self, sequence_id: str) -> tuple[int, ...]:
        """Return sorted resident token positions for one sequence."""
        positions = []
        for virtual_page, physical_page in self._page_table.get(
            str(sequence_id), {}
        ).items():
            page_start = virtual_page * self.page_size_tokens
            for offset, item in enumerate(self._pages[physical_page]):
                if item is not None:
                    positions.append(page_start + offset)
        return tuple(sorted(positions))

    def stats(self) -> PagedKVCacheStats:
        """Return aggregate page-pool occupancy and eviction counts."""
        resident_pages = len(self._reverse_page_table)
        return PagedKVCacheStats(
            total_pages=self.total_pages,
            resident_pages=resident_pages,
            free_pages=len(self._free_pages),
            peak_resident_pages=self._peak_resident_pages,
            evictions=self._evictions,
            page_size_tokens=self.page_size_tokens,
            memory_budget_bytes=self.config.memory_budget_bytes,
            bytes_per_page=self.config.bytes_per_page,
        )

    def clear(self) -> None:
        """Release all resident pages and reset counters."""
        self._free_pages = list(range(self.total_pages))
        self._page_table.clear()
        self._reverse_page_table.clear()
        for page in self._pages.values():
            page[:] = [None] * self.page_size_tokens
        self._lru_pages.clear()
        self._next_positions.clear()
        self._evictions = 0
        self._peak_resident_pages = 0

    def _allocate_page(
        self,
        sequence_id: str,
        virtual_page: int,
        *,
        protected_min_position: int,
    ) -> int:
        if self._free_pages:
            physical_page = self._free_pages.pop(0)
        else:
            physical_page = self._select_eviction_page(
                sequence_id,
                protected_min_position=protected_min_position,
            )
            self._evict_page(physical_page)

        self._page_table.setdefault(sequence_id, {})[virtual_page] = physical_page
        self._reverse_page_table[physical_page] = (sequence_id, virtual_page)
        self._pages[physical_page] = [None] * self.page_size_tokens
        self._touch(physical_page)
        self._peak_resident_pages = max(
            self._peak_resident_pages,
            len(self._reverse_page_table),
        )
        return physical_page

    def _select_eviction_page(
        self,
        sequence_id: str,
        *,
        protected_min_position: int,
    ) -> int:
        for physical_page in self._lru_pages:
            owner = self._reverse_page_table.get(physical_page)
            if owner is None:
                continue
            owner_sequence_id, owner_virtual_page = owner
            page_end = (owner_virtual_page + 1) * self.page_size_tokens
            if owner_sequence_id == sequence_id and page_end <= protected_min_position:
                return physical_page

        try:
            return next(iter(self._lru_pages))
        except StopIteration as exc:
            raise RuntimeError("paged KV-cache has no page to evict") from exc

    def _evict_page(self, physical_page: int) -> None:
        sequence_id, virtual_page = self._reverse_page_table.pop(physical_page)
        table = self._page_table.get(sequence_id)
        if table is not None:
            table.pop(virtual_page, None)
            if not table:
                self._page_table.pop(sequence_id, None)
        self._pages[physical_page] = [None] * self.page_size_tokens
        self._lru_pages.pop(physical_page, None)
        self._evictions += 1

    def _touch(self, physical_page: int) -> None:
        self._lru_pages.pop(physical_page, None)
        self._lru_pages[physical_page] = None


@dataclass
class SpeculativeDecodeMetrics:
    """Aggregate counters for one speculative decode call."""

    requested: bool = False
    enabled: bool = False
    fallback_reason: str | None = None
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    target_batches: int = 0
    draft_steps: int = 0
    rollback_count: int = 0
    speculative_batches: int = 0
    target_sampled_tokens: int = 0
    generated_tokens: int = 0
    elapsed_seconds: float = 0.0
    max_speculative_depth: int = 0

    @property
    def rejected_tokens(self) -> int:
        """Return draft tokens rejected by target verification."""
        return max(self.drafted_tokens - self.accepted_tokens, 0)

    @property
    def acceptance_rate(self) -> float:
        """Return accepted draft tokens divided by proposed draft tokens."""
        if self.drafted_tokens <= 0:
            return 0.0
        return self.accepted_tokens / self.drafted_tokens

    @property
    def average_speculative_depth(self) -> float:
        """Return mean proposed depth per speculative verification batch."""
        if self.speculative_batches <= 0:
            return 0.0
        return self.drafted_tokens / self.speculative_batches

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics without prompt, token, or PHI-bearing fields."""
        return {
            "requested": self.requested,
            "enabled": self.enabled,
            "fallback_reason": self.fallback_reason,
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "acceptance_rate": self.acceptance_rate,
            "target_batches": self.target_batches,
            "draft_steps": self.draft_steps,
            "rollback_count": self.rollback_count,
            "speculative_batches": self.speculative_batches,
            "target_sampled_tokens": self.target_sampled_tokens,
            "generated_tokens": self.generated_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "max_speculative_depth": self.max_speculative_depth,
            "average_speculative_depth": self.average_speculative_depth,
        }


@dataclass
class SpeculativeDecodeResult:
    """Generated text plus non-PHI speculative decode metrics."""

    text: str
    metrics: SpeculativeDecodeMetrics


def resolve_mlx_language_model(model_name: str, config: Any = None) -> str:
    """Resolve an OpenMed/source/local name to an MLX-LM artifact path.

    Args:
        model_name: Registry alias, source model id, OpenMed MLX repo id, or
            local artifact directory.
        config: Optional OpenMed config. If present, ``cache_dir`` is passed to
            Hugging Face snapshot downloads.

    Returns:
        Local directory path containing an MLX-LM-compatible artifact.
    """

    local_path = Path(model_name).expanduser()
    if local_path.is_dir() and (local_path / "config.json").exists():
        return str(local_path)

    from openmed.core.model_registry import OPENMED_MODELS

    if model_name in OPENMED_MODELS:
        model_name = OPENMED_MODELS[model_name].model_id

    repo_id = _MLX_LANGUAGE_MODEL_MAP.get(model_name, model_name)
    if "/" not in repo_id:
        repo_id = f"OpenMed/{repo_id}"

    cache_dir = getattr(config, "cache_dir", None) if config is not None else None
    return _download_mlx_lm_artifact(repo_id, cache_dir=cache_dir)


def resolve_mlx_draft_language_model(
    target_model_name: str,
    *,
    draft_model_name: str | None = None,
    config: Any = None,
) -> str | None:
    """Resolve an MLX draft model artifact for a target language model.

    Explicit draft paths or repo ids are respected. Without an explicit draft,
    the core registry is used so draft weights remain a separate artifact from
    the target model package.
    """

    if draft_model_name:
        return resolve_mlx_language_model(draft_model_name, config=config)

    from openmed.core.model_registry import resolve_draft_model_for

    draft_info = resolve_draft_model_for(target_model_name)
    if draft_info is None:
        return None
    return resolve_mlx_language_model(draft_info.draft_model_id, config=config)


def _download_mlx_lm_artifact(repo_id: str, cache_dir: str | None = None) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required for OpenMed MLX language models. "
            "Install with: pip install openmed[mlx]"
        ) from exc

    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=_MLX_LM_ALLOW_PATTERNS,
    )


def _unwrap_tokenizer(tokenizer: Any) -> Any:
    return getattr(tokenizer, "tokenizer", tokenizer)


def _tokenizer_vocab(tokenizer: Any) -> dict[str, int] | None:
    tokenizer = _unwrap_tokenizer(tokenizer)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, Mapping):
            return {str(token): int(token_id) for token, token_id in vocab.items()}

    vocab = getattr(tokenizer, "vocab", None)
    if isinstance(vocab, Mapping):
        return {str(token): int(token_id) for token, token_id in vocab.items()}
    return None


def _special_token_ids(tokenizer: Any) -> tuple[tuple[str, int], ...]:
    tokenizer = _unwrap_tokenizer(tokenizer)
    ids: list[tuple[str, int]] = []
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            ids.append((name, int(value)))
    return tuple(ids)


def tokenizer_alignment_fingerprint(tokenizer: Any) -> str | None:
    """Return a stable tokenizer alignment fingerprint when vocabulary is visible."""

    vocab = _tokenizer_vocab(tokenizer)
    if vocab is None:
        return None

    digest = hashlib.sha256()
    for token, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        digest.update(str(token_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    for name, token_id in _special_token_ids(tokenizer):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(token_id).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def tokenizers_are_aligned(target_tokenizer: Any, draft_tokenizer: Any) -> bool:
    """Return True when target and draft tokenizers have the same vocabulary."""

    if target_tokenizer is draft_tokenizer:
        return True

    target_fingerprint = tokenizer_alignment_fingerprint(target_tokenizer)
    draft_fingerprint = tokenizer_alignment_fingerprint(draft_tokenizer)
    return (
        target_fingerprint is not None
        and draft_fingerprint is not None
        and target_fingerprint == draft_fingerprint
    )


def _encode_prompt(tokenizer: Any, prompt: str | Sequence[int]) -> list[int]:
    if not isinstance(prompt, str):
        return [int(token_id) for token_id in prompt]

    tokenizer = _unwrap_tokenizer(tokenizer)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            token_ids = encode(prompt, add_special_tokens=False)
        except TypeError:
            token_ids = encode(prompt)
        return [int(token_id) for token_id in token_ids]

    call = getattr(tokenizer, "__call__", None)
    if callable(call):
        encoded = call(prompt)
        input_ids = (
            encoded.get("input_ids")
            if isinstance(encoded, Mapping)
            else getattr(encoded, "input_ids", encoded)
        )
        return [int(token_id) for token_id in input_ids]

    raise ValueError("The loaded tokenizer cannot encode prompts.")


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""

    tokenizer = _unwrap_tokenizer(tokenizer)
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            return str(decode(list(token_ids), skip_special_tokens=True))
        except TypeError:
            return str(decode(list(token_ids)))

    raise ValueError("The loaded tokenizer cannot decode generated tokens.")


def _eos_token_ids(tokenizer: Any) -> set[int]:
    tokenizer = _unwrap_tokenizer(tokenizer)
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, int):
        return {int(value)}
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value if isinstance(item, int)}
    return set()


def _bos_token_id(tokenizer: Any) -> int | None:
    value = getattr(_unwrap_tokenizer(tokenizer), "bos_token_id", None)
    return int(value) if isinstance(value, int) else None


def _as_model_input(token_ids: Sequence[int]) -> Any:
    try:
        import mlx.core as mx
    except ImportError:
        return [[int(token_id) for token_id in token_ids]]

    return mx.array([[int(token_id) for token_id in token_ids]])


def _forward_logits(model: Any, token_ids: Sequence[int]) -> list[list[float]]:
    output = model(_as_model_input(token_ids))
    if isinstance(output, tuple):
        output = output[0]
    if hasattr(output, "logits"):
        output = output.logits
    try:
        import mlx.core as mx

        mx.eval(output)
    except ImportError:
        pass
    except TypeError:
        pass

    values = output.tolist() if hasattr(output, "tolist") else output
    if not isinstance(values, list) or not values:
        raise ValueError("Model did not return logits.")

    # Normalize [batch, seq, vocab] and [seq, vocab] to [seq, vocab].
    if isinstance(values[0], list) and values[0] and isinstance(values[0][0], list):
        values = values[0]
    if values and values[0] and not isinstance(values[0], list):
        values = [values]
    return [[float(item) for item in row] for row in values]


def _last_token_logits(model: Any, token_ids: Sequence[int]) -> list[float]:
    if not token_ids:
        raise ValueError("Cannot decode from an empty prompt without a BOS token.")
    logits = _forward_logits(model, token_ids)
    return logits[len(token_ids) - 1]


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _one_hot(token_id: int, size: int) -> list[float]:
    probs = [0.0] * size
    if 0 <= token_id < size:
        probs[token_id] = 1.0
    return probs


def _apply_top_p(probs: list[float], top_p: float) -> list[float]:
    if top_p >= 1.0:
        return probs
    if top_p <= 0.0:
        return _one_hot(_argmax(probs), len(probs))

    sorted_indices = sorted(
        range(len(probs)),
        key=lambda index: probs[index],
        reverse=True,
    )
    kept: set[int] = set()
    cumulative = 0.0
    for index in sorted_indices:
        kept.add(index)
        cumulative += probs[index]
        if cumulative >= top_p:
            break

    filtered = [prob if index in kept else 0.0 for index, prob in enumerate(probs)]
    total = sum(filtered)
    if total <= 0.0:
        return _one_hot(sorted_indices[0], len(probs))
    return [prob / total for prob in filtered]


def _distribution_from_logits(
    logits: Sequence[float],
    *,
    temp: float,
    top_p: float,
) -> list[float]:
    if temp <= 0.0:
        return _one_hot(_argmax(logits), len(logits))

    scaled = [float(value) / temp for value in logits]
    max_logit = max(scaled)
    exp_values = [math.exp(value - max_logit) for value in scaled]
    total = sum(exp_values)
    if total <= 0.0:
        return _one_hot(_argmax(logits), len(logits))
    probs = [value / total for value in exp_values]
    return _apply_top_p(probs, top_p)


def _sample_from_distribution(probs: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, prob in enumerate(probs):
        cumulative += max(float(prob), 0.0)
        if threshold <= cumulative:
            return index
    return _argmax(probs)


def _correction_distribution(
    target_probs: Sequence[float],
    draft_probs: Sequence[float],
) -> list[float]:
    length = min(len(target_probs), len(draft_probs))
    corrected = [
        max(float(target_probs[index]) - float(draft_probs[index]), 0.0)
        for index in range(length)
    ]
    if len(target_probs) > length:
        corrected.extend(float(value) for value in target_probs[length:])
    total = sum(corrected)
    if total <= 0.0:
        return [float(value) for value in target_probs]
    return [value / total for value in corrected]


def _record_speculative_metrics(
    metrics_sink: Any,
    metrics: SpeculativeDecodeMetrics,
) -> None:
    if metrics_sink is None:
        return
    recorder = getattr(metrics_sink, "record_speculative_decode", None)
    if callable(recorder):
        recorder(metrics.to_dict())


class OpenMedMLXLanguageModel:
    """Small OpenMed wrapper around an MLX-LM causal language model."""

    def __init__(
        self,
        model_name: str = LANEFORMER_MLX_MODEL,
        config: Any = None,
        *,
        draft_model_name: str | None = None,
        metrics: Any = None,
    ):
        """Load an MLX-LM artifact.

        Args:
            model_name: Source model id, OpenMed MLX repo id, registry alias, or
                local artifact directory.
            config: Optional OpenMed config. If present, its ``cache_dir`` is
                passed to Hugging Face snapshot downloads.
            draft_model_name: Optional draft model id or artifact directory to
                load lazily for speculative decoding.
            metrics: Optional metrics registry with ``record_speculative_decode``.
        """

        self.model_name = model_name
        self.config = config
        self.draft_model_name = draft_model_name
        self.metrics = metrics
        self.last_speculative_metrics: SpeculativeDecodeMetrics | None = None
        self.model_path = resolve_mlx_language_model(model_name, config=config)
        self.paged_kv_cache_config = _coerce_paged_kv_cache_config(
            getattr(config, "paged_kv_cache", None) if config is not None else None
        )
        self.last_paged_kv_cache_plan: PagedKVCachePlan | None = None
        self._draft_model: Any | None = None
        self._draft_tokenizer: Any | None = None

        try:
            from mlx_lm import load
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for OpenMed MLX language models. "
                "Install with: pip install openmed[mlx]"
            ) from exc

        self.model, self.tokenizer = load(self.model_path)

    def _load_draft_model(self, draft_model_name: str | None = None) -> bool:
        if self._draft_model is not None and self._draft_tokenizer is not None:
            return True

        draft_path = resolve_mlx_draft_language_model(
            self.model_name,
            draft_model_name=draft_model_name or self.draft_model_name,
            config=self.config,
        )
        if draft_path is None:
            return False

        try:
            from mlx_lm import load
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for OpenMed MLX language models. "
                "Install with: pip install openmed[mlx]"
            ) from exc

        self._draft_model, self._draft_tokenizer = load(draft_path)
        return True

    def format_chat_prompt(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render chat messages with the model tokenizer's chat template.

        Args:
            messages: Chat messages accepted by the tokenizer chat template.
            add_generation_prompt: Whether to append the assistant generation
                marker when rendering the prompt.

        Returns:
            Rendered prompt text.
        """

        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("The loaded tokenizer does not expose a chat template.")
        return tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    def generate(
        self,
        prompt: str | Sequence[int] | None = None,
        *,
        messages: Sequence[Mapping[str, str]] | None = None,
        max_tokens: int = 256,
        temp: float = 0.0,
        top_p: float = 1.0,
        verbose: bool = False,
        speculative: bool | None = None,
        draft_model_name: str | None = None,
        max_speculative_tokens: int = DEFAULT_SPECULATIVE_TOKENS,
        seed: int | None = None,
        return_metrics: bool = False,
        paged_kv_cache: PagedKVCacheConfig | Mapping[str, Any] | bool | None = None,
        metrics: Any = None,
        **kwargs: Any,
    ) -> str | SpeculativeDecodeResult:
        """Generate text with the loaded MLX language model.

        Args:
            prompt: Plain prompt text or token ids. Mutually exclusive with
                ``messages``.
            messages: Chat messages to render with the model chat template.
            max_tokens: Maximum generated token count.
            temp: Sampling temperature forwarded to ``mlx_lm.generate``.
            top_p: Top-p sampling value forwarded to ``mlx_lm.generate``.
            verbose: Whether MLX-LM should print generation details.
            speculative: Enable draft-model speculative decoding. When ``None``,
                speculative decoding is enabled only if a draft model is supplied.
            draft_model_name: Optional draft model id or artifact directory.
            max_speculative_tokens: Maximum draft tokens proposed per target pass.
            seed: Optional Python RNG seed for speculative sampling tests.
            return_metrics: Return ``SpeculativeDecodeResult`` instead of text.
            paged_kv_cache: Optional paged-cache config or ``True`` to use the
                runner's configured default. Plain generation uses it for
                chunked prefill and bounded KV-cache hints where supported.
            metrics: Optional metrics registry for this call.
            **kwargs: Additional keyword arguments forwarded to
                ``mlx_lm.generate``.

        Returns:
            Generated text, or text plus speculative metrics when requested.
        """

        if messages is not None:
            if prompt is not None:
                raise ValueError("Pass either prompt or messages, not both.")
            prompt = self.format_chat_prompt(messages)
        if prompt is None:
            raise ValueError("prompt or messages is required.")

        should_speculate = speculative
        if should_speculate is None:
            should_speculate = (
                draft_model_name is not None or self.draft_model_name is not None
            )

        if should_speculate:
            result = self._generate_speculative_or_fallback(
                prompt,
                max_tokens=max_tokens,
                temp=temp,
                top_p=top_p,
                verbose=verbose,
                draft_model_name=draft_model_name,
                max_speculative_tokens=max_speculative_tokens,
                seed=seed,
                metrics=metrics,
                generate_kwargs=kwargs,
            )
            return result if return_metrics else result.text

        text = self._generate_plain(
            prompt,
            max_tokens=max_tokens,
            temp=temp,
                top_p=top_p,
                verbose=verbose,
                generate_kwargs=kwargs,
                paged_kv_cache=paged_kv_cache,
                metrics_sink=metrics,
            )
        if return_metrics:
            metrics_payload = SpeculativeDecodeMetrics(requested=False)
            return SpeculativeDecodeResult(text=text, metrics=metrics_payload)
        return text

    def _generate_plain(
        self,
        prompt: str | Sequence[int],
        *,
        max_tokens: int,
        temp: float,
        top_p: float,
        verbose: bool,
        generate_kwargs: Mapping[str, Any],
        paged_kv_cache: PagedKVCacheConfig | Mapping[str, Any] | bool | None = None,
        metrics_sink: Any = None,
    ) -> str:
        try:
            from mlx_lm import generate
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for OpenMed MLX language models. "
                "Install with: pip install openmed[mlx]"
            ) from exc

        prepared_kwargs = dict(generate_kwargs)
        raw_paged_config = paged_kv_cache
        if raw_paged_config is None:
            raw_paged_config = prepared_kwargs.pop("paged_kv_cache_config", None)
        cache_config = self._resolve_paged_kv_cache_config(raw_paged_config)
        self.last_paged_kv_cache_plan = None
        if cache_config is not None:
            token_count = _token_count_for_prompt(prompt, self.tokenizer)
            plan = cache_config.plan(token_count)
            self.last_paged_kv_cache_plan = plan
            _apply_paged_kv_generation_kwargs(generate, prepared_kwargs, plan)
            _record_paged_kv_metrics(metrics_sink, plan)

        if "sampler" not in prepared_kwargs and (temp > 0.0 or top_p < 1.0):
            try:
                from mlx_lm.sample_utils import make_sampler
            except ImportError as exc:
                raise ImportError(
                    "mlx-lm sampler utilities are required for non-greedy "
                    "OpenMed MLX language-model generation. Install with: "
                    "pip install openmed[mlx]"
                ) from exc

            sampler_top_p = top_p if top_p < 1.0 else 0.0
            prepared_kwargs["sampler"] = make_sampler(temp=temp, top_p=sampler_top_p)

        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=verbose,
            **prepared_kwargs,
        )

    def _resolve_paged_kv_cache_config(
        self,
        value: PagedKVCacheConfig | Mapping[str, Any] | bool | None,
    ) -> PagedKVCacheConfig | None:
        if value is True:
            if self.paged_kv_cache_config is None:
                raise ValueError(
                    "paged_kv_cache=True requires a PagedKVCacheConfig on the "
                    "runner or an explicit paged_kv_cache config"
                )
            return self.paged_kv_cache_config
        if value is None:
            return self.paged_kv_cache_config
        return _coerce_paged_kv_cache_config(value)

    def _generate_speculative_or_fallback(
        self,
        prompt: str | Sequence[int],
        *,
        max_tokens: int,
        temp: float,
        top_p: float,
        verbose: bool,
        draft_model_name: str | None,
        max_speculative_tokens: int,
        seed: int | None,
        metrics: Any,
        generate_kwargs: Mapping[str, Any],
    ) -> SpeculativeDecodeResult:
        metrics_payload = SpeculativeDecodeMetrics(requested=True)
        self.last_speculative_metrics = metrics_payload
        metrics_sink = metrics if metrics is not None else self.metrics
        started = time.perf_counter()

        fallback_reason: str | None = None
        if "sampler" in generate_kwargs:
            fallback_reason = "custom_sampler"
        elif max_speculative_tokens <= 0:
            fallback_reason = "non_positive_speculative_depth"
        elif not self._load_draft_model(draft_model_name):
            fallback_reason = "draft_model_unavailable"
        elif not tokenizers_are_aligned(self.tokenizer, self._draft_tokenizer):
            fallback_reason = "tokenizer_mismatch"

        if fallback_reason is not None:
            text = self._fallback_generate(
                prompt,
                metrics_payload,
                fallback_reason=fallback_reason,
                max_tokens=max_tokens,
                temp=temp,
                top_p=top_p,
                verbose=verbose,
                metrics_sink=metrics_sink,
                generate_kwargs=generate_kwargs,
                started=started,
            )
            return SpeculativeDecodeResult(text=text, metrics=metrics_payload)

        try:
            text = self._generate_speculative(
                prompt,
                max_tokens=max_tokens,
                temp=temp,
                top_p=top_p,
                max_speculative_tokens=max_speculative_tokens,
                seed=seed,
                metrics=metrics_payload,
            )
        except Exception as exc:
            text = self._fallback_generate(
                prompt,
                metrics_payload,
                fallback_reason=f"speculative_decode_failed:{type(exc).__name__}",
                max_tokens=max_tokens,
                temp=temp,
                top_p=top_p,
                verbose=verbose,
                metrics_sink=metrics_sink,
                generate_kwargs=generate_kwargs,
                started=started,
            )
            return SpeculativeDecodeResult(text=text, metrics=metrics_payload)

        metrics_payload.elapsed_seconds = time.perf_counter() - started
        _record_speculative_metrics(metrics_sink, metrics_payload)
        return SpeculativeDecodeResult(text=text, metrics=metrics_payload)

    def _fallback_generate(
        self,
        prompt: str | Sequence[int],
        metrics_payload: SpeculativeDecodeMetrics,
        *,
        fallback_reason: str,
        max_tokens: int,
        temp: float,
        top_p: float,
        verbose: bool,
        metrics_sink: Any,
        generate_kwargs: Mapping[str, Any],
        started: float,
    ) -> str:
        metrics_payload.enabled = False
        metrics_payload.fallback_reason = fallback_reason
        text = self._generate_plain(
            prompt,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            verbose=verbose,
            generate_kwargs=generate_kwargs,
        )
        metrics_payload.elapsed_seconds = time.perf_counter() - started
        _record_speculative_metrics(metrics_sink, metrics_payload)
        return text

    def _generate_speculative(
        self,
        prompt: str | Sequence[int],
        *,
        max_tokens: int,
        temp: float,
        top_p: float,
        max_speculative_tokens: int,
        seed: int | None,
        metrics: SpeculativeDecodeMetrics,
    ) -> str:
        if self._draft_model is None:
            raise ValueError("Draft model is not loaded.")

        prompt_tokens = _encode_prompt(self.tokenizer, prompt)
        if not prompt_tokens:
            bos = _bos_token_id(self.tokenizer)
            if bos is None:
                raise ValueError("Empty prompts require a tokenizer BOS token.")
            prompt_tokens = [bos]

        rng = random.Random(seed)
        eos_ids = _eos_token_ids(self.tokenizer)
        generated: list[int] = []
        metrics.enabled = True

        while len(generated) < max_tokens:
            prefix = [*prompt_tokens, *generated]
            remaining = max_tokens - len(generated)
            depth = min(max_speculative_tokens, remaining)
            draft_tokens, draft_probs = self._propose_draft_tokens(
                prefix,
                depth=depth,
                temp=temp,
                top_p=top_p,
                rng=rng,
                eos_ids=eos_ids,
                metrics=metrics,
            )
            if not draft_tokens:
                break

            verify_tokens = [*prefix, *draft_tokens]
            target_logits = _forward_logits(self.model, verify_tokens)
            metrics.target_batches += 1
            metrics.speculative_batches += 1
            metrics.drafted_tokens += len(draft_tokens)
            metrics.max_speculative_depth = max(
                metrics.max_speculative_depth,
                len(draft_tokens),
            )

            accepted_all = True
            for index, draft_token in enumerate(draft_tokens):
                target_position = len(prefix) - 1 + index
                logits = target_logits[target_position]
                if temp <= 0.0:
                    target_token = _argmax(logits)
                    if target_token == draft_token:
                        generated.append(draft_token)
                        metrics.accepted_tokens += 1
                        if draft_token in eos_ids:
                            metrics.generated_tokens = len(generated)
                            return _decode_tokens(self.tokenizer, generated)
                        continue

                    generated.append(target_token)
                    metrics.rollback_count += 1
                    metrics.target_sampled_tokens += 1
                    accepted_all = False
                    break

                target_probs = _distribution_from_logits(logits, temp=temp, top_p=top_p)
                target_token_prob = (
                    target_probs[draft_token]
                    if draft_token < len(target_probs)
                    else 0.0
                )
                draft_token_prob = (
                    draft_probs[index][draft_token]
                    if draft_token < len(draft_probs[index])
                    else 0.0
                )
                accept_prob = min(
                    1.0,
                    target_token_prob / max(draft_token_prob, 1e-12),
                )
                if rng.random() <= accept_prob:
                    generated.append(draft_token)
                    metrics.accepted_tokens += 1
                    if draft_token in eos_ids:
                        metrics.generated_tokens = len(generated)
                        return _decode_tokens(self.tokenizer, generated)
                    continue

                corrected = _correction_distribution(target_probs, draft_probs[index])
                sampled_token = _sample_from_distribution(corrected, rng)
                generated.append(sampled_token)
                metrics.rollback_count += 1
                metrics.target_sampled_tokens += 1
                accepted_all = False
                break

            if len(generated) >= max_tokens:
                break

            if accepted_all:
                bonus_position = len(prefix) + len(draft_tokens) - 1
                bonus_logits = target_logits[bonus_position]
                bonus_token = self._sample_target_token(
                    bonus_logits,
                    temp=temp,
                    top_p=top_p,
                    rng=rng,
                )
                generated.append(bonus_token)
                metrics.target_sampled_tokens += 1
                if bonus_token in eos_ids:
                    break

            if generated and generated[-1] in eos_ids:
                break

        metrics.generated_tokens = len(generated)
        return _decode_tokens(self.tokenizer, generated)

    def _propose_draft_tokens(
        self,
        prefix: Sequence[int],
        *,
        depth: int,
        temp: float,
        top_p: float,
        rng: random.Random,
        eos_ids: set[int],
        metrics: SpeculativeDecodeMetrics,
    ) -> tuple[list[int], list[list[float]]]:
        if self._draft_model is None:
            raise ValueError("Draft model is not loaded.")

        draft_tokens: list[int] = []
        draft_probs: list[list[float]] = []
        for _ in range(depth):
            logits = _last_token_logits(self._draft_model, [*prefix, *draft_tokens])
            metrics.draft_steps += 1
            probs = _distribution_from_logits(logits, temp=temp, top_p=top_p)
            token = (
                _argmax(logits)
                if temp <= 0.0
                else _sample_from_distribution(probs, rng)
            )
            draft_tokens.append(token)
            draft_probs.append(probs)
            if token in eos_ids:
                break
        return draft_tokens, draft_probs

    @staticmethod
    def _sample_target_token(
        logits: Sequence[float],
        *,
        temp: float,
        top_p: float,
        rng: random.Random,
    ) -> int:
        if temp <= 0.0:
            return _argmax(logits)
        probs = _distribution_from_logits(logits, temp=temp, top_p=top_p)
        return _sample_from_distribution(probs, rng)


def generate_text(
    prompt: str | Sequence[int] | None = None,
    *,
    messages: Sequence[Mapping[str, str]] | None = None,
    model_name: str = LANEFORMER_MLX_MODEL,
    config: Any = None,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 1.0,
    verbose: bool = False,
    speculative: bool | None = None,
    draft_model_name: str | None = None,
    max_speculative_tokens: int = DEFAULT_SPECULATIVE_TOKENS,
    seed: int | None = None,
    return_metrics: bool = False,
    paged_kv_cache: PagedKVCacheConfig | Mapping[str, Any] | bool | None = None,
    metrics: Any = None,
    **kwargs: Any,
) -> str | SpeculativeDecodeResult:
    """Load an OpenMed MLX language model and generate text.

    Args:
        prompt: Plain prompt text or token ids. Mutually exclusive with
            ``messages``.
        messages: Chat messages to render with the model chat template.
        model_name: Source model id, OpenMed MLX repo id, registry alias, or
            local artifact directory.
        config: Optional OpenMed config for cache placement.
        max_tokens: Maximum generated token count.
        temp: Sampling temperature forwarded to ``mlx_lm.generate``.
        top_p: Top-p sampling value forwarded to ``mlx_lm.generate``.
        verbose: Whether MLX-LM should print generation details.
        speculative: Enable draft-model speculative decoding. When ``None``,
            speculative decoding is enabled only if a draft model is supplied.
        draft_model_name: Optional draft model id or artifact directory.
        max_speculative_tokens: Maximum draft tokens proposed per target pass.
        seed: Optional Python RNG seed for speculative sampling tests.
        return_metrics: Return ``SpeculativeDecodeResult`` instead of text.
        paged_kv_cache: Optional paged-cache config or ``True`` to use the
            model runner's configured default.
        metrics: Optional metrics registry for this call.
        **kwargs: Additional keyword arguments forwarded to ``mlx_lm.generate``.

    Returns:
        Generated text from MLX-LM.
    """

    runner = OpenMedMLXLanguageModel(
        model_name=model_name,
        config=config,
        draft_model_name=draft_model_name,
        metrics=metrics,
    )
    return runner.generate(
        prompt,
        messages=messages,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p,
        verbose=verbose,
        speculative=speculative,
        draft_model_name=draft_model_name,
        max_speculative_tokens=max_speculative_tokens,
        seed=seed,
        return_metrics=return_metrics,
        paged_kv_cache=paged_kv_cache,
        metrics=metrics,
        **kwargs,
    )


def _chunk_token_ranges(
    total_tokens: int,
    chunk_size_tokens: int,
) -> tuple[TokenRange, ...]:
    if total_tokens <= 0:
        return ()
    return tuple(
        TokenRange(start, min(start + chunk_size_tokens, total_tokens))
        for start in range(0, total_tokens, chunk_size_tokens)
    )


def _coerce_paged_kv_cache_config(
    value: PagedKVCacheConfig | Mapping[str, Any] | bool | None,
) -> PagedKVCacheConfig | None:
    if value is None or value is False:
        return None
    if isinstance(value, PagedKVCacheConfig):
        return value
    if isinstance(value, Mapping):
        return PagedKVCacheConfig(**dict(value))
    if value is True:
        raise ValueError("paged KV-cache configuration requires a memory budget")
    raise TypeError(
        "paged_kv_cache must be a PagedKVCacheConfig, mapping, boolean, or None"
    )


def _is_token_id_sequence(prompt: Any) -> bool:
    return not isinstance(prompt, (str, bytes)) and isinstance(prompt, Sequence)


def _token_count_for_prompt(prompt: str | Sequence[int], tokenizer: Any) -> int:
    if _is_token_id_sequence(prompt):
        return len(prompt)

    text = str(prompt)
    tokenizer_obj = getattr(tokenizer, "tokenizer", tokenizer)
    encode = getattr(tokenizer_obj, "encode", None)
    if callable(encode):
        encoded = encode(text)
        ids = getattr(encoded, "ids", encoded)
        try:
            return len(ids)
        except TypeError:
            pass

    if callable(tokenizer_obj):
        try:
            encoded = tokenizer_obj(text, return_tensors=None)
        except TypeError:
            encoded = tokenizer_obj(text)
        input_ids = _extract_input_ids(encoded)
        if input_ids is not None:
            return len(input_ids)

    return len(text.split()) if text else 0


def _extract_input_ids(encoded: Any) -> Sequence[Any] | None:
    if isinstance(encoded, Mapping):
        input_ids = encoded.get("input_ids")
    else:
        input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None:
        return None
    try:
        first = input_ids[0]
    except (IndexError, TypeError):
        return input_ids
    if not isinstance(first, (str, bytes)) and isinstance(first, Sequence):
        return first
    return input_ids


def _apply_paged_kv_generation_kwargs(
    generate_fn: Any,
    generate_kwargs: dict[str, Any],
    plan: PagedKVCachePlan,
) -> None:
    if "prefill_step_size" not in generate_kwargs and _accepts_keyword(
        generate_fn, "prefill_step_size"
    ):
        generate_kwargs["prefill_step_size"] = plan.prefill_step_size
    if "max_kv_size" not in generate_kwargs and _accepts_keyword(
        generate_fn, "max_kv_size"
    ):
        generate_kwargs["max_kv_size"] = max(plan.resident_window_tokens, 1)


def _accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters


def _record_paged_kv_metrics(metrics: Any, plan: PagedKVCachePlan) -> None:
    if metrics is None:
        return
    record = getattr(metrics, "record_mlx_paged_kv_cache", None)
    if not callable(record):
        record = getattr(metrics, "record_paged_kv_cache", None)
    if not callable(record):
        return

    stats = plan.stats()
    try:
        record(
            total_pages=stats.total_pages,
            resident_pages=stats.resident_pages,
            evictions=stats.evictions,
            peak_pages=stats.peak_resident_pages,
            memory_budget_bytes=stats.memory_budget_bytes,
        )
    except TypeError:
        record(stats)


__all__ = [
    "LANEFORMER_MLX_MODEL",
    "LANEFORMER_SOURCE_MODEL",
    "LANEFORMER_DRAFT_MLX_MODEL",
    "DEFAULT_SPECULATIVE_TOKENS",
    "OpenMedMLXLanguageModel",
    "OpenMedPagedKVCache",
    "PagedKVCacheConfig",
    "PagedKVCachePlan",
    "PagedKVCacheStats",
    "SpeculativeDecodeMetrics",
    "SpeculativeDecodeResult",
    "TokenRange",
    "generate_text",
    "resolve_mlx_draft_language_model",
    "resolve_mlx_language_model",
    "tokenizer_alignment_fingerprint",
    "tokenizers_are_aligned",
]
