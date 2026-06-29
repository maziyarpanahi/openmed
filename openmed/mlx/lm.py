"""MLX language-model support for OpenMed.

The PII/NER MLX backend uses OpenMed's artifact contract. Causal language
models use the MLX-LM artifact contract so OpenMed can load custom `model_file`
implementations such as Laneformer without bundling model weights.
"""

from __future__ import annotations

import inspect
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

LANEFORMER_SOURCE_MODEL = "kogai/laneformer-2b-it"
LANEFORMER_MLX_MODEL = "OpenMed/laneformer-2b-it-q4-mlx"

_MLX_LANGUAGE_MODEL_MAP: dict[str, str] = {
    "laneformer-2b-it": LANEFORMER_MLX_MODEL,
    LANEFORMER_SOURCE_MODEL: LANEFORMER_MLX_MODEL,
    LANEFORMER_MLX_MODEL: LANEFORMER_MLX_MODEL,
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
    """Configuration for block-paged MLX-LM KV-cache planning.

    ``bytes_per_token`` is the model-specific KV footprint for one token across
    all layers. For a transformer this is typically:
    ``layers * kv_heads * head_dim * 2(K,V) * bytes_per_scalar``.
    """

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
    """Pure-Python page table for deterministic KV-cache accounting.

    The cache stores arbitrary key/value payloads by sequence id and token
    position. It is intentionally dependency-free so unit tests can stress page
    edges and eviction behavior without importing MLX.
    """

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
        position = self._next_positions.get(sequence_id, 0)
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
        """Return a stored token KV payload.

        Raises:
            KeyError: If the token's page has been evicted or the position has
                not been written.
        """
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
        owner = self._reverse_page_table.pop(physical_page)
        sequence_id, virtual_page = owner
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


class OpenMedMLXLanguageModel:
    """Small OpenMed wrapper around an MLX-LM causal language model."""

    def __init__(self, model_name: str = LANEFORMER_MLX_MODEL, config: Any = None):
        """Load an MLX-LM artifact.

        Args:
            model_name: Source model id, OpenMed MLX repo id, registry alias, or
                local artifact directory.
            config: Optional OpenMed config. If present, its ``cache_dir`` is
                passed to Hugging Face snapshot downloads.
        """

        self.model_name = model_name
        self.config = config
        self.model_path = resolve_mlx_language_model(model_name, config=config)
        self.paged_kv_cache_config = _coerce_paged_kv_cache_config(
            getattr(config, "paged_kv_cache", None) if config is not None else None
        )
        self.last_paged_kv_cache_plan: PagedKVCachePlan | None = None

        try:
            from mlx_lm import load
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for OpenMed MLX language models. "
                "Install with: pip install openmed[mlx]"
            ) from exc

        self.model, self.tokenizer = load(self.model_path)

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
        paged_kv_cache: PagedKVCacheConfig | Mapping[str, Any] | bool | None = None,
        metrics: Any = None,
        **kwargs: Any,
    ) -> str:
        """Generate text with the loaded MLX language model.

        Args:
            prompt: Plain prompt text or token ids. Mutually exclusive with
                ``messages``.
            messages: Chat messages to render with the model chat template.
            max_tokens: Maximum generated token count.
            temp: Sampling temperature forwarded to ``mlx_lm.generate``.
            top_p: Top-p sampling value forwarded to ``mlx_lm.generate``.
            verbose: Whether MLX-LM should print generation details.
            paged_kv_cache: Optional paged-cache config or ``True`` to use the
                runner's configured default. When enabled, OpenMed adds
                chunked-prefill and bounded KV-cache arguments where supported.
            metrics: Optional service metrics registry receiving aggregate
                paged-cache occupancy and eviction counts.
            **kwargs: Additional keyword arguments forwarded to
                ``mlx_lm.generate``.

        Returns:
            Generated text from MLX-LM.
        """

        if messages is not None:
            if prompt is not None:
                raise ValueError("Pass either prompt or messages, not both.")
            prompt = self.format_chat_prompt(messages)
        if prompt is None:
            raise ValueError("prompt or messages is required.")

        try:
            from mlx_lm import generate
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for OpenMed MLX language models. "
                "Install with: pip install openmed[mlx]"
            ) from exc

        generate_kwargs = dict(kwargs)
        raw_paged_config = paged_kv_cache
        if raw_paged_config is None:
            raw_paged_config = generate_kwargs.pop("paged_kv_cache_config", None)
        cache_config = self._resolve_paged_kv_cache_config(raw_paged_config)
        self.last_paged_kv_cache_plan = None
        if cache_config is not None:
            token_count = _token_count_for_prompt(prompt, self.tokenizer)
            plan = cache_config.plan(token_count)
            self.last_paged_kv_cache_plan = plan
            _apply_paged_kv_generation_kwargs(generate, generate_kwargs, plan)
            _record_paged_kv_metrics(metrics, plan)

        if "sampler" not in generate_kwargs and (temp > 0.0 or top_p < 1.0):
            try:
                from mlx_lm.sample_utils import make_sampler
            except ImportError as exc:
                raise ImportError(
                    "mlx-lm sampler utilities are required for non-greedy "
                    "OpenMed MLX language-model generation. Install with: "
                    "pip install openmed[mlx]"
                ) from exc

            sampler_top_p = top_p if top_p < 1.0 else 0.0
            generate_kwargs["sampler"] = make_sampler(temp=temp, top_p=sampler_top_p)

        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=verbose,
            **generate_kwargs,
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
    paged_kv_cache: PagedKVCacheConfig | Mapping[str, Any] | bool | None = None,
    metrics: Any = None,
    **kwargs: Any,
) -> str:
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
        paged_kv_cache: Optional paged-cache config or ``True`` to use the
            model runner's configured default.
        metrics: Optional service metrics registry receiving aggregate
            paged-cache occupancy and eviction counts.
        **kwargs: Additional keyword arguments forwarded to ``mlx_lm.generate``.

    Returns:
        Generated text from MLX-LM.
    """

    runner = OpenMedMLXLanguageModel(model_name=model_name, config=config)
    return runner.generate(
        prompt,
        messages=messages,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p,
        verbose=verbose,
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
    "OpenMedMLXLanguageModel",
    "OpenMedPagedKVCache",
    "PagedKVCacheConfig",
    "PagedKVCachePlan",
    "PagedKVCacheStats",
    "TokenRange",
    "generate_text",
    "resolve_mlx_language_model",
]
