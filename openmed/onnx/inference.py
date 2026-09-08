"""Local ONNX Runtime inference for OpenMed token-classification artifacts."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution import OnnxExecutionControl

MODEL_FILENAMES = {
    "fp32": "model.onnx",
    "fp16": "model_fp16.onnx",
    "int8": "model_int8.onnx",
}
DEFAULT_VARIANT_ORDER = ("int8", "fp32", "fp16")


@dataclass(frozen=True)
class _TokenWindow:
    """One input window with document-global token and character coordinates."""

    document: int
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    offsets: list[tuple[int, int]]
    token_indices: list[int | None]


class _TokenizersTokenizerAdapter:
    """Expose the small tokenizer surface needed by :class:`OnnxModel`."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._default_truncation = dict(tokenizer.truncation or {})
        self.model_max_length = self._default_truncation.get("max_length")
        padding = tokenizer.padding or {}
        self.pad_token_id = padding.get("pad_id")
        if self.pad_token_id is None:
            for token in ("[PAD]", "<pad>", "<|endoftext|>"):
                token_id = tokenizer.token_to_id(token)
                if token_id is not None:
                    self.pad_token_id = token_id
                    break
        self._lock = threading.Lock()
        # The Transformers wrapper does not apply tokenizer.json padding unless
        # callers request it. Match that behavior for single-note inference.
        self._tokenizer.no_padding()

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[Any]]:
        """Encode one text without importing a model framework."""
        if kwargs.get("return_offsets_mapping") is not True:
            raise ValueError("ONNX tokenization requires return_offsets_mapping=True")
        if kwargs.get("return_tensors") != "np":
            raise ValueError("ONNX tokenization requires return_tensors='np'")

        truncation = bool(kwargs.get("truncation", False))
        max_length = kwargs.get("max_length")
        with self._lock:
            try:
                if truncation:
                    options = dict(self._default_truncation)
                    if max_length is not None:
                        options["max_length"] = int(max_length)
                    if options:
                        self._tokenizer.enable_truncation(**options)
                else:
                    self._tokenizer.no_truncation()
                encoded = self._tokenizer.encode(text)
            finally:
                if self._default_truncation:
                    self._tokenizer.enable_truncation(**self._default_truncation)
                else:
                    self._tokenizer.no_truncation()

        return {
            "input_ids": [encoded.ids],
            "attention_mask": [encoded.attention_mask],
            "token_type_ids": [encoded.type_ids],
            "offset_mapping": [encoded.offsets],
        }

    def encode_windows(
        self, text: str, *, document: int, max_length: int, stride: int
    ) -> list[_TokenWindow]:
        """Encode every token with overlap, retaining source-text offsets."""
        with self._lock:
            try:
                # Build windows from the full encoding. Some tokenizer versions
                # return incomplete overflow chains; those cannot prove coverage.
                self._tokenizer.no_truncation()
                encoded = self._tokenizer.encode(text)
            finally:
                if self._default_truncation:
                    self._tokenizer.enable_truncation(**self._default_truncation)
                else:
                    self._tokenizer.no_truncation()
        return _windows_from_encoding(
            encoded.ids,
            encoded.offsets,
            encoded.attention_mask,
            encoded.type_ids,
            [
                i
                for i, (special, mask) in enumerate(
                    zip(encoded.special_tokens_mask, encoded.attention_mask)
                )
                if not special and mask
            ],
            document,
            max_length,
            stride,
        )


class _TokenizersTokenizerFactory:
    """Load a fast tokenizer directly from a pinned artifact directory."""

    def __init__(self, tokenizer_type: Any) -> None:
        self._tokenizer_type = tokenizer_type

    def from_pretrained(self, model_id: str, **kwargs: Any) -> Any:
        """Load ``tokenizer.json`` while preserving the former factory API."""
        del kwargs
        tokenizer_path = Path(model_id) / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Required ONNX artifact metadata is missing: {tokenizer_path.name}"
            )
        tokenizer = self._tokenizer_type.from_file(str(tokenizer_path))
        return _TokenizersTokenizerAdapter(tokenizer)


@dataclass(frozen=True)
class OnnxEntity:
    """One text entity predicted by an OpenMed ONNX model."""

    label: str
    score: float
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable entity dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class OnnxPrediction:
    """A complete document prediction and raw-text-free coverage metadata."""

    entities: tuple[OnnxEntity, ...]
    token_count: int
    processed_tokens: int
    window_count: int
    complete: bool


class OnnxModel:
    """CPU-first token-classification model backed by ONNX Runtime.

    Use :meth:`from_pretrained` with either an OpenMed Hugging Face repository
    id or a local artifact directory. Inference never performs network calls
    after the artifact and tokenizer files have been downloaded.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        model_path: str | Path | None = None,
        variant: str = "auto",
        providers: Sequence[str] = ("CPUExecutionProvider",),
        session_options: Any | None = None,
        tokenizer: Any | None = None,
        session: Any | None = None,
    ) -> None:
        np, ort, auto_tokenizer = _load_runtime_dependencies()
        self._np = np
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.model_path, self.variant = _resolve_model_path(
            self.artifact_dir,
            model_path=model_path,
            variant=variant,
        )
        self.config = _read_json(self.artifact_dir / "config.json")
        self.id2label = _read_id2label(self.artifact_dir, self.config)
        self.tokenizer = tokenizer or auto_tokenizer.from_pretrained(
            str(self.artifact_dir),
            local_files_only=True,
            use_fast=True,
        )
        self.session = session or ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=list(providers),
        )
        self.input_names = tuple(item.name for item in self.session.get_inputs())
        outputs = tuple(self.session.get_outputs())
        self.output_name = next(
            (item.name for item in outputs if item.name == "logits"),
            outputs[0].name if outputs else None,
        )
        if self.output_name is None:
            raise RuntimeError("ONNX model exposes no outputs")
        input_shape = getattr(self.session.get_inputs()[0], "shape", ()) or ()
        self._fixed_batch_size = (
            input_shape[0] if input_shape and isinstance(input_shape[0], int) else None
        )
        self._fixed_sequence_length = (
            input_shape[1]
            if len(input_shape) > 1 and isinstance(input_shape[1], int)
            else None
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path,
        *,
        variant: str = "auto",
        revision: str = "main",
        cache_dir: str | Path | None = None,
        token: str | None = None,
        local_files_only: bool = False,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        session_options: Any | None = None,
    ) -> "OnnxModel":
        """Load an OpenMed ONNX artifact from the Hub or local storage.

        Args:
            model_id: Hugging Face repository id, artifact directory, or ONNX
                file path.
            variant: ``"auto"``, ``"int8"``, ``"fp32"``, ``"fp16"``, or an
                ONNX filename. ``"auto"`` prefers the CPU-oriented INT8 graph.
            revision: Hub revision to download.
            cache_dir: Optional Hugging Face cache directory.
            token: Optional Hugging Face read token.
            local_files_only: Refuse network access and use cached files only.
            providers: ONNX Runtime execution providers in priority order.
            session_options: Optional ONNX Runtime ``SessionOptions`` object.

        Returns:
            A ready-to-run :class:`OnnxModel`.
        """

        path = Path(model_id).expanduser()
        explicit_model_path: Path | None = None
        if path.exists():
            if path.is_file():
                # Keep the snapshot filename: Hub graph symlinks point into a
                # blob directory without tokenizer metadata or external data.
                explicit_model_path = path.absolute()
                artifact_dir = explicit_model_path.parent
                if (
                    artifact_dir.name == "onnx"
                    and not (artifact_dir / "config.json").is_file()
                    and (artifact_dir.parent / "config.json").is_file()
                ):
                    artifact_dir = artifact_dir.parent
            else:
                artifact_dir = path.resolve()
        else:
            snapshot_download = _load_snapshot_download()
            artifact_dir = _download_artifact(
                snapshot_download,
                model_id=str(model_id),
                variant=variant,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )

        return cls(
            artifact_dir,
            model_path=explicit_model_path,
            variant=variant,
            providers=providers,
            session_options=session_options,
        )

    def predict(
        self,
        text: str,
        *,
        threshold: float = 0.0,
        max_length: int | None = None,
        stride: int | None = None,
    ) -> list[OnnxEntity]:
        """Predict aggregated entities and source-text offsets.

        Args:
            text: Text to classify.
            threshold: Minimum mean token probability for returned entities.
            max_length: Token window size, including special tokens. Long
                documents use overlapping windows and are never truncated.
            stride: Number of overlapping content tokens between windows.

        Returns:
            Aggregated entities ordered by source offset.
        """

        return self.predict_batch(
            [text], threshold=threshold, max_length=max_length, stride=stride
        )[0]

    def predict_batch(
        self,
        texts: Sequence[str],
        *,
        threshold: float = 0.0,
        max_length: int | None = None,
        stride: int | None = None,
        batch_size: int = 8,
        max_batch_tokens: int = 4096,
        max_windows: int = 4096,
    ) -> list[list[OnnxEntity]]:
        """Predict complete documents using length-bucketed tensor batches.

        Args:
            texts: Documents in desired output order.
            threshold: Minimum mean token probability for an entity.
            max_length: Maximum tokens per window, including special tokens.
            stride: Overlapping content tokens; defaults to at most 96.
            batch_size: Maximum windows in one runtime call.
            max_batch_tokens: Maximum padded token slots per runtime call.
            max_windows: Maximum windows in the request before rejecting it.

        Returns:
            One list of source-aligned entities for each input document.
        """
        return [
            list(result.entities)
            for result in self.predict_batch_detailed(
                texts,
                threshold=threshold,
                max_length=max_length,
                stride=stride,
                batch_size=batch_size,
                max_batch_tokens=max_batch_tokens,
                max_windows=max_windows,
            )
        ]

    def predict_batch_detailed(
        self,
        texts: Sequence[str],
        *,
        threshold: float = 0.0,
        max_length: int | None = None,
        stride: int | None = None,
        batch_size: int = 8,
        max_batch_tokens: int = 4096,
        max_windows: int = 4096,
        execution_control: OnnxExecutionControl | None = None,
    ) -> list[OnnxPrediction]:
        """Return batch predictions with verified document coverage counts.

        Arguments match :meth:`predict_batch`. Overlapping tokens use the
        window with the most surrounding context, then decode once in original
        token order. A boundary cannot truncate or duplicate an entity merely
        because it is split across runtime calls. Incomplete or malformed
        runtime output raises instead of returning a successful partial result.
        ``execution_control`` optionally applies one deadline and native
        cancellation scope across all windows in the document batch.
        """
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of documents")
        documents = list(texts)
        if any(not isinstance(text, str) or not text for text in documents):
            raise ValueError("text must be a non-empty string")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        for name, value in (
            ("batch_size", batch_size),
            ("max_batch_tokens", max_batch_tokens),
            ("max_windows", max_windows),
        ):
            _positive_integer(name, value)
        limits = [
            int(value)
            for value in (
                self.config.get("max_position_embeddings"),
                getattr(self.tokenizer, "model_max_length", None),
                self._fixed_sequence_length,
            )
            if isinstance(value, int) and 2 < value < 10**7
        ]
        model_limit = min(limits) if limits else 512
        window_size = min(model_limit, 512) if max_length is None else max_length
        _positive_integer("max_length", window_size)
        if window_size > model_limit:
            raise ValueError("max_length exceeds the model token limit")
        overlap = min(96, max(0, (window_size - 2) // 4)) if stride is None else stride
        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, int)
            or overlap < 0
            or overlap >= window_size
        ):
            raise ValueError("stride must be non-negative and smaller than max_length")
        if self._fixed_batch_size is not None:
            if self._fixed_batch_size != 1:
                raise ValueError(
                    "ONNX batching requires a dynamic or size-one batch axis"
                )
            batch_size = 1
        windows: list[_TokenWindow] = []
        counts = [0] * len(documents)
        token_counts = [0] * len(documents)
        for index, text in enumerate(documents):
            if execution_control is not None:
                execution_control.check()
            if hasattr(self.tokenizer, "encode_windows"):
                encoded_windows = self.tokenizer.encode_windows(
                    text, document=index, max_length=window_size, stride=overlap
                )
            else:
                encoded_windows = _custom_tokenizer_windows(
                    self.tokenizer, text, index, window_size, overlap
                )
            for window in encoded_windows:
                for token_index, offset in zip(window.token_indices, window.offsets):
                    if token_index is not None:
                        if not 0 <= offset[0] < offset[1] <= len(text):
                            raise RuntimeError(
                                "tokenizer returned invalid source offsets"
                            )
                        token_counts[index] = max(token_counts[index], token_index + 1)
            windows.extend(encoded_windows)
            counts[index] = len(encoded_windows)
            if len(windows) > max_windows:
                raise ValueError(
                    "request exceeds max_windows; no partial result returned"
                )
        # Each token keeps its best-context logit vector, not every window's
        # logits. No document text or mapping is stored on the shared model.
        votes: list[dict[int, tuple[int, tuple[int, int], Any]]] = [
            {} for _ in documents
        ]
        ordered = sorted(windows, key=lambda window: len(window.input_ids))
        cursor = 0
        while cursor < len(ordered):
            batch = []
            padded_length = 0
            while cursor < len(ordered) and len(batch) < batch_size:
                window = ordered[cursor]
                length = self._fixed_sequence_length or len(window.input_ids)
                next_length = max(length, padded_length)
                if next_length * (len(batch) + 1) > max_batch_tokens:
                    if not batch:
                        raise ValueError("one window exceeds max_batch_tokens")
                    break
                batch.append(window)
                padded_length = next_length
                cursor += 1
            feed = self._pad_windows(batch, padded_length)
            output = (
                execution_control.run(self.session, [self.output_name], feed)
                if execution_control is not None
                else self.session.run([self.output_name], feed)
            )
            logits = self._np.asarray(output[0])
            expected = (len(batch), padded_length, len(self.id2label))
            if logits.shape != expected or not self._np.isfinite(logits).all():
                raise RuntimeError("ONNX output has invalid shape or non-finite logits")
            for row, window in enumerate(batch):
                positions = [
                    index
                    for index, value in enumerate(window.token_indices)
                    if value is not None
                ]
                for rank, position in enumerate(positions):
                    token_index = window.token_indices[position]
                    assert token_index is not None
                    weight = 1 + min(rank, len(positions) - rank - 1)
                    existing = votes[window.document].get(token_index)
                    offset = tuple(window.offsets[position])
                    if existing is not None and existing[1] != offset:
                        raise RuntimeError("overlap token offsets are inconsistent")
                    if existing is None or weight > existing[0]:
                        votes[window.document][token_index] = (
                            weight,
                            offset,
                            logits[row, position].copy(),
                        )
        results = []
        for index, text in enumerate(documents):
            if execution_control is not None:
                execution_control.check()
            selected = votes[index]
            if sorted(selected) != list(range(token_counts[index])):
                raise RuntimeError("ONNX processing left an uncovered token interval")
            entities = []
            if selected:
                entities = _decode_entities(
                    self._np,
                    self._np.stack([selected[i][2] for i in range(len(selected))]),
                    [selected[i][1] for i in range(len(selected))],
                    self.id2label,
                    text,
                    threshold=threshold,
                )
            results.append(
                OnnxPrediction(
                    tuple(entities),
                    token_counts[index],
                    len(selected),
                    counts[index],
                    True,
                )
            )
        if execution_control is not None:
            execution_control.check()
        return results

    def _pad_windows(
        self, windows: Sequence[_TokenWindow], length: int
    ) -> dict[str, Any]:
        pad_id = self.config.get("pad_token_id")
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None and any(
            len(window.input_ids) != length for window in windows
        ):
            raise ValueError("dynamic padding requires a configured pad token")
        shape = (len(windows), length)
        encoded = {
            "input_ids": self._np.full(shape, pad_id or 0, dtype=self._np.int64),
            "attention_mask": self._np.zeros(shape, dtype=self._np.int64),
            "token_type_ids": self._np.zeros(shape, dtype=self._np.int64),
        }
        for row, window in enumerate(windows):
            size = len(window.input_ids)
            for name in encoded:
                encoded[name][row, :size] = getattr(window, name)
        return self._build_feed(encoded)

    def __call__(
        self,
        text: str,
        *,
        threshold: float = 0.0,
        max_length: int | None = None,
    ) -> list[OnnxEntity]:
        """Delegate to :meth:`predict` for pipeline-style use."""

        return self.predict(text, threshold=threshold, max_length=max_length)

    def _build_feed(self, encoded: Mapping[str, Any]) -> dict[str, Any]:
        inputs_by_name = {item.name: item for item in self.session.get_inputs()}
        feed: dict[str, Any] = {}
        for name in self.input_names:
            value = encoded.get(name)
            if value is None and name == "token_type_ids":
                input_ids = encoded.get("input_ids")
                if input_ids is not None:
                    value = self._np.zeros_like(input_ids)
            if value is None:
                raise RuntimeError(f"tokenizer did not provide required input {name!r}")
            dtype = _numpy_input_dtype(self._np, inputs_by_name[name])
            feed[name] = self._np.asarray(value, dtype=dtype)
        return feed


def load_onnx_model(
    model_id: str | Path,
    **kwargs: Any,
) -> OnnxModel:
    """Load an OpenMed ONNX model with :meth:`OnnxModel.from_pretrained`."""

    return OnnxModel.from_pretrained(model_id, **kwargs)


def _resolve_model_path(
    artifact_dir: Path,
    *,
    model_path: str | Path | None,
    variant: str,
) -> tuple[Path, str]:
    if model_path is not None:
        resolved = Path(model_path).expanduser().absolute()
        if not resolved.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {resolved}")
        return resolved, _variant_for_filename(resolved.name)

    normalized = variant.strip().lower()
    candidates = DEFAULT_VARIANT_ORDER if normalized == "auto" else (normalized,)
    for candidate in candidates:
        filename = MODEL_FILENAMES.get(candidate, candidate)
        for relative in (filename, f"onnx/{filename}"):
            path = artifact_dir / relative
            if path.is_file():
                return path, _variant_for_filename(Path(filename).name)
    expected = ", ".join(MODEL_FILENAMES.get(item, item) for item in candidates)
    raise FileNotFoundError(
        f"No ONNX graph for variant {variant!r} in {artifact_dir}; expected {expected}"
    )


def _download_artifact(
    snapshot_download: Any,
    *,
    model_id: str,
    variant: str,
    revision: str,
    cache_dir: str | Path | None,
    token: str | None,
    local_files_only: bool,
) -> Path:
    normalized = variant.strip().lower()
    candidates = DEFAULT_VARIANT_ORDER if normalized == "auto" else (normalized,)
    artifact_dir: Path | None = None
    for candidate in candidates:
        filename = MODEL_FILENAMES.get(candidate, candidate)
        artifact_dir = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                token=token,
                local_files_only=local_files_only,
                allow_patterns=[
                    filename,
                    f"{filename}.data",
                    f"{filename}_data",
                    f"onnx/{filename}",
                    f"onnx/{filename}.data",
                    f"onnx/{filename}_data",
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.jinja",
                ],
            )
        )
        if any(
            (artifact_dir / name).is_file() for name in (filename, f"onnx/{filename}")
        ):
            return artifact_dir
    expected = ", ".join(MODEL_FILENAMES.get(item, item) for item in candidates)
    raise FileNotFoundError(
        f"No ONNX graph for variant {variant!r} in {model_id}; expected {expected}"
    )


def _variant_for_filename(filename: str) -> str:
    for variant, candidate in MODEL_FILENAMES.items():
        if filename == candidate:
            return variant
    return filename


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required ONNX artifact metadata is missing: {path.name}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _read_id2label(
    artifact_dir: Path,
    config: Mapping[str, Any],
) -> dict[int, str]:
    label_path = artifact_dir / "id2label.json"
    payload: Any = (
        json.loads(label_path.read_text(encoding="utf-8"))
        if label_path.is_file()
        else config.get("id2label")
    )
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("OpenMed ONNX artifact must provide a non-empty id2label map")
    labels = {int(key): str(value) for key, value in payload.items()}
    if sorted(labels) != list(range(len(labels))):
        raise ValueError("ONNX id2label indices must be contiguous from zero")
    return labels


def _positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _custom_tokenizer_windows(
    tokenizer: Any, text: str, document: int, max_length: int, stride: int
) -> list[_TokenWindow]:
    """Adapt injected tokenizers while requiring an untruncated encoding."""
    encoded = dict(
        tokenizer(
            text, return_offsets_mapping=True, return_tensors="np", truncation=False
        )
    )
    if encoded.get("offset_mapping") is None:
        raise RuntimeError("tokenizer did not return offset_mapping")
    ids = list(encoded["input_ids"][0])
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"][0]]
    masks = list(encoded.get("attention_mask", [[1] * len(ids)])[0])
    types = list(encoded.get("token_type_ids", [[0] * len(ids)])[0])
    if not len(ids) == len(offsets) == len(masks) == len(types):
        raise RuntimeError("tokenizer returned inconsistent field lengths")
    content = [i for i, (start, end) in enumerate(offsets) if start != end and masks[i]]
    return _windows_from_encoding(
        ids, offsets, masks, types, content, document, max_length, stride
    )


def _windows_from_encoding(
    ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    masks: Sequence[int],
    types: Sequence[int],
    content: Sequence[int],
    document: int,
    max_length: int,
    stride: int,
) -> list[_TokenWindow]:
    """Slice the complete content encoding, preserving the model's affixes."""
    if not content:
        return []
    if list(content) != list(range(content[0], content[-1] + 1)):
        raise ValueError("ONNX windows require contiguous single-document content")
    prefix = list(range(content[0]))
    suffix = list(range(content[-1] + 1, len(ids)))
    capacity = max_length - len(prefix) - len(suffix)
    if capacity <= stride:
        raise ValueError("stride must be smaller than the content window")
    windows = []
    for start in range(0, len(content), capacity - stride):
        selected = content[start : start + capacity]
        positions = prefix + selected + suffix
        indices = [None] * len(prefix) + list(range(start, start + len(selected)))
        indices += [None] * len(suffix)
        windows.append(
            _TokenWindow(
                document,
                [ids[i] for i in positions],
                [masks[i] for i in positions],
                [types[i] for i in positions],
                [offsets[i] for i in positions],
                indices,
            )
        )
        if start + capacity >= len(content):
            break
    return windows


def _numpy_input_dtype(np: Any, input_info: Any) -> Any:
    input_type = str(getattr(input_info, "type", "tensor(int64)")).lower()
    if "int32" in input_type:
        return np.int32
    return np.int64


def _decode_entities(
    np: Any,
    logits: Any,
    offsets: Any,
    id2label: Mapping[int, str],
    text: str,
    *,
    threshold: float,
) -> list[OnnxEntity]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    label_ids = probabilities.argmax(axis=-1)
    scores = probabilities.max(axis=-1)
    entities: list[OnnxEntity] = []
    current: dict[str, Any] | None = None

    for label_id, score, offset in zip(label_ids, scores, offsets):
        start, end = int(offset[0]), int(offset[1])
        prefix, label = _split_label(id2label.get(int(label_id), f"LABEL_{label_id}"))
        if start == end or label.upper() == "O":
            current = _flush_entity(entities, current, text, threshold)
            continue

        starts_new = (
            current is None
            or current["label"] != label
            or prefix in {"B", "S", "U"}
            or (prefix not in {"I", "E", "L"} and start > int(current["end"]))
        )
        if starts_new:
            current = _flush_entity(entities, current, text, threshold)
            current = {
                "label": label,
                "start": start,
                "end": end,
                "scores": [float(score)],
            }
        else:
            current["end"] = max(int(current["end"]), end)
            current["scores"].append(float(score))

        if prefix in {"E", "L", "S", "U"}:
            current = _flush_entity(entities, current, text, threshold)

    _flush_entity(entities, current, text, threshold)
    return entities


def _split_label(raw_label: str) -> tuple[str, str]:
    normalized = str(raw_label).strip()
    if len(normalized) > 2 and normalized[1] in {"-", "_"}:
        prefix = normalized[0].upper()
        if prefix in {"B", "I", "E", "L", "S", "U"}:
            return prefix, normalized[2:]
    return "", normalized


def _flush_entity(
    entities: list[OnnxEntity],
    current: dict[str, Any] | None,
    text: str,
    threshold: float,
) -> None:
    if current is None:
        return None
    scores = current["scores"]
    score = sum(scores) / len(scores)
    start, end = int(current["start"]), int(current["end"])
    if score >= threshold and 0 <= start < end <= len(text):
        entities.append(
            OnnxEntity(
                label=str(current["label"]),
                score=score,
                start=start,
                end=end,
                text=text[start:end],
            )
        )
    return None


def _load_runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ImportError(
            "OpenMed ONNX inference requires the ONNX Runtime extra. "
            "Install with: pip install 'openmed[edge-sbc]' for local artifacts "
            "or pip install 'openmed[onnx-runtime]' for Hub downloads"
        ) from exc
    return np, ort, _TokenizersTokenizerFactory(Tokenizer)


def _load_snapshot_download() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Remote ONNX model downloads require the ONNX Runtime download "
            "profile. Install with: pip install 'openmed[onnx-runtime]', or "
            "copy a local artifact directory and use 'openmed[edge-sbc]'"
        ) from exc
    return snapshot_download


__all__ = ["OnnxEntity", "OnnxModel", "OnnxPrediction", "load_onnx_model"]
