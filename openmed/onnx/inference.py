"""Local ONNX Runtime inference for OpenMed token-classification artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL_FILENAMES = {
    "fp32": "model.onnx",
    "fp16": "model_fp16.onnx",
    "int8": "model_int8.onnx",
}
DEFAULT_VARIANT_ORDER = ("int8", "fp32", "fp16")


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
        np, ort, auto_tokenizer, _ = _load_runtime_dependencies()
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
                explicit_model_path = path.resolve()
                artifact_dir = explicit_model_path.parent
            else:
                artifact_dir = path.resolve()
        else:
            _, _, _, snapshot_download = _load_runtime_dependencies()
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
    ) -> list[OnnxEntity]:
        """Predict aggregated entities and source-text offsets.

        Args:
            text: Text to classify.
            threshold: Minimum mean token probability for returned entities.
            max_length: Optional tokenizer truncation length.

        Returns:
            Aggregated entities ordered by source offset.
        """

        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        tokenizer_kwargs: dict[str, Any] = {
            "return_offsets_mapping": True,
            "return_tensors": "np",
            "truncation": True,
        }
        if max_length is not None:
            if max_length <= 0:
                raise ValueError("max_length must be positive")
            tokenizer_kwargs["max_length"] = max_length
        encoded = dict(self.tokenizer(text, **tokenizer_kwargs))
        offsets = encoded.pop("offset_mapping", None)
        if offsets is None:
            raise RuntimeError("tokenizer did not return offset_mapping")

        feed = self._build_feed(encoded)
        logits = self.session.run([self.output_name], feed)[0]
        sequence_logits = self._np.asarray(logits)[0]
        sequence_offsets = self._np.asarray(offsets)[0]
        return _decode_entities(
            self._np,
            sequence_logits,
            sequence_offsets,
            self.id2label,
            text,
            threshold=threshold,
        )

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
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {resolved}")
        return resolved, _variant_for_filename(resolved.name)

    normalized = variant.strip().lower()
    candidates = DEFAULT_VARIANT_ORDER if normalized == "auto" else (normalized,)
    for candidate in candidates:
        filename = MODEL_FILENAMES.get(candidate, candidate)
        path = artifact_dir / filename
        if path.is_file():
            return path, _variant_for_filename(filename)
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
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.jinja",
                ],
            )
        )
        if (artifact_dir / filename).is_file():
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
    return {int(key): str(value) for key, value in payload.items()}


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


def _load_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "OpenMed ONNX inference requires the ONNX Runtime extra. "
            "Install with: pip install 'openmed[onnx-runtime]'"
        ) from exc
    return np, ort, AutoTokenizer, snapshot_download


__all__ = ["OnnxEntity", "OnnxModel", "load_onnx_model"]
