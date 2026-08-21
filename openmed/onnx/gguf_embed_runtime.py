"""Out-of-process llama.cpp-compatible GGUF embedding runtime."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 120.0
DEFAULT_CONTEXT_SIZE = 512
DEFAULT_BATCH_SIZE = 32
MAX_EMBEDDING_TEXTS = 256
MAX_EMBEDDING_TEXT_CHARS = 32 * 1024
MAX_EMBEDDING_TOTAL_CHARS = 1024 * 1024
MAX_EMBEDDING_OUTPUT_CHARS = 4 * 1024 * 1024
MAX_EMBEDDING_DIMENSION = 65_536
MAX_OUTPUT_PARSE_CANDIDATES = 256
MAX_COMMAND_PARTS = 256
MAX_COMMAND_PART_CHARS = 32 * 1024
MAX_COMMAND_TOTAL_CHARS = 256 * 1024
MAX_CONTEXT_SIZE = 1_048_576
MAX_BATCH_SIZE = 65_536
LLAMA_CPP_EMBEDDING_BINARY_NAMES = (
    "llama-embedding",
    "embedding",
    "llama_embedding",
)


class GgufEmbeddingRuntimeError(RuntimeError):
    """Raised when the external llama.cpp embedding runtime cannot run."""


class LlamaCppEmbeddingRuntime:
    """Run a local GGUF embedding model through a llama.cpp subprocess.

    The runtime deliberately does not import a llama.cpp Python binding. Each
    input is passed as an argument to a local embedding executable, keeping
    the integration out of OpenMed's process and dependency graph.
    """

    def __init__(
        self,
        model_path: str | Path,
        executable: str | Path | None = None,
        *,
        llama_cpp_dir: str | Path | None = None,
        command: Sequence[str] | None = None,
        timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        context_size: int | None = DEFAULT_CONTEXT_SIZE,
        batch_size: int | None = DEFAULT_BATCH_SIZE,
        extra_args: Sequence[str] = (),
    ) -> None:
        if command is not None and (
            executable is not None or llama_cpp_dir is not None
        ):
            raise ValueError("provide command or executable/llama_cpp_dir, not both")
        validated_timeout = _positive_finite_float(
            timeout_seconds,
            name="timeout_seconds",
        )
        validated_context_size = _bounded_positive_int(
            context_size,
            name="context_size",
            maximum=MAX_CONTEXT_SIZE,
            allow_none=True,
        )
        validated_batch_size = _bounded_positive_int(
            batch_size,
            name="batch_size",
            maximum=MAX_BATCH_SIZE,
            allow_none=True,
        )

        resolved_model = Path(model_path).expanduser().resolve()
        if not resolved_model.is_file():
            raise FileNotFoundError(f"GGUF model not found: {resolved_model}")

        if command is not None:
            base_command = _normalize_command_parts(command, name="command")
        else:
            resolved_executable = resolve_llama_cpp_embedding_binary(
                executable,
                llama_cpp_dir=llama_cpp_dir,
            )
            base_command = (str(resolved_executable),)

        self.model_path = resolved_model
        self.command = base_command
        self.timeout_seconds = validated_timeout
        self.context_size = validated_context_size
        self.batch_size = validated_batch_size
        self.extra_args = _normalize_command_parts(
            extra_args,
            name="extra_args",
            allow_empty=True,
        )

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector for each non-empty input string."""

        normalized = _normalize_texts(texts)
        return [self._encode_one(text) for text in normalized]

    def embed(self, text: str) -> list[float]:
        """Return one embedding vector for a single non-empty string."""

        vectors = self.encode([text])
        return vectors[0]

    def _encode_one(self, text: str) -> list[float]:
        command = [*self.command, "--model", str(self.model_path)]
        command.extend(("--embeddings", "--pooling", "mean"))
        if self.context_size is not None:
            command.extend(("--ctx-size", str(self.context_size)))
        if self.batch_size is not None:
            command.extend(("--batch-size", str(self.batch_size)))
        command.extend(("--log-disable", "--prompt", text))
        command.extend(self.extra_args)

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise GgufEmbeddingRuntimeError(
                f"llama.cpp embedding exceeded {self.timeout_seconds} seconds"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise GgufEmbeddingRuntimeError(
                "llama.cpp embedding subprocess failed"
            ) from exc
        except (OSError, UnicodeError, ValueError) as exc:
            raise GgufEmbeddingRuntimeError(
                "could not start llama.cpp embedding subprocess"
            ) from exc

        try:
            if not isinstance(completed.stdout, str):
                raise ValueError("embedding output must be UTF-8 text")
            return _parse_embedding_output(completed.stdout)
        except ValueError as exc:
            raise GgufEmbeddingRuntimeError(
                "llama.cpp embedding output did not contain one finite vector"
            ) from exc


GgufEmbeddingRuntime = LlamaCppEmbeddingRuntime
GgufGroundingRuntime = LlamaCppEmbeddingRuntime
LlamaCppEmbeddingRunner = LlamaCppEmbeddingRuntime


def resolve_llama_cpp_embedding_binary(
    executable: str | Path | None = None,
    *,
    llama_cpp_dir: str | Path | None = None,
) -> Path:
    """Resolve a local llama.cpp embedding executable without downloading it."""

    if executable is not None and llama_cpp_dir is not None:
        raise ValueError("provide executable or llama_cpp_dir, not both")

    if executable is not None:
        resolved = Path(executable).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"llama.cpp embedding executable not found: {resolved}"
            )
        return resolved

    checkout_value = llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
    if checkout_value is None:
        raise FileNotFoundError(
            "llama.cpp embedding executable is not configured; pass executable, "
            "llama_cpp_dir, or set LLAMA_CPP_DIR"
        )

    checkout = Path(checkout_value).expanduser().resolve()
    for name in LLAMA_CPP_EMBEDDING_BINARY_NAMES:
        candidate = checkout / name
        if candidate.is_file():
            return candidate
        candidate_with_bin = checkout / "build" / "bin" / name
        if candidate_with_bin.is_file():
            return candidate_with_bin
    names = " or ".join(LLAMA_CPP_EMBEDDING_BINARY_NAMES)
    raise FileNotFoundError(f"{names} not found in llama.cpp checkout: {checkout}")


def _normalize_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)):
        raise ValueError("texts must be a sequence of non-empty strings")
    try:
        values = list(islice(iter(texts), MAX_EMBEDDING_TEXTS + 1))
    except TypeError as exc:
        raise ValueError("texts must be a sequence of non-empty strings") from exc
    if len(values) > MAX_EMBEDDING_TEXTS:
        raise ValueError(f"texts must contain at most {MAX_EMBEDDING_TEXTS} items")
    normalized = [text.strip() for text in values if isinstance(text, str)]
    if (
        len(normalized) != len(values)
        or not normalized
        or any(not text for text in normalized)
    ):
        raise ValueError("texts must contain only non-empty strings")
    if any("\0" in text for text in normalized):
        raise ValueError("texts must not contain NUL characters")
    if any(len(text) > MAX_EMBEDDING_TEXT_CHARS for text in normalized):
        raise ValueError(
            f"each text must contain at most {MAX_EMBEDDING_TEXT_CHARS} characters"
        )
    if sum(len(text) for text in normalized) > MAX_EMBEDDING_TOTAL_CHARS:
        raise ValueError(
            f"texts must contain at most {MAX_EMBEDDING_TOTAL_CHARS} characters total"
        )
    return normalized


def _parse_embedding_output(output: str) -> list[float]:
    if not isinstance(output, str) or not output.strip():
        raise ValueError("embedding output is empty")
    if len(output) > MAX_EMBEDDING_OUTPUT_CHARS:
        raise ValueError("embedding output exceeds the parsing limit")

    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    stripped = output.strip()
    try:
        candidates.append(json.loads(stripped))
    except (json.JSONDecodeError, RecursionError):
        pass

    for match in islice(
        re.finditer(r"[\[{]", output),
        MAX_OUTPUT_PARSE_CANDIDATES,
    ):
        try:
            payload, _ = decoder.raw_decode(output[match.start() :])
        except (json.JSONDecodeError, RecursionError):
            continue
        candidates.append(payload)

    for payload in reversed(candidates):
        try:
            vector = _extract_vector(payload)
        except RecursionError:
            continue
        if vector is not None:
            return vector

    bracket_vectors = list(
        islice(
            re.finditer(r"\[([^\[\]]+)\]", output, flags=re.DOTALL),
            MAX_OUTPUT_PARSE_CANDIDATES,
        )
    )
    for match in reversed(bracket_vectors):
        body = match.group(1)
        vector = _parse_number_sequence(body)
        if vector is not None:
            return vector

    for line in reversed(output.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        if ":" in candidate:
            candidate = candidate.rsplit(":", 1)[-1].strip()
        vector = _parse_number_sequence(candidate)
        if vector is not None:
            return vector

    raise ValueError("could not parse an embedding vector")


def _extract_vector(payload: Any) -> list[float] | None:
    if isinstance(payload, Mapping):
        for key in ("embedding", "embeddings", "vector", "data"):
            if key in payload:
                vector = _extract_vector(payload[key])
                if vector is not None:
                    return vector
        return None

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        values = list(islice(iter(payload), MAX_EMBEDDING_DIMENSION + 1))
        if len(values) > MAX_EMBEDDING_DIMENSION:
            raise ValueError("embedding vector exceeds the dimension limit")
        if values and all(_is_number(value) for value in values):
            return _validate_vector(values)
        if len(values) == 1:
            return _extract_vector(values[0])
        return None

    return None


def _parse_number_sequence(value: str) -> list[float] | None:
    tokens = [token for token in re.split(r"[\s,;]+", value.strip()) if token]
    if not tokens:
        return None
    if len(tokens) > MAX_EMBEDDING_DIMENSION:
        raise ValueError("embedding vector exceeds the dimension limit")
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        return None
    return _validate_vector(values)


def _validate_vector(values: Sequence[Any]) -> list[float]:
    if not values:
        raise ValueError("embedding vector is empty")
    if len(values) > MAX_EMBEDDING_DIMENSION:
        raise ValueError("embedding vector exceeds the dimension limit")
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding vector must contain numbers") from exc
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding vector must contain finite numbers")
    return vector


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return parsed


def _bounded_positive_int(
    value: Any,
    *,
    name: str,
    maximum: int,
    allow_none: bool,
) -> int | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer or None")
    if value <= 0 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}, or None")
    return value


def _normalize_command_parts(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        parts = list(islice(iter(values), MAX_COMMAND_PARTS + 1))
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if len(parts) > MAX_COMMAND_PARTS:
        raise ValueError(f"{name} must contain at most {MAX_COMMAND_PARTS} items")
    if not parts and not allow_empty:
        raise ValueError(f"{name} must contain an executable")
    if any(not isinstance(part, str) for part in parts):
        raise ValueError(f"{name} must contain only strings")
    if parts and not allow_empty and not parts[0]:
        raise ValueError(f"{name} executable must not be empty")
    if any("\0" in part for part in parts):
        raise ValueError(f"{name} must not contain NUL characters")
    if any(len(part) > MAX_COMMAND_PART_CHARS for part in parts):
        raise ValueError(
            f"each {name} item must contain at most {MAX_COMMAND_PART_CHARS} characters"
        )
    if sum(len(part) for part in parts) > MAX_COMMAND_TOTAL_CHARS:
        raise ValueError(
            f"{name} must contain at most {MAX_COMMAND_TOTAL_CHARS} characters total"
        )
    return tuple(parts)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CONTEXT_SIZE",
    "DEFAULT_EMBEDDING_TIMEOUT_SECONDS",
    "GgufEmbeddingRuntimeError",
    "GgufEmbeddingRuntime",
    "GgufGroundingRuntime",
    "LLAMA_CPP_EMBEDDING_BINARY_NAMES",
    "LlamaCppEmbeddingRunner",
    "LlamaCppEmbeddingRuntime",
    "MAX_EMBEDDING_DIMENSION",
    "MAX_EMBEDDING_OUTPUT_CHARS",
    "MAX_EMBEDDING_TEXT_CHARS",
    "MAX_EMBEDDING_TEXTS",
    "MAX_EMBEDDING_TOTAL_CHARS",
    "resolve_llama_cpp_embedding_binary",
]
