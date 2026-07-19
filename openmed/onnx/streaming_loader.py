"""Deterministic, local-only streaming for sharded model weights.

The loader understands standard Safetensors shard indexes and ONNX graphs
whose initializers use ONNX external data. It maps one deterministic layer
group at a time and releases those mappings before advancing to the next
group. A consumer can therefore copy each group into its runtime model without
temporarily materializing every source weight at once.
"""

from __future__ import annotations

import gc
import json
import math
import mmap
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

from openmed.onnx.ram_budget import (
    PeakRamProbe,
    PeakRamReport,
    RamBudget,
    RssSampler,
)

_SAFETENSORS_INDEX_SUFFIX = ".safetensors.index.json"
_PARAMETER_SUFFIX = re.compile(
    r"(?:[./](?:weight|bias|gamma|beta|running_mean|running_var))$",
    re.IGNORECASE,
)
_NUMBERED_LAYER = re.compile(
    r"(?:^|[./])(?:encoder[./])?"
    r"(?:layer|layers|block|blocks|transformer[./]h|h)[./](\d+)(?:[./]|$)",
    re.IGNORECASE,
)

_SAFETENSORS_ITEM_SIZES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class ShardFormatError(ValueError):
    """Raised when a shard or its metadata is incomplete or unsafe."""


class LocalWeightsRequired(ValueError):
    """Raised when a source could resolve outside its local artifact root."""


class BufferReleaseError(RuntimeError):
    """Raised when a consumer retains a mapped tensor beyond its callback."""


@dataclass(frozen=True)
class LayerGroupSpec:
    """Metadata for one deterministic group without any mapped tensors."""

    name: str
    tensor_names: tuple[str, ...]
    source_files: tuple[Path, ...]
    mapped_bytes: int


@dataclass(frozen=True)
class StreamingLoadReport:
    """Memory and shard counts recorded for a completed streaming load."""

    source: Path
    source_format: str
    groups_loaded: int
    tensors_loaded: int
    bytes_mapped: int
    peak_ram: PeakRamReport


@dataclass(frozen=True)
class _TensorSpec:
    name: str
    path: Path
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _PlannedGroup:
    public: LayerGroupSpec
    tensors: tuple[_TensorSpec, ...]


@dataclass(frozen=True)
class _LoadPlan:
    source: Path
    source_format: str
    groups: tuple[_PlannedGroup, ...]


class StreamedLayerGroup:
    """Read-only tensor views valid for the duration of one load callback.

    Call :meth:`release` after copying or applying the tensors. A loader-owned
    group is released automatically when iteration advances. Retaining an
    array view after that boundary is rejected because it would defeat the RAM
    bound.
    """

    def __init__(
        self,
        spec: LayerGroupSpec,
        tensors: dict[str, Any],
        mappings: list[mmap.mmap],
    ) -> None:
        self.spec = spec
        self._tensor_storage = tensors
        self._tensors = MappingProxyType(tensors)
        self._mappings = mappings
        self._released = False

    @property
    def name(self) -> str:
        """Return the deterministic layer-group name."""

        return self.spec.name

    @property
    def tensors(self) -> Mapping[str, Any]:
        """Return read-only, memory-mapped NumPy tensor views."""

        if self._released:
            raise RuntimeError(f"Layer group {self.name!r} has been released")
        return self._tensors

    @property
    def released(self) -> bool:
        """Return whether all file mappings have been closed."""

        return self._released

    def release(self) -> None:
        """Drop tensor views and close their file mappings immediately."""

        if self._released:
            return
        self._tensor_storage.clear()
        gc.collect()
        failed: list[mmap.mmap] = []
        for mapping in self._mappings:
            _discard_mapped_pages(mapping)
            try:
                mapping.close()
            except BufferError:
                failed.append(mapping)
        self._mappings = failed
        if failed:
            raise BufferReleaseError(
                f"Layer group {self.name!r} still has retained tensor views; "
                "copy values needed after the streaming callback instead of "
                "retaining mapped arrays"
            )
        self._released = True

    def __enter__(self) -> "StreamedLayerGroup":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.release()
        return False

    def __del__(self) -> None:
        if not getattr(self, "_released", True):
            try:
                self.release()
            except Exception:
                pass


class StreamingWeightLoader:
    """Stream local sharded weights under a measured incremental-RSS budget.

    Args:
        source: Local ``.onnx`` graph, ``.safetensors`` file, standard
            ``.safetensors.index.json`` file, or artifact directory.
        ram_budget: Positive incremental-RSS budget in bytes or a
            :class:`RamBudget`.
        ram_budget_bytes: Byte-oriented alternative to ``ram_budget``.
        layers_per_group: Consecutive logical layers mapped together.
        rss_sampler: Optional current-RSS sampler for platform integration or
            deterministic testing.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        ram_budget: RamBudget | int | None = None,
        ram_budget_bytes: int | None = None,
        layers_per_group: int = 1,
        rss_sampler: RssSampler | None = None,
    ) -> None:
        if ram_budget is None and ram_budget_bytes is None:
            raise ValueError("A fail-closed RAM budget is required")
        if ram_budget is not None and ram_budget_bytes is not None:
            raise ValueError("Pass ram_budget or ram_budget_bytes, not both")
        selected_budget = ram_budget if ram_budget is not None else ram_budget_bytes
        assert selected_budget is not None
        if isinstance(layers_per_group, bool) or layers_per_group <= 0:
            raise ValueError("layers_per_group must be a positive integer")

        self.source = _resolve_local_source(source)
        self.ram_budget = (
            selected_budget
            if isinstance(selected_budget, RamBudget)
            else RamBudget(selected_budget)
        )
        self.layers_per_group = layers_per_group
        self._rss_sampler = rss_sampler
        self._plan: _LoadPlan | None = None
        self.last_report: StreamingLoadReport | None = None

    @property
    def layer_groups(self) -> tuple[LayerGroupSpec, ...]:
        """Return deterministic group metadata without mapping weight data."""

        if self._plan is None:
            with PeakRamProbe(
                self.ram_budget,
                rss_sampler=self._rss_sampler,
            ) as probe:
                self._plan = self._discover(probe)
        return tuple(group.public for group in self._plan.groups)

    @property
    def source_format(self) -> str:
        """Return ``"onnx-external-data"`` or ``"safetensors"``."""

        if self._plan is None:
            self.layer_groups
        assert self._plan is not None
        return self._plan.source_format

    def iter_layer_groups(self) -> Iterator[StreamedLayerGroup]:
        """Yield and automatically release one mapped layer group at a time."""

        probe = PeakRamProbe(self.ram_budget, rss_sampler=self._rss_sampler)
        groups_loaded = 0
        tensors_loaded = 0
        bytes_mapped = 0
        plan: _LoadPlan | None = None

        try:
            with probe:
                plan = self._plan or self._discover(probe)
                self._plan = plan
                for planned in plan.groups:
                    probe.reserve(
                        planned.public.mapped_bytes,
                        f"mapping layer group {planned.public.name!r}",
                    )
                    group = _open_group(planned)
                    groups_loaded += 1
                    tensors_loaded += len(planned.tensors)
                    bytes_mapped += planned.public.mapped_bytes
                    try:
                        probe.checkpoint(f"mapping layer group {planned.public.name!r}")
                        yield group
                        probe.checkpoint(
                            f"applying layer group {planned.public.name!r}"
                        )
                    finally:
                        group.release()
                    probe.checkpoint(f"releasing layer group {planned.public.name!r}")
        finally:
            if probe.started:
                report_source = plan.source if plan is not None else self.source
                report_format = (
                    plan.source_format if plan is not None else "undiscovered"
                )
                self.last_report = StreamingLoadReport(
                    source=report_source,
                    source_format=report_format,
                    groups_loaded=groups_loaded,
                    tensors_loaded=tensors_loaded,
                    bytes_mapped=bytes_mapped,
                    peak_ram=probe.report,
                )

    def stream(
        self,
        consumer: Callable[[StreamedLayerGroup], Any],
    ) -> StreamingLoadReport:
        """Apply ``consumer`` to each group and return measured load metadata."""

        if not callable(consumer):
            raise TypeError("consumer must be callable")
        for group in self.iter_layer_groups():
            consumer(group)
        assert self.last_report is not None
        return self.last_report

    def load(
        self,
        consumer: Callable[[StreamedLayerGroup], Any],
    ) -> StreamingLoadReport:
        """Alias for :meth:`stream` for loader-style integrations."""

        return self.stream(consumer)

    def _discover(self, probe: PeakRamProbe) -> _LoadPlan:
        selected = _select_source(self.source)
        if selected.name.endswith(_SAFETENSORS_INDEX_SUFFIX):
            tensors = _discover_safetensors_index(selected, probe)
            source_format = "safetensors"
        elif selected.is_dir() or selected.suffix.lower() == ".safetensors":
            paths = (
                tuple(sorted(selected.glob("*.safetensors")))
                if selected.is_dir()
                else (selected,)
            )
            tensors = _discover_safetensors_files(paths, probe)
            source_format = "safetensors"
        elif selected.suffix.lower() == ".onnx":
            tensors = _discover_onnx_external_data(selected, probe)
            source_format = "onnx-external-data"
        else:
            raise ShardFormatError(f"Unsupported weight source: {selected}")

        groups = _group_tensors(tensors, self.layers_per_group)
        if not groups:
            raise ShardFormatError(f"No weight tensors found in {selected}")
        return _LoadPlan(
            source=selected,
            source_format=source_format,
            groups=groups,
        )


def _resolve_local_source(source: str | Path) -> Path:
    raw = str(source)
    if "://" in raw or raw.startswith(("hf:", "http:", "https:")):
        raise LocalWeightsRequired(
            "Streaming weights must come from a local path; network identifiers "
            "are not allowed"
        )
    path = Path(source).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Weight source not found: {path}") from exc
    if not (resolved.is_file() or resolved.is_dir()):
        raise FileNotFoundError(f"Weight source is not a file or directory: {path}")
    return resolved


def _select_source(source: Path) -> Path:
    if source.is_file():
        return source

    indexes = tuple(sorted(source.glob(f"*{_SAFETENSORS_INDEX_SUFFIX}")))
    if len(indexes) == 1:
        return indexes[0]
    if len(indexes) > 1:
        raise ShardFormatError(
            f"Artifact directory contains multiple Safetensors indexes: {source}"
        )

    standard_onnx = source / "model.onnx"
    if standard_onnx.is_file():
        return standard_onnx

    safetensors_paths = tuple(sorted(source.glob("*.safetensors")))
    if safetensors_paths:
        return source

    onnx_paths = tuple(sorted(source.glob("*.onnx")))
    if len(onnx_paths) == 1:
        return onnx_paths[0]
    if len(onnx_paths) > 1:
        raise ShardFormatError(
            f"Artifact directory contains multiple ONNX graphs: {source}"
        )
    raise ShardFormatError(f"No sharded ONNX or Safetensors weights in {source}")


def _discover_safetensors_index(
    index_path: Path,
    probe: PeakRamProbe,
) -> tuple[_TensorSpec, ...]:
    probe.reserve(index_path.stat().st_size, "reading the Safetensors shard index")
    payload = _read_json_object(index_path)
    probe.checkpoint("reading the Safetensors shard index")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ShardFormatError(
            f"Safetensors index must contain a non-empty weight_map: {index_path}"
        )

    expected: dict[str, Path] = {}
    root = index_path.parent.resolve()
    for name, location in weight_map.items():
        if not isinstance(name, str) or not name:
            raise ShardFormatError("Safetensors tensor names must be non-empty strings")
        if not isinstance(location, str) or not location:
            raise ShardFormatError(f"Shard location for {name!r} must be a string")
        expected[name] = _resolve_artifact_file(root, location)

    discovered_by_file: dict[Path, dict[str, _TensorSpec]] = {}
    for path in sorted(set(expected.values())):
        discovered_by_file[path] = {
            tensor.name: tensor for tensor in _read_safetensors_header(path, probe)
        }

    tensors: list[_TensorSpec] = []
    for name in sorted(expected, key=_natural_key):
        path = expected[name]
        tensor = discovered_by_file[path].get(name)
        if tensor is None:
            raise ShardFormatError(
                f"Safetensors index maps {name!r} to {path.name}, but the tensor "
                "is absent from that shard"
            )
        tensors.append(tensor)
    return tuple(tensors)


def _discover_safetensors_files(
    paths: tuple[Path, ...],
    probe: PeakRamProbe,
) -> tuple[_TensorSpec, ...]:
    if not paths:
        raise ShardFormatError("No Safetensors shard files were found")
    by_name: dict[str, _TensorSpec] = {}
    for path in sorted(paths):
        for tensor in _read_safetensors_header(path, probe):
            if tensor.name in by_name:
                raise ShardFormatError(
                    f"Duplicate tensor {tensor.name!r} across Safetensors shards"
                )
            by_name[tensor.name] = tensor
    return tuple(by_name[name] for name in sorted(by_name, key=_natural_key))


def _read_safetensors_header(
    path: Path,
    probe: PeakRamProbe,
) -> tuple[_TensorSpec, ...]:
    file_size = path.stat().st_size
    if file_size < 8:
        raise ShardFormatError(f"Safetensors shard is shorter than its header: {path}")
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        header_length = int.from_bytes(raw_length, "little", signed=False)
        if header_length <= 0 or header_length > file_size - 8:
            raise ShardFormatError(f"Invalid Safetensors header length in {path}")
        probe.reserve(
            header_length + 8,
            f"reading Safetensors metadata from {path.name}",
        )
        header_bytes = handle.read(header_length)
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShardFormatError(f"Invalid Safetensors JSON header in {path}") from exc
    if not isinstance(header, Mapping):
        raise ShardFormatError(f"Safetensors header must be an object: {path}")
    probe.checkpoint(f"reading Safetensors metadata from {path.name}")

    data_start = 8 + header_length
    tensors: list[_TensorSpec] = []
    for name in sorted(
        (item for item in header if item != "__metadata__"),
        key=_natural_key,
    ):
        metadata = header[name]
        if not isinstance(name, str) or not name or not isinstance(metadata, Mapping):
            raise ShardFormatError(f"Invalid tensor entry in {path}")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if dtype not in _SAFETENSORS_ITEM_SIZES:
            raise ShardFormatError(
                f"Unsupported Safetensors dtype {dtype!r} for tensor {name!r}"
            )
        normalized_shape = _validate_shape(shape, name)
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets
            )
        ):
            raise ShardFormatError(f"Invalid data_offsets for tensor {name!r}")
        start, end = offsets
        if start < 0 or end < start:
            raise ShardFormatError(f"Invalid data_offsets for tensor {name!r}")
        expected_bytes = _shape_size(normalized_shape) * _SAFETENSORS_ITEM_SIZES[dtype]
        if end - start != expected_bytes:
            raise ShardFormatError(
                f"Safetensors byte length does not match shape for tensor {name!r}"
            )
        absolute_end = data_start + end
        if absolute_end > file_size:
            raise ShardFormatError(f"Tensor {name!r} extends beyond shard {path.name}")
        tensors.append(
            _TensorSpec(
                name=name,
                path=path,
                offset=data_start + start,
                nbytes=expected_bytes,
                shape=normalized_shape,
                dtype=dtype,
            )
        )
    return tuple(tensors)


def _discover_onnx_external_data(
    model_path: Path,
    probe: PeakRamProbe,
) -> tuple[_TensorSpec, ...]:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "ONNX external-data loading requires the ONNX extra. "
            "Install with: pip install 'openmed[onnx]'"
        ) from exc
    np = _load_numpy()
    probe.reserve(model_path.stat().st_size, "reading ONNX graph metadata")
    model = onnx.load(str(model_path), load_external_data=False)
    probe.checkpoint("reading ONNX graph metadata")

    root = model_path.parent.resolve()
    tensors: list[_TensorSpec] = []
    try:
        for tensor in model.graph.initializer:
            name = str(tensor.name)
            if not name:
                raise ShardFormatError("ONNX initializers must have stable names")
            try:
                dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type))
            except (KeyError, TypeError, ValueError) as exc:
                raise ShardFormatError(
                    f"Unsupported ONNX dtype for initializer {name!r}"
                ) from exc
            if dtype.hasobject:
                raise ShardFormatError(
                    f"Object/string ONNX initializer {name!r} cannot be memory mapped"
                )
            shape = tuple(int(value) for value in tensor.dims)
            nbytes = _shape_size(shape) * dtype.itemsize
            external = {str(item.key): str(item.value) for item in tensor.external_data}
            if not external:
                if nbytes == 0:
                    continue
                raise ShardFormatError(
                    f"ONNX initializer {name!r} is inline; streaming requires "
                    "external-data weights"
                )
            location = external.get("location")
            if not location:
                raise ShardFormatError(
                    f"ONNX initializer {name!r} has no external-data location"
                )
            path = _resolve_artifact_file(root, location)
            offset = _parse_non_negative_int(
                external.get("offset", "0"), name, "offset"
            )
            declared_length = _parse_non_negative_int(
                external.get("length", str(nbytes)),
                name,
                "length",
            )
            if declared_length < nbytes:
                raise ShardFormatError(
                    f"ONNX external data is shorter than initializer {name!r}"
                )
            if offset + declared_length > path.stat().st_size:
                raise ShardFormatError(
                    f"ONNX initializer {name!r} extends beyond {path.name}"
                )
            tensors.append(
                _TensorSpec(
                    name=name,
                    path=path,
                    offset=offset,
                    nbytes=nbytes,
                    shape=shape,
                    dtype=dtype.newbyteorder("<").str,
                )
            )
    finally:
        del model
        gc.collect()
        probe.checkpoint("releasing ONNX graph metadata")
    return tuple(sorted(tensors, key=lambda item: _natural_key(item.name)))


def _resolve_artifact_file(root: Path, location: str) -> Path:
    if "://" in location:
        raise LocalWeightsRequired(
            f"Shard location must be local to the artifact: {location!r}"
        )
    relative = Path(location)
    if relative.is_absolute():
        raise LocalWeightsRequired(
            f"Shard location must be relative to the artifact: {location!r}"
        )
    try:
        candidate = (root / relative).resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Weight shard not found: {location}") from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise LocalWeightsRequired(
            f"Shard location escapes the local artifact directory: {location!r}"
        ) from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Weight shard is not a file: {candidate}")
    return candidate


def _group_tensors(
    tensors: tuple[_TensorSpec, ...],
    layers_per_group: int,
) -> tuple[_PlannedGroup, ...]:
    buckets: dict[str, list[_TensorSpec]] = {}
    ordering: dict[str, tuple[Any, ...]] = {}
    for tensor in tensors:
        key, order = _layer_bucket(tensor.name)
        buckets.setdefault(key, []).append(tensor)
        ordering[key] = order

    layer_keys = sorted(buckets, key=lambda key: ordering[key])
    groups: list[_PlannedGroup] = []
    for start in range(0, len(layer_keys), layers_per_group):
        keys = layer_keys[start : start + layers_per_group]
        selected = tuple(
            sorted(
                (tensor for key in keys for tensor in buckets[key]),
                key=lambda item: _natural_key(item.name),
            )
        )
        name = keys[0] if len(keys) == 1 else f"{keys[0]}..{keys[-1]}"
        public = LayerGroupSpec(
            name=name,
            tensor_names=tuple(tensor.name for tensor in selected),
            source_files=tuple(sorted({tensor.path for tensor in selected})),
            mapped_bytes=sum(tensor.nbytes for tensor in selected),
        )
        groups.append(_PlannedGroup(public=public, tensors=selected))
    return tuple(groups)


def _layer_bucket(name: str) -> tuple[str, tuple[Any, ...]]:
    normalized = name.replace("/", ".")
    numbered = _NUMBERED_LAYER.search(normalized)
    if numbered:
        index = int(numbered.group(1))
        return f"layer-{index:04d}", (1, index, _natural_key(normalized))
    lowered = normalized.lower()
    if "embedding" in lowered:
        return "embeddings", (0, 0, _natural_key(normalized))
    if any(
        part in lowered for part in ("classifier", "classification_head", "lm_head")
    ):
        return "classifier", (3, 0, _natural_key(normalized))
    module = _PARAMETER_SUFFIX.sub("", normalized)
    return module, (2, 0, _natural_key(module))


def _open_group(planned: _PlannedGroup) -> StreamedLayerGroup:
    np = _load_numpy()
    tensors: dict[str, Any] = {}
    mappings: list[mmap.mmap] = []
    try:
        for spec in planned.tensors:
            if spec.nbytes == 0:
                array = np.empty(spec.shape, dtype=_numpy_dtype(np, spec.dtype))
                array.setflags(write=False)
                tensors[spec.name] = array
                continue
            mapping, delta = _map_region(spec.path, spec.offset, spec.nbytes)
            mappings.append(mapping)
            array = np.ndarray(
                spec.shape,
                dtype=_numpy_dtype(np, spec.dtype),
                buffer=mapping,
                offset=delta,
            )
            array.setflags(write=False)
            tensors[spec.name] = array
        return StreamedLayerGroup(planned.public, tensors, mappings)
    except Exception:
        tensors.clear()
        gc.collect()
        for mapping in mappings:
            try:
                mapping.close()
            except BufferError:
                pass
        raise


def _map_region(path: Path, offset: int, length: int) -> tuple[mmap.mmap, int]:
    file_size = path.stat().st_size
    if offset < 0 or length < 0 or offset + length > file_size:
        raise ShardFormatError(f"Mapped tensor range is outside shard {path.name}")
    granularity = mmap.ALLOCATIONGRANULARITY
    map_offset = offset - (offset % granularity)
    delta = offset - map_offset
    map_length = delta + length
    with path.open("rb") as handle:
        mapping = mmap.mmap(
            handle.fileno(),
            length=map_length,
            access=mmap.ACCESS_READ,
            offset=map_offset,
        )
    return mapping, delta


def _discard_mapped_pages(mapping: mmap.mmap) -> None:
    advice = getattr(mmap, "MADV_DONTNEED", None)
    madvise = getattr(mapping, "madvise", None)
    if advice is None or madvise is None:
        return
    try:
        madvise(advice)
    except (OSError, ValueError):
        pass


def _numpy_dtype(np: Any, dtype: str) -> Any:
    if dtype == "BF16":
        try:
            return np.dtype("bfloat16")
        except TypeError:
            try:
                from ml_dtypes import bfloat16
            except ImportError as exc:
                raise ShardFormatError(
                    "BF16 Safetensors require NumPy bfloat16 support or ml_dtypes"
                ) from exc
            return np.dtype(bfloat16)
    safetensors_dtypes = {
        "BOOL": "?",
        "I8": "i1",
        "U8": "u1",
        "I16": "<i2",
        "U16": "<u2",
        "F16": "<f2",
        "I32": "<i4",
        "U32": "<u4",
        "F32": "<f4",
        "I64": "<i8",
        "U64": "<u8",
        "F64": "<f8",
    }
    return np.dtype(safetensors_dtypes.get(dtype, dtype))


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Streaming weight loading requires NumPy. "
            "Install with: pip install 'openmed[onnx-runtime]'"
        ) from exc
    return np


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShardFormatError(f"Invalid JSON metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise ShardFormatError(f"JSON metadata must contain an object: {path}")
    return payload


def _validate_shape(value: Any, tensor_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
        for dimension in value
    ):
        raise ShardFormatError(f"Invalid shape for tensor {tensor_name!r}")
    return tuple(value)


def _shape_size(shape: tuple[int, ...]) -> int:
    return math.prod(shape, start=1)


def _parse_non_negative_int(value: str, name: str, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ShardFormatError(
            f"Invalid ONNX external-data {field} for initializer {name!r}"
        ) from exc
    if parsed < 0:
        raise ShardFormatError(
            f"Invalid ONNX external-data {field} for initializer {name!r}"
        )
    return parsed


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


__all__ = [
    "BufferReleaseError",
    "LayerGroupSpec",
    "LocalWeightsRequired",
    "ShardFormatError",
    "StreamedLayerGroup",
    "StreamingLoadReport",
    "StreamingWeightLoader",
]
