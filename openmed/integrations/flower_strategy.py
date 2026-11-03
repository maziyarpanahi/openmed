"""Optional Flower bridge that gates federated rounds on OpenMed contracts.

The strategy wrapper validates every client fit result against the trusted
:class:`~openmed.training.federated_update_metadata.FederatedUpdatePolicy`,
the policy's schema fingerprint, and a digest over the transported tensor
bytes before it delegates aggregation to an operator-supplied strategy. Any
violation refuses the whole round: the delegate is never called and the global
parameters stay unchanged.

The client wrapper refuses to return a local update that does not declare
clipping, omits an allowlisted parameter, or carries a parameter outside the
allowlist. It attaches validated metadata so the server-side gate can check it.

Flower is imported lazily only when a framework adapter is requested, so this
module imports with the core package. The bridge reads no client identifiers,
never logs tensor values, and does not configure transport security or deploy
the framework; both remain the operator's responsibility.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import logging
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from openmed.training.federated_schema_fingerprint import fingerprint_update_schema
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdateMetadataError,
    FederatedUpdatePolicy,
)

FLOWER_BRIDGE_SCHEMA_VERSION = "openmed.integrations.flower_strategy.v1"
FLOWER_METADATA_METRIC = "openmed.update_metadata"
FLOWER_TENSOR_TYPE = "numpy.ndarray"

FLOWER_REASON_CODES: Final = frozenset(
    {
        "client_evaluate_failed",
        "client_fit_failed",
        "clipping_not_declared",
        "delegate_aggregation_failed",
        "duplicate_update",
        "empty_round",
        "internal_error",
        "invalid_client_update",
        "invalid_configuration",
        "invalid_delegate",
        "invalid_metric_allowlist",
        "invalid_policy",
        "invalid_result",
        "invalid_round",
        "invalid_tensor_payload",
        "metadata_invalid",
        "metadata_missing",
        "model_digest_mismatch",
        "parameter_count_mismatch",
        "parameter_missing",
        "parameter_not_allowlisted",
        "parameter_set_mismatch",
        "parameter_shape_mismatch",
        "round_failures_present",
        "schema_fingerprint_mismatch",
        "serializer_failed",
        "too_many_results",
        "update_digest_mismatch",
    }
)

_INSTALL_HINT: Final = (
    "Flower support requires an operator-installed flwr release; the bridge is "
    "tested with flwr>=1.42,<2. The openmed[flower] extra installs no framework."
)
_PAYLOAD_DOMAIN: Final = b"openmed.integrations.flower_strategy.payload.v1\0"
_ROUND_DOMAIN: Final = b"openmed.integrations.flower_strategy.round.v1\0"
_PLACEHOLDER_DIGEST: Final = "sha256:" + "0" * 64
_MAX_RESULTS = 10_000
_STATUS_METRIC: Final = "openmed.round_status"
_REASON_METRIC: Final = "openmed.reason_code"
# Fixed messages raised by ``FederatedUpdateMetadata`` mapped to bridge codes.
_METADATA_CODES: Final = {
    "update must declare clipping": "clipping_not_declared",
    "parameters do not match policy": "parameter_set_mismatch",
    "model digest does not match policy": "model_digest_mismatch",
}
_LOGGER = logging.getLogger(__name__)


class FlowerBridgeError(RuntimeError):
    """Fail-closed bridge error that carries only a fixed reason code.

    Args:
        code: One of :data:`FLOWER_REASON_CODES`. Unknown values collapse to
            ``internal_error`` so caller data can never become the message.
    """

    def __init__(self, code: str) -> None:
        if type(code) is not str or code not in FLOWER_REASON_CODES:
            code = "internal_error"
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class FlowerRoundReceipt:
    """Content-free transcript entry for one gated fit round.

    Args:
        server_round: Round number supplied by the framework, or ``0`` when the
            framework supplied an invalid value.
        status: ``aggregated`` when the delegate ran, otherwise ``refused``.
        reason_code: Fixed refusal code, or ``None`` for aggregated rounds.
        result_count: Number of fit results presented for the round.
        failure_count: Number of framework-reported failures for the round.
        schema_fingerprint: Trusted policy schema fingerprint.
        updates_root: Order-independent digest over the accepted update
            digests, or ``None`` when the round was refused.
        schema_version: Receipt schema identifier.
    """

    server_round: int
    status: str
    reason_code: str | None
    result_count: int
    failure_count: int
    schema_fingerprint: str
    updates_root: str | None
    schema_version: str = FLOWER_BRIDGE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible mapping of the receipt fields."""
        return {
            "schema_version": self.schema_version,
            "server_round": self.server_round,
            "status": self.status,
            "reason_code": self.reason_code,
            "result_count": self.result_count,
            "failure_count": self.failure_count,
            "schema_fingerprint": self.schema_fingerprint,
            "updates_root": self.updates_root,
        }


@dataclass(frozen=True, slots=True)
class FlowerClientUpdate:
    """A named local update produced by an OpenMed-aware training client.

    Args:
        arrays: Mapping from allowlisted parameter name to an array-like value
            exposing ``shape`` and ``dtype``. Hidden from ``repr``.
        num_examples: Positive number of local training examples.
        clipped: ``True`` only when the local trainer clipped the update.
        metrics: Optional local metrics; only allowlisted numeric values are
            forwarded. Hidden from ``repr``.
    """

    arrays: Mapping[str, Any] = field(repr=False)
    num_examples: int
    clipped: bool
    metrics: Mapping[str, Any] = field(default_factory=dict, repr=False)


class FederatedTrainingClient(Protocol):
    """Protocol for the local trainer wrapped by :class:`OpenMedFlowerClient`."""

    def fit(self, parameters: Any, config: Mapping[str, Any]) -> FlowerClientUpdate:
        """Train locally and return a named, clipped update."""


class FlowerStrategyProtocol(Protocol):
    """Subset of the Flower ``Strategy`` interface the bridge delegates to."""

    def aggregate_fit(
        self, server_round: int, results: list[Any], failures: list[Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Aggregate validated fit results."""


def flower_update_digest(
    parameters: Sequence[FederatedParameterMetadata], tensors: Sequence[bytes]
) -> str:
    """Return the domain-separated digest binding names to tensor bytes.

    Args:
        parameters: Trusted parameters in canonical (name-sorted) order.
        tensors: Serialized tensors in the same order as ``parameters``.

    Returns:
        A ``sha256:`` reference suitable for ``update_digest``.

    Raises:
        FlowerBridgeError: If the sequences differ in length or a tensor is
            not ``bytes``.
    """
    if type(tensors) not in (list, tuple) or len(tensors) != len(parameters):
        raise FlowerBridgeError("parameter_count_mismatch")
    digest = hashlib.sha256(_PAYLOAD_DOMAIN)
    digest.update(len(parameters).to_bytes(4, "big"))
    for parameter, tensor in zip(parameters, tensors):
        if type(parameter) is not FederatedParameterMetadata:
            raise FlowerBridgeError("invalid_policy")
        if type(tensor) is not bytes:
            raise FlowerBridgeError("invalid_tensor_payload")
        name = parameter.name.encode("ascii")
        digest.update(len(name).to_bytes(2, "big"))
        digest.update(name)
        digest.update(len(tensor).to_bytes(8, "big"))
        digest.update(tensor)
    return "sha256:" + digest.hexdigest()


def policy_schema_fingerprint(policy: FederatedUpdatePolicy) -> str:
    """Return the schema fingerprint every update under ``policy`` must have.

    Args:
        policy: Trusted coordinator policy.

    Returns:
        The fingerprint of the policy's model and parameter schema. Update
        digests and clipping status are not part of the schema.

    Raises:
        FlowerBridgeError: If ``policy`` is not a ``FederatedUpdatePolicy``.
    """
    if type(policy) is not FederatedUpdatePolicy:
        raise FlowerBridgeError("invalid_policy")
    reference = FederatedUpdateMetadata(
        model_digest=policy.model_digest,
        adapter_format=policy.adapter_format,
        parameters=policy.parameters,
        total_elements=sum(item.element_count for item in policy.parameters),
        update_digest=_PLACEHOLDER_DIGEST,
        clipped=True,
        policy=policy,
    )
    return fingerprint_update_schema(reference)


class OpenMedFlowerStrategy:
    """Strategy wrapper that validates every fit result before aggregation.

    Each round is checked as a whole. A result must carry metadata under
    :data:`FLOWER_METADATA_METRIC` that validates against the trusted policy,
    has the policy's schema fingerprint, and declares an ``update_digest``
    equal to :func:`flower_update_digest` over the transported tensor bytes.
    One violation refuses the round, the delegate is not called, and the
    framework keeps the previous global parameters.

    Args:
        delegate: Operator-supplied strategy, such as Flower's ``FedAvg``.
        policy: Trusted coordinator policy. It must require clipping.
        expected_schema_fingerprint: Optional independently pinned schema
            fingerprint. It must equal the policy's fingerprint.
        tensor_type: Required ``Parameters.tensor_type`` value, or ``None`` to
            accept any string.
        accept_failures: When ``False`` (default) any framework-reported
            failure refuses the round. When ``True`` failures are counted but
            never forwarded, so unvalidated payloads cannot reach the delegate.
        raise_on_violation: Raise :class:`FlowerBridgeError` instead of
            returning an empty aggregate on refusal.
        max_transcript: Number of most recent receipts retained.
        logger: Optional logger; messages contain only codes and counts.
    """

    def __init__(
        self,
        delegate: FlowerStrategyProtocol,
        *,
        policy: FederatedUpdatePolicy,
        expected_schema_fingerprint: str | None = None,
        tensor_type: str | None = FLOWER_TENSOR_TYPE,
        accept_failures: bool = False,
        raise_on_violation: bool = False,
        max_transcript: int = 1024,
        logger: logging.Logger | None = None,
    ) -> None:
        if type(policy) is not FederatedUpdatePolicy or not policy.require_clipped:
            raise FlowerBridgeError("invalid_policy")
        if not callable(getattr(delegate, "aggregate_fit", None)):
            raise FlowerBridgeError("invalid_delegate")
        if (
            (tensor_type is not None and type(tensor_type) is not str)
            or type(accept_failures) is not bool
            or type(raise_on_violation) is not bool
            or type(max_transcript) is not int
            or not 1 <= max_transcript <= 1_000_000
        ):
            raise FlowerBridgeError("invalid_configuration")
        fingerprint = policy_schema_fingerprint(policy)
        if expected_schema_fingerprint is not None and (
            type(expected_schema_fingerprint) is not str
            or not hmac.compare_digest(expected_schema_fingerprint, fingerprint)
        ):
            raise FlowerBridgeError("schema_fingerprint_mismatch")
        self._delegate = delegate
        self._policy = policy
        self._fingerprint = fingerprint
        self._tensor_type = tensor_type
        self._accept_failures = accept_failures
        self._raise = raise_on_violation
        self._transcript: deque[FlowerRoundReceipt] = deque(maxlen=max_transcript)
        self._logger = logger or _LOGGER

    def __repr__(self) -> str:
        return f"OpenMedFlowerStrategy(schema_fingerprint={self._fingerprint!r})"

    @property
    def schema_fingerprint(self) -> str:
        """Return the trusted schema fingerprint enforced for every result."""
        return self._fingerprint

    @property
    def transcript(self) -> tuple[FlowerRoundReceipt, ...]:
        """Return the retained content-free round receipts, oldest first."""
        return tuple(self._transcript)

    def aggregate_fit(
        self, server_round: int, results: Any, failures: Any
    ) -> tuple[Any, dict[str, Any]]:
        """Validate a whole fit round, then delegate aggregation.

        Args:
            server_round: Positive round number from the framework.
            results: ``(client_proxy, fit_res)`` pairs. The proxy is never read.
            failures: Framework-reported failures for the round.

        Returns:
            The delegate's ``(parameters, metrics)`` result, or ``(None,
            {...})`` with a fixed refusal code when the round is refused.

        Raises:
            FlowerBridgeError: On refusal when ``raise_on_violation`` is set.
        """
        result_count = len(results) if type(results) in (list, tuple) else 0
        failure_count = len(failures) if type(failures) in (list, tuple) else 0
        round_number = (
            server_round if type(server_round) is int and server_round > 0 else 0
        )
        try:
            digests = self._validate_round(server_round, results, failures)
        except FlowerBridgeError as error:
            return self._refuse(round_number, error.code, result_count, failure_count)
        try:
            outcome = self._delegate.aggregate_fit(server_round, list(results), [])
        except Exception:
            outcome = None
        if type(outcome) is not tuple or len(outcome) != 2:
            return self._refuse(
                round_number,
                "delegate_aggregation_failed",
                result_count,
                failure_count,
            )
        self._record(
            FlowerRoundReceipt(
                server_round=round_number,
                status="aggregated",
                reason_code=None,
                result_count=result_count,
                failure_count=failure_count,
                schema_fingerprint=self._fingerprint,
                updates_root=_updates_root(digests),
            )
        )
        return outcome

    def initialize_parameters(self, client_manager: Any) -> Any:
        """Forward initial parameter selection to the delegate."""
        return self._delegate.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Any, client_manager: Any
    ) -> Any:
        """Forward fit configuration to the delegate."""
        return self._delegate.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Any, client_manager: Any
    ) -> Any:
        """Forward evaluation configuration to the delegate."""
        return self._delegate.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_evaluate(self, server_round: int, results: Any, failures: Any) -> Any:
        """Forward evaluation aggregation; evaluation carries no model update."""
        return self._delegate.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round: int, parameters: Any) -> Any:
        """Forward centralized evaluation to the delegate."""
        return self._delegate.evaluate(server_round, parameters)

    def as_flower_strategy(self) -> Any:
        """Return a ``flwr.server.strategy.Strategy`` subclass instance.

        Returns:
            A Flower strategy whose methods call this bridge.

        Raises:
            ImportError: If no compatible ``flwr`` release is installed.
        """
        base = _load_flower_attribute("flwr.server.strategy", "Strategy")
        bridge = self

        class _OpenMedStrategy(base):  # type: ignore[misc, valid-type]
            def __repr__(self) -> str:
                return repr(bridge)

            def initialize_parameters(self, client_manager):
                return bridge.initialize_parameters(client_manager)

            def configure_fit(self, server_round, parameters, client_manager):
                return bridge.configure_fit(server_round, parameters, client_manager)

            def aggregate_fit(self, server_round, results, failures):
                return bridge.aggregate_fit(server_round, results, failures)

            def configure_evaluate(self, server_round, parameters, client_manager):
                return bridge.configure_evaluate(
                    server_round, parameters, client_manager
                )

            def aggregate_evaluate(self, server_round, results, failures):
                return bridge.aggregate_evaluate(server_round, results, failures)

            def evaluate(self, server_round, parameters):
                return bridge.evaluate(server_round, parameters)

        return _OpenMedStrategy()

    def _validate_round(
        self, server_round: Any, results: Any, failures: Any
    ) -> list[str]:
        if type(server_round) is not int or server_round < 1:
            raise FlowerBridgeError("invalid_round")
        if type(results) not in (list, tuple) or type(failures) not in (list, tuple):
            raise FlowerBridgeError("invalid_result")
        if not results:
            raise FlowerBridgeError("empty_round")
        if len(results) > _MAX_RESULTS:
            raise FlowerBridgeError("too_many_results")
        if failures and not self._accept_failures:
            raise FlowerBridgeError("round_failures_present")
        digests: list[str] = []
        for item in results:
            if type(item) not in (list, tuple) or len(item) != 2:
                raise FlowerBridgeError("invalid_result")
            digests.append(self._validate_result(item[1]))
        if len(set(digests)) != len(digests):
            raise FlowerBridgeError("duplicate_update")
        return digests

    def _validate_result(self, fit_res: Any) -> str:
        try:
            return self._check_result(fit_res)
        except FlowerBridgeError:
            raise
        except Exception:
            raise FlowerBridgeError("invalid_result") from None

    def _check_result(self, fit_res: Any) -> str:
        metrics = getattr(fit_res, "metrics", None)
        if type(metrics) is not dict:
            raise FlowerBridgeError("invalid_result")
        payload = metrics.get(FLOWER_METADATA_METRIC)
        if type(payload) is not str:
            raise FlowerBridgeError("metadata_missing")
        try:
            metadata = FederatedUpdateMetadata.from_json(payload, policy=self._policy)
        except FederatedUpdateMetadataError as error:
            code = _METADATA_CODES.get(str(error), "metadata_invalid")
            raise FlowerBridgeError(code) from None
        if not hmac.compare_digest(
            fingerprint_update_schema(metadata), self._fingerprint
        ):
            raise FlowerBridgeError("schema_fingerprint_mismatch")
        parameters = getattr(fit_res, "parameters", None)
        tensors = getattr(parameters, "tensors", None)
        tensor_type = getattr(parameters, "tensor_type", None)
        if type(tensors) not in (list, tuple) or type(tensor_type) is not str:
            raise FlowerBridgeError("invalid_tensor_payload")
        if self._tensor_type is not None and tensor_type != self._tensor_type:
            raise FlowerBridgeError("invalid_tensor_payload")
        if len(tensors) != len(self._policy.parameters):
            raise FlowerBridgeError("parameter_count_mismatch")
        observed = flower_update_digest(self._policy.parameters, tensors)
        if not hmac.compare_digest(observed, metadata.update_digest):
            raise FlowerBridgeError("update_digest_mismatch")
        num_examples = getattr(fit_res, "num_examples", None)
        if type(num_examples) is not int or num_examples < 1:
            raise FlowerBridgeError("invalid_result")
        return metadata.update_digest

    def _refuse(
        self, server_round: int, code: str, result_count: int, failure_count: int
    ) -> tuple[None, dict[str, Any]]:
        self._record(
            FlowerRoundReceipt(
                server_round=server_round,
                status="refused",
                reason_code=code,
                result_count=result_count,
                failure_count=failure_count,
                schema_fingerprint=self._fingerprint,
                updates_root=None,
            )
        )
        if self._raise:
            raise FlowerBridgeError(code) from None
        return None, {_STATUS_METRIC: "refused", _REASON_METRIC: code}

    def _record(self, receipt: FlowerRoundReceipt) -> None:
        self._transcript.append(receipt)
        level = logging.INFO if receipt.reason_code is None else logging.WARNING
        self._logger.log(
            level,
            "OpenMed Flower round %d %s reason=%s results=%d failures=%d",
            receipt.server_round,
            receipt.status,
            receipt.reason_code or "none",
            receipt.result_count,
            receipt.failure_count,
        )


class OpenMedFlowerClient:
    """Client wrapper that only returns clipped, allowlisted updates.

    The wrapped trainer returns a :class:`FlowerClientUpdate` with arrays keyed
    by parameter name. The wrapper refuses the update unless clipping is
    declared, every allowlisted parameter is present with the trusted shape and
    dtype, and no other parameter is present. Accepted updates are returned in
    the policy's canonical order with validated metadata attached under
    :data:`FLOWER_METADATA_METRIC`.

    Args:
        delegate: Local OpenMed-aware trainer.
        policy: Trusted coordinator policy. It must require clipping.
        serializer: Callable that turns one array into the exact bytes the
            framework transports. Defaults to Flower's ``ndarray_to_bytes``.
        metric_allowlist: Metric names forwarded from the trainer. Only
            ``bool``, ``int`` and ``float`` values are forwarded.
    """

    def __init__(
        self,
        delegate: FederatedTrainingClient,
        *,
        policy: FederatedUpdatePolicy,
        serializer: Callable[[Any], bytes] | None = None,
        metric_allowlist: Iterable[str] = (),
    ) -> None:
        if type(policy) is not FederatedUpdatePolicy or not policy.require_clipped:
            raise FlowerBridgeError("invalid_policy")
        if not callable(getattr(delegate, "fit", None)):
            raise FlowerBridgeError("invalid_delegate")
        if serializer is not None and not callable(serializer):
            raise FlowerBridgeError("invalid_configuration")
        self._delegate = delegate
        self._policy = policy
        self._serializer = serializer
        self._metric_allowlist = _metric_allowlist(metric_allowlist)

    def __repr__(self) -> str:
        return "OpenMedFlowerClient()"

    def get_parameters(self, config: Mapping[str, Any]) -> Any:
        """Forward the initial parameter request to the wrapped trainer."""
        return self._delegate.get_parameters(config)

    def fit(
        self, parameters: Any, config: Mapping[str, Any]
    ) -> tuple[list[Any], int, dict[str, Any]]:
        """Train locally and return only a validated, clipped update.

        Args:
            parameters: Global parameters supplied by the framework.
            config: Fit configuration supplied by the framework.

        Returns:
            Arrays in canonical parameter order, the example count, and
            metrics containing the validated metadata plus allowlisted values.

        Raises:
            FlowerBridgeError: If the update is not clipped, not allowlisted,
                malformed, or the trainer or serializer fails.
            ImportError: If no serializer was supplied and Flower is missing.
        """
        try:
            update = self._delegate.fit(parameters, config)
        except Exception:
            raise FlowerBridgeError("client_fit_failed") from None
        if type(update) is not FlowerClientUpdate:
            raise FlowerBridgeError("invalid_client_update")
        if update.clipped is not True:
            raise FlowerBridgeError("clipping_not_declared")
        arrays = update.arrays
        if not isinstance(arrays, Mapping) or any(
            type(name) is not str for name in arrays
        ):
            raise FlowerBridgeError("invalid_client_update")
        expected = {parameter.name for parameter in self._policy.parameters}
        names = set(arrays)
        if names - expected:
            raise FlowerBridgeError("parameter_not_allowlisted")
        if expected - names:
            raise FlowerBridgeError("parameter_missing")
        if type(update.num_examples) is not int or update.num_examples < 1:
            raise FlowerBridgeError("invalid_client_update")
        ordered = [arrays[parameter.name] for parameter in self._policy.parameters]
        for parameter, array in zip(self._policy.parameters, ordered):
            _check_array(parameter, array)
        serializer = self._serializer or _load_flower_attribute(
            "flwr.common", "ndarray_to_bytes"
        )
        blobs = [_serialize(serializer, array) for array in ordered]
        try:
            metadata = FederatedUpdateMetadata(
                model_digest=self._policy.model_digest,
                adapter_format=self._policy.adapter_format,
                parameters=self._policy.parameters,
                total_elements=sum(
                    item.element_count for item in self._policy.parameters
                ),
                update_digest=flower_update_digest(self._policy.parameters, blobs),
                clipped=True,
                policy=self._policy,
            )
        except FederatedUpdateMetadataError:
            raise FlowerBridgeError("invalid_client_update") from None
        metrics = self._filter_metrics(update.metrics)
        metrics[FLOWER_METADATA_METRIC] = metadata.to_json()
        return ordered, update.num_examples, metrics

    def evaluate(
        self, parameters: Any, config: Mapping[str, Any]
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Forward evaluation and keep only allowlisted numeric metrics.

        Args:
            parameters: Global parameters supplied by the framework.
            config: Evaluation configuration supplied by the framework.

        Returns:
            The trainer's loss and example count with filtered metrics.

        Raises:
            FlowerBridgeError: If the trainer fails or returns a malformed value.
        """
        try:
            outcome = self._delegate.evaluate(parameters, config)
        except Exception:
            raise FlowerBridgeError("client_evaluate_failed") from None
        if type(outcome) is not tuple or len(outcome) != 3:
            raise FlowerBridgeError("client_evaluate_failed")
        loss, num_examples, metrics = outcome
        return loss, num_examples, self._filter_metrics(metrics)

    def as_flower_client(self) -> Any:
        """Return a ``flwr.client.NumPyClient`` subclass instance.

        Returns:
            A Flower NumPy client whose methods call this bridge; call
            ``to_client()`` on it where the framework expects a ``Client``.

        Raises:
            ImportError: If no compatible ``flwr`` release is installed.
        """
        base = _load_flower_attribute("flwr.client", "NumPyClient")
        bridge = self

        class _OpenMedClient(base):  # type: ignore[misc, valid-type]
            def __repr__(self) -> str:
                return repr(bridge)

            def get_parameters(self, config):
                return bridge.get_parameters(config)

            def fit(self, parameters, config):
                return bridge.fit(parameters, config)

            def evaluate(self, parameters, config):
                return bridge.evaluate(parameters, config)

        return _OpenMedClient()

    def _filter_metrics(self, metrics: Any) -> dict[str, Any]:
        if not isinstance(metrics, Mapping):
            return {}
        return {
            key: value
            for key, value in metrics.items()
            if type(key) is str
            and key in self._metric_allowlist
            and type(value) in (bool, int, float)
        }


def _metric_allowlist(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise FlowerBridgeError("invalid_metric_allowlist")
    try:
        names = tuple(values)
    except TypeError:
        raise FlowerBridgeError("invalid_metric_allowlist") from None
    if len(names) > 256 or any(
        type(name) is not str
        or not 1 <= len(name) <= 128
        or name in (FLOWER_METADATA_METRIC, _STATUS_METRIC, _REASON_METRIC)
        for name in names
    ):
        raise FlowerBridgeError("invalid_metric_allowlist")
    return frozenset(names)


def _check_array(parameter: FederatedParameterMetadata, array: Any) -> None:
    try:
        shape = tuple(array.shape)
        dtype = str(array.dtype)
    except Exception:
        raise FlowerBridgeError("invalid_tensor_payload") from None
    if shape != parameter.shape or dtype != parameter.dtype:
        raise FlowerBridgeError("parameter_shape_mismatch")


def _serialize(serializer: Callable[[Any], bytes], array: Any) -> bytes:
    try:
        blob = serializer(array)
    except Exception:
        raise FlowerBridgeError("serializer_failed") from None
    if type(blob) is not bytes:
        raise FlowerBridgeError("serializer_failed")
    return blob


def _updates_root(digests: Sequence[str]) -> str:
    digest = hashlib.sha256(_ROUND_DOMAIN)
    for value in sorted(digests):
        digest.update(value.encode("ascii") + b"\n")
    return "sha256:" + digest.hexdigest()


def _load_flower_attribute(module_name: str, attribute: str) -> Any:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (ImportError, AttributeError):
        raise ImportError(_INSTALL_HINT) from None


__all__ = [
    "FLOWER_BRIDGE_SCHEMA_VERSION",
    "FLOWER_METADATA_METRIC",
    "FLOWER_REASON_CODES",
    "FLOWER_TENSOR_TYPE",
    "FederatedTrainingClient",
    "FlowerBridgeError",
    "FlowerClientUpdate",
    "FlowerRoundReceipt",
    "FlowerStrategyProtocol",
    "OpenMedFlowerClient",
    "OpenMedFlowerStrategy",
    "flower_update_digest",
    "policy_schema_fingerprint",
]
