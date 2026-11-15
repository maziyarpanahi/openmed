"""Standard gRPC health status bound to explicit preload and drain lifecycle."""

from __future__ import annotations

import threading
from typing import Any

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc


class ReadinessGrpcServer(grpc.Server):
    """Delegate listener credentials while owning status-only health lifecycle.

    The wrapper registers the standard health protocol for overall status and
    the existing fully qualified OpenMed service. It does not load models in
    its constructor, enable reflection or coordinate runtime request admission.
    """

    def __init__(self, server: grpc.Server, runtime: Any, service_name: str) -> None:
        self._server = server
        self._runtime = runtime
        self._state_lock = threading.Lock()
        self._preload_lock = threading.Lock()
        self._ready = False
        self._draining = False
        self._health = health.HealthServicer()
        self._services = ("", service_name)
        for name in self._services:
            self._health.set(name, health_pb2.HealthCheckResponse.NOT_SERVING)
        health_pb2_grpc.add_HealthServicer_to_server(self._health, self._server)

    def preload_runtime(self) -> None:
        """Call actual configured runtime preload; serving follows success only.

        Concurrent callers share one successful preload. Drain is terminal and
        cannot be reversed by a preload completing later. Failed preload leaves
        health not-serving and exposes only a controlled error without its cause.
        """
        with self._preload_lock:
            with self._state_lock:
                if self._draining:
                    raise RuntimeError("grpc_runtime_draining")
                if self._ready:
                    return
            failed = False
            try:
                self._runtime.preload()
            except Exception:
                failed = True
            if failed:
                # Raise outside the except block so neither cause nor context
                # retains input-bearing runtime failure detail.
                raise RuntimeError("grpc_preload_failed")
            with self._state_lock:
                if not self._draining:
                    self._ready = True
                    for name in self._services:
                        self._health.set(name, health_pb2.HealthCheckResponse.SERVING)

    def begin_drain(self) -> None:
        """Publish terminal not-serving before a host drains requests or stops.

        Coordinated runtime admission and lease draining belong to the host; this
        is the health notification seam and does not execute clinical work.
        """
        with self._state_lock:
            self._draining = True
            self._ready = False
            self._health.enter_graceful_shutdown()

    def add_generic_rpc_handlers(self, generic_rpc_handlers):
        """Delegate RPC registration without changing existing listener policy."""
        return self._server.add_generic_rpc_handlers(generic_rpc_handlers)

    def add_registered_method_handlers(self, service_name, method_handlers):
        """Delegate registered methods to the original grpcio server."""
        return self._server.add_registered_method_handlers(
            service_name, method_handlers
        )

    def add_insecure_port(self, address):
        """Delegate an explicitly configured insecure listener unchanged."""
        return self._server.add_insecure_port(address)

    def add_secure_port(self, address, server_credentials):
        """Delegate TLS credentials unchanged; never introduce an insecure fallback."""
        return self._server.add_secure_port(address, server_credentials)

    def start(self):
        """Start listeners without claiming runtime preload has completed."""
        return self._server.start()

    def stop(self, grace):
        """Publish drain status before delegating the standard grpcio stop call."""
        self.begin_drain()
        return self._server.stop(grace)

    def wait_for_termination(self, timeout=None):
        """Delegate standard termination waiting."""
        return self._server.wait_for_termination(timeout)
