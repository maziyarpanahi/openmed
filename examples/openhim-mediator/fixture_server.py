"""Synthetic OpenHIM-core handshake fixture for container smoke tests."""

from __future__ import annotations

import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

USERNAME = os.environ.get("OPENHIM_FIXTURE_USERNAME", "openhim@example.org")
PASSWORD = os.environ.get("OPENHIM_FIXTURE_PASSWORD", "synthetic-password")
EXPECTED_AUTH = "Basic " + base64.b64encode(
    f"{USERNAME}:{PASSWORD}".encode("utf-8")
).decode("ascii")
STATE = {"registration_count": 0, "heartbeat_count": 0, "urn": None}


class FixtureHandler(BaseHTTPRequestHandler):
    """Serve only the OpenHIM endpoints exercised by the mediator client."""

    server_version = "OpenHIMFixture/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == f"/authenticate/{USERNAME.replace('@', '%40')}":
            self._json(200, {"salt": "synthetic", "ts": "2026-01-01T00:00:00Z"})
            return
        if self.path == "/health":
            self._json(200, {"status": "ok", **STATE})
            return
        self._json(404, {"status": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.headers.get("Authorization") != EXPECTED_AUTH:
            self._json(401, {"status": "unauthorized"})
            return
        try:
            payload = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
        except json.JSONDecodeError:
            self._json(400, {"status": "invalid_json"})
            return

        if self.path == "/mediators":
            STATE["registration_count"] += 1
            STATE["urn"] = payload.get("urn")
            self._json(201, {"registered": True})
            return

        if self.path.startswith("/mediators/") and self.path.endswith("/heartbeat"):
            encoded_urn = self.path.removeprefix("/mediators/").removesuffix(
                "/heartbeat"
            )
            if unquote(encoded_urn) != STATE["urn"]:
                self._json(404, {"status": "unknown_mediator"})
                return
            STATE["heartbeat_count"] += 1
            self._json(200, {"config": {}} if payload.get("config") else {})
            return

        self._json(404, {"status": "not_found"})

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8081), FixtureHandler).serve_forever()
