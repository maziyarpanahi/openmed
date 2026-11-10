"""Loopback-only recorded OpenMRS FHIR2 response server for the demo."""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

_FHIR_TYPES = frozenset({"Patient", "Encounter", "Observation"})
_REST_TYPES = frozenset({"patient", "encounter", "obs"})


@contextmanager
def openmrs_fixture_server(
    fhir_resources: Mapping[str, Sequence[Mapping[str, Any]]],
    rest_resources: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Iterator[str]:
    """Serve deterministic OpenMRS REST and FHIR2 pages on loopback."""

    fhir_recording = {
        resource_type: tuple(copy.deepcopy(list(items)))
        for resource_type, items in fhir_resources.items()
    }
    rest_recording = {
        resource_type: tuple(copy.deepcopy(list(items)))
        for resource_type, items in rest_resources.items()
    }
    handler = _handler_for(fhir_recording, rest_recording)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="openmrs-fixture",
        daemon=True,
    )
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/openmrs"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _handler_for(
    fhir_recording: Mapping[str, Sequence[Mapping[str, Any]]],
    rest_recording: Mapping[str, Sequence[Mapping[str, Any]]],
) -> type[BaseHTTPRequestHandler]:
    class FixtureHandler(BaseHTTPRequestHandler):
        server_version = "OpenMRSFixture/1.0"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            fhir_prefix = "/openmrs/ws/fhir2/R4/"
            rest_prefix = "/openmrs/ws/rest/v1/"
            if parsed.path.startswith(fhir_prefix):
                resource_type = parsed.path.removeprefix(fhir_prefix)
                if resource_type not in _FHIR_TYPES:
                    self._json(404, {"error": "unsupported_resource"})
                    return
                self._fhir_search(
                    parsed.path,
                    parse_qs(parsed.query),
                    resource_type,
                )
                return
            if parsed.path.startswith(rest_prefix):
                resource_type = parsed.path.removeprefix(rest_prefix)
                if resource_type not in _REST_TYPES:
                    self._json(404, {"error": "unsupported_resource"})
                    return
                self._rest_search(parse_qs(parsed.query), resource_type)
                return
            self._json(404, {"error": "not_found"})

        def _fhir_search(
            self,
            path: str,
            query: Mapping[str, list[str]],
            resource_type: str,
        ) -> None:
            page_size = _positive_int(query.get("_count", ["25"])[0], default=25)
            page = _positive_int(query.get("page", ["1"])[0], default=1)
            resources = list(fhir_recording.get(resource_type, ()))
            start = (page - 1) * page_size
            selected = resources[start : start + page_size]
            base_url = (
                f"http://{self.server.server_address[0]}:"
                f"{self.server.server_address[1]}"
            )
            search_url = f"{base_url}{path}"
            bundle: dict[str, Any] = {
                "resourceType": "Bundle",
                "type": "searchset",
                "total": len(resources),
                "entry": [
                    {
                        "fullUrl": f"{search_url}/{resource['id']}",
                        "resource": resource,
                    }
                    for resource in selected
                ],
                "link": [
                    {
                        "relation": "self",
                        "url": f"{search_url}?{urlencode({'_count': page_size, 'page': page})}",
                    }
                ],
            }
            if start + page_size < len(resources):
                bundle["link"].append(
                    {
                        "relation": "next",
                        "url": (
                            f"{search_url}?"
                            f"{urlencode({'_count': page_size, 'page': page + 1})}"
                        ),
                    }
                )
            self._json(200, bundle)

        def _rest_search(
            self,
            query: Mapping[str, list[str]],
            resource_type: str,
        ) -> None:
            page_size = _positive_int(query.get("limit", ["25"])[0], default=25)
            start = _nonnegative_int(
                query.get("startIndex", ["0"])[0],
                default=0,
            )
            resources = list(rest_recording.get(resource_type, ()))
            self._json(
                200,
                {
                    "results": resources[start : start + page_size],
                    "startIndex": start,
                },
            )

        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/fhir+json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            return

    return FixtureHandler


def _positive_int(value: str, *, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: str, *, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default
