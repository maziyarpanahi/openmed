#!/usr/bin/env python3
"""Check browser request traces for unexpected network egress.

The checker is deliberately transport-free. It observes request objects or a
JSON trace supplied by the caller; it never opens a socket or resolves a
host. Model assets are the only network requests that can pass, and only when
their URL is explicitly configured by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import SplitResult, unquote, urlsplit, urlunsplit

NETWORK_SCHEMES = frozenset({"ftp", "http", "https", "ws", "wss"})
NON_NETWORK_SCHEMES = frozenset({"about", "blob", "data"})
_MODEL_ASSET_SCHEMES = frozenset({"http", "https"})
_MAX_MODEL_ASSET_PATTERNS = 256
_MAX_REQUESTS = 10_000
_MAX_TRACE_BYTES = 8 * 1024 * 1024
_MAX_URL_LENGTH = 8_192
_HTTP_METHODS = frozenset(
    {"CONNECT", "DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "TRACE"}
)
_RESOURCE_TYPES = frozenset(
    {
        "document",
        "eventsource",
        "fetch",
        "font",
        "image",
        "manifest",
        "media",
        "other",
        "script",
        "stylesheet",
        "texttrack",
        "websocket",
        "xhr",
    }
)
_CLASSIFICATIONS = frozenset(
    {"browser-internal", "invalid-url", "model-asset", "unexpected-network"}
)
_REPORT_SCHEMES = NETWORK_SCHEMES | NON_NETWORK_SCHEMES | {"unknown"}


@dataclass(frozen=True, slots=True)
class _ParsedNetworkURL:
    """Canonical URL data kept private so raw URLs cannot enter reports."""

    canonical: str
    origin: str
    path: str
    query: str
    scheme: str


@dataclass(frozen=True, slots=True)
class _ObservedRequest:
    """Raw request data retained only long enough to classify one event."""

    method: str
    resource_type: str
    url: str


@dataclass(frozen=True, slots=True)
class RequestSummary:
    """A privacy-safe summary of one observed browser request.

    Paths, query strings, fragments, headers, and request bodies are never
    retained here. The two digests let an operator correlate repeated events
    without putting a potentially sensitive URL in an exception or report.
    """

    index: int
    method: str
    resource_type: str
    scheme: str
    url_digest: str
    origin_digest: str
    classification: str

    def __post_init__(self) -> None:
        """Reject caller-supplied values that could make a report unsafe."""

        digest_fields = (self.url_digest, self.origin_digest)
        if (
            type(self.index) is not int
            or self.index < 0
            or type(self.method) is not str
            or self.method not in _HTTP_METHODS | {"UNKNOWN"}
            or type(self.resource_type) is not str
            or self.resource_type not in _RESOURCE_TYPES
            or type(self.scheme) is not str
            or self.scheme not in _REPORT_SCHEMES
            or type(self.classification) is not str
            or self.classification not in _CLASSIFICATIONS
            or any(
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in digest_fields
            )
        ):
            raise ValueError("request summary contains invalid safe metadata")

    @property
    def is_unexpected(self) -> bool:
        """Return whether the request violates the configured policy."""

        return self.classification in {"invalid-url", "unexpected-network"}

    def to_dict(self) -> dict[str, object]:
        """Return the stable, raw-value-free representation for reports."""

        return {
            "classification": self.classification,
            "index": self.index,
            "method": self.method,
            "origin_digest": self.origin_digest,
            "resource_type": self.resource_type,
            "scheme": self.scheme,
            "url_digest": self.url_digest,
        }


@dataclass(frozen=True, slots=True)
class EgressReport:
    """Deterministic result of checking a browser request sequence."""

    requests: tuple[RequestSummary, ...]

    def __post_init__(self) -> None:
        """Require a complete, immutable sequence of validated summaries."""

        if type(self.requests) is not tuple or any(
            type(request) is not RequestSummary or request.index != index
            for index, request in enumerate(self.requests)
        ):
            raise ValueError("egress report contains invalid request summaries")

    @property
    def request_count(self) -> int:
        """Return the number of observed request events."""

        return len(self.requests)

    @property
    def network_request_count(self) -> int:
        """Return the number of requests that used a network-like scheme."""

        return sum(
            request.classification
            in {"model-asset", "unexpected-network", "invalid-url"}
            for request in self.requests
        )

    @property
    def allowed_model_asset_count(self) -> int:
        """Return the number of explicitly allowlisted model asset requests."""

        return sum(request.classification == "model-asset" for request in self.requests)

    @property
    def unexpected_requests(self) -> tuple[RequestSummary, ...]:
        """Return the privacy-safe summaries that failed the policy."""

        return tuple(request for request in self.requests if request.is_unexpected)

    @property
    def unexpected_request_count(self) -> int:
        """Return the number of unexpected network requests."""

        return len(self.unexpected_requests)

    @property
    def passed(self) -> bool:
        """Return whether every network request was explicitly permitted."""

        return not self.unexpected_requests

    def to_dict(self) -> dict[str, object]:
        """Return a stable report that contains no raw request values."""

        return {
            "allowed_model_asset_count": self.allowed_model_asset_count,
            "network_request_count": self.network_request_count,
            "passed": self.passed,
            "request_count": self.request_count,
            "requests": [request.to_dict() for request in self.requests],
            "schema_version": 1,
            "unexpected_request_count": self.unexpected_request_count,
        }

    def to_json(self) -> str:
        """Return the canonical JSON report representation."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def assert_clean(self) -> EgressReport:
        """Raise a safe exception when an unexpected request was observed."""

        if not self.passed:
            raise NetworkEgressViolation(self)
        return self


class NetworkEgressViolation(AssertionError):
    """Raised when browser egress is outside the explicit model-asset policy."""

    def __init__(self, report: EgressReport) -> None:
        self.report = report
        super().__init__(
            "Unexpected browser network egress detected: "
            f"{report.unexpected_request_count} request(s) were not in the "
            "explicit model-asset allowlist."
        )


class NetworkEgressProbe:
    """Record browser request events and assert a local-first egress policy.

    The object works as a Playwright-style ``page.on("request", callback)``
    callback without importing Playwright. It can therefore be used by a
    browser test with the optional browser dependency installed, or with a
    small synthetic page in an offline unit test.
    """

    def __init__(self, *, allowed_model_assets: Iterable[str] | str = ()) -> None:
        self._allowed_model_assets = _normalise_model_asset_patterns(
            allowed_model_assets
        )
        self._requests: list[RequestSummary] = []
        self._page: Any | None = None
        self._listener = self.record

    @property
    def request_count(self) -> int:
        """Return the number of events recorded so far."""

        return len(self._requests)

    def record(self, request: Any) -> None:
        """Record one browser request event without inspecting its payload.

        ``request`` may be a Playwright request object, a mapping containing a
        ``url`` field, or a URL string. Headers and bodies are intentionally
        ignored so a trace cannot accidentally collect sensitive data.
        """

        if len(self._requests) >= _MAX_REQUESTS:
            raise ValueError("request trace exceeds the supported event count")
        observed = _coerce_request(request)
        self._requests.append(
            _summarise_request(
                len(self._requests),
                observed,
                self._allowed_model_assets,
            )
        )

    def report(self) -> EgressReport:
        """Classify all recorded events into a deterministic safe report."""

        return EgressReport(requests=tuple(self._requests))

    def assert_clean(self) -> EgressReport:
        """Assert that recorded requests contain no unexpected network call."""

        return self.report().assert_clean()

    def attach(self, page: Any) -> NetworkEgressProbe:
        """Attach this probe to a Playwright-style page request event."""

        if self._page is not None:
            raise RuntimeError("network egress probe is already attached")
        on = getattr(page, "on", None)
        if not callable(on):
            raise TypeError("browser page does not expose an event subscription API")
        on("request", self._listener)
        self._page = page
        return self

    def detach(self) -> None:
        """Detach this probe from its page, if it is attached."""

        page = self._page
        if page is None:
            return
        remove_listener = getattr(page, "remove_listener", None)
        if not callable(remove_listener):
            remove_listener = getattr(page, "off", None)
        if not callable(remove_listener):
            raise TypeError("browser page does not expose an event removal API")
        remove_listener("request", self._listener)
        self._page = None


@contextmanager
def capture_browser_requests(
    page: Any,
    *,
    allowed_model_assets: Iterable[str] | str = (),
) -> Iterator[NetworkEgressProbe]:
    """Capture request events from ``page`` for one synthetic browser action."""

    probe = NetworkEgressProbe(allowed_model_assets=allowed_model_assets)
    probe.attach(page)
    try:
        yield probe
    finally:
        probe.detach()


def check_network_egress(
    requests: Iterable[Any],
    *,
    allowed_model_assets: Iterable[str] | str = (),
) -> EgressReport:
    """Check supplied browser requests without making any network call."""

    probe = NetworkEgressProbe(allowed_model_assets=allowed_model_assets)
    try:
        iterator = iter(requests)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ValueError("request trace could not be read") from None
    while True:
        try:
            request = next(iterator)
        except StopIteration:
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("request trace could not be read") from None
        probe.record(request)
    return probe.report()


def assert_no_unexpected_requests(
    requests: Iterable[Any],
    *,
    allowed_model_assets: Iterable[str] | str = (),
) -> EgressReport:
    """Check requests and raise if any network request is not allowlisted."""

    return check_network_egress(
        requests,
        allowed_model_assets=allowed_model_assets,
    ).assert_clean()


def _coerce_request(request: Any) -> _ObservedRequest:
    if isinstance(request, str):
        url: Any = request
        method: Any = "GET"
        resource_type: Any = "other"
    else:
        url = _request_value(request, "url")
        method = _request_value(request, "method")
        resource_type = _request_value(request, "resource_type")

    plain_url = _plain_text(url)
    candidate = plain_url.strip() if plain_url is not None else ""
    if len(candidate) > _MAX_URL_LENGTH:
        candidate = ""

    return _ObservedRequest(
        method=_normalise_method(method),
        resource_type=_normalise_resource_type(resource_type),
        url=candidate,
    )


def _request_value(request: Any, name: str) -> Any:
    try:
        if isinstance(request, Mapping):
            value = request.get(name)
        else:
            value = getattr(request, name, None)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return None
    return None if callable(value) else value


def _normalise_method(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    plain_value = _plain_text(value)
    if plain_value is None:
        return "UNKNOWN"
    candidate = plain_value.strip().upper()
    return candidate if candidate in _HTTP_METHODS else "UNKNOWN"


def _normalise_resource_type(value: Any) -> str:
    plain_value = _plain_text(value)
    if plain_value is None:
        return "other"
    candidate = plain_value.strip().casefold()
    return candidate if candidate in _RESOURCE_TYPES else "other"


def _plain_text(value: Any) -> str | None:
    """Copy a string without executing hooks on a string subclass."""

    if not isinstance(value, str):
        return None
    try:
        return str.encode(value, "utf-8").decode("utf-8")
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return None


def _normalise_model_asset_patterns(
    patterns: Iterable[str] | str,
) -> tuple[str, ...]:
    if isinstance(patterns, str):
        values: Iterator[object] = iter((patterns,))
    else:
        try:
            values = iter(patterns)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError(
                "model asset allowlist must be an iterable of URLs"
            ) from None

    normalised: list[str] = []
    seen: set[str] = set()
    for index in range(_MAX_MODEL_ASSET_PATTERNS + 1):
        try:
            pattern = next(values)
        except StopIteration:
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("model asset allowlist could not be read") from None
        if index == _MAX_MODEL_ASSET_PATTERNS:
            raise ValueError("model asset allowlist exceeds the supported entry count")
        plain_pattern = _plain_text(pattern)
        if plain_pattern is None or not plain_pattern.strip():
            raise ValueError("model asset allowlist entries must be non-empty URLs")
        candidate = plain_pattern.strip()
        if len(candidate) > _MAX_URL_LENGTH:
            raise ValueError("model asset allowlist entries exceed the URL size limit")
        if "*" in candidate:
            raise ValueError("model asset allowlist entries cannot use wildcards")
        parsed = _parse_network_url(candidate, model_asset=True)
        if parsed is None or "#" in candidate:
            raise ValueError(
                "model asset allowlist entries must be absolute HTTP(S) URLs"
            )
        if parsed.path == "/":
            raise ValueError("model asset directory prefixes must include a path")
        if parsed.path.endswith("/") and parsed.query:
            raise ValueError("model asset directory prefixes cannot include a query")
        if parsed.canonical not in seen:
            seen.add(parsed.canonical)
            normalised.append(parsed.canonical)
    return tuple(normalised)


def _parse_network_url(
    value: str,
    *,
    model_asset: bool = False,
) -> _ParsedNetworkURL | None:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return None

    scheme = parsed.scheme.casefold()
    if model_asset and scheme not in _MODEL_ASSET_SCHEMES:
        return None
    if not parsed.netloc or not hostname or parsed.username or parsed.password:
        return None

    host = hostname.casefold()
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    else:
        netloc = host
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{netloc}:{port}"

    path = parsed.path or "/"
    if not _is_safe_url_path(path):
        return None
    canonical = urlunsplit((scheme, netloc, path, parsed.query, ""))
    origin = urlunsplit((scheme, netloc, "", "", ""))
    return _ParsedNetworkURL(
        canonical=canonical,
        origin=origin,
        path=path,
        query=parsed.query,
        scheme=scheme,
    )


def _is_safe_url_path(path: str) -> bool:
    """Reject path forms that a browser, proxy, or server may reinterpret."""

    candidate = path
    for _ in range(4):
        lowered = candidate.casefold()
        if (
            any(ord(character) < 32 or ord(character) == 127 for character in candidate)
            or "\\" in candidate
            or "%2e" in lowered
            or "%2f" in lowered
            or "%5c" in lowered
            or any(part in {".", ".."} for part in candidate.split("/"))
        ):
            return False
        try:
            decoded = unquote(candidate, errors="strict")
        except UnicodeError:
            return False
        if decoded == candidate:
            return True
        candidate = decoded
    try:
        return unquote(candidate, errors="strict") == candidate
    except UnicodeError:
        return False


def _summarise_request(
    index: int,
    request: _ObservedRequest,
    allowed_model_assets: tuple[str, ...],
) -> RequestSummary:
    raw_url = request.url
    url_digest = _digest(raw_url or "<missing-url>")
    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        parsed = SplitResult("", "", "", "", "")

    raw_scheme = parsed.scheme.casefold()
    scheme = (
        raw_scheme if raw_scheme in NETWORK_SCHEMES | NON_NETWORK_SCHEMES else "unknown"
    )
    if raw_scheme in NON_NETWORK_SCHEMES:
        classification = "browser-internal"
        origin_digest = _digest(raw_scheme)
    else:
        network_url = _parse_network_url(raw_url)
        if network_url is None:
            classification = (
                "invalid-url"
                if raw_scheme in {"", "http", "https"}
                else "unexpected-network"
            )
            origin_digest = _digest(raw_scheme or "invalid")
        elif network_url.scheme not in NETWORK_SCHEMES:
            classification = "unexpected-network"
            origin_digest = _digest(network_url.origin)
        elif request.method == "GET" and _matches_model_asset(
            network_url.canonical,
            allowed_model_assets,
        ):
            classification = "model-asset"
            origin_digest = _digest(network_url.origin)
        else:
            classification = "unexpected-network"
            origin_digest = _digest(network_url.origin)

    return RequestSummary(
        classification=classification,
        index=index,
        method=request.method,
        origin_digest=origin_digest,
        resource_type=request.resource_type,
        scheme=scheme,
        url_digest=url_digest,
    )


def _matches_model_asset(url: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/"):
            try:
                request_query = urlsplit(url).query
            except ValueError:
                continue
            if not request_query and url.startswith(pattern):
                return True
        if url == pattern:
            return True
    return False


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _load_trace(path: Path) -> list[Any]:
    try:
        with path.open("rb") as trace_file:
            raw_trace = trace_file.read(_MAX_TRACE_BYTES + 1)
        if len(raw_trace) > _MAX_TRACE_BYTES:
            raise ValueError("request trace exceeds the supported file size")
        payload = json.loads(raw_trace.decode("utf-8"))
    except (OSError, RecursionError, UnicodeError, json.JSONDecodeError):
        raise ValueError("could not read the request trace") from None

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("requests"), list):
        return payload["requests"]
    raise ValueError("request trace must be a JSON list or an object with requests")


def build_parser() -> argparse.ArgumentParser:
    """Build the offline network-egress checker CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="JSON request trace to inspect")
    parser.add_argument(
        "--allow-model-asset",
        action="append",
        default=[],
        metavar="URL_OR_PREFIX",
        help="explicit HTTP(S) model asset URL or slash-terminated prefix",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path for the privacy-safe JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checker against a local JSON trace and return its exit code."""

    args = build_parser().parse_args(argv)
    try:
        report = check_network_egress(
            _load_trace(args.trace),
            allowed_model_assets=args.allow_model_asset,
        )
        rendered = report.to_json()
        if args.report is not None:
            args.report.write_text(rendered, encoding="utf-8")
    except (OSError, ValueError):
        print(
            "Network egress check could not read or validate the supplied trace.",
            file=sys.stderr,
        )
        return 2

    print(rendered, end="")
    return 0 if report.passed else 1


__all__ = [
    "EgressReport",
    "NetworkEgressProbe",
    "NetworkEgressViolation",
    "RequestSummary",
    "assert_no_unexpected_requests",
    "capture_browser_requests",
    "check_network_egress",
]


if __name__ == "__main__":
    raise SystemExit(main())
