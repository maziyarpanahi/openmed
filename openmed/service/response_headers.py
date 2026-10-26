"""No-store, nosniff, referrer and framing headers for OpenMed REST responses.

REST responses can carry PHI, reversible de-identification mappings and job
results. Browsers and intermediaries may retain such responses, so every
non-probe response receives a strict header set:

- ``Cache-Control: no-store`` and ``Pragma: no-cache`` forbid storage by
  browsers, proxies and HTTP/1.0 caches.
- ``X-Content-Type-Options: nosniff`` stops browsers from reinterpreting JSON
  or NDJSON payloads as HTML or script.
- ``Referrer-Policy: no-referrer`` keeps request URLs, including job and
  resource identifiers, out of outbound ``Referer`` headers.
- ``X-Frame-Options: DENY`` and ``Content-Security-Policy: frame-ancestors
  'none'`` refuse framing. The CSP directive is the current standard; the
  legacy header covers clients that predate it. The CSP sets no other
  directive, so it does not restrict scripts on the generated API documents.

The middleware is pure ASGI and rewrites only the ``http.response.start``
message, so streamed bodies such as NDJSON are never buffered.
:func:`install_response_headers` wraps the application's complete middleware
stack, which places it outside Starlette's ``ServerErrorMiddleware``; the
JSON envelopes rendered for unhandled exceptions therefore carry the headers
too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .throttle import PROBE_PATHS

NO_STORE_CACHE_CONTROL = "no-store"
NO_CACHE_PRAGMA = "no-cache"
STATIC_DOCUMENT_CACHE_CONTROL = "no-cache"
ENFORCED_HEADERS = (
    ("x-content-type-options", "nosniff"),
    ("referrer-policy", "no-referrer"),
    ("x-frame-options", "DENY"),
)
FRAMING_POLICY = "frame-ancestors 'none'"


def _frozen_mapping(values: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class ResponseHeaderPolicy:
    """Route exemptions for the REST privacy response headers.

    Paths are exact, application-relative route paths; prefixes never match.

    Attributes:
        probe_paths: Liveness, readiness and health probe paths whose
            responses pass through unchanged.
        static_cache_control: Static, non-PHI documents mapped to the
            ``Cache-Control`` value they declare instead of ``no-store``.
            These responses still receive the nosniff, referrer and framing
            headers.
    """

    probe_paths: frozenset[str] = PROBE_PATHS
    static_cache_control: Mapping[str, str] = field(
        default_factory=lambda: _frozen_mapping({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_paths", frozenset(self.probe_paths))
        object.__setattr__(
            self, "static_cache_control", _frozen_mapping(self.static_cache_control)
        )
        for path, value in self.static_cache_control.items():
            if path in self.probe_paths:
                raise ValueError("A probe path cannot declare a cache policy")
            if not value or not value.isascii() or "\r" in value or "\n" in value:
                raise ValueError("Static cache policies must be single-line ASCII")


def static_document_paths(app: Any) -> tuple[str, ...]:
    """Return the generated API-document paths an application serves.

    Args:
        app: FastAPI application whose OpenAPI, Swagger UI and ReDoc settings
            are read. Disabled documents (``None`` URLs) are skipped.

    Returns:
        The application-relative paths of the enabled documents.
    """
    docs_url = getattr(app, "docs_url", None)
    candidates = [
        getattr(app, "openapi_url", None),
        docs_url,
        getattr(app, "redoc_url", None),
    ]
    if docs_url:
        candidates.append(getattr(app, "swagger_ui_oauth2_redirect_url", None))
    return tuple(path for path in candidates if isinstance(path, str) and path)


def policy_for_app(app: Any) -> ResponseHeaderPolicy:
    """Build the default policy: probes exempt, API documents revalidated.

    The generated OpenAPI schema, Swagger UI and ReDoc pages contain no PHI,
    so they declare ``Cache-Control: no-cache``, which allows storage but
    requires revalidation. Every other route keeps ``no-store``.

    Args:
        app: FastAPI application whose document routes are declared.

    Returns:
        The response-header policy for the application.
    """
    return ResponseHeaderPolicy(
        static_cache_control={
            path: STATIC_DOCUMENT_CACHE_CONTROL for path in static_document_paths(app)
        }
    )


def _has_no_store(values: list[str]) -> bool:
    for value in values:
        for directive in value.split(","):
            if directive.strip().lower() == NO_STORE_CACHE_CONTROL:
                return True
    return False


def apply_response_headers(
    headers: MutableHeaders,
    *,
    static_cache_control: Optional[str] = None,
) -> None:
    """Add the privacy headers to one response without duplicating any.

    A ``Cache-Control`` value that already contains ``no-store`` is kept;
    any other value is replaced, as is any ``Pragma`` other than
    ``no-cache``. ``X-Content-Type-Options`` is always exactly ``nosniff``,
    its only defined value. Referrer and legacy framing headers are enforced.
    Existing CSP policies remain in force alongside a no-framing policy.

    Args:
        headers: Mutable view over the response-start headers.
        static_cache_control: Declared cache policy of a static, non-PHI
            document. When given, it is used only if the handler set no
            ``Cache-Control``, and ``Pragma`` is left alone.
    """
    if static_cache_control is None:
        cache_values = headers.getlist("cache-control")
        headers["cache-control"] = (
            ", ".join(cache_values)
            if _has_no_store(cache_values)
            else NO_STORE_CACHE_CONTROL
        )
        if headers.getlist("pragma") != [NO_CACHE_PRAGMA]:
            headers["pragma"] = NO_CACHE_PRAGMA
    else:
        headers.setdefault("cache-control", static_cache_control)
    for name, value in ENFORCED_HEADERS:
        if headers.getlist(name) != [value]:
            headers[name] = value
    policies = headers.getlist("content-security-policy")
    if not any(
        policy.strip() == FRAMING_POLICY
        for value in policies
        for policy in value.split(",")
    ):
        policies.append(FRAMING_POLICY)
    # CSP lists enforce every policy independently; keeping their boundaries
    # preserves existing script, connection and other restrictions.
    headers["content-security-policy"] = ", ".join(policies)


def _route_path(scope: Scope) -> str:
    path = scope.get("path", "")
    root_path = scope.get("root_path", "")
    if root_path and path.startswith(root_path):
        remainder = path[len(root_path) :]
        if not remainder or remainder.startswith("/"):
            return remainder or "/"
    return path


class ResponseHeadersMiddleware:
    """Pure ASGI middleware that applies :class:`ResponseHeaderPolicy`.

    Args:
        app: Wrapped ASGI application.
        policy: Exemptions to apply. Defaults to probe exemptions only, with
            no static cache policies.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        policy: Optional[ResponseHeaderPolicy] = None,
    ) -> None:
        self.app = app
        self.policy = policy if policy is not None else ResponseHeaderPolicy()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process one ASGI connection, adding headers to HTTP responses."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = _route_path(scope)
        if path in self.policy.probe_paths:
            await self.app(scope, receive, send)
            return
        static_cache_control = self.policy.static_cache_control.get(path)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                apply_response_headers(
                    MutableHeaders(scope=message),
                    static_cache_control=static_cache_control,
                )
            await send(message)

        await self.app(scope, receive, send_with_headers)


def install_response_headers(
    app: Any,
    *,
    policy: Optional[ResponseHeaderPolicy] = None,
) -> None:
    """Wrap an application's full middleware stack with the header policy.

    ``add_middleware`` would place the middleware inside Starlette's
    ``ServerErrorMiddleware``, so 500 envelopes for unhandled exceptions
    would miss the headers. Wrapping the built stack covers them, leaves
    ``app.user_middleware`` untouched and keeps the headers independent of
    user-middleware ordering. Installing twice is a no-op.

    Args:
        app: Starlette or FastAPI application that has not started serving.
        policy: Exemptions to apply. Defaults to :func:`policy_for_app`,
            evaluated when the middleware stack is first built.

    Raises:
        RuntimeError: If the application's middleware stack is already built.
    """
    if getattr(app, "middleware_stack", None) is not None:
        raise RuntimeError("Cannot install response headers after app startup")
    if getattr(app.state, "response_headers_installed", False):
        return
    build_stack: Callable[[], ASGIApp] = app.build_middleware_stack

    def build_with_response_headers() -> ASGIApp:
        active = policy if policy is not None else policy_for_app(app)
        app.state.response_header_policy = active
        return ResponseHeadersMiddleware(build_stack(), policy=active)

    app.build_middleware_stack = build_with_response_headers
    app.state.response_headers_installed = True
