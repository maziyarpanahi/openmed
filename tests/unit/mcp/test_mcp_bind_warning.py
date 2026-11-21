import logging
import warnings

import pytest

from openmed.mcp import server


@pytest.mark.parametrize(
    ("transport", "host", "warns"),
    [
        ("stdio", "private-fixture.example", False),
        ("streamable-http", "127.0.0.1", False),
        ("streamable-http", "::1", False),
        ("streamable-http", "localhost", False),
        ("streamable-http", "::ffff:127.0.0.2", False),
        ("streamable-http", "0.0.0.0", True),
        ("http", "192.168.1.23", True),
        ("streamable-http", "private-fixture.example", True),
    ],
)
def test_warning_is_at_listener_startup_only(
    transport, host, warns, monkeypatch, caplog
):
    fastmcp = pytest.importorskip("mcp.server.fastmcp").FastMCP
    started = []

    async def start(self):
        started.append(self.settings.host)

    monkeypatch.setattr(fastmcp, "run_streamable_http_async", start)
    monkeypatch.setattr(fastmcp, "run_stdio_async", start)
    with (
        caplog.at_level(logging.WARNING, logger=server.__name__),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always", DeprecationWarning)
        assert server.main(["--transport", transport, "--host", host]) == 0
    assert started == [host]
    notices = [
        str(w.message) for w in captured if "non_loopback_bind" in str(w.message)
    ]
    logs = [r.getMessage() for r in caplog.records if r.name == server.__name__]
    assert len(notices) == len(logs) == int(warns)
    for message in notices + logs:
        assert "5.0" in message and "authenticated reverse proxy" in message
        assert host not in message
