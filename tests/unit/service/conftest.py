"""Shared fixtures for the service unit tests."""

from __future__ import annotations

import os

import pytest

from openmed.service.keyed_digest import SERVICE_DIGEST_KEY_ENV_VAR

SYNTHETIC_SERVICE_DIGEST_KEY = "synthetic-service-digest-key-" + "0" * 32


@pytest.fixture(autouse=True)
def _synthetic_service_digest_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give production apps with authentication their durable digest key.

    Outside the dev and test profiles, an app with service authentication
    refuses to start without ``OPENMED_SERVICE_DIGEST_KEY``. Tests that cover
    the missing-key path delete the variable themselves.
    """
    if not os.environ.get(SERVICE_DIGEST_KEY_ENV_VAR):
        monkeypatch.setenv(SERVICE_DIGEST_KEY_ENV_VAR, SYNTHETIC_SERVICE_DIGEST_KEY)
