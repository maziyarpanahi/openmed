"""Synthetic, process-local adapter for exercising the refresh protocol only."""

from contextlib import contextmanager
from threading import Lock

from openmed.interop.fhir.smart_refresh import SmartCredential

REQUESTED = "system/SyntheticObservation.cu offline_access"
HANDLE = "synthetic-handle"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Slot:
    def __init__(self, credential=None):
        self.credential = credential
        self.revoked = False
        self.replacements = 0

    def read(self):
        return self.credential

    def replace(self, credential):
        if self.revoked:
            raise ValueError("synthetic-revoked")
        self.credential = credential
        self.replacements += 1

    def revoke(self):
        self.credential = None
        self.revoked = True


class Custody:
    def __init__(self, credential=None):
        self.slot = Slot(credential)
        self.lock = Lock()

    @contextmanager
    def transaction(self, handle):
        assert handle == HANDLE
        with self.lock:
            yield self.slot


def old_credential(*, refresh="synthetic-refresh-0", expires_at=1030):
    return SmartCredential(
        "synthetic-access-0", refresh, expires_at, frozenset(REQUESTED.split())
    )


def response(**changes):
    payload = {
        "access_token": "synthetic-access-1",
        "refresh_token": "synthetic-refresh-1",
        "token_type": "Bearer",
        "expires_in": 300,
        "scope": REQUESTED,
    }
    payload.update(changes)
    return payload
