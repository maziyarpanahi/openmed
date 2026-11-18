"""Synthetic continuation-authority harness; no identity service or network."""

from __future__ import annotations

from openmed.agent.correlation import ActionId, RunId
from openmed.agent.identifiers import ToolId
from openmed.agent.permissions.revocation import (
    AuthorityRuntime,
    InMemoryAuthorityGenerationStore,
)
from openmed.agent.workflows.recovery import (
    CompensationLimit,
    EffectKind,
    EffectRecord,
)
from openmed.service.auth import AuthPrincipal
from openmed.service.continuation_auth import (
    ContinuationGuard,
    InMemoryContinuationLedger,
    consent_authority,
    principal_authority,
)
from tests.fixtures.agent.authority_status import (
    SyntheticClock,
    SyntheticStatusProvider,
)

SENTINEL = "SENTINEL-PHI-Jane-Roe-555-0100"
WORK_ID = "job-synthetic-0001"
RUN_ID = RunId("run_" + "7" * 32)
POLICY_REVISION = "policy:org.example/deid@3"
CONSENT_DIGEST = "sha256:" + "c" * 64
COMMIT_EVIDENCE = "sha256:" + "e" * 64


class ContinuationAuthority:
    """One synthetic authority service plus a worker-process guard.

    The provider and clock model the authority service, which outlives a worker
    restart. The generation store and ledger belong to the worker process unless
    a test passes durable copies into :meth:`restart`.
    """

    def __init__(
        self,
        *,
        provider: SyntheticStatusProvider | None = None,
        clock: SyntheticClock | None = None,
        store: InMemoryAuthorityGenerationStore | None = None,
        ledger: InMemoryContinuationLedger | None = None,
    ) -> None:
        self.clock = clock if clock is not None else SyntheticClock()
        self.provider = (
            provider if provider is not None else SyntheticStatusProvider(self.clock)
        )
        self.store = store if store is not None else InMemoryAuthorityGenerationStore()
        self.ledger = ledger if ledger is not None else InMemoryContinuationLedger()
        self.runtime = AuthorityRuntime(self.provider, self.store, clock=self.clock)
        self.guard = ContinuationGuard(self.runtime, ledger=self.ledger)
        self.principal = AuthPrincipal(
            subject=SENTINEL,
            credential_type="api_key",
            scopes=("deidentify", "jobs"),
            key_id="synthetic-key",
        )
        self.scope = principal_authority(
            self.principal, policy_revision=POLICY_REVISION, expires_at=100
        )
        self.consent = consent_authority(CONSENT_DIGEST, expires_at=100)
        self.contracts = (self.scope, self.consent)
        if provider is None:
            self.provider.register(self.contracts)

    def restart(
        self,
        *,
        store: InMemoryAuthorityGenerationStore | None = None,
        ledger: InMemoryContinuationLedger | None = None,
    ) -> "ContinuationAuthority":
        """Return a new worker process against the same authority service."""

        return ContinuationAuthority(
            provider=self.provider, clock=self.clock, store=store, ledger=ledger
        )


def pending_effect(ordinal: int) -> EffectRecord:
    """Return one content-free pending effect for the synthetic run."""

    return EffectRecord.create(
        ordinal=ordinal,
        run_id=RUN_ID,
        action_id=ActionId("act_" + str(ordinal + 1) * 32),
        tool_id=ToolId("tool:org.example/sink@1.0.0"),
        kind=EffectKind.LOCAL_TOOL,
        operation_digest="sha256:" + str(ordinal + 1) * 64,
        approval_required=False,
        compensation_limit=CompensationLimit.NONE,
    )


class SyntheticSink:
    """Effect sink that durably commits once per call and returns evidence."""

    def __init__(self) -> None:
        self.commits: list[str] = []
        self.fail = False

    def commit(self, key: str) -> str:
        if self.fail:
            raise RuntimeError(SENTINEL)
        self.commits.append(key)
        return COMMIT_EVIDENCE


class SyntheticSource:
    """Closable synchronous stream source that records how far it was read."""

    def __init__(self, chunks: list[str], *, fail_at: int | None = None) -> None:
        self.chunks = chunks
        self.fail_at = fail_at
        self.read = 0
        self.closed = False

    def __iter__(self) -> "SyntheticSource":
        return self

    def __next__(self) -> str:
        if self.fail_at is not None and self.read == self.fail_at:
            raise RuntimeError(SENTINEL)
        if self.read >= len(self.chunks):
            raise StopIteration
        chunk = self.chunks[self.read]
        self.read += 1
        return chunk

    def close(self) -> None:
        self.closed = True


class SyntheticAsyncSource:
    """Closable asynchronous stream source mirroring :class:`SyntheticSource`."""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.read = 0
        self.closed = False

    def __aiter__(self) -> "SyntheticAsyncSource":
        return self

    async def __anext__(self) -> str:
        if self.read >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.read]
        self.read += 1
        return chunk

    async def aclose(self) -> None:
        self.closed = True
