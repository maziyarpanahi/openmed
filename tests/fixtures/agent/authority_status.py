"""Synthetic local authority contracts and status provider; no identity service."""

from dataclasses import replace

from openmed.agent.correlation import RunId
from openmed.agent.permissions.access_tickets import (
    AccessTicket,
    AccessTicketRequest,
    AccessTicketVerifier,
    RecordSelector,
    ToolAction,
)
from openmed.agent.permissions.delegation import (
    DelegationGrantSigner,
    DelegationGrantVerifier,
    DelegationRequest,
    DelegationScope,
)
from openmed.agent.permissions.grants import (
    CapabilityGrantConstraint,
    CapabilityGrantRequest,
    CapabilityGrantSigner,
    CapabilityGrantVerifier,
)
from openmed.agent.permissions.revocation import (
    AuthorityContract,
    AuthorityKind,
    AuthorityRuntime,
    AuthorityStatus,
    InMemoryAuthorityGenerationStore,
)
from openmed.agent.permissions.runtime_authority import (
    delegation_authorities,
    grant_authority,
    ticket_authority,
)

KEY = b"synthetic-offline-signing-key-32-bytes"
ACTION_DIGEST = "sha256:" + "a" * 64


class SyntheticClock:
    def __init__(self) -> None:
        self.now = 10

    def __call__(self) -> int:
        return self.now


class SyntheticStatusProvider:
    """Provider with explicit unknown records and monotonic revocation."""

    def __init__(self, clock: SyntheticClock) -> None:
        self.clock = clock
        self.records: dict[tuple[AuthorityKind, str], tuple[int, bool]] = {}
        self.calls = 0
        self.override: AuthorityStatus | None = None
        self.unavailable = False

    def register(self, contracts: tuple[AuthorityContract, ...]) -> None:
        for contract in contracts:
            self.records[(contract.kind, contract.digest)] = (0, False)

    def revoke(self, contract: AuthorityContract) -> None:
        key = (contract.kind, contract.digest)
        generation, _ = self.records[key]
        self.records[key] = (generation + 1, True)

    def get_status(self, kind: AuthorityKind, digest: str) -> AuthorityStatus | None:
        self.calls += 1
        if self.unavailable:
            raise RuntimeError("synthetic-private-provider-payload")
        if self.override is not None:
            return self.override
        record = self.records.get((kind, digest))
        if record is None:
            return None
        generation, revoked = record
        return AuthorityStatus(kind, digest, generation, revoked, self.clock())


class SyntheticAuthority:
    """Complete existing static contracts and an injected local runtime."""

    def __init__(self) -> None:
        self.clock = SyntheticClock()
        self.provider = SyntheticStatusProvider(self.clock)
        self.store = InMemoryAuthorityGenerationStore()
        self.runtime = AuthorityRuntime(self.provider, self.store, clock=self.clock)
        self.constraint = CapabilityGrantConstraint(
            tool="tool:org.example/synthetic@1.0.0",
            resource="resource:org.example/synthetic@1.0.0",
            action="action:org.example/read@1.0.0",
            policy_profile="policy:org.example/local@1.0.0",
        )
        self.grant = CapabilityGrantSigner(KEY).issue([self.constraint], expires_at=100)
        self.grant_request = CapabilityGrantRequest(**self.constraint.to_dict())
        self.grant_verifier = CapabilityGrantVerifier(KEY, clock=self.clock)
        selector = RecordSelector.from_value(
            kind="selector:org.example/record@1.0.0",
            value="synthetic-record",
            key=KEY,
        )
        action = ToolAction(self.constraint.tool, self.constraint.action)
        self.ticket = AccessTicket(
            RunId("run_" + "1" * 32),
            "purpose:org.example/synthetic@1.0.0",
            ("data:org.example/synthetic@1.0.0",),
            (selector,),
            (action,),
            100,
        )
        self.ticket_request = AccessTicketRequest(
            self.ticket.run_id,
            self.ticket.purpose,
            self.ticket.permitted_data_classes,
            (selector,),
            action,
        )
        self.ticket_verifier = AccessTicketVerifier(clock=self.clock)
        scope = DelegationScope(
            (self.constraint,),
            self.ticket.permitted_data_classes,
            (self.ticket.purpose,),
        )
        signer = DelegationGrantSigner(KEY)
        self.delegation_verifier = DelegationGrantVerifier(KEY, clock=self.clock)
        root = signer.issue_root(
            principal="agent:org.example/parent@1.0.0",
            scope=scope,
            expires_at=100,
            remaining_depth=2,
        )
        child = signer.derive_child(
            root,
            DelegationRequest(
                principal="agent:org.example/child@1.0.0",
                scope=scope,
                expires_at=100,
                remaining_depth=1,
            ),
            self.delegation_verifier,
            now=self.clock(),
        ).grant
        grandchild = signer.derive_child(
            child,
            DelegationRequest(
                principal="agent:org.example/grandchild@1.0.0",
                scope=scope,
                expires_at=100,
                remaining_depth=0,
            ),
            self.delegation_verifier,
            now=self.clock(),
        ).grant
        self.chain = (root, child, grandchild)
        self.contracts = (
            grant_authority(self.grant),
            ticket_authority(self.ticket),
            *delegation_authorities(self.chain),
        )
        self.provider.register(self.contracts)

    def current_status(self, contract: AuthorityContract) -> AuthorityStatus:
        status = self.provider.get_status(contract.kind, contract.digest)
        assert status is not None
        return replace(status)
