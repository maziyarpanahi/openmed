"""Regression tests for aggregate guarded-relation public exports."""

from openmed.clinical import (
    AssertionState,
    RelationAssertionState,
    RelationEvidence,
    RelationEvidenceLocation,
    validate_guarded_relation,
    validate_guarded_relation_direction,
    validate_guarded_relation_evidence,
)
from openmed.clinical.radiology_profile import AssertionState as RadiologyAssertionState
from openmed.clinical.relations.deduplicate import (
    RelationEvidence as DeduplicationEvidence,
)
from openmed.clinical.relations.directionality import (
    validate_guarded_relation as validate_direction,
)
from openmed.clinical.relations.evidence_binding import (
    AssertionState as EvidenceAssertionState,
)
from openmed.clinical.relations.evidence_binding import (
    EvidenceSpan,
)
from openmed.clinical.relations.evidence_binding import (
    validate_guarded_relation as validate_evidence,
)


def test_colliding_contract_names_have_explicit_public_exports() -> None:
    """Keep aggregate exports bound to their intended contract modules."""

    assert RelationEvidence is EvidenceSpan
    assert RelationEvidenceLocation is DeduplicationEvidence
    assert validate_guarded_relation is validate_direction
    assert validate_guarded_relation_direction is validate_direction
    assert validate_guarded_relation_evidence is validate_evidence
    assert AssertionState is RadiologyAssertionState
    assert RelationAssertionState is EvidenceAssertionState
