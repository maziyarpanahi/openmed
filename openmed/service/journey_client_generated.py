"""Generated Journey workflow client methods. Do not edit manually."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Optional

JourneyResourceType = Literal[
    "artifact",
    "job",
    "fact",
    "conflict",
    "journey",
    "cohort",
    "dataset",
    "registry",
    "measure",
    "trial_review",
    "evidence",
    "current_fact",
    "journey_event",
    "mapping",
    "cohort_run",
    "dataset_manifest",
]
JourneyWorkflowName = Literal[
    "journey", "cohort", "dataset", "registry", "measure", "trial_review"
]


class JourneyWorkflowClientMixin:
    """Generated convenience methods for fixed-resource workflows."""

    def journey_resources(
        self,
        resource_type: JourneyResourceType,
        *,
        namespace: str = "default",
        purpose: str = "care_review",
        first: int = 20,
        after: Optional[str] = None,
        fields: Sequence[str] = (),
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Implemented by the concrete REST client."""

        raise NotImplementedError

    def journey(
        self,
        *,
        namespace: str = "default",
        purpose: str = "care_review",
        first: int = 20,
        after: Optional[str] = None,
        fields: Sequence[str] = (),
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Read the journey workflow page."""

        return self.journey_resources(
            "journey",
            namespace=namespace,
            purpose=purpose,
            first=first,
            after=after,
            fields=fields,
            request_id=request_id,
        )

    def cohort(
        self,
        *,
        namespace: str = "default",
        purpose: str = "care_review",
        first: int = 20,
        after: Optional[str] = None,
        fields: Sequence[str] = (),
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Read the cohort workflow page."""

        return self.journey_resources(
            "cohort",
            namespace=namespace,
            purpose=purpose,
            first=first,
            after=after,
            fields=fields,
            request_id=request_id,
        )

    def dataset(
        self,
        *,
        namespace: str = "default",
        purpose: str = "care_review",
        first: int = 20,
        after: Optional[str] = None,
        fields: Sequence[str] = (),
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Read the dataset workflow page."""

        return self.journey_resources(
            "dataset",
            namespace=namespace,
            purpose=purpose,
            first=first,
            after=after,
            fields=fields,
            request_id=request_id,
        )

    def registry(
        self,
        *,
        namespace: str = "default",
        purpose: str = "care_review",
        first: int = 20,
        after: Optional[str] = None,
        fields: Sequence[str] = (),
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Read the registry workflow page."""

        return self.journey_resources(
            "registry",
            namespace=namespace,
            purpose=purpose,
            first=first,
            after=after,
            fields=fields,
            request_id=request_id,
        )

    def measure(
        self,
        *,
        namespace: str = "default",
        purpose: str = "care_review",
        first: int = 20,
        after: Optional[str] = None,
        fields: Sequence[str] = (),
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Read the measure workflow page."""

        return self.journey_resources(
            "measure",
            namespace=namespace,
            purpose=purpose,
            first=first,
            after=after,
            fields=fields,
            request_id=request_id,
        )

    def trial_review(
        self,
        *,
        namespace: str = "default",
        purpose: str = "care_review",
        first: int = 20,
        after: Optional[str] = None,
        fields: Sequence[str] = (),
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Read the trial_review workflow page."""

        return self.journey_resources(
            "trial_review",
            namespace=namespace,
            purpose=purpose,
            first=first,
            after=after,
            fields=fields,
            request_id=request_id,
        )


__all__ = ["JourneyResourceType", "JourneyWorkflowClientMixin", "JourneyWorkflowName"]
