"""Health Universe A2A agent for draft PHI-replacement Markdown."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections import Counter
from typing import Any, Protocol

# The Health Universe SDK includes optional tracing integrations. Keep them
# disabled unless the deployer deliberately opts in before importing this module.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("DO_NOT_TRACK", "1")

from health_universe_a2a import (  # noqa: E402
    Agent,
    AgentContext,
    UpdateImportance,
    ValidationAccepted,
    ValidationRejected,
)
from health_universe_a2a.documents import DocumentClient  # noqa: E402

from .health_universe_phi import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    OpenMedReplacementEngine,
    PhiReplacementSettings,
    ReplacementResult,
)

LOGGER = logging.getLogger(__name__)
FILE_ACCESS_URI = "https://healthuniverse.com/ext/file_access/v2"
DRAFT_BANNER = (
    "> **DRAFT PHI-REPLACEMENT OUTPUT — HUMAN REVIEW REQUIRED.**  \n"
    "> Do not treat this file as verified de-identified data.\n\n"
)


class ReplacementEngine(Protocol):
    """Interface used to inject a model-free engine in tests."""

    def replace(self, markdown: str, *, seed_scope: str) -> ReplacementResult:
        """Return replacement Markdown and raw-text-free counts."""

        ...


def opaque_source_id(document_id: str) -> str:
    """Hash a platform document ID for saved evidence."""

    return hashlib.sha256(document_id.encode()).hexdigest()


class PhiReplacementAgent(Agent):
    """Create draft replacements from Health Universe extracted Markdown."""

    def __init__(
        self,
        *,
        settings: PhiReplacementSettings | None = None,
        engine: ReplacementEngine | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings or PhiReplacementSettings.from_environment()
        self.engine = engine or OpenMedReplacementEngine(self.settings)
        self._model_lock = asyncio.Lock()

    def get_agent_name(self) -> str:
        return "Clinical PHI Replacement Agent"

    def get_agent_description(self) -> str:
        return (
            "Uses Health Universe platform extraction and a local OpenMed model to "
            "create draft Markdown with detected identifiers replaced by synthetic "
            "values. Human review remains required."
        )

    def get_agent_version(self) -> str:
        return "0.1.0"

    def get_max_duration_seconds(self) -> int:
        return 14400

    async def validate_message(
        self,
        message: str,
        metadata: dict[str, Any],
    ) -> ValidationAccepted | ValidationRejected:
        """Require at least one accessible source document."""

        del message
        file_extension = metadata.get(FILE_ACCESS_URI)
        if not isinstance(file_extension, dict):
            return ValidationRejected(reason="Upload at least one source document.")
        extension_context = file_extension.get("context") or {}
        if not isinstance(extension_context, dict):
            return ValidationRejected(reason="Document access context is missing.")
        access_token = file_extension.get("access_token")
        thread_id = extension_context.get("thread_id")
        if not access_token or not thread_id:
            return ValidationRejected(reason="Document access is incomplete.")

        document_client = DocumentClient(
            base_url=os.getenv(
                "HU_NESTJS_URL",
                "https://apps.healthuniverse.com/api/v1",
            ),
            access_token=access_token,
            thread_id=thread_id,
        )
        try:
            documents = await document_client.list_documents(
                role="source",
                flatten_attachments=True,
            )
            if not documents:
                return ValidationRejected(reason="Upload at least one source document.")
        except Exception as error:  # noqa: BLE001
            LOGGER.warning(
                "Source-document validation could not be completed: %s",
                type(error).__name__,
            )
        finally:
            try:
                await document_client.close()
            except Exception as error:  # noqa: BLE001
                LOGGER.warning(
                    "Document-validation connection cleanup failed: %s",
                    type(error).__name__,
                )
        return ValidationAccepted(estimated_duration_seconds=900)

    async def _processing_statuses(
        self,
        context: AgentContext,
        document_ids: list[str],
    ) -> list[Any]:
        """Wait for platform extraction and return available statuses."""

        try:
            return await context.document_client.wait_for_ready(
                document_ids=document_ids,
                timeout=float(self.settings.extraction_wait_seconds),
            )
        except TimeoutError:
            LOGGER.warning("Platform extraction wait timed out")
            statuses = []
            for document_id in document_ids:
                try:
                    statuses.append(
                        await context.document_client.get_processing_status(document_id)
                    )
                except Exception as error:  # noqa: BLE001
                    LOGGER.warning(
                        "Could not retrieve one extraction status: %s",
                        type(error).__name__,
                    )
            return statuses

    async def process_message(self, message: str, context: AgentContext) -> str:
        """Run platform extraction, local replacement, and artifact upload."""

        del message
        await context.update_progress(
            "Finding source documents...",
            0.05,
            importance=UpdateImportance.NOTICE,
        )
        documents = await context.document_client.list_documents(
            role="source",
            flatten_attachments=True,
        )
        unique_documents = {document.id: document for document in documents}
        documents = list(unique_documents.values())
        if not documents:
            return "No source documents were found."
        if len(documents) > self.settings.maximum_documents:
            return (
                f"The job contains {len(documents)} documents; the configured limit is "
                f"{self.settings.maximum_documents}. Split it into smaller jobs."
            )

        await context.update_progress(
            "Waiting for Health Universe platform OCR/extraction...",
            0.10,
            importance=UpdateImportance.NOTICE,
        )
        statuses = await self._processing_statuses(
            context,
            [document.id for document in documents],
        )
        status_by_id = {status.document_id: status for status in statuses}

        run_scope = str(getattr(context, "thread_id", "") or "thread")
        rows: list[dict[str, object]] = []
        status_counts: Counter[str] = Counter()
        action_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        collision_label_counts: Counter[str] = Counter()
        total_entities = 0
        total_collisions = 0
        ip_checked = 0
        invalid_ips = 0
        outputs_created = 0

        for index, document in enumerate(documents, start=1):
            if context.is_cancelled():
                status_counts["cancelled"] += len(documents) - index + 1
                break
            row: dict[str, object] = {
                "document_number": index,
                "source_document_id_sha256": opaque_source_id(document.id),
                "output": None,
            }
            status = status_by_id.get(document.id)
            if status is not None and not status.is_ready:
                row["status"] = "platform_extraction_not_ready"
                status_counts["platform_extraction_not_ready"] += 1
                rows.append(row)
                continue
            try:
                markdown = await context.document_client.download_extracted(document.id)
            except Exception as error:  # noqa: BLE001
                row.update(
                    status="platform_extraction_error",
                    error_type=type(error).__name__,
                )
                status_counts["platform_extraction_error"] += 1
                rows.append(row)
                continue
            if (
                sum(character.isalnum() for character in markdown)
                < self.settings.minimum_alphanumeric_characters
            ):
                row["status"] = "no_readable_extracted_text"
                status_counts["no_readable_extracted_text"] += 1
                rows.append(row)
                continue

            await context.update_progress(
                f"Replacing identifiers in document {index} of {len(documents)}...",
                0.15 + 0.75 * index / len(documents),
            )
            try:
                async with self._model_lock:
                    replacement = await asyncio.to_thread(
                        self.engine.replace,
                        markdown,
                        seed_scope=run_scope,
                    )
            except Exception as error:  # noqa: BLE001
                row.update(status="model_error", error_type=type(error).__name__)
                status_counts["model_error"] += 1
                rows.append(row)
                continue

            output_name = f"Deidentified_Document_{index:04d}.md"
            try:
                await context.document_client.write(
                    output_name,
                    DRAFT_BANNER + replacement.markdown,
                    filename=output_name,
                )
            except Exception as error:  # noqa: BLE001
                row.update(status="output_write_error", error_type=type(error).__name__)
                status_counts["output_write_error"] += 1
                rows.append(row)
                continue
            row.update(
                status="draft_replacement_created",
                output=output_name,
                entity_count=replacement.entity_count,
                replacement_collision_count=replacement.replacement_collision_count,
                ip_surrogates_checked=replacement.ip_surrogates_checked,
                invalid_ip_surrogates=replacement.invalid_ip_surrogates,
            )
            rows.append(row)
            status_counts["draft_replacement_created"] += 1
            action_counts.update(replacement.action_counts)
            label_counts.update(replacement.entity_label_counts)
            collision_label_counts.update(
                replacement.replacement_collision_label_counts
            )
            total_entities += replacement.entity_count
            total_collisions += replacement.replacement_collision_count
            ip_checked += replacement.ip_surrogates_checked
            invalid_ips += replacement.invalid_ip_surrogates
            outputs_created += 1

        report = {
            "report_schema": "openmed.phi_replacement_agent.v1",
            "contains_raw_phi": False,
            "model": MODEL_ID,
            "model_snapshot_revision": MODEL_REVISION,
            "model_integrity_registry_verified": False,
            "offline_model_inference": True,
            "method": "replace",
            "policy": None,
            "confidence_threshold": self.settings.confidence_threshold,
            "safety_sweep": True,
            "keep_mapping": False,
            "consistent_within_thread": True,
            "source_documents": len(documents),
            "draft_outputs_created": outputs_created,
            "status_counts": dict(status_counts),
            "total_model_entities": total_entities,
            "action_counts": dict(action_counts),
            "entity_label_counts": dict(label_counts),
            "replacement_collision_count": total_collisions,
            "replacement_collision_label_counts": dict(collision_label_counts),
            "ip_surrogates_checked": ip_checked,
            "invalid_ip_surrogates": invalid_ips,
            "documents": rows,
            "warning": (
                "Draft outputs may retain PHI. Human review is required before use "
                "or disclosure. Replacement does not establish HIPAA Safe Harbor."
            ),
        }
        report_name = "Deidentification_Safety_Report.json"
        await context.document_client.write(
            report_name,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            filename=report_name,
        )
        await context.update_progress(
            "Draft replacement Markdown and safety report created.",
            1.0,
            importance=UpdateImportance.NOTICE,
        )
        return (
            f"Created {outputs_created} draft replacement Markdown file(s) from "
            f"{len(documents)} source document(s). Human review is required."
        )
