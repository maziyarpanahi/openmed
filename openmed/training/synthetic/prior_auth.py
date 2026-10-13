"""Traceable synthetic prior-authorization packets with explicit evidence gaps."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from itertools import islice

_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_KINDS = frozenset({"note", "order", "form"})


def _code(value: str) -> None:
    if not isinstance(value, str) or not _CODE.fullmatch(value):
        raise ValueError("invalid synthetic authorization code")


@dataclass(frozen=True, slots=True)
class AuthorizationCriterion:
    """A user-declared synthetic evidence criterion, never an insurer rule."""

    code: str
    document_kinds: tuple[str, ...]
    required: bool = True

    def __post_init__(self) -> None:
        _code(self.code)
        kinds = tuple(islice(self.document_kinds, 4))
        if (
            not kinds
            or len(kinds) > 3
            or any(not isinstance(k, str) or k not in _KINDS for k in kinds)
            or len(set(kinds)) != len(kinds)
        ):
            raise ValueError("invalid synthetic authorization criterion kinds")
        if type(self.required) is not bool:
            raise ValueError("invalid synthetic authorization required state")
        object.__setattr__(self, "document_kinds", tuple(sorted(kinds)))


@dataclass(frozen=True, slots=True)
class AuthorizationEvidence:
    """A synthetic documented fact or deliberate distractor."""

    document_kind: str
    code: str
    present: bool
    distractor: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.document_kind, str) or self.document_kind not in _KINDS:
            raise ValueError("invalid synthetic authorization document kind")
        _code(self.code)
        if type(self.present) is not bool or type(self.distractor) is not bool:
            raise ValueError("invalid synthetic authorization evidence state")


@dataclass(frozen=True, slots=True)
class AuthorizationCaseSpec:
    """Bounded diagnosis/treatment placeholders and declared criterion evidence."""

    diagnosis_code: str
    treatment_code: str
    criteria: tuple[AuthorizationCriterion, ...]
    evidence: tuple[AuthorizationEvidence, ...]

    def __post_init__(self) -> None:
        _code(self.diagnosis_code)
        _code(self.treatment_code)
        criteria = tuple(islice(self.criteria, 129))
        evidence = tuple(islice(self.evidence, 1025))
        if (
            not criteria
            or len(criteria) > 128
            or any(not isinstance(c, AuthorizationCriterion) for c in criteria)
        ):
            raise ValueError("invalid synthetic authorization criteria")
        if len(evidence) > 1024 or any(
            not isinstance(e, AuthorizationEvidence) for e in evidence
        ):
            raise ValueError("invalid synthetic authorization evidence")
        if len({c.code for c in criteria}) != len(criteria):
            raise ValueError("duplicate synthetic authorization criterion")
        if len(set(evidence)) != len(evidence):
            raise ValueError("duplicate synthetic authorization evidence")
        codes = {c.code for c in criteria}
        if any(not e.distractor and e.code not in codes for e in evidence):
            raise ValueError("undeclared synthetic authorization criterion evidence")
        object.__setattr__(
            self, "criteria", tuple(sorted(criteria, key=lambda c: c.code))
        )
        object.__setattr__(
            self,
            "evidence",
            tuple(
                sorted(
                    evidence,
                    key=lambda e: (e.document_kind, e.code, e.present, e.distractor),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class AuthorizationDocument:
    """A synthetic note/order/form sharing case identity and clinical placeholders."""

    document_id: str
    case_id: str
    kind: str
    diagnosis_code: str
    treatment_code: str
    text: str
    evidence: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class AuthorizationPacket:
    """Synthetic documents and gold criterion statuses for offline evaluation."""

    case_id: str
    documents: tuple[AuthorizationDocument, ...]
    criteria: tuple[dict[str, object], ...]
    review_state: str


def generate_authorization_packet(
    spec: AuthorizationCaseSpec, *, seed: int = 0
) -> AuthorizationPacket:
    """Assemble note/order/form fixtures and gold gaps without deciding coverage."""
    if (
        not isinstance(spec, AuthorizationCaseSpec)
        or type(seed) is not int
        or not 0 <= seed <= 2**63 - 1
    ):
        raise ValueError("invalid synthetic authorization generation settings")
    case_id = "authcase_" + hashlib.sha256(f"{seed}:{spec!r}".encode()).hexdigest()[:24]
    documents = []
    references = {}
    for kind in sorted(_KINDS):
        document_id = (
            kind + "_" + hashlib.sha256(f"{case_id}:{kind}".encode()).hexdigest()[:24]
        )
        text = f"Synthetic {kind}.\nCase: {case_id}\nDiagnosis: {spec.diagnosis_code}\nTreatment: {spec.treatment_code}\n"
        truth = []
        for index, item in enumerate(
            e for e in spec.evidence if e.document_kind == kind
        ):
            line = f"{item.code}: {'present' if item.present else 'absent'}\n"
            start = len(text)
            text += line
            reference = f"{document_id}:{index}"
            truth.append(
                {
                    "reference": reference,
                    "code": item.code,
                    "present": item.present,
                    "role": "distractor" if item.distractor else "criterion_evidence",
                    "start": start,
                    "end": start + len(line) - 1,
                }
            )
            if not item.distractor:
                references.setdefault(item.code, []).append((item, reference))
        documents.append(
            AuthorizationDocument(
                document_id,
                case_id,
                kind,
                spec.diagnosis_code,
                spec.treatment_code,
                text,
                tuple(truth),
            )
        )
    criteria = []
    for criterion in spec.criteria:
        matches = [
            (e, ref)
            for e, ref in references.get(criterion.code, [])
            if e.document_kind in criterion.document_kinds
        ]
        states = {e.present for e, _ref in matches}
        if not states:
            status = "missing"
        elif len(states) > 1:
            status = "conflicting"
        elif True in states:
            status = "satisfied"
        else:
            status = "not_satisfied"
        criteria.append(
            {
                "code": criterion.code,
                "required": criterion.required,
                "document_kinds": list(criterion.document_kinds),
                "status": status,
                "evidence_references": sorted(ref for _e, ref in matches),
            }
        )
    required = [c["status"] for c in criteria if c["required"]]
    if any(status in {"missing", "conflicting"} for status in required):
        state = "abstain"
    elif "not_satisfied" in required:
        state = "requirements_not_satisfied"
    else:
        state = "ready_for_review"
    return AuthorizationPacket(case_id, tuple(documents), tuple(criteria), state)


def validate_authorization_packet(packet: AuthorizationPacket) -> tuple[str, ...]:
    """Independently verify shared identities, evidence spans and gold review state."""
    if (
        not isinstance(packet, AuthorizationPacket)
        or not isinstance(packet.documents, tuple)
        or len(packet.documents) != 3
    ):
        raise ValueError("invalid synthetic authorization packet")
    if not isinstance(packet.criteria, tuple) or not 1 <= len(packet.criteria) <= 128:
        return ("invalid_criteria",)
    codes = set()
    for criterion in packet.criteria:
        if not isinstance(criterion, dict) or set(criterion) != {
            "code",
            "required",
            "document_kinds",
            "status",
            "evidence_references",
        }:
            return ("invalid_criteria",)
        if (
            not isinstance(criterion["code"], str)
            or not _CODE.fullmatch(criterion["code"])
            or criterion["code"] in codes
            or type(criterion["required"]) is not bool
        ):
            return ("invalid_criteria",)
        codes.add(criterion["code"])
        kinds, refs = criterion["document_kinds"], criterion["evidence_references"]
        if (
            not isinstance(kinds, (list, tuple))
            or not 1 <= len(kinds) <= 3
            or any(not isinstance(k, str) or k not in _KINDS for k in kinds)
            or len(set(kinds)) != len(kinds)
        ):
            return ("invalid_criteria",)
        if (
            not isinstance(refs, (list, tuple))
            or len(refs) > 1024
            or any(not isinstance(ref, str) or len(ref) > 128 for ref in refs)
            or len(set(refs)) != len(refs)
        ):
            return ("invalid_criteria",)
        if not isinstance(criterion["status"], str) or criterion["status"] not in {
            "missing",
            "conflicting",
            "satisfied",
            "not_satisfied",
        }:
            return ("invalid_criteria",)
    for document in packet.documents:
        if (
            not isinstance(document, AuthorizationDocument)
            or any(
                not isinstance(value, str)
                for value in (
                    document.kind,
                    document.document_id,
                    document.case_id,
                    document.diagnosis_code,
                    document.treatment_code,
                    document.text,
                )
            )
            or len(document.text) > 1000000
        ):
            return ("invalid_document",)
        if not isinstance(document.evidence, tuple) or len(document.evidence) > 1024:
            return ("invalid_evidence",)
        for row in document.evidence:
            if not isinstance(row, dict) or set(row) != {
                "reference",
                "code",
                "present",
                "role",
                "start",
                "end",
            }:
                return ("invalid_evidence",)
            if (
                not isinstance(row["reference"], str)
                or len(row["reference"]) > 128
                or not isinstance(row["code"], str)
                or not _CODE.fullmatch(row["code"])
                or type(row["present"]) is not bool
                or not isinstance(row["role"], str)
                or row["role"] not in {"distractor", "criterion_evidence"}
            ):
                return ("invalid_evidence",)
    errors = set()
    if {d.kind for d in packet.documents} != {"note", "order", "form"}:
        errors.add("document_kind_mismatch")
    if (
        len({d.diagnosis_code for d in packet.documents}) != 1
        or len({d.treatment_code for d in packet.documents}) != 1
    ):
        errors.add("clinical_placeholder_mismatch")
    references = set()
    declared = {}
    for document in packet.documents:
        header = f"Synthetic {document.kind}.\nCase: {document.case_id}\nDiagnosis: {document.diagnosis_code}\nTreatment: {document.treatment_code}\n"
        if not document.text.startswith(header):
            errors.add("document_header_mismatch")
        if document.case_id != packet.case_id:
            errors.add("case_identity_mismatch")
        for row in document.evidence:
            start, end = row.get("start"), row.get("end")
            expected = f"{row['code']}: {'present' if row['present'] else 'absent'}"
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(document.text)
                or document.text[start:end] != expected
            ):
                errors.add("evidence_span_mismatch")
            if row["reference"] in references:
                errors.add("duplicate_evidence_reference")
            references.add(row["reference"])
            declared[row["reference"]] = (document.kind, row)
    for criterion in packet.criteria:
        if any(
            reference not in references
            for reference in criterion["evidence_references"]
        ):
            errors.add("dangling_evidence_reference")
        matches = [
            (ref, row)
            for ref, (kind, row) in declared.items()
            if kind in criterion["document_kinds"]
            and row["code"] == criterion["code"]
            and row["role"] == "criterion_evidence"
        ]
        states = {row["present"] for _ref, row in matches}
        expected = (
            "missing"
            if not states
            else (
                "conflicting"
                if len(states) > 1
                else "satisfied"
                if True in states
                else "not_satisfied"
            )
        )
        if criterion["status"] != expected or sorted(
            criterion["evidence_references"]
        ) != sorted(ref for ref, _row in matches):
            errors.add("criterion_truth_mismatch")
    required = [c["status"] for c in packet.criteria if c["required"]]
    expected_state = (
        "abstain"
        if any(s in {"missing", "conflicting"} for s in required)
        else (
            "requirements_not_satisfied"
            if "not_satisfied" in required
            else "ready_for_review"
        )
    )
    if packet.review_state != expected_state:
        errors.add("review_state_mismatch")
    return tuple(sorted(errors))
