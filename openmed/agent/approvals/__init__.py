"""Human approval contracts for high-impact agent actions."""

from __future__ import annotations

from .tokens import (
    APPROVAL_NONCE_BYTES,
    APPROVAL_RECEIPT_SCHEMA_VERSION,
    APPROVAL_TOKEN_SCHEMA_VERSION,
    APPROVAL_TOKEN_SIGNATURE_ALGORITHM,
    ApprovalActionMismatchError,
    ApprovalExpiredError,
    ApprovalNonceStore,
    ApprovalNonceStoreError,
    ApprovalReceipt,
    ApprovalReplayError,
    ApprovalReviewerRoleMismatchError,
    ApprovalSignatureError,
    ApprovalToken,
    ApprovalTokenError,
    ApprovalTokenSigner,
    ApprovalTokenValidationError,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
    dispatch_with_approval_token,
)

__all__ = [
    "APPROVAL_NONCE_BYTES",
    "APPROVAL_RECEIPT_SCHEMA_VERSION",
    "APPROVAL_TOKEN_SCHEMA_VERSION",
    "APPROVAL_TOKEN_SIGNATURE_ALGORITHM",
    "ApprovalActionMismatchError",
    "ApprovalExpiredError",
    "ApprovalNonceStore",
    "ApprovalNonceStoreError",
    "ApprovalReceipt",
    "ApprovalReplayError",
    "ApprovalReviewerRoleMismatchError",
    "ApprovalSignatureError",
    "ApprovalToken",
    "ApprovalTokenError",
    "ApprovalTokenSigner",
    "ApprovalTokenValidationError",
    "ApprovalTokenVerifier",
    "InMemoryApprovalNonceStore",
    "dispatch_with_approval_token",
]
