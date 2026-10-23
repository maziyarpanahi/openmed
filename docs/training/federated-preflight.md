# Federated Preflight

## Purpose

The federated preflight report is the deterministic, fail-closed
orchestration boundary used before federated round enrollment.

It combines independently computed findings and produces one of:

- `eligible`
- `review-required`
- `blocked`

## Fail-closed behavior

Mandatory checks are supplied by the caller.

If a declared mandatory check has no finding, the preflight runner
creates a `blocked` finding with:

`MANDATORY_CHECK_MISSING`

A blocked finding cannot be overridden by eligible or
review-required findings.

## Independent findings

Each check produces its own finding.

The report does not calculate an aggregate score and does not allow
one successful check to compensate for a blocked safety check.

## Existing validation contracts

The preflight orchestration layer consumes existing contracts for:

- federated scheduling
- federated update metadata
- federated metric envelopes
- environment lock digest

It does not reimplement capability-envelope or round-manifest
validation owned by the dependencies tracked in #2821 and #2822.

## Digest-only output

Reports contain digest references rather than participant identities,
local data, or training payloads.

Digest references use the canonical form:

`sha256:<64 lowercase hexadecimal characters>`

## Scope

This module only produces the preflight report.

It does not:

- enroll clients
- open network connections
- execute training
- aggregate updates
- promote models
- provide compliance certification