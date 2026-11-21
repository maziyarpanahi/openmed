# Recall backstops on the 4.3 LTS line

Start this procedure when a supported release falls below its applicable G1a
floor for a direct-identifier label, or when a supported workflow allows a
redaction bypass. Record the model revision, format, library version,
configured profile, affected label, denominator and before-evidence digest.
A pooled recall score can hide a failing label. If the denominator is missing,
record the evidence gap and investigate it separately.

Treat any suspected PHI exposure as a private security report. The project's
[severity rules](https://github.com/maziyarpanahi/openmed/blob/master/SECURITY.md#severity)
classify systematic identifier false negatives under a documented profile as
high and real leakage or a redaction bypass as critical. Use synthetic cases
for public regression coverage. Keep identifiers, source records and bypass
details within the private incident process.

1. **Open a corrective action.** Use `openmed compliance capa new --help` to
   create a record in the local ledger with source `incident`, assessed
   severity, affected capability and version. Associate the before-evidence
   digest with the record. Use `openmed compliance capa transition` to track
   the investigation. Keep only its opaque reference in release evidence.
   See the [corrective-action ledger](../compliance/waiver-ledger.md) and
   [CAPA CLI](https://github.com/maziyarpanahi/openmed/pull/5728).

2. **Contain a model regression in Stream A.** Resolve the last-green pointer
   from `gates/baseline.json` with
   `openmed release rollback PII --dry-run`. Supply `--tier` and `--format`
   for the affected model variant. Check the target revision against its
   recorded green evidence. If this rollback restores the tested behavior, repeat the command
   without `--dry-run` on a review branch and inspect the diff in
   `models.jsonl`, generated cards and rollback log. If a library defect
   affects both model revisions, record that the pointer offers no mitigation.
   [Release streams and channels](semver-and-channels.md) describes the two
   independent release paths.

3. **Prepare the Stream B library patch.** When the defect is in the SDK,
   add a synthetic regression for the failing label or affected redaction
   path. Prepare the smallest eligible change for `release/v4.3`, run the
   regression there and add an entry to `CHANGELOG.md`. Follow the
   [4.3 LTS backport policy](https://github.com/maziyarpanahi/openmed/blob/358357b53998dd79a84bfd7172034a6008458aef/docs/release/lts-backport-policy.md).
   For a shared defect the policy lands the fix on `master` first and
   backports the reviewed commit. A defect confined to 4.3 can be fixed
   directly on the LTS branch. Record whether model containment alone
   resolved the incident before proposing a library patch.

4. **Account for the forward port.** Compare the affected paths using
   `git diff release/v4.3 master -- <affected-path>`. Link the corresponding
   `master` fix, forward-port an LTS-first correction if applicable, or
   document why current code is unaffected. Run the same regression on each
   affected line and retain its result for review.

5. **Re-verify the affected build.** Run
   `python -m pytest tests/unit/eval/test_release_gates.py -q` to check the
   gate implementation. Separately rerun the affected evaluation on the
   pinned model, format and profile. Compare each G1a label and denominator
   with its floor and confirm the leakage regression is covered. Refresh the
   aggregate reports, rebuild and sign the capability pack, then run
   `openmed evidence verify` with the review key, evaluation time and age
   limit. Record the after-evidence digest and remaining blockers in the
   ledger. The [evidence pack instructions](v3.0-journey-release-gate.md#capability-validation-evidence-packs)
   give the offline commands.

6. **Decide whether a field-safety notice is needed.** Use the affected
   versions, residual exposure and verified mitigation to determine whether
   operators must pin a model, install a patch or reprocess affected outputs.
   The responsible maintainer records the decision and rationale privately
   against the CAPA ID. Follow the [incident notification runbook](../compliance/breach-notification-runbook.md)
   with the site's incident owner when privacy exposure is suspected.

7. **Coordinate the advisory.** Follow the private reporting and disclosure
   process in [SECURITY.md](https://github.com/maziyarpanahi/openmed/blob/master/SECURITY.md).
   Identify affected versions and formats, fixed pointers or patch versions,
   operator action and the available aggregate verification evidence.
   Coordinate timing with affected parties. Close the corrective action
   after verification and link the advisory reference in the ledger.
