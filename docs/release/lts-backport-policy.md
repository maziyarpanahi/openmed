# Backports to the 4.3 LTS line

The planned 4.3 Library/SDK support window runs for 12 months from its first
stable release through February 2028. Python 3.11 remains supported throughout
that window. Maintainers issue patch releases as eligible fixes become ready.

## Maintenance branch

Use `release/v4.3` for all 4.3 patch backports. After the first stable 4.3 tag
exists, the maintainer creates the branch from that exact tag:

```sh
git fetch origin tag v4.3.0
git branch release/v4.3 v4.3.0
git push origin refs/heads/release/v4.3
```

The release workflow branch filters match this name. Patch candidates must
declare a 4.3 version and use the previous reachable stable SDK tag as their
API baseline. The protected release-branch rules and commit attribution check
apply to every backport.

## Eligibility and approval

The OpenMed maintainer approves backports that address a security defect,
restore a previously met privacy recall threshold, or update a dependency
within the line's existing supported version floors for one of those fixes.
Report security defects privately under the
[security reporting policy](https://github.com/maziyarpanahi/openmed/blob/master/SECURITY.md).
For a recall regression, record the affected label or gate and the aggregate
before-and-after result using the [validation procedure](../clinical-validation-protocol.md#running-a-study).
Use synthetic examples in public PRs.

Features, public API additions and changes to defaults belong on the current
development line. A dependency update that raises a minimum supported runtime
or dependency version also belongs there.

## Landing a fix

1. Land the fix on `master` with regression coverage. If the affected code
   exists only on 4.3, explain that difference in the LTS PR and implement the
   correction directly against the LTS line.
2. Apply the `backport-4.3` label to identify the candidate. Link the original
   fix and explain which eligibility criterion it meets. For a confidential
   security fix, coordinate disclosure through the private reporting process.
3. Open a focused PR against the 4.3 maintenance branch. Preserve the original
   commit reference when cherry-picking and adapt it to the supported 4.3 API.
   Record the fix in the patch changelog and run the affected regression checks
   and normal release gates on that branch.
4. The maintainer reviews compatibility, evidence and CI before approving the
   backport for the next 4.3 patch release.

Stream A model releases follow their own promotion gates and cadence.
See [release streams and channels](semver-and-channels.md#channels) and the
[release gates](semver-and-channels.md#release-gates).
