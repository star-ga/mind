# Seed lineage before advancement execution

Author: STARGA Inc.

The advancement harness now validates the captured seed's lineage before running
that seed, any generated compiler, or the corroborating oracle. Previously, an
unresolved lineage prevented publication only after compiler execution. The
captured bytes remain the bytes executed; fixed-point, fresh-oracle, publication
and rollback requirements are unchanged.

This checks consistency with the trusted checkout and its Git history. It does
not authenticate attacker-controlled history or replace signature verification.

## Regression evidence

On 2026-10-29, the same strengthened tests were run against base
`5e0b3008262981b33e346805ec96901ab4cf9a1d` and the corrected four-file patch:

```sh
python3 tests/self_host_loop_advance_test.py
```

The baseline retained the original harness implementation. Both trees included
the new tests and the two existing process-refusal fixtures' explicit synthetic
lineage receipts. Full Git history was supplied for the existing chain test.

| Variant | Exit | Passed assertions | Failed assertions | stderr |
| --- | ---: | ---: | ---: | --- |
| Original execution order | 1 | 346 | 12 | Empty |
| Lineage checked first | 0 | 358 | 0 | Empty |

Four call-log controls detect compiler execution with absent or unreadable
history, for one-hop and two-hop advancement. A real, harmless shell seed also
writes a marker: admitted history must produce the marker and its intended
process-refusal result; absent/unreadable history must return the lineage refusal
without the marker. Each hop's positive and negative controls use exactly the
same script bytes and pathname. The twelve baseline failures are these ordering
and marker assertions, not crashes or missing dependencies.

Both source snapshots were hash-checked before and after execution. The single
Linux test job had a 2 GiB address-space and physical-memory limit, no swap,
one CPU and a 720-second outer deadline. Peak cgroup memory was 214,990,848 bytes;
memory-limit, OOM and swap events were zero.

The tested patch SHA-256 is
`722fa84f97e88def103bb6479cc1d81813b9d558bc7700d47fa41ec52ed879a4`.
Retained raw evidence archive SHA-256:
`3aade51fff31874cc4bd7d84205a061392aad4a4f5a071f3fc8874eb7eaa6c8f`.
Baseline/fixed stdout SHA-256, respectively:
`cf4921149992b2a190bcbe6207c47d92ab01b70c4a1d23b54c38ca4f311ad7e0`
and `b97970520301dc4f971ce80230cff54056a57295d9ad04b7c10aec4363d98e72`.
An independent source review accepted the ordering change with explicit trust
and coverage limitations; the exact-artifact marker control addresses its
requested improvement.

These are harness regression results using synthetic images and a stub oracle,
plus the harmless marker process. They do not demonstrate a real compiler fixed
point, authorize a seed refresh, qualify a new compiler pin, or satisfy consumer
native/MLIR compatibility or performance gates.
