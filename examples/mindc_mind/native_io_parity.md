# Four-argument native I/O parity

Author: STARGA Inc.

`__mind_read(fd, buffer, count, offset)` and `__mind_write` use one guarded
lowering for every offset expression. Negative offsets select stream I/O;
nonnegative offsets select positioned I/O. A null buffer or nonpositive count
returns zero without a syscall; a syscall failure returns `-1`; success returns
the actual byte count. Positioned operations leave the file cursor unchanged.

This fixes the previous exception for compile-time `-1`: literal and computed
forms could previously return different errors or disagree on no-op guards.
Native normalization does **not** write libc TLS errno. That remains a dynamic
runtime obligation, including the downstream consumer campaign's `nc_errno()` use.
Raw `__mind_sys_*` and `std.posix` keep their negative-errno contracts.

The pre-offset three-argument compatibility form retains its raw syscall bytes
and behavior. Other argument counts are refused before the emitter loads any
intrinsic argument slots; the existing function/module refusal mechanism emits
no executable artifact. This change does not widen native C ABI admission.

## Regression contract

`self_host_native_io_smoke.py` declares 80 cases before execution:

- Seven offset spellings for both operations, testing invalid descriptors,
  null buffers, zero counts and negative counts.
- Literal/computed nonblocking `EAGAIN`, short writes, short reads, EOF and
  successful writes.
- Positioned I/O with independently checked file content and cursor movement.
- Raw negative errno, documented three-argument compatibility, and malformed
  arities 0, 1, 2 and 5 for both intrinsics.

An additional `--contexts` packet contains 16 malformed-call controls in
statement, let-binding, nested arithmetic and conditional positions. Enumerate
and run it in a separate evidence directory; the original 80 cases are unchanged.

The executable programs and assertions are MIND. The diagnostic host runner
builds them and records exit status, stdout, stderr and resulting files.
Every valid four-argument case must match the MLIR build and its independently
specified result. Legacy three-argument cases are native compatibility controls;
malformed arity cases require diagnosed native refusal with no output artifact.
A signal, timeout, allocator trap or failed valid-source build is never a pass.

```sh
python3 examples/mindc_mind/self_host_native_io_smoke.py \
  --enumerate --evidence /path/to/receipts
python3 examples/mindc_mind/self_host_native_io_smoke.py \
  --native /path/to/current-bounded-compiler.elf \
  --mlir /path/to/official-mindc --evidence /path/to/receipts
```

Run inside the documented 2 GiB, zero-swap cgroup and one-CPU quota. Each child
also receives a 2 GiB address-space limit and CPU/time limits. `--baseline`
records the explicitly predicted old defects; it is an observation mode and
must never be reported as qualification. It does not execute malformed programs.

## Deliberate byte-contract change

The original `POSITIONED`, `FILESYSTEM`, `ERRORS`, `SENTINEL_ONLY`,
`SENTINEL_MIXED` and `LEGACY_THREE_ARG` fixture sources in
`self_host_native_sys_smoke.py` are retained unchanged, including historical
comments describing the superseded exception. Their runtime checks are
retained, with stderr equality added. The old requirement that four-argument
literal sites match the frozen raw-syscall bytes is superseded because those
bytes encode the demonstrated defect. Its historical source and failing
execution receipts are preserved in the private compiler delivery evidence.

The corrected gate instead requires five guarded dispatches in the original
literal fixture and seven in the original mixed fixture, then executes them
with exact expected outputs. Three-argument bytes and the emitter closure's
seed parity remain required. No frozen seed or compiler pin is changed by this
lane. Integration must separately advance/reseed, prove the new fixed point,
and qualify the actual consumers before publishing a compiler pin.

The new diagnostic runner is classified as manual in `SMOKE_WIRING.tsv` until
CI supplies its exact source-bound bounded compiler. Existing syscall and
process gates retain their CI wiring. Passing this slice does not qualify the
ten downstream-consumer programs or establish general C FFI/errno support.

## Observed verification

The isolated 2026-10-15 verification passed all 80 original cases and the
16 additional malformed-call contexts. The original packet has SHA-256
`624958a627009df98a1c5c5d7331341193ab80d87ea22047692ecd6897242852`;
the supplement has SHA-256
`5a8f32d2c4d4a2e5e4a9951dbfb2998e2d116cf03561963be7f1eb0317e375ed`.
The source-bound bounded compiler reached an exact stage-2/stage-3 fixed point,
and its default whole-compiler ELF output matched a fresh MLIR malloc oracle.
Bootstrap, whole-module MIC3 and the existing syscall gate passed. The latter
used a temporary bounded current compiler as its seed and added the existing
bounded profile header to emissions; it did not qualify or modify the frozen
production seed. Original fixture sources and all remaining gate assertions
were retained. All phases used a 2 GiB zero-swap cgroup and one CPU; the largest
observed cgroup peak was 888,647,680 bytes, with zero OOM events.

Independent source review found no blocker in this slice and explicitly kept
libc TLS errno and cancellation semantics open. Review of execution receipts
is distinct from an independent rerun. This evidence supports the scoped I/O
correction, not a new compiler pin or completion of downstream-consumer qualification.
