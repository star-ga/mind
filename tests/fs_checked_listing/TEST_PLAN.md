# Checked-listing acceptance plan

Author: STARGA Inc.

This is the original test contract, preserved after execution. See
[EVIDENCE.md](EVIDENCE.md) for observed MLIR results and the native blocker.
The production module is unchanged from source-r1, SHA256
338467cc428802e0d5a3147e90c9c98f876fcf9645761da1904e751bce7cc7c5.
The assertions and expectations below remain unchanged. The completed runs do
not qualify consumer gates or promote a compiler pin. This plan adds controls
without changing the original36 unit cases.

1. Original module plus `unit.mind`:24 raw-block cases and12 argument cases;
   exact normal exit0, stdout `0` plus LF, empty stderr. These check bytes and
   local argument handling, not syscall completion.
2. Original module plus `real.mind`: declared host fixture contains empty
   `fse/` and `fsc/d/`, private to one fresh case. There are initially no regular
   files. The MIND test verifies the empty listing and creates exclusive raw
   names `a` and byteFF using the candidate's own open/close primitives. It
   checks all three retained names, exact bounds, ENOENT and ENOTDIR, then
   repeats the complete listing/free operation10,000 times under original
   NOFILE1024. It removes only its own declared fixture on success. Host mkdir
   is input preparation, not a test oracle. No new mkdir intrinsic or extern
   is introduced. Exact normal exit0, stdout `0` plus LF, empty stderr.
3. Explicitly instrumented module plus `fault_support.mind`,
   `fault_driver.mind` and `fault_main.mind`:36 cases from FAULT_CASES.json,
   exact normal exit0, stdout `0` plus LF, empty stderr. No real descriptors
   are opened by the injected syscall functions. Raw errno and dirent blocks
   are controlled fixture inputs; actual platform behavior remains case2.
4. Seven actual module-source mutations in MUTATIONS.json, each run with its
   designated original control first. The original must normally exit0 with
   `0` LF and empty stderr. The mutant must normally exit1 and print precisely
   the predeclared numeric diagnostic triples, with empty stderr. Build failure,
   crash, timeout, OOM or an unexpected diagnostic is a failed experiment, not
   successful fault detection. Original and mutant source hashes are frozen.

## Test-only instrumentation boundary

INSTRUMENTATION.json contains29 ordered exact source replacements. They thread
one explicit control pointer through the module's allocation-owning helpers and
its public API, replace the three primitive wrapper bodies with controlled MIND
functions, and wrap all six allocation expressions and13 free expressions.
The instrumented signature is deliberately a distinct test ABI, not an exported
replacement for `fs_read_dir_checked`. Source control flow, arithmetic, guards,
publication and ownership paths are otherwise unchanged. Assembly checks each
needle's exact occurrence count and complete input/output source SHA256.

The control block belongs to the test harness;64 tracked allocation slots cover
the bounded cases. The wrappers call the actual allocator/free implementation.
A selected allocation ordinal returns null exactly once. Unexpected real OOM,
registry exhaustion, foreign/double/null free or bad descriptor use increments
an independently checked error count. Fifteen failure ordinals cover all six
allocation sites including growth after four retained entries. Every failed
allocation must have the exact expected allocation count and free all prior
successful allocations. Remaining allocations are cleaned only *after* failed
ownership observations have been recorded. This avoids masking detected leaks.

EOF, immediate read failure, partial enumeration then read failure, malformed
second block and close failures are distinct cases. EINTR close consumes the
simulated descriptor; exactly one close must be observed. The successful fd0
case prevents using `fd > 0` as an accidental admission condition. Both input
bytes and an output-adjacent guard are checked. The output stays the sentinel on
failure; no supplied PASS or status receipt is accepted as product authority.

## Remaining obligations

The original-source real test used the built and reviewed candidate identified
in EVIDENCE.md. The accepted a1444 compiler and
the separate bounded-SHA/open diagnostic build do not supply those primitives.
No private fallback or source-level replacement can qualify that real route.
Windows is unsupported by this Linux primitive profile. Host toolchain, exact
compiled source, executable hashes and resource controls require a separate
frozen transport. Static source review does not establish backend semantics.

Root owns parser integration. It must propagate checked-listing failures before
any MANIFEST replacement or journal deletion and demonstrate unchanged on-disk
authority after partial scans/open/read/close/allocation failures. This module's
tests do not claim that consumer obligation or an atomic directory snapshot.
