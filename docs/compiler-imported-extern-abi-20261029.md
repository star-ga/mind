# Imported extern ABI validation — 2026-10-29

Author: STARGA Inc.

The scoped ABI successor passes its ten added regression functions and three retained Linux runtime controls. This is not a compiler release, seed transition, full-suite pass or qualified MIND-Lab pin. The accepted `a1444` compiler pin and the ten-program/35-case consumer qualification contract remain unchanged.

## Source and behavior

The tested successor is based on `8d2616d3c5b3f7fa5fdb36dda463a6ea81da05df`. Its frozen six-file patch has SHA-256 `988c33787d308f654111e4cf27788519866feb19f0f09beb75457439a255ce7c`.

The source checker now rejects conflicting declarations of the same foreign symbol within one source owner, including declarations nested in function and control-flow bodies that lowering hoists into the owner's foreign-symbol namespace. Identical declarations and default-C versus explicitly equivalent host conventions remain valid. The checker uses the existing calling-convention resolver and maintained AST child walker.

Imported extern lowering now resolves default C consistently with local declarations, rather than passing an unresolved `C` tag to the ABI classifier. Export collection preserves conflicts for refusal instead of silently retaining the first declaration. Scoped lookup also refuses conflicting declarations belonging to one owner.

Nested extern annotations are admitted only when the existing primitive/pointer/callback checker can validate their raw lowered types without a lexical alias or aggregate-layout mapping. Unsupported nested aliases/layouts receive ordinary diagnostics; this change does not claim to implement their lowering.

## Observed checks

| Check | Baseline product source plus frozen tests | Fixed source | Actual evidence |
| --- | --- | --- | --- |
| Declaration integration suite | 3 passed, 3 assertion failures | 6 passed, 0 failed, 0 ignored | Contradictory signatures, nested conflicts, nested unresolved annotations; valid identical declarations and convention equivalence |
| Internal ABI controls | 1 passed, 3 assertion failures; exit 101 | 4 passed, 0 failed, 0 ignored; exit 0 | Same-owner ambiguity; stable unimported siblings; resolved host convention; imported/local convention agreement |
| Retained imported-extern runtime suite | 3 passed | 3 passed, 0 failed, 0 ignored | Each test actually built and loaded MIND and C shared objects, then checked 13 signed/unsigned argument/return values and error-return behavior |

The baseline internal failures were ordinary assertion failures: `C` versus `SysV` metadata, imported `C` versus local `SysV`, and an incorrectly visible conflicting declaration. A crash or build failure was not counted as detection. Fixed execution stdout contains all expected test names; stderr is empty. `MIND_BENCH_REQUIRE=1` made missing runtime prerequisites fail instead of skipping.

Both baseline and fixed controls used the same retained dependency lock, SHA-256 `460b6a63eaff3a268b0abaf7b40feff92f5ce13a72bb80c9913cccf83fd529da`. The existing repository does not track a Cargo.lock; this validation lock is an external evidence input, not a new repository dependency policy. All 187 required public registry archives matched their lock checksums. Credential configuration and unrelated private material were not copied.

## Toolchain and reproduction

Observed host: Linux x86-64 in WSL on G1. Rust/Cargo were version 1.99.1; external LLVM/MLIR/Clang were 23.1.3. The Rust compiler's internal LLVM version differs from the external toolchain and is not substituted for it.

For each immutable source snapshot, use a distinct `CARGO_TARGET_DIR`. Supply the recorded validation lock, an offline verified registry cache, one Cargo build job, disabled incremental compilation, and test/dev debug information set to zero. Explicitly select `MLIR_OPT`, `MLIR_TRANSLATE`, and `CLANG` from the same LLVM 23.1.3 installation. The retained runtime test invokes `cc` and `python3`; the test-only `cc` path resolves to that Clang executable, and Python is an external test instrument.

```sh
cargo test --offline --locked \
  --features std-surface,mlir-build,cross-module-imports \
  --test extern_declaration_conflicts \
  --test imported_extern_narrow_ret_run \
  --verbose --no-run --message-format=json

cargo test --offline --locked \
  --features std-surface,mlir-build,cross-module-imports \
  --lib --verbose --no-run --message-format=json
```

Extract exact test executables from Cargo's `compiler-artifact` records, hash them, then run the two integration executables with `--test-threads=1 --nocapture`. Run the library executable with `extern_abi_tests --test-threads=1 --nocapture`. Execution receipts must match both source and executable hashes; merely seeing Cargo report `fresh` is insufficient.

| Artifact | SHA-256 |
| --- | --- |
| Fixed frontend used for runtime tests | `2b128231f425cda5455f9b673cf104607c7ca8e354f31a95d00707ef3c052136` |
| Fixed declaration test executable | `5f26017000a2216a77fe342f3c805ea665081709ba1bf49bc267a4419918522b` |
| Fixed retained runtime test executable | `a3db812dde48b8af3887d30113fc592a6b0ea32b7542fa1e0273d9bf5757b65b` |
| Baseline library test executable | `a7c30a9a8c834b6a6ef0e92e95b99008ac9575e85a13991d12bd76bddf529730` |
| Fixed library test executable | `4577e241cdb0e843726009d0a07db915e989f081bd3d24d130c9b6fd34b9a6e1` |
| Rust compiler | `b53b780f77f952062f8ca8a3b70aef2390b1e45ad91939be94afdc9543026853` |
| Cargo | `a670e02cfa39ac2e53f77a2d911ca1fc9e2c479b0603ec956625be1daf24e3af` |
| mlir-opt | `12758a64a51d0b623b0091f5fcdec93cdd5dcadcf93dde30998bbecb48885b69` |
| mlir-translate | `738f59331cd57730da43b12c602f80b87e0c16fca2d925cb57c8a59ae5acb17f` |
| Clang / test-only cc | `913453afb492abab17025139852413ba7fd7a52b4fb75fcce58ff3fe399d0a9e` |
| llc | `9f294abce5106717cb0fe94cd564c32b825789f11783f0921b44009c4af7ae6c` |
| External Python test instrument | `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f` |

The frontend is a validation debug build, not the published accepted compiler and not a byte-identical rebuild of the earlier frontend built with Rust 1.93.

## Resource evidence and invalid attempts

All jobs used one CPU and no swap. Fixed integration compilation/execution ran in a 2 GiB RAM cgroup, with peak charged memory 1,782,841,344 bytes; no address-space qualification is claimed for that unit. The four library controls subsequently executed in a separate 2 GiB RAM **and** 2 GiB address-space unit, peaking at 16,236,544 bytes, with all memory events zero.

External Rust library-test compilation used an explicitly approved build-only 4 GiB cgroup. It peaked at 2,277,605,376 charged bytes, including file cache; this is not a compiler RSS measurement. The sampled working-memory estimate peaked at 2,027,921,408 bytes. Windows host free memory remained above 9,249,596 KiB during this build; a 2-second host guardian would stop only the owned unit below 8.5 GiB, and stale telemetry failed closed. Neither the build nor execution used swap or had an OOM event. These build-instrument limits do not relax the final native consumer/compiler 2 GiB contract.

Earlier combined-target build attempts stopped at their 1.9 GB soft guard before tests. Another setup attempt refused before Cargo because an optional cgroup telemetry file was absent. These are incomplete attempts, not passing checks.

The first integration attempt's baseline results are valid, but its fixed results are invalid: sharing one sequential Cargo target directory allowed Cargo to reuse the baseline artifacts for a different source snapshot. That target and the receipts were preserved. The accepted fixed integration results came from a new target directory, with actual rustc invocations and binary hashes retained. Library compilation likewise used separate source-bound targets.

## Evidence inventory and review boundary

The private operator handoff retains exact commands, source/tool/executable hashes, stdout, stderr, exits, runtime prerequisite output, source-before/after receipts, resource counters and guard logs. Raw archives are identified here without importing unrelated private material:

| Evidence archive | SHA-256 |
| --- | --- |
| `compiler-abi-g1-r2-attempt1.tar` | `6b4aa14391e6166e95ec889269b79ce052bf52eeaa70272577da06e0da393c80` |
| `compiler-abi-g1-r2-attempts2-3.tar` | `a4e7a95d1fea3a721635d6ec83eb1cabab7f23e735fcc2b8b26856c35e72eb87` |
| `compiler-abi-g1-r2-integration4.tar` | `78117b4c491b526d747a9b6b3764fde6a143c191b0a758979fb48bc773460635` |
| `compiler-abi-g1-r2-fixed5.tar` | `783a48650372b5024961d1b065769410a65f99ed87917b4b16df13dedec3ac2c` |
| `compiler-abi-g1-r2-library6-7.tar` | `95cbd7ab2c7cd69eeaadcf3bc169b0791c30f7a302b6019ef70df71aa294650e` |

An independent source reviewer marked R2 `PASS_WITH_NOTES`; the earlier R1 failure remains recorded. That source review is distinct from the implementation owner's observed execution results and from the integrator's review. Review did not establish Windows executable behavior, AAPCS implementation correctness, full compiler-suite health or native consumer qualification. An inherited asymmetry between nested local extern hiding and imported-symbol lookup remains recorded as a broader resolution obligation; no wrong-result execution was established for it in this scope.

No seed files, source-integrity pins, accepted compiler hashes or MIND-Lab proof/release statuses were changed by this patch.
