# Audit Report

## Summary Table
| Category | Severity | Location(s) | Impact | Suggested Fix |
| --- | --- | --- | --- | --- |
| FFI allocation overflow | High | `src/ffi/mod.rs::mind_alloc` | Passing a `size` larger than `usize::MAX` truncates the allocation request and can return a pointer to too-small memory or fail unpredictably. | Reject oversized requests and surface an error before calling `malloc`. |
| Codebase not `cargo fmt --check` clean | Medium | Multiple files (e.g., `src/eval/conv2d_grad.rs`) | CI-quality gap: formatting drift causes `cargo fmt --check` to fail, blocking automated pipelines. | Run `cargo fmt` or update CI expectations to keep tree format-clean. |
| Missing LLVM toolchain for MLIR/tblgen builds | High | `tblgen` build scripts invoked by `cargo clippy/test` | Lints and tests fail early because `llvm-config` 19.x is unavailable, preventing verification of MLIR-dependent crates. | Provide LLVM 19 `llvm-config` in CI/dev env or gate MLIR-dependent targets when unavailable. |
| Missing security tooling | Medium | `cargo audit`, `cargo deny` commands absent | Vulnerability/license scanning cannot run, leaving supply-chain issues unchecked. | Install `cargo-audit` and `cargo-deny` (or provide alternate scanning) in CI. |

## Detailed Findings

### 1) FFI allocation overflow
- **Location:** `src/ffi/mod.rs`, `mind_alloc`.
- **Issue:** The function casts a user-provided `u64` size directly to `usize` before calling `libc::malloc`. On 32-bit or other limited-pointer targets, values greater than `usize::MAX` wrap, leading to undersized allocations and potential buffer overruns for callers that trust the requested size.
- **Impact:** Potential undefined behavior or memory corruption when the FFI boundary is used on narrower architectures or with unvalidated inputs.
- **Evidence:** Code path lacks bounds check prior to `malloc`.
- **Fix:** Reject requests exceeding `usize::MAX`, surface an error via the existing `LAST_ERROR` channel before returning a null pointer, and cover the guard with regression test `ffi::capi::tests::alloc_rejects_oversized_request`. (Implemented.)

### 2) Formatting drift blocks `cargo fmt --check`
- **Location:** Several files including `src/eval/conv2d_grad.rs`, `src/eval/ir_interp.rs`, `src/eval/mod.rs`, `src/eval/stdlib/tensor.rs`, `src/ir/mod.rs`, `tests/conv2d_exec.rs`, and `tests/conv2d_grad.rs`.
- **Issue:** `cargo fmt --all -- --check` reports diffs in the current tree, so format-check CI would fail even without code changes.
- **Impact:** Prevents automated linting pipelines from passing and obscures signal from other checks.
- **Repro:** `cargo fmt --all -- --check` (fails with diff output).
- **Suggested Fix:** Run `cargo fmt` across the repository or update CI to enforce/accept the existing style.

### 3) LLVM toolchain missing for MLIR/tblgen targets
- **Location:** Build step for `tblgen` dependency used by MLIR components.
- **Issue:** `cargo clippy --all-targets --all-features` and both test suites fail because `llvm-config` 19.x is not available in the environment, so `tblgen`'s build script aborts.
- **Impact:** Lints and tests cannot run, hiding potential regressions or UB in MLIR-dependent code.
- **Repro:** `cargo clippy --all-targets --all-features -- -D warnings` or `cargo test --all --all-features` fails with `failed to find correct version (19.x.x) of llvm-config`.
- **Suggested Fix:** Ensure LLVM 19 toolchain is installed and on PATH in CI/dev images, or add feature gating to skip MLIR-dependent crates when the toolchain is absent.

### 4) Missing vulnerability/license scanning tooling
- **Location:** Workspace root tooling.
- **Issue:** `cargo audit` and `cargo deny check` are unavailable, so dependency vulnerability and license policies are unchecked.
- **Impact:** Potential security or compliance issues could pass unnoticed.
- **Repro:** Running `cargo audit` or `cargo deny check` fails because the subcommands are not installed.
- **Suggested Fix:** Add `cargo-audit` and `cargo-deny` to CI images and wire them into pipelines.

## False Positives / Won't Fix
- None identified beyond the environment/tooling gaps noted above.

## Risky Areas to Re-check After Changes
- Re-run FFI tests to ensure `LAST_ERROR` handling behaves correctly across threads after the new allocation guard.
- Once LLVM 19 tooling is available, re-run `cargo clippy` and the full test suite (including doc tests) with all features to uncover any MLIR/inkwell regressions.
- After repository-wide formatting is addressed, re-run `cargo fmt --check` in CI to confirm consistency.
