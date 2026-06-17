//! Runnable-artifact ABI gate (release-readiness P1.1).
//!
//! A pure, read-only pre-pass over the parsed + type-checked AST that fails
//! LOUD on the constructs the *runnable* (object / shared-library) lowering
//! path would otherwise SILENTLY MISCOMPILE — so a stranger never receives a
//! wrong-but-`rc=0` executable artifact. For a deterministic, evidence-signing
//! compiler a confidently-wrong `.so` is the worst outcome: it carries a valid
//! signature over incorrect code.
//!
//! Scope is exactly the i64-scalar ABI the shipped backend lowers correctly.
//! The gate is enforced ONLY when emitting a runnable artifact (`--emit-obj` /
//! `--emit-shared`). The inspection / IR surfaces (`mindc check`, `--emit-ir`,
//! `--emit-mlir`, `--emit-mic3`) are intentionally left untouched: there a
//! `i32`/`u32`/`tensor` annotation is a perfectly valid *type* (and the test
//! suite type-checks such programs heavily) — it simply does not yet *lower*
//! to a correct runnable artifact.
//!
//! The walk NEVER mutates the module/IR and produces zero diagnostics for an
//! all-i64 program, so the mic@3 self-host fixed point stays byte-identical
//! (keystone 7/7). Verified false-positive-free against `std/`, `examples/`,
//! and `examples/mindc_mind/main.mind`.
//!
//! Gate set (the *silent* sub-i64-ABI miscompiles; genuinely loud paths —
//! e.g. float struct fields, unsupported tensor ops — are left to the existing
//! lowering errors). Sub-i64 struct *fields* are no longer gated: the
//! width-aware struct ABI lowers them with a canonical offset table + typed
//! store/load. The remaining gated constructs:
//!   * a `tensor` / `diff tensor` function parameter or return: erases to the
//!     i64 ABI and is silently treated as a scalar integer.
//!   * a function parameter or return declared at a sub-i64 integer width
//!     (`i32`/`u32`/`i8`/`u8`/`i16`/`u16`): the width and, for unsigned types,
//!     the semantics are silently lost.
//!
//! `extern "C"` blocks are EXEMPT — the C-ABI boundary legitimately declares
//! narrow ints and maps them to the platform contract.

use crate::ast::{Module, Node, Span as AstSpan, TypeAnn};
use crate::diagnostics::{Diagnostic, Span};

const PHASE: &str = "lower";
const HELP: &str = "the shipped backend lowers only the i64-scalar ABI; this construct is not yet \
     lowerable to a runnable artifact (RUNS burndown). Run it with the `mind` interpreter, or keep \
     the compiled path to the i64 subset.";

/// Walk the module and return one error diagnostic per construct the runnable
/// (object / shared-library) lowering path would silently miscompile. The
/// returned `Vec` is empty for an all-i64 program (no allocation on the happy
/// path beyond the empty `Vec`), and the module is never mutated.
pub fn check_runnable_lowerable(module: &Module, src: &str, file: Option<&str>) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    // Only function signatures gate now. Struct declarations no longer gate: the
    // width-aware struct ABI lowers sub-i64 fields (i32/u32/i16/u16/i8/u8/bool)
    // with a canonical per-field offset table + typed store/load, and float
    // fields remain loud via the downstream non-i64-call check. `extern "C"`
    // declarations and every other item are exempt — no other top-level item
    // reaches the runnable path with a silently-miscompiled non-i64 signature.
    for item in &module.items {
        let Node::FnDef {
            name,
            params,
            ret_type,
            span,
            ..
        } = item
        else {
            continue;
        };
        for p in params {
            if let Some(reason) = sig_non_i64(&p.ty) {
                out.push(mk(
                    src,
                    file,
                    p.span,
                    "lower::non_i64_param",
                    format!(
                        "parameter `{}` of `{name}` is not lowerable to a runnable artifact: \
                         {reason}",
                        p.name
                    ),
                ));
            }
        }
        if let Some(reason) = ret_type.as_ref().and_then(sig_non_i64) {
            out.push(mk(
                src,
                file,
                *span,
                "lower::non_i64_return",
                format!(
                    "return type of `{name}` is not lowerable to a runnable artifact: {reason}"
                ),
            ));
        }
    }
    out
}

/// Reason a function parameter/return `TypeAnn` cannot lower in the runnable
/// i64 ABI, or `None` when it lowers correctly (`i64`, `f32`, `f64`, `bool`,
/// a struct handle (`Named`), slice/ref/array/tuple — handled elsewhere or
/// already loud).
fn sig_non_i64(ty: &TypeAnn) -> Option<&'static str> {
    match ty {
        // `i32`/`u32` params & returns now lower correctly (real i32 MLIR with
        // signed/unsigned op selection + deterministic two's-complement wrap), so
        // they are no longer gated.
        TypeAnn::Tensor { .. } => Some(
            "a tensor-typed parameter/return erases to the i64 ABI and is treated as a scalar \
             integer",
        ),
        TypeAnn::DiffTensor { .. } => Some("a diff-tensor parameter/return erases to the i64 ABI"),
        // i8/u8/i16/u16 (as `Named`) still widen to i64 in a signature — they have
        // no dedicated ValueKind yet, so the i32 ABI does not cover them.
        TypeAnn::Named(n) if is_narrow_int_name(n) => {
            Some("a sub-i64 integer type silently widens to `i64`")
        }
        _ => None,
    }
}

/// Narrow integer type names that may reach the AST as `TypeAnn::Named`
/// (the lexer maps `i32`/`u32` to dedicated scalars; `i8`/`u8`/`i16`/`u16`
/// can arrive as named scalars). `u64`/`i64` are full-width and lower fine.
fn is_narrow_int_name(n: &str) -> bool {
    matches!(n, "i8" | "u8" | "i16" | "u16")
}

fn mk(src: &str, file: Option<&str>, span: AstSpan, code: &'static str, msg: String) -> Diagnostic {
    Diagnostic::error(PHASE, code, msg)
        .with_span(Span::from_offsets(src, span.start(), span.end(), file))
        .with_help(HELP)
}
