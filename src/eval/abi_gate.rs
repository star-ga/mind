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
//! lowering errors):
//!   * a struct field declared at a sub-i64 width (`i32`/`u32`/`i8`/`u8`/
//!     `i16`/`u16`/`bool`): the heap-record layout uses a fixed 8-byte stride
//!     and a full-width store, incompatible with the declared width.
//!   * a `tensor` / `diff tensor` function parameter or return: erases to the
//!     i64 ABI and is silently treated as a scalar integer.
//!   * a function parameter or return declared at a sub-i64 integer width
//!     (`i32`/`u32`/`i8`/`u8`/`i16`/`u16`): the width and, for unsigned types,
//!     the semantics are silently lost.
//!
//! `extern "C"` blocks are EXEMPT — the C-ABI boundary legitimately declares
//! narrow ints and maps them to the platform contract.

use crate::ast::{Attribute, Module, Node, Span as AstSpan, TypeAnn};
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
    for item in &module.items {
        match item {
            Node::FnDef {
                name,
                params,
                ret_type,
                span,
                ..
            } => {
                for p in params {
                    if let Some(reason) = sig_non_i64(&p.ty) {
                        out.push(mk(
                            src,
                            file,
                            p.span,
                            "lower::non_i64_param",
                            format!(
                                "parameter `{}` of `{name}` is not lowerable to a runnable \
                                 artifact: {reason}",
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
                            "return type of `{name}` is not lowerable to a runnable artifact: \
                             {reason}"
                        ),
                    ));
                }
            }
            Node::StructDef {
                name,
                fields,
                attrs,
                ..
            } => {
                // A `#[repr(C)]` struct is a foreign-boundary contract; narrow
                // fields there are intentional (handled by the extern path).
                if is_repr_c(attrs) {
                    continue;
                }
                for f in fields {
                    if let Some(reason) = field_non_i64(&f.ty) {
                        out.push(mk(
                            src,
                            file,
                            f.span,
                            "lower::non_i64_struct_field",
                            format!(
                                "field `{}` of struct `{name}` is not lowerable: {reason}",
                                f.name
                            ),
                        ));
                    }
                }
            }
            // `extern "C"` declarations and every other item are exempt: the
            // foreign boundary legitimately uses narrow ints, and no other
            // top-level item reaches the runnable lowering path with a
            // silently-miscompiled non-i64 signature.
            _ => {}
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
        TypeAnn::ScalarI32 => {
            Some("`i32` silently widens to `i64`, losing 32-bit width and wraparound")
        }
        TypeAnn::ScalarU32 => {
            Some("`u32` silently lowers to a signed `i64`, losing width and unsigned semantics")
        }
        TypeAnn::Tensor { .. } => Some(
            "a tensor-typed parameter/return erases to the i64 ABI and is treated as a scalar \
             integer",
        ),
        TypeAnn::DiffTensor { .. } => Some("a diff-tensor parameter/return erases to the i64 ABI"),
        TypeAnn::Named(n) if is_narrow_int_name(n) => {
            Some("a sub-i64 integer type silently widens to `i64`")
        }
        _ => None,
    }
}

/// Reason a struct field `TypeAnn` cannot lower (a sub-i64 width breaks the
/// fixed 8-byte-stride heap-record layout). Floats are left to the existing
/// loud lowering error; `i64` and struct-handle (`Named`) fields lower
/// correctly and are the records the self-host stack relies on.
fn field_non_i64(ty: &TypeAnn) -> Option<&'static str> {
    match ty {
        TypeAnn::ScalarI32 | TypeAnn::ScalarU32 | TypeAnn::ScalarBool => Some(
            "a sub-i64 field uses a fixed 8-byte stride and full-width store, incompatible with \
             its declared width",
        ),
        TypeAnn::Named(n) if is_narrow_int_name(n) => {
            Some("a sub-i64 field uses a fixed 8-byte stride, incompatible with its declared width")
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

fn is_repr_c(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.name == "repr" && a.args.iter().any(|x| x == "C"))
}

fn mk(src: &str, file: Option<&str>, span: AstSpan, code: &'static str, msg: String) -> Diagnostic {
    Diagnostic::error(PHASE, code, msg)
        .with_span(Span::from_offsets(src, span.start(), span.end(), file))
        .with_help(HELP)
}
