// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! E2005 for a wrong-arity `__mind_*` intrinsic call, in EVERY expression
//! position.
//!
//! A call to a registered `__mind_*` intrinsic carries exactly the intrinsic's
//! declared arity — the i64 ABI of RFC 0005. The rule used to be enforced only
//! inside `check_std_surface_intrinsic`, which runs only where `infer_expr`
//! reaches the call, and `infer_expr` skips whole positions: it types an `if`
//! by the last statement of its then-branch and a `for` by the last statement
//! of its body, types a `while` without visiting its condition or body (RFC
//! 0005 Gap 1), and never re-infers the arguments of an intra-module or
//! loose-ABI call (see `check_intra_fn_call`). A wrong-arity call in a skipped
//! position passed `mindc check` and was refused only at build time, by the IR
//! verifier (E3001).
//!
//! This pass owns the rule outright. It walks every node the AST has — the
//! exhaustive `node_children_ref` classification plus the function, closure and
//! impl-method bodies that walk treats as scope leaves — and reports E2005 at
//! each intrinsic call site whose argument count differs from the declared
//! arity, wherever the call sits. `check_std_surface_intrinsic` still validates
//! the argument types on the `infer_expr` path; the arity refusal itself has
//! exactly one source. `mindc check` desugars closures and impl methods into
//! functions before this pass runs; the closure and impl arms below serve the
//! callers that type-check a module as parsed (`build::native_scope`).

use super::AstSpan;
use super::CALL_ARITY_CODE;
use super::TypeEnv;
use super::TypeErrSpan;
use super::ValueType;
use super::describe_value_type;
use super::diag_from_span;
use super::infer_expr;
use crate::ast::{Module, Node};
use crate::diagnostics::Diagnostic as Pretty;

// RFC 0005 Phase 1 + 1.5 — pure-MIND standard surface intrinsics.
//
// The table itself moved to `crate::intrinsics` so that ONE row carries a
// primitive's name, its i64 arity, AND whether calling it can make the program's
// result vary between runs. It used to be split: the arity here, the determinism
// verdict in two hand-maintained parallel arrays (this module's and
// `ir::evidence`'s) that drifted identically and let the registered
// `__mind_nerve_rt_*` clock / entropy / stdin surface be attested
// `deterministic`. See the `crate::intrinsics` module docs.
pub(super) fn std_surface_intrinsic_arity(name: &str) -> Option<usize> {
    crate::intrinsics::std_surface_intrinsic_arity(name)
}

pub(super) fn check_std_surface_intrinsic(
    callee: &str,
    args: &[Node],
    span: AstSpan,
    env: &TypeEnv,
) -> Result<(ValueType, AstSpan), TypeErrSpan> {
    let arity = std_surface_intrinsic_arity(callee).ok_or_else(|| TypeErrSpan {
        msg: format!("unsupported call to `{callee}`"),
        span,
    })?;
    if args.len() != arity {
        // The arity rule is `intrinsic_call_arity`'s (E2005 at EVERY expression
        // position); this function only runs where `infer_expr` reaches the
        // call. Type the call as its i64 result and let that pass refuse it —
        // one source for one rule.
        return Ok((ValueType::ScalarI64, span));
    }
    for (i, a) in args.iter().enumerate() {
        let (ty, _) = infer_expr(a, env)?;
        // Integer literals come in as `ScalarI32` (line 562) but lower
        // to `ConstI64` in IR — accept both; reject anything else with
        // a clear i64-ABI message that points to RFC 0005 phase 2+.
        if !matches!(ty, ValueType::ScalarI32 | ValueType::ScalarI64) {
            return Err(TypeErrSpan {
                msg: format!(
                    "argument {i} to intrinsic `{callee}` must be i64 (got {}); the std-surface intrinsics use the i64 ABI — aggregate types are RFC 0005 phase 2+",
                    describe_value_type(&ty)
                ),
                span: a.span(),
            });
        }
    }
    Ok((ValueType::ScalarI64, span))
}

/// Report E2005 for every wrong-arity registered intrinsic call reachable in
/// `module`, at any depth and in any expression position.
pub(super) fn check(module: &Module, src: &str, file: Option<&str>, errs: &mut Vec<Pretty>) {
    for item in &module.items {
        check_node(item, src, file, errs);
    }
}

fn check_node(node: &Node, src: &str, file: Option<&str>, errs: &mut Vec<Pretty>) {
    if let Node::Call { callee, args, span } = node
        && let Some(arity) = std_surface_intrinsic_arity(callee)
        && args.len() != arity
    {
        errs.push(diag_from_span(
            src,
            file,
            format!(
                "intrinsic `{callee}` expects {arity} i64 argument(s) (i64 ABI — RFC 0005); got {}",
                args.len()
            ),
            *span,
            CALL_ARITY_CODE,
        ));
    }
    // `node_children_ref`'s exhaustive classification stops at `FnDef` /
    // `Closure` / `ImplBlock` (they open a scope, so the closure rewriter must
    // not descend). This walk is scope-blind — a wrong-arity intrinsic call in
    // a body is still a call site — so descend into those bodies explicitly.
    // `TraitDef` carries signatures only (no body) and is a leaf for both.
    match node {
        Node::FnDef(fd, _) => {
            for stmt in &fd.body {
                check_node(stmt, src, file, errs);
            }
        }
        Node::Closure(data, _) => {
            for stmt in &data.body {
                check_node(stmt, src, file, errs);
            }
        }
        Node::ImplBlock { methods, .. } => {
            for method in methods {
                check_node(method, src, file, errs);
            }
        }
        _ => {}
    }
    for child in crate::eval::closures::node_children_ref(node) {
        check_node(child, src, file, errs);
    }
}

/// E2041: a call the MLIR backend cannot lower, refused by name. Compiled
/// only with `mlir-build`: the refusal belongs to the MLIR route.
///
/// The set is read off the registry, never a second list: a `__mind_*` row
/// registered in some profile that the Rust/MLIR profile does not select (the
/// native-only AVX2 ymm, CPU-feature, timing and shared-memory rows). Such a
/// call used to type-check, lower to an external call and fail in the linker
/// with an undefined reference. `pipeline` records these with the runnable
/// blockers, so every MLIR build refuses before it lowers or links. `mindc
/// check`, `--emit-mic3` and `--emit-evidence` stay backend-neutral: the native
/// backend lowers these rows.
#[cfg(feature = "mlir-build")]
pub(crate) const MLIR_UNLOWERED_INTRINSIC_CODE: &str = "E2041";

/// Report E2041 at every call, in any expression position, whose callee has no
/// MLIR lowering.
#[cfg(feature = "mlir-build")]
pub(crate) fn mlir_unlowered_intrinsic_calls(
    module: &Module,
    src: &str,
    file: Option<&str>,
) -> Vec<Pretty> {
    let mut out = Vec::new();
    for item in &module.items {
        super::nerve_walk::walk(item, &mut |node| {
            if let Node::Call { callee, span, .. } = node
                && mlir_has_no_lowering(callee)
            {
                out.push(diag_from_span(
                    src,
                    file,
                    format!(
                        "intrinsic `{callee}` has no MLIR lowering: it is a native-backend row. \
                         Build with `--backend native`, or remove the call"
                    ),
                    *span,
                    MLIR_UNLOWERED_INTRINSIC_CODE,
                ));
            }
        });
    }
    out
}

/// A registered intrinsic that the Rust/MLIR profile does not select.
#[cfg(feature = "mlir-build")]
fn mlir_has_no_lowering(callee: &str) -> bool {
    crate::intrinsics::registered_intrinsic_arity(callee).is_some()
        && std_surface_intrinsic_arity(callee).is_none()
}

#[cfg(all(test, feature = "mlir-build"))]
mod mlir_unlowered_tests {
    use super::*;
    use crate::intrinsics::{IntrinsicProfile, STD_SURFACE_INTRINSICS};

    /// The E2041 diagnostics for one call to `name` with `arity` zero arguments.
    fn refusals(name: &str, arity: usize) -> Vec<Pretty> {
        let args = vec!["0"; arity].join(", ");
        let src =
            format!("fn main() -> i64 {{\n    let r: i64 = {name}({args});\n    return 0;\n}}\n");
        let module = crate::parser::parse(&src).expect("fixture parses");
        mlir_unlowered_intrinsic_calls(&module, &src, Some("p.mind"))
    }

    /// E2041 refuses exactly the registry rows without the Rust/MLIR profile,
    /// each by name. The expected set is read off the registry, never a literal
    /// count, so a row registered later is covered the day it lands.
    #[test]
    fn every_native_only_row_is_refused_by_name_and_no_mlir_row_is() {
        let mut expected = Vec::new();
        let mut refused = Vec::new();
        for spec in STD_SURFACE_INTRINSICS {
            let mlir = spec
                .contract
                .is_none_or(|c| c.profiles.contains(&IntrinsicProfile::RustMlir));
            if !mlir {
                expected.push(spec.physical_name);
            }
            let diags = refusals(spec.physical_name, spec.arity);
            match diags.as_slice() {
                [] => {}
                [d] => {
                    assert_eq!(d.code, MLIR_UNLOWERED_INTRINSIC_CODE);
                    assert!(
                        d.message.contains(&format!("`{}`", spec.physical_name)),
                        "{}",
                        d.message
                    );
                    refused.push(spec.physical_name);
                }
                many => panic!("{}: {} E2041 diagnostics", spec.physical_name, many.len()),
            }
        }
        println!("E2041 refused rows ({}): {refused:?}", refused.len());
        assert_eq!(
            refused, expected,
            "E2041 must refuse exactly the rows without the Rust/MLIR profile"
        );
        assert!(
            !expected.is_empty(),
            "no native-only row: the refusal leg ran on nothing"
        );
        for name in [
            "__mind_ymm_vmovdqu_ld",
            "__mind_vpaddd_i32x8",
            "__mind_cpu_relax",
        ] {
            assert!(refused.contains(&name), "{name}");
        }
        for name in ["__mind_argc", "__mind_argv", "__mind_open"] {
            assert!(refusals(name, 0).is_empty(), "{name} lowers on MLIR");
        }
    }

    /// A call in a position the type inference skips (a `while` body) is
    /// still refused.
    #[test]
    fn a_native_only_call_in_a_loop_body_is_refused() {
        let src = "fn main() -> i64 {\n    let mut i: i64 = 0;\n    while i < 1 {\n        __mind_cpu_relax();\n        i = i + 1;\n    }\n    return 0;\n}\n";
        let module = crate::parser::parse(src).expect("fixture parses");
        let diags = mlir_unlowered_intrinsic_calls(&module, src, None);
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("`__mind_cpu_relax`"));
    }
}
