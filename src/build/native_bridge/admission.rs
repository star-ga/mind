// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Source-level refusals the native bridge applies to the merged program.
//!
//! * An entry `main` that declares parameters is refused, except exactly
//!   `fn main(argc: i64, argv: i64)` (W4b). A compiler seed that predates W4b
//!   does not bind `main`'s parameters to the process arguments, so such a
//!   program ran with every parameter 0 (`fn main(argc: i64, argv: i64)` built
//!   and exited 40 natively where the MLIR build exited 23). The admitted shape
//!   is therefore only CONDITIONALLY admitted: the bridge checks the emitted
//!   entry for the argc/argv binding and refuses the artifact when it is missing
//!   (`native_bridge::entry::check_entry_binds_main_params`).
//! * The name check the MLIR build path treats as fatal (an unknown identifier
//!   E2002, a function used as a value E2037) and an unknown enum variant
//!   (E2008) are refused with the type checker's own diagnostics. Without the
//!   check these reached lowering's fail-closed `panic!` and the build died
//!   with exit 101.
//! * A symbolic tensor dimension read as a VALUE (`reshape(x, [K, 1])` where
//!   `K` is only a dimension of a parameter's type) is refused. The type
//!   checker deliberately lets the name resolve, but lowering has no value for
//!   it and panicked on the same undefined-identifier guard.
//! * A binding that re-declares a name from an enclosing scope is refused
//!   (`native.loop_body_shadow` inside a loop, `native.block_shadow` outside
//!   one). The frozen compiler resolves nested bindings by name and merges the
//!   two, so such programs built and ran with the inner value.
//! * Lowering itself runs behind a panic boundary, so any other construct that
//!   still reaches a lowering `panic!` becomes a refusal with its message, not
//!   an exit 101.

use std::collections::BTreeSet;

use crate::ast::{Module, Node};
use crate::diagnostics::{
    ColorChoice, Diagnostic, DiagnosticEmitter, DiagnosticFormat, Severity, Span,
};
use crate::type_checker::resolve;

/// What the entry `main`'s parameter list asks of the native compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MainParams {
    /// `fn main()` (or no entry `main` at all).
    None,
    /// Exactly `fn main(argc: i64, argv: i64)`: admitted on condition that the
    /// compiler in use binds both to the process argc / `char **argv`, which the
    /// bridge verifies on the emitted entry before writing the artifact.
    ArgcArgv,
}

/// Refuse an entry `main` that declares parameters, except the one shape the
/// native compiler binds (W4b): exactly two `i64` parameters, read as argc and
/// argv (`char **`). Any other count or type -- `(argc)` alone, a third `envp`,
/// an `i32`/`f64` or aliased type -- is still refused by name.
pub(super) fn refuse_entry_main_with_params(
    ast: &Module,
    path: &str,
) -> Result<MainParams, String> {
    for item in &ast.items {
        let Node::FnDef(f, _) = item else { continue };
        if f.name != "main" || f.params.is_empty() {
            continue;
        }
        if f.params.len() == 2
            && f.params
                .iter()
                .all(|p| matches!(p.ty, crate::ast::TypeAnn::ScalarI64))
        {
            return Ok(MainParams::ArgcArgv);
        }
        let params: Vec<&str> = f.params.iter().map(|p| p.name.as_str()).collect();
        return Err(format!(
            "`{path}`: the entry `fn main` declares parameter(s) {params:?}, but the native \
             backend binds only `fn main(argc: i64, argv: i64)` to the process arguments; \
             any other parameter list would run with garbage values (a silent miscompile). \
             Declare `fn main()` or `fn main(argc: i64, argv: i64)`, or rerun with \
             `--backend mlir`. No artifact written. (native.entry_main_has_params)"
        ));
    }
    Ok(MainParams::None)
}

/// Refuse a binding that re-declares a name from an enclosing scope.
///
/// The frozen native compiler resolves variables by name across nested
/// blocks, so it merges such a binding into the outer one. Inside a loop:
/// `let mut x = 5; while k < 2 { k = k + 1; let mut x = 7; x = 3; } return x;`
/// built natively and exited 3 where the MLIR build exits 5
/// (`native.loop_body_shadow`). Outside any loop, a write from a further-nested
/// block lands on the outer binding: `let mut x = 5; if c { let mut x = 9;
/// if c { x = 40; } } return x;` exited 40 natively, 5 under MLIR
/// (`native.block_shadow`). Refused, with every offending binding named, until
/// the seed scopes nested bindings. The scope rules are in `shadow`; when
/// unsure they refuse.
///
/// `compiler_scopes` reports whether the compiler that will run carries the
/// scoped-nested-bindings mark (`seed::compiler_scopes_nested_bindings`); it
/// is asked only when a shadow exists. When it does, the shapes `shadow` marks
/// `proven` (a plain `let` over an outer `let` or parameter across statement
/// `if`/`else` blocks and `while` loops only) are admitted; every other shadow
/// is still refused. A compiler without the mark merges nested bindings, so
/// nothing is admitted.
///
/// `origins` maps the composed user region back to its source files: each
/// entry is a file's display path and the byte offset where it starts in
/// `text`, in ascending order, so a binding in an imported module is reported
/// at that module's own file and line.
pub(super) fn refuse_shadowing(
    ast: &Module,
    text: &str,
    origins: &[(String, usize)],
    compiler_scopes: &dyn Fn() -> bool,
) -> Result<(), String> {
    let mut shadows = super::shadow::find(ast);
    if shadows.is_empty() {
        return Ok(());
    }
    if shadows.iter().any(|s| s.proven) && compiler_scopes() {
        shadows.retain(|s| !s.proven);
        if shadows.is_empty() {
            return Ok(());
        }
    }
    let locate = |offset: usize| {
        let offset = offset.min(text.len());
        let i = origins
            .iter()
            .rposition(|(_, start)| *start <= offset)
            .unwrap_or(0);
        let (path, start) = origins
            .get(i)
            .map_or(("<input>", 0), |(p, s)| (p.as_str(), *s));
        let end = origins.get(i + 1).map_or(text.len(), |(_, s)| *s);
        (path, start.min(text.len()), end.min(text.len()))
    };
    let mut rendered = Vec::new();
    for s in &shadows {
        let (code, place) = if s.in_loop {
            ("native.loop_body_shadow", "inside a loop")
        } else {
            ("native.block_shadow", "in a nested block")
        };
        let (outer_path, outer_lo, _) = locate(s.outer.start());
        let outer_end = s.outer.start().clamp(outer_lo, text.len());
        let outer_line = text
            .get(outer_lo..outer_end)
            .map_or(0, |t| t.matches('\n').count())
            + 1;
        let mut d = Diagnostic::error(
            "native",
            code,
            format!(
                "{} `{}` {place} in fn `{}` re-declares `{}`, already bound as a {} at \
                 {outer_path}:{outer_line}; the frozen native compiler resolves nested \
                 bindings by name and would merge the two (a silent miscompile)",
                s.inner_kind, s.name, s.fn_name, s.name, s.outer_kind,
            ),
        );
        let (path, lo, hi) = locate(s.inner.start());
        // `text` is valid UTF-8 and every origin starts a file, so `lo..hi` is
        // on char boundaries; a `\n` joiner at `hi - 1` is outside the file.
        let file = text.get(lo..hi).unwrap_or("");
        let start = s.inner.start().saturating_sub(lo).min(file.len());
        let end = s.inner.end().saturating_sub(lo).min(file.len()).max(start);
        d.span = Some(Span::from_offsets(file, start, end, Some(path)));
        d.help = Some(format!(
            "rename the inner `{}`, or rerun with `--backend mlir`",
            s.name
        ));
        rendered.push(render(std::slice::from_ref(&d), file));
    }
    let names: BTreeSet<&str> = shadows.iter().map(|s| s.name.as_str()).collect();
    let mut codes: Vec<&str> = Vec::new();
    if shadows.iter().any(|s| s.in_loop) {
        codes.push("native.loop_body_shadow");
    }
    if shadows.iter().any(|s| !s.in_loop) {
        codes.push("native.block_shadow");
    }
    Err(format!(
        "{}\nbinding(s) {names:?} re-declare a name from an enclosing scope; the native \
         backend would merge them with the outer binding. Rerun with `--backend mlir`. \
         No artifact written. ({})",
        rendered.join("\n"),
        codes.join(", ")
    ))
}

/// The type-check codes whose construct lowering cannot materialise.
const NAME_FATAL_CODES: [&str; 3] = [
    resolve::UNKNOWN_IDENT_CODE,
    resolve::UNKNOWN_VARIANT_CODE,
    crate::type_checker::FN_AS_VALUE_CODE,
];

fn render(diags: &[Diagnostic], text: &str) -> String {
    let emitter = DiagnosticEmitter::new(DiagnosticFormat::Human, ColorChoice::Never);
    diags
        .iter()
        .map(|d| emitter.render_human(d, Some(text)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Refuse unresolved names, using the type checker's own diagnostics.
pub(super) fn refuse_unresolved_names(ast: &Module, text: &str, path: &str) -> Result<(), String> {
    let fatal: Vec<Diagnostic> = crate::type_checker::check_module_types_in_file(
        ast,
        text,
        Some(path),
        &crate::type_checker::TypeEnv::default(),
    )
    .into_iter()
    .filter(|d| d.severity == Severity::Error && NAME_FATAL_CODES.contains(&d.code))
    .collect();
    if fatal.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{}\n`{path}`: unresolved name(s) above; the native backend refuses a program it \
         cannot lower. No artifact written. (native.unresolved_name)",
        render(&fatal, text)
    ))
}

/// Refuse a symbolic tensor dimension that a function body reads as a value.
pub(super) fn refuse_symbolic_dims_read_as_values(
    ast: &Module,
    text: &str,
    path: &str,
) -> Result<(), String> {
    let mut found: Vec<Diagnostic> = Vec::new();
    for item in &ast.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| {
            let Node::FnDef(f, _) = node else { return };
            let mut dims = BTreeSet::new();
            for p in &f.params {
                resolve::collect_shape_vars(&p.ty, &mut dims);
            }
            if let Some(rt) = &f.ret_type {
                resolve::collect_shape_vars(rt, &mut dims);
            }
            if dims.is_empty() {
                return;
            }
            // Resolve WITHOUT the dimension names the type checker pre-binds:
            // what is then unresolved and IS a dimension name is a dimension
            // read as a value. Everything else keeps the type checker's verdict.
            let injected: BTreeSet<String> = f.type_params.iter().cloned().collect();
            let params: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
            let (unresolved, _) = resolve::resolve_fn_body(&f.body, &params, ast, &injected);
            for u in unresolved {
                if u.is_call || u.variant_of.is_some() || u.undeclared_assign {
                    continue;
                }
                if !dims.contains(&u.name) {
                    continue;
                }
                let start = u.span.start().min(text.len());
                let end = u.span.end().min(text.len());
                let mut d = Diagnostic::error(
                    "native",
                    "native.symbolic_dim_as_value",
                    format!(
                        "symbolic tensor dimension `{}` of fn `{}` is read as a value; the \
                         native backend has no runtime value for a symbolic dimension",
                        u.name, f.name
                    ),
                );
                d.span = Some(Span::from_offsets(text, start, end, Some(path)));
                d.help = Some(
                    "pass the extent as an `i64` parameter, or rerun with `--backend mlir`"
                        .to_string(),
                );
                found.push(d);
            }
        });
    }
    if found.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{}\n`{path}`: symbolic dimension(s) read as values; no artifact written. \
         (native.symbolic_dim_as_value)",
        render(&found, text)
    ))
}

/// Lower behind a panic boundary: a lowering `panic!` becomes a refusal.
pub(super) fn lower_without_panicking(
    ast: &Module,
    path: &str,
) -> Result<crate::ir::IRModule, String> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let lowered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::eval::lower_to_ir(ast)
    }));
    std::panic::set_hook(previous);
    match lowered {
        Ok(Ok(ir)) => Ok(ir),
        Ok(Err(refusal)) => Err(format!(
            "`{path}`: {refusal} (lower.materialization_refused)"
        )),
        Err(payload) => {
            let why = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("(no message)");
            Err(format!(
                "`{path}`: lowering refused the program: {why} No artifact written. \
                 (lower.refused_construct)"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(src: &str) -> Result<(), String> {
        let ast = crate::parser::parse(src).expect("parse");
        refuse_symbolic_dims_read_as_values(&ast, src, "t.mind")
    }

    #[test]
    fn a_dimension_read_as_a_value_is_refused() {
        let err =
            dims("fn f(x: tensor<f32[K]>) -> tensor<f32[K, 1]> { return reshape(x, [K, 1]); }\n")
                .unwrap_err();
        assert!(err.contains("`K`") && err.contains("t.mind:1:"), "{err}");
    }

    #[test]
    fn a_dimension_used_only_in_types_is_admitted() {
        assert!(dims("fn f(x: tensor<f32[K]>) -> tensor<f32[K]> { return x; }\n").is_ok());
    }

    #[test]
    fn a_local_or_module_binding_of_the_same_name_is_admitted() {
        assert!(dims("fn f(x: tensor<f32[K]>) -> i64 { let K = 3; return K; }\n").is_ok());
        assert!(dims("const K: i64 = 4;\nfn f(x: tensor<f32[K]>) -> i64 { return K; }\n").is_ok());
        assert!(dims("fn f(x: tensor<f32[K]>, K: i64) -> i64 { return K; }\n").is_ok());
    }

    #[test]
    fn only_the_argc_argv_main_is_conditionally_admitted() {
        let main_params = |src: &str| {
            let ast = crate::parser::parse(src).expect("parse");
            refuse_entry_main_with_params(&ast, "t.mind")
        };
        assert_eq!(
            main_params("fn main() -> i64 { return 0; }\n"),
            Ok(MainParams::None)
        );
        assert_eq!(
            main_params("fn h(a: i64) -> i64 { return a; }\nfn main() -> i64 { return h(1); }\n"),
            Ok(MainParams::None)
        );
        assert_eq!(
            main_params("fn main(argc: i64, argv: i64) -> i64 { return argc; }\n"),
            Ok(MainParams::ArgcArgv)
        );
        for refused in [
            "fn main(argc: i64) -> i64 { return argc; }\n",
            "fn main(a: i64, b: i64, c: i64) -> i64 { return a; }\n",
            "fn main(argc: i32, argv: i64) -> i64 { return 0; }\n",
            "fn main(argc: i64, argv: f64) -> i64 { return argc; }\n",
        ] {
            assert!(main_params(refused).is_err(), "{refused}");
        }
    }

    fn shadowed(src: &str) -> Vec<String> {
        let ast = crate::parser::parse(src).expect("parse");
        super::super::shadow::find(&ast)
            .into_iter()
            .map(|s| {
                let place = if s.in_loop { "loop" } else { "block" };
                format!("{}:{}<{}@{place}", s.name, s.inner_kind, s.outer_kind)
            })
            .collect()
    }

    #[test]
    fn loop_body_shadows_are_found_at_every_nesting() {
        let w = "fn main() -> i64 { let mut x = 5; let mut k = 0; \
                 while k < 2 { k = k + 1; let mut x = 7; x = 3; } return x; }";
        assert_eq!(shadowed(w), ["x:let<let@loop"]);
        let p = "fn f(x: i64) -> i64 { for i in 0..2 { let x = 1; } return x; }";
        assert_eq!(shadowed(p), ["x:let<parameter@loop"]);
        let in_if = "fn f(x: i64) -> i64 { while x < 2 { if x == 1 { let x = 1; } } return x; }";
        assert_eq!(shadowed(in_if), ["x:let<parameter@loop"]);
        let nested = "fn f() -> i64 { while 1 { let a = 1; while 1 { let a = 2; } } return 0; }";
        assert_eq!(shadowed(nested), ["a:let<let@loop"]);
        let for_var = "fn f() -> i64 { for i in 0..3 { let i = 10; } return 0; }";
        assert_eq!(shadowed(for_var), ["i:let<for variable@loop"]);
        let nested_for = "fn f() -> i64 { for i in 0..3 { for i in 0..2 { } } return 0; }";
        assert_eq!(shadowed(nested_for), ["i:for variable<for variable@loop"]);
        let header = "fn f(i: i64) -> i64 { for i in 0..3 { } return i; }";
        assert_eq!(shadowed(header), ["i:for variable<parameter@loop"]);
        let tuple = "fn f(b: i64) -> i64 { while 1 { let (a, b) = (1, 2); } return b; }";
        assert_eq!(shadowed(tuple), ["b:let<parameter@loop"]);
        let konst = "const N: i64 = 4;\nfn f() -> i64 { while 1 { let N = 3; } return N; }";
        assert_eq!(shadowed(konst), ["N:let<const@loop"]);
    }

    #[test]
    fn block_shadows_outside_loops_are_found_at_every_nesting() {
        let one = "fn f(x: i64) -> i64 { if x == 1 { let x = 7; } return x; }";
        assert_eq!(shadowed(one), ["x:let<parameter@block"]);
        let two = "fn f() -> i64 { let mut x = 5; if 1 { let mut x = 9; if 1 { x = 40; } } \
                   return x; }";
        assert_eq!(shadowed(two), ["x:let<let@block"]);
        let deeper = "fn f() -> i64 { let x = 5; if 1 { if 1 { let x = 9; } } return x; }";
        assert_eq!(shadowed(deeper), ["x:let<let@block"]);
        let els = "fn f() -> i64 { let x = 5; if 1 { } else { let x = 9; } return x; }";
        assert_eq!(shadowed(els), ["x:let<let@block"]);
        let arm = "fn f(v: i64) -> i64 { let y = 5; match v { y => { return y; } } }";
        assert_eq!(shadowed(arm), ["y:match binding<let@block"]);
        let konst = "const N: i64 = 4;\nfn f() -> i64 { let N = 3; return N; }";
        assert_eq!(shadowed(konst), ["N:let<const@block"]);
        let param_konst = "const N: i64 = 4;\nfn f(N: i64) -> i64 { return N; }";
        assert_eq!(shadowed(param_konst), ["N:parameter<const@block"]);
        let block_then_loop = "fn f() -> i64 { let x = 5; if 1 { let x = 9; while 1 { } } \
                               return x; }";
        assert_eq!(shadowed(block_then_loop), ["x:let<let@block"]);
    }

    #[test]
    fn same_frame_redeclarations_and_fresh_names_are_admitted() {
        for ok in [
            "fn f(x: i64) -> i64 { let x = x + 1; if x == 1 { x = 7; } return x; }",
            "fn f() -> i64 { let x = 1; let x = x + 1; if 1 { if 1 { x = 3; } } return x; }",
            "fn f() -> i64 { if 1 { let t = 1; } if 1 { let t = 2; if 1 { t = 3; } } return 0; }",
            "fn f() -> i64 { let mut s = 0; while s < 3 { let t = 1; s = s + t; } return s; }",
            "fn f() -> i64 { while 1 { let t = 1; } while 1 { let t = 2; } return 0; }",
            "fn f() -> i64 { while 1 { let a = 1; let a = a + 1; } return 0; }",
            "fn f() -> i64 { while 1 { let t = 1; } let t = 2; return t; }",
            "fn f() -> i64 { for j in 0..2 { } for j in 0..4 { } return 0; }",
            "fn f() -> i64 { while 1 { if 1 { let q = 1; } let q = 2; } return 0; }",
            "fn f() -> i64 { while 1 { let _ = 1; let _ = 2; } return 0; }",
        ] {
            assert_eq!(shadowed(ok), Vec::<String>::new(), "{ok}");
        }
    }

    fn proven(src: &str) -> Vec<(String, bool)> {
        let ast = crate::parser::parse(src).expect("parse");
        super::super::shadow::find(&ast)
            .into_iter()
            .map(|s| (s.name, s.proven))
            .collect()
    }

    #[test]
    fn only_lets_across_if_blocks_and_while_loops_are_proven() {
        let yes = [
            "fn f() -> i64 { let mut x: i64 = 5; if 1 { let mut x: i64 = 9; if 1 { x = 40; } } \
             return x; }",
            "fn f(x: i64) -> i64 { if x == 1 { let x: i64 = 7; } return x; }",
            "fn f() -> i64 { let x: i64 = 5; if 1 { } else { let x: i64 = 9; } return x; }",
            "fn f() -> i64 { let mut x: i64 = 5; let mut k: i64 = 0; \
             while k < 2 { k = k + 1; let mut x: i64 = 7; x = 3; } return x; }",
            "fn f() -> i64 { while 1 { let a: i64 = 1; while 1 { if 1 { let a: i64 = 2; } } } \
             return 0; }",
            "fn f(x: i64) -> i64 { while x < 2 { if x == 1 { let x: i64 = 1; } } return x; }",
            // the other proven scalar types, mixed across the pair
            "fn f(x: i32) -> i64 { if x == 1 { let x: i64 = 7; } return 0; }",
            "fn f() -> i64 { let x: i64 = 5; if 1 { let x: f64 = 2.5; } return x; }",
        ];
        for src in yes {
            let got = proven(src);
            assert!(
                !got.is_empty() && got.iter().all(|(_, p)| *p),
                "{src}: {got:?}"
            );
        }
        let no = [
            // for / for-each variables and bodies
            "fn f(x: i64) -> i64 { for i in 0..2 { let x = 1; } return x; }",
            "fn f() -> i64 { for i in 0..3 { let i = 10; } return 0; }",
            "fn f(i: i64) -> i64 { for i in 0..3 { } return i; }",
            // tuple let, module constant, match binding / arm
            "fn f(b: i64) -> i64 { while 1 { let (a, b) = (1, 2); } return b; }",
            "const N: i64 = 4;\nfn f() -> i64 { while 1 { let N = 3; } return N; }",
            "fn f(v: i64) -> i64 { let y = 5; match v { y => { return y; } } }",
            "fn f(v: i64) -> i64 { let y = 5; match v { 1 => { let y = 2; return y; } _ => 0 } }",
            // a `let` inside a value-`if` branch is an expression position
            "fn f() -> i64 { let x: i64 = 5; let y: i64 = if 1 { let x: i64 = 9; x } else { 0 }; \
             return x + y; }",
            // an unannotated binding on either side, or a type the proof does not cover
            "fn f() -> i64 { let mut x = 5; if 1 { let mut x: i64 = 9; x = 1; } return x; }",
            "fn f() -> i64 { let mut x: i64 = 5; if 1 { let mut x = 9; x = 1; } return x; }",
            "fn f() -> i64 { let x: bool = true; if 1 { let x: i64 = 9; } return 0; }",
            "fn f(x: f32) -> i64 { if 1 { let x: i64 = 9; } return 0; }",
        ];
        for src in no {
            let got = proven(src);
            assert!(
                !got.is_empty() && got.iter().all(|(_, p)| !*p),
                "{src}: {got:?}"
            );
        }
        // A `for` anywhere between the two bindings makes the pair unproven,
        // even when the inner frame is an `if`.
        let mixed = "fn f() -> i64 { let x: i64 = 1; for i in 0..2 { if 1 { let x: i64 = 2; } } return x; }";
        assert_eq!(proven(mixed), [("x".to_string(), false)]);
    }

    #[test]
    fn a_scoping_compiler_admits_only_the_proven_shadows() {
        let ok = "fn main() -> i64 {\n    let mut x: i64 = 5;\n    if 1 {\n        \
                  let mut x: i64 = 9;\n        if 1 { x = 40; }\n    }\n    return x;\n}\n";
        let ast = crate::parser::parse(ok).expect("parse");
        let o = [("t.mind".to_string(), 0)];
        assert!(refuse_shadowing(&ast, ok, &o, &|| true).is_ok());
        let err = refuse_shadowing(&ast, ok, &o, &|| false).unwrap_err();
        assert!(err.contains("native.block_shadow"), "{err}");
        // Mixed: the proven `x` is admitted, the `for` variable is still refused
        // and is the only binding named.
        let mixed = "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let i: i64 = 1;\n    \
                     if 1 { let x: i64 = 2; }\n    for i in 0..2 { }\n    return x + i;\n}\n";
        let ast = crate::parser::parse(mixed).expect("parse");
        let err = refuse_shadowing(&ast, mixed, &o, &|| true).unwrap_err();
        assert!(err.contains("`i`") && !err.contains("`x`"), "{err}");
        // The mark probe is not consulted when nothing shadows.
        let none = "fn main() -> i64 { let x: i64 = 1; return x; }";
        let ast = crate::parser::parse(none).expect("parse");
        assert!(refuse_shadowing(&ast, none, &o, &|| panic!("asked")).is_ok());
    }

    #[test]
    fn each_function_starts_a_fresh_scope_chain() {
        // `x` is a parameter of `g` only; the loop in `h` binds a fresh `x`.
        let src = "fn g(x: i64) -> i64 { return x; }\n\
                   fn h() -> i64 { while 1 { let x = 1; } return 0; }";
        assert_eq!(shadowed(src), Vec::<String>::new());
    }

    #[test]
    fn the_refusal_names_the_binding_and_its_line() {
        let src = "fn main() -> i64 {\n    let mut x: i64 = 5;\n    while 1 {\n        \
                   let mut x: i64 = 7;\n    }\n    return x;\n}\n";
        let ast = crate::parser::parse(src).expect("parse");
        let err = refuse_shadowing(&ast, src, &[("t.mind".to_string(), 0)], &|| false).unwrap_err();
        assert!(err.contains("native.loop_body_shadow"), "{err}");
        assert!(!err.contains("native.block_shadow"), "{err}");
        assert!(err.contains("t.mind:4:"), "{err}");
        assert!(err.contains("at t.mind:2"), "{err}");
        assert!(
            err.contains("`x`") && err.contains("--backend mlir"),
            "{err}"
        );
    }

    #[test]
    fn a_refusal_in_a_second_file_names_that_file_and_its_own_line() {
        // Two files joined with one `\n`, as `compose_user_region` does.
        let main = "fn main() -> i64 {\n    return h(3);\n}\n";
        let helper = "fn h(a: i64) -> i64 {\n    let mut acc: i64 = a;\n    if a > 0 {\n        \
                      let mut acc: i64 = 0;\n    }\n    return acc;\n}\n";
        let text = format!("{main}\n{helper}");
        let ast = crate::parser::parse(&text).expect("parse");
        let origins = [
            ("src/main.mind".to_string(), 0),
            ("src/helper.mind".to_string(), main.len() + 1),
        ];
        let err = refuse_shadowing(&ast, &text, &origins, &|| false).unwrap_err();
        assert!(err.contains("native.block_shadow"), "{err}");
        assert!(err.contains("src/helper.mind:4:"), "{err}");
        assert!(err.contains("at src/helper.mind:2"), "{err}");
        assert!(!err.contains("src/main.mind:"), "{err}");
    }

    #[test]
    fn a_lowering_panic_becomes_a_refusal() {
        // An unknown variant of a known enum reaches lowering's fail-closed
        // panic when the name check is skipped; the boundary reports it.
        let src =
            "enum Color { Red, Blue }\nfn main() -> i64 { let c = Color::Green; return 0; }\n";
        let ast = crate::parser::parse(src).expect("parse");
        let err = lower_without_panicking(&ast, "t.mind").unwrap_err();
        assert!(err.contains("lower.refused_construct"), "{err}");
    }
}
