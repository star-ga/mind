// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Source-level refusals the native bridge applies to the merged program.
//!
//! * An entry `main` that declares parameters is refused. The frozen compiler
//!   does not bind `main`'s parameters to the process arguments, so such a
//!   program would run with every parameter 0: `fn main(argc: i64, argv: i64)`
//!   built and exited 40 natively where the MLIR build of the same source
//!   exited 23. Refused until the seed binds them.
//! * The name check the MLIR build path treats as fatal (an unknown identifier
//!   E2002, a function used as a value E2037) and an unknown enum variant
//!   (E2008) are refused with the type checker's own diagnostics. Without the
//!   check these reached lowering's fail-closed `panic!` and the build died
//!   with exit 101.
//! * A symbolic tensor dimension read as a VALUE (`reshape(x, [K, 1])` where
//!   `K` is only a dimension of a parameter's type) is refused. The type
//!   checker deliberately lets the name resolve, but lowering has no value for
//!   it and panicked on the same undefined-identifier guard.
//! * Lowering itself runs behind a panic boundary, so any other construct that
//!   still reaches a lowering `panic!` becomes a refusal with its message, not
//!   an exit 101.

use std::collections::BTreeSet;

use crate::ast::{Module, Node};
use crate::diagnostics::{
    ColorChoice, Diagnostic, DiagnosticEmitter, DiagnosticFormat, Severity, Span,
};
use crate::type_checker::resolve;

/// Refuse an entry `main` that declares parameters.
pub(super) fn refuse_entry_main_with_params(ast: &Module, path: &str) -> Result<(), String> {
    for item in &ast.items {
        let Node::FnDef(f, _) = item else { continue };
        if f.name != "main" || f.params.is_empty() {
            continue;
        }
        let params: Vec<&str> = f.params.iter().map(|p| p.name.as_str()).collect();
        return Err(format!(
            "`{path}`: the entry `fn main` declares parameter(s) {params:?}, but the native \
             backend does not bind `main`'s parameters to the process arguments yet, so the \
             program would run with every one of them 0 (a silent miscompile). Declare \
             `fn main()` or rerun with `--backend mlir`. No artifact written. \
             (native.entry_main_has_params)"
        ));
    }
    Ok(())
}

/// Refuse a binding inside a loop that re-declares an enclosing name.
///
/// The frozen native compiler resolves loop-carried variables by name, so it
/// merges such a binding into the outer one: `let mut x = 5; while k < 2 {
/// k = k + 1; let mut x = 7; x = 3; } return x;` built natively and exited 3
/// where the MLIR build exits 5. Refused (with every offending binding named)
/// until the seed scopes loop-body bindings. The scope rules are in
/// `loop_shadow`; when unsure they refuse.
pub(super) fn refuse_loop_body_shadowing(
    ast: &Module,
    text: &str,
    path: &str,
) -> Result<(), String> {
    let shadows = super::loop_shadow::find(ast);
    if shadows.is_empty() {
        return Ok(());
    }
    let line_of = |offset: usize| text[..offset.min(text.len())].matches('\n').count() + 1;
    let found: Vec<Diagnostic> = shadows
        .iter()
        .map(|s| {
            let mut d = Diagnostic::error(
                "native",
                "native.loop_body_shadow",
                format!(
                    "{} `{}` inside a loop in fn `{}` re-declares `{}`, already bound as a {} \
                     at line {}; the frozen native compiler resolves loop-carried variables by \
                     name and would merge the two bindings (a silent miscompile)",
                    s.inner_kind,
                    s.name,
                    s.fn_name,
                    s.name,
                    s.outer_kind,
                    line_of(s.outer.start())
                ),
            );
            let start = s.inner.start().min(text.len());
            let end = s.inner.end().min(text.len()).max(start);
            d.span = Some(Span::from_offsets(text, start, end, Some(path)));
            d.help = Some(format!(
                "rename the inner `{}`, or rerun with `--backend mlir`",
                s.name
            ));
            d
        })
        .collect();
    let names: BTreeSet<&str> = shadows.iter().map(|s| s.name.as_str()).collect();
    Err(format!(
        "{}\n`{path}`: binding(s) {names:?} inside a loop re-declare a name from an enclosing \
         scope; the native backend would merge them with the outer binding. Rerun with \
         `--backend mlir`. No artifact written. (native.loop_body_shadow)",
        render(&found, text)
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
    fn only_a_parameterised_main_is_refused() {
        let ok = crate::parser::parse("fn main() -> i64 { return 0; }\n").expect("parse");
        assert!(refuse_entry_main_with_params(&ok, "t.mind").is_ok());
        let helper = crate::parser::parse(
            "fn h(a: i64) -> i64 { return a; }\nfn main() -> i64 { return h(1); }\n",
        )
        .expect("parse");
        assert!(refuse_entry_main_with_params(&helper, "t.mind").is_ok());
        let bad =
            crate::parser::parse("fn main(argc: i64) -> i64 { return argc; }\n").expect("parse");
        assert!(refuse_entry_main_with_params(&bad, "t.mind").is_err());
    }

    fn shadowed(src: &str) -> Vec<String> {
        let ast = crate::parser::parse(src).expect("parse");
        super::super::loop_shadow::find(&ast)
            .into_iter()
            .map(|s| format!("{}:{}<{}", s.name, s.inner_kind, s.outer_kind))
            .collect()
    }

    #[test]
    fn loop_body_shadows_are_found_at_every_nesting() {
        let w = "fn main() -> i64 { let mut x = 5; let mut k = 0; \
                 while k < 2 { k = k + 1; let mut x = 7; x = 3; } return x; }";
        assert_eq!(shadowed(w), ["x:let<let"]);
        let p = "fn f(x: i64) -> i64 { for i in 0..2 { let x = 1; } return x; }";
        assert_eq!(shadowed(p), ["x:let<parameter"]);
        let in_if = "fn f(x: i64) -> i64 { while x < 2 { if x == 1 { let x = 1; } } return x; }";
        assert_eq!(shadowed(in_if), ["x:let<parameter"]);
        let nested = "fn f() -> i64 { while 1 { let a = 1; while 1 { let a = 2; } } return 0; }";
        assert_eq!(shadowed(nested), ["a:let<let"]);
        let for_var = "fn f() -> i64 { for i in 0..3 { let i = 10; } return 0; }";
        assert_eq!(shadowed(for_var), ["i:let<for variable"]);
        let nested_for = "fn f() -> i64 { for i in 0..3 { for i in 0..2 { } } return 0; }";
        assert_eq!(shadowed(nested_for), ["i:for variable<for variable"]);
        let header = "fn f(i: i64) -> i64 { for i in 0..3 { } return i; }";
        assert_eq!(shadowed(header), ["i:for variable<parameter"]);
        let tuple = "fn f(b: i64) -> i64 { while 1 { let (a, b) = (1, 2); } return b; }";
        assert_eq!(shadowed(tuple), ["b:let<parameter"]);
        let konst = "const N: i64 = 4;\nfn f() -> i64 { while 1 { let N = 3; } return N; }";
        assert_eq!(shadowed(konst), ["N:let<const"]);
    }

    #[test]
    fn non_loop_shadows_and_fresh_loop_locals_are_admitted() {
        for ok in [
            "fn f(x: i64) -> i64 { if x == 1 { let x = 7; } return x; }",
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
        let err = refuse_loop_body_shadowing(&ast, src, "t.mind").unwrap_err();
        assert!(err.contains("native.loop_body_shadow"), "{err}");
        assert!(err.contains("t.mind:4:"), "{err}");
        assert!(err.contains("at line 2"), "{err}");
        assert!(
            err.contains("`x`") && err.contains("--backend mlir"),
            "{err}"
        );
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
